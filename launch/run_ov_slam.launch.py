import hashlib
import os
import re
import tempfile
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


OPENVINS_CONFIG_FILENAME = "estimator_config.yaml"
OV_SECONDARY_CONFIG_FILENAME = "master_config.yaml"
PACKAGE_SHARE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_SHARE_DIR / "config"
DEFAULT_IMAGE_CONVERSION_SCRIPT = (
    PACKAGE_SHARE_DIR / "scripts" / "dataset_utils" / "image_conversion_node.py"
)
DEFAULT_RVIZ_CONFIG_PATH = PACKAGE_SHARE_DIR / "launch" / "ov_slam.rviz"
DEFAULT_OPENVINS_CONFIG_PATH = CONFIG_DIR / "open_vins" / OPENVINS_CONFIG_FILENAME
DEFAULT_OV_SECONDARY_CONFIG_PATH = (
    CONFIG_DIR / "ov_secondary" / OV_SECONDARY_CONFIG_FILENAME
)


launch_args = [
    DeclareLaunchArgument(
        name="bag",
        default_value="",
        description="path to ros2 bag to play (if empty, do not play)",
    ),
    DeclareLaunchArgument(
        name="bag_rate",
        default_value="1.0",
        description="ros2 bag play rate (1.0 = realtime speed)",
    ),
    DeclareLaunchArgument(
        name="namespace", default_value="ov_msckf", description="namespace"
    ),
    DeclareLaunchArgument(
        name="ov_enable", default_value="true", description="enable OpenVINS node"
    ),
    DeclareLaunchArgument(
        name="rviz_enable", default_value="true", description="enable rviz node"
    ),
    DeclareLaunchArgument(
        name="openvins_config_path",
        default_value=str(DEFAULT_OPENVINS_CONFIG_PATH),
        description="Path to OpenVINS estimator_config.yaml",
    ),
    DeclareLaunchArgument(
        name="verbosity",
        default_value="WARNING",
        description="ALL, DEBUG, INFO, WARNING, ERROR, SILENT",
    ),
    DeclareLaunchArgument(
        name="use_stereo",
        default_value="false",
        description=(
            "if we have more than 1 camera, if we should try to track stereo "
            "constraints between pairs"
        ),
    ),
    DeclareLaunchArgument(
        name="max_cameras",
        default_value="1",
        description=(
            "how many cameras we have 1 = mono, 2 = stereo, >2 = binocular "
            "(all mono tracking)"
        ),
    ),
    DeclareLaunchArgument(
        name="save_total_state",
        default_value="false",
        description="record the total state with calibration and features to a txt file",
    ),
    DeclareLaunchArgument(
        name="filepath_est",
        default_value="/tmp/ov_estimate.txt",
        description="output path for full state estimate",
    ),
    DeclareLaunchArgument(
        name="filepath_std",
        default_value="/tmp/ov_estimate_std.txt",
        description="output path for state standard deviation",
    ),
    DeclareLaunchArgument(
        name="filepath_gt",
        default_value="/tmp/ov_groundtruth.txt",
        description="output path for groundtruth state",
    ),
    DeclareLaunchArgument(
        name="filepath_odom",
        default_value="/tmp/ov_odometry.txt",
        description="output path for odom in TUM format",
    ),
    DeclareLaunchArgument(
        name="image_conversion",
        default_value="true",
        description="Enable compressed-to-image conversion node",
    ),
    DeclareLaunchArgument(
        name="posegraph_enable",
        default_value="true",
        description="enable OpenVINS secondary posegraph node",
    ),
    DeclareLaunchArgument(
        name="ov_secondary_config_path",
        default_value=str(DEFAULT_OV_SECONDARY_CONFIG_PATH),
        description="Path to ov_secondary master_config.yaml",
    ),
    DeclareLaunchArgument(
        name="output_path",
        default_value="",
        description=(
            "override ov_secondary_loop_fusion output_path; if empty, use "
            "ov_secondary_config_path value"
        ),
    ),
    DeclareLaunchArgument(
        name="pose_graph_load_path",
        default_value="",
        description=(
            "override ov_secondary_loop_fusion pose_graph_load_path; if empty, use "
            "ov_secondary_config_path value"
        ),
    ),
]


def _shutdown_when_process_fails(action, process_name, shutdown_state):
    def on_exit(event, context):
        if context.is_shutdown or shutdown_state["requested"]:
            return []

        if event.returncode == 0:
            return [LogInfo(msg=f"{process_name} exited cleanly.")]

        reason = f"{process_name} failed with return code {event.returncode}"
        shutdown_state["requested"] = True
        return [
            LogInfo(msg=f"{reason}; shutting down launch."),
            EmitEvent(event=Shutdown(reason=reason)),
        ]

    return RegisterEventHandler(
        OnProcessExit(
            target_action=action,
            on_exit=on_exit,
        )
    )


def _cleanup_generated_file_on_shutdown(file_path):
    def on_shutdown(event, context):
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        return []

    return RegisterEventHandler(OnShutdown(on_shutdown=on_shutdown))


def _image_conversion_topics(max_cameras):
    if max_cameras == "1":
        return [
            "/insta/cam0/image_raw/compressed",
            "/insta/cam0/image_raw",
        ]

    return [
        "/insta/cam0/image_raw/compressed",
        "/insta/cam1/image_raw/compressed",
        "/insta/cam0/image_raw",
        "/insta/cam1/image_raw",
    ]


def _quoted_opencv_yaml_string(value):
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _opencv_yaml_value(value):
    if isinstance(value, str):
        return _quoted_opencv_yaml_string(value)
    return str(value)


def _write_ov_secondary_config_override(
    config_path,
    output_path,
    pose_graph_load_path,
):
    overrides = {
        "output_path": output_path,
        "pose_graph_load_path": pose_graph_load_path,
    }
    if pose_graph_load_path:
        overrides["load_previous_pose_graph"] = 1
    overrides = {key: value for key, value in overrides.items() if value}
    if not overrides:
        return config_path, None

    with open(config_path, "r", encoding="utf-8") as config_file:
        config_lines = config_file.readlines()

    seen_keys = set()
    updated_lines = []
    for line in config_lines:
        updated_line = line
        for key, value in overrides.items():
            match = re.match(
                rf"^(\s*{re.escape(key)}\s*:\s*)([^#\n]*?)(\s*#.*)?$",
                line,
            )
            if match:
                comment = match.group(3) or ""
                updated_line = (
                    f"{match.group(1)}{_opencv_yaml_value(value)}{comment}\n"
                )
                seen_keys.add(key)
                break
        updated_lines.append(updated_line)

    missing_keys = set(overrides.keys()) - seen_keys
    if missing_keys:
        updated_lines.append("\n# launch-time ov_secondary_loop_fusion overrides\n")
        for key in sorted(missing_keys):
            updated_lines.append(f"{key}: {_opencv_yaml_value(overrides[key])}\n")

    config_digest = hashlib.sha1(
        os.path.abspath(config_path).encode("utf-8")
    ).hexdigest()[:12]
    generated_config_path = os.path.join(
        tempfile.gettempdir(),
        f"ov_secondary_loop_fusion_{config_digest}.yaml",
    )
    with open(generated_config_path, "w", encoding="utf-8") as generated_config:
        generated_config.writelines(updated_lines)

    return generated_config_path, generated_config_path


def launch_setup(context):
    openvins_actions = []
    shutdown_handlers = []
    shutdown_state = {"requested": False}
    max_cameras = LaunchConfiguration("max_cameras").perform(context)
    openvins_config_path = LaunchConfiguration("openvins_config_path").perform(context)
    if not os.path.isfile(openvins_config_path):
        openvins_actions.append(
            LogInfo(
                msg=(
                    "ERROR: openvins_config_path file: '{}' - "
                    "does not exist. - not starting OpenVINS"
                ).format(
                    openvins_config_path
                )
            )
        )

    if not openvins_actions:
        openvins_node = Node(
            package="ov_msckf",
            executable="run_subscribe_msckf",
            condition=IfCondition(LaunchConfiguration("ov_enable")),
            namespace=LaunchConfiguration("namespace"),
            output="screen",
            parameters=[
                {"verbosity": LaunchConfiguration("verbosity")},
                {"use_stereo": LaunchConfiguration("use_stereo")},
                {"max_cameras": LaunchConfiguration("max_cameras")},
                {"config_path": openvins_config_path},
                {"save_total_state": LaunchConfiguration("save_total_state")},
                {"filepath_est": LaunchConfiguration("filepath_est")},
                {"filepath_std": LaunchConfiguration("filepath_std")},
                {"filepath_gt": LaunchConfiguration("filepath_gt")},
                {"filepath_odom": LaunchConfiguration("filepath_odom")},
            ],
        )
        openvins_actions.append(openvins_node)
        shutdown_handlers.append(
            _shutdown_when_process_fails(openvins_node, "OpenVINS", shutdown_state)
        )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        condition=IfCondition(LaunchConfiguration("rviz_enable")),
        arguments=[
            "-d" + str(DEFAULT_RVIZ_CONFIG_PATH),
            "--ros-args",
            "--log-level",
            "warn",
        ],
    )

    image_conversion_actions = []
    image_conversion_script = str(DEFAULT_IMAGE_CONVERSION_SCRIPT)
    if not os.path.isfile(image_conversion_script):
        image_conversion_actions.append(
            LogInfo(
                msg=(
                    "ERROR: image_conversion_node script: '{}' - "
                    "does not exist. - not starting image conversion"
                ).format(image_conversion_script)
            )
        )
    else:
        image_conversion_process = ExecuteProcess(
            condition=IfCondition(LaunchConfiguration("image_conversion")),
            cmd=[
                image_conversion_script,
                *_image_conversion_topics(max_cameras),
                "--ros-args",
                "-r",
                PythonExpression(
                    [
                        "'__ns:=/' + '",
                        LaunchConfiguration("namespace"),
                        "'.lstrip('/')",
                    ]
                ),
            ],
            output="screen",
        )
        image_conversion_actions.append(image_conversion_process)
        shutdown_handlers.append(
            _shutdown_when_process_fails(
                image_conversion_process,
                "image_conversion_node",
                shutdown_state,
            )
        )

    ov_secondary_actions = []
    ov_secondary_config_path = LaunchConfiguration("ov_secondary_config_path").perform(
        context
    )
    if not os.path.isfile(ov_secondary_config_path):
        ov_secondary_actions.append(
            LogInfo(
                msg=(
                    "ERROR: ov_secondary_config_path file: '{}' - "
                    "does not exist. - not starting OV secondary"
                ).format(ov_secondary_config_path)
            )
        )
    else:
        ov_secondary_config_path, generated_ov_secondary_config_path = (
            _write_ov_secondary_config_override(
                ov_secondary_config_path,
                LaunchConfiguration("output_path").perform(context),
                LaunchConfiguration("pose_graph_load_path").perform(context),
            )
        )
        if generated_ov_secondary_config_path:
            shutdown_handlers.append(
                _cleanup_generated_file_on_shutdown(generated_ov_secondary_config_path)
            )
        ov_secondary_node = Node(
            package="ov_secondary_loop_fusion",
            executable="loop_fusion_node",
            name="loop_fusion_node",
            output="screen",
            condition=IfCondition(LaunchConfiguration("posegraph_enable")),
            remappings=[
                (
                    "/ov_msckf/cam0/image_raw/compressed",
                    "/insta/cam0/image_raw/compressed",
                )
            ],
            emulate_tty=True,
            parameters=[
                {
                    "config_file": ov_secondary_config_path,
                    "vocabulary_file": PathJoinSubstitution(
                        [
                            FindPackageShare("ov_secondary_loop_fusion"),
                            "data",
                            "brief_k10L6.bin",
                        ]
                    ),
                    "brief_pattern_file": PathJoinSubstitution(
                        [
                            FindPackageShare("ov_secondary_loop_fusion"),
                            "data",
                            "brief_pattern.yml",
                        ]
                    ),
                }
            ],
        )
        ov_secondary_actions.append(ov_secondary_node)
        shutdown_handlers.append(
            _shutdown_when_process_fails(
                ov_secondary_node, "OV secondary", shutdown_state
            )
        )

    bag_play = ExecuteProcess(
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("bag"), "' != ''"])
        ),
        cmd=[
            "ros2",
            "bag",
            "play",
            LaunchConfiguration("bag"),
            "--clock",
            "--rate",
            LaunchConfiguration("bag_rate"),
        ],
        output="screen",
    )

    return shutdown_handlers + openvins_actions + [
        rviz_node,
        *image_conversion_actions,
        *ov_secondary_actions,
        bag_play,
    ]


def generate_launch_description():
    opfunc = OpaqueFunction(function=launch_setup)
    ld = LaunchDescription(launch_args)
    ld.add_action(opfunc)
    return ld
