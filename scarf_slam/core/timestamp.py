from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class MappingTimestamp:
    sec: int
    nsec: int

    def __repr__(self) -> str:
        return f"MappingTimestamp(sec={self.sec}, nsec={self.nsec})"

    @property
    def total_nanoseconds(self) -> int:
        return int(self.sec) * 1_000_000_000 + int(self.nsec)
