import typing as tp

Point = tp.Tuple[float, float]


class Polygon:
    def __init__(self, shell: tp.List[Point]) -> None:
        ...

    @property
    def area(self) -> float:
        ...

    @property
    def is_simple(self) -> bool:
        ...

    def equals(self, other: "Polygon") -> bool:
        ...
