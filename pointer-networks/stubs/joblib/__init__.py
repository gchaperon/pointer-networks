import typing as tp
import pathlib

import typing_extensions as tpx

T = tp.TypeVar("T")
P = tpx.ParamSpec("P")


class Memory:
    def __init__(
        self, location: tp.Union[str, pathlib.Path, None], verbose: tp.Optional[int]
    ) -> None:
        ...

    def cache(
        self,
        func: tp.Callable[P, T],
        ignore: tp.Optional[tp.List[str]] = None,
        verbose: tp.Optional[int] = None,
        mmap_mode: tp.Optional[tp.Literal["r+", "r", "w+", "c"]] = None,
    ) -> tp.Callable[P, T]:
        ...
