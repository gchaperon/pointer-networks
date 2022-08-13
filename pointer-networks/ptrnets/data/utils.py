import contextlib
import io
import itertools
import logging
import pathlib
import typing as tp
import zipfile

import numpy as np
import numpy.ma as ma
import numpy.typing as npt
import tqdm


def load_file(
    fname: pathlib.Path,
    inner_path: tp.Optional[str] = None,
) -> tp.Tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    point_sets: tp.List[npt.NDArray[np.float32]] = []
    targets: tp.List[npt.NDArray[np.int64]] = []
    with uopen(fname, inner_path) as file:
        for line in tqdm.tqdm(
            file,
            desc=f"Loading {fname}",
            unit="lines",
            disable=logging.getLogger().level > logging.WARNING,
        ):
            point_set, target = parse_line(line)
            point_sets.append(point_set)
            targets.append(target)

    _DT = tp.TypeVar("_DT", bound=np.generic)

    def to_masked(array: npt.NDArray[_DT], max_len: int) -> npt.NDArray[_DT]:
        masked: npt.NDArray[_DT] = ma.array(
            np.empty((max_len, *array.shape[1:]), dtype=array.dtype)
        )
        masked[: len(array)] = array
        masked[len(array) :] = ma.masked
        return masked

    max_len_points = max(map(len, point_sets))
    max_len_targets = max(map(len, targets))
    return (
        ma.stack([to_masked(arr, max_len_points) for arr in point_sets]),
        ma.stack([to_masked(arr, max_len_targets) for arr in targets]),
    )


def parse_line(line: str) -> tp.Tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    coord_str, _, indices_str = line.partition("output")
    points = np.array(coord_str.split(), dtype=np.float32).reshape(-1, 2)
    indices = np.array(indices_str.split(), dtype=np.int64)
    return points, indices


# I am not completely sure about this usage of contextmanager
@contextlib.contextmanager
def uopen(
    path: pathlib.Path, inner_path: tp.Optional[str] = None
) -> tp.Generator[tp.Iterator[str], None, None]:
    """I think there are 4 cases, path might be
    * a .txt file
    * a .zip file with a single .txt inside
    * a .zip file with a bunch of .txt that should be concatenated
    * a .zip file with train/test splits inside, in .txt format, the second
      optional argument is for this case"""

    if path.suffix == ".txt":
        # Just a .txt, open it and return
        with open(path) as txt_file:
            yield txt_file
    elif path.suffix == ".zip":
        # 3 cases for zip, always open it
        with zipfile.ZipFile(path) as myzip:
            if inner_path:
                # If there is an inner_path open that and return
                with myzip.open(inner_path) as bytes_file:
                    yield io.TextIOWrapper(bytes_file)
            else:
                # If there is one or more files inside the zip,
                # open them all and concatenate them
                with contextlib.ExitStack() as stack:
                    yield itertools.chain.from_iterable(
                        io.TextIOWrapper(stack.enter_context(myzip.open(fname)))
                        for fname in myzip.namelist()
                    )
    else:
        raise ValueError(f"Invalid suffix in {path=}")
