import io
import math
import zipfile
import typing as tp
import pathlib
import itertools
import torch
import pytorch_lightning as pl
import contextlib

NExamplesT = tp.Literal["5", "10", "50", "5-50", "200", "500"]


def parse_line(line):
    pts, _, ans = line.partition("output")
    return list(zip(*[map(float, pts.split())] * 2)), list(map(int, ans.split()))


def load(lines):
    for line in lines:
        yield parse_line(line)


def distance(pt1: tp.Tuple[float, float], pt2: tp.Tuple[float, float]) -> float:
    return math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def tour_distance(pts, seq):
    dist = 0.0
    # NOTE: seq indices are 1-indexed
    for i, j in zip(seq, seq[1:]):
        dist += distance(pts[i - 1], pts[j - 1])
    return dist


# I am not completely sure about this usage of contextmanager
@contextlib.contextmanager
def _open(path: pathlib.Path, inner_path: tp.Optional[str] = None) -> tp.Iterator[str]:
    """I think there are 4 cases
    * a .txt file
    * a .zip file with a single .txt inside
    * a .zip file with a bunch of .txt that should be concatenated
    * a .zip file with train/test splits inside, in .txt format, the second
      optional argument is for this case"""

    if path.suffix == ".txt":
        # Just a .txt, open it and return
        with open(path) as file:
            yield file
    elif path.suffix == ".zip":
        # 3 cases for zip, always open it
        with zipfile.ZipFile(path) as myzip:
            if inner_path:
                # If there is an inner_path open that and return
                with myzip.open(inner_path) as file:
                    yield io.TextIOWrapper(file)
            else:
                # If there is one or more files inside the zip,
                # open them all and concatenate them
                with contextlib.ExitStack() as stack:
                    yield itertools.chain.from_iterable(
                        io.TextIOWrapper(stack.enter_context(myzip.open(fname)))
                        for fname in myzip.namelist()
                    )
    else:
        raise ValueError(f"Invalid {path=}")


class ConvexHull(torch.utils.data.Dataset):
    """WIP"""

    _file_template = "convex_hull_{n_examples}_{split}"
    point_sets: tp.List[tp.List[tp.Tuple[float, float]]]
    answers: tp.List[tp.List[int]]

    def __init__(
        self,
        datadir: str,
        n_examples: tp.Literal[5, 10, 50, "5-50", 200, 500],
        split: tp.Literal["train", "test"],
    ):
        preffix = self._file_template.format(n_examples=n_examples, split=split)
        datapath = next(
            (p for p in pathlib.Path(datadir).iterdir() if p.name.startswith(preffix)),
            None,
        )
        if not datapath:
            raise ValueError(
                f"Combi not found in {datadir=}: {n_examples=} and {split=}"
            )

        with _open(datapath) as file:
            self.point_sets, self.answers = zip(*map(parse_line, file))

    def __len__(self):
        return len(self.point_sets)


class TSP(torch.utils.data.Dataset):
    _options_to_file = {
        (5, "train", "optimal"): ("tsp_5_train.zip", "tsp5.txt"),
        (5, "test", "optimal"): ("tsp_5_train.zip", "tsp5_test.txt"),
        (10, "train", "optimal"): ("tsp_10_train_exact.txt",),
        (10, "test", "optimal"): ("tsp_10_train_exact.txt",),
        (10, "test", "a1"): ("tsp_10_train.zip", "tsp10_test.txt"),
        (20, "test", "a1"): ("tsp_20_test.txt",),
        (40, "test", "a1"): ("tsp_40_test.txt",),
        (50, "train", "a1"): ("tsp_50_train.zip",),
        (50, "test", "a1"): ("tsp_50_test.txt.zip",),
        ("5-20", "train", "optimal"): ("tsp_5-20_train.zip",),
    }

    tuples = tp.List[tuple]

    def __init__(
        self,
        datadir: str,
        nexamples: tp.Literal[5, 10, 20, 40, 50, "5-20"],
        split: tp.Literal["train", "test"],
        algorithm: tp.Literal["optimal", "a1", "a2", "a3"],
    ):
        open_args = self._options_to_file.get((nexamples, split, algorithm))
        if open_args is None:
            raise ValueError(
                "Invalid option combination. "
                'Check the paper and "Notes on data" in the readme.'
            )

        fname, *rest = open_args

        with _open(pathlib.Path(datadir) / fname, *rest) as file:
            breakpoint()
            self.tuples = list(map(parse_line, file))

    def __getitem__(self, key: int) -> tuple:
        points, answer = self.tuples[key]
        return torch.tensor(points), torch.tensor(answer)

    def __len__(self) -> int:
        return len(self.tuples)


class ConvexHullDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datadir: str = "data",
    ):
        pass
