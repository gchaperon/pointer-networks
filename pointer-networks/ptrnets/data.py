import os
import io
import math
import zipfile
import typing as tp
import pathlib
import functools
import itertools
import torch
import pytorch_lightning as pl
import contextlib


_Point = tp.Tuple[float, float]


def parse_line(line: str) -> tp.Tuple[tp.List[_Point], tp.List[int]]:
    coord_str, _, indices_str = line.partition("output")
    coords = [float(el) for el in coord_str.split()]
    points = list(zip(coords[::2], coords[1::2]))
    # reveal_type(points)
    indices = list(map(int, indices_str.split()))
    return points, indices


def distance(pt1: tp.Tuple[float, float], pt2: tp.Tuple[float, float]) -> float:
    return math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def tour_distance(pts: tp.List[_Point], seq: tp.List[int]) -> float:
    dist = 0.0
    # NOTE: seq indices are 1-indexed
    for i, j in zip(seq, seq[1:]):
        dist += distance(pts[i - 1], pts[j - 1])
    return dist


# I am not completely sure about this usage of contextmanager
@contextlib.contextmanager
def _open(
    path: pathlib.Path, inner_path: tp.Optional[str] = None
) -> tp.Generator[tp.Iterator[str], None, None]:
    """I think there are 4 cases
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
        raise ValueError(f"Invalid {path=}")


_NExamples = tp.Literal[5, 10, 20, 40, 50, "5-20"]
_Split = tp.Literal["train", "test"]
_Algorithm = tp.Literal["optimal", "a1", "a2", "a3"]

_PtrNetItem = tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class TSP(torch.utils.data.Dataset[_PtrNetItem]):
    _options_to_file: tp.ClassVar[
        tp.Dict[
            tp.Tuple[_NExamples, _Split, _Algorithm],
            tp.Tuple[str, ...],
        ]
    ] = {
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

    tuples: tp.List[tp.Tuple[tp.List[_Point], tp.List[int]]]

    def __init__(
        self,
        datadir: str,
        nexamples: _NExamples,
        split: _Split,
        algorithm: _Algorithm,
    ) -> None:
        super().__init__()

        open_args = self._options_to_file.get((nexamples, split, algorithm))
        if open_args is None:
            raise ValueError(
                "Invalid option combination. "
                'Check the paper and "Notes on data" in thetuple readme.'
            )

        fname, *rest = open_args

        # reveal_type(rest)
        with _open(pathlib.Path(datadir) / fname, *rest) as file:
            var = list(map(parse_line, file))
            self.tuples = var
            # reveal_type(var)

    def __getitem__(self, key: int) -> _PtrNetItem:
        points, idx_sequence = map(torch.tensor, self.tuples[key])
        # NOTE: idx_sequence is 1-indexed
        # prepend target_point_sequence with bos token (=> in the paper)
        # i will arbitrarily define that token to be [-1, -1]
        target_point_sequence = torch.vstack(
            (torch.ones(2) * -1, points[idx_sequence - 1])
        )
        # append 0 to idx_sequence to simulate eos token
        return (
            points,
            target_point_sequence,
            torch.cat([idx_sequence, torch.tensor([0])]),
        )

    def __len__(self) -> int:
        return len(self.tuples)


_Batch = tp.Tuple[
    torch.nn.utils.rnn.PackedSequence,
    torch.nn.utils.rnn.PackedSequence,
    torch.nn.utils.rnn.PackedSequence,
]


def collate_items(items: tp.Sequence[_PtrNetItem]) -> _Batch:
    point_sets: tp.List[torch.Tensor] = []
    target_point_sequences: tp.List[torch.Tensor] = []
    answers: tp.List[torch.Tensor] = []
    for points, target_point_sequence, answer in items:
        point_sets.append(points)
        target_point_sequences.append(target_point_sequence)
        answers.append(answer)
    pack = functools.partial(torch.nn.utils.rnn.pack_sequence, enforce_sorted=False)
    return pack(point_sets), pack(target_point_sequences), pack(answers)


class TSPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datadir: str,
        train_params: tp.Tuple[_NExamples, _Algorithm],
        test_params: tp.Tuple[_NExamples, _Algorithm],
        batch_size: int,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.save_hyperparameters(ignore=(datadir,))

        self.datadir = datadir
        # NOTE: the explicit type here is because mypy gets confused otherwise
        self.train_split_params: tp.Tuple[_NExamples, _Split, _Algorithm] = (
            train_params[0],
            "train",
            train_params[1],
        )
        self.test_split_params: tp.Tuple[_NExamples, _Split, _Algorithm] = (
            test_params[0],
            "test",
            test_params[1],
        )
        self.batch_size = batch_size

    def setup(self, stage: tp.Optional[str] = None) -> None:
        if stage in ("fit", "validate", None):
            self.tsp_test = TSP(self.datadir, *self.test_split_params)
        if stage in ("fit", None):
            self.tsp_train = TSP(self.datadir, *self.train_split_params)

    def train_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            # for cast explanation see this
            # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py#L30
            dataset=tp.cast(torch.utils.data.Dataset[_Batch], self.tsp_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() or 0,
            collate_fn=collate_items,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            # for cast explanation see train_dataloader
            dataset=tp.cast(torch.utils.data.Dataset[_Batch], self.tsp_test),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() or 0,
            collate_fn=collate_items,
            pin_memory=True,
        )


# class ConvexHull(torch.utils.data.Dataset):
#     """WIP"""
#
#     _file_template = "convex_hull_{n_examples}_{split}"
#     point_sets: tp.List[tp.List[tp.Tuple[float, float]]]
#     answers: tp.List[tp.List[int]]
#
#     def __init__(
#         self,
#         datadir: str,
#         n_examples: tp.Literal[5, 10, 50, "5-50", 200, 500],
#         split: tp.Literal["train", "test"],
#     ):
#         preffix = self._file_template.format(n_examples=n_examples, split=split)
#         datapath = next(
#             (p for p in pathlib.Path(datadir).iterdir() if p.name.startswith(preffix)),
#             None,
#         )
#         if not datapath:
#             raise ValueError(
#                 f"Combi not found in {datadir=}: {n_examples=} and {split=}"
#             )
#
#         with _open(datapath) as file:
#             self.point_sets, self.answers = zip(*map(parse_line, file))
#
#     def __len__(self) -> int:
#         return len(self.point_sets)
#
#
# class ConvexHullDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         datadir: str = "data",
#     ):
#         pass
