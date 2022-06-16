import typing as tp
import pathlib
import functools
import torch
import pytorch_lightning as pl
import tqdm
import os
import numpy as np
import numpy.typing as npt

from .utils import parse_line, uopen


_PtrNetItem = tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_Batch = tp.Tuple[
    torch.nn.utils.rnn.PackedSequence,
    torch.nn.utils.rnn.PackedSequence,
    torch.nn.utils.rnn.PackedSequence,
]


class TSP(torch.utils.data.Dataset[_PtrNetItem]):
    NPointsT = tp.Literal["5", "10", "20", "40", "50", "5-20", "5-10"]
    SplitT = tp.Literal["train", "test", "debug"]
    AlgorithmT = tp.Literal["optimal", "a1", "a2", "a3"]

    _options_to_file: tp.ClassVar[
        tp.Dict[
            tp.Tuple[NPointsT, SplitT, AlgorithmT],
            tp.Tuple[str, ...],
        ]
    ] = {
        ("5", "train", "optimal"): ("tsp_5_train.zip", "tsp5.txt"),
        ("5", "test", "optimal"): ("tsp_5_train.zip", "tsp5_test.txt"),
        ("10", "train", "optimal"): ("tsp_10_train_exact.txt",),
        ("10", "test", "optimal"): ("tsp_10_test_exact.txt",),
        ("10", "test", "a1"): ("tsp_10_train.zip", "tsp10_test.txt"),
        ("20", "test", "a1"): ("tsp_20_test.txt",),
        ("40", "test", "a1"): ("tsp_40_test.txt",),
        ("50", "train", "a1"): ("tsp_50_train.zip",),
        ("50", "test", "a1"): ("tsp_50_test.txt.zip",),
        ("5-20", "train", "optimal"): ("tsp_5-20_train.zip",),
        # for debugging
        ("5-10", "train", "optimal"): ("tsp_5-10_debug.txt",),
        ("5-10", "test", "optimal"): ("tsp_5-10_debug.txt",),
    }

    BOS_TOKEN = np.array([-1.0, -1.0], dtype=np.float32)
    EOS_TOKEN = np.array([0])

    point_sets: tp.List[npt.NDArray[np.float32]]
    targets: tp.List[npt.NDArray[np.int64]]

    def __init__(
        self,
        datadir: str,
        npoints: NPointsT,
        split: SplitT,
        algorithm: AlgorithmT,
    ) -> None:
        super().__init__()

        self.datadir = datadir
        self.npoints = npoints
        self.split = split
        self.algorithm = algorithm

        open_args = self._options_to_file.get((npoints, split, algorithm))
        if open_args is None:
            raise ValueError(
                "Invalid option combination. "
                'Check the paper and "Notes on data" in the readme.'
            )

        fname, *rest = open_args

        with uopen(pathlib.Path(datadir) / fname, *rest) as file:
            self.point_sets = []
            self.targets = []
            for line in tqdm.tqdm(file, desc=f"Loading {self}", unit="lines"):
                point_set, target = parse_line(line)
                self.point_sets.append(point_set)
                self.targets.append(target)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"datadir={self.datadir!r}, "
            f"npoints={self.npoints!r}, "
            f"split={self.split!r}, "
            f"algorithm={self.algorithm!r})"
        )

    def __getitem__(self, key: int) -> _PtrNetItem:
        points = torch.from_numpy(self.point_sets[key])
        # append 0 to idx_sequence to simulate eos token
        idx_sequence = torch.from_numpy(
            np.concatenate([self.targets[key], self.EOS_TOKEN])
        )
        # NOTE: idx_sequence is 1-indexed
        # prepend target_point_sequence with bos token (=> in the paper)
        # i will arbitrarily define that token to be [-1, -1]
        # note that eos token must be ignore when constructing
        # the target point sequence
        target_point_sequence = torch.from_numpy(
            np.vstack([self.BOS_TOKEN, points[idx_sequence[:-1] - 1]])
        )
        return points, target_point_sequence, idx_sequence

    def __len__(self) -> int:
        return len(self.point_sets)


def collate_into_packed_sequence(items: tp.Sequence[_PtrNetItem]) -> _Batch:
    point_sets, target_point_sequences, answers = zip(*items)
    pack = functools.partial(torch.nn.utils.rnn.pack_sequence, enforce_sorted=False)
    return pack(point_sets), pack(target_point_sequences), pack(answers)


class TSPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datadir: str,
        train_params: tp.Tuple[TSP.NPointsT, TSP.AlgorithmT],
        test_params: tp.Tuple[TSP.NPointsT, TSP.AlgorithmT],
        batch_size: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=(datadir,))

        self.datadir = datadir
        # NOTE: the explicit type here is because mypy gets confused otherwise
        self.train_split_params: tp.Tuple[TSP.NPointsT, TSP.SplitT, TSP.AlgorithmT] = (
            train_params[0],
            "train",
            train_params[1],
        )
        self.test_split_params: tp.Tuple[TSP.NPointsT, TSP.SplitT, TSP.AlgorithmT] = (
            test_params[0],
            "test",
            test_params[1],
        )
        self.batch_size = batch_size

    def setup(self, stage: tp.Optional[str] = None) -> None:
        if stage in ("fit", "validate", "test", None):
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
            # num_workers=0,
            collate_fn=collate_into_packed_sequence,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            # for cast explanation see train_dataloader
            dataset=tp.cast(torch.utils.data.Dataset[_Batch], self.tsp_test),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() or 0,
            # num_workers=0,
            collate_fn=collate_into_packed_sequence,
            pin_memory=True,
        )

    # I am using the same dataloader here because the only thing I want
    # to do differently at test time is measure different/more expensive
    # metrics.
    # At train time, val_dataloader is used to kinda check if
    # training makes sense, but the early stopping is done based
    # on train loss. Plus, all data points come from the same distribution
    # so it's whatever.
    test_dataloader = val_dataloader


class ConvexHull(torch.utils.data.Dataset[_PtrNetItem]):
    NPointsT = tp.Literal["5", "10", "50", "200", "500", "5-50"]
    SplitT = tp.Literal["train", "test"]

    _options_to_file: tp.ClassVar[tp.Dict[tp.Tuple[NPointsT, SplitT], str]] = {
        ("50", "train"): "convex_hull_50_train.txt",
        ("5-50", "train"): "convex_hull_5-50_train.txt.zip",
        ("5", "test"): "convex_hull_5_test.txt",
        ("10", "test"): "convex_hull_10_test.txt",
        ("50", "test"): "convex_hull_50_test.txt",
        ("200", "test"): "convex_hull_200_test.txt",
        ("500", "test"): "convex_hull_500_test.txt.zip",
    }

    point_sets: tp.List[npt.NDArray[np.float32]]
    targets: tp.List[npt.NDArray[np.int64]]

    def __init__(self, datadir: str, npoints: NPointsT, split: SplitT) -> None:
        super().__init__()
        self.datadir = datadir
        self.npoints = npoints
        self.split = split

        fname = self._options_to_file.get((npoints, split))
        if fname is None:
            raise ValueError(
                "Invalid option combo. "
                "Check the paper (or the data dir) for available combinations"
            )

        with uopen(pathlib.Path(datadir) / fname) as file:
            point_sets = []
            targets = []
            for line in tqdm.tqdm(file, desc=f"Loading {self}", unit="lines"):
                point_set, target = parse_line(line)
                point_sets.append(point_set)
                targets.append(target)
            self.point_sets = point_sets
            self.targets = targets

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"datadir={self.datadir!r}, "
            f"npoints={self.npoints!r}, "
            f"split={self.split!r})"
        )

    def __len__(self) -> int:
        return len(self.point_sets)

    def __getitem__(self, key: int) -> _PtrNetItem:
        points = torch.from_numpy(self.point_sets[key])
        idx_sequence = torch.from_numpy(self.targets[key])
        target_point_sequence = points[idx_sequence - 1]
        return points, target_point_sequence, idx_sequence


class ConvexHullDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datadir: str,
        train_npoints: ConvexHull.NPointsT,
        test_npoints: ConvexHull.NPointsT,
        batch_size: int,
        val_fraction: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=(datadir,))

        self.datadir = datadir
        self.train_npoints = train_npoints
        self.test_npoints = test_npoints
        self.batch_size = batch_size
        self.val_fraction = val_fraction

    def setup(self, stage: tp.Optional[str] = None) -> None:
        if stage in ("fit", "validate", None):
            train_full = ConvexHull(
                self.datadir, npoints=self.test_npoints, split="train"
            )
            nval = int(len(train_full) * self.val_fraction)
            ntrain = len(train_full) - nval
            (
                self.convex_hull_train,
                self.convex_hull_val,
            ) = torch.utils.data.random_split(train_full, lengths=[ntrain, nval])

        if stage in ("test", None):
            self.convex_hull_test = ConvexHull(
                self.datadir, npoints=self.train_npoints, split="test"
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            dataset=tp.cast(torch.utils.data.Dataset[_Batch], self.convex_hull_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() or 0,
            collate_fn=collate_into_packed_sequence,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            dataset=tp.cast(torch.utils.data.Dataset[_Batch], self.convex_hull_val),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_into_packed_sequence,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            dataset=tp.cast(torch.utils.data.Dataset[_Batch], self.convex_hull_test),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_into_packed_sequence,
        )
