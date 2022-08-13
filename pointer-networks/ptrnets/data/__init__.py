import functools
import logging
import pathlib
import typing as tp

import joblib
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch

from .utils import load_file

_PtrNetItem = tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_Batch = tp.Tuple[
    torch.nn.utils.rnn.PackedSequence,
    torch.nn.utils.rnn.PackedSequence,
    torch.nn.utils.rnn.PackedSequence,
]


def collate_into_packed_sequence(items: tp.Sequence[_PtrNetItem]) -> _Batch:
    point_sets, target_point_sequences, answers = zip(*items)
    pack = functools.partial(torch.nn.utils.rnn.pack_sequence, enforce_sorted=False)
    return pack(point_sets), pack(target_point_sequences), pack(answers)


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

    datadir: str
    npoints: NPointsT
    split: SplitT

    point_sets: npt.NDArray[np.float32]
    targets: npt.NDArray[np.int64]

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

        print(f"Loading {self!r}")
        memory = joblib.Memory(
            datadir, verbose=10 if logging.getLogger().level < logging.ERROR else 0
        )
        self.point_sets, self.targets = memory.cache(load_file)(
            pathlib.Path(datadir) / fname
        )

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
        points = torch.from_numpy(self.point_sets[key].compressed().reshape(-1, 2))
        idx_sequence = torch.from_numpy(self.targets[key].compressed())
        target_point_sequence = points[idx_sequence - 1]
        return (
            points,
            target_point_sequence,
            torch.hstack([idx_sequence, torch.tensor(0)]),
        )


class ConvexHullDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datadir: str,
        train_npoints: ConvexHull.NPointsT,
        test_npointss: tp.List[ConvexHull.NPointsT],
        batch_size: int,
        val_fraction: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))

        self.datadir = datadir
        self.train_npoints = train_npoints
        self.test_npointss = test_npointss
        self.batch_size = batch_size
        self.val_fraction = val_fraction

        assert len(set(test_npointss)) == len(
            test_npointss
        ), "there shouldn't be repeated splits in test_npointss"

    def setup(self, stage: tp.Optional[str] = None) -> None:
        if stage in ("fit", "validate", None):
            train_full = ConvexHull(
                self.datadir, npoints=self.train_npoints, split="train"
            )
            nval = int(len(train_full) * self.val_fraction)
            ntrain = len(train_full) - nval
            (
                self.convex_hull_train,
                self.convex_hull_val,
            ) = torch.utils.data.random_split(train_full, lengths=[ntrain, nval])

        if stage in ("test", None):
            self.convex_hull_test_datasets = [
                ConvexHull(self.datadir, npoints=npoints, split="test")
                for npoints in self.test_npointss
            ]

    def train_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            # NOTE: for cast explanation see this
            # https://github.com/pytorch/pytorch/blob/v1.11.0/torch/utils/data/dataloader.py#L30
            dataset=tp.cast(torch.utils.data.Dataset[_Batch], self.convex_hull_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_into_packed_sequence,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            dataset=tp.cast(torch.utils.data.Dataset[_Batch], self.convex_hull_val),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_into_packed_sequence,
        )

    def test_dataloader(self) -> tp.List[torch.utils.data.DataLoader[_Batch]]:
        return [
            torch.utils.data.DataLoader(
                dataset=tp.cast(torch.utils.data.Dataset[_Batch], dataset),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=collate_into_packed_sequence,
            )
            for dataset in self.convex_hull_test_datasets
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
    }

    datadir: str
    npoint: NPointsT
    split: SplitT
    algorithm: AlgorithmT

    point_sets: npt.NDArray[np.float32]
    targets: npt.NDArray[np.int64]

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

        print(f"Loading {self!r}")
        fname, *extras = open_args
        memory = joblib.Memory(
            datadir, verbose=10 if logging.getLogger().level < logging.ERROR else 0
        )
        self.point_sets, self.targets = memory.cache(load_file)(
            pathlib.Path(datadir) / fname, *extras
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"datadir={self.datadir!r}, "
            f"npoints={self.npoints!r}, "
            f"split={self.split!r}, "
            f"algorithm={self.algorithm!r})"
        )

    def __len__(self) -> int:
        return len(self.point_sets)

    def __getitem__(self, key: int) -> _PtrNetItem:
        points = torch.from_numpy(self.point_sets[key].compressed().reshape(-1, 2))
        idx_sequence = torch.from_numpy(self.targets[key].compressed())
        target_point_sequence = points[idx_sequence - 1]
        return (
            points,
            target_point_sequence,
            torch.hstack([idx_sequence, torch.tensor(0)]),
        )


class TSPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datadir: str,
        train_opts: tp.Tuple[TSP.NPointsT, TSP.AlgorithmT],
        test_optss: tp.List[tp.Tuple[TSP.NPointsT, TSP.AlgorithmT]],
        batch_size: int,
        val_fraction: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))

        self.datadir = datadir
        self.train_opts = train_opts
        self.test_optss = test_optss
        self.batch_size = batch_size
        self.val_fraction = val_fraction

    def setup(self, stage: tp.Optional[str] = None) -> None:
        if stage in ("fit", "validate", None):
            train_full = TSP(
                self.datadir,
                npoints=self.train_opts[0],
                split="train",
                algorithm=self.train_opts[1],
            )
            nval = int(len(train_full) * self.val_fraction)
            ntrain = len(train_full) - nval
            self.tsp_train, self.tsp_val = torch.utils.data.random_split(
                train_full, lengths=[ntrain, nval]
            )
        if stage in ("test", None):
            self.tsp_test_datasets = [
                TSP(self.datadir, npoints=opts[0], split="test", algorithm=opts[1])
                for opts in self.test_optss
            ]

    def train_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            # NOTE: for cast explanation see this
            # https://github.com/pytorch/pytorch/blob/v1.11.0/torch/utils/data/dataloader.py#L30
            dataset=tp.cast(torch.utils.data.Dataset[_Batch], self.tsp_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_into_packed_sequence,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            dataset=tp.cast(torch.utils.data.Dataset[_Batch], self.tsp_val),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_into_packed_sequence,
        )

    def test_dataloader(self) -> tp.List[torch.utils.data.DataLoader[_Batch]]:
        return [
            torch.utils.data.DataLoader(
                dataset=tp.cast(torch.utils.data.Dataset[_Batch], dataset),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=collate_into_packed_sequence,
            )
            for dataset in self.tsp_test_datasets
        ]
