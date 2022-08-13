import logging
import pathlib
import typing as tp

import click
import pytorch_lightning as pl
import tabulate
import torch

import ptrnets


@click.group()
def cli() -> None:
    pass


@cli.group()
def train() -> None:
    pass


def train_single(
    model: ptrnets.PointerNetwork,
    datamodule: tp.Union[ptrnets.ConvexHullDataModule, ptrnets.TSPDataModule],
    experiment_name: str,
    max_grad_norm: float,
) -> tp.List[tp.Dict["str", float]]:
    _CkptCbKwargs = tp.TypedDict("_CkptCbKwargs", {"monitor": str, "mode": str})
    checkpoint_callback_kwargs: _CkptCbKwargs = {
        "monitor": "val/loss",
        "mode": "min",
    }
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="epoch={epoch}-val_loss={val/loss:.3f}",
        auto_insert_metric_name=False,
        **checkpoint_callback_kwargs,
    )
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        logger=pl.loggers.TensorBoardLogger(
            save_dir="logs",
            name=experiment_name,
            default_hp_metric=False,
        ),
        gradient_clip_val=max_grad_norm,
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.EarlyStopping(
                patience=3,
                min_delta=1e-3,
                **checkpoint_callback_kwargs,
            ),
            checkpoint_callback,
        ],
        max_epochs=1000,
    )
    trainer.fit(model, datamodule)
    test_results = trainer.test(
        model,
        datamodule,
        ckpt_path=checkpoint_callback.best_model_path or None,
        verbose=logging.getLogger().level < logging.ERROR,
    )
    return test_results


@train.command(name="convex-hull", context_settings={"show_default": True})
@click.option(
    "--train-npoints",
    type=click.Choice(tp.get_args(ptrnets.ConvexHull.NPointsT)),
    required=True,
)
@click.option(
    "--test-npoints",
    "test_npointss",
    type=click.Choice(tp.get_args(ptrnets.ConvexHull.NPointsT)),
    required=True,
    multiple=True,
)
@click.option("--learn-rate", default=1.0)
@click.option("--hidden-size", default=256)
@click.option("--init-range", nargs=2, default=(-0.08, 0.08))
@click.option("--batch-size", default=128)
@click.option("--max-grad-norm", default=2.0)
@click.option("--seed", default=42)
def train_convex_hull(
    train_npoints: ptrnets.ConvexHull.NPointsT,
    test_npointss: tp.List[ptrnets.ConvexHull.NPointsT],
    learn_rate: float,
    hidden_size: int,
    init_range: tp.Tuple[float, float],
    batch_size: int,
    max_grad_norm: float,
    seed: int,
) -> None:
    pl.seed_everything(seed, workers=True)
    datamodule = ptrnets.ConvexHullDataModule(
        "data", train_npoints, test_npointss, batch_size
    )
    model = ptrnets.PointerNetworkForConvexHull(
        input_size=2,
        hidden_size=hidden_size,
        learn_rate=learn_rate,
        init_range=init_range,
    )
    train_single(
        model,
        datamodule,
        "convex-hull",
        max_grad_norm,
    )


@train.command(name="tsp", context_settings={"show_default": True})
@click.option(
    "--train-opts",
    type=(
        click.Choice(tp.get_args(ptrnets.TSP.NPointsT)),
        click.Choice(tp.get_args(ptrnets.TSP.AlgorithmT)),
    ),
    required=True,
)
@click.option(
    "--test-opts",
    "test_optss",
    type=(
        click.Choice(tp.get_args(ptrnets.TSP.NPointsT)),
        click.Choice(tp.get_args(ptrnets.TSP.AlgorithmT)),
    ),
    required=True,
    multiple=True,
)
@click.option("--learn-rate", default=1.0)
@click.option("--hidden-size", default=256)
@click.option("--init-range", nargs=2, default=(-0.08, 0.08))
@click.option("--batch-size", default=128)
@click.option("--max-grad-norm", default=2.0)
@click.option("--seed", default=42)
def train_tsp(
    train_opts: tp.Tuple[ptrnets.TSP.NPointsT, ptrnets.TSP.AlgorithmT],
    test_optss: tp.List[tp.Tuple[ptrnets.TSP.NPointsT, ptrnets.TSP.AlgorithmT]],
    learn_rate: float,
    hidden_size: int,
    init_range: tp.Tuple[float, float],
    batch_size: int,
    max_grad_norm: float,
    seed: int,
) -> None:
    pl.seed_everything(seed, workers=True)
    datamodule = ptrnets.TSPDataModule("data", train_opts, test_optss, batch_size)
    model = ptrnets.PointerNetworkForTSP(
        input_size=2,
        hidden_size=hidden_size,
        learn_rate=learn_rate,
        init_range=init_range,
    )
    train_single(
        model,
        datamodule,
        "tsp",
        max_grad_norm,
    )


@cli.command()
@click.option("--write", is_flag=True)
def replicate(write: bool) -> None:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    pl.seed_everything(42, workers=True)

    # Convex Hull results
    # ===================
    print("Running Convex Hull experiments")
    table1: tp.List[tp.Dict[str, str]] = []
    for i, (train_npoints, test_npointss) in enumerate(
        [
            ("50", ["50"]),
            ("5-50", ["5", "10", "50", "200", "500"]),
        ]
    ):
        print(f"Running exp {i} with {train_npoints=} and {test_npointss=}")
        # NOTE: on type ignore, mypy complains because train_npoints has type
        # str, but it's unable to see that I am only using valid values for
        # ntrain_points. Same for test_npointss.
        convex_hull_datamodule = ptrnets.ConvexHullDataModule(
            "data", train_npoints, test_npointss, batch_size=128  # type:ignore
        )
        convex_hull_model = ptrnets.PointerNetworkForConvexHull(
            input_size=2, hidden_size=256, learn_rate=1.0, init_range=(-0.08, 0.08)
        )
        results = train_single(
            convex_hull_model, convex_hull_datamodule, "convex-hull", max_grad_norm=2.0
        )
        for test_npoints, result in zip(test_npointss, results):
            # if there are multiple dataloader pytorch lightning adds extra
            # text to the end of each metric key, so simple pick the first key
            # that contains what im looking for
            acc_key = next(k for k in result.keys() if "test/polygon_accuracy" in k)
            area_key = next(k for k in result.keys() if "test/polygon_coverage" in k)
            poly_area = result[area_key]
            table1.append(
                dict(
                    method="ptr-net",
                    trained_n=train_npoints,
                    n=test_npoints,
                    accuracy=format(result[acc_key], ".1%"),
                    area=format(poly_area, ".1%") if poly_area > 0.0 else "FAIL",
                )
            )

    # TSP results
    # ===========
    print("Running TSP experiments")
    table2: tp.List[tp.Dict[str, str]] = []
    for i, (train_opts, test_optss) in enumerate(
        [
            (("5", "optimal"), [("5", "optimal")]),
            (("10", "optimal"), [("10", "optimal")]),
            (("50", "a1"), [("50", "a1")]),
            (
                ("5-20", "optimal"),
                [
                    ("5", "optimal"),
                    ("10", "optimal"),
                    ("20", "a1"),
                    ("40", "a1"),
                    ("50", "a1"),
                ],
            ),
        ]
    ):
        print(f"Running exp {i} with {train_opts=} and {test_optss=}")
        tsp_datamodule = ptrnets.TSPDataModule(
            "data", train_opts, test_optss, batch_size=128  # type: ignore
        )
        tsp_model = ptrnets.PointerNetworkForTSP(
            input_size=2, hidden_size=256, learn_rate=1.0, init_range=(-0.08, 0.08)
        )
        results = train_single(tsp_model, tsp_datamodule, "tsp", max_grad_norm=2.0)
        for test_opts, result in zip(test_optss, results):
            tour_dist_key = next(k for k in result.keys() if "tour_distance" in k)
            if train_opts[0] == "50":
                n_value = "50 (a1 trained)"
            elif train_opts[0] == "5-20":
                n_value = f"{test_opts[0]} (5-20 trained)"
            else:
                n_value = train_opts[0]
            table2.append(
                {"n": n_value, "ptr-net": format(result[tour_dist_key], ".2f")}
            )

    # Print to screen and maybe write
    # ===============================
    # sometimes tqdm does some weird stuff, so flush streams
    table1_str = tabulate.tabulate(table1, headers="keys", tablefmt="github")
    table2_str = tabulate.tabulate(table2, headers="keys", tablefmt="github")
    print("\n\n" + "=" * 10 + " RESULTS " + "=" * 10)
    print("\nCompare with Table 1 in the paper.")
    print(table1_str)
    print("\nCompare with Table 2 in the paper.")
    print(table2_str)

    if write:
        reports_path = pathlib.Path("reports")
        reports_path.mkdir(exist_ok=True)
        with open(reports_path / "table1.md", "w") as fout:
            fout.write(table1_str)
        with open(reports_path / "table2.md", "w") as fout:
            fout.write(table2_str)

    # print("comming soon...")
