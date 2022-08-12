import typing as tp

import click
import pytorch_lightning as pl
import torch

import ptrnets

# import typing_extensions as tpx


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
    learn_rate: float,
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
                patience=5,
                **checkpoint_callback_kwargs,
            ),
            checkpoint_callback,
        ],
        max_epochs=1000,
        deterministic=True,
    )
    trainer.fit(model, datamodule)
    test_results = trainer.test(
        model,
        datamodule,
        ckpt_path=checkpoint_callback.best_model_path or None,
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
@click.option("--learn-rate", default=0.001)
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
        learn_rate,
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
@click.option("--learn-rate", default=1e-3)
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
        learn_rate,
        max_grad_norm,
    )


@cli.command()
def replicate() -> None:
    print("comming soon...")
