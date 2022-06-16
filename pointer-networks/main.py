import pathlib
import click
import typing as tp
import torch
import pytorch_lightning as pl

import ptrnets


# breakpoint()
@click.command()
@click.option(
    "--train-split",
    type=(
        click.Choice(tp.get_args(ptrnets.TSP.NPointsT)),
        click.Choice(tp.get_args(ptrnets.TSP.AlgorithmT)),
    ),
    required=True,
)
@click.option(
    "--test-split",
    type=(
        click.Choice(tp.get_args(ptrnets.TSP.NPointsT)),
        click.Choice(tp.get_args(ptrnets.TSP.AlgorithmT)),
    ),
    required=True,
)
@click.option("--learn-rate", default=0.001)
@click.option("--hidden-size", default=256)
@click.option("--init-range", nargs=2, default=(-0.08, 0.08))
@click.option("--batch-size", default=128)
@click.option("--max-grad-norm", default=2.0)
@click.option("--limit-train-batches", default=1.0)
def train_tsp(
    train_split: tp.Tuple[ptrnets.TSP.NPointsT, ptrnets.TSP.AlgorithmT],
    test_split: tp.Tuple[ptrnets.TSP.NPointsT, ptrnets.TSP.AlgorithmT],
    learn_rate: float,
    hidden_size: int,
    init_range: tp.Tuple[float, float],
    batch_size: int,
    max_grad_norm: float,
    limit_train_batches: float,
) -> tp.List[tp.Dict[str, float]]:
    dm = ptrnets.TSPDataModule(
        datadir="data",
        train_params=train_split,
        test_params=test_split,
        batch_size=batch_size,
    )
    model = ptrnets.PointerNetworkForTSP(
        input_size=2,
        hidden_size=hidden_size,
        learn_rate=learn_rate,
        init_range=init_range,
    )
    trainer = pl.Trainer(
        gpus=-1,
        gradient_clip_val=max_grad_norm,
        limit_train_batches=limit_train_batches,
        limit_val_batches=0,
        callbacks=[
            # pl.callbacks.EarlyStopping(monitor="train_loss", patience=1),
            pl.callbacks.ModelCheckpoint(monitor="train_loss"),
        ],
    )
    trainer.fit(model, dm)
    # test_results = trainer.test(model, dm)
    # return test_results


@click.command(context_settings={"show_default": True})
@click.option(
    "--train-npoints",
    type=click.Choice(tp.get_args(ptrnets.ConvexHull.NPointsT)),
    required=True,
)
@click.option(
    "--test-npoints",
    type=click.Choice(tp.get_args(ptrnets.ConvexHull.NPointsT)),
    required=True,
)
@click.option("--learn-rate", default=0.001)
@click.option("--hidden-size", default=256)
@click.option("--init-range", nargs=2, default=(-0.08, 0.08))
@click.option("--batch-size", default=128)
@click.option("--max-grad-norm", default=2.0)
def train_convex_hull(
    train_npoints: ptrnets.ConvexHull.NPointsT,
    test_npoints: ptrnets.ConvexHull.NPointsT,
    learn_rate: float,
    hidden_size: int,
    init_range: tp.Tuple[float, float],
    batch_size: int,
    max_grad_norm: float,
) -> tp.List[tp.Dict[str, float]]:

    datamodule = ptrnets.ConvexHullDataModule(
        "data", train_npoints, test_npoints, batch_size
    )
    model = ptrnets.PointerNetworkForConvexHull(
        input_size=2,
        hidden_size=hidden_size,
        learn_rate=learn_rate,
        init_range=init_range,
    )

    checkpoint_callback_kwargs = {
        "monitor": "val/sequence_acc",
        "mode": "max",
    }
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**checkpoint_callback_kwargs)
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        logger=pl.loggers.TensorBoardLogger(
            save_dir="logs",
            name="convex-hull",
            default_hp_metric=False,
        ),
        gradient_clip_val=max_grad_norm,
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.EarlyStopping(
                **checkpoint_callback_kwargs,
                patience=10,
            ),
            checkpoint_callback,
        ],
        max_epochs=200,
        deterministic=True,
    )
    # trainer.fit(model, datamodule)
    trainer.validate(
        model,
        datamodule,
        ckpt_path=next(
            pathlib.Path("r2_logs/convex-hull/version_0/checkpoints").iterdir()
        ),
    )
    results = trainer.test(
        model,
        datamodule=datamodule,
        # ckpt_path=checkpoint_callback.best_model_path,
        ckpt_path=next(
            pathlib.Path("r2_logs/convex-hull/version_0/checkpoints").iterdir()
        ),
    )
    print(results)
    return results


@click.group()
def train() -> None:
    pass


train.add_command(train_tsp, name="tsp")
train.add_command(train_convex_hull, name="convex-hull")


@click.group()
def replicate() -> None:
    pass


@replicate.command(name="tsp")
def replicate_tsp() -> None:
    pass


@replicate.command(name="convex-hull")
def replicate_convex_hull() -> None:
    pass


@click.group()
def optimize() -> None:
    pass


@optimize.command(name="tsp")
def optimize_tsp() -> None:
    pass


@optimize.command(name="convex-hull")
def optimize_convex_hull() -> None:
    pass


@click.group()
def main() -> None:
    pass


main.add_command(train)
main.add_command(replicate)
main.add_command(optimize)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*(val|test)_dataloader.*num_workers",
        category=UserWarning,
    )

    pl.seed_everything(42, workers=True)
    res = train_convex_hull()
    print(res)
