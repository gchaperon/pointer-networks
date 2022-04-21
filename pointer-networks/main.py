import click
import typing as tp
import pytorch_lightning as pl

import ptrnets


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
@click.option("--hidden-size", default=512)
@click.option("--init-range", nargs=2, default=(-0.08, 0.08))
@click.option("--batch-size", default=128)
@click.option("--max-grad-norm", default=2.0)
def main_tsp(
    train_split: tp.Tuple[ptrnets.TSP.NPointsT, ptrnets.TSP.AlgorithmT],
    test_split: tp.Tuple[ptrnets.TSP.NPointsT, ptrnets.TSP.AlgorithmT],
    learn_rate: float,
    hidden_size: int,
    init_range: tp.Tuple[float, float],
    batch_size: int,
    max_grad_norm: float,
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
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="train_loss", patience=1),
            pl.callbacks.ModelCheckpoint(monitor="train_loss"),
        ],
    )
    trainer.fit(model, dm)
    test_results = trainer.test(model, dm)
    return test_results


@click.command()
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
@click.option("--hidden-size", default=512)
@click.option("--init-range", nargs=2, default=(-0.08, 0.08))
@click.option("--batch-size", default=128)
@click.option("--max-grad-norm", default=2.0)
def main_convex_hull(
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
    trainer = pl.Trainer(
        gpus=-1,
        gradient_clip_val=max_grad_norm,
        # limit_test_batches=2,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="train_loss", patience=1),
            pl.callbacks.ModelCheckpoint(monitor="train_loss"),
        ],
    )
    trainer.fit(model, datamodule)
    results = trainer.test(model, datamodule=datamodule)
    return results


if __name__ == "__main__":
    main_convex_hull()
