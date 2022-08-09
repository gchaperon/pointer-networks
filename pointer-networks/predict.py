import torch
import pytorch_lightning as pl
import pathlib
import typing as tp
import ptrnets
import click


@click.command()
@click.argument("experiment-dir")
@click.option(
    "--test-npoints",
    type=click.Choice(tp.get_args(ptrnets.ConvexHull.NPointsT)),
    required=True,
    multiple=True,
)
@click.option("--batch-size", default=128)
@click.option("--limit-batches", type=int)
def main(
    experiment_dir: str,
    test_npoints: str,
    batch_size: int,
    limit_batches: tp.Optional[int] = None,
) -> None:

    print(experiment_dir, test_npoints, batch_size)
    ckpt_path = str(next((pathlib.Path(experiment_dir) / "checkpoints").iterdir()))
    print(ckpt_path)
    model = ptrnets.PointerNetworkForConvexHull.load_from_checkpoint(ckpt_path)
    datamodule = ptrnets.ConvexHullDataModule("data", "50", test_npoints, batch_size)
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        limit_test_batches=limit_batches or 1.0,
    )

    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
