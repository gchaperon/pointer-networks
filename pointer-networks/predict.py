import tqdm
import itertools
import os
import torch
import pytorch_lightning as pl
import pathlib
import typing as tp
import ptrnets
import ptrnets.metrics as metrics
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
def main(experiment_dir, test_npoints, batch_size) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(42, workers=True)

    print(experiment_dir, test_npoints, batch_size)
    ckpt_path = str(next((pathlib.Path(experiment_dir) / "checkpoints").iterdir()))
    print(ckpt_path)
    model = ptrnets.PointerNetworkForConvexHull.load_from_checkpoint(ckpt_path)
    datamodule = ptrnets.ConvexHullDataModule("data", "50", test_npoints, batch_size)
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        deterministic=True,
        limit_test_batches=100 // batch_size,
    )

    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
