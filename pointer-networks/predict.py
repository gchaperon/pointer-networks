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
def main(experiment_dir, test_npoints, batch_size):
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
    # assert 100 % batch_size == 0
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        deterministic=True,
        limit_test_batches=100 // batch_size,
    )

    dm = ptrnets.ConvexHullDataModule("data", "50", test_npoints, 1)
    dm.setup("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ans_single = []
    assert 100 % batch_size == 0
    for i, b in itertools.islice(enumerate(tqdm.tqdm(dm.test_dataloader()[0])), 0, 100):
        enc_in, dec_in, tgt = (el.to(device) for el in b)
        ans_single.append(model.batch_beam_search(enc_in))

    dm = ptrnets.ConvexHullDataModule("data", "50", test_npoints, batch_size)
    dm.setup("test")
    ans_batch = []
    for i, b in itertools.islice(
        enumerate(tqdm.tqdm(dm.test_dataloader()[0])), 0, 100 // batch_size
    ):
        enc_in, dec_in, tgt = (el.to(device) for el in b)
        ans_batch.append(model.batch_beam_search(enc_in))

    max_len = max(
        max(len(s.batch_sizes) for s in ans_single),
        max(len(s.batch_sizes) for s in ans_batch),
    )
    ans_single = torch.vstack(
        [
            torch.nn.utils.rnn.pad_packed_sequence(
                s, batch_first=True, total_length=max_len
            )[0]
            for s in ans_single
        ]
    )
    ans_batch = torch.vstack(
        [
            torch.nn.utils.rnn.pad_packed_sequence(
                s, batch_first=True, total_length=max_len
            )[0]
            for s in ans_batch
        ]
    )
    diff = (~(ans_single == ans_batch).all(dim=1)).nonzero()
    breakpoint()

    # trainer.test(model, datamodule)

    # datamodule.setup("test")
    # inputs, decoder_inputs, targets = next(iter(datamodule.test_dataloader()[0]))
    # res = model.batch_beam_search(inputs)
    # breakpoint()
    # pass


if __name__ == "__main__":
    main()
