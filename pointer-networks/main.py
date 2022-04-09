#!/usr/bin/env python3
import torch
from torch.nn.utils.rnn import pad_packed_sequence
import argparse
import pytorch_lightning as pl
import ptrnets
import tqdm


def main(args=None) -> None:
    dm = ptrnets.TSPDataModule(
        datadir="data",
        train_params=tuple(args.train_split),
        test_params=tuple(args.test_split),
        batch_size=args.batch_size,
    )
    model = ptrnets.PointerNetworkForTSP(
        input_size=2,
        hidden_size=args.hidden_size,
        learn_rate=args.learn_rate,
        init_range=args.init_range,
    )
    trainer = pl.Trainer(
        gpus=-1,
        gradient_clip_val=args.max_grad_norm,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="train_loss", patience=2),
            pl.callbacks.ModelCheckpoint(monitor="train_loss"),
        ],
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learn-rate", default=0.001, type=float)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--init-range", nargs=2, default=(-0.08, 0.08), type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--max-grad-norm", default=2.0, type=float)

    parser.add_argument("--train-split", nargs=2, required=True)
    parser.add_argument("--test-split", nargs=2, required=True)

    args = parser.parse_args()
    main(args)
