#!/usr/bin/env python3
import torch
import argparse
import pytorch_lightning as pl
import ptrnets
import tqdm


def main(args=None) -> None:
    dm = ptrnets.TSPDataModule(
        datadir="data",
        train_params=(5, "optimal"),
        test_params=(5, "optimal"),
        batch_size=128,
    )
    dm.setup()
#    for _ in tqdm.tqdm(dm.train_dataloader()):
#        pass
    model = ptrnets.PointerNetwork(
        input_size=2,
        hidden_size=512,
        learn_rate=args.learn_rate,
        init_range=(-0.08, 0.08),
    )

    trainer = pl.Trainer(gpus=-1, gradient_clip_val=2.0)
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learn-rate", required=True, type=float)
    args = parser.parse_args()
    main(args)
