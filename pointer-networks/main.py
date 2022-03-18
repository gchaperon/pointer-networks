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
        # learn_rate=args.learn_rate,
        learn_rate=0.001,
        init_range=(-0.08, 0.08),
    )
    breakpoint()
    trainer = pl.Trainer(gpus=-1, gradient_clip_val=2.0)
    trainer.fit(
        model,
        dm,
        # ckpt_path="lightning_logs/version_15/checkpoints/epoch=7-step=62503.ckpt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--learn-rate", required=True, type=float)
    args = parser.parse_args()
    main(args)
