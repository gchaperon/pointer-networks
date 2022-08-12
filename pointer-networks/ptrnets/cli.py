import optuna
import torch
import pytorch_lightning as pl
import typing as tp
import click
import ptrnets


@click.group()
def cli() -> None:
    pass


@cli.group()
def train() -> None:
    pass


@train.command(name="convex-hull", context_settings={"show_default": True})
@click.option(
    "--train-npoints",
    type=click.Choice(tp.get_args(ptrnets.ConvexHull.NPointsT)),
    required=True,
)
@click.option(
    "--test-npoints",
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
    test_npoints: tp.List[ptrnets.ConvexHull.NPointsT],
    learn_rate: float,
    hidden_size: int,
    init_range: tp.Tuple[float, float],
    batch_size: int,
    max_grad_norm: float,
    seed: int,
) -> None:
    pl.seed_everything(seed, workers=True)
    datamodule = ptrnets.ConvexHullDataModule(
        "data", train_npoints, test_npoints, batch_size
    )
    model = ptrnets.PointerNetworkForConvexHull(
        input_size=2,
        hidden_size=hidden_size,
        learn_rate=learn_rate,
        init_range=init_range,
    )

    _CkptCbKwargs = tp.TypedDict("_CkptCbKwargs", {"monitor": str, "mode": str})
    checkpoint_callback_kwargs: _CkptCbKwargs = {
        "monitor": "val/sequence_accuracy",
        "mode": "max",
    }
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=(
            "epoch={epoch}-"
            "val_sequence_acc={" + checkpoint_callback_kwargs["monitor"] + ":.3f}"
        ),
        auto_insert_metric_name=False,
        **checkpoint_callback_kwargs,
    )
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        logger=pl.loggers.TensorBoardLogger(
            save_dir="logs",
            name="convex-hull",
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
    trainer.test(
        model,
        datamodule=datamodule,
        ckpt_path=checkpoint_callback.best_model_path or None,
    )


def train_single_tsp(
    train_opts,
    test_optss,
    learn_rate,
    hidden_size,
    init_range,
    batch_size,
    max_grad_norm,
    seed,
) -> None:
    pl.seed_everything(seed, workers=True)
    datamodule = ptrnets.TSPDataModule("data", train_opts, test_optss, batch_size)
    model = ptrnets.PointerNetworkForTSP(
        input_size=2,
        hidden_size=hidden_size,
        learn_rate=learn_rate,
        init_range=init_range,
    )
    checkpoint_callback_kwargs = {
        "monitor": "val/sequence_accuracy",
        "mode": "max",
    }
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=(
            "epoch={epoch}-"
            "val_sequence_acc={" + checkpoint_callback_kwargs["monitor"] + ":.3f}"
        ),
        auto_insert_metric_name=False,
        **checkpoint_callback_kwargs,
    )
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        logger=pl.loggers.TensorBoardLogger(
            save_dir="logs",
            name="tsp",
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
    trainer.test(
        model,
        datamodule,
        ckpt_path=checkpoint_callback.best_model_path or None,
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
@click.option("--hidden-size", default=512)
@click.option("--init-range", nargs=2, default=(-0.08, 0.08))
@click.option("--batch-size", default=128)
@click.option("--max-grad-norm", default=2.0)
@click.option("--seed", default=42)
def train_tsp(
    train_opts,
    test_optss,
    learn_rate,
    hidden_size,
    init_range,
    batch_size,
    max_grad_norm,
    seed,
) -> None:
    train_single_tsp(**locals())


@cli.command()
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
@click.option("--ntrials", default=10)
def hpsearch(train_opts, test_optss, ntrials) -> None:
    def objective(trial):
        init_range = trial.suggest_float("init_range", 0.02, 2.0, step=0.01)
        best_seq_acc = train_single_tsp(
            train_opts,
            test_optss,
            trial.suggest_int("lr_mul", 1, 10)
            * 10 ** -trial.suggest_int("lr_exp", 3, 5),
            trial.suggest_int("hidden_size", 256, 1024, step=64),
            (-init_range, init_range),
            trial.suggest_int("batch_size", 64, 1024, step=64),
            trial.suggest_float("max_grad_norm", 1.0, 5.0, step=0.1),
            seed=None,
        )
        return best_seq_acc

    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(), direction="maximize"
    )
    study.optimize(objective, n_trials=ntrials)
