# Pointer Networks
My replication code for the paper [Pointer Networks](https://arxiv.org/abs/1506.03134).

**tldr:** go to [results](#results)

# The data
The official data is hosted at http://goo.gl/NDcOIG. It can be downloaded using `gdown`
```console
$ gdown --folder --output data https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU
```
or manually using the web interface.

Alternatively, since gdown has sometimes failed me, I'm hosting it myself (hopefully
it's not illegal).  You can download it like so
```console
$ mkdir -p data && wget -qO- https://users.dcc.uchile.cl/~gchapero/datasets/ptr-nets-data.tar.gz | tar -C data -xzv
```

In any case, the data dir should end up looking like this
```
data
├── convex_hull_10_test.txt
├── convex_hull_200_test.txt
├── ...
├── convex_hull_5_test.txt
├── README.txt
├── tsp_10_test_exact.txt
├── tsp_10_train_exact.txt
├── ...
└── tsp_5_train.zip
```

## Notes

Not all data is available, only data for convex hull and tsp. Even then, some splits are
missing. See
[here](https://github.com/gchaperon/replication/blob/63f3d0b73e44f93baad5b6106375208ecec2627d/pointer-networks/ptrnets/data/__init__.py#L31)
for available splits for convex hull and
[here](https://github.com/gchaperon/replication/blob/63f3d0b73e44f93baad5b6106375208ecec2627d/pointer-networks/ptrnets/data/__init__.py#L172)
for tsp splits.

Notice also that some splits in tsp were identified by computing the average solution
distance, since not all of them are tagged with the algorithm used to solve the problem.

# User Guide
The code was tested on python 3.9 using pytorch 1.11. Other versions might work but your
mileage may vary.
## Install
To install the dependencies run
```console
$ pip install -r requirements.txt
```
Note that the pytorch _and_ cuda versions are pinned, so if you are using a gpu
with different cuda capabilities you should modify the requirements file accordingly.

## CLI
The cli has two commands, `train` and `replicate`
```console
$ python -m ptrnets --help
Usage: python -m ptrnets [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  replicate
  train
```

If you want to replicate the results of the paper run 
```console
$ python -m ptrnets replicate
```
This should handle loading the data and running all the experiments of the paper

I also provide a simple `train` command where you can choose to either train the network
for convex hull or tsp, and tweak hyperparameters. Here is the synopsis for training in
convex hull.
```console
$ python -m ptrnets train convex-hull --help
Usage: python -m ptrnets train convex-hull [OPTIONS]

Options:
  --train-npoints [5|10|50|200|500|5-50]
                                  [required]
  --test-npoints [5|10|50|200|500|5-50]
                                  [required]
  --learn-rate FLOAT              [default: 1.0]
  --hidden-size INTEGER           [default: 256]
  --init-range FLOAT...           [default: -0.08, 0.08]
  --batch-size INTEGER            [default: 128]
  --max-grad-norm FLOAT           [default: 2.0]
  --seed INTEGER                  [default: 42]
  --help                          Show this message and exit.
```


# Results
These results were produced by running `python -m ptrnets replicate --write`, and
copy-pasted from the [reports](reports/) dir. The training curves can be seen in the experiment
at [TensorBoard.dev](https://tensorboard.dev/experiment/F80SgWxKRO2xmof6vKALkQ/).
The command took ~12h to run on a single RTX2080 (8gb). I didn't see more than ~5gb of VRAM usage.

The results are considerably off, so I will do some hparam tweaking in the future.

## Convex Hull
Compare with Table 1 of the paper.

| method   | trained n   |   n | accuracy   | area   |
|----------|-------------|-----|------------|--------|
| ptr-net  | 50          |  50 | 41.1%      | 99.8%  |
| ptr-net  | 5-50        |   5 | 85.4%      | 99.0%  |
| ptr-net  | 5-50        |  10 | 69.7%      | 99.6%  |
| ptr-net  | 5-50        |  50 | 30.5%      | 99.8%  |
| ptr-net  | 5-50        | 200 | 0.9%       | 99.4%  |
| ptr-net  | 5-50        | 500 | 0.0%       | 99.0%  |

## TSP
Compare with Table 2 of the paper.

| n                 |   ptr-net |
|-------------------|-----------|
| 5                 |      2.12 |
| 10                |      2.89 |
| 50 (a1 trained)   |      7.77 |
| 5 (5-20 trained)  |      2.18 |
| 10 (5-20 trained) |      3.04 |
| 20 (5-20 trained) |      4.25 |
| 40 (5-20 trained) |      8.4  |
| 50 (5-20 trained) |     11.15 |

# Changes
Here is the list of changes I did to the original paper description

* The paper states that the decoding process for the convex hull task was unconstrained.
  I added some conditions to the decoding process for convex hull, see the details
  [here](https://github.com/gchaperon/replication/blob/125e9d9a2de3790ffb502cc6cd10b8c1578003ca/pointer-networks/ptrnets/model.py#L376).
<!---
* Optimizer: SGD to Adam and add learn rate scheduler. I obtained similar results with
  eoth but Adam converged faster.
--->


# Other implementations
Some were useful, some weren't

* https://github.com/devsisters/pointer-network-tensorflow
* https://github.com/keon/pointer-networks
* https://github.com/vshallc/PtrNets
* https://github.com/Chanlaw/pointer-networks
* https://github.com/devnag/tensorflow-pointer-networks
* https://github.com/ast0414/pointer-networks-pytorch
