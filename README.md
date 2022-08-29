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
$ mkdir -p data && wget -O- https://users.dcc.uchile.cl/~gchapero/datasets/ptr-nets-data.tar.gz | tar -C data -xz
```

In any case, the project root should have a data dir that looks like this:
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

Not all data is available, only data for convex hull and tsp. Even then, some
splits are missing. See
[here](https://github.com/gchaperon/pointer-networks/blob/master/ptrnets/data/__init__.py#L32)
for available splits for convex hull and
[here](https://github.com/gchaperon/pointer-networks/blob/master/ptrnets/data/__init__.py#L171)
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
at [TensorBoard.dev](https://tensorboard.dev/experiment/O5QdzV6OQMy2TXf2kLjp5Q/).
The command took ~6h to run on a single RTX3090 (24gb). I didn't see more than ~5gb of VRAM usage.


## Convex Hull
Compare with Table 1 of the paper.

| method   | trained n   |   n | accuracy   | area   |
|----------|-------------|-----|------------|--------|
| ptr-net  | 50          |  50 | 66.1%      | 100.0% |
| ptr-net  | 5-50        |   5 | 93.8%      | 99.8%  |
| ptr-net  | 5-50        |  10 | 87.9%      | 99.9%  |
| ptr-net  | 5-50        |  50 | 63.1%      | 100.0% |
| ptr-net  | 5-50        | 200 | 10.1%      | 99.9%  |
| ptr-net  | 5-50        | 500 | 0.1%       | 99.6%  |

## TSP
Compare with Table 2 of the paper.

| n                 |   ptr-net |
|-------------------|-----------|
| 5                 |      2.12 |
| 10                |      2.88 |
| 50 (a1 trained)   |      6.7  |
| 5 (5-20 trained)  |      2.18 |
| 10 (5-20 trained) |      3.05 |
| 20 (5-20 trained) |      4.23 |
| 40 (5-20 trained) |      7.25 |
| 50 (5-20 trained) |      9.14 |

# Changes
Here is the list of changes I did to the original paper description

* Convex Hull decoding: The paper states that the decoding process for the
  convex hull task was unconstrained.  I added some conditions to the decoding
  process for convex hull, see the details
  [here](https://github.com/gchaperon/pointer-networks/blob/master/ptrnets/model.py#L376).

* Optimizer and Scheduler: Changed SGD to Adam, learn rate from 1.0 to 1e-3
  (inline with what's recommended for Adam) and added a learn rate scheduler
  (exponential decay). The optimizer largely improved and sped up convergence,
  and the scheduler helped with reducing noise in the final steps of training.

# Other implementations
Some were useful, some weren't

* https://github.com/devsisters/pointer-network-tensorflow
* https://github.com/keon/pointer-networks
* https://github.com/vshallc/PtrNets
* https://github.com/Chanlaw/pointer-networks
* https://github.com/devnag/tensorflow-pointer-networks
* https://github.com/ast0414/pointer-networks-pytorch
