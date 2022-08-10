# Pointer Networks
My replication code for the paper [Pointer Networks](https://arxiv.org/abs/1506.03134).

# The data
The official data is hosted at http://goo.gl/NDcOIG. It can be downloaded using `gdown`
```bash
$ gdown --folder --output data https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU
```
or manually using the web interface.

Alternatively, since gdown has sometimes failed me, I'm hosting it myself (hopefully it's not illegal).
You can download it like so
```bash
$ wget -qO- https://users.dcc.uchile.cl/~gchapero/datasets/ptr-nets-data.tar.gz | tar -C data -xzv
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

# Notes on data
## Train
Training splits with data files correspondance

| train split         | file                         |
|---------------------|------------------------------|
| `n=5 (optimal)`     | `tsp_5_train.zip/tsp5.txt`   |
| `n=10 (optimal)`    | `tsp_10_train_exact.txt`     |
| `n=50 (a1 trained)` | `tsp_50_train.zip`           |
| `n=50 (a3 trained)` | `???`                        |
| `n=5-20 (optimal)`  | `tsp_5-20_train.zip/*`       |

Some files were determined by inspecting the average tour distance. In particular, there are 2 `tsp_10_*` train/test pairs and only one `tsp_50_*` train/test pair.

## Test
Data files correspondance with Table 2 of the paper

| n                 | optimal                         | a1                                | a2    | a3    |
|-------------------|---------------------------------|-----------------------------------|-------|-------|
| 5                 | `tsp_5_train.zip/tsp5_test.txt` | `???`                             | `???` | `???` |
| 10                | `tsp_10_test_exact.txt`         | `tsp_10_train.zip/tsp10_test.txt` | `???` | `???` |
| 50 (a1 trained)   | `na`                            | `tsp_50_test.txt.zip`             | `???` | `???` |
| 50 (a3 trained)   | `na`                            | `same`                            | `???` | `???` |
| 5 (5-20 trained)  | `same`                          | `???`                             | `???` | `???` |
| 10 (5-20 trained) | `same`                          | `???`                             | `???` | `???` |
| 20 (5-20 trained) | `???`                           | `tsp_20_test.txt`                 | `???` | `???` |
| 25 (5-20 trained) | `na`                            | `???`                             | `???` | `???` |
| 30 (5-20 trained) | `na`                            | `???`                             | `???` | `???` |
| 40 (5-20 trained) | `na`                            | `tsp_40_test.txt`                 | `???` | `???` |
| 50 (5-20 trained) | `na`                            | `same`                            | `???` | `???` |

`na` means it shouldn't exist, `???` means it should exist but isn't found in `./data` and `same` means its the same file from a previous row.

Again, some were determined by inspecting the average tour distance.

# Changes
Here is the list of changes I did to the original paper description

* Optimizer: SGD to Adam and add learn rate scheduler. I obtained similar results with both but Adam converged faster.

# Other implementations
* https://github.com/devsisters/pointer-network-tensorflow
* https://github.com/keon/pointer-networks
* https://github.com/vshallc/PtrNets
* https://github.com/Chanlaw/pointer-networks
* https://github.com/devnag/tensorflow-pointer-networks
* https://github.com/ast0414/pointer-networks-pytorch

