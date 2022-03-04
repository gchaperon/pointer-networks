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

In any case, the data dir should look like this
```
data
├── convex_hull_10_test.txt
├── convex_hull_200_test.txt
├── convex_hull_500_test.txt.zip
├── convex_hull_50_test.txt
├── convex_hull_50_train.txt
├── convex_hull_5-50_train.txt.zip
├── ...
└── tsp_5_train.zip
```

# Notes on data
Training splits with data files correspondance

| train split         | file                         |
|---------------------|------------------------------|
| `n=5 (optimal)`     | `tsp_5_train.zip/tsp5.txt`   |
| `n=10 (optimal)`    | `tsp_10_train_exact.txt`     |
| `n=50 (a1 trained)` | `tsp_50_train.zip/tsp50.txt` |
| `n=50 (a3 trained)` | `???`                        |
| `n=5-20 (optimal)`  | `tsp_5-20_train.zip/*`       |

Some files where determined by inspecting the average tour distance. In particular, there are 2 `tsp_10_*` train/test pairs and only one `tsp_50_*` train/test pair.

