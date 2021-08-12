# Advanced options

Several command line options can be passed to `dp train`, which can be checked with
```bash
$ dp train --help
```
An explanation will be provided
```
positional arguments:
  INPUT                 the input json database

optional arguments:
  -h, --help            show this help message and exit
  --init-model INIT_MODEL
                        Initialize a model by the provided checkpoint
  --restart RESTART     Restart the training from the provided checkpoint
```

**`--init-model model.ckpt`**, initializes the model training with an existing model that is stored in the checkpoint `model.ckpt`, the network architectures should match.

**`--restart model.ckpt`**, continues the training from the checkpoint `model.ckpt`.

On some resources limited machines, one may want to control the number of threads used by DeePMD-kit. This is achieved by three environmental variables: `OMP_NUM_THREADS`, `TF_INTRA_OP_PARALLELISM_THREADS` and `TF_INTER_OP_PARALLELISM_THREADS`. `OMP_NUM_THREADS` controls the multithreading of DeePMD-kit implemented operations. `TF_INTRA_OP_PARALLELISM_THREADS` and `TF_INTER_OP_PARALLELISM_THREADS` controls `intra_op_parallelism_threads` and `inter_op_parallelism_threads`, which are  Tensorflow configurations for multithreading. An explanation is found [here](https://stackoverflow.com/questions/41233635/meaning-of-inter-op-parallelism-threads-and-intra-op-parallelism-threads).

For example if you wish to use 3 cores of 2 CPUs on one node, you may set the environmental variables and run DeePMD-kit as follows:
```bash
export OMP_NUM_THREADS=6
export TF_INTRA_OP_PARALLELISM_THREADS=3
export TF_INTER_OP_PARALLELISM_THREADS=2
dp train input.json
```

One can set other environmental variables:

| Environment variables | Allowed value          | Default value | Usage                      |
| --------------------- | ---------------------- | ------------- | -------------------------- |
| DP_INTERFACE_PREC     | `high`, `low`          | `high`        | Control high (double) or low (float) precision of training. |