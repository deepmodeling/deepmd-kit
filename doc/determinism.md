# Running DeepMD in full deterministic mode

With the default settings DeepMD does not guaranty that two successive training on the same data will return the same results. The results will also depend on the processing units GPU vs CPU and variations might also be observed between different families of GPUs. This document explains how to set up DeepMD runs to get reproducible results for a given set of training data and hardware architecture. It only apply to the forces and stress calculations during the training and inference phases.

The GPU kernels calculating the forces and stress in DeepMD are deterministic. Calls to the TensorFlow API however, do not guaranty that unless a set of environment variables affecting its executation are set up at runtime or if specific api calls are used during the TensorFlow initialization steps. The most important environment variable is `TF_DETERMINITIC_OPS` that selects the deterministic variants of tensorflow GPU functions if set to 1. Two others variables controlling the tensorflow threading; `TF_INTER_OP_PARALLELISM_THREADS` and `TF_INTER_OP_PARALLELISM_THREADS`; should be set to 0. More information about running tensorflow in deterministic mode and it imply can be found [here](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism). The `OMP_NUM_THREADS` variable seems to have less or no impact when the GPU version of DeepMD is used.

Adding these three lines of code in the run scripts is enough to get reproducible results on the same hardware.

```[sh]
export TF_DETERMINISTIC_OPS=1
export TF_INTER_OP_PARALLELISM_THREADS=0
export TF_INTER_OP_PARALLELISM_THREADS=0
```


