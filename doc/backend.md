# Backend

## Supported backends

DeePMD-kit supports multiple backends: TensorFlow and PyTorch.
To use DeePMD-kit, you must install at least one backend.
Each backend does not support all features.
In the documentation, TensorFlow {{ tensorflow_icon }} and PyTorch {{ pytorch_icon }} icons are used to mark whether a backend supports a feature.

### TensorFlow {{ tensorflow_icon }}

- Model filename extension: `.pb`
- Checkpoint filename extension: `.meta`, `.index`, `.data-00000-of-00001`

[TensorFlow](https://tensorflow.org) 2.7 or above is required, since NumPy 1.21 or above is required.
DeePMD-kit does not use the TensorFlow v2 API but uses the TensorFlow v1 API (`tf.compat.v1`) in the graph mode.

### PyTorch {{ pytorch_icon }}

- Model filename extension: `.pth`
- Checkpoint filename extension: `.pt`

[PyTorch](https://pytorch.org/) 2.0 or above is required.
While `.pth` and `.pt` are the same in the PyTorch package, they have different meanings in the DeePMD-kit to distinguish the model and the checkpoint.

### DP {{ dpmodel_icon }}

:::{note}
This backend is only for development and should not take into production.
:::

- Model filename extension: `.dp`, `.yaml`, `.yml`

DP is a reference backend for development, which uses pure [NumPy](https://numpy.org/) to implement models without using any heavy deep-learning frameworks.
Due to the limitation of NumPy, it doesn't support gradient calculation and thus cannot be used for training.
As a reference backend, it is not aimed at the best performance, but only the correct results.
The DP backend has two formats, both of which are backend-independent:
The `.dp` format uses [HDF5](https://docs.h5py.org/) to store model serialization data, which has good performance.
The `.yaml` or `.yml` use [YAML](https://yaml.org/) to save the data as plain texts, which is easy to read for human beings.
Only Python inference interface can load these formats.

NumPy 1.21 or above is required.

## Switch the backend

### Training

When training and freezing a model, you can use `dp --tf` or `dp --pt` in the command line to switch the backend.

### Inference

When doing inference, DeePMD-kit detects the backend from the model filename.
For example, when the model filename ends with `.pb` (the ProtoBuf file), DeePMD-kit will consider it using the TensorFlow backend.

## Convert model files between backends

If a model is supported by two backends, one can use [`dp convert-backend`](./cli.rst) to convert the model file between these two backends.

:::{warning}
Currently, only the `se_e2_a` model fully supports the backend conversion between TensorFlow {{ tensorflow_icon }} and PyTorch {{ pytorch_icon }}.
:::
