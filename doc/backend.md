# Backend

## Supported backends

DeePMD-kit supports multiple backends: TensorFlow and PyTorch.
To use DeePMD-kit, you must install at least one backend.
Each backend does not support all features.
In the documentation, TensorFlow {{ tensorflow_icon }} and PyTorch {{ pytorch_icon }} icons are used to mark whether a backend supports a feature.

### TensorFlow {{ tensorflow_icon }}

TensorFlow 2.2 or above is required.
DeePMD-kit does not use the TensorFlow v2 API but uses the TensorFlow v1 API (`tf.compat.v1`) in the graph mode.

### PyTorch {{ pytorch_icon }}

PyTorch 2.0 or above is required.

## Switch the backend

### Training

When training and freezing a model, you can use `dp --tf` or `dp --pt` in the command line to switch the backend.

### Inference

When doing inference, DeePMD-kit detects the backend from the model filename.
For example, when the model filename ends with `.pb` (the ProtoBuf file), DeePMD-kit will consider it using the TensorFlow backend.
