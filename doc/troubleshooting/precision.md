# Why does a model have low precision?

Many phenomena are caused by model accuracy.
For example, during simulations, temperatures explode, structures fall apart, and atoms are lost.
One can [test the model](../test/test.md) to confirm whether the model has the enough accuracy.

There are many reasons for a low-quality model.
Some common reasons are listed below.

## Data

### Data units and signs

The unit of training data should follow what is listed in [data section](../data/system.md).
Usually, the package to calculate the training data has different units from those of the DeePMD-kit.
It is noted that some software label the energy gradient as forces, instead of the negative energy gradient.
It is necessary to check them carefully to avoid inconsistent data.

### SCF coverage and data accuracy

The accuracy of models will not exceed the accuracy of training data, so the training data should reach enough accuracy.
Here is a checklist for the accuracy of data:

- SCF should converge to a suitable threshold for all points in the training data.
- The convergence of the energy, force and virial with respect to the energy cutoff and k-spacing sample is checked.
- Sometimes, QM software may generate unstable outliers, which should be removed.
- The data should be extracted with enough digits and stored with the proper precision. Large energies may have low precision when they are stored as the single-precision floating-point format (FP32).

### Enough data

If the model performs good on the training data, but has bad accuracy on another data, this means some data space is not covered by the training data.
It can be validated by evaluating the [model deviation](../test/model-deviation.md) with multiple models.
If the model deviation of these data is high for some data, try to collect more data using [DP-GEN](../third-party/out-of-deepmd-kit.md#dp-gen).

### Values of data

One should be aware that the errors of some data is also affected by the absolute values of this data.
Stable structures tend to be more precise than unstable structures because unstable structures may have larger forces.
Also, errors will be introduced in the Projector augmented wave (PAW) DFT calculations when the atoms are very close due to the overlap of pseudo-potentials.
It is expected to see that data with large forces has larger errors and it is better to compare different models only with the same data.

## Model

### Enough `sel`

The [`sel`](../model/sel.md) of the descriptors must be enough for both training and test data.
Otherwise, the model will be unreliable and give wrong results.

### Cutoff radius

The model cannot fit the long-term interaction out of the cutoff radius.
This is a designed approximation for performance, but one has to choose proper cutoff radius for the system.

### Neural network size

The size of neural networks will affect the accuracy, but if one follows the parameters in the examples, this effect is insignificant.
See [FAQ: How to tune Fitting/embedding-net size](./howtoset_netsize.md) for details.

### Neural network precision

In some cases, one may want to use the FP32 precision to make the model faster.
For some applications, FP32 is enough and thus is recommended, but one should still be aware that the precision of FP32 is not as high as that of FP64.
See [Floating-point precision of the model](../model/precision.md) section for how to set the precision.

## Training

### Training steps

Generally speaking, the longer the number of training steps, the better the model.
A balance between model accuracy and training time can be achieved.
If one finds that model accuracy decreases with training time, there may be a problem with the data. See the [data section](#data) for details.

### Learning rate

Both too large and too small learning rate may affect the training.
It is recommended to start with a large learning rate and end with a small learning rate.
The learning rate from the examples is a good choice to start.
