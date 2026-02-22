# Fit other properties {{ pytorch_icon }} {{ jax_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, DP {{ dpmodel_icon }}
:::

Here we present an API to DeepProperty model, which can be used to fit other properties like band gap, bulk modulus, critical temperature, etc.

In this example, we will show you how to train a model to fit properties of `humo`, `lumo` and `band gap`. A complete training input script of the examples can be found in

```bash
$deepmd_source_dir/examples/property/train
```

The training and validation data are also provided our examples. But note that **the data provided along with the examples are of limited amount, and should not be used to train a production model.**

Similar to the `input.json` used in `ener` mode, training JSON is also divided into {ref}`model <model>`, {ref}`learning_rate <learning_rate>`, {ref}`loss <loss>` and {ref}`training <training>`. Most keywords remain the same as `ener` mode, and their meaning can be found [here](train-se-atten.md). To fit the `property`, one needs to modify {ref}`model[standard]/fitting_net <model[standard]/fitting_net>` and {ref}`loss <loss>`.

## The fitting Network

The {ref}`fitting_net <model[standard]/fitting_net>` section tells DP which fitting net to use.

The JSON of `property` type should be provided like

```json
"fitting_net" : {
	"type": "property",
        "intensive": true,
        "property_name": "band_prop",
        "task_dim": 3,
	"neuron": [240,240,240],
	"resnet_dt": true,
	"seed": 1,
},
```

- `type` specifies which type of fitting net should be used. It should be `property`.
- `intensive` indicates whether the fitting property is intensive. If `intensive` is `true`, the model output is the average of the property contribution of each atom. If `intensive` is `false`, the model output is the sum of the property contribution of each atom.
- `property_name` is the name of the property to be predicted. It should be consistent with the property name in the dataset. In each system, code will read `set.*/{property_name}.npy` file as prediction label if you use NumPy format data.
- `fitting_net/task_dim` is the dimension of model output. It should be consistent with the property dimension in the dataset, which means if the shape of data stored in `set.*/{property_name}.npy` is `batch size * 3`, `fitting_net/task_dim` should be set to 3.
- The rest arguments have the same meaning as they do in `ener` mode.

## Loss

DeepProperty supports trainings of the global system (one or more global labels are provided in a frame). For example, when fitting `property`, each frame will provide a `1 x task_dim` vector which gives the fitting properties.

The loss section should be provided like

```json
"loss" : {
	"type": "property",
	"metric": ["mae"],
	"loss_func": "smooth_mae"
},
```

- {ref}`type <loss/type>` should be written as `property` as a distinction from `ener` mode.
- `metric`: The metric for display, which will be printed in `lcurve.out`. This list can include 'smooth_mae', 'mae', 'mse' and 'rmse'.
- `loss_func`: The loss function to minimize, you can use 'mae','smooth_mae', 'mse' and 'rmse'.

## Training Data Preparation

The label should be named `{property_name}.npy/raw`, `property_name` is defined by `fitting_net/property_name` in `input.json`.

To prepare the data, you can use `dpdata` tools, for example:

```py
import dpdata
import numpy as np
from dpdata.data_type import (
    Axis,
    DataType,
)

property_name = "band_prop"  # fittng_net/property_name
task_dim = 3  # fitting_net/task_dim

# register datatype
datatypes = [
    DataType(
        property_name,
        np.ndarray,
        shape=(Axis.NFRAMES, task_dim),
        required=False,
    ),
]
datatypes.extend(
    [
        DataType(
            "energies",
            np.ndarray,
            shape=(Axis.NFRAMES, 1),
            required=False,
        ),
        DataType(
            "forces",
            np.ndarray,
            shape=(Axis.NFRAMES, Axis.NATOMS, 1),
            required=False,
        ),
    ]
)

for datatype in datatypes:
    dpdata.System.register_data_type(datatype)
    dpdata.LabeledSystem.register_data_type(datatype)

ls = dpdata.MultiSystems()
frame = dpdata.System("POSCAR", fmt="vasp/poscar")
labelframe = dpdata.LabeledSystem()
labelframe.append(frame)
labelframe.data[property_name] = np.array([[-0.236, 0.056, 0.292]], dtype=np.float32)
ls.append(labelframe)
ls.to_deepmd_npy_mixed("deepmd")
```

## Train the Model

The training command is the same as `ener` mode, i.e.

::::{tab-set}

:::{tab-item} PyTorch {{ pytorch_icon }}

```bash
dp --pt train input.json
```

:::

::::

The detailed loss can be found in `lcurve.out`:

```
# step        mae_val     mae_trn   lr
# If there is no available reference data, rmse_*_{val,trn} will print nan
      1      2.72e-02    2.40e-02    2.0e-04
    100      1.79e-02    1.34e-02    2.0e-04
    200      1.45e-02    1.86e-02    2.0e-04
    300      1.61e-02    4.90e-03    2.0e-04
    400      2.04e-02    1.05e-02    2.0e-04
    500      9.09e-03    1.85e-02    2.0e-04
    600      1.01e-02    5.63e-03    2.0e-04
    700      1.10e-02    1.76e-02    2.0e-04
    800      1.14e-02    1.50e-02    2.0e-04
    900      9.54e-03    2.70e-02    2.0e-04
   1000      1.00e-02    2.73e-02    2.0e-04
```

## Test the Model

We can use `dp test` to infer the properties for given frames.

::::{tab-set}

:::{tab-item} PyTorch {{ pytorch_icon }}

```bash

dp --pt freeze -o frozen_model.pth

dp --pt test -m frozen_model.pth -s ../data/data_0/ -d ${output_prefix} -n 100
```

:::

::::

if `dp test -d ${output_prefix}` is specified, the predicted properties for each frame are output in the working directory

```
${output_prefix}.property.out.0   ${output_prefix}.property.out.1  ${output_prefix}.property.out.2  ${output_prefix}.property.out.3
```

for `*.property.out.*`, it contains matrix with shape of `(2, task_dim)`,

```
# ../data/data_0 - 0: data_property pred_property
-2.449000030755996704e-01 -2.315840660495154801e-01
6.400000303983688354e-02 5.810663314446311983e-02
3.088999986648559570e-01 2.917143316092784544e-01
```

## Data Normalization

When `fitting_net/type` is `ener`, the energy bias layer “$e_{bias}$” adds a constant bias to the atomic energy contribution according to the atomic number.i.e.,
$$e_{bias} (Z_i) (MLP(D_i))= MLP(D_i) + e_{bias} (Z_i)$$

But when `fitting_net/type` is `property`. The property bias layer is used to normalize the property output of the model.i.e.,
$$p_{bias} (MLP(D_i))= MLP(D_i) * std+ mean$$

1. `std`: The standard deviation of the property label
2. `mean`: The average value of the property label
