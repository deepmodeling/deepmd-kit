# Train a Deep Potential model to fit `tensor` like `Dipole` and `Polarizability`

Unlike `energy` which is a scalar, one may want to fit some high dimensional physical quantity, like `dipole` (vector) and `polarizability` (matrix, shorted as `polar`). Deep Potential has provided different API to allow this. In this example we will show you how to train a model to fit them for a water system. A complete training input script of the examples can be found in 

```bash
$deepmd_source_dir/examples/water_tensor/dipole/dipole_input.json
$deepmd_source_dir/examples/water_tensor/polar/polar_input.json
```

The training and validation data are also provided our examples. But note that **the data provided along with the examples are of limited amount, and should not be used to train a productive model.**



The directory of this examples:

-   [The training input script](#the-training-input-script)
-   [Training Data Preparation](#training-data-preparation)
- 	[Train the Model](#train-the-model)

## The training input script

Similar to the `input.json` used in `ener` mode, training json is also divided into `model`, `learning_rate`, `loss` and `training`. Most keywords remains the same as `ener` mode, and their meaning can be found [here](train-se-e2-a.md). To fit a tensor, one need to modify `model.fitting_net` and `loss`.

### Model

The `fitting_net` section tells DP which fitting net to use.

The json of `dipole` type should be provided like

```json
	"fitting_net" : {
		"type": "dipole",
		"sel_type": [0],
		"neuron": [100,100,100],
		"resnet_dt": true,
		"seed": 1,
	},
```

The json of `polar` type should be provided like

```json
	"fitting_net" : {
	   	"type": "polar",
		"sel_type": [0],
		"neuron": [100,100,100],
		"resnet_dt": true,
		"seed": 1,
	},
```

-   `type` specifies which type of fitting net should be used. It should be either `dipole` or `polar`. Note that `global_polar` mode in version 1.x is already **deprecated** and is merged into `polar`. To specify whether a system is global or atomic, please see [here](train-se-e2-a.md).
-   `sel_type` is a list specifying which type of atoms have the quantity you want to fit. For example, in water system, `sel_type` is `[0]` since `0` represents for atom `O`. If left unset, all type of atoms will be fitted.
-   The rest `args` has the same meaning as they do in `ener` mode.

### Loss

DP supports a combinational training of global system (only a global `tensor` label, i.e. dipole or polar, is provided in a frame) and atomic system (labels for **each** atom included in `sel_type` are provided). In a global system, each frame has just **one** `tensor` label. For example, when fitting `polar`, each frame will just provide a `1 x 9` vector which gives the elements of the polarizability tensor of that frame in order XX, XY, XZ, YX, YY, YZ, XZ, ZY, ZZ. By contrast, in a atomic system, each atom in `sel_type` has a `tensor` label. For example, when fitting dipole, each frame will provide a `#sel_atom x 3` matrix, where `#sel_atom` is the number of atoms whose type are in `sel_type`.

The `loss` section tells DP the weight of this two kind of loss, i.e.

```python
loss = pref * global_loss + pref_atomic * atomic_loss
```

The loss section should be provided like 

```json
	"loss" : {
		"type":		"tensor",
		"pref":		1.0,
		"pref_atomic":	1.0,
	},
```

-   `type` should be written as `tensor` as a distinction from `ener` mode.
-   `pref` and `pref_atomic` respectively specify the weight of global loss and atomic loss. It can not be left unset. If set to 0, system with corresponding label will NOT be included in the training process.

## Training Data Preparation

In tensor mode, the identification of label's type (global or atomic) is derived from the file name. The global label should be named as `dipole.npy/raw` or `polarizability.npy/raw`, while the atomic label should be named as `atomic_dipole.npy/raw` or `atomic_polarizability.npy/raw`. If wrongly named, DP will report an error

```bash
ValueError: cannot reshape array of size xxx into shape (xx,xx). This error may occur when your label mismatch it's name, i.e. you might store global tensor in `atomic_tensor.npy` or atomic tensor in `tensor.npy`.
```

In this case, please check the file name of label.

## Train the Model

The training command is the same as `ener` mode, i.e.

```bash
dp train input.json
```

The detailed loss can be found in `lcurve.out`:

```
#  step    rmse_val   rmse_trn  rmse_lc_val rmse_lc_trn rmse_gl_val rmse_gl_trn  lr
     0     8.34e+00   8.26e+00   8.34e+00   8.26e+00    0.00e+00    0.00e+00   1.0e-02
   100     3.51e-02   8.55e-02   0.00e+00   8.55e-02    4.38e-03    0.00e+00   5.0e-03
   200     4.77e-02   5.61e-02   0.00e+00   5.61e-02    5.96e-03    0.00e+00   2.5e-03
   300     5.68e-02   1.47e-02   0.00e+00   0.00e+00    7.10e-03    1.84e-03   1.3e-03
   400     3.73e-02   3.48e-02   1.99e-02   0.00e+00    2.18e-03    4.35e-03   6.3e-04
   500     2.77e-02   5.82e-02   1.08e-02   5.82e-02    2.11e-03    0.00e+00   3.2e-04
   600     2.81e-02   5.43e-02   2.01e-02   0.00e+00    1.01e-03    6.79e-03   1.6e-04
   700     2.97e-02   3.28e-02   2.03e-02   0.00e+00    1.17e-03    4.10e-03   7.9e-05
   800     2.25e-02   6.19e-02   9.05e-03   0.00e+00    1.68e-03    7.74e-03   4.0e-05
   900     3.18e-02   5.54e-02   9.93e-03   5.54e-02    2.74e-03    0.00e+00   2.0e-05
  1000     2.63e-02   5.02e-02   1.02e-02   5.02e-02    2.01e-03    0.00e+00   1.0e-05
  1100     3.27e-02   5.89e-02   2.13e-02   5.89e-02    1.43e-03    0.00e+00   5.0e-06
  1200     2.85e-02   2.42e-02   2.85e-02   0.00e+00    0.00e+00    3.02e-03   2.5e-06
  1300     3.47e-02   5.71e-02   1.07e-02   5.71e-02    3.00e-03    0.00e+00   1.3e-06
  1400     3.13e-02   5.76e-02   3.13e-02   5.76e-02    0.00e+00    0.00e+00   6.3e-07
  1500     3.34e-02   1.11e-02   2.09e-02   0.00e+00    1.57e-03    1.39e-03   3.2e-07
  1600     3.11e-02   5.64e-02   3.11e-02   5.64e-02    0.00e+00    0.00e+00   1.6e-07
  1700     2.97e-02   5.05e-02   2.97e-02   5.05e-02    0.00e+00    0.00e+00   7.9e-08
  1800     2.64e-02   7.70e-02   1.09e-02   0.00e+00    1.94e-03    9.62e-03   4.0e-08
  1900     3.28e-02   2.56e-02   3.28e-02   0.00e+00    0.00e+00    3.20e-03   2.0e-08
  2000     2.59e-02   5.71e-02   1.03e-02   5.71e-02    1.94e-03    0.00e+00   1.0e-08
```

One may notice that in each step, some of local loss and global loss will be `0.0`. This is because our training data and validation data consist of global system and atomic system, i.e.
```
	--training_data
		>atomic_system
		>global_system
	--validation_data
		>atomic_system
		>global_system
```
During training, at each step when the lcurve.out is printed, the system used for evaluating the training (validation) error may be either with only global or only atomic labels, thus the corresponding atomic or global errors are missing and are printed as zeros. 


