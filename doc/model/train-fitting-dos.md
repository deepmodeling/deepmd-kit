# Fit electronic density of states (DOS) {{ tensorflow_icon }} {{ pytorch_icon }} {{ jax_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, DP {{ dpmodel_icon }}
:::

Here we present an API to DeepDOS model, which can be used to fit electronic density of state (DOS) (which is a vector).

See the [PRB paper](https://doi.org/10.1103/PhysRevB.105.174109) for details.

In this example, we will show you how to train a model to fit a silicon system. A complete training input script of the examples can be found in

```bash
$deepmd_source_dir/examples/dos/input.json
```

The training and validation data are also provided our examples. But note that **the data provided along with the examples are of limited amount, and should not be used to train a production model.**

Similar to the `input.json` used in `ener` mode, training JSON is also divided into {ref}`model <model>`, {ref}`learning_rate <learning_rate>`, {ref}`loss <loss>` and {ref}`training <training>`. Most keywords remain the same as `ener` mode, and their meaning can be found [here](train-se-e2-a.md). To fit the `dos`, one needs to modify {ref}`model[standard]/fitting_net <model[standard]/fitting_net>` and {ref}`loss <loss>`.

## The fitting Network

The {ref}`fitting_net <model[standard]/fitting_net>` section tells DP which fitting net to use.

The JSON of `dos` type should be provided like

```json
	"fitting_net" : {
		"type": "dos",
		"numb_dos": 250,
		"sel_type": [0],
		"neuron": [120,120,120],
		"resnet_dt": true,
		"fparam": 0,
		"seed": 1,
	},
```

- `type` specifies which type of fitting net should be used. It should be `dos`.
- `numb_dos` specifies the length of output vector (density of states), which the same as the `NEDOS` set in VASP software, this argument defines the output length of the neural network. We note that the length of `dos` provided in training set should be the same.
- The rest arguments have the same meaning as they do in `ener` mode.

## Loss

DeepDOS supports trainings of the global system (a global `dos` label is provided in a frame) or atomic system (atomic labels `atom_dos` is provided for **each** atom in a frame). In a global system, each frame has just **one** `dos` label. For example, when fitting `dos`, each frame will just provide a `1 x numb_dos` vector which gives the total electronic density of states. By contrast, in an atomic system, each atom in has a `atom_dos` label. For example, when fitting the site-projected electronic density of states, each frame will provide a `natom x numb_dos` matrices,

The {ref}`loss <loss>` section tells DP the weight of these two kinds of loss, i.e.

```python
loss = pref * global_loss + pref_atomic * atomic_loss
```

The loss section should be provided like

```json
	"loss" : {
		"type": "dos",
		"start_pref_dos": 0.0,
		"limit_pref_dos": 0.0,
		"start_pref_cdf": 0.0,
		"limit_pref_cdf": 0.0,
		"start_pref_ados": 1.0,
		"limit_pref_ados": 1.0,
		"start_pref_acdf": 0.0,
		"limit_pref_acdf": 0.0
	},
```

- {ref}`type <loss/type>` should be written as `dos` as a distinction from `ener` mode.
- `pref_dos` and `pref_ados`, respectively specify the weight of global and atomic loss. If set to 0, the corresponding label will not be included in the training process.
- We also provides a combination training of vector and its cumulative distribution function `cdf`, which can be defined as

$$D(\epsilon) = \int_{e_{min}}^{\epsilon} g(\epsilon')d\epsilon'$$

## Training Data Preparation

The global label should be named `dos.npy/raw`, while the atomic label should be named `atomic_dos.npy/raw`. If wrongly named, DP will report an error.

To prepare the data, we recommend shifting the DOS data by the Fermi level.

## Train the Model

The training command is the same as `ener` mode, i.e.

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

```bash
dp --tf train input.json
```

:::

:::{tab-item} PyTorch {{ pytorch_icon }}

```bash
dp --pt train input.json
```

:::

::::

The detailed loss can be found in `lcurve.out`:

```
#  step      rmse_trn   rmse_ados_trn   rmse_ados_lr
      0      1.11e+00      1.11e+00    1.0e-03
    100      5.00e-02      5.00e-02    1.0e-03
    200      4.70e-02      4.70e-02    1.0e-03
    300      6.45e-02      6.45e-02    1.0e-03
    400      3.39e-02      3.39e-02    1.0e-03
    500      4.60e-02      4.60e-02    1.0e-03
    600      3.98e-02      3.98e-02    1.0e-03
    700      9.50e-02      9.50e-02    1.0e-03
    800      5.49e-02      5.49e-02    1.0e-03
    900      5.57e-02      5.57e-02    1.0e-03
   1000      3.73e-02      3.73e-02    1.0e-03
   1100      4.33e-02      4.33e-02    1.0e-03
   1200      3.27e-02      3.27e-02    1.0e-03
   1300      3.68e-02      3.68e-02    1.0e-03
   1400      3.09e-02      3.09e-02    1.0e-03
   1500      3.42e-02      3.42e-02    1.0e-03
   1600      5.62e-02      5.62e-02    1.0e-03
   1700      6.12e-02      6.12e-02    1.0e-03
   1800      4.10e-02      4.10e-02    1.0e-03
   1900      5.30e-02      5.30e-02    1.0e-03
   2000      3.85e-02      3.85e-02    1.0e-03
```

## Test the Model

In this earlier version, we can use `dp test` to infer the electronic density of state for given frames.

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

```bash

dp --tf freeze -o frozen_model.pb

dp --tf test -m frozen_model.pb -s ../data/111/$k -d ${output_prefix} -a -n 100
```

:::

:::{tab-item} PyTorch {{ pytorch_icon }}

```bash

dp --pt freeze -o frozen_model.pth

dp --pt test -m frozen_model.pth -s ../data/111/$k -d ${output_prefix} -a -n 100
```

:::

::::

if `dp test -d ${output_prefix} -a` is specified, the predicted DOS and atomic DOS for each frame are output in the working directory

```
${output_prefix}.ados.out.0   ${output_prefix}.ados.out.1  ${output_prefix}.ados.out.2  ${output_prefix}.ados.out.3
${output_prefix}.dos.out.0   ${output_prefix}.dos.out.1  ${output_prefix}.dos.out.2  ${output_prefix}.dos.out.3
```

for `*.dos.out.*`, it contains matrix with shape of `(2, numb_dos)`,
for `*.ados.out.*`, it contains matrix with shape of `(2, natom x numb_dos)`,

```
# frame - 0: data_dos pred_dos
0.000000000000000000e+00 1.963193264917645342e-03
0.000000000000000000e+00 1.178440836781313727e-03
0.000000000000000000e+00 1.441258071790407769e-04
0.000000000000000000e+00 1.787297933314058174e-03
0.000000000000000000e+00 1.901603280243024940e-03
0.000000000000000000e+00 2.279848925571981155e-03
0.000000000000000000e+00 2.149355854688561607e-03
0.000000000000000000e+00 1.829848459515726056e-03
0.000000000000000000e+00 1.905156512419792225e-03
```
