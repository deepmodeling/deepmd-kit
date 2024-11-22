# Fit `tensor` like `Dipole` and `Polarizability` {{ tensorflow_icon }} {{ pytorch_icon }} {{ jax_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, DP {{ dpmodel_icon }}
:::

Unlike `energy`, which is a scalar, one may want to fit some high dimensional physical quantity, like `dipole` (vector) and `polarizability` (matrix, shorted as `polar`). Deep Potential has provided different APIs to do this. In this example, we will show you how to train a model to fit a water system. A complete training input script of the examples can be found in

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

```bash
$deepmd_source_dir/examples/water_tensor/dipole/dipole_input.json
$deepmd_source_dir/examples/water_tensor/polar/polar_input.json
```

:::

:::{tab-item} PyTorch {{ pytorch_icon }}

```bash
$deepmd_source_dir/examples/water_tensor/dipole/dipole_input_torch.json
$deepmd_source_dir/examples/water_tensor/polar/polar_input_torch.json
```

:::

::::

The training and validation data are also provided our examples. But note that **the data provided along with the examples are of limited amount, and should not be used to train a production model.**

Similar to the `input.json` used in `ener` mode, training JSON is also divided into {ref}`model <model>`, {ref}`learning_rate <learning_rate>`, {ref}`loss <loss>` and {ref}`training <training>`. Most keywords remain the same as `ener` mode, and their meaning can be found [here](train-se-e2-a.md).
To fit a tensor, one needs to modify {ref}`fitting_net <model[standard]/fitting_net>` and {ref}`loss <loss>`.

## Theory

To represent the first-order tensorial properties (i.e. vector properties), we let the fitting network, denoted by $\mathcal F_{1}$, output an $M$-dimensional vector; then we have the representation,

```math
(T_i^{(1)})_\alpha =
\frac{1}{N_c}
\sum_{j=1}^{N_c}\sum_{m=1}^M (\mathcal G^i)_{jm} (\mathcal R^i)_{j,\alpha+1}
(\mathcal F_{1}(\mathcal D^i))_m, \ \alpha=1,2,3.
```

We let the fitting network $\mathcal F_{2}$ output an $M$-dimensional vector, and the second-order tensorial properties (matrix properties) are formulated as

```math
(T_i^{(2)})_{\alpha\beta} =
\frac{1}{N_c^2}
\sum_{j=1}^{N_c}\sum_{k=1}^{N_c}\sum_{m=1}^M
(\mathcal G^i)_{jm}
(\mathcal R^i)_{j,\alpha+1}
(\mathcal R^i)_{k,\beta+1}
(\mathcal G^i)_{km}
(\mathcal F_{2}(\mathcal D^i))_m,
\ \alpha,\beta=1,2,3,
```

where $\mathcal{G}^i$ and $\mathcal{R}^i$ can be found in [`se_e2_a`](./train-se-e2-a.md).
Thus, the tensor fitting network requires the descriptor to have the same or similar form as the DeepPot-SE descriptor.
$\mathcal{F}_1$ and $\mathcal F_2$ are the neural network functions.
The total tensor $\boldsymbol{T}$ (total dipole $\boldsymbol{T}^{(1)}$ or total polarizability $\boldsymbol{T}^{(2)}$) is the sum of the atomic tensor:

```math
    \boldsymbol{T} = \sum_i \boldsymbol{T}_i.
```

The tensorial models can be used to calculate IR spectrum and Raman spectrum.[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

## The fitting Network

The {ref}`fitting_net <model[standard]/fitting_net>` section tells DP which fitting net to use.

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

The JSON of `dipole` type should be provided like

```json
	"fitting_net" : {
		"type": "dipole",
		"sel_type": [0],
		"neuron": [100,100,100],
		"resnet_dt": true,
		"seed": 1,
	},
```

The JSON of `polar` type should be provided like

```json
	"fitting_net" : {
	   	"type": "polar",
		"sel_type": [0],
		"neuron": [100,100,100],
		"resnet_dt": true,
		"seed": 1,
	},
```

- `type` specifies which type of fitting net should be used. It should be either `dipole` or `polar`. Note that `global_polar` mode in version 1.x is already **deprecated** and is merged into `polar`. To specify whether a system is global or atomic, please see [here](train-se-e2-a.md).
- `sel_type` is a list specifying which type of atoms have the quantity you want to fit. For example, in the water system, `sel_type` is `[0]` since `0` represents atom `O`. If left unset, all types of atoms will be fitted.
- The rest arguments have the same meaning as they do in `ener` mode.

:::

:::{tab-item} PyTorch {{ pytorch_icon }}

The JSON of `dipole` type should be provided like

```json
	"atom_exclude_types": [
      1
    ],
	"fitting_net" : {
		"type": "dipole",
		"neuron": [100,100,100],
		"resnet_dt": true,
		"seed": 1,
	},
```

The JSON of `polar` type should be provided like

```json
	"atom_exclude_types": [
      1
    ],
	"fitting_net" : {
	   	"type": "polar",
		"neuron": [100,100,100],
		"resnet_dt": true,
		"seed": 1,
	},
```

- `type` specifies which type of fitting net should be used. It should be either `dipole` or `polar`. Note that `global_polar` mode in version 1.x is already **deprecated** and is merged into `polar`. To specify whether a system is global or atomic, please see [here](train-se-e2-a.md).
- `atom_exclude_types` is a list specifying the which type of atoms have the quantity you want to set to zero. For example, in the water system, `atom_exclude_types` is `[1]` since `1` represents atom `H`.
- The rest arguments have the same meaning as they do in `ener` mode.
  :::

::::

## Loss

DP supports a combinational training of the global system (only a global `tensor` label, i.e. dipole or polar, is provided in a frame) and atomic system (labels for **each** atom included in `sel_type`/ not included in `atom_exclude_types` are provided). In a global system, each frame has just **one** `tensor` label. For example, when fitting `polar`, each frame will just provide a `1 x 9` vector which gives the elements of the polarizability tensor of that frame in order XX, XY, XZ, YX, YY, YZ, XZ, ZY, ZZ. By contrast, in an atomic system, each atom in `sel_type` has a `tensor` label. For example, when fitting a dipole, each frame will provide a `#sel_atom x 3` matrices, where `#sel_atom` is the number of atoms whose type are in `sel_type`.

The {ref}`loss <loss>` section tells DP the weight of these two kinds of loss, i.e.

```python
loss = pref * global_loss + pref_atomic * atomic_loss
```

The loss section should be provided like

```json
	"loss" : {
		"type":		"tensor",
		"pref":		1.0,
		"pref_atomic":	1.0
	},
```

- {ref}`type <loss/type>` should be written as `tensor` as a distinction from `ener` mode.
- {ref}`pref <loss[tensor]/pref>` and {ref}`pref_atomic <loss[tensor]/pref_atomic>` respectively specify the weight of global loss and atomic loss. It can not be left unset. If set to 0, the corresponding label will NOT be included in the training process.

## Training Data Preparation

In tensor mode, the identification of the label's type (global or atomic) is derived from the file name. The global label should be named `dipole.npy/raw` or `polarizability.npy/raw`, while the atomic label should be named `atomic_dipole.npy/raw` or `atomic_polarizability.npy/raw`. If wrongly named, DP will report an error

```bash
ValueError: cannot reshape array of size xxx into shape (xx,xx). This error may occur when your label mismatch it's name, i.e. you might store global tensor in `atomic_tensor.npy` or atomic tensor in `tensor.npy`.
```

In this case, please check the file name of the label.

## Train the Model

The training command is the same as `ener` mode, i.e.

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

```bash
dp train input.json
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

One may notice that in each step, some of the local loss and global loss will be `0.0`. This is because our training data and validation data consist of the global system and atomic system, i.e.

```
	--training_data
		>atomic_system
		>global_system
	--validation_data
		>atomic_system
		>global_system
```

During training, at each step when the `lcurve.out` is printed, the system used for evaluating the training (validation) error may be either with only global or only atomic labels, thus the corresponding atomic or global errors are missing and are printed as zeros.

## Difference among different backends

To only fit against a subset of atomic types, in the TensorFlow backend, {ref}`fitting_net/sel_type <model[standard]/fitting_net[dipole]/sel_type>` should be set to selected types;
in other backends, {ref}`atom_exclude_types <model/atom_exclude_types>` should be set to excluded types.
The TensorFlow backend does not support {ref}`numb_fparam <model[standard]/fitting_net[dipole]/numb_fparam>` and {ref}`numb_aparam <model[standard]/fitting_net[dipole]/numb_aparam>`.
