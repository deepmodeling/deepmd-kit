# Fit spin energy {{ tensorflow_icon }} {{ pytorch_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}, DP {{ dpmodel_icon }}
:::

To train a model that takes additional spin information as input, you only need to modify the following sections to define the spin-specific settings,
keeping other sections the same as the normal energy model's input script.

:::{warning}
Note that when adding spin into the model, there will be some implicit modifications automatically done by the program:

- In the TensorFlow backend, the `se_e2_a` descriptor will treat those atom types with spin as new (virtual) types,
  and duplicate their corresponding selected numbers of neighbors ({ref}`sel <model/descriptor[se_e2_a]/sel>`) from their real atom types.
- In the PyTorch backend, if spin settings are added, all the types (with or without spin) will have their virtual types.
  The `se_e2_a` descriptor will thus double the {ref}`sel <model/descriptor[se_e2_a]/sel>` list,
  while in other descriptors with mixed types (such as `dpa1` or `dpa2`), the sel number will not be changed for clarity.
  If you are using descriptors with mixed types, to achieve better performance,
  you should manually extend your sel number (maybe double) depending on the balance between performance and efficiency.
  :::

## Spin

The spin settings are given by the {ref}`spin <model/spin>` section, which sets the magnetism for each type of atoms as described in the following sections.

:::{note}
Note that the construction of spin settings is different between TensorFlow and PyTorch/DP.
:::

### Spin settings in TensorFlow

The implementation in TensorFlow only supports `se_e2_a` descriptor. See examples in `$deepmd_source_dir/examples/spin/se_e2_a/input_tf.json`, the {ref}`spin <model/spin>` section is defined as the following:

```json
    "spin" : {
        "use_spin":         [true, false],
        "virtual_len":      [0.4],
        "spin_norm":        [1.2737],
    },
```

- {ref}`use_spin <model/spin[ener_spin]/use_spin>` is a list of boolean values indicating whether to use atomic spin for each atom type.
  True for spin and False for not. The index of this option matches option `type_map <model/type_map>`.
- {ref}`virtual_len <model/spin[ener_spin]/virtual_len>` specifies the distance between virtual atom and the belonging real atom.
- {ref}`spin_norm <model/spin[ener_spin]/spin_norm>` gives the magnitude of the magnetic moment for each magnatic atom.

### Spin settings in PyTorch/DP

In PyTorch/DP, the spin implementation is more flexible and so far supports the following descriptors:

- `se_e2_a`
- `dpa1`(`se_atten`)
- `dpa2`

See `se_e2_a` examples in `$deepmd_source_dir/examples/spin/se_e2_a/input_torch.json`, the {ref}`spin <model/spin>` section is defined as the following with a much more clear interface:

```json
    "spin": {
      "use_spin": [true, false],
      "virtual_scale": [0.3140]
    },
```

- {ref}`use_spin <model/spin[ener_spin]/use_spin>` is a list of boolean values indicating whether to use atomic spin for each atom type, or a list of type indexes that use atomic spin.
  The index of this option matches option `type_map <model/type_map>`.
- {ref}`virtual_len <model/spin[ener_spin]/virtual_scale>` defines the scaling factor to determine the virtual distance
  between a virtual atom representing spin and its corresponding real atom
  for each atom type with spin. This factor is defined as the virtual distance
  divided by the magnitude of atomic spin for each atom type with spin.
  The virtual coordinate is defined as the real coordinate plus spin \* virtual_scale.
  List of float values with shape of `ntypes` or `ntypes_spin` or one single float value for all types,
  only used when {ref}`use_spin <model/spin[ener_spin]/use_spin>` is True for each atom type.

## Spin Loss

The spin loss function $L$ for training energy is given by

$$L = p_e L_e + p_{fr} L_{fr} + p_{fm} L_{fm} + p_v L_v$$

where $L_e$, $L_{fr}$, $L_{fm}$ and $L_v$ denote the loss in energy, atomic force, magnatic force and virial, respectively. $p_e$, $p_{fr}$, $p_{fm}$ and $p_v$ give the prefactors of the energy, atomic force, magnatic force and virial losses.

The prefectors may not be a constant, rather it changes linearly with the learning rate. Taking the atomic force prefactor for example, at training step $t$, it is given by

$$p_{fr}(t) = p_{fr}^0 \frac{ \alpha(t) }{ \alpha(0) } + p_{fr}^\infty ( 1 - \frac{ \alpha(t) }{ \alpha(0) })$$

where $\alpha(t)$ denotes the learning rate at step $t$. $p_{fr}^0$ and $p_{fr}^\infty$ specifies the $p_f$ at the start of the training and at the limit of $t \to \infty$ (set by {ref}`start_pref_fr <loss[ener_spin]/start_pref_fr>` and {ref}`limit_pref_f <loss[ener_spin]/limit_pref_fr>`, respectively), i.e.

```math
pref_fr(t) = start_pref_fr * ( lr(t) / start_lr ) + limit_pref_fr * ( 1 - lr(t) / start_lr )
```

The {ref}`loss <loss>` section in the `input.json` is

```json
    "loss" :{
	"type":		        "ener_spin",
	"start_pref_e":	    0.02,
	"limit_pref_e":	    1,
	"start_pref_fr":	1000,
    "limit_pref_fr":	1.0,
	"start_pref_fm":	10000,
	"limit_pref_fm":	10.0,
	"start_pref_v":	    0,
	"limit_pref_v":	    0,
    },
```

The options {ref}`start_pref_e <loss[ener_spin]/start_pref_e>`, {ref}`limit_pref_e <loss[ener_spin]/limit_pref_e>`, {ref}`start_pref_fr <loss[ener_spin]/start_pref_fr>`, {ref}`limit_pref_fm <loss[ener_spin]/limit_pref_fm>`, {ref}`start_pref_v <loss[ener_spin]/start_pref_v>` and {ref}`limit_pref_v <loss[ener_spin]/limit_pref_v>` determine the start and limit prefactors of energy, atomic force, magnatic force and virial, respectively.

If one does not want to train with virial, then he/she may set the virial prefactors {ref}`start_pref_v <loss[ener_spin]/start_pref_v>` and {ref}`limit_pref_v <loss[ener_spin]/limit_pref_v>` to 0.

## Data preparation

(Need a documentation for data format for TensorFlow and PyTorch/DP.)
