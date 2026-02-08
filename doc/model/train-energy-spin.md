# Fit spin energy {{ tensorflow_icon }} {{ pytorch_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}, DP {{ dpmodel_icon }}
:::

To train a model that takes additional spin information as input, you only need to modify the following sections to define the spin-specific settings,
keeping other sections the same as the normal energy model's input script.

:::{warning}
Note that when adding spin into the model, there will be some implicit modifications automatically done by the program:

- In the TensorFlow backend, the `se_e2_a` descriptor will treat those atom types with spin as new (virtual) types,
  and duplicate their corresponding selected numbers of neighbors ({ref}`sel <model[standard]/descriptor[se_e2_a]/sel>`) from their real atom types.
- In the PyTorch backend, if spin settings are added, all the types (with or without spin) will have their virtual types.
  The `se_e2_a` descriptor will thus double the {ref}`sel <model[standard]/descriptor[se_e2_a]/sel>` list,
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
- `dpa3`

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

:::{note}
It should be noted that the spin models in PyTorch/DP are capable of addressing scenarios where the spin approaches zero
(indicating the virtual atom is in close proximity to the real atom) by adjusting the non-zero
{ref}`env_protection <model[standard]/descriptor[se_e2_a]/env_protection>` parameter within the descriptor.
This parameter is set to 0.01 by default in the spin model. It appears that a value of 0.01 is generally sufficient for maintaining model stability.
For systems with nearly zero spin, users can also consider tuning this parameter to potentially enhance stability.
:::

## Spin Loss

The spin loss function $L$ for training energy is given by

$$L = p_e L_e + p_{fr} L_{fr} + p_{fm} L_{fm} + p_v L_v$$

where $L_e$, $L_{fr}$, $L_{fm}$ and $L_v$ denote the loss in energy, atomic force, magnatic force and virial, respectively. $p_e$, $p_{fr}$, $p_{fm}$ and $p_v$ give the prefactors of the energy, atomic force, magnatic force and virial losses.

:::{note}
Please note that the virial and atomic virial are not currently supported in spin models.
:::

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

## Data format

:::{note}
Note that the spin data format is different between TensorFlow and PyTorch/DP.
:::

### Spin data format in TensorFlow

In the TensorFlow backend, the spin system data format may contain the following files:

```
type.raw
set.*/box.npy
set.*/coord.npy
set.*/energy.npy
set.*/force.npy
```

This system contains `Nframes` frames with the same atom number `Natoms` and magnetic atom number `Nspins`, the total number of element and virtual types contained in all frames is `Ntypes`. The `box` and `energy` files are the same as those in [standard formats](../data/system.md). The `type` file contains the types of both real atoms and virtual atoms. In `coord` and `force` files, virtual atomic coordinates are integrated with real atomic coordinates, and magnetic forces are combined with atomic forces. Specifically, magnetic forces are obtained from [DeltaSpin](https://github.com/caizefeng/DeltaSpin) and virtual atomic coordinates are given by:

$$\bm{R}_{i^p} = \bm{R}_i + \frac{\eta_{\zeta_i}}{\mu_{\vert \bm{S}_i \vert}} \cdot \bm{S}_i$$

where $\bm{R}_{i^p}$, $\bm{R}_i$, and $\bm{S}_i$ denote the virtual atomic coordinate, atomic coordinate and spin, respectively. $\eta_{\zeta_i}$ and $\mu_{\vert \bm{S}_i \vert}$ correspond to the `virtual_len` and `spin_norm` defined in [spin settings](#spin-settings-in-tensorflow).

We list the details about spin system data format in TensorFlow backend:

| ID     | Property                   | Raw file   | Unit | Shape                             | Description                                                                                                                                               |
| ------ | -------------------------- | ---------- | ---- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| type   | Atom type indexes          | type.raw   | \    | Natoms + Nspins                   | Integers that start with 0. The first `Natoms` entries represent real atom types, followed by `Nspins` entries representing virtual atom types.           |
| coord  | Coordinates                | coord.raw  | Å    | Nframes \* (Natoms + Nspins) \* 3 | The first `3 \* Natoms` columns represent the coordinates of real atoms, followed by `3 \* Nspins` columns representing the coordinates of virtual atoms. |
| box    | Boxes                      | box.raw    | Å    | Nframes \* 3 \* 3                 | in the order `XX XY XZ YX YY YZ ZX ZY ZZ`                                                                                                                 |
| energy | Frame energies             | energy.raw | eV   | Nframes                           |
| force  | Atomic and magnetic forces | force.raw  | eV/Å | Nframes \* (Natoms + Nspins) \* 3 | The first `3 \* Natoms` columns represent atomic forces, followed by `3 \* Nspins` columns representing magnetic forces.                                  |

### Spin data format in PyTorch/DP

In the PyTorch backend, spin and magnetic forces are listed in separate files, and the data format may contain the following files:

```
type.raw
set.*/box.npy
set.*/coord.npy
set.*/spin.npy
set.*/energy.npy
set.*/force.npy
set.*/force_mag.npy
```

This system contains `Nframes` frames with the same atom number `Natoms`, the total number of element contained in all frames is `Ntypes`. Most files are the same as those in [standard formats](../data/system.md), here we only list the distinct ones:

| ID             | Property         | Raw file      | Unit    | Shape                  | Description                                                         |
| -------------- | ---------------- | ------------- | ------- | ---------------------- | ------------------------------------------------------------------- |
| spin           | Magnetic moments | spin.raw      | $\mu_B$ | Nframes \* Natoms \* 3 | Spin for magnetic atoms and zero for non-magnetic atoms.            |
| magnetic force | Magnetic forces  | force_mag.raw | eV/Å    | Nframes \* Natoms \* 3 | Magnetic forces for magnetic atoms and zero for non-magnetic atoms. |
