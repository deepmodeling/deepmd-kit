# LAMMPS commands

## Enable DeePMD-kit plugin (plugin mode)

If you are using the plugin mode, enable DeePMD-kit package in LAMMPS with `plugin` command:

```lammps
plugin load libdeepmd_lmp.so
```

After LAMMPS version `patch_24Mar2022`, another way to load plugins is to set the environmental variable `LAMMPS_PLUGIN_PATH`:

```sh
LAMMPS_PLUGIN_PATH=$deepmd_root/lib/deepmd_lmp
```

where `$deepmd_root` is the directory to [install C++ interface](../install/install-from-source.md).

The built-in mode doesn't need this step.

## pair_style `deepmd`

The DeePMD-kit package provides the pair_style `deepmd`

```lammps
pair_style deepmd models ... keyword value ...
```
- deepmd = style of this pair_style
- models = frozen model(s) to compute the interaction.
If multiple models are provided, then only the first model serves to provide energy and force prediction for each timestep of molecular dynamics,
and the model deviation will be computed among all models every `out_freq` timesteps.
- keyword = *out_file* or *out_freq* or *fparam* or *fparam_from_compute* or *atomic* or *relative* or *relative_v* or *aparam* or *ttm*
<pre>
    <i>out_file</i> value = filename
        filename = The file name for the model deviation output. Default is model_devi.out
    <i>out_freq</i> value = freq
        freq = Frequency for the model deviation output. Default is 100.
    <i>fparam</i> value = parameters
        parameters = one or more frame parameters required for model evaluation.
    <i>fparam_from_compute</i> value = id
        id = compute id used to update the frame parameter.
    <i>atomic</i> = no value is required.
        If this keyword is set, the model deviation of each atom will be output.
    <i>relative</i> value = level
        level = The level parameter for computing the relative model deviation of the force
    <i>relative_v</i> value = level
        level = The level parameter for computing the relative model deviation of the virial
    <i>aparam</i> value = parameters
        parameters = one or more atomic parameters of each atom required for model evaluation
    <i>ttm</i> value = id
        id = fix ID of fix ttm
</pre>

### Examples
```lammps
pair_style deepmd graph.pb
pair_style deepmd graph.pb fparam 1.2
pair_style deepmd graph_0.pb graph_1.pb graph_2.pb out_file md.out out_freq 10 atomic relative 1.0
pair_coeff * * O H

pair_style deepmd cp.pb fparam_from_compute TEMP
compute    TEMP all temp
```

### Description
Evaluate the interaction of the system by using [Deep Potential][DP] or [Deep Potential Smooth Edition][DP-SE]. It is noticed that deep potential is not a "pairwise" interaction, but a multi-body interaction.

This pair style takes the deep potential defined in a model file that usually has the .pb extension. The model can be trained and frozen by package [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit), which can have either double or single float precision interface.

The model deviation evalulates the consistency of the force predictions from multiple models. By default, only the maximal, minimal and average model deviations are output. If the key `atomic` is set, then the model deviation of force prediction of each atom will be output.

By default, the model deviation is output in absolute value. If the keyword `relative` is set, then the relative model deviation of the force will be output, including values output by the keyword `atomic`. The relative model deviation of the force on atom $i$ is defined by

$$E_{f_i}=\frac{\left|D_{f_i}\right|}{\left|f_i\right|+l}$$

where $D_{f_i}$ is the absolute model deviation of the force on atom $i$, $f_i$ is the norm of the force and $l$ is provided as the parameter of the keyword `relative`.
If the keyword `relative_v` is set, then the relative model deviation of the virial will be output instead of the absolute value, with the same definition of that of the force:

$$E_{v_i}=\frac{\left|D_{v_i}\right|}{\left|v_i\right|+l}$$

If the keyword `fparam` is set, the given frame parameter(s) will be fed to the model.
If the keyword `fparam_from_compute` is set, the global parameter(s) from compute command (e.g., temperature from [compute temp command](https://docs.lammps.org/compute_temp.html)) will be fed to the model as the frame parameter(s).
If the keyword `aparam` is set, the given atomic parameter(s) will be fed to the model, where each atom is assumed to have the same atomic parameter(s).
If the keyword `ttm` is set, electronic temperatures from [fix ttm command](https://docs.lammps.org/fix_ttm.html) will be fed to the model as the atomic parameters.

Only a single `pair_coeff` command is used with the deepmd style which specifies atom names. These are mapped to LAMMPS atom types (integers from 1 to Ntypes) by specifying Ntypes additional arguments after `* *` in the `pair_coeff` command.
If atom names are not set in the `pair_coeff` command, the training parameter {ref}`type_map <model/type_map>` will be used by default.
If the training parameter {ref}`type_map <model/type_map>` is not set, atom names in the `pair_coeff` command cannot be set. In this case, atom type indexes in [`type.raw`](../data/system.md) (integers from 0 to Ntypes-1) will map to LAMMPS atom types.

Spin is specified by keywords `virtual_len` and `spin_norm`. If the keyword `virtual_len` is set, the distance between virtual atom and its corresponding real atom for each type of magnetic atoms will be fed to the model as the spin parameters. If the keyword `spin_norm` is set, the magnitude of the magnetic moment for each type of magnetic atoms will be fed to the model as the spin parameters.

### Restrictions
- The `deepmd` pair style is provided in the USER-DEEPMD package, which is compiled from the DeePMD-kit, visit the [DeePMD-kit website](https://github.com/deepmodeling/deepmd-kit) for more information.


## Compute tensorial properties

The DeePMD-kit package provides the compute `deeptensor/atom` for computing atomic tensorial properties.

```lammps
compute ID group-ID deeptensor/atom model_file
```
- ID: user-assigned name of the computation
- group-ID: ID of the group of atoms to compute
- deeptensor/atom: the style of this compute
- model_file: the name of the binary model file.

At this time, the training parameter {ref}`type_map <model/type_map>` will be mapped to LAMMPS atom types.

### Examples
```lammps
compute         dipole all deeptensor/atom dipole.pb
```
The result of the compute can be dumped to trajectory file by
```lammps
dump            1 all custom 100 water.dump id type c_dipole[1] c_dipole[2] c_dipole[3]
```

### Restrictions
- The `deeptensor/atom` compute is provided in the USER-DEEPMD package, which is compiled from the DeePMD-kit, visit the [DeePMD-kit website](https://github.com/deepmodeling/deepmd-kit) for more information.


## Long-range interaction
The reciprocal space part of the long-range interaction can be calculated by LAMMPS command `kspace_style`. To use it with DeePMD-kit, one writes
```lammps
pair_style	deepmd graph.pb
pair_coeff  * *
kspace_style	pppm 1.0e-5
kspace_modify	gewald 0.45
```
Please notice that the DeePMD does nothing to the direct space part of the electrostatic interaction, because this part is assumed to be fitted in the DeePMD model (the direct space cut-off is thus the cut-off of the DeePMD model). The splitting parameter `gewald` is modified by the `kspace_modify` command.

## Use of the centroid/stress/atom to get the full 3x3 "atomic-virial"

The [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) allows also the computation of per-atom stress tensor defined as:

$$dvatom=\sum_{m}( \mathbf{r}_n- \mathbf{r}_m) \frac{de_m}{d\mathbf{r}_n}$$

Where $\mathbf{r}_n$ is the atomic position of nth atom, $\mathbf{v}_n$ velocity of the atom and $\frac{de_m}{d\mathbf{r}_n}$ the derivative of the atomic energy.

In LAMMPS one can get the per-atom stress using the command `centroid/stress/atom`:
```lammps
compute ID group-ID centroid/stress/atom NULL virial
```
see [LAMMPS doc page](https://docs.lammps.org/compute_stress_atom.html#thompson2) for more details on the meaning of the keywords.
### Examples
In order of computing the 9-component per-atom stress
```lammps
compute stress all centroid/stress/atom NULL virial
```
Thus `c_stress` is an array with 9 components in the order `xx,yy,zz,xy,xz,yz,yx,zx,zy`.

If you use this feature please cite [D. Tisi, L. Zhang, R. Bertossa, H. Wang, R. Car, S. Baroni - arXiv preprint arXiv:2108.10850, 2021](https://arxiv.org/abs/2108.10850)

## Computation of heat flux
Using a per-atom stress tensor one can, for example, compute the heat flux defined as:

$$\mathbf J = \sum_n e_n \mathbf v_n + \sum_{n,m} ( \mathbf r_m- \mathbf r_n) \frac{de_m}{d\mathbf r_n} \mathbf v_n$$

to compute the heat flux with LAMMPS:
```lammps
compute ke_ID all ke/atom
compute pe_ID all pe/atom
compute stress_ID group-ID centroid/stress/atom NULL virial
compute flux_ID all heat/flux ke_ID pe_ID stress_ID
```

### Examples

```lammps
compute ke all ke/atom
compute pe all pe/atom
compute stress all centroid/stress/atom NULL virial
compute flux all heat/flux ke pe stress
```
`c_flux` is a global vector of length 6. The first three components are the $x$, $y$ and $z$ components of the full heat flux vector. The others are the components of the so-called convective portion, see [LAMMPS doc page](https://docs.lammps.org/compute_heat_flux.html) for more detailes.

If you use these features please cite [D. Tisi, L. Zhang, R. Bertossa, H. Wang, R. Car, S. Baroni - arXiv preprint arXiv:2108.10850, 2021](https://arxiv.org/abs/2108.10850)


[DP]:https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.143001
[DP-SE]:https://dl.acm.org/doi/10.5555/3327345.3327356
