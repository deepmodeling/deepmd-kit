# LAMMPS commands

## Enable DeePMD-kit plugin (plugin mode)

If you are using the plugin mode, enable DeePMD-kit package in LAMMPS with `plugin` command:

```
plugin load path/to/deepmd/lib/libdeepmd_lmp.so
```

The built-in mode doesn't need this step.

## pair_style `deepmd`

The DeePMD-kit package provides the pair_style `deepmd`

```
pair_style deepmd models ... keyword value ...
```
- deepmd = style of this pair_style
- models = frozen model(s) to compute the interaction. If multiple models are provided, then the model deviation will be computed
- keyword = *out_file* or *out_freq* or *fparam* or *atomic* or *relative*
<pre>
    <i>out_file</i> value = filename
        filename = The file name for the model deviation output. Default is model_devi.out
    <i>out_freq</i> value = freq
        freq = Frequency for the model deviation output. Default is 100.
    <i>fparam</i> value = parameters
        parameters = one or more frame parameters required for model evaluation.
    <i>atomic</i> = no value is required. 
        If this keyword is set, the model deviation of each atom will be output.
    <i>relative</i> value = level
        level = The level parameter for computing the relative model deviation
</pre>

### Examples
```
pair_style deepmd graph.pb
pair_style deepmd graph.pb fparam 1.2
pair_style deepmd graph_0.pb graph_1.pb graph_2.pb out_file md.out out_freq 10 atomic relative 1.0
```

### Description
Evaluate the interaction of the system by using [Deep Potential][DP] or [Deep Potential Smooth Edition][DP-SE]. It is noticed that deep potential is not a "pairwise" interaction, but a multi-body interaction. 

This pair style takes the deep potential defined in a model file that usually has the .pb extension. The model can be trained and frozen by package [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit).

The model deviation evalulate the consistency of the force predictions from multiple models. By default, only the maximal, minimal and averge model deviations are output. If the key `atomic` is set, then the model deviation of force prediction of each atom will be output.

By default, the model deviation is output in absolute value. If the keyword `relative` is set, then the relative model deviation will be output. The relative model deviation of the force on atom `i` is defined by
```math
           |Df_i|
Ef_i = -------------
       |f_i| + level
```
where `Df_i` is the absolute model deviation of the force on atom `i`, `|f_i|` is the norm of the the force and `level` is provided as the parameter of the keyword `relative`.

### Restrictions
- The `deepmd` pair style is provided in the USER-DEEPMD package, which is compiled from the DeePMD-kit, visit the [DeePMD-kit website](https://github.com/deepmodeling/deepmd-kit) for more information.


## Compute tensorial properties

The DeePMD-kit package provide the compute `deeptensor/atom` for computing atomic tensorial properties. 

```
compute ID group-ID deeptensor/atom model_file
```
- ID: user-assigned name of the computation
- group-ID: ID of the group of atoms to compute
- deeptensor/atom: the style of this compute
- model_file: the name of the binary model file.

### Examples
```
compute         dipole all deeptensor/atom dipole.pb
```
The result of the compute can be dump to trajctory file by 
```
dump            1 all custom 100 water.dump id type c_dipole[1] c_dipole[2] c_dipole[3] 
```

### Restrictions
- The `deeptensor/atom` compute is provided in the USER-DEEPMD package, which is compiled from the DeePMD-kit, visit the [DeePMD-kit website](https://github.com/deepmodeling/deepmd-kit) for more information.


## Long-range interaction
The reciprocal space part of the long-range interaction can be calculated by LAMMPS command `kspace_style`. To use it with DeePMD-kit, one writes 
```bash
pair_style	deepmd graph.pb
pair_coeff
kspace_style	pppm 1.0e-5
kspace_modify	gewald 0.45
```
Please notice that the DeePMD does nothing to the direct space part of the electrostatic interaction, because this part is assumed to be fitted in the DeePMD model (the direct space cut-off is thus the cut-off of the DeePMD model). The splitting parameter `gewald` is modified by the `kspace_modify` command.

[DP]:https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.143001
[DP-SE]:https://dl.acm.org/doi/10.5555/3327345.3327356