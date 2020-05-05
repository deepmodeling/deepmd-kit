# pair_style deepmd command

## Syntax
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

## Examples
```
pair_style deepmd graph.pb
pair_style deepmd graph.pb fparam 1.2
pair_style deepmd graph_0.pb graph_1.pb graph_2.pb out_file md.out out_freq 10 atomic relative 1.0
```

## Description

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


## Restrictions

- The `deepmd` pair style is provided in the USER-DEEPMD package, which is compiled from the DeePMD-kit, visit the [DeePMD-kit website](https://github.com/deepmodeling/deepmd-kit) for more information.

- The `atom_style` of the system should be `atomic`.

- When using the `atomic` key word of `deepmd` is set, one should not use this pair style with MPI parallelization.


[DP]:https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.143001
[DP-SE]:https://arxiv.org/abs/1805.09003
