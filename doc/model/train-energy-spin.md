# Fit spin energy

In this section, we will take `$deepmd_source_dir/examples/NiO/se_e2_a/input.json` as an example of the input file.

## Spin

The construction of the fitting net is give by section {ref}`spin <model/spin>`
```json
    "spin" : {
        "use_spin":         [true, false],
        "virtual_len":      [0.4],
        "spin_norm":        [1.2737],
    },
```
* {ref}`use_spin <model/spin[ener_spin]/use_spin>` determines whether to turn on the magnetism of the atoms.The index of this option matches option `type_map <model/type_map>`.
* {ref}`virtual_len <model/spin[ener_spin]/virtual_len>` specifies the distance between virtual atom and the belonging real atom.
* {ref}`spin_norm <model/spin[ener_spin]/spin_norm>` gives the magnitude of the magnetic moment for each magnatic atom.

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
