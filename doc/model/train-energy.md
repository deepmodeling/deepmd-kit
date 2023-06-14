# Fit energy

In this section, we will take `$deepmd_source_dir/examples/water/se_e2_a/input.json` as an example of the input file.

## The fitting network

The construction of the fitting net is given by section {ref}`fitting_net <model/fitting_net>`
```json
	"fitting_net" : {
	    "neuron":		[240, 240, 240],
	    "resnet_dt":	true,
	    "seed":		1
	},
```
* {ref}`neuron <model/fitting_net[ener]/neuron>` specifies the size of the fitting net. If two neighboring layers are of the same size, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is built between them.
* If the option {ref}`resnet_dt <model/fitting_net[ener]/resnet_dt>` is set to `true`, then a timestep is used in the ResNet.
* {ref}`seed <model/fitting_net[ener]/seed>` gives the random seed that is used to generate random numbers when initializing the model parameters.

## Loss

The loss function $L$ for training energy is given by

$$L = p_e L_e + p_f L_f + p_v L_v$$

where $L_e$, $L_f$, and $L_v$ denote the loss in energy, forces and virials, respectively. $p_e$, $p_f$, and $p_v$ give the prefactors of the energy, force and virial losses. The prefectors may not be a constant, rather it changes linearly with the learning rate. Taking the force prefactor for example, at training step $t$, it is given by

$$p_f(t) = p_f^0 \frac{ \alpha(t) }{ \alpha(0) } + p_f^\infty ( 1 - \frac{ \alpha(t) }{ \alpha(0) })$$

where $\alpha(t)$ denotes the learning rate at step $t$. $p_f^0$ and $p_f^\infty$ specifies the $p_f$ at the start of the training and the limit of $t \to \infty$ (set by {ref}`start_pref_f <loss[ener]/start_pref_f>` and {ref}`limit_pref_f <loss[ener]/limit_pref_f>`, respectively), i.e.
```math
pref_f(t) = start_pref_f * ( lr(t) / start_lr ) + limit_pref_f * ( 1 - lr(t) / start_lr )
```

The {ref}`loss <loss>` section in the `input.json` is
```json
    "loss" : {
	"start_pref_e":	0.02,
	"limit_pref_e":	1,
	"start_pref_f":	1000,
	"limit_pref_f":	1,
	"start_pref_v":	0,
	"limit_pref_v":	0
    }
```
The options {ref}`start_pref_e <loss[ener]/start_pref_e>`, {ref}`limit_pref_e <loss[ener]/limit_pref_e>`, {ref}`start_pref_f <loss[ener]/start_pref_f>`, {ref}`limit_pref_f <loss[ener]/limit_pref_f>`, {ref}`start_pref_v <loss[ener]/start_pref_v>` and {ref}`limit_pref_v <loss[ener]/limit_pref_v>` determine the start and limit prefactors of energy, force and virial, respectively.

If one does not want to train with virial, then he/she may set the virial prefactors {ref}`start_pref_v <loss[ener]/start_pref_v>` and {ref}`limit_pref_v <loss[ener]/limit_pref_v>` to 0.
