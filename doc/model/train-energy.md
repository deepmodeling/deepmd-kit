# Fit energy

In this section, we will take `$deepmd_source_dir/examples/water/se_e2_a/input.json` as an example of the input file.

## The fitting network

The construction of the fitting net is give by section `fitting_net`
```json
	"fitting_net" : {
	    "neuron":		[240, 240, 240],
	    "resnet_dt":	true,
	    "seed":		1
	},
```
* `neuron` specifies the size of the fitting net. If two neighboring layers are of the same size, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is built between them. 
* If the option `resnet_dt` is set `true`, then a timestep is used in the ResNet. 
* `seed` gives the random seed that is used to generate random numbers when initializing the model parameters.

## Loss

The loss function $L$ for training energy is given by

$$L = p_e L_e + p_f L_f + p_v L_v$$

where $L_e$, $L_f$, and $L_v$ denote the loss in energy, force and virial, respectively. $p_e$, $p_f$, and $p_v$ give the prefactors of the energy, force and virial losses. The prefectors may not be a constant, rather it changes linearly with the learning rate. Taking the force prefactor for example, at training step $t$, it is given by

$$p_f(t) = p_f^0 \frac{ \alpha(t) }{ \alpha(0) } + p_f^\infty ( 1 - \frac{ \alpha(t) }{ \alpha(0) })$$

where $\alpha(t)$ denotes the learning rate at step $t$. $p_f^0$ and $p_f^\infty$ specifies the $p_f$ at the start of the training and at the limit of $t \to \infty$ (set by `start_pref_f` and `limit_pref_f`, respectively), i.e.
```math
pref_f(t) = start_pref_f * ( lr(t) / start_lr ) + limit_pref_f * ( 1 - lr(t) / start_lr )
```

The `loss` section in the `input.json` is 
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
The options `start_pref_e`, `limit_pref_e`, `start_pref_f`, `limit_pref_f`, `start_pref_v` and `limit_pref_v` determine the start and limit prefactors of energy, force and virial, respectively.

If one does not want to train with virial, then he/she may set the virial prefactors `start_pref_v` and `limit_pref_v` to 0.
