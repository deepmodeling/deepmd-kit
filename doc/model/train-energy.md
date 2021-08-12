# Fit energy

## Fitting network

The `fitting_net` section tells DP which fitting net to use.

The json of `dipole` type should be provided like

```json
	"fitting_net" : {
		"type": "dipole",
		"sel_type": [0],
		"neuron": [100,100,100],
		"resnet_dt": true,
		"seed": 1,
	},
```

The json of `polar` type should be provided like

```json
	"fitting_net" : {
	   	"type": "polar",
		"sel_type": [0],
		"neuron": [100,100,100],
		"resnet_dt": true,
		"seed": 1,
	},
```

-   `type` specifies which type of fitting net should be used. It should be either `dipole` or `polar`. Note that `global_polar` mode in version 1.x is already **deprecated** and is merged into `polar`. To specify whether a system is global or atomic, please see [here](train-se-e2-a.md).
-   `sel_type` is a list specifying which type of atoms have the quantity you want to fit. For example, in water system, `sel_type` is `[0]` since `0` represents for atom `O`. If left unset, all type of atoms will be fitted.
-   The rest `args` has the same meaning as they do in `ener` mode.

## Loss

The loss function for training energy is given by
```
loss = pref_e * loss_e + pref_f * loss_f + pref_v * loss_v
```
where `loss_e`, `loss_f` and `loss_v` denote the loss in energy, force and virial, respectively. `pref_e`, `pref_f` and `pref_v` give the prefactors of the energy, force and virial losses. The prefectors may not be a constant, rather it changes linearly with the learning rate. Taking the force prefactor for example, at training step `t`, it is given by
```math
pref_f(t) = start_pref_f * ( lr(t) / start_lr ) + limit_pref_f * ( 1 - lr(t) / start_lr )
```
where `lr(t)` denotes the learning rate at step `t`. `start_pref_f` and `limit_pref_f` specifies the `pref_f` at the start of the training and at the limit of `t -> inf`.

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
