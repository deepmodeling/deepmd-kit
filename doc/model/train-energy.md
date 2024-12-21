# Fit energy {{ tensorflow_icon }} {{ pytorch_icon }} {{ jax_icon }} {{ paddle_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, Paddle {{ paddle_icon }}, DP {{ dpmodel_icon }}
:::

In this section, we will take `$deepmd_source_dir/examples/water/se_e2_a/input.json` as an example of the input file.

## Theory

In the DP model, we let the fitting network $\mathcal{F}_ 0$ maps the descriptor $\mathcal{D}^i$ to a scalar, where the subscript $0$ means that the output is a zero-order tensor (i.e. scalar). The model can then be used to predict the total potential energy of the system by

```math
    E  =  \sum_i E_i = \sum_i \mathcal F_0 (\mathcal D^i),
```

where the output of the fitting network is treated as the atomic potential energy contribution, i.e. $E_i$.
The output scalar can also be treated as other scalar properties defined on an atom, for example, the partial charge of atom $i$.

In some cases, atomic-specific or frame-specific parameters, such as electron temperature, may be treated as extra input to the fitting network.
We denote the atomic and frame-specific parameters by $\boldsymbol{P}^i\in \mathbb{R}^{N_p}$ (with $N_p$ being the dimension) and $\boldsymbol{Q}\in \mathbb{R}^{N_q}$ (with $N_q$ being the dimension), respectively.

```math
    E_i=\mathcal{F}_0(\{\mathcal{D}^i, \boldsymbol{P}^i, \boldsymbol Q\}).
```

The atomic force $\boldsymbol{F}_ {i}$ and the virial tensor $\boldsymbol{\Xi} = (\Xi_{\alpha\beta})$ (if PBC is applied) can be derived from the potential energy $E$:

```math
    F_{i,\alpha}=-\frac{\partial E}{\partial r_{i,\alpha}},
```

```math
    \Xi_{\alpha\beta}=-\sum_{\gamma} \frac{\partial E}{\partial h_{\gamma\alpha}} h_{\gamma\beta},
```

where $r_{i,\alpha}$ and $F_{i,\alpha}$ denotes the $\alpha$-th component of the coordinate and force of atom $i$. $h_{\alpha\beta}$ is the $\beta$-th component of the $\alpha$-th basis vector of the simulation region.

The properties $\eta$ of the energy loss function could be energy $E$, force $\boldsymbol{F}$, virial $\boldsymbol{\Xi}$, relative energy $\Delta E$, or any combination among them, and the loss functions of them are

```math
    L_E(\boldsymbol{x};\boldsymbol{\theta})=\frac{1}{N}(E(\boldsymbol{x};\boldsymbol{\theta})-E^*)^2,
```

```math
    L_F(\boldsymbol{x};\boldsymbol{\theta})=\frac{1}{3N}\sum_{k=1}^{N}\sum_{\alpha=1}^3(F_{k,\alpha}(\boldsymbol{x};\boldsymbol{\theta})-F_{k,\alpha}^*)^2,
```

```math
    L_\Xi(\boldsymbol{x};\boldsymbol{\theta})=\frac{1}{9N}\sum_{\alpha,\beta=1}^{3}(\Xi_{\alpha\beta}(\boldsymbol{x};\boldsymbol{\theta})-\Xi_{\alpha\beta}^*)^2,
```

```math
    L_{\Delta E}(\boldsymbol{x};\boldsymbol{\theta})=\frac{1}{N}({\Delta E}(\boldsymbol{x};\boldsymbol{\theta})-{\Delta E}^*)^2,
```

where $F_{k,\alpha}$ is the $\alpha$-th component of the force on atom $k$, and the superscript $\ast$ indicates the label of the property that should be provided in advance.
Using $N$ ensures that each loss of fitting property is averaged over atomic contributions before they contribute to the total loss by weight.

If part of atoms is more important than others, for example, certain atoms play an essential role when calculating free energy profiles or kinetic isotope effects, the MSE of atomic forces with prefactors $q_{k}$ can also be used as the loss function:

```math
    L_F^p(\mathbf{x};\boldsymbol{\theta})=\frac{1}{3N}\sum_{k=1}^{N} \sum_{\alpha} q_{k} (F_{k,\alpha}(\mathbf{x};\boldsymbol{\theta})-F_{k,\alpha}^*)^2.
```

The atomic forces with larger prefactors will be fitted more accurately than those in other atoms.

If some forces are quite large, for example, forces can be greater than 60 eV/Å in high-temperature reactive simulations, one may also prefer the force loss is relative to the magnitude:

```math
    L^r_F(\boldsymbol{x};\boldsymbol{\theta})=\frac{1}{3N}\sum_{k=1}^{N}\sum_\alpha \left(\frac{F_{k,\alpha}(\boldsymbol{x};\boldsymbol{\theta})-F_{k,\alpha}^*}{\lvert\boldsymbol{F}^\ast_k\lvert + \nu}\right)^2.
```

where $\nu$ is a small constant used to protect
an atom where the magnitude of $\boldsymbol{F}^\ast_k$ is small from having a large $L^r_F$.
Benefiting from the relative force loss, small forces can be fitted more accurately.[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

## The fitting network

The construction of the fitting net is given by section {ref}`fitting_net <model[standard]/fitting_net>`

```json
	"fitting_net" : {
	    "neuron":		[240, 240, 240],
	    "resnet_dt":	true,
	    "seed":		1
	},
```

- {ref}`neuron <model[standard]/fitting_net[ener]/neuron>` specifies the size of the fitting net. If two neighboring layers are of the same size, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is built between them.
- If the option {ref}`resnet_dt <model[standard]/fitting_net[ener]/resnet_dt>` is set to `true`, then a timestep is used in the ResNet.
- {ref}`seed <model[standard]/fitting_net[ener]/seed>` gives the random seed that is used to generate random numbers when initializing the model parameters.

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
