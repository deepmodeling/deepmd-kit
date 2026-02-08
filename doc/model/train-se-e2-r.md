# Descriptor `"se_e2_r"` {{ tensorflow_icon }} {{ pytorch_icon }} {{ jax_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, DP {{ dpmodel_icon }}
:::

The notation of `se_e2_r` is short for the Deep Potential Smooth Edition (DeepPot-SE) constructed from the radial information of atomic configurations. The `e2` stands for the embedding with two-atom information.

## Theory

The descriptor, using either radial-only information, is given by

```math
    \mathcal{D}^i = \frac{1}{N_c} \sum_j (\mathcal{G}^i)_{jk},
```

where
$N_c$ is the expected maximum number of neighboring atoms, which is the same constant for all atoms over all frames.
A matrix with a dimension of $N_c$ will be padded if the number of neighboring atoms is less than $N_c$.

Each row of the embedding matrix $\mathcal{G}^i \in \mathbb{R}^{N_c \times M}$ consists of $M$ nodes from the output layer of an NN function $\mathcal{N}_ {g}$ of $s(r_{ij})$:

```math
    (\mathcal{G}^i)_j = \mathcal{N}_{e,2}(s(r_{ij})),
```

where $\boldsymbol{r}_ {ij}=\boldsymbol{r}_ j-\boldsymbol{r}_ i = (x_{ij}, y_{ij}, z_{ij})$ is the relative coordinate and $r_{ij}=\lVert \boldsymbol{r}_{ij} \lVert$ is its norm. The switching function $s(r)$ is defined as

```math
    s(r)=
    \begin{cases}
    \frac{1}{r}, & r \lt r_s, \\
    \frac{1}{r} \big[ x^3 (-6 x^2 +15 x -10) +1 \big], & r_s \leq r \lt r_c, \\
    0, & r \geq r_c,
    \end{cases}
```

where $x=\frac{r - r_s}{ r_c - r_s}$ switches from 1 at $r_s$ to 0 at the cutoff radius $r_c$.
The switching function $s(r)$ is smooth in the sense that the second-order derivative is continuous.

In the above equations, the network parameters are not explicitly written.
$r_s$, $r_c$ and $M$ are hyperparameters provided by the user.
The DeepPot-SE is continuous up to the second-order derivative in its domain.[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

## Instructions

A complete training input script of this example can be found in the directory

```bash
$deepmd_source_dir/examples/water/se_e2_r/input.json
```

The training input script is very similar to that of [`se_e2_a`](train-se-e2-a.md). The only difference lies in the {ref}`descriptor <model[standard]/descriptor>` section

```json
	"descriptor": {
	    "type":		"se_e2_r",
	    "sel":		[46, 92],
	    "rcut_smth":	0.50,
	    "rcut":		6.00,
	    "neuron":		[5, 10, 20],
        "type_one_side": true,
	    "resnet_dt":	false,
	    "seed":		1,
	    "_comment": " that's all"
	},
```

The type of the descriptor is set by the key {ref}`type <model[standard]/descriptor/type>`.

## Type embedding support

Type embdding is only supported in the TensorFlow backends.

## Difference among different backends

In the TensorFlow backend, {ref}`env_protection <model[standard]/descriptor[se_e2_r]/env_protection>` cannot be set to a non-zero value.
In the PyTorch, JAX, and DP backend, {ref}`type_one_side <model[standard]/descriptor[se_e2_r]/type_one_side>` cannot be set to `false`.

## Model compression

Model compression is supported when type embedding is not used.
