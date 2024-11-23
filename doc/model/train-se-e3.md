# Descriptor `"se_e3"` {{ tensorflow_icon }} {{ pytorch_icon }} {{ jax_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, DP {{ dpmodel_icon }}
:::

The notation of `se_e3` is short for three-body embedding DeepPot-SE, which incorporates embedded bond-angle information.
The embedding takes bond angles between a central atom and its two neighboring atoms as input (denoted by `e3`).

## Theory

The three-body embedding DeepPot-SE descriptor incorporates bond-angle information, making the model more accurate. The descriptor $\mathcal{D}^i$ can be represented as

```math
    \mathcal{D}^i = \frac{1}{N_c^2}(\mathcal{R}^i(\mathcal{R}^i)^T):\mathcal{G}^i,
```

where
$N_c$ is the expected maximum number of neighboring atoms, which is the same constant for all atoms over all frames.
$\mathcal{R}^i$ is constructed as

```math
    (\mathcal{R}^i)_j =
    \{
    \begin{array}{cccc}
    s(r_{ij}) & \frac{s(r_{ij})x_{ij}}{r_{ij}} & \frac{s(r_{ij})y_{ij}}{r_{ij}} & \frac{s(r_{ij})z_{ij}}{r_{ij}}
    \end{array}
    \},
```

Currently, only the full information case of $\mathcal{R}^i$ is supported by the three-body embedding.
Each element of $\mathcal{G}^i \in \mathbb{R}^{N_c \times N_c \times M}$ comes from $M$ nodes from the output layer of an NN $\mathcal{N}_{e,3}$ function:

```math
    (\mathcal{G}^i)_{jk}=\mathcal{N}_{e,3}((\theta_i)_{jk}),
```

where $(\theta_i)_ {jk} = (\mathcal{R}^i)_ {j,\\{2,3,4\\}}\cdot (\mathcal{R}^i)_ {k,\\{2,3,4\\}}$ considers the angle form of two neighbours ($j$ and $k$).
The notation $:$ in the equation indicates the contraction between matrix $\mathcal{R}^i(\mathcal{R}^i)^T$ and the first two dimensions of tensor $\mathcal{G}^i$.[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

## Instructions

A complete training input script of this example can be found in the directory

```bash
$deepmd_source_dir/examples/water/se_e3/input.json
```

The training input script is very similar to that of [`se_e2_a`](train-se-e2-a.md). The only difference lies in the `descriptor <model[standard]/descriptor>` section

```json
	"descriptor": {
	    "type":		"se_e3",
	    "sel":		[40, 80],
	    "rcut_smth":	0.50,
	    "rcut":		6.00,
	    "neuron":		[2, 4, 8],
	    "resnet_dt":	false,
	    "seed":		1,
	    "_comment":		" that's all"
	},
```

The type of the descriptor is set by the key {ref}`type <model[standard]/descriptor/type>`.

## Type embedding

Use [`se_e3_tebd`](./train-se-e3-tebd.md) for type embedding support.

## Difference among different backends

In the TensorFlow backend, {ref}`env_protection <model[standard]/descriptor[se_e3]/env_protection>` cannot be set to a non-zero value.

## Model compression

Model compression is supported.
