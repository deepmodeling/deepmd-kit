# Descriptor `"se_e2_a"` {{ tensorflow_icon }} {{ pytorch_icon }} {{ jax_icon }} {{ paddle_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, Paddle {{ paddle_icon }}, DP {{ dpmodel_icon }}
:::

The notation of `se_e2_a` is short for the Deep Potential Smooth Edition (DeepPot-SE) constructed from all information (both angular and radial) of atomic configurations. The `e2` stands for the embedding with two-atoms information. This descriptor was described in detail in [the DeepPot-SE paper](https://arxiv.org/abs/1805.09003).

Note that it is sometimes called a "two-atom embedding descriptor" which means the input of the embedding net is atomic distances. The descriptor **does** encode multi-body information (both angular and radial information of neighboring atoms).

## Theory

The two-body embedding smooth edition of the DP descriptor $\mathcal{D}^i \in \mathbb{R}^{M \times M_{<}}$, is usually named DeepPot-SE descriptor.
It is noted that the descriptor is a multi-body representation of the local environment of the atom $i$.
We call it two-body embedding because the embedding network takes only the distance between atoms $i$ and $j$ (see below), but it is not implied that the descriptor takes only the pairwise information between $i$ and its neighbors.
The descriptor, using full information, is given by

```math
    \mathcal{D}^i = \frac{1}{N_c^2} (\mathcal{G}^i)^T \mathcal{R}^i (\mathcal{R}^i)^T \mathcal{G}^i_<,
```

where
$N_c$ is the expected maximum number of neighboring atoms, which is the same constant for all atoms over all frames.
A matrix with a dimension of $N_c$ will be padded if the number of neighboring atoms is less than $N_c$. $\mathcal{R}^i \in \mathbb{R}^{N_c \times 4}$ is the coordinate matrix, and each row of $\mathcal{R}^i$ can be constructed as

```math
    (\mathcal{R}^i)_j =
    \{
    \begin{array}{cccc}
    s(r_{ij}) & \frac{s(r_{ij})x_{ij}}{r_{ij}} & \frac{s(r_{ij})y_{ij}}{r_{ij}} & \frac{s(r_{ij})z_{ij}}{r_{ij}}
    \end{array}
    \},
```

where $\boldsymbol{r}_{ij}=\boldsymbol{r}_j-\boldsymbol{r}_i = (x_{ij}, y_{ij}, z_{ij})$ is the relative coordinate and $r_{ij}=\lVert \boldsymbol{r}_{ij} \lVert$ is its norm. The switching function $s(r)$ is defined as

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

Each row of the embedding matrix $\mathcal{G}^i \in \mathbb{R}^{N_c \times M}$ consists of $M$ nodes from the output layer of an NN function $\mathcal{N}_ {g}$ of $s(r_{ij})$:

```math
    (\mathcal{G}^i)_j = \mathcal{N}_{e,2}(s(r_{ij})),
```

where the subscript $e,2$ is used to distinguish the NN from other NNs used in the DP model.
In the above equation, the network parameters are not explicitly written.
$\mathcal{G}^i_< \in \mathbb{R}^{N_c \times M_<}$ only takes first $M_<$ columns of $\mathcal{G}^i$ to reduce the size of $\mathcal D^i$.
$r_s$, $r_c$, $M$ and $M_<$ are hyperparameters provided by the user.
The DeepPot-SE is continuous up to the second-order derivative in its domain.[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

## Instructions

In this example, we will train a DeepPot-SE model for a water system. A complete training input script of this example can be found in the directory.

```bash
$deepmd_source_dir/examples/water/se_e2_a/input.json
```

With the training input script, data are also provided in the example directory. One may train the model with the DeePMD-kit from the directory.

The construction of the descriptor is given by section {ref}`descriptor <model[standard]/descriptor>`. An example of the descriptor is provided as follows

```json
	"descriptor" :{
	    "type":		"se_e2_a",
	    "rcut_smth":	0.50,
	    "rcut":		6.00,
	    "sel":		[46, 92],
	    "neuron":		[25, 50, 100],
	    "type_one_side":	true,
	    "axis_neuron":	16,
	    "resnet_dt":	false,
	    "seed":		1
	}
```

- The {ref}`type <model[standard]/descriptor/type>` of the descriptor is set to `"se_e2_a"`.
- {ref}`rcut <model[standard]/descriptor[se_e2_a]/rcut>` is the cut-off radius for neighbor searching, and the {ref}`rcut_smth <model[standard]/descriptor[se_e2_a]/rcut_smth>` gives where the smoothing starts.
- {ref}`sel <model[standard]/descriptor[se_e2_a]/sel>` gives the maximum possible number of neighbors in the cut-off radius. It is a list, the length of which is the same as the number of atom types in the system, and `sel[i]` denotes the maximum possible number of neighbors with type `i`.
- The {ref}`neuron <model[standard]/descriptor[se_e2_a]/neuron>` specifies the size of the embedding net. From left to right the members denote the sizes of each hidden layer from the input end to the output end, respectively. If the outer layer is twice the size of the inner layer, then the inner layer is copied and concatenated, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is built between them.
- If the option {ref}`type_one_side <model[standard]/descriptor[se_e2_a]/type_one_side>` is set to `true`, the embedding network parameters vary by types of neighbor atoms only, so there will be $N_\text{types}$ sets of embedding network parameters. Otherwise, the embedding network parameters vary by types of centric atoms and types of neighbor atoms, so there will be $N_\text{types}^2$ sets of embedding network parameters.
- The {ref}`axis_neuron <model[standard]/descriptor[se_e2_a]/axis_neuron>` specifies the size of the submatrix of the embedding matrix, the axis matrix as explained in the [DeepPot-SE paper](https://arxiv.org/abs/1805.09003)
- If the option {ref}`resnet_dt <model[standard]/descriptor[se_e2_a]/resnet_dt>` is set to `true`, then a timestep is used in the ResNet.
- {ref}`seed <model[standard]/descriptor[se_e2_a]/seed>` gives the random seed that is used to generate random numbers when initializing the model parameters.

## Type embedding support

Type embdding is only supported in the TensorFlow backends.
`se_e2_a` with type embedding and [`se_atten`](./train-se-atten.md) (or its updated version) without any attention layer are mathematically equivalent, so `se_atten` can be a substitute in other backends.

## Difference among different backends

In the TensorFlow backend, {ref}`env_protection <model[standard]/descriptor[se_e2_a]/env_protection>` cannot be set to a non-zero value.
In the JAX backend, {ref}`type_one_side <model[standard]/descriptor[se_e2_a]/type_one_side>` cannot be set to `false`.

## Model compression

Model compression is supported when type embedding is not used.
To use model compression with type embedding in the TensorFlow backend, use `se_a_tebd_v2` instead.
