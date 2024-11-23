# Type embedding approach {{ tensorflow_icon }} {{ pytorch_icon }} {{ jax_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, DP {{ dpmodel_icon }}
:::

We generate specific a type embedding vector for each atom type so that we can share one descriptor embedding net and one fitting net in total, which decline training complexity largely.

The training input script is similar to that of [`se_e2_a`](train-se-e2-a.md), but different by adding the {ref}`type_embedding <model/type_embedding>` section.

## Theory

Usually, when the type embedding approach is not enabled, for a system with multiple chemical species ($|\{\alpha_i\}| > 1$), parameters of the embedding network $\mathcal{N}_{e,\{2,3\}}$ are as follows chemical-species-wise:

```math
    (\mathcal{G}^i)_j = \mathcal{N}^{\alpha_i, \alpha_j}_{e,2}(s(r_{ij})) \quad \mathrm{or}\quad
    (\mathcal{G}^i)_j = \mathcal{N}^{ \alpha_j}_{e,2}(s(r_{ij})),
```

```math
    (\mathcal{G}^i)_{jk} =\mathcal{N}^{\alpha_j, \alpha_k}_{e,3}((\theta_i)_{jk}).
```

Thus, there will be $N_t^2$ or $N_t$ embedding networks where $N_t$ is the number of chemical species.
To improve the performance of matrix operations, $n(i)$ is divided into blocks of different chemical species.
Each matrix with a dimension of $N_c$ is divided into corresponding blocks, and each block is padded to $N_c^{\alpha_j}$ separately.
The limitation of this approach is that when there are large numbers of chemical species, the number of embedding networks will increase, requiring large memory and decreasing computing efficiency.

Similar to the embedding networks, if the type embedding approach is not used, the fitting network parameters are chemical-species-wise, and there are $N_t$ sets of fitting network parameters.
For performance, atoms are sorted by their chemical species $\alpha_i$ in advance.
Take an example, the atomic energy $E_i$ is represented as follows:

```math
E_i=\mathcal{F}_0^{\alpha_i}(\mathcal{D}^i).
```

To reduce the number of NN parameters and improve computing efficiency when there are large numbers of chemical species,
the type embedding $\mathcal{A}$ is introduced, represented as a NN function $\mathcal{N}_t$ of the atomic type $\alpha$:

```math
    \mathcal{A}^i = \mathcal{N}_t\big( \text{one hot}(\alpha_i) \big),
```

where $\alpha_i$ is converted to a one-hot vector representing the chemical species before feeding to the NN.
The type embeddings of central and neighboring atoms $\mathcal{A}^i$ and $\mathcal{A}^j$ are added as an extra input of the embedding network $\mathcal{N}_{e,\{2,3\}}$:

```math
    (\mathcal{G}^i)_j = \mathcal{N}_{e,2}(\{s(r_{ij}), \mathcal{A}^i, \mathcal{A}^j\})  \quad \mathrm{or}\quad
    (\mathcal{G}^i)_j = \mathcal{N}_{e,2}(\{s(r_{ij}), \mathcal{A}^j\}) ,
```

```math
    (\mathcal{G}^i)_{jk} =\mathcal{N}_{e,3}(\{(\theta_i)_{jk}, \mathcal{A}^j, \mathcal{A}^k\}).
```

In fitting networks, the type embedding is inserted into the input of the fitting networks:

```math
E_i=\mathcal{F}_0(\{\mathcal{D}^i, \mathcal{A}^i\}).
```

In this way, all chemical species share the same network parameters through the type embedding.[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

## Instructions for TensorFlow backend {{ tensorflow_icon }}

In the TensorFlow backend, the type embedding is at the model level.
The {ref}`model <model>` defines how the model is constructed, adding a section of type embedding net:

```json
    "model": {
	"type_map":	["O", "H"],
	"type_embedding":{
			...
	},
	"descriptor" :{
            ...
	},
	"fitting_net" : {
            ...
	}
    }
```

The model will automatically apply the type embedding approach and generate type embedding vectors. If the type embedding vector is detected, the descriptor and fitting net would take it as a part of the input.

The construction of type embedding net is given by {ref}`type_embedding <model/type_embedding>`. An example of {ref}`type_embedding <model/type_embedding>` is provided as follows

```json
	"type_embedding":{
	    "neuron":		[2, 4, 8],
	    "resnet_dt":	false,
	    "seed":		1
	}
```

- The {ref}`neuron <model/type_embedding/neuron>` specifies the size of the type embedding net. From left to right the members denote the sizes of each hidden layer from the input end to the output end, respectively. It takes a one-hot vector as input and output dimension equals to the last dimension of the {ref}`neuron <model/type_embedding/neuron>` list. If the outer layer is twice the size of the inner layer, then the inner layer is copied and concatenated, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is built between them.
- If the option {ref}`resnet_dt <model/type_embedding/resnet_dt>` is set to `true`, then a timestep is used in the ResNet.
- {ref}`seed <model/type_embedding/seed>` gives the random seed that is used to generate random numbers when initializing the model parameters.

A complete training input script of this example can be found in the directory.

```bash
$deepmd_source_dir/examples/water/se_e2_a_tebd/input.json
```

See [here](../development/type-embedding.md) for further explanation of `type embedding`.

See documentation for each descriptor for details.

## Instructions for other backends

In other backends, the type embedding is within the descriptor itself.

See documentation for each descriptor for details.
