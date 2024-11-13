# Descriptor `"se_e3_tebd"` {{ pytorch_icon }} {{ jax_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, DP {{ dpmodel_icon }}
:::

The notation of `se_e3_tebd` is short for the three-body embedding descriptor with type embeddings, where the notation `se` denotes the Deep Potential Smooth Edition (DeepPot-SE).
The embedding takes bond angles between a central atom and its two neighboring atoms (denoted by `e3`) and their type embeddings (denoted by `tebd`) as input.

## Theory

The three-body embedding DeepPot-SE descriptor with type embeddings incorporates bond-angle and type information, making the model more accurate. The descriptor $\mathcal{D}^i$ can be represented as

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

where $s(r_{ij})$ is the switch function between central atom $i$ and neighbor atom $j$, which is the same as that in `se_e2_a`.

Currently, only the full information case of $\mathcal{R}^i$ is supported by the three-body embedding.
Each element of $\mathcal{G}^i \in \mathbb{R}^{N_c \times N_c \times M}$ comes from $M$ nodes from the output layer of an NN function $\mathcal{N}_{e,3}$.
If `tebd_input_mode` is set to `concat`, the formulation is:

```math
    (\mathcal{G}^i)_{jk}=\mathcal{N}_{e,3}((\theta_i)_{jk}, \mathcal{A}^j, \mathcal{A}^k)
```

Otherwise, if `tebd_input_mode` is set to `strip`, the angular and type information will be taken into two separate NNs $\mathcal{N}_{e,3}^{s}$ and $\mathcal{N}_{e,3}^{t}$. The formulation is:

```math
    (\mathcal{G}^i)_{jk}=\mathcal{N}_{e,3}^{s}((\theta_i)_{jk}) + \mathcal{N}_{e,3}^{s}((\theta_i)_{jk}) \odot ( \mathcal{N}_{e,3}^{t}(\mathcal{A}^j, \mathcal{A}^k) \odot s(r_{ij}) \odot s(r_{ik}))
```

where $(\theta_i)_ {jk} = (\mathcal{R}^i)_ {j,\\{2,3,4\\}}\cdot (\mathcal{R}^i)_ {k,\\{2,3,4\\}}$ considers the angle form of two neighbours ($j$ and $k$).
The type embeddings of neighboring atoms $\mathcal{A}^j$ and $\mathcal{A}^k$ are added as an extra input of the embedding network.
The notation $:$ in the equation indicates the contraction between matrix $\mathcal{R}^i(\mathcal{R}^i)^T$ and the first two dimensions of tensor $\mathcal{G}^i$.

## Instructions

A complete training input script of this example can be found in the directory

```bash
$deepmd_source_dir/examples/water/se_e3_tebd/input.json
```

The training input script is very similar to that of [`se_e2_a`](train-se-e2-a.md). The only difference lies in the {ref}`descriptor <model[standard]/descriptor>` section

```json
	"descriptor": {
      "type": "se_e3_tebd",
      "sel": 40,
      "rcut_smth": 0.5,
      "rcut": 4.0,
      "neuron": [
        2,
        4,
        8
      ],
      "tebd_dim": 8,
      "tebd_input_mode": "concat",
      "activation_function": "tanh"
	},
```

The type of the descriptor is set by the key {ref}`type <model[standard]/descriptor/type>`.

## Type embedding

Type embedding is within this descriptor with the {ref}`tebd_dim <model[standard]/descriptor[se_e3_tebd]/tebd_dim>` argument.

## Model compression

Model compression is not supported.
