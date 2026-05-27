# Descriptor DPA3 {{ pytorch_icon }} {{ jax_icon }} {{ paddle_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, DP {{ dpmodel_icon }}
:::

DPA3 is an advanced interatomic potential based on message passing.
As a large atomic model (LAM), it is designed to integrate and jointly train on datasets from different domains,
covering diverse chemical and materials systems.
Its architecture provides high fitting accuracy and robust generalization both within and beyond the training domain.
DPA3 also preserves energy conservation and the physical symmetries of the potential energy surface,
making it a reliable model for a wide range of scientific applications.

Reference: [DPA3 paper](https://arxiv.org/abs/2506.01686).

Training example: `examples/water/dpa3/input_torch.json`.

## Theory

DPA3 is a graph neural network operating on the Line Graph Series (LiGS) constructed from atomic configurations.

### Line Graph Series (LiGS)

Given an initial graph $G^{(1)}$ representing the atomic system, where atoms are vertices and pairs of neighboring atoms within a cutoff radius $r_c$ are edges, the line graph transform $\mathcal{L}$ constructs a new graph $G^{(2)} = \mathcal{L}(G^{(1)})$ by:

1. Converting each edge in $G^{(1)}$ to a vertex in $G^{(2)}$
1. Creating edges in $G^{(2)}$ between vertices whose corresponding edges in $G^{(1)}$ share a common vertex

Recursively applying this transform generates a series of graphs $\{G^{(1)}, G^{(2)}, \ldots, G^{(K)}\}$, where $G^{(k)} = \mathcal{L}(G^{(k-1)})$. This sequence is called the Line Graph Series (LiGS) of order $K$.

Geometrically, vertices in $G^{(1)}$, $G^{(2)}$, $G^{(3)}$, and $G^{(4)}$ correspond to atoms, bonds (pairs of atoms), angles (three atoms with two bonds sharing a common atom), and dihedral angles (four atoms with two angles sharing a common bond), respectively.

### Message Passing on LiGS

DPA3 performs message passing across all graphs in the LiGS. At layer $l$, the vertex and edge features on graph $G^{(k)}$ are denoted as $\mathbf{v}_\alpha^{(k,l)} \in \mathbb{R}^{d_v^{(k)}}$ and $\mathbf{e}_{\alpha\beta}^{(k,l)} \in \mathbb{R}^{d_e^{(k)}}$, where $\alpha$ and $\alpha\beta$ denote vertex and edge indices, and $d_v^{(k)}$, $d_e^{(k)}$ are per-graph feature dimensions (for example, in `RepFlowArgs`: $d_v^{(1)}=n_\text{dim}$, $d_e^{(1)}=e_\text{dim}$, $d_v^{(2)}=e_\text{dim}$, and $d_e^{(2)}=a_\text{dim}$).

The feature update follows a recursive formulation with residual connections. We use $\text{Update}_V$ and $\text{Update}_E$ to distinguish vertex and edge update modules, respectively:

**Edge updates (all graphs $G^{(k)}$):**
Edge features are updated based on messages from connected vertices:

```math
\mathbf{e}_{\alpha\beta}^{(k,l+1)} = \mathbf{e}_{\alpha\beta}^{(k,l)} + \text{Update}_E^{(k)}\left(\mathbf{e}_{\alpha\beta}^{(k,l)}, \mathbf{v}_\alpha^{(k,l)}, \mathbf{v}_\beta^{(k,l)}\right)
```

**For $G^{(1)}$ (initial graph, vertex update):**
Vertex features are updated through self-message and symmetrization:

```math
\mathbf{v}_\alpha^{(1,l+1)} = \mathbf{v}_\alpha^{(1,l)} + \text{Update}_V^{(1)}\left(\mathbf{v}_\alpha^{(1,l)}, \{\mathbf{e}_{\alpha\beta}^{(1,l)}\}_{\beta \in \mathcal{N}(\alpha)}\right)
```

**For $G^{(k)}$ with $k > 1$ (vertex identity):**
The vertex feature of $G^{(k)}$ is identical to the edge feature of $G^{(k-1)}$:

```math
\mathbf{v}_\alpha^{(k,l)} = \mathbf{e}_{\alpha\beta}^{(k-1,l)}
```

where $(\alpha,\beta)$ denotes the edge in $G^{(k-1)}$ corresponding to vertex $\alpha$ in $G^{(k)}$. This identity eliminates redundant storage.

The same edge update rule also applies to $G^{(1)}$ edge features $\mathbf{e}_{\alpha\beta}^{(1,l)}$ (i.e., with $k=1$ in $\text{Update}_E^{(k)}$). Therefore, these features evolve across layers and, via the $\mathbf{v}^{(2,l)}$-$\mathbf{e}^{(1,l)}$ identity, drive the updates on $G^{(2)}$.

### Descriptor Construction

The final vertex features of $G^{(1)}$ serve as the descriptor representing the local environment of each atom:

```math
\mathcal{D}^\alpha = \mathbf{v}_\alpha^{(1,L)}
```

where $L$ is the total number of layers.

The descriptor output is then consumed by downstream fitting/model components for property prediction (e.g., energy). See the model/fitting documentation for those equations and training objectives.

### Physical Symmetries and Conservative Forces

DPA3 respects the physical symmetries of the potential energy surface:

1. **Translational invariance**: The model depends only on relative coordinates $\mathbf{r}_{\alpha\beta} = \mathbf{r}_\beta - \mathbf{r}_\alpha$, not absolute positions.

1. **Rotational invariance**: The final descriptor is rotationally invariant; intermediate equivariant representations are used internally and contracted to produce invariant atomic features.

1. **Permutational invariance**: Atoms of the same chemical species are treated identically under permutation symmetry operations (re-labeling) of identical atoms.

In addition, DPA3 is inherently conservative: forces are derived from energy gradients:

```math
\mathbf{F}_\alpha = -\frac{\partial E}{\partial \mathbf{r}_\alpha}
```

Virials are similarly derived from cell tensor gradients, ensuring the model is conservative and suitable for molecular dynamics simulations.

### Default Configuration

DPA3 uses LiGS order $K=2$ as the default configuration, which was found effective in prior work ([DPA3 paper](https://arxiv.org/abs/2506.01686)). The model supports scaling through increasing the number of layers $L$ (e.g., DPA3-L3, DPA3-L6, DPA3-L12, DPA3-L24).

## Hyperparameter tests

We systematically trained DPA3 on six representative DFT datasets (available at [AIS-Square](https://www.aissquare.com/datasets/detail?pageType=datasets&name=DPA3_hyperparameter_search&id=316)): metallic systems (`Alloy`, `AlMgCu`, `W`), a covalent material (`Boron`), a molecular system (`Drug`), and liquid water (`Water`).
Under consistent training conditions (0.5M training steps, `batch_size` = `auto:128`),
we evaluated the impact of key hyperparameters on validation accuracy.

The comparative analysis focused on average RMSEs (Root Mean Square Error) for energy, force, and virial predictions across the six systems.
The results are summarized below to guide scenario-specific hyperparameter selection:

| Model            | comment         | nlayers | n_dim   | e_dim  | a_dim | e_sel   | a_sel  | start_lr | stop_lr  | loss prefactors           | rmse_e (meV/atom) | rmse_f (meV/Å) | rmse_v (meV/atom) | Training wall time (h) |
| ---------------- | --------------- | ------- | ------- | ------ | ----- | ------- | ------ | -------- | -------- | ------------------------- | ----------------- | -------------- | ----------------- | ---------------------- |
| DPA3-L3          | Default         | 3       | 256     | 128    | 32    | 120     | 30     | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 5.74              | 85.4           | 43.1              | 9.8                    |
|                  | Small dimension | 3       | **128** | **64** | 32    | 120     | 30     | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 6.99              | 93.6           | 46.7              | 8.0                    |
|                  | Large sel       | 3       | 256     | 128    | 32    | **154** | **48** | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 5.70              | 83.7           | 43.4              | 14.1                   |
| DPA3-L6          | Default         | 6       | 256     | 128    | 32    | 120     | 30     | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 4.85              | 79.9           | 39.7              | 19.2                   |
|                  | Small dimension | 6       | **128** | **64** | 32    | 120     | 30     | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 5.11              | 77.7           | 41.2              | 14.1                   |
|                  | Large sel       | 6       | 256     | 128    | 32    | **154** | **48** | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 4.76              | 78.4           | 40.2              | 31.8                   |
| DPA2-L6 (medium) | Default         | 6       | -       | -      | -     | -       | -      | 1e-3     | 3.51e-08 | 0.02\|1, 1000\|1, 0.02\|1 | 12.12             | 109.3          | 83.1              | 12.2                   |

The loss prefactors (0.2|20, 100|60, 0.02|1) correspond to (`start_pref_e`|`limit_pref_e`, `start_pref_f`|`limit_pref_f`, `start_pref_v`|`limit_pref_v`), respectively.
Virial RMSEs were averaged exclusively for systems containing virial labels (`Alloy`, `AlMgCu`, `W`, and `Boron`).

Note that all DPA3 models use `float32`, while other models use `float64` by default.

## Requirements of installation from source code {{ pytorch_icon }} {{ paddle_icon }}

::::{tab-set}

:::{tab-item} PyTorch {{ pytorch_icon }}

To run the DPA3 model on LAMMPS via source code installation
(users can skip this step if using [easy installation](../install/easy-install.md)),
the custom OP library for Python interface integration must be compiled and linked
during the [model freezing process](../freeze/freeze.md).

The customized OP library for the Python interface can be installed by setting environment variable {envvar}`DP_ENABLE_PYTORCH` to `1` during installation.

If one runs LAMMPS with MPI, the customized OP library for the C++ interface should be compiled against the same MPI library as the runtime MPI.
If one runs LAMMPS with MPI and CUDA devices, it is recommended to compile the customized OP library for the C++ interface with a [CUDA-Aware MPI](https://developer.nvidia.com/mpi-solutions-gpus) library and CUDA,
otherwise the communication between GPU cards falls back to the slower CPU implementation.
:::

:::{tab-item} Paddle {{ paddle_icon }}

The customized OP library for the Python interface can be installed by

```sh
cd deepmd-kit/source/op/pd
python setup.py install
```

If one runs LAMMPS with MPI, the customized OP library for the C++ interface should be compiled against the same MPI library as the runtime MPI.
If one runs LAMMPS with MPI and CUDA devices, it is recommended to compile the customized OP library for the C++ interface with a [CUDA-Aware MPI](https://developer.nvidia.com/mpi-solutions-gpus) library and CUDA,
otherwise the communication between GPU cards falls back to the slower CPU implementation.
:::

::::

## Limitations of the JAX backend with LAMMPS {{ jax_icon }}

When using the JAX backend, 2 or more MPI ranks are not supported. One must set `map` to `yes` using the [`atom_modify`](https://docs.lammps.org/atom_modify.html) command.

```lammps
atom_modify map yes
```

See the example `examples/water/lmp/jax_dpa.lammps`.

## Data format

DPA3 supports both the [standard data format](../data/system.md) and the [mixed type data format](../data/system.md#mixed-type).

## Type embedding

Type embedding is within this descriptor with the same dimension as the node embedding: {ref}`n_dim <model[standard]/descriptor[dpa3]/repflow/n_dim>` argument.

## Model compression

Model compression is not supported in this descriptor.
