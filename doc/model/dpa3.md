# Descriptor DPA3 {{ pytorch_icon }} {{ jax_icon }} {{ paddle_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, DP {{ dpmodel_icon }}
:::

DPA3 is an advanced interatomic potential leveraging the message passing architecture.
Designed as a large atomic model (LAM), DPA3 is tailored to integrate and simultaneously train on datasets from various disciplines,
encompassing diverse chemical and materials systems across different research domains.
Its model design ensures exceptional fitting accuracy and robust generalization both within and beyond the training domain.
Furthermore, DPA3 maintains energy conservation and respects the physical symmetries of the potential energy surface,
making it a dependable tool for a wide range of scientific applications.

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

DPA3 performs message passing across all graphs in the LiGS. At layer $l$, the vertex and edge features on graph $G^{(k)}$ are denoted as $\mathbf{v}_\alpha^{(k,l)} \in \mathbb{R}^{d_v}$ and $\mathbf{e}_{\alpha\beta}^{(k,l)} \in \mathbb{R}^{d_e}$, where $\alpha$ and $\alpha\beta$ denote vertex and edge indices, and $d_v$, $d_e$ are feature dimensions.

The feature update follows a recursive formulation with residual connections:

**For $G^{(1)}$ (initial graph):**
The vertex features are updated through self-message and symmetrization:

```math
\mathbf{v}_\alpha^{(1,l+1)} = \mathbf{v}_\alpha^{(1,l)} + \text{Update}^{(1)}\left(\mathbf{v}_\alpha^{(1,l)}, \{\mathbf{e}_{\alpha\beta}^{(1,l)}\}_{\beta \in \mathcal{N}(\alpha)}\right)
```

**For $G^{(k)}$ with $k > 1$:**
The vertex feature of $G^{(k)}$ is identical to the edge feature of $G^{(k-1)}$. This identity eliminates redundant storage:

```math
\mathbf{v}_\alpha^{(k,l)} = \mathbf{e}_{\alpha}^{(k-1,l)}
```

The edge features are updated based on messages from connected vertices:

```math
\mathbf{e}_{\alpha\beta}^{(k,l+1)} = \mathbf{e}_{\alpha\beta}^{(k,l)} + \text{Update}^{(k)}\left(\mathbf{e}_{\alpha\beta}^{(k,l)}, \mathbf{v}_\alpha^{(k,l)}, \mathbf{v}_\beta^{(k,l)}\right)
```

### Descriptor Construction

The final vertex features of $G^{(1)}$ serve as the descriptor representing the local environment of each atom:

```math
\mathcal{D}^i = \mathbf{v}_i^{(1,L)}
```

where $L$ is the total number of layers.

For multi-task training, the descriptor is augmented with dataset encoding (typically a one-hot vector) and passed through a fitting network to predict atomic energies:

```math
E_i = \mathcal{N}_{\text{fit}}(\mathcal{D}^i \oplus \mathbf{d}_{\text{dataset}})
```

The total system energy is the sum of atomic contributions:

```math
E = \sum_i E_i
```

### Physical Symmetries

DPA3 respects all physical symmetries of the potential energy surface:

1. **Translational invariance**: The model depends only on relative coordinates $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$, not absolute positions.

1. **Rotational invariance**: The descriptor is constructed from scalar features that are invariant under global rotations.

1. **Permutational invariance**: Atoms of the same chemical species are treated identically, respecting quantum statistics.

1. **Energy conservation**: Forces are derived from energy gradients:

```math
\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}
```

Virials are similarly derived from cell tensor gradients, ensuring the model is conservative and suitable for molecular dynamics simulations.

### Default Configuration

Based on extensive hyperparameter tests, DPA3 uses LiGS order $K=2$ as the default, which provides optimal balance between accuracy and computational cost. The model supports scaling through increasing the number of layers $L$ (e.g., DPA3-L3, DPA3-L6, DPA3-L12, DPA3-L24).

## Hyperparameter tests

We systematically conducted DPA3 training on six representative DFT datasets (available at [AIS-Square](https://www.aissquare.com/datasets/detail?pageType=datasets&name=DPA3_hyperparameter_search&id=316)):
metallic systems (`Alloy`, `AlMgCu`, `W`), covalent material (`Boron`), molecular system (`Drug`), and liquid water (`Water`).
Under consistent training conditions (0.5M training steps, batch_size "auto:128"),
we rigorously evaluated the impacts of some critical hyperparameters on validation accuracy.

The comparative analysis focused on average RMSEs (Root Mean Square Error) for both energy, force and virial predictions across all six systems,
with results tabulated below to guide scenario-specific hyperparameter selection:

| Model            | comment         | nlayers | n_dim   | e_dim  | a_dim | e_sel   | a_sel  | start_lr | stop_lr  | loss prefactors           | rmse_e (meV/atom) | rmse_f (meV/Å) | rmse_v (meV/atom) | Training wall time (h) |
| ---------------- | --------------- | ------- | ------- | ------ | ----- | ------- | ------ | -------- | -------- | ------------------------- | ----------------- | -------------- | ----------------- | ---------------------- |
| DPA3-L3          | Default         | 3       | 256     | 128    | 32    | 120     | 30     | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 5.74              | 85.4           | 43.1              | 9.8                    |
|                  | Small dimension | 3       | **128** | **64** | 32    | 120     | 30     | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 6.99              | 93.6           | 46.7              | 8.0                    |
|                  | Large sel       | 3       | 256     | 128    | 32    | **154** | **48** | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 5.70              | 83.7           | 43.4              | 14.1                   |
| DPA3-L6          | Default         | 6       | 256     | 128    | 32    | 120     | 30     | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 4.85              | 79.9           | 39.7              | 19.2                   |
|                  | Small dimension | 6       | **128** | **64** | 32    | 120     | 30     | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 5.11              | 77.7           | 41.2              | 14.1                   |
|                  | Large sel       | 6       | 256     | 128    | 32    | **154** | **48** | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 4.76              | 78.4           | 40.2              | 31.8                   |
| DPA2-L6 (medium) | Default         | 6       | -       | -      | -     | -       | -      | 1e-3     | 3.51e-08 | 0.02\|1, 1000\|1, 0.02\|1 | 12.12             | 109.3          | 83.1              | 12.2                   |

The loss prefactors (0.2|20, 100|60, 0.02|1) correspond to (`start_pref_e`|`limit_pref_e`, `start_pref_f`|`limit_pref_f`, `start_pref_v`|`limit_pref_v`) respectively.
Virial RMSEs were averaged exclusively for systems containing virial labels (`Alloy`, `AlMgCu`, `W`, and `Boron`).

Note that we set `float32` in all DPA3 models, while `float64` in other models by default.

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
