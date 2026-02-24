# Descriptor DPA-2 {{ pytorch_icon }} {{ jax_icon }} {{ paddle_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, Paddle {{ paddle_icon }}, DP {{ dpmodel_icon }}
:::

The DPA-2 model implementation. See [DPA-2 paper](https://doi.org/10.1038/s41524-024-01493-2) for more details.

Training example: `examples/water/dpa2/input_torch_medium.json`, see [README](../../examples/water/dpa2/README.md) for inputs in different levels.

## Theory

DPA-2 is an attention-based descriptor designed to learn expressive local atomic representations while preserving the physical symmetries required by interatomic potentials.

### Local environment and representation

For each central atom $\alpha$, neighbors $\beta \in \mathcal{N}(\alpha)$ are selected within a cutoff radius. DPA-2 encodes each local environment through geometric features (relative coordinates and derived invariants) and element/type information.

The descriptor is built hierarchically:

1. **Initial embedding**: geometric and type features are projected into latent channels.
1. **Attention-based interaction**: stacked attention layers model neighbor-neighbor and center-neighbor correlations in the local environment.
1. **Output descriptor**: atom-wise latent features after the final layer are used as descriptor outputs for downstream fitting/model components.

### Attention-based message passing

DPA-2 uses attention to aggregate neighbor information with data-dependent weights. Conceptually, each layer computes:

```math
\mathbf{h}_\alpha^{(l+1)} = \mathbf{h}_\alpha^{(l)} + \mathrm{Attn}^{(l)}\left(\mathbf{h}_\alpha^{(l)}, \{\mathbf{h}_\beta^{(l)}\}_{\beta\in\mathcal{N}(\alpha)}, \{\mathbf{g}_{\alpha\beta}\}_{\beta\in\mathcal{N}(\alpha)}\right)
```

where $\mathbf{h}$ denotes latent node features and $\mathbf{g}_{\alpha\beta}$ denotes geometry-conditioned pair features. Residual updates enable stable deep stacking.

### Physical symmetries

DPA-2 is constructed to satisfy key symmetry requirements of atomistic modeling:

1. **Translational invariance**: only relative coordinates are used.
1. **Rotational behavior**: internal geometric constructions are designed so that final scalar descriptor channels used downstream are rotationally invariant.
1. **Permutational invariance**: atoms of the same species are treated identically under permutation (re-labeling) operations.

### Multi-task training context

DPA-2 is commonly used in a multi-task setting. The descriptor is shared, while task-specific heads/objectives are handled downstream. See [Multi-task training](../train/multi-task-training.md) for framework details.

## Requirements of installation {{ pytorch_icon }}

If one wants to run the DPA-2 model on LAMMPS, the customized OP library for the Python interface must be installed when [freezing the model](../freeze/freeze.md).

The customized OP library for the Python interface can be installed by setting environment variable {envvar}`DP_ENABLE_PYTORCH` to `1` during installation.

If one runs LAMMPS with MPI, the customized OP library for the C++ interface should be compiled against the same MPI library as the runtime MPI.
If one runs LAMMPS with MPI and CUDA devices, it is recommended to compile the customized OP library for the C++ interface with a [CUDA-Aware MPI](https://developer.nvidia.com/mpi-solutions-gpus) library and CUDA,
otherwise the communication between GPU cards falls back to the slower CPU implementation.

## Limiations of the JAX backend with LAMMPS {{ jax_icon }}

When using the JAX backend, 2 or more MPI ranks are not supported. One must set `map` to `yes` using the [`atom_modify`](https://docs.lammps.org/atom_modify.html) command.

```lammps
atom_modify map yes
```

See the example `examples/water/lmp/jax_dpa.lammps`.

## Data format

DPA-2 supports both the [standard data format](../data/system.md) and the [mixed type data format](../data/system.md#mixed-type).

## Type embedding

Type embedding is within this descriptor with the {ref}`tebd_dim <model[standard]/descriptor[dpa2]/tebd_dim>` argument.

## Model compression

Model compression is supported when {ref}`repinit/tebd_input_mode <model[standard]/descriptor[dpa2]/repinit/tebd_input_mode>` is `strip`.

- If {ref}`repinit/attn_layer <model[standard]/descriptor[dpa2]/repinit/attn_layer>` is `0`, both the type embedding and geometric parts inside `repinit` are compressed.
- If `repinit/attn_layer` is not `0`, only the type embedding tables are compressed and the geometric attention layers remain as neural networks.

An example is given in `examples/water/dpa2/input_torch_compressible.json`.
The performance improvement will be limited if other parts are more expensive.
