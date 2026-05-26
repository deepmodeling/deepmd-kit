# Descriptor DPA-2 {{ pytorch_icon }} {{ jax_icon }} {{ paddle_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, Paddle {{ paddle_icon }}, DP {{ dpmodel_icon }}
:::

The DPA-2 model implementation. See [DPA-2 paper](https://doi.org/10.1038/s41524-024-01493-2) for more details.

Training example: `examples/water/dpa2/input_torch_medium.json`, see [README](../../examples/water/dpa2/README.md) for inputs in different levels.

## Theory

DPA-2 is an attention-based descriptor architecture proposed for large atomic models (LAMs); see the [DPA-2 paper](https://doi.org/10.1038/s41524-024-01493-2).

At a high level, DPA-2 builds local representations with three coupled channels (paper notation):

- **Single-atom channel** $\mathbf{f}_\alpha$
- **Rotationally invariant pair channel** $\mathbf{g}_{\alpha\beta}$
- **Rotationally equivariant pair channel** $\mathbf{h}_{\alpha\beta}$

for neighbors $\beta\in\mathcal{N}(\alpha)$ within cutoffs.

### Descriptor pipeline

The descriptor follows two main stages:

1. **repinit (representation initializer)**
   - Initializes and fuses type and geometry information from local environments.
1. **repformer (representation transformer)**
   - Stacked message-passing layers that update $\mathbf{f}$, $\mathbf{g}$, and per-atom representations $\mathbf{h}$ through convolution/symmetrization/MLP and attention-style interactions.

The final descriptor is formed from learned single-atom representations and then passed to downstream fitting/model components.

### Message-passing intuition

DPA-2 updates local features layer-by-layer with residual connections. Conceptually, each layer performs neighborhood aggregation using geometry-conditioned interactions:

```math
\mathbf{h}_\alpha^{(l+1)} = \mathbf{h}_\alpha^{(l)} + \mathrm{MP}^{(l)}\left(\mathbf{h}_\alpha^{(l)}, \{\mathbf{h}_\beta^{(l)}\}_{\beta\in\mathcal{N}(\alpha)}, \{\mathbf{g}_{\alpha\beta}\}_{\beta\in\mathcal{N}(\alpha)}\right)
```

where $\mathrm{MP}^{(l)}$ denotes the layer-specific message-passing operator.

### Physical properties

Consistent with the DPA-2 design goals in the paper, the model family is built to satisfy:

1. **Translational invariance** (depends on relative coordinates)
1. **Rotational and permutational symmetry requirements**
1. **Conservative formulation** when used in energy models (forces/virials from energy gradients)
1. **Smoothness up to second-order derivatives**

### Multi-task training context

DPA-2 is designed for multi-task pre-training with a shared descriptor and task-specific downstream heads. See [Multi-task training](../train/multi-task-training.md) for workflow details.

## Requirements of installation {{ pytorch_icon }}

If one wants to run the DPA-2 model on LAMMPS, the customized OP library for the Python interface must be installed when [freezing the model](../freeze/freeze.md).

The customized OP library for the Python interface can be installed by setting environment variable {envvar}`DP_ENABLE_PYTORCH` to `1` during installation.

If one runs LAMMPS with MPI, the customized OP library for the C++ interface should be compiled against the same MPI library as the runtime MPI.
If one runs LAMMPS with MPI and CUDA devices, it is recommended to compile the customized OP library for the C++ interface with a [CUDA-Aware MPI](https://developer.nvidia.com/mpi-solutions-gpus) library and CUDA,
otherwise the communication between GPU cards falls back to the slower CPU implementation.

## Limitations of the JAX backend with LAMMPS {{ jax_icon }}

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
