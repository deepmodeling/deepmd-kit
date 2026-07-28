# Descriptor DPA-2 {{ pytorch_icon }} {{ jax_icon }} {{ paddle_icon }} {{ dpmodel_icon }}

> [!NOTE]
> **Supported backends**: PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, Paddle {{ paddle_icon }}, DP {{ dpmodel_icon }}

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

The customized OP library for the Python interface is installed by default when building DeePMD-kit from source.

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

## Graph-native inference route (pt_expt) {{ pytorch_icon }}

In the pt_expt backend, a graph-eligible DPA-2 descriptor (`repinit/use_three_body` `false` -- the three-body sub-block is not graph-eligible -- and not compressed) can be frozen through a NeighborGraph-native inference path instead of the legacy dense neighbor-list path:

```bash
dp --pt_expt freeze -o model.pt2 --lower-kind graph
```

As with DPA-1's graph path (see [Difference among different backends](train-se-atten.md#difference-among-different-backends)), the graph route considers all neighbors within the cutoff rather than a fixed, padded selection, so its numeric result can differ slightly (down to the AOTInductor floating-point noise floor at non-binding `sel`, larger if `sel` is binding) from the dense/`nlist` path.

> [!NOTE]
> **Default route change in pt_expt (eager & training).** For a graph-eligible DPA-2 descriptor, the pt_expt backend now defaults to the carry-all graph route not only for `--lower-kind graph` freezing but also in **eager inference/evaluation and in (compiled) training** (`neighbor_graph_method=None` resolves to the graph). This changes the numerical behavior of existing pt_expt configurations relative to the dense neighbor-list route (by the amounts described above — negligible at non-binding `sel`). The other backends (dpmodel/PyTorch/Paddle/TensorFlow/JAX) are unaffected: they keep the dense route as their only path.
>
> To retain the legacy dense route on pt_expt:
>
> - **Inference / evaluation:** pass `neighbor_graph_method="legacy"` to `forward_common` / `call_common` (forces the dense neighbor-list path).
> - **Training:** call `model.atomic_model.descriptor.disable_graph_lower()` on the constructed model before training. This flips `uses_graph_lower()` to `False`, which both the eager forward and the compiled-training lower honor, so both run the dense route consistently (the same mechanism the spin model uses to stay on the dense path). A first-class training-config knob for this is planned as a follow-up.

> [!NOTE]
> **Smoothness at the cutoff.** The graph route is exactly smooth at the cutoff, like the dense path. The non-attention channels (environment matrix, switch envelope, convolution, drrd/grrg, g1g1, symmetrization) are smooth by construction. The repformer *attention* channels (`update_g1_has_attn`, `update_g2_has_attn`) additionally use a fixed-phantom-count softmax: the dense smooth-attention denominator keeps exactly `sel − n_real` padding terms at $e^{-\mathrm{attnw\_shift}}$ (a geometry-independent count); the graph kernels reproduce this by excluding masked pairs from the softmax and adding $\max(\mathrm{sel} - n_\mathrm{real}, 0)$ phantom denominator terms per center. An edge entering the cutoff sphere does so at logit $-\mathrm{attnw\_shift}$ exactly while the phantom count drops by one, so the swap is value-preserving and the energy/force are continuous (verified at the float64 noise floor, $\lesssim 10^{-13}$). This also makes the carry-all graph attention agree with the dense attention term-for-term at non-binding `sel`. The only residual $e^{-20}$-scale discontinuity remains for a center with `sel` or more *real* neighbors within the block cutoff — a regime where the dense path itself suffers a far larger discontinuity from truncating a real neighbor, i.e. where `sel` is misconfigured.

DPA-2's repformer block performs message passing (per-layer neighbor feature aggregation), so a graph-frozen `.pt2` archive additionally embeds a with-comm AOTInductor artifact. Multi-rank LAMMPS runs dispatch to this artifact and drive an MPI ghost-atom exchange (`border_op`) once per repformer layer, instead of folding ghosts onto local owners as the non-message-passing (e.g. DPA-1) graph path does. `.pt2` archives frozen with `--lower-kind graph` before this artifact was introduced do not carry it and must be re-frozen to support multi-rank inference.

Per-atom virial on the graph route uses a different (but equally valid) decomposition than the dense path: each edge's full bond-virial contribution is assigned to its source (neighbor) atom, whereas the dense path's autograd-based per-atom decomposition distributes message-passing contributions across atoms differently. For non-message-passing descriptors the two conventions coincide; for DPA-2 they differ elementwise. Only the *total* virial (summed over all atoms) is convention-independent between the two paths; per-atom virial values from the graph and dense routes should not be compared directly.

Multi-rank message-passing inference on the graph route requires every MPI rank to own or ghost at least one atom -- a rank with zero atoms in both categories raises an error rather than silently desynchronizing the collective per-layer ghost exchange. Pick a domain decomposition that keeps every rank non-empty, or use the dense (`nlist`, the default) artifact, which has no such restriction.
