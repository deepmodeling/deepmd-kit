# Descriptor DPA4 {{ pytorch_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}
:::

DPA4 is the DeePMD-kit implementation of the SeZM (Smooth Equivariant
Zone-bridging Model) architecture. Use `model.type: "dpa4"` in new input
files. The aliases `DPA4`, `SeZM`, and `sezm` are accepted for the same
implementation. The DPA4 model scaffold uses the SeZM descriptor and the
`dpa4_ener` fitting network, so `descriptor.type` and `fitting_net.type`
may be omitted in ordinary DPA4 inputs.

Reference: [DPA4 paper](https://arxiv.org/abs/2606.02419).

Training example: `examples/water/dpa4/input.json`.

Quick start:

```bash
cd examples/water/dpa4
dp --pt train input.json
```

## Overview

DPA4/SeZM is an SO(3)-equivariant message-passing model for conservative
interatomic potentials. It predicts atomic energies and obtains forces and
virials by differentiating the energy, following the same conservative
formulation used by standard DeePMD energy models:

```math
\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}.
```

The model keeps vector and higher-order angular information while building
the descriptor. Only the final descriptor sent to the fitting network is
scalar. This separates geometric representation from energy prediction:
equivariant layers encode local environments, and the fitting network maps
the resulting scalar features to atomic energies.

## Model scaffold

The DPA4 model type is a convenience scaffold around the SeZM descriptor and
the `dpa4_ener` energy fitting network. A minimal input therefore only needs
the model type, `type_map`, and descriptor settings such as `sel` and `rcut`:

```json
{
  "model": {
    "type": "dpa4",
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "sel": 120,
      "rcut": 6.0
    }
  }
}
```

Options that are not written in the input use their documented defaults.
The neighbor selection `sel` may be an integer total neighbor limit, a
per-type list, or `auto` / `auto:factor`.

Internally, the PyTorch model builds a standard DeePMD neighbor list for the
public forward path. When `use_compile` is enabled, the model additionally
uses a compact sparse-edge path for compiled training. Both paths share the
same descriptor and fitting definitions.

## Descriptor construction

For each frame, DPA4/SeZM first builds a local neighbor graph within cutoff
radius `rcut`. Each edge stores the displacement vector, smooth cutoff
weights, radial basis features, and the rotation between the global frame and
an edge-aligned local frame. These edge features are built once per forward
call and reused by all interaction blocks.

One DPA4/SeZM interaction block consists of the following operations:

1. Gather source-atom equivariant features on each edge.
1. Rotate them into the edge-local frame.
1. Apply SO(2)-equivariant convolution on the retained angular orders.
1. Rotate messages back to the global frame.
1. Aggregate messages at destination atoms with smooth envelope weights or
   attention weights.
1. Update atom features with an equivariant feed-forward block.

After the last block, DPA4/SeZM keeps the `l = 0` scalar channels:

```math
\mathcal{D}_i = \mathrm{Scalar}\left(\mathbf{h}_i^{(L)}\right),
```

where $\mathbf{h}_i^{(L)}$ is the final equivariant feature of atom `i`.

## Angular representation

DPA4/SeZM stores intermediate features as SO(3)-equivariant coefficients. A
feature block with maximum degree `lmax` contains all degrees
`l = 0, ..., lmax`, and each degree has `2l + 1` angular components.

The model reduces angular cost by working in a local frame on each edge. In
that frame, rotations around the edge axis become SO(2) operations. The SO(2)
convolution retains orders `|m| <= mmax`, or the per-block value specified by
`m_schedule`, while preserving the required equivariant transformation
behavior.

Two schedules control the angular width:

- `l_schedule` sets the SO(3) degree used by each block. A schedule such as
  `[3, 3, 2]` uses higher degrees in early blocks and truncates them in later
  blocks.
- `mmax` or `m_schedule` sets how many SO(2) orders are retained in the
  edge-local convolution.

The angular schedule is one of the primary accuracy-cost controls in
DPA4/SeZM. Larger angular spaces can represent more complex local chemistry,
but the cost grows quickly with `lmax`. For many systems, a non-increasing
`l_schedule` provides a practical compromise.

## Radial basis and smooth cutoff

Every edge uses a radial basis multiplied by a smooth envelope. The default
basis is Bessel-like, and a Gaussian basis is also available through
`basis_type`. The cutoff envelope is constructed so that its value and first
three derivatives vanish at `rcut`. This smoothness is important for
molecular dynamics because nonsmooth descriptor cutoffs would be inherited by
force derivatives.

DPA4/SeZM uses two envelope exponents through `env_exp`:

- the first exponent controls the radial basis envelope,
- the second exponent controls message-passing edge weights.

Increasing an exponent keeps the corresponding envelope closer to one for
more of the cutoff range before it drops near `rcut`.

## Attention and focus streams

DPA4/SeZM can aggregate edge messages either by envelope-weighted scatter or
by attention. When attention is enabled with `n_atten_head > 0`, the cutoff
envelope also participates in the softmax normalization. Edges near the
cutoff are therefore smoothly suppressed in both the numerator and the
denominator, avoiding nonsmooth contributions from the normalization term.

The SO(2) convolution can also use multiple focus streams through `n_focus`.
These streams process the same edge geometry in parallel and are then
combined through scalar weights. This design is not a sparse mixture of
experts: all focus streams are evaluated before soft reweighting. The
additional capacity helps the convolution distinguish different local
patterns while preserving equivariance.

## Grid nonlinearities

Several DPA4/SeZM branches can use sphere-grid or SO(3)-grid nonlinearities
inside the equivariant network. The most commonly used public switches are:

- `s2_activation`, which enables S2-grid nonlinearities for the SO(2) branch
  and/or the block-internal feed-forward branch.
- `ffn_so3_grid`, which uses an SO(3) Wigner-D grid in the block-internal
  feed-forward path.
- `lebedev_quadrature`, which selects packaged Lebedev quadrature rules for
  enabled S2-grid branches.
- `grid_mlp` and `grid_branch`, which select the polynomial point-wise MLP or
  the scalar-routed polynomial branch mixer for each grid path. Each is either
  a single value applied to every path or a list
  `[node_wise, message_node, ffn]`.

These options affect the expressiveness and cost of the equivariant
nonlinearity. The final `l = 0` output descriptor remains a scalar feature
tensor consumed by the fitting network.

## Environment-seeded initial features

When `use_env_seed` is enabled, DPA4/SeZM seeds the initial node state from
the local environment before the equivariant message-passing blocks. The
scalar seed uses a DeePMD-style local environment matrix with radial
information and normalized directions, then produces FiLM-like scale and
shift values for the first scalar features. When non-scalar degrees are
present, the same switch also enables the geometric initial embedding.

When `use_env_seed` is disabled, the initial node state contains only
atom-local scalar features before message passing. This keeps a one-block
model closed over the one-hop neighbor shell.

## Zone bridging and ZBL

DPA4/SeZM includes an optional short-range bridge for analytical repulsion.
The typical use case is ZBL:

```math
E_i = E_i^{\mathrm{DPA4/SeZM}} + E_i^{\mathrm{ZBL}}.
```

The purpose of zone bridging is to combine the analytical short-range
repulsion with the learned model while preventing uncontrolled learned forces
in the same protected region.

Zone bridging has two pieces:

1. Distances below `bridging_r_inner` are clamped before they enter the
   descriptor. Between `bridging_r_inner` and `bridging_r_outer`, a smooth
   polynomial transitions back to the true distance.
1. A source gate suppresses message propagation from atoms involved in frozen
   short-range pairs. This blocks multi-hop leakage, where a third atom could
   otherwise carry information about the frozen pair back into the learned
   energy.

This gives a controlled decomposition in the protected region:

```math
E_\mathrm{total}(r) = E_\mathrm{ZBL}(r) + E_\mathrm{model}(\tilde r),
```

where $r$ is the true distance and $\tilde r$ is the clamped distance seen by
the descriptor.

Enable zone bridging with:

```json
{
  "model": {
    "bridging_method": "zbl",
    "bridging_r_inner": 0.5,
    "bridging_r_outer": 0.8
  }
}
```

When ZBL bridging is enabled, set `training.training_data.min_pair_dist` to
the same value as `bridging_r_inner` so that frames with shorter atom pairs
are excluded from training. See `examples/water/dpa4/input-zbl.json` for a
complete ZBL input example.

## Fitting network

DPA4/SeZM uses the `dpa4_ener` energy fitting implementation. It is selected
automatically by the DPA4 model scaffold and maps scalar descriptors to atomic
energies.

The fitting network uses the same common keys as DeePMD's standard energy
fitting network:

- `neuron`
- `activation_function`
- `precision`
- `seed`
- `numb_fparam`
- `numb_aparam`

The hidden layers use GLU-style transformations. If `neuron` is `[0]`, the
fitting network uses a direct projection from descriptor channels to atomic
energy. This compact setting is useful for small examples and quick
validation tests.

For shared-fitting multitask training, DPA4/SeZM supports case embeddings.
With `case_film_embd: true`, the case vector modulates the fitting network
instead of being concatenated directly to the descriptor. This keeps the
descriptor case-independent while allowing the energy map to depend on the
task branch.

## Configuration

For a complete training input, see `examples/water/dpa4/input.json`. The
example uses a compact water setup with the DPA4 model type, SeZM descriptor
options, `dpa4_ener` fitting settings, and the standard conservative energy
loss. Its structure is closer to a DPA4-Neo-style compact configuration than
to the DPA4-Air pretrained configuration.

Common descriptor controls include:

- `sel` and `rcut` for the neighbor list.
- `channels`, `n_radial`, and `basis_type` for feature width and radial
  resolution.
- `lmax`, `l_schedule`, `mmax`, and `m_schedule` for angular resolution.
- `n_blocks`, `so2_layers`, and `ffn_blocks` for network depth.
- `n_focus` and `n_atten_head` for focus streams and attention aggregation.
- `use_env_seed`, `s2_activation`, `ffn_so3_grid`, and `message_node_so3` for
  the main geometric feature paths.
- `use_amp` and `precision` for training precision.

## Training modes

The recommended training objective is the standard conservative energy loss:

```json
{
  "loss": {
    "type": "ener"
  }
}
```

In this mode, the model predicts energies, and forces are computed by
autograd. See [training energy models](train-energy.md) for the general
energy-training workflow.

DPA4/SeZM also has an experimental direct-force denoising mode selected by:

```json
{
  "loss": {
    "type": "dens"
  }
}
```

Use `dens` only when the direct-force denoising head is required. It is not
the default training path. See `examples/water/dpa4/input_dens.json` for an
example input.

## Spin

DPA4/SeZM supports the DeePMD-kit spin convention in the PyTorch backend.
Keep the DPA4/SeZM type string and add the standard `model.spin` block:

```json
{
  "model": {
    "type": "dpa4",
    "type_map": [
      "Ni",
      "O"
    ],
    "spin": {
      "use_spin": [
        true,
        false
      ],
      "virtual_scale": [
        0.314
      ]
    },
    "descriptor": {
      "sel": 120,
      "rcut": 6.0
    }
  }
}
```

The spin path supports the conservative `ener_spin` loss. The direct-force
denoising mode is not used together with spin. See
[training spin energy models](train-energy-spin.md) for the common spin
training settings, and `examples/water/dpa4/input-spin.json` for a DPA4-style
input example.

## Performance and hardware recommendations

### bfloat16 automatic mixed precision

DPA4/SeZM supports automatic mixed precision (AMP) during training through the
descriptor option `use_amp`, whose default value is `true`. This option uses
bfloat16 (bf16) autocast for eligible CUDA operations. In typical DPA4/SeZM
workloads, bf16 AMP reduces memory usage and may improve throughput while
preserving fitted accuracy; no visible accuracy degradation is expected in
normal DPA4/SeZM training. Numerically sensitive geometric operations are kept
in promoted precision.

When the GPU provides native bf16 support, enabling `use_amp` is recommended:

```json
{
  "model": {
    "descriptor": {
      "use_amp": true
    }
  }
}
```

On GPUs without native bf16 support, explicitly set `use_amp` to `false` to
avoid runtime errors or additional conversion overhead:

```json
{
  "model": {
    "descriptor": {
      "use_amp": false
    }
  }
}
```

On NVIDIA hardware, native bf16 support starts with the Ampere generation,
including A100-series accelerators and RTX 30-series GPUs, and continues on
newer architectures.

### Experimental `torch.compile` path

DPA4/SeZM can train through an experimental `torch.compile` path:

```json
{
  "model": {
    "use_compile": true
  }
}
```

This path is useful for force-loss training, where differentiating the force
loss requires higher-order derivatives through the conservative
energy-gradient path. DPA4/SeZM traces this path before passing it to
Inductor.

This path is experimental and may expose PyTorch compiler issues. It currently
requires `torch==2.11`; other PyTorch versions are not supported for this
compiled DPA4/SeZM training path. On NVIDIA GPUs, CUDA must be >= 12.6. Apple
Silicon Macs are also supported. It has been tested with Python 3.13. If the
compiled path fails or produces unexpected behavior, please report the issue
with the PyTorch version, CUDA version, GPU model, and a minimal input file.

### Inference environment variables

DPA4/SeZM reads inference-related environment variables when the PyTorch model
is constructed. If these variables are already exported in the shell, they
take precedence over values written in the input file. Changing them after
model construction does not affect that model instance.

`DP_COMPILE_INFER` controls whether evaluation and inference forwards use the
DPA4/SeZM compile path:

```bash
export DP_COMPILE_INFER=1
```

Accepted true values are `1`, `true`, `yes`, and `on`; accepted false values
are `0`, `false`, `no`, and `off`. Enabling this path has the same PyTorch
version requirements as `model.use_compile`.

During training validation, the same setting can be requested in the input
file:

```json
{
  "validating": {
    "compiled_infer": true
  }
}
```

The trainer translates this option into `DP_COMPILE_INFER=1` before model
construction, unless the shell environment already defines `DP_COMPILE_INFER`.

`DP_TF32_INFER` controls the float32 matmul precision used by evaluation and
inference forwards on CUDA:

- `0`: use PyTorch `highest` precision. This is the default.
- `1`: use PyTorch `high` precision.
- `2`: use PyTorch `medium` precision.

During training validation, the input option
`validating.tf32_infer: true` is translated into `DP_TF32_INFER=1` before
model construction, again without overriding an explicitly exported
environment variable. Training forwards are controlled separately by
`model.enable_tf32`, independently of whether `model.use_compile` selects the
compiled or eager training path.

For molecular dynamics and other workflows that are sensitive to potential
energy surface smoothness, keep `DP_TF32_INFER=0`. Enabling TF32 inference may
leave energy and force MAE nearly unchanged while making the potential energy
surface less smooth. For less smoothness-sensitive evaluation or screening
workloads, `DP_TF32_INFER=1` or `2` may be useful for improving throughput.

`DP_TRITON_INFER` enables fused block-diagonal Triton kernels for the SO(2)
Wigner-D rotation. It applies to evaluation and inference on CUDA in eval mode
only and is disabled by default:

```bash
export DP_TRITON_INFER=1
```

The kernels operate on the block-diagonal (by degree `l`) structure of the
Wigner-D matrix and are numerically equivalent to the default dense rotation up
to floating-point rounding. They retain full float32 accumulation regardless of
`DP_TF32_INFER` and are therefore appropriate for smoothness-sensitive
workflows. They are compatible with the compile path (`DP_COMPILE_INFER=1`) and
reduce both latency and peak memory.

When exporting DPA4/SeZM to `.pt2`, set inference environment variables before
running `dp --pt freeze`. The exported package is an AOTInductor artifact, so
graph-level choices and compiler precision settings are fixed during export and
are not re-evaluated when the `.pt2` file is later loaded by ASE or LAMMPS.
In particular, `DP_TRITON_INFER` selects the SO(2) rotation branch that is
captured into the exported graph, and `DP_TF32_INFER` should be set before
export if TF32 inference is desired. `DP_ACT_INFER` is not a runtime control for
`.pt2` inference: activation checkpointing is a Python/autograd memory-saving
strategy, while `.pt2` inference runs a forward-only AOTI package whose force
and virial computations have already been lowered into the exported graph.

### Hardware selection

DPA4/SeZM is designed for fp32 training and inference. Hardware selection
should therefore be based primarily on fp32 throughput rather than fp64
throughput. In contrast to workloads dominated by double-precision linear
algebra, DPA4/SeZM does not require GPUs with especially strong fp64
performance.

For practical training, prefer GPUs that combine high fp32 FLOPS with native
bf16 support. Native bf16 enables the recommended AMP path, lowering memory
usage and often improving throughput. Because AMP can substantially reduce the
activation memory footprint, DPA4/SeZM training usually does not require
unusually large-memory GPUs once the target system and batch size fit. In that
regime, native bf16 support and fp32 FLOPS are usually more important
selection criteria than maximum device memory.

## LoRA fine-tuning

DPA4/SeZM supports LoRA adapters on its SO(3) and SO(2) linear layers. This
mode is intended for single-task fine-tuning. A typical input block is:

```json
{
  "model": {
    "type": "dpa4",
    "lora": {
      "rank": 16,
      "alpha": 16.0
    }
  }
}
```

Then fine-tune from a checkpoint:

```bash
dp --pt train lora_ft.json --finetune pretrained.pt
```

See `examples/water/dpa4/lora_ft.json` for a complete example.

## Export

DPA4/SeZM checkpoints use the PyTorch `.pt2` export path. Run the standard
freeze command:

```bash
dp --pt freeze -c model.ckpt.pt -o frozen_model
```

The PyTorch backend detects DPA4/SeZM and writes `frozen_model.pt2`. Use this
`.pt2` file with LAMMPS:

```lammps
pair_style deepmd frozen_model.pt2
pair_coeff * * O H
```

The ordinary TorchScript freeze path is not used for DPA4/SeZM checkpoints.
A small LAMMPS example is in `examples/water/dpa4/lmp/`.

## Data format

DPA4/SeZM uses the [standard DeePMD-kit data format](../data/system.md). Keep
the `type_map` order consistent across the dataset, input file, and any
downstream `pair_coeff` mapping.

## Limitations

- DPA4/SeZM is currently implemented for the PyTorch backend.
- Model compression is not supported.
- Export uses `.pt2`; the ordinary TorchScript freeze path is not used for
  DPA4/SeZM checkpoints.

## Citation

If you use DPA4/SeZM, please cite the [DPA4 paper](https://arxiv.org/abs/2606.02419):

```bibtex
@article{li2026dpa4,
  title = {{DPA4}: Pushing the Accuracy-Cost Frontier of Interatomic
           Potentials with {EMFA} {SO(2)} Convolution},
  author = {Li, Tiancheng and Li, Wentao and Peng, Anyang and Xue, Jianming
            and Zhang, Linfeng and Zhang, Duo and Wang, Han},
  journal = {arXiv preprint arXiv:2606.02419},
  year = {2026},
  eprint = {2606.02419},
  archivePrefix = {arXiv},
  primaryClass = {physics.chem-ph},
  doi = {10.48550/arXiv.2606.02419},
  url = {https://arxiv.org/abs/2606.02419}
}
```
