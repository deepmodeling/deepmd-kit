# Descriptor DPA4 {{ pytorch_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}
:::

DPA4/SeZM is the DPA-series implementation of SeZM, the Smooth Equivariant
Zone-bridging Model. The recommended input type is `dpa4`; `DPA4`, `SeZM`,
and `sezm` are accepted aliases. For new input files, set
`model.type: "dpa4"` and `descriptor.type: "dpa4"`.

Training example: `examples/water/dpa4/input.json`.

## Overview

DPA4/SeZM is an SO(3)-equivariant message-passing model for conservative
interatomic potentials. It predicts atomic energies and obtains forces
and virials by differentiating the energy, following the same
conservative formulation used by standard DeePMD energy models:

```math
\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}.
```

The model retains vector and higher-order angular information during
descriptor construction. Only the final descriptor passed to the fitting
network is scalar. This separates the geometric representation from the
energy mapping: equivariant layers encode local geometry, and the fitting
network maps the resulting scalar features to atomic energies.

## Descriptor construction

For each frame, DPA4/SeZM first builds a local neighbor graph within cutoff
radius `rcut`. Each edge stores the displacement vector, a smooth cutoff
weight, radial basis features, and a rotation from the global coordinate
frame to an edge-aligned local frame.

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

DPA4/SeZM avoids the most expensive part of a full SO(3) operation by working
in a local frame on each edge. In that frame, rotations around the edge
axis become SO(2) operations. The descriptor retains only orders
`|m| <= mmax` inside the SO(2) convolution, reducing angular cost while
preserving the required rotation behavior.

Two schedules control the angular width:

- `l_schedule` sets the SO(3) degree used by each block. A schedule such as
  `[3, 3, 2]` uses higher degrees in early blocks and truncates them in
  later blocks.
- `mmax` or `m_schedule` sets how many SO(2) orders are retained in the
  edge-local convolution.

The angular schedule is one of the primary accuracy-cost controls in
DPA4/SeZM. Larger angular spaces can represent more complex local chemistry,
but the cost grows quickly with `lmax`. For many systems, a
non-increasing `l_schedule` provides a practical compromise.

## Radial basis and smooth cutoff

Every edge uses a radial basis multiplied by a smooth envelope. The
default basis is Bessel-like, and a Gaussian basis is also available. The
cutoff envelope is constructed so that its value and first three
derivatives vanish at `rcut`. This smoothness is important for molecular
dynamics because nonsmooth descriptor cutoffs would be inherited by force
derivatives.

DPA4/SeZM uses two envelope exponents through `env_exp`:

- the first exponent controls the radial basis envelope,
- the second controls message-passing edge weights.

Increasing the exponent keeps the envelope closer to one for more of the
cutoff range before it drops near `rcut`.

## Attention and focus streams

DPA4/SeZM can aggregate edge messages either by envelope-weighted scatter or by
attention. When attention is enabled, the cutoff envelope also
participates in the softmax normalization. Edges near the cutoff are therefore
smoothly suppressed in both the numerator and the denominator, avoiding
nonsmooth contributions from the normalization term.

The SO(2) convolution can also use multiple focus streams. These streams
process the same edge geometry in parallel and are then combined through
scalar weights. This design is not a sparse mixture of experts: all focus
streams are evaluated before soft reweighting. The additional capacity
helps the convolution distinguish different local patterns while
preserving equivariance.

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

DPA4/SeZM includes an optional short-range bridge for analytical repulsion. The
typical use case is ZBL:

```math
E_i = E_i^{\mathrm{DPA4/SeZM}} + E_i^{\mathrm{ZBL}}.
```

The purpose of zone bridging is to combine the analytical short-range
repulsion with the learned model while preventing uncontrolled learned
forces in the same protected region.

Zone bridging has two pieces:

1. Distances below `bridging_r_inner` are clamped before they enter the
   descriptor. Between `bridging_r_inner` and `bridging_r_outer`, a smooth
   polynomial transitions back to the true distance.
1. A source gate suppresses message propagation from atoms involved in
   frozen short-range pairs. This blocks multi-hop leakage, where a third
   atom could otherwise carry information about the frozen pair back into
   the learned energy.

This gives a controlled decomposition in the protected region:

```math
E_\mathrm{total}(r) = E_\mathrm{ZBL}(r) + E_\mathrm{model}(\tilde r),
```

where $r$ is the true distance and $\tilde r$ is the clamped distance seen
by the descriptor.

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

DPA4/SeZM uses `dpa4_ener` as the energy fitting network name in input files.
It is a GLU-based fitting network that maps scalar descriptors to atomic
energies.

The fitting network uses the same common keys as DeePMD's standard energy
fitting network:

- `neuron`
- `activation_function`
- `precision`
- `seed`
- `numb_fparam`
- `numb_aparam`

The hidden layers use GLU-style transformations. If `neuron` is `[0]`,
the fitting network uses a direct projection from descriptor channels to
atomic energy. This compact setting is useful for small examples and quick
validation tests.

For shared-fitting multitask training, DPA4/SeZM supports case embeddings. With
`case_film_embd: true`, the case vector modulates the fitting network
instead of being concatenated directly to the descriptor. This keeps the
descriptor case-independent while allowing the energy map to depend on the
task branch.

## Configuration

For a complete training input, see `examples/water/dpa4/input.json`. The
example keeps the water dataset paths local to the repository while using a
parameter set close to the pretrained DPA4-Air model.

## Training modes

The recommended training objective is the standard conservative energy
loss:

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

Use `dens` only when the direct-force denoising head is required. It is
not the default training path.

## Spin

DPA4/SeZM supports the DeePMD-kit spin convention in the PyTorch backend. Keep
the DPA4/SeZM type string and add the standard `model.spin` block:

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
      "type": "dpa4",
      "sel": 120,
      "rcut": 6.0
    }
  }
}
```

The spin path supports the conservative `ener_spin` loss. The direct-force
denoising mode is not used together with spin. See
[training spin energy models](train-energy-spin.md) for the common spin
training settings.

## Performance and hardware recommendations

### bfloat16 automatic mixed precision

DPA4/SeZM supports automatic mixed precision (AMP) during training through the
descriptor option `use_amp`. This option uses bfloat16 (bf16) autocast for
eligible CUDA operations. In typical DPA4/SeZM workloads, bf16 AMP reduces
memory usage and may improve throughput while preserving fitted accuracy, but
the final accuracy should be validated for the target system. Numerically
sensitive geometric operations are kept in promoted precision.

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

On GPUs without native bf16 support, leave `use_amp` disabled to avoid runtime
errors or additional conversion overhead. On NVIDIA hardware, native bf16
support starts with the Ampere generation, including A100-series accelerators
and RTX 30-series GPUs, and continues on newer architectures.

### Experimental `torch.compile` path

DPA4/SeZM can train through an experimental `torch.compile` path:

```json
{
  "model": {
    "use_compile": true
  }
}
```

This path is useful for force-loss training because the model first
differentiates energy to obtain forces and then differentiates the force
loss with respect to model parameters. The training graph therefore contains
higher-order autograd operations, including mixed derivatives induced by
differentiating force losses with respect to model parameters. DPA4/SeZM
traces this graph before passing it to Inductor.

This path is experimental and may expose PyTorch compiler issues. It currently
requires `torch==2.11`; other PyTorch versions are not supported for this
compiled DPA4/SeZM training path. On NVIDIA GPUs, CUDA must be >= 12.6. Apple
Silicon Macs are also supported. It has been tested with Python 3.13. If the
compiled path fails or produces unexpected behavior, please report the issue
with the PyTorch version, CUDA version, GPU model, and a minimal input file.

For evaluation-time compile during validation, set:

```json
{
  "validating": {
    "compiled_infer": true
  }
}
```

You can also set `DP_COMPILE_INFER=1` in the environment before training.

### Hardware selection

DPA4/SeZM is designed for fp32 training and inference. Hardware selection
should therefore be based primarily on fp32 throughput rather than fp64
throughput. In contrast to workloads dominated by double-precision linear
algebra, DPA4/SeZM does not require GPUs with especially strong fp64 performance.

For practical training, prefer GPUs that combine high fp32 FLOPS with native
bf16 support. Native bf16 enables the recommended AMP path, lowering memory
usage and often improving throughput. Because AMP can substantially reduce the
activation memory footprint, DPA4/SeZM training usually does not require
unusually large-memory GPUs once the target system and batch size fit. In that
regime, native bf16 support and fp32 FLOPS are usually more important selection
criteria than maximum device memory.

## LoRA fine-tuning

DPA4/SeZM supports LoRA adapters on its SO(3) and SO(2) linear layers. A typical
input block is:

```json
{
  "model": {
    "type": "dpa4",
    "descriptor": {
      "type": "dpa4"
    },
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
file with LAMMPS:

```lammps
pair_style deepmd frozen_model.pt2
pair_coeff * * O H
```

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
