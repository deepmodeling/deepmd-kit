# Descriptor DPA4 {{ pytorch_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}
:::

DPA4 is the DeePMD-kit implementation of the SeZM (Smooth Equivariant
Zone-bridging Model) architecture: an SO(3)-equivariant message-passing model
for conservative interatomic potentials. The aliases `DPA4`, `SeZM`, and
`sezm` all select the same implementation.

`model.type: "dpa4"` is a convenience scaffold that selects the SeZM descriptor
and defaults to the `dpa4_ener` energy fitting network, so `descriptor.type` and
`fitting_net.type` may be omitted for energy training. A new energy input then
needs only the model type, `type_map`, and a few descriptor options.

Reference: [DPA4 paper](https://arxiv.org/abs/2606.02419).

## Quick start

DPA4 is a PyTorch-only model. Train it with the standard `dp --pt` workflow:

```bash
cd examples/water/dpa4
dp --pt train input.json
```

`examples/water/dpa4/input.json` is a complete, compact training input you can
copy and adapt. See [training energy models](train-energy.md) for the general
training workflow shared by all energy models.

## Overview

DPA4/SeZM predicts atomic energies and obtains forces and virials by
differentiating the energy, the same conservative formulation used by standard
DeePMD energy models:

```math
\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}.
```

Internally the model keeps vector and higher-order angular (SO(3)-equivariant)
information while building the descriptor, and only the final descriptor sent
to the fitting network is a scalar. This separates geometric representation
(equivariant message-passing layers that encode local environments) from
energy prediction (the fitting network that maps scalar features to atomic
energies). The architecture targets a favorable accuracy–cost trade-off; if you
want the design details, see [Architecture details](#architecture-details) at
the end of this page.

## Configuration

### Minimal input

A minimal DPA4 model only needs the model type, `type_map`, and the neighbor
list (`sel`, `rcut`). Every other option uses its documented default.

```json
{
  "model": {
    "type": "dpa4",
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "rcut": 6.0
    }
  }
}
```

DPA4/SeZM defaults to `float32`
({ref}`precision <model[dpa4]/descriptor[dpa4]/precision>`); double precision is
unnecessary and not recommended (see [Hardware selection](#hardware-selection)).

:::{note}
{ref}`sel <model[dpa4]/descriptor[dpa4]/sel>` behaves differently from classic
descriptors. On the conservative **energy** path it is only an initial
neighbor-search capacity that grows on demand, so it never truncates the
neighbor list and you do not need to size it to the true maximum neighbor count.
The native spin scheme shares this energy path, so it grows on demand too; only
the denoising (`dens`) path and the `deepspin` spin scheme cap the list at
`sum(sel)`. You can also set `sel` to `auto` or `auto:factor` to size it from
the training data.
:::

### Main options

Every descriptor option, with its default and full description, is listed in
the {ref}`argument reference <model[dpa4]/descriptor[dpa4]>`. The options worth
tuning first group into four accuracy–cost levers:

- **Angular width** — the primary control. {ref}`lmax <model[dpa4]/descriptor[dpa4]/lmax>`
  with the per-block pyramid {ref}`l_schedule <model[dpa4]/descriptor[dpa4]/l_schedule>`
  (which overrides `lmax` and `n_blocks`), and the SO(2) order
  {ref}`mmax <model[dpa4]/descriptor[dpa4]/mmax>` /
  {ref}`m_schedule <model[dpa4]/descriptor[dpa4]/m_schedule>`. Cost grows
  quickly with `lmax`; a non-increasing `l_schedule` is often a good compromise.
- **Depth** — {ref}`n_blocks <model[dpa4]/descriptor[dpa4]/n_blocks>`,
  {ref}`so2_layers <model[dpa4]/descriptor[dpa4]/so2_layers>`,
  {ref}`ffn_blocks <model[dpa4]/descriptor[dpa4]/ffn_blocks>`.
- **Width** — {ref}`channels <model[dpa4]/descriptor[dpa4]/channels>`,
  {ref}`n_radial <model[dpa4]/descriptor[dpa4]/n_radial>`.
- **Aggregation** — {ref}`n_focus <model[dpa4]/descriptor[dpa4]/n_focus>`,
  {ref}`n_atten_head <model[dpa4]/descriptor[dpa4]/n_atten_head>` (`0` falls
  back to a plain envelope-weighted scatter).

The neighbor list is set by {ref}`rcut <model[dpa4]/descriptor[dpa4]/rcut>` and
{ref}`sel <model[dpa4]/descriptor[dpa4]/sel>`, the initial node features by
{ref}`use_env_seed <model[dpa4]/descriptor[dpa4]/use_env_seed>`, and the energy
head by the fitting `neuron` list (`[0]` is a direct projection). The quickest
starting point is to copy `examples/water/dpa4/input.json` and adjust the
levers above.

## Training

### Energy training (default)

The recommended objective is the standard conservative energy loss. The model
predicts energies and forces are obtained by autograd:

```json
{
  "loss": {
    "type": "ener"
  }
}
```

See [training energy models](train-energy.md) for the general workflow.

### Property training

DPA4/SeZM can also train invariant structure properties by selecting the
standard property fitting network. Set `fitting_net.type` to `property`, provide
the property name used by the data file, and use the property loss:

```json
{
  "model": {
    "type": "dpa4",
    "fitting_net": {
      "type": "property",
      "property_name": "band_prop",
      "task_dim": 3,
      "intensive": true
    }
  },
  "loss": {
    "type": "property"
  }
}
```

The property label follows the usual DeePMD property-data convention: each
system provides `band_prop.npy` when `property_name` is `band_prop`. Property
fitting is not a water task, so the complete DPA4/SeZM property input lives with
the property dataset: see `examples/property/train/input_dpa4.json`, which trains
on the QM9 property subset in `examples/property/data`.

### Direct-force denoising (`dens`, experimental)

DPA4/SeZM has an experimental direct-force denoising head:

```json
{
  "loss": {
    "type": "dens"
  }
}
```

Use `dens` only when the direct-force denoising head is required; it is not the
default training path. See `examples/water/dpa4/input_dens.json` for an example.

### Spin

DPA4/SeZM supports the DeePMD-kit spin convention through the standard
`model.spin` block. Two schemes are available, selected by `model.spin.scheme`:

- `native` — the per-atom spin vector enters the descriptor as an
  equivariant feature, and the magnetic force is the spin gradient of the
  energy. No virtual atoms are introduced, so the neighbor list and type map
  keep their real-system sizes.
- `deepspin` (default) — the classical DeepSpin representation, in which each
  magnetic atom is paired with a virtual atom displaced along its spin. It is
  the default, so a `model.spin` block without an explicit `scheme` reproduces
  the classical behaviour shared with every non-SeZM spin model.

Both schemes train against the conservative `ener_spin` loss, share the same
`spin` / `force_mag` data convention, and are not combined with the `dens` mode.
Because the dataset and loss are identical, switching `scheme` does not require
any change to the data or the loss block. Complete inputs are in
`examples/spin/dpa4/`: `input.json` for the native scheme and
`input-deepspin.json` for the deepspin scheme. See
[training spin energy models](train-energy-spin.md) for the general workflow.

`use_spin` accepts a per-type boolean list or, to avoid enumerating a large
`type_map`, the list of magnetic species as type indices or element symbols
(e.g. `"use_spin": ["Fe"]`), expanded against `type_map`. For the native scheme,
`model.spin.allow_missing_label` additionally admits training systems that lack a
`spin` data file by filling their spin with zeros; since a zero spin reduces the
native descriptor to its spin-free form, a model can be trained on a mixture of
spin-labelled and spin-free systems, or pretrained on spin-free data and
fine-tuned on spin-labelled data without changing its type map or parameters.

#### Native scheme

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
      "scheme": "native"
    },
    "descriptor": {
      "rcut": 6.0
    }
  }
}
```

`use_spin` marks which atom types carry spin. Both the conservative force and
the magnetic force come from a single energy gradient:

```math
\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i},
\qquad
\mathbf{F}^{\mathrm{mag}}_i = -\frac{\partial E}{\partial \mathbf{s}_i},
```

where $\mathbf{s}_i$ is the input spin vector. The model does not rescale the
spin internally, so $\mathbf{s}_i$ keeps the dataset's `spin` convention and the
magnetic force is reported in the matching units of `force_mag` -- there is no
`virtual_scale` factor in this scheme. The magnetic force is reported on the
magnetic atom types only, matching the `force_mag` label. The native scheme
relies on the descriptor's angular degrees to represent the spin direction, so
it requires `lmax >= 1` (the default).

#### DeepSpin virtual-atom scheme (default)

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
      ],
      "scheme": "deepspin"
    },
    "descriptor": {
      "sel": 120,
      "rcut": 6.0
    }
  }
}
```

The `deepspin` scheme augments each magnetic atom with a virtual atom at
`coord + spin * virtual_scale`, which doubles the internal neighbor capacity and
type map. `virtual_scale` is required by this scheme and ignored by the native
scheme.

### Multi-task / shared fitting

DPA4/SeZM supports shared-fitting multitask training. With
`case_film_embd: true`, the case vector modulates the fitting network instead
of being concatenated to the descriptor, which keeps the descriptor
case-independent while letting the energy map depend on the task branch. See
[multi-task training](../train/multi-task-training.md) for the workflow and
`examples/water/dpa4/input_multitask.json` for an example.

### LoRA fine-tuning

DPA4/SeZM supports LoRA adapters on its SO(3) and SO(2) linear layers, intended
for single-task fine-tuning:

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

Fine-tune from a checkpoint:

```bash
dp --pt train lora_ft.json --finetune pretrained.pt
```

Best checkpoints fold the LoRA deltas back into the base weights, producing
plain DPA4/SeZM checkpoints suitable for deployment. See
`examples/water/dpa4/lora_ft.json`.

## Zone bridging (ZBL)

DPA4/SeZM can add an analytical short-range repulsion (typically ZBL) to the
learned energy in a protected region:

```math
E_i = E_i^{\mathrm{DPA4/SeZM}} + E_i^{\mathrm{ZBL}}.
```

Below `bridging_r_inner` the distance seen by the descriptor is clamped, with a
smooth transition back to the true distance up to `bridging_r_outer`; a source
gate additionally blocks the learned model from leaking information about the
frozen short-range pairs. Enable it with:

```json
{
  "model": {
    "bridging_method": "zbl",
    "bridging_r_inner": 0.5,
    "bridging_r_outer": 0.8
  }
}
```

When ZBL bridging is enabled, set `training.training_data.min_pair_dist` to the
same value as `bridging_r_inner` so frames with shorter atom pairs are excluded
from training. See `examples/water/dpa4/input-zbl.json` for a complete example.

## Performance and precision

### Training-time settings

Three options control training precision and the compiled path:

- {ref}`use_amp <model[dpa4]/descriptor[dpa4]/use_amp>` — bf16 automatic mixed
  precision on CUDA. Reduces memory and often improves throughput with no
  expected accuracy loss. Recommended on GPUs with native bf16 (NVIDIA Ampere
  and newer, e.g. A100 / RTX 30-series); set it off on GPUs without native bf16
  to avoid runtime errors and conversion overhead.
- {ref}`enable_tf32 <model[dpa4]/enable_tf32>` — TF32 matmul precision for CUDA
  **training** forwards (independent of `use_compile`, and separate from the
  inference TF32 control below).
- {ref}`use_compile <model[dpa4]/use_compile>` — experimental `torch.compile`
  training path. Useful for force-loss training (higher-order derivatives
  through the energy gradient) and can speed training markedly on supported
  setups.

### Inference and deployment settings

Inference behavior is controlled by environment variables, each with an
equivalent input-file option used during training validation:

| Environment variable | Input-file option           | Default       | Effect                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| -------------------- | --------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DP_COMPILE_INFER`   | `validating.compiled_infer` | off           | Use the compile path for evaluation/inference. Same `torch==2.11` / CUDA ≥ 12.6 requirements as `model.use_compile`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `DP_TF32_INFER`      | `validating.tf32_infer`     | `0` (highest) | float32 matmul precision for inference: `0` highest, `1` high, `2` medium. Higher values improve throughput but make the potential energy surface less smooth.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `DP_AMP_INFER`       | `validating.amp_infer`      | off           | bf16 autocast inside the descriptor interaction blocks for inference when `descriptor.use_amp=true`. Usually keeps aggregate MAE similar but can make the potential energy surface less smooth.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `DP_TRITON_INFER`    | —                           | `0`           | Triton inference kernel level `0`-`3` (CUDA eval only, compatible with `DP_COMPILE_INFER`). `1`: universal fused kernels, numerically equivalent to the dense path with full float32 accumulation. `2`: adds the table-configured fused SO(2) value path and edge-block backward kernels (still exact float32). `3`: additionally runs the SO(2) mixing stack on fp16 tensor cores with split compensation — roughly float32-level accuracy (maximum force deviation about 4e-6 eV/Å on a 4-thousand-atom system) at a substantial speedup; only shapes validated by the tuning sweep are affected. Levels 2 and 3 read launch tables tuned per GPU model (H20 ships built in); on other GPUs the kernels fall back to conservative configurations, and `dp --pt freeze` tunes the missing entries automatically on the local GPU before exporting (a one-off sweep of a few minutes, baked into the `.pt2`). |

Accepted boolean values for the other switches are `1`/`true`/`yes`/`on` and
`0`/`false`/`no`/`off`; `DP_TRITON_INFER` accepts only the numeric levels.
Shell exports take precedence over the input-file options and over values
written in the input; they are read when the model is constructed and changing
them afterward has no effect.

For molecular dynamics and other workflows sensitive to the smoothness of the
potential energy surface, keep `DP_TF32_INFER=0` and `DP_AMP_INFER=0`.
`DP_AMP_INFER` can coexist with `DP_TF32_INFER`, but bf16 autocast dominates
the eligible operations it covers, so TF32 usually adds little extra throughput
there. `DP_TRITON_INFER` levels `1` and `2` retain full float32 accumulation
regardless of the precision policy and are therefore safe for those workflows;
level `3` perturbs forces at the 2^-22 rounding scale (three orders of
magnitude finer than TF32) and is the recommended fast setting once validated
for the target system.

:::{important}
Set these variables **before** running `dp --pt freeze`. The exported `.pt2` is
an AOTInductor artifact, so the SO(2) rotation branch (`DP_TRITON_INFER`), the
matmul precision (`DP_TF32_INFER`), and inference AMP (`DP_AMP_INFER`) are
captured into the graph at export time and are **not** re-evaluated when the
`.pt2` is later loaded by ASE or LAMMPS. A frozen `.pt2` runs a forward-only
package, so training-time memory-saving switches do not apply to it.
:::

### Hardware selection

DPA4/SeZM is designed for fp32 training and inference, so prefer GPUs with high
fp32 throughput and native bf16 support rather than strong fp64 performance.
Because bf16 AMP substantially reduces the activation memory footprint, very
large device memory is usually less important than fp32 FLOPS and bf16 support
once the target system and batch size fit.

## Export and running in LAMMPS

### Freeze to `.pt2`

DPA4/SeZM checkpoints use the PyTorch `.pt2` (AOTInductor) export path; the
ordinary TorchScript freeze path is not used. Run the standard freeze command:

```bash
dp --pt freeze -c model.ckpt.pt -o frozen_model
```

The PyTorch backend detects DPA4/SeZM and writes `frozen_model.pt2`.

### Single GPU

Use the frozen `.pt2` with the `deepmd` pair style. A small example is in
`examples/water/dpa4/lmp/`.

```lammps
pair_style deepmd frozen_model.pt2
pair_coeff * * O H
```

### Multi-GPU (MPI) inference

:::{important}
**Multi-rank support depends on the freeze `--lower-kind`.** DPA4/SeZM reads
ghost-neighbour features at every interaction block. A graph-kind `.pt2`
(`dp --pt_expt freeze --lower-kind graph`, see [Graph-native inference route
(pt_expt)](#graph-native-inference-route-pt_expt) below) embeds a with-comm
AOTInductor artifact and supports multi-rank LAMMPS out of the box. A
dense-kind `.pt2` (the default `dp --pt freeze` path used elsewhere on this
page) has no cross-rank ghost-feature exchange implemented for its adapter,
so a multi-rank LAMMPS run on a dense `.pt2` raises an error at the first
force evaluation instead of silently running without the exchange; use a
single MPI rank for dense-frozen DPA4/SeZM. Models with spin, charge/spin
conditioning, or ZBL zone bridging are not graph-eligible (see below) and so
stay dense-only and single-rank regardless of `--lower-kind`. The remainder
of this subsection describes the multi-GPU launch recipe, which applies to
both the supported graph-kind and the single-rank dense-kind archives.
:::

The exported `.pt2` runs across multiple GPUs in LAMMPS using MPI domain
decomposition, with the same `.pt2` file serving both single- and multi-GPU
runs and no extra freeze options needed beyond `--lower-kind graph`. The
launch recipe:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 lmp -in in.lammps
```

Each MPI rank uses at most one GPU, so `CUDA_VISIBLE_DEVICES` must list every
GPU the run may use. DPA4/SeZM exchanges neighbor information across the
domain boundary, so the LAMMPS atom map must be enabled:

```lammps
atom_modify map yes
pair_style deepmd frozen_model.pt2
pair_coeff * * O H
```

Two settings improve multi-GPU runs:

- For fast GPU-to-GPU exchange, build the C++ interface against a
  [CUDA-Aware MPI](https://developer.nvidia.com/mpi-solutions-gpus) library;
  otherwise the cross-rank exchange falls back to a slower CPU path.
- Use a non-zero neighbor skin, e.g. `neighbor 2.0 bin`, to keep per-step GPU
  memory stable. A zero skin rebuilds the neighbor list every step and can
  substantially increase memory use.

## Graph-native inference route (pt_expt)

In the pt_expt backend, a plain-energy DPA4/SeZM descriptor (no native or
`deepspin` spin, no frame-level charge/spin conditioning `add_chg_spin_ebd`,
and no ZBL zone bridging) can be frozen through a NeighborGraph-native
inference path instead of the legacy dense neighbor-list path:

```bash
dp --pt_expt freeze -o model.pt2 --lower-kind graph
```

As with DPA-1's and DPA-2's graph paths (see [Difference among different
backends](train-se-atten.md#difference-among-different-backends) and the
"Graph-native inference route (pt_expt)" section of [DPA-2's
documentation](dpa2.md)), the graph route is a sel-free, carry-all builder:
it considers every neighbor within `rcut` rather than the capacity fixed by
`sel`. For the default conservative-energy path, `sel` is already only an
initial search capacity that grows on demand (see the note in [Minimal
input](#minimal-input)), so graph and dense results agree there down to
floating-point noise. Where `sel` does act as a hard cap -- the `dens`
denoising path, or the `deepspin` virtual-atom spin scheme, both of which
size the neighbor list to `sum(sel)` -- the graph route's larger neighbor set
can diverge from the dense route, the same deliberate divergence documented
for DPA-1/DPA-2.

A DPA4/SeZM descriptor configured with spin, charge/spin conditioning, or ZBL
zone bridging is not graph-eligible and always runs the dense route
regardless of `--lower-kind`; `--lower-kind graph` on such a model raises an
error at freeze time instead of exporting a silently-dense-only artifact.

Unlike the dense route (see [Multi-GPU (MPI)
inference](#multi-gpu-mpi-inference) above), a graph-frozen `.pt2` embeds a
with-comm AOTInductor artifact and supports multi-rank LAMMPS: each block's
cross-rank ghost-feature exchange runs through the `border_op` MPI path once
per interaction block, the same mechanism used by DPA-2's graph route (see
the "Graph-native inference route (pt_expt)" section of [DPA-2's
documentation](dpa2.md)). As on DPA-2, multi-rank inference on the graph
route requires every MPI rank to own or ghost at least one atom; a rank with
zero atoms in both categories aborts the run collectively rather than
silently desynchronizing the per-block exchange. Pick a domain decomposition
that keeps every rank non-empty, or use the dense route, which has no such
restriction (but is single-rank only, as noted above).

## Embedding extraction

A trained DPA4/SeZM model can export learned representations for downstream
analysis with `dp embed`. A single forward pass (no force or virial) produces
three embeddings per system:

- `descriptor`: per-atom local-environment representation, shape
  `(nframes, natoms, dim_descriptor)`.
- `atomic_feature`: per-atom activation after the last fitting hidden layer,
  shape `(nframes, natoms, dim_hidden)`.
- `structural_feature`: whole-structure summary obtained by summing
  `atomic_feature` over atoms, shape `(nframes, dim_hidden)`.

```bash
dp embed -m model.ckpt.pt -s /path/to/system -o embedding.hdf5
```

The results are written to a single HDF5 file in which each system is a group
holding the three float32 datasets above:

```python
import h5py

with h5py.File("embedding.hdf5", "r") as f:
    type_map = f.attrs["type_map"]
    group = f[next(iter(f.keys()))]
    descriptor = group["descriptor"][:]
    atomic_feature = group["atomic_feature"][:]
    structural_feature = group["structural_feature"][:]
```

This command operates on the training checkpoint (`.pt`), not the frozen
`.pt2`, and honors both `DP_COMPILE_INFER` and `DP_TRITON_INFER`. See
[model embeddings](../inference/embedding.md) for the full description.

## Data format

DPA4/SeZM uses the [standard DeePMD-kit data format](../data/system.md) and
also supports the [mixed-type data format](../data/system.md#mixed-type), which
is convenient for datasets that mix many element combinations (and is the usual
choice for multi-task training). Keep the `type_map` order consistent across the
dataset, the input file, and any downstream `pair_coeff` mapping.

## Architecture details

Optional background on how the descriptor works, linking each part to the
options that control it. Skip it unless you are tuning those options.

### Equivariant representation and the l = 0 read-out

DPA4/SeZM stores intermediate features as SO(3)-equivariant coefficients: a
feature block of maximum degree `lmax` holds all degrees `l = 0, …, lmax`, each
with `2l + 1` angular components (controlled by `lmax` / `l_schedule`).

For each frame the model first builds a local neighbor graph within `rcut`.
Each edge stores the displacement vector, smooth cutoff weights, radial basis
features, and the rotation between the global frame and an edge-aligned local
frame; these are built once and reused by all blocks. One interaction block
then (1) gathers source-atom features on each edge, (2) rotates them into the
edge-local frame, (3) applies an SO(2)-equivariant convolution on the retained
angular orders, (4) rotates the messages back, (5) aggregates them at
destination atoms with envelope or attention weights, and (6) updates atom
features with an equivariant feed-forward block.

Working in the edge-local frame turns rotations around the edge axis into SO(2)
operations, so the cost scales with `lmax` instead of cubically. The SO(2)
convolution retains orders `|m| ≤ mmax` (or `m_schedule`). After the last block,
only the `l = 0` scalar channels are read out and passed to the fitting network:

```math
\mathcal{D}_i = \mathrm{Scalar}\left(\mathbf{h}_i^{(L)}\right).
```

### Radial basis and smooth cutoff

Every edge uses a radial basis (`basis_type`, with `n_radial` functions)
multiplied by a smooth envelope whose value and first three derivatives vanish
at `rcut`. This smoothness matters for MD because nonsmooth descriptor cutoffs
would be inherited by the force derivatives. The two `env_exp` exponents control
the radial-basis envelope and the message-passing edge weights respectively;
larger values keep an envelope closer to one for more of the cutoff range.

### Attention and focus streams

Messages are aggregated either by envelope-weighted scatter or by attention
(`n_atten_head > 0`). With attention, the cutoff envelope participates in the
softmax normalization, so edges near `rcut` are smoothly suppressed in both the
numerator and the denominator. The SO(2) convolution can also use multiple
`n_focus` streams that process the same edge geometry in parallel and are
combined by scalar weights, adding capacity while preserving equivariance.

### Grid nonlinearities

Several branches can use sphere-grid (S2) or SO(3) Wigner-D grid
nonlinearities. The main switches are `s2_activation` (S2-grid nonlinearity for
the SO(2) and/or FFN branch), `ffn_so3_grid` (SO(3) grid in the block-internal
FFN), `lebedev_quadrature` (Lebedev rules for enabled S2 branches), and
`grid_mlp` / `grid_branch` (point-wise polynomial MLP or scalar-routed branch
mixer per grid path). These trade expressiveness for cost; the final `l = 0`
output remains a scalar.

### Environment-seeded initial features

With `use_env_seed` enabled, the initial node state is seeded from the local
environment: a DeePMD-style environment matrix produces FiLM-like scale and
shift values for the first scalar features, and the geometric initial embedding
is enabled when non-scalar degrees are present. With it disabled, the initial
state contains only atom-local scalar features, which keeps a one-block model
closed over the one-hop neighbor shell.

## Limitations

- DPA4/SeZM is implemented for the PyTorch backend only.
- Export uses `.pt2` (AOTInductor); the TorchScript freeze path is not used.
- Model compression is not supported.
- Multi-rank (multi-GPU/MPI) LAMMPS inference is supported only for a
  graph-kind `.pt2` (`--lower-kind graph`); the dense-kind `.pt2` fails fast
  at the first force evaluation, so dense-frozen models must run on a single
  MPI rank. See [Multi-GPU (MPI) inference](#multi-gpu-mpi-inference).
- The pt_expt graph-native inference route (`--lower-kind graph`) is
  unavailable for spin, charge/spin conditioning, or ZBL zone-bridging
  configurations, which stay on the dense route and therefore single-rank.
  On the supported graph route, multi-rank inference additionally requires
  every MPI rank to own or ghost at least one atom. See
  [Graph-native inference route (pt_expt)](#graph-native-inference-route-pt_expt).

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
