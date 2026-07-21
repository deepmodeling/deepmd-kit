# Descriptor DPA4C {{ pytorch_icon }}

> [!NOTE]
> **Supported backends**: PyTorch Exportable {{ pytorch_icon }} (`dp --pt-expt`)

DPA4C is the compact and compressible degree-wise descriptor of the DPA4
family. Where DPA4/SeZM targets the accuracy frontier through equivariant
message passing, DPA4C targets the throughput frontier: it reads each local
environment once, keeps no message-passing state, and admits a compressed CUDA
inference path in which its radial functions are replaced by tabulated splines.
It is intended for large-scale molecular dynamics and as a distillation student
of a DPA4 teacher.

DPA4C is selected as a descriptor, `descriptor.type: "dpa4c"`, and pairs with
the standard energy fitting network. There is no separate `model.type` scaffold.

## Quick start

```bash
cd examples/water/dpa4c
dp --pt-expt train input.json
```

`examples/water/dpa4c/input.json` is a complete energy-training input you can
copy and adapt. See [training energy models](train-energy.md) for the general
workflow shared by all energy models.

## Overview

DPA4C predicts atomic energies and obtains forces and virials by
differentiating the energy, the same conservative formulation used by every
standard DeePMD energy model:

```math
\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}.
```

For each atom the descriptor accumulates angular moments of its neighbors up to
degree {ref}`lmax <model[standard]/descriptor[dpa4c]/lmax>`, contracts them into
rotationally invariant scalars, and passes only those scalars to the fitting
network. The neighbor shell is read exactly once: there is no message passing,
so an atom's descriptor depends only on the atoms within
{ref}`rcut <model[standard]/descriptor[dpa4c]/rcut>` of it. This one-hop
locality is what keeps the per-step cost low and makes the compressed inference
path possible.

Two properties follow from the construction and matter in practice. The radial
map is exactly zero at and beyond `rcut`, with continuous derivatives, so the
potential energy surface stays smooth as neighbors cross the cutoff. And every
per-atom quantity is bounded analytically, which is why compression needs
neither an extrapolation region nor overflow checking.

If you want the design details, see
[Architecture details](#architecture-details) at the end of this page.

## Configuration

### Minimal input

A minimal DPA4C descriptor needs nothing but its type; every option has a
documented default.

```json
{
  "model": {
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "type": "dpa4c",
      "rcut": 6.0
    },
    "fitting_net": {
      "neuron": [
        128,
        128,
        128
      ],
      "activation_function": "silu"
    }
  }
}
```

DPA4C has no `sel` option. It is graph-native: the descriptor consumes a
carry-all neighbor graph that holds every neighbor within `rcut`, rather than a
fixed-capacity neighbor list. There is no capacity to size, no dependence on the
densest frame in the dataset, and no truncation to guard against.

DPA4C defaults to `float32`
({ref}`precision <model[standard]/descriptor[dpa4c]/precision>`), which is also
what the compressed CUDA path requires. Double precision is neither necessary
nor supported for compressed inference.

### Main options

Every option, with its default and full description, is listed in the
{ref}`argument reference <model[standard]/descriptor[dpa4c]>`. Four of them
carry the accuracy–cost trade-off:

- **Width** — {ref}`channels <model[standard]/descriptor[dpa4c]/channels>`, one
  of 8, 16, 32, 64, or 128. This is the primary scaling knob. It widens the
  scalar and edge features, the per-atom angular state, and the descriptor
  output together, so it costs both throughput and the largest system that fits
  in memory.
- **Angular degree** — {ref}`lmax <model[standard]/descriptor[dpa4c]/lmax>`, one
  of 2, 3, or 4. Each additional degree adds angular components to the per-atom
  state. Its absolute cost is fixed by the degree, so its *relative* cost is
  largest at narrow widths.
- **Radial resolution** —
  {ref}`radial_modes <model[standard]/descriptor[dpa4c]/radial_modes>`. Zero
  leaves every ordered atom-type pair with a rescaled copy of one shared radial
  function; larger values let each pair select its own radial shape from several
  shared profiles. It spends per-edge work without enlarging the per-atom state,
  which makes it the lever to reach for when memory rather than throughput is
  the binding constraint.
- **Radial basis** —
  {ref}`basis_type <model[standard]/descriptor[dpa4c]/basis_type>` and
  {ref}`n_radial <model[standard]/descriptor[dpa4c]/n_radial>` select the
  analytic basis that feeds the radial network.

> [!IMPORTANT]
> Compressed inference is compiled for
> `radial_modes` in `{0, 2, 4, 8}` only. A model trained with any other value
> trains and runs correctly on the portable path, but `dp --pt-expt compress`
> will reject it. Choose the value with compression in mind if you intend to
> deploy the compressed model.

### Recommended configurations

The released grades pair each descriptor width with a fitting width sized
against it, in ascending cost. They are good starting points; `Neo` is the
general-purpose default.

| Grade | `channels` | `lmax` | `radial_modes` | Fitting hidden width |
| ----- | ---------: | -----: | -------------: | -------------------: |
| Nano  |          8 |      2 |              0 |                   96 |
| Mini  |         32 |      2 |              0 |                  192 |
| Neo   |         32 |      2 |              4 |                  192 |
| Air   |         64 |      3 |              4 |                  256 |
| Plus  |        128 |      3 |              4 |                  384 |

`Mini` and `Neo` share a descriptor width and differ only in the radial modes,
which buys accuracy at a per-edge cost while leaving the largest tractable
system unchanged. `Air` and `Plus` additionally raise the angular degree.

The fitting network is sized against the descriptor because the invariant
output grows with `channels`. Unlike `radial_modes`, fitting width is not a free
trade against memory: it adds per-atom activations and the derivatives saved for
the force backward pass, so widening or deepening it costs throughput and
capacity together. Widen it only when validation error is limited by fitting
capacity rather than by the descriptor.

## Training

The recommended objective is the standard conservative energy loss:

```json
{
  "loss": {
    "type": "ener"
  }
}
```

See [training energy models](train-energy.md) for the general workflow.

### Mixed precision

{ref}`use_amp <model[standard]/descriptor[dpa4c]/use_amp>` runs the per-edge
stage under bfloat16 automatic mixed precision on CUDA during training. The
activation footprint of that stage scales with the edge count and dominates
memory, so enabling it lowers peak memory substantially; the destination
reduction and the invariant readout stay in the descriptor precision. Use it on
GPUs with native bf16 support.

`use_amp` is an execution policy rather than model state: it is not serialized,
and evaluation and inference are governed independently by `DP_AMP_INFER`. A
model trained in full precision can therefore be evaluated under mixed
precision, and the reverse.

## Model compression

Compression replaces the analytic radial functions and their type-pair
modulation with tabulated splines evaluated by fused CUDA kernels. Because the
radial map is analytically bounded and vanishes at `rcut`, the table needs no
extrapolation region and no overflow checking.

The workflow is the standard three steps:

```bash
dp --pt-expt train input.json
dp --pt-expt freeze -c model.ckpt.pt -o frozen_model --lower-kind graph
dp --pt-expt compress -i frozen_model.pt2 -o compressed_model.pt2
```

Only `-s, --step` applies to DPA4C; it sets the uniform spline spacing in Å, and
a smaller value means a finer table and a larger model.
The `--extrapolate`, `--frequency`, and `--training-script` options exist for
descriptors whose tables need a second region, an overflow guard, or a minimum
neighbor distance computed from data; DPA4C needs none of them and ignores them.

Compression requires:

- the PyTorch Exportable backend on CUDA;
- `precision: "float32"`;
- `channels` in `{8, 16, 32, 64, 128}`, `lmax` in `{2, 3, 4}`, and
  `radial_modes` in `{0, 2, 4, 8}`;
- an empty
  {ref}`exclude_types <model[standard]/descriptor[dpa4c]/exclude_types>`, since
  the fused kernel has no type-exclusion branch. A compressed model with
  excluded pairs falls back to the portable path.

`dp --pt-expt compress` reports an explicit error when the configuration falls
outside these sets.

## Export and running in LAMMPS

DPA4C uses the PyTorch `.pt2` (AOTInductor) export path. Freeze with the graph
lower, which is the form the C++ graph path consumes:

```bash
dp --pt-expt freeze -c model.ckpt.pt -o frozen_model --lower-kind graph
```

Use the frozen or compressed `.pt2` with the `deepmd` pair style:

```lammps
pair_style deepmd compressed_model.pt2
pair_coeff * * O H
```

Because DPA4C performs no message passing, it needs no cross-rank halo exchange
of intermediate features, and MPI domain decomposition follows the ordinary
pair-style path. Launch one MPI rank per GPU and make every target device
visible:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 lmp -in in.lammps
```

Use a non-zero neighbor skin, for example `neighbor 2.0 bin`, to keep per-step
GPU memory stable; a zero skin rebuilds the neighbor list every step.

## Inference settings

Inference behavior is controlled by environment variables read when the model is
constructed:

| Environment variable | Default | Effect                                                                                                                                                                                                                                                       |
| -------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `DP_CUDA_INFER`      | `0`     | Fused CUDA kernel level. `0` disables them. `1` uses the fused descriptor and fitting operators with the force from autograd. `2` additionally collapses descriptor, fitting, and force/virial assembly into one operator, numerically identical to level 1. |
| `DP_AMP_INFER`       | off     | bf16 autocast over the per-edge stage during inference. Independent of the training-time `use_amp`.                                                                                                                                                          |
| `DP_TF32_INFER`      | `0`     | float32 matmul precision: `0` highest, `1` high, `2` medium.                                                                                                                                                                                                 |

A compressed model requires `DP_CUDA_INFER` of at least `1` to reach its fused
path; at `0` it evaluates through the portable path and the compression brings
no speedup. For molecular dynamics sensitive to the smoothness of the potential
energy surface, keep `DP_TF32_INFER=0` and `DP_AMP_INFER=0`.

> [!IMPORTANT]
> Set these variables **before** running `dp --pt-expt freeze` or
> `dp --pt-expt compress`. The exported `.pt2` is an AOTInductor artifact, so the
> kernel level and precision policy are captured into the graph at export time
> and are **not** re-evaluated when the `.pt2` is later loaded by LAMMPS.

## Data format

DPA4C consumes a mixed-type neighbor list, so it supports both the
[standard DeePMD-kit data format](../data/system.md) and the
[mixed-type data format](../data/system.md#mixed-type). Keep the `type_map`
order consistent across the dataset, the input file, and any downstream
`pair_coeff` mapping.

## Architecture details

Optional background on how the descriptor works, linking each part to the
options that control it. Skip it unless you are tuning those options.

### Edge features

For every neighbor pair within `rcut`, the interatomic distance is expanded on
an analytic radial basis (`basis_type`, with `n_radial` functions) and passed
through a radial network that produces one amplitude per channel. The ordered
pair of atom types modulates that amplitude with a learned scale and shift, so
the radial shape depends on which species face each other without instantiating
a separate network per pair. With `radial_modes` greater than zero, each ordered
pair additionally mixes several shared radial profiles with its own
coefficients, which lets pairs differ in shape rather than only in scale.

Each amplitude is multiplied by a smooth cutoff envelope whose value and first
derivatives vanish at `rcut`, and by the real spherical harmonics of the
neighbor direction up to degree `lmax`.

### Degree-wise moments and invariant read-out

The per-atom state is the sum of these edge contributions, held separately for
each angular degree. Degree zero carries `channels` scalar values; higher
degrees carry progressively fewer channels, each with `2l + 1` angular
components. This tapering keeps the state small, which is what bounds both the
per-step cost and the memory per atom.

The read-out contracts the moments into rotationally invariant scalars — norms
and cross-channel products within each degree, together with couplings across
degrees — and appends two measures of neighborhood density. Only these scalars
reach the fitting network:

```math
\mathcal{D}_i = \mathrm{Invariants}\left(\{\mathbf{M}_i^{(l)}\}_{l=0}^{l_{\max}}\right).
```

Because the contraction is exactly rotationally invariant, the descriptor and
hence the energy are invariant under global rotation, and the forces obtained by
differentiation are equivariant.

### Output calibration

Descriptor statistics are used once, at initialization, to record a fixed
per-coordinate scale that puts the invariant outputs on a comparable footing
before they enter the fitting network. This is an initialization preconditioner,
not a running normalization: no sample-dependent statistic is evaluated during
training or inference, so the model remains a pure function of the atomic
positions and types.

## Limitations

- DPA4C is implemented for the PyTorch Exportable backend (`dp --pt-expt`).
- Export uses `.pt2` (AOTInductor); the TorchScript freeze path is not used.
- Model compression requires CUDA, `float32`, and a configuration inside the
  compiled sets listed under [Model compression](#model-compression).
- The descriptor is one-hop local by construction. Interactions beyond `rcut`
  are not represented, and unlike a message-passing model the effective range
  cannot be extended by adding layers.
