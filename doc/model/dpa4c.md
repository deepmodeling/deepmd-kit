# Descriptor DPA4C {{ pytorch_icon }}

> [!NOTE]
> **Supported backends**: PyTorch Exportable {{ pytorch_icon }} (`dp --pt-expt`)

DPA4C is the compact and compressible degree-wise descriptor of the DPA4
family. Where DPA4/SeZM targets the accuracy frontier through equivariant
message passing, DPA4C targets the throughput frontier: it reads each local
environment once, keeps no message-passing state, and admits a compressed CUDA
inference path in which its radial functions are replaced by tabulated splines.

Choose DPA4C when the run is limited by simulation speed or system size rather
than by the last increment of accuracy: large-scale molecular dynamics, long
trajectories, and distillation from a DPA4 teacher. Choose DPA4 when accuracy
is the binding constraint.

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

## How it works

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
{ref}`rcut <model[standard]/descriptor[dpa4c]/rcut>` of it.

Three consequences shape how the model is used in practice.

- **One-hop locality** keeps the per-step cost low and removes the cross-rank
  halo exchange of intermediate features that a message-passing model needs, so
  domain decomposition follows the ordinary pair-style path.
- **Exact smoothness at the cutoff.** The radial map is exactly zero at and
  beyond `rcut`, with continuous derivatives, so the potential energy surface
  stays smooth as neighbors cross the cutoff.
- **Analytic bounds on every per-atom quantity**, which is why compression needs
  neither an extrapolation region nor overflow checking.

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

### Options that matter

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
> The compressed CUDA path is compiled for `channels` in `{8, 16, 32, 64, 128}`,
> `lmax` in `{2, 3, 4}`, and `radial_modes` in `{0, 2, 4, 8}` only. A model
> trained outside those sets trains and evaluates correctly, but
> `dp --pt-expt compress` rejects it. Choose these values with deployment in
> mind.

### Recommended configurations and presets

The released grades, Nano, Mini, Neo, Air and Plus in ascending cost, pair each
descriptor width with a fitting width sized against it. They are good starting
points; `Neo` is the general-purpose default. Each grade is available as a named
model preset, `dpa4c-nano-v20260901`, `dpa4c-mini-v20260901`,
`dpa4c-neo-v20260901`, `dpa4c-air-v20260901` and `dpa4c-plus-v20260901`:
setting `model.preset` fills in `type_map` (all 118 elements), `descriptor` and
`fitting_net` from the release configuration, and entries written next to the
preset take precedence, as a whole for `type_map` and key by key inside
`descriptor` and `fitting_net`. Run-specific options such as `use_amp` and
`seed` are added alongside:

```json
{
  "model": {
    "preset": "dpa4c-neo-v20260901",
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "seed": 42
    },
    "fitting_net": {
      "seed": 42
    }
  }
}
```

The version tag identifies the release a preset reproduces: a later release
with different settings gets a new version, and existing presets are never
changed. The expansion and merge rules are described on the
[DPA4 page](dpa4.md#presets).

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

Compression is the deployment step. It replaces the analytic radial functions
and their type-pair modulation with tabulated splines evaluated by fused CUDA
kernels, and re-exports the model in the compact canonical graph form that the
fast inference path consumes. Because the radial map is analytically bounded and
vanishes at `rcut`, the table needs no extrapolation region and no overflow
checking.

Train, freeze, then compress the frozen archive:

```bash
dp --pt-expt train input.json
dp --pt-expt freeze -c model.ckpt.pt -o frozen_model --lower-kind graph
dp --pt-expt compress -i frozen_model.pt2 -o compressed_model.pt2
```

The two archives are not interchangeable. `frozen_model.pt2` carries the plain
graph lower and is the uncompressed intermediate; `compressed_model.pt2` carries
the compact canonical graph lower and is what you deploy. Compression selects
that lower on its own, so it takes no lower-kind option of its own.

Only `-s, --step` applies to DPA4C; it sets the uniform spline spacing in Å, and
a smaller value means a finer table and a larger model. The `--extrapolate`,
`--frequency`, and `--training-script` options exist for descriptors whose
tables need a second region, an overflow guard, or a minimum neighbor distance
computed from data; DPA4C needs none of them and ignores them.

Compression requires:

- the PyTorch Exportable backend on CUDA;
- `precision: "float32"`;
- `channels`, `lmax` and `radial_modes` inside the compiled sets listed above;
- an empty
  {ref}`exclude_types <model[standard]/descriptor[dpa4c]/exclude_types>`, since
  the fused kernel has no type-exclusion branch.

`dp --pt-expt compress` reports an explicit error when any of these is not met.
A model that excludes type pairs still trains and runs; deploy it as the
uncompressed graph archive.

## Running in LAMMPS

DPA4C uses the PyTorch `.pt2` (AOTInductor) export path and is served by the
`deepmd` pair style:

```lammps
pair_style deepmd compressed_model.pt2
pair_coeff * * O H
```

### Choosing a pair style

The compact canonical graph form exists so that the whole step can stay on the
device. Only the Kokkos pair styles use that device-resident entry point; the
host styles run the same archive through a per-step host round trip. On a GPU,
reaching DPA4C's advertised throughput therefore takes three things together: a
Kokkos-enabled LAMMPS build on the GPU backend, the compressed archive, and
`DP_CUDA_INFER` set at export time as described under
[Inference settings](#inference-settings). On a CPU host none of that applies --
see [CPU hosts](#cpu-hosts).

| Pair style    | Build                    | Accepted archive          | Execution                                      |
| ------------- | ------------------------ | ------------------------- | ---------------------------------------------- |
| `deepmd`      | any                      | graph lower or compressed | host round trip each step                      |
| `deepmd/kk`   | Kokkos, GPU backend only | graph lower or compressed | device-resident; compressed uses fused kernels |
| `dpa4spin`    | any, `atom_style spin`   | graph lower or compressed | host round trip each step                      |
| `dpa4spin/kk` | Kokkos, GPU backend only | compressed only           | device-resident                                |

Run under Kokkos with one GPU:

```bash
lmp -k on g 1 -sf kk -in in.lammps
```

### CPU hosts

DPA4C has a second set of hand-written operators for the CPU, so a compressed
archive runs the same fused pipeline on a host without a GPU. They are selected
automatically -- there is no level to set, because they replace a lowering of
the same arithmetic and are faster wherever they apply -- and they carry the
same numerical contract as the CUDA ones: float32 model computation, no
reduced-precision path.

```bash
export OMP_NUM_THREADS=$(nproc --all)
export DP_INTRA_OP_PARALLELISM_THREADS=$OMP_NUM_THREADS
export DP_INTER_OP_PARALLELISM_THREADS=1
lmp -in in.lammps
```

Three conditions have to hold for the fused CPU path to be taken:

- the archive is **compressed**, because the operator reads the radial table
  rather than evaluating the radial network;
- `channels` is one of 8, 16, 32, 64, 128, `lmax` one of 2, 3, 4, `radial_modes`
  one of 0, 2, 4, 8, the parameters are float32, and no type pairs are excluded;
- the descriptor is **not** spin-conditioned. The magnetic families need a
  source-major counterpart of the destination scan, which only the CUDA kernels
  carry; a spin model on a CPU host falls back to the portable path.

`deepmd/kk` brings nothing on a CPU. Its purpose is to keep the neighbor list
device-resident and avoid host-device synchronization, which on a host is
already absent, and everything outside the pair style is about 2% of the step.
Use the plain `deepmd` style.

Two settings are worth knowing:

- **Thread count.** The C++ interface reads
  `DP_INTRA_OP_PARALLELISM_THREADS`, not `OMP_NUM_THREADS`, and warns when it
  is unset. Set both, and prefer one thread per *physical* core.
- **Processor affinity.** If LAMMPS is launched from a process that has already
  initialized an OpenMP runtime under `OMP_PROC_BIND` -- a Python driver, for
  instance -- it inherits that process's affinity mask, which may be a single
  core, and will then run the model on one thread whatever `OMP_NUM_THREADS`
  says. The symptom is a low `CPU use` percentage in the LAMMPS timing summary
  next to a high thread count. Launch LAMMPS directly, or reset the mask in the
  child.

Resident memory is roughly 16 to 31 KB per atom depending on the grade, so a 4
GiB budget holds between 124 000 atoms (Plus) and 230 000 atoms (Nano). Most of
that is the neighbor graph, which depends on the cutoff rather than on the model
width.

`DP_CPU_MALLOC_RETAIN` controls a heap policy that matters at these sizes. By
default the operator library keeps large blocks that a step reuses instead of
returning them to the kernel on every free; without it a molecular-dynamics step
re-faults its whole working set and loses more than half its throughput above
about 32 000 atoms. Retaining them costs a few hundred megabytes to a gigabyte
of resident memory. Set `DP_CPU_MALLOC_RETAIN=0` on a memory-constrained host to
trade that back.

### Multiple GPUs

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

| Environment variable   | Default | Effect                                                                                                                                                          |
| ---------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DP_CUDA_INFER`        | `0`     | Fused CUDA kernel level: `0` off, `1` fused descriptor and fitting, `2` additionally fuses force and virial assembly. Levels 1 and 2 are numerically identical. |
| `DP_AMP_INFER`         | off     | bf16 autocast over the per-edge stage during inference. Independent of the training-time `use_amp`.                                                             |
| `DP_TF32_INFER`        | `0`     | float32 matmul precision: `0` highest, `1` high, `2` medium.                                                                                                    |
| `DP_CPU_MALLOC_RETAIN` | `1`     | Whether the CPU operator library retains large heap blocks between steps. See [CPU hosts](#cpu-hosts).                                                          |

A compressed model needs `DP_CUDA_INFER` of at least `1` to reach its fused
path on a GPU; at `0` it evaluates through the portable path and the compression
brings no speedup. On a CPU host there is no equivalent level: the fused
operators are always selected when the model is eligible. For molecular dynamics sensitive to the smoothness of the potential
energy surface, keep `DP_TF32_INFER=0` and `DP_AMP_INFER=0`.

> [!IMPORTANT]
> Set these variables **before** running `dp --pt-expt freeze` or
> `dp --pt-expt compress`. The exported `.pt2` is an AOTInductor artifact, so the
> kernel level and precision policy are captured into the graph at export time
> and are **not** re-evaluated when the `.pt2` is later loaded by LAMMPS.

## Native spin

DPA4C accepts a per-atom magnetic moment as an equivariant descriptor input.
The moment is not represented by a virtual atom: the atom count of the model
equals the number of physical atoms, and the magnetic force is the negative
spin gradient of the same energy that yields the conservative force,

```math
\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i},
\qquad
\mathbf{F}^{m}_i = -\frac{\partial E}{\partial \mathbf{s}_i} .
```

### What the descriptor represents

The magnetic moment is an axial vector: it is even under spatial inversion and
odd under time reversal, whereas a displacement is odd under inversion and even
under time reversal. The descriptor therefore emits only invariants of even
total spin order, which leaves it invariant under the full orthogonal group
acting jointly on positions and moments, including improper operations, and
invariant under time reversal. Reversing every moment leaves the energy
unchanged and reverses the magnetic force.

Four families of spin channels are accumulated over the neighbor shell and
contracted against one another and against the geometric moments:

| Family                              | Content                                                                                           | Interaction it represents                   |
| ----------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Isotropic vector                    | $\sum_j \varphi_c(r_{ij})\,\hat{\mathbf{s}}_j$                                                    | Heisenberg exchange                         |
| Bond-projected vector               | $\sum_j \varphi_c(r_{ij})\,(\hat{\mathbf{s}}_j\cdot\hat{\mathbf{u}}_{ij})\,\hat{\mathbf{u}}_{ij}$ | Symmetric anisotropic exchange              |
| Quadrupole                          | $\sum_j \varphi_c(r_{ij})\,B_2(\hat{\mathbf{s}}_j)$                                               | Biquadratic exchange, single-ion anisotropy |
| Magnitude and magnetic coordination | $\sum_j \varphi_c(r_{ij})\,\lvert\mathbf{s}_j\rvert^2$ and the gated neighbor count               | Longitudinal and stoichiometric terms       |

Two-body Heisenberg exchange, biquadratic exchange and single-ion anisotropy
are represented exactly rather than approximately: each corresponds to a single
emitted invariant times a learned radial profile. The Dzyaloshinskii-Moriya
interaction is not representable at any order, because the invariant read-out
contains no antisymmetric contraction.

The width of the spin block follows the degree-two width of the geometric
descriptor, so it is set by
{ref}`channels <model[standard]/descriptor[dpa4c]/channels>` and has no knob of
its own.

### Enabling native spin

Native spin is requested at the model level, not on the descriptor. The
`use_spin` list marks the magnetic types, either as booleans over the type map
or by element name:

```json
{
  "model": {
    "type_map": [
      "Ni",
      "O"
    ],
    "spin": {
      "scheme": "native",
      "use_spin": [
        true,
        false
      ]
    },
    "descriptor": {
      "type": "dpa4c",
      "rcut": 6.0,
      "channels": 32,
      "lmax": 2,
      "precision": "float32"
    }
  }
}
```

The `native` scheme is required; the virtual-atom `deepspin` scheme is not
supported by this descriptor. Training uses the `ener_spin` loss, whose
`start_pref_fm` and `limit_pref_fm` weight the magnetic force. A complete
example is provided in `examples/spin/dpa4c/input.json`.

A moment is conditioned by a per-type gate and a reference magnitude measured
from the training corpus, so a non-magnetic type contributes exactly zero to
every spin channel and the magnitude of a magnetic type is normalized to order
unity. A model that declares a magnetic type but receives no moment is
rejected rather than evaluated at zero, since the latter is indistinguishable
from a broken data pipeline and reports a vanishing magnetic force.

### Running a spin model in LAMMPS

Freeze and compress exactly as for any other DPA4C model, then select a spin
pair style from the table under
[Choosing a pair style](#choosing-a-pair-style). Both spin styles require
`atom_style spin`:

```lammps
atom_style        spin
pair_style        dpa4spin compressed_model.pt2
pair_coeff        * * Ni O
```

Ghost moments are supplied by the forward communication that `atom_style spin`
already performs, and the magnetic force is reduced back onto owning atoms
alongside the conservative force, so domain decomposition needs no additional
exchange.

`min_style spin` reads the magnetic force from the pair style and relaxes the
moment directions. Spin dynamics through `fix nve/spin` requires a LAMMPS build
whose fix recognizes this pair style, because the stock fix accumulates the
magnetic force only from pair styles matching its own name pattern.

A worked example, a rocksalt NiO cell in its type-II antiferromagnetic order,
is provided in `examples/spin/dpa4c/lmp/`.

## Data format

DPA4C consumes a mixed-type neighbor list, so it supports both the
[standard DeePMD-kit data format](../data/system.md) and the
[mixed-type data format](../data/system.md#mixed-type). Keep the `type_map`
order consistent across the dataset, the input file, and any downstream
`pair_coeff` mapping.

## Limitations

- DPA4C is implemented for the PyTorch Exportable backend (`dp --pt-expt`).
- Export uses `.pt2` (AOTInductor); the TorchScript freeze path is not used.
- Model compression requires `float32` and a configuration inside the compiled
  sets listed under [Model compression](#model-compression). The resulting
  archive runs fused kernels on either a CUDA device or a CPU host; only a
  spin-conditioned model is CUDA-only.
- The device-resident inference path requires a Kokkos-enabled LAMMPS build on
  the GPU backend.
- The descriptor is one-hop local by construction. Interactions beyond `rcut`
  are not represented, and unlike a message-passing model the effective range
  cannot be extended by adding layers.
- Native spin requires the `native` scheme; the virtual-atom `deepspin` scheme
  is not supported. The Dzyaloshinskii-Moriya interaction is not representable,
  as explained under [Native spin](#native-spin).

## Architecture details

Background on how the descriptor works, linking each part to the options that
control it. Skip it unless you are tuning those options.

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
