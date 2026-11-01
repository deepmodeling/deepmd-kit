# Descriptor DPA4C {{ pytorch_icon }}

> [!NOTE]
> **Supported backends**: PyTorch Exportable {{ pytorch_icon }} (`dp --pt-expt`)

DPA4C is the compact and compressible degree-wise descriptor of the DPA4
family. Where DPA4/SeZM targets the accuracy frontier through equivariant
message passing, DPA4C targets the throughput frontier: by default it reads
each local environment once, keeps no message-passing state, and compresses
into an archive that runs fast on GPUs and CPU hosts. A second
message-passing block makes the model more accurate at a higher cost (see
[Presets](#presets)).

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
network. With one message-passing block, the default, an atom's descriptor
depends only on the atoms within
{ref}`rcut <model[standard]/descriptor[dpa4c]/rcut>` of it, which keeps the
cost of a step low.

{ref}`n_blocks <model[standard]/descriptor[dpa4c]/n_blocks>` adds further
blocks. Each one reads the features that every neighbor computed in the
previous block, reweights the message that the neighbor sends, and adds the
directions of the neighbor's own environment to the moments of the atom, so
two atoms of the same element at the same distance no longer send the same
message. Every block extends the range of the descriptor by its cutoff.

The radial functions vanish at `rcut` with continuous derivatives, and the
effect of every further block fades out at its own cutoff in the same way, so
the potential energy surface stays smooth as neighbors cross a cutoff.
[Architecture details](#architecture-details) describes the descriptor more
fully.

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

DPA4C has no `sel` option: it always uses every neighbor within `rcut`, so
there is no neighbor capacity to size and nothing is truncated.

DPA4C defaults to `float32`
({ref}`precision <model[standard]/descriptor[dpa4c]/precision>`), which
compression requires. Double precision is neither necessary nor supported for
compressed inference.

### Options that matter

Every option, with its default and full description, is listed in the
{ref}`argument reference <model[standard]/descriptor[dpa4c]>`. Five of them
carry the accuracy–cost trade-off:

- **Depth** — {ref}`n_blocks <model[standard]/descriptor[dpa4c]/n_blocks>`.
  A second block makes the model more accurate than a single-block model one
  or two sizes up, and costs several times as much per step once compressed
  (see [Presets](#presets)); choose the depth with deployment in mind. `rcut`
  then takes either one cutoff for every block or a list of non-increasing
  cutoffs, one per block, such as `[6.0, 5.0]`. A further block evaluates only
  the neighbors within its own cutoff, so a smaller cutoff makes it markedly
  cheaper; the default of 6 Å for both blocks is the most accurate.
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
  shared profiles. It spends per-neighbor work without enlarging the per-atom
  state, which makes it the lever to reach for when memory rather than
  throughput is the binding constraint.
- **Radial basis** —
  {ref}`basis_type <model[standard]/descriptor[dpa4c]/basis_type>` and
  {ref}`n_radial <model[standard]/descriptor[dpa4c]/n_radial>` select the
  analytic basis that feeds the radial network; the `bessel/fix` and
  `gaussian/fix` forms keep the frequencies or centres at their initial
  values instead of training them.

The fitting network is sized against the descriptor because the invariant
output grows with `channels`. Unlike `radial_modes`, fitting width is not a free
trade against memory: widening or deepening the network costs throughput and
memory together. Widen it only when validation error is limited by fitting
capacity rather than by the descriptor.

> [!IMPORTANT]
> Compression supports `channels` in `{8, 16, 32, 64, 128}`, `lmax` in
> `{2, 3, 4}`, and `radial_modes` in `{0, 2, 4, 8}` only. A model trained
> outside those sets trains and evaluates correctly, but
> `dp --pt-expt compress` rejects it. Choose these values with deployment in
> mind.

### Presets

DPA4C preset names use `dpa4c-<size>-<version>`. The single-block sizes are
listed in ascending computational cost, followed by the two-block sizes; `neo`
is the general-purpose starting point:

| Version     | Available sizes                                                                    |
| ----------- | ---------------------------------------------------------------------------------- |
| `v20260911` | `nano`, `mini`, `neo`, `air`, `plus`, `mini-duo`, `neo-duo`, `air-duo`, `plus-duo` |
| `v20260901` | `nano`, `mini`, `neo`, `air`, `plus`                                               |

A `duo` size is the size of the same name with two message-passing blocks, for
example `dpa4c-air-duo-v20260911`. It is more accurate than the single-block
size one or two steps up. Uncompressed, it costs about twice as much per step
as its single-block size; compressed, roughly five to ten times as much on
large systems, because compression speeds up the second block far less than
the first. There is no `nano-duo`, because the single-block `mini` is more
accurate at the same cost.

Setting `model.preset` supplies `type_map` (all 118 elements), `descriptor`
and `fitting_net`. Run-specific settings are written alongside the preset:

```json
{
  "model": {
    "preset": "dpa4c-neo-v20260911",
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

Explicit settings override the preset: `type_map` is replaced as a whole,
while `descriptor` and `fitting_net` are merged key by key. See the
[DPA4 preset rules](dpa4.md#presets) for expansion details and multi-task use.

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

{ref}`use_amp <model[standard]/descriptor[dpa4c]/use_amp>` runs the
per-neighbor computations of the descriptor under bfloat16 automatic mixed
precision on CUDA during training, which lowers peak memory substantially; the
per-atom results stay in the descriptor precision. Use it on GPUs with native
bf16 support.

`use_amp` applies to training only and is not stored in the model; inference
has its own setting, `DP_AMP_INFER` (see
[Inference settings](#inference-settings)). A model trained in full precision
can therefore be evaluated under mixed precision, and the reverse.

## Zone bridging

DPA4C can add an analytical short-range repulsion to the learned energy, so that
close approaches in high-temperature, high-pressure or collision simulations meet
a physical repulsive wall instead of an extrapolated network output:

```math
E_i = E_i^{\mathrm{DPA4C}} + E_i^{\mathrm{pair}}.
```

A model with several message-passing blocks needs it most: in contacts closer
than about 0.45 of the covalent bond length, which no training set samples, it
extrapolates worse than a single block.

Two potentials are available. `bridging_method: "zbl"` uses the
Ziegler-Biersack-Littmark screened nuclear potential, which is derived from
frozen atomic electron densities and covers every element.
`bridging_method: "nlh"` uses the Nordlund-Lehtola-Hobler form, whose
coefficients are fitted to self-consistent quantum-chemical pair energies and
therefore follow the real repulsive wall much more closely; it leaves a smaller
residual repulsion at ordinary bond lengths, which is energy the learned model
would otherwise have to cancel. `zbl` is the default; whether `nlh` yields a
better trained model has not been established.

> [!NOTE]
> The `nlh` coefficients shipped with DeePMD-kit are this project's own refit and
> are **not** the coefficients published by Nordlund, Lehtola and Hobler. They are
> fitted independently to the same open reference data — K. Nordlund, G. Hobler and
> S. Lehtola, Zenodo [10.5281/zenodo.14172633](https://doi.org/10.5281/zenodo.14172633)
> v1.0 (2024), CC BY 4.0 — under conditions this use requires, and are therefore
> modified material. The functional form is that of
> [Phys. Rev. A **111**, 032818 (2025)](https://doi.org/10.1103/PhysRevA.111.032818).
> Pairs containing an element heavier than uranium, which the reference data does
> not cover, fall back to ZBL.

The analytical term acts on every pair inside `rcut`, and a window between an
inner and an outer radius hands each pair over between the two descriptions.
Below the inner radius a pair leaves the learned model entirely: both atoms are
described exactly as if the other one were absent, and the pair interacts
through the analytical term alone. Across the window the pair enters the
learned model smoothly, and beyond the outer radius the learned model is the
same as without bridging. The energy stays three times continuously
differentiable through the window, as it does at the outer cutoff.

The two radii follow the size of the pair they apply to. Each is a fraction of
the covalent bond length of that element pair, so a hydrogen pair is described
down to a much shorter separation than a caesium pair, and one setting covers
every element combination in the type map. The defaults place the outer radius
at 0.80 of the bond length, below all but the shortest multiple bonds, and
the inner radius at 0.26, deep inside the repulsive wall where the analytical
term is the responsible description.

Enable it on the model:

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
    "bridging_method": "zbl"
  }
}
```

`type_map` must name chemical elements, because the analytical term is built
from their nuclear charges and the window from their covalent radii.
`bridging_fraction_inner` and `bridging_fraction_outer` adjust the two
fractions. Giving `bridging_r_inner` and `bridging_r_outer` instead replaces
the rule with one window in Å applied to every pair alike, which is worth doing
only for a single-element system or to reproduce an earlier model. See
`examples/water/dpa4c/input-zbl.json` for a complete input.

Training leaves out frames that contain a pair closer than the midpoint of its
window, where the learned model has too little influence on the energy to fit a
label. The threshold follows from the window of each element pair, so
`training_data.min_pair_dist` must stay unset on a bridged model. The data
statistics, including those of `dp change-bias`, use the same filter, while
validation batches are not filtered, so a validation error also covers frames
that training leaves out.

Bridging therefore relies on short-range training data. The labels that are
kept have to reach into the window of every element pair, which in practice
means adding dimer scans that run down to about half the covalent bond length.
Labels inside the window keep the hand-over smooth; a pair whose window holds
no label can show spurious force maxima there.

The same model can be written explicitly as a `linear_ener` model over the
DPA4C model and an `inner_potential` sub-model with `weights: "sum"`; the
concise form above expands to it.

Bridging changes nothing after training: a bridged model freezes, compresses
and runs in LAMMPS, on one GPU or several, like the plain DPA4C model of the
same depth. Native spin and the frame charge state combine with bridging; the
analytical term depends on the separation alone and leaves the magnetic force
to the learned model.

A model trained without bridging fine-tunes into its bridged form, and the
other way round. Add the bridging keys above to the input, keep the
architecture, and start from the earlier checkpoint:

```bash
dp --pt-expt train input-zbl.json --finetune model.ckpt.pt
```

The descriptor and the fitting network carry over. Fine-tuning is required
rather than optional: the bridging window changes what the descriptor sees at
short range, so the learned part is trained against the analytical term it is
now summed with.

## Model compression

Compression is the deployment step: it tabulates the radial functions of the
model, which makes inference much faster on a GPU or a CPU host. Freeze and
compress a checkpoint in one operation:

```bash
dp --pt-expt train input.json
dp --pt-expt freeze -c model.ckpt.pt -o compressed_model --compress
```

This writes `compressed_model.pt2` with the default compression settings. Set
the [inference settings](#inference-settings) before running it.

A branch of a multi-task model is frozen by naming it with `--head`:

```bash
dp --pt-expt freeze -c model.ckpt.pt -o compressed_branch --head CHOSEN_BRANCH --compress
```

A branch whose fitting network is shared with the other branches compresses
like a single-task model, unless the widths of the descriptor and the case
embedding add up to the width of the first fitting layer or to half of it.
Such a branch is exported in a slower form, and the export reports it with a
warning.

On a GPU, compression also tunes the archive for the GPU model it runs on,
which adds from a few seconds to several minutes. The tuning changes only the
speed, never the results. On a GPU of another model the archive runs with
untuned defaults, so compress on the GPU model that will run the simulations.

To compress an existing frozen archive or choose the spline spacing, use the
compression command:

```bash
dp --pt-expt compress -i frozen_model.pt2 -o compressed_model.pt2 --step 0.01
```

`-s, --step` sets the uniform spline spacing in Å; a smaller value means a
finer table and a larger archive. DPA4C ignores `--extrapolate`, `--frequency`
and `--training-script`.

For ASE, `DP(model="model.pt", freeze=True)` automatically compresses supported
DPA4C checkpoints and exports a single-process `.pt2`. If compression is not
supported, ASE warns and retains the ordinary freeze path. See the
[ASE interface](../third-party/ase.md).

Compression requires:

- the PyTorch Exportable backend;
- `precision: "float32"`;
- `channels`, `lmax` and `radial_modes` inside the supported sets listed under
  [Options that matter](#options-that-matter);
- an empty
  {ref}`exclude_types <model[standard]/descriptor[dpa4c]/exclude_types>`.

`dp --pt-expt compress` and `freeze --compress` report an explicit error when
any of these is not met. A model with excluded type pairs still trains and
runs; deploy it as an uncompressed archive.

A model with several message-passing blocks compresses on the same terms and
runs compressed on one GPU or several, also with zone bridging, native spin or
a [latent-charge long-range module](long-range.md). For a GPU, its compressed
export additionally requires:

- a fitting network with `float32` weights, `tanh` or `silu` hidden layers whose
  widths are multiples of four, no `resnet_dt`, and no frame or atomic
  parameters;
- no `pair_exclude_types` or `atom_exclude_types`;
- for a branch of a multi-task model, widths with which the branch compresses
  like a single-task model, as described above;
- a long-range module, if any, that meets the conditions given in
  [Long-range latent charges](long-range.md#training-and-inference), such as a
  single charge channel.

The compressed export of any other model with several blocks for a GPU reports
an error. On a CPU host, a model with several blocks runs through the slower
general path (see [CPU hosts](#cpu-hosts)).

## Running in LAMMPS

DPA4C runs in LAMMPS from its `.pt2` archive through the `deepmd` pair style:

```lammps
pair_style deepmd compressed_model.pt2
pair_coeff * * O H
```

A model with several message-passing blocks needs `atom_modify map yes` when it
runs on a single MPI rank.

### Choosing a pair style

On a GPU, the fastest runs take three things together: a LAMMPS build with the
KOKKOS package on its GPU backend, running the `/kk` pair styles; a compressed
archive; and `DP_CUDA_INFER` set when the archive is exported, as described
under [Inference settings](#inference-settings). The other pair styles run the
same archive more slowly, because they copy the data of every step through
host memory. For a host without a GPU, see [CPU hosts](#cpu-hosts).

| Pair style    | LAMMPS build                            | Archive                    | Use                                               |
| ------------- | --------------------------------------- | -------------------------- | ------------------------------------------------- |
| `deepmd`      | any                                     | uncompressed or compressed | CPU hosts, and GPUs without KOKKOS                |
| `deepmd/kk`   | KOKKOS on a GPU                         | uncompressed or compressed | GPUs                                              |
| `dpa4spin`    | any, with `atom_style spin`             | uncompressed or compressed | spin models on CPU hosts, and GPUs without KOKKOS |
| `dpa4spin/kk` | KOKKOS on a GPU, with `atom_style spin` | compressed                 | spin models on GPUs                               |

Run under Kokkos with one GPU:

```bash
lmp -k on g 1 -sf kk -in in.lammps
```

Steps that compute the per-atom virial, which `compute centroid/stress/atom`
requests, are slower than the steps around them, so request it at a coarse
interval. The Kokkos pair styles provide the nine-component per-atom virial
only.

### CPU hosts

A compressed archive also runs fast on a host without a GPU. The CPU path is
selected automatically; there is no setting to choose it. Run with one thread
per physical core:

```bash
export OMP_NUM_THREADS=$(nproc --all)
export DP_INTRA_OP_PARALLELISM_THREADS=$OMP_NUM_THREADS
export DP_INTER_OP_PARALLELISM_THREADS=1
lmp -in in.lammps
```

The fast CPU path serves compressed models with one message-passing block and
without native spin. A model with several blocks or with native spin runs on a
CPU host through the slower general path. Use the plain `deepmd` pair style;
the Kokkos pair styles bring no speedup on a CPU host.

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

A CPU run needs roughly 15 to 30 KB of memory per atom, most of it for the
neighbor list, which grows with `rcut` rather than with the size of the model.
By default the CPU path also keeps the memory that it reuses from step to step,
which makes large systems much faster and costs a few hundred megabytes to a
gigabyte; on a host short of memory, set `DP_CPU_MALLOC_RETAIN=0`.

### Multiple GPUs

Run one MPI rank per GPU and make every GPU visible:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 lmp -k on g 4 -sf kk -in in.lammps
```

Use an MPI library that can access GPU memory (CUDA-aware MPI), and launch it
with the options given in
[Install LAMMPS](../install/install-lammps.md#mpi-for-runs-on-several-gpus).
With an MPI library without that support, LAMMPS prints a warning and the run
proceeds, but noticeably more slowly. A model with several message-passing
blocks exchanges data between the GPUs at every step and depends most on these
options; it needs no other settings of its own.

Use a non-zero neighbor skin, for example `neighbor 2.0 bin`, to keep the GPU
memory of every step stable; a zero skin rebuilds the neighbor list every step.

For a model with several message-passing blocks, keep the default CUDA memory
allocator: under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, Open MPI
over UCX aborts the run at its first step.

## Inference settings

Environment variables control inference:

| Environment variable   | Default | Effect                                                                                                                                |
| ---------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `DP_CUDA_INFER`        | `0`     | Fast GPU kernels: `0` off, `1` on, `2` on also for the forces and virials, the fastest level. Levels 1 and 2 give identical results.  |
| `DP_AMP_INFER`         | off     | bfloat16 automatic mixed precision over the per-neighbor computations during inference. Independent of the training option `use_amp`. |
| `DP_TF32_INFER`        | `0`     | float32 matrix-multiplication precision: `0` highest, `1` high, `2` medium.                                                           |
| `DP_CPU_MALLOC_RETAIN` | `1`     | Whether the CPU path keeps the memory it reuses from step to step. See [CPU hosts](#cpu-hosts).                                       |

A compressed model runs fast on a GPU only with `DP_CUDA_INFER` of at least
`1`; at `0` compression brings no speedup there. A CPU host needs no such
setting. For molecular dynamics sensitive to the smoothness of the potential
energy surface, keep `DP_TF32_INFER=0` and `DP_AMP_INFER=0`.

> [!IMPORTANT]
> Set `DP_CUDA_INFER`, `DP_AMP_INFER` and `DP_TF32_INFER` **before** running
> `dp --pt-expt freeze` or `dp --pt-expt compress`. They are fixed into the
> `.pt2` archive when it is exported, and LAMMPS does **not** read them when it
> loads the archive. `DP_CPU_MALLOC_RETAIN` is read when the model runs.

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
under time reversal. The descriptor is invariant under rotations and
reflections acting jointly on positions and moments, and under time reversal:
reversing every moment leaves the energy unchanged and reverses the magnetic
force.

It represents two-body Heisenberg exchange, biquadratic exchange and
single-ion anisotropy exactly, together with symmetric anisotropic exchange and
longitudinal terms in the magnitude of the moments. The
Dzyaloshinskii-Moriya interaction is not representable at any order.

The spin part of the descriptor has no options of its own; its width follows
{ref}`channels <model[standard]/descriptor[dpa4c]/channels>`.

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

A non-magnetic type contributes nothing to the spin part of the descriptor,
and the moments of each magnetic type are normalized by a reference magnitude
measured from the training data. A model with a magnetic type requires the
moments as input: evaluating it without them is an error, not an evaluation at
zero moments.

### Running a spin model in LAMMPS

Freeze and compress exactly as for any other DPA4C model, then select a spin
pair style from the table under [Choosing a pair style](#choosing-a-pair-style).
Both spin styles require `atom_style spin` and, on a single MPI rank, an atom
map; the
[LAMMPS command reference](../third-party/lammps-command.md#pair_style-dpa4spin)
describes their keywords. A minimal input reads:

```lammps
atom_style        spin
atom_modify       map yes
pair_style        dpa4spin compressed_model.pt2
pair_coeff        * * Ni O
```

Spin models run on several MPI ranks without further settings.

By default the pair styles store the precession magnetic force that the LAMMPS
SPIN package expects, the negative energy gradient `-dE/dS` scaled by the
moment magnitude over `hbar`. `min_style spin` reads this force and relaxes the
moment directions. The `fix nve/spin` of a stock LAMMPS build does not use this
force: the fix evaluates the magnetic force itself, through the pair styles
whose names begin with `spin`, and `dpa4spin` and `dpa4spin/kk` are not among
them. With no such pair style, the fix runs silently on a single MPI rank and
stops with a sectoring error on several. Spin dynamics therefore needs a LAMMPS
build whose integrator reads the magnetic force of these pair styles.

A Newtonian integrator of the Cartesian spin vector needs `-dE/dS` itself
rather than the precession force. The keyword `magnetic_force cartesian`
selects `-dE/dS` and requires `units metal`:

```lammps
pair_style        dpa4spin/kk compressed_model.pt2 magnetic_force cartesian
```

A worked example, a rocksalt NiO cell in its type-II antiferromagnetic order,
is provided in `examples/spin/dpa4c/lmp/`.

## Embedding extraction

A trained DPA4C model exports learned representations for downstream analysis
with `dp embed`. A single forward pass (no force or virial) produces the per-atom
`descriptor`, the per-atom `atomic_feature` (the activation after the last
fitting hidden layer), and the per-frame `structural_feature` (the sum of
`atomic_feature` over real atoms not excluded by `atom_exclude_types`):

```bash
dp embed -m model.ckpt.pt -s /path/to/system -o embedding.hdf5
```

Atoms excluded by `atom_exclude_types` retain their per-atom `descriptor` and
`atomic_feature`; only the structural sum leaves them out.

The command reads the training checkpoint (`.pt`) or the frozen model (`.pt2`),
compressed or not, and a ZBL-bridged model reports the embeddings of its learned
model. Unlike a LAMMPS run, `dp embed` reads the
[inference settings](#inference-settings) when it runs rather than at export: on
a GPU, a compressed model computes its descriptor fast only when
`DP_CUDA_INFER` is at least `1` for the command. See
[model embeddings](../inference/embedding.md) for the output format and the
Python interface.

## Data format

DPA4C supports both the [standard DeePMD-kit data format](../data/system.md)
and the [mixed-type data format](../data/system.md#mixed-type). Keep the
`type_map` order consistent across the dataset, the input file, and any
downstream `pair_coeff` mapping.

## Limitations

- DPA4C is implemented for the PyTorch Exportable backend (`dp --pt-expt`).
- Export uses `.pt2` (AOTInductor); the TorchScript freeze path is not used.
- Model compression requires `float32` and supported values of `channels`,
  `lmax` and `radial_modes` (see [Model compression](#model-compression)). A
  compressed archive runs fast on a GPU or a CPU host; a spin model and a model
  with several blocks run fast on GPUs only.
- The fastest GPU runs require a LAMMPS build with the KOKKOS package on its
  GPU backend.
- A single-block descriptor is one-hop local: interactions beyond `rcut` are
  not represented. Each further block extends the range by its own cutoff.
- A model with several blocks costs several times its single-block size per
  step even when compressed, and on a GPU it compresses only under the
  conditions given under [Model compression](#model-compression). In pair
  contacts closer than about 0.45 of the covalent bond length, which no
  training set samples, it extrapolates worse than a single block; enable
  [zone bridging](#zone-bridging) wherever such contacts can occur.
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

The per-atom state is the sum of these neighbor contributions, held separately
for each angular degree. Degree zero carries `channels` scalar values; higher
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

A further block starts from the invariants and moments that every atom holds
after the previous block. Each neighbor turns them into a few saturated
features: equivariant keys, which describe how its own environment is oriented,
and an invariant state. For every neighbor pair an attention logit combines the
state of the neighbor, the orientation of its keys relative to the bond, and
the amplitude of the pair in one small hidden layer, so the effect of a
neighbor depends on its chemical environment and on the distance together. The
logit rescales the amplitude of the pair by a factor between zero and two, and
a softmax over the neighbors adds their keys to the moments of the atom. The
block then sums the neighbor contributions again and reads out the invariants
as before. Every map that produces the output of a block starts at zero, so a
model with several blocks starts out as its single-block counterpart, and every
block vanishes smoothly at its own cutoff.
