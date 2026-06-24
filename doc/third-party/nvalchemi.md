# Molecular dynamics with nvalchemi-toolkit

[`nvalchemi-toolkit`](https://github.com/NVIDIA/nvalchemi-toolkit) is NVIDIA's
GPU-accelerated framework for batched molecular dynamics and structure
optimization with machine-learning interatomic potentials. DeePMD-kit ships a
thin adapter, `DPA4Wrapper`, that exposes a trained DPA-4 / SeZM model to any
`nvalchemi` dynamics engine (NVE, NVT, NPT, FIRE, ...). The model itself runs
unmodified; the wrapper only translates between the `nvalchemi` graph batch and
the model's internal interface.

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}, for DPA-4 / SeZM energy
models. `nvalchemi-toolkit` is an optional dependency and must be installed
separately. A CUDA device is recommended, since `nvalchemi`'s neighbour-list and
integrator kernels are GPU-accelerated.
:::

## Installation

Install the optional toolkit through the DeePMD-kit extra:

```bash
pip install deepmd-kit[nvalchemi]
```

This pulls in the `nvalchemi-toolkit` package; equivalently, install it directly
with `pip install nvalchemi-toolkit`. Refer to the `nvalchemi-toolkit`
documentation for the build that matches your Python, platform, and CUDA
environment.

The DeePMD-kit adapter lives in `deepmd.pt.nvalchemi`; importing it without
`nvalchemi-toolkit` present raises an actionable error.

## Loading a model

A trained DeePMD-kit checkpoint (`.pt`) is loaded and wrapped in one call:

```python
import torch
from deepmd.pt.nvalchemi import DPA4Wrapper

model = DPA4Wrapper.from_checkpoint(
    "model.ckpt.pt",
    device=torch.device("cuda"),
    compute_stress=True,  # enable the Cauchy stress output (needs a periodic cell)
)
```

For a multi-task checkpoint, pass the branch name with `head="..."`. An
already-instantiated model can be wrapped directly with `DPA4Wrapper(model)`.

`from_checkpoint` also accepts a frozen `.pt2` (AOTInductor) package produced by
`dp --pt freeze`; it is loaded through its precompiled callable (float64 I/O,
and device-locked to the host it was frozen on).

### Performance

The model runs eagerly by default. To use DeePMD-kit's compiled inference path,
set the environment variables **before** loading the model:

- `DP_COMPILE_INFER=1` — compile the model. The first call pays a one-time
  compile cost (~1–2 min); subsequent steps are roughly 3x faster, and the
  dynamic-shape graph handles the changing neighbour count during MD without
  recompiling.
- `DP_TRITON_INFER=1` — additionally enable the Triton inference kernels for a
  further speedup on larger cells.

A frozen `.pt2` package bakes the compilation in — and, when you run
`dp --pt freeze` with `DP_TRITON_INFER=1` set, the Triton kernels too — so it
skips the warm-up at the cost of being device-locked.

## Single-point evaluation

Build an `AtomicData` object, batch it, compute a neighbour list, and call the
model. The wrapper returns a dictionary with `energy` (shape `(B, 1)`),
`forces` (shape `(N, 3)`), and, when enabled, `stress` (shape `(B, 3, 3)`):

```python
from nvalchemi.data import AtomicData, Batch
from nvalchemi.neighbors import compute_neighbors

data = AtomicData(
    atomic_numbers=atomic_numbers,  # (N,) integer atomic numbers
    positions=positions,  # (N, 3) in Angstrom
    cell=cell,  # (1, 3, 3) lattice vectors, or omit for a cluster
    pbc=pbc,  # (1, 3) booleans, or omit for a cluster
)
batch = Batch.from_data_list([data], device="cuda")
compute_neighbors(batch, config=model.model_config.neighbor_config)

out = model(batch)
energy = out["energy"]  # (B, 1) eV
forces = out["forces"]  # (N, 3) eV/A
stress = out["stress"]  # (B, 3, 3) eV/A^3 (Cauchy stress = virial / volume)
```

Forces and stress are computed conservatively inside the model and returned
directly, so no gradient bookkeeping is required on the caller side.

## Molecular dynamics

For dynamics, register a neighbour-list hook so the list is rebuilt before each
force evaluation, then drive the batch with an integrator. The following snippet
runs canonical (NVT) dynamics with a Langevin thermostat:

```python
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import initialize_velocities
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.integrators import NVTLangevin
from nvalchemi.hooks import NeighborListHook
from nvalchemi.neighbors import compute_neighbors

# ``forces`` and ``energy`` are pre-allocated so the integrator can read forces
# and the engine can write results back into the batch in place.
data = AtomicData(
    atomic_numbers=atomic_numbers,
    positions=positions,
    cell=cell,
    pbc=pbc,
    forces=torch.zeros_like(positions),
    energy=torch.zeros((1, 1), dtype=positions.dtype, device=positions.device),
)
batch = Batch.from_data_list([data], device="cuda")

# Seed Maxwell-Boltzmann velocities at the target temperature.
temperature = torch.full(
    (batch.num_graphs,), 330.0, dtype=positions.dtype, device="cuda"
)
initialize_velocities(
    batch.velocities, batch.atomic_masses, temperature, batch.batch_idx.int()
)

nl_hook = NeighborListHook(
    model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE
)
nvt = NVTLangevin(model, dt=0.5, temperature=330.0, friction=0.01, hooks=[nl_hook])

# Prime the neighbour list and forces, then integrate.
compute_neighbors(batch, config=model.model_config.neighbor_config)
nvt.compute(batch)
batch = nvt.run(batch, n_steps=1000)
```

Switching ensemble is a one-line change: use `NVE(model, dt=...)` for the
microcanonical ensemble or `NPT(...)` for constant pressure (which consumes the
`stress` output). `nvalchemi` also provides logging and monitoring hooks (e.g.
`LoggingHook`, `EnergyDriftMonitorHook`) that attach to the same engine.

## Geometry optimization

The same model drives the FIRE optimizer. Convergence is controlled by a
maximum-force criterion:

```python
from nvalchemi.dynamics.base import ConvergenceHook
from nvalchemi.dynamics.optimizers import FIRE2

opt = FIRE2(
    model,
    dt=1.0,
    hooks=[nl_hook],
    convergence_hook=ConvergenceHook.from_fmax(threshold=0.05),
)
compute_neighbors(batch, config=model.model_config.neighbor_config)
opt.compute(batch)
batch = opt.run(batch, n_steps=200)  # stops early once fmax <= 0.05 eV/A
```

## Outputs and configuration

The wrapper advertises its capabilities through `model.model_config`
(an `nvalchemi` `ModelConfig`):

- `outputs` — `energy`, `forces`, and `stress`.
- `active_outputs` — the subset computed on each call. `energy` and `forces` are
  active by default; `stress` is added when `compute_stress=True` (or via
  `model.set_config("active_outputs", {"energy", "forces", "stress"})`).
- `neighbor_config` — the cutoff and neighbour-list format the model requires.
  Pass it to `compute_neighbors` or `NeighborListHook` as shown above.

## Heterogeneous batches

A single `Batch` may contain several structures of different sizes and cells.
The wrapper evaluates the whole batch in one pass and returns per-structure
energy and stress (`(B, 1)` and `(B, 3, 3)`) together with the concatenated
per-atom forces (`(N, 3)`), making it straightforward to evaluate many systems
at once.

## Units and conventions

- Lengths are in Angstrom, energies in eV, masses in amu, and time in
  femtoseconds.
- Atomic numbers are mapped to model types using the checkpoint `type_map`. Pass
  `atomic_number_to_type={Z: type_index, ...}` to `DPA4Wrapper` to override this
  for non-standard type maps.
- The reported `stress` is the Cauchy stress, equal to the virial divided by the
  cell volume; it requires a periodic cell.

## Limitations

- Only DPA-4 / SeZM energy models are supported.
- Acceleration uses DeePMD-kit's own compiled inference (`DP_COMPILE_INFER` or a
  frozen `.pt2`); `nvalchemi`'s `FusedStage` `torch.compile` is not used.
- Embeddings (`compute_embeddings`) require the `.pt` backend, not `.pt2`.
- Charge / spin conditioning is applied as a single global value per batch.

## Examples

Complete, runnable scripts for single-point evaluation, NVE, NVT, and geometry
optimization are provided in
[`examples/water/dpa4/nvalchemi/`](https://github.com/deepmodeling/deepmd-kit/tree/master/examples/water/dpa4/nvalchemi).
