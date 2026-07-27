# Running DPA-4 / SeZM with nvalchemi-toolkit

This directory contains runnable examples for driving a trained DPA-4 / SeZM
model with NVIDIA's [`nvalchemi-toolkit`](https://github.com/NVIDIA/nvalchemi-toolkit)
molecular-dynamics framework. The model is loaded through
`deepmd.pt.nvalchemi.DPA4Wrapper`, a thin adapter that exposes a DeePMD-kit
PyTorch model to any `nvalchemi` dynamics engine.

For a conceptual overview and the full API reference, see the user guide at
`doc/third-party/nvalchemi.md`.

## Prerequisites

- A DeePMD-kit installation with the PyTorch backend.
- The optional `nvalchemi-toolkit` package (`pip install deepmd-kit[nvalchemi]`,
  or `pip install nvalchemi-toolkit`; see its documentation for the build matching
  your Python, platform, and CUDA environment). A CUDA device is
  recommended, since `nvalchemi`'s neighbour-list and integrator kernels are
  GPU-accelerated.
- A trained DPA-4 / SeZM checkpoint (`.pt`, or a frozen `.pt2`). The examples
  default to the smoke-test checkpoint shipped at `../lmp/pretrained.pt`; replace
  it with your own model for production runs.

For faster inference, export `DP_COMPILE_INFER=1` (optionally `DP_TRITON_INFER=1`)
before running any script to enable the compiled path, or pass a frozen `.pt2`
package as `--model`.

The example structures are read from the bundled water dataset
(`../../data/data_0`), which provides a 192-atom periodic liquid-water cell.

## Examples

Each script is self-contained and documented; run any of them with `--help` to
see all options.

| Script            | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| `single_point.py` | Evaluate potential energy, atomic forces, and the Cauchy stress tensor. |
| `run_nve.py`      | Microcanonical (NVE) MD; reports total-energy conservation.             |
| `run_nvt.py`      | Canonical (NVT) MD with a Langevin thermostat at a target temperature.  |
| `relax.py`        | Fixed-cell geometry optimization with the FIRE2 optimizer.              |

## Quick start

```bash
cd examples/water/dpa4/nvalchemi

# Single-point energy / forces / stress
python single_point.py

# 200-step NVE trajectory seeded at 300 K
python run_nve.py --steps 200 --dt 0.5 --temperature 300

# NVT at 330 K with a Langevin thermostat
python run_nvt.py --steps 300 --temperature 330 --friction 0.01

# Relax to a maximum force of 0.05 eV/A
python relax.py --fmax 0.05 --max-steps 200
```

To use your own model and structure:

```bash
python run_nvt.py --model /path/to/model.ckpt.pt --data /path/to/deepmd/system
```

## Notes

- **Units** follow the standard atomistic-MD convention: lengths in Angstrom,
  energies in eV, masses in amu, and time in femtoseconds.
- **Element mapping** is derived automatically from the model `type_map`. Atoms
  whose element is absent from the type map raise a clear error.
- **Stress** requires a periodic cell. The Cauchy stress equals the virial
  divided by the cell volume.
- The shipped `pretrained.pt` is a 500-step smoke-test model, not a
  production-quality water potential; use it only to verify the workflow.
