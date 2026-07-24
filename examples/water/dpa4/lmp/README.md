# LAMMPS example for DPA4 / SeZM

This directory contains a minimal end-to-end pipeline for running a
DPA4 model in LAMMPS via `pair_style deepmd`. DPA4 and SeZM refer to the
same PyTorch implementation; DPA4 is the DPA-series user-facing name.

## Files

| File            | Description                                                                           |
| --------------- | ------------------------------------------------------------------------------------- |
| `input.json`    | Demo DPA4/SeZM trained for 500 HybridMuon steps on `examples/water/data/data_{0..3}`. |
| `pretrained.pt` | Checkpoint produced from `input.json` for the LAMMPS smoke test.                      |
| `in.lammps`     | 20-step NVT run at 330 K on 192 water molecules.                                      |
| `water.lmp`     | LAMMPS data file (192-atom liquid water cell).                                        |

The frozen `.pt2` archive is not included because AOTInductor packages
are target-specific: they depend on the host's CPU/GPU, GPU compute
capability, and libtorch version. Freeze locally before running.

## Usage

Optionally retrain:

```bash
dp --pt train input.json --skip-neighbor-stat
```

Freeze the checkpoint (the pt backend detects DPA4 / SeZM and writes a
`.pt2` archive automatically):

```bash
dp --pt freeze -c model.ckpt.pt -o frozen_model
```

To use the shipped smoke-test checkpoint instead of retraining, replace
`model.ckpt.pt` with `pretrained.pt`.

Run the MD:

```bash
lmp -in in.lammps
```

The run should load the `.pt2` archive with a cutoff of 6 Å and two atom types,
then complete 20 steps with finite thermodynamic values. Exact values depend on
the trained checkpoint.

```
load model from: frozen_model.pt2 to gpu 0
  rcut in model:      6
  ntypes in model:    2
```

## Notes

- `pair_coeff * * O H` pins LAMMPS atom types 1 and 2 to `type_map`
  entries `"O"` and `"H"` respectively. When the element names are
  omitted, the mapping falls back to the `type_map` order stored in
  the `.pt2` metadata.
- `atom_modify map yes` keeps the ghost / periodic-image to local-atom
  mapping explicit for `.pt2` graph inference. GNN-style `.pt2` models
  fail fast when this atom map is required but absent.
- The 500-step `pretrained.pt` is intended as a smoke test, not a
  physically accurate water potential. Retrain with a longer schedule
  for production.
