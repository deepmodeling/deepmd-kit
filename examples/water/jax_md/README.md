# JAX-MD water example

This example runs a short JAX-MD NVE smoke simulation using a DeePMD JAX
checkpoint and the same 192-atom water configuration used by the LAMMPS example
in `../lmp/water.lmp`.

It is intentionally small so it can be used as an integration check. The
JAX-MD run itself is short; the checkpoint should be produced from the existing
`../se_e2_a` water training directory or supplied with `--model`. The script
uses dpdata to read the LAMMPS data file.

## Train a JAX checkpoint

Reuse the existing `se_e2_a` training input:

```bash
cd ../se_e2_a
dp --jax train input.json --skip-neighbor-stat
```

This writes `model.ckpt.jax`, a stable checkpoint pointer to the latest
checkpoint directory. The checked-in `../se_e2_a/input.json` is a full training
example; for a quick integration check, use an existing checkpoint or make a
scratch copy with smaller `training.numb_steps` and `training.save_freq`.

## Run JAX-MD

```bash
cd ../jax_md
python run_jax_md.py --model ../se_e2_a/model.ckpt.jax --steps 10
```

The script prints the JAX backend/device, neighbor-list shape, and a small
thermo table. It uses:

- `jax_md.space.periodic` for the periodic cubic water box,
- `jax_md.partition.neighbor_list` for a dense JAX-MD neighbor list,
- `deepmd.jax.jax_md.as_jax_md` to adapt the DeePMD checkpoint to a JAX-MD
  potential,
- `dpdata.System` to load the LAMMPS water data file.

The default timestep, temperature, and random seed follow `../lmp/in.lammps`
(`0.0005`, `330 K`, `23456789`). Masses are taken from the LAMMPS input
(`O=16`, `H=2`) and converted to `eV ps^2 / A^2` for metal-style units.
