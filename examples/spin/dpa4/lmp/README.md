# LAMMPS example for DPA4 / SeZM native spin

Runs a native-spin DPA4 / SeZM model in LAMMPS through `pair_style deepspin`
backed by an AOTInductor `.pt2` archive. For the classical DeepSpin
(virtual-atom) scheme, see `examples/spin/lmp`.

## Files

| File        | Description                                               |
| ----------- | --------------------------------------------------------- |
| `in.lammps` | Single-point evaluation of a 4-atom NiO cell.             |
| `init.data` | `atom_style spin` data: 2 magnetic Ni + 2 non-magnetic O. |

## Usage

Train (configuration in `../input.json`) and freeze to a `.pt2` archive. The
freeze CLI detects DPA4 / SeZM and rewrites the suffix to `.pt2`; the archive is
target-specific and is not shipped, so freeze locally:

```bash
dp --pt train ../input.json --skip-neighbor-stat
dp --pt freeze -c model.ckpt.pt -o frozen_model
```

Run:

```bash
lmp -in in.lammps
```

`spin.dump` holds the per-atom force (`fx fy fz`) and magnetic force
(`c_fmag[1..3]`), which is non-zero on the magnetic Ni atoms and zero on O.

The same archive runs under domain decomposition without any change:

```bash
mpirun -np 2 lmp -in in.lammps
```
