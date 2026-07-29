# LAMMPS example for DPA4C native spin

Runs a native-spin DPA4C model in LAMMPS through `pair_style dpa4spin`, and
through `dpa4spin/kk` under Kokkos. The magnetic moment enters the descriptor
as an equivariant input, so no virtual atoms are created and the atom count
equals the number of physical atoms. For the classical DeepSpin (virtual-atom)
scheme, see `examples/spin/lmp`; for DPA4 / SeZM native spin, see
`examples/spin/dpa4/lmp`.

## Files

| File        | Description                                                                                          |
| ----------- | ---------------------------------------------------------------------------------------------------- |
| `in.lammps` | Single-point evaluation, spin relaxation and lattice dynamics of a NiO supercell.                    |
| `init.data` | `atom_style spin` data: rocksalt NiO, 32 magnetic Ni + 32 O, in its type-II antiferromagnetic order. |

## Usage

Train with the configuration in `../input.json`, compress, and freeze. The
pair style requires the compact canonical graph lower, which `--lower-kind auto` selects for a compressed DPA4C; the archive is target-specific and is not
shipped, so freeze locally:

```bash
dp --pt-expt train ../input.json
dp --pt-expt compress -i model.ckpt.pt -o compressed.pt
dp --pt-expt freeze -c compressed.pt -o frozen_model --lower-kind auto
```

Run on the host:

```bash
lmp -in in.lammps
```

Run device-resident under Kokkos, where the graph and the moment stay in device
memory for the whole step:

```bash
lmp -k on g 1 -sf kk -in in.lammps
```

Either path runs under domain decomposition without any change. Ghost moments
arrive through the forward communication that `atom_style spin` already
performs, and the magnetic force is reduced onto owning atoms alongside the
conservative force:

```bash
mpirun -np 2 lmp -in in.lammps
```

`nio.dump` holds the per-atom moment (`c_spin[1..4]`), the magnetic force
(`c_spin[5..7]`), which is non-zero on Ni and exactly zero on O, and the
conservative force (`fx fy fz`).

The run has three stages. A single-point evaluation reports the energy and both
forces of the antiferromagnetic reference state. `min_style spin` then reads
the magnetic force from the pair style and relaxes the moment directions at
fixed positions. Finally `fix nvt` integrates positions at the relaxed magnetic
configuration, which exercises the conservative force, the neighbor rebuilds
and the domain decomposition; the thermostat does not touch the moments, so
they stay fixed through that stage.

Spin dynamics through `fix nve/spin` requires a LAMMPS build whose fix
recognizes this pair style, because the stock fix accumulates the magnetic
force only from pair styles matching its own name pattern.
