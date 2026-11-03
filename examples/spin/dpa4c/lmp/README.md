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

Train with the configuration in `../input.json` and freeze to a `.pt2` archive.
The archive is target-specific and is not shipped, so freeze locally:

```bash
dp --pt-expt train ../input.json
dp --pt-expt freeze -c model.ckpt.pt -o frozen_model --lower-kind graph
```

`dpa4spin` accepts either a graph-lower archive, as frozen above, or the
compressed archive that `dp --pt-expt compress` produces and that runs the
fused CUDA kernels. See
[model compression](../../../../doc/model/dpa4c.md#model-compression) for that
workflow, and point `pair_style` at the compressed archive if you follow it.

Run on the host:

```bash
lmp -in in.lammps
```

Under Kokkos the graph and the moments stay in device memory for the whole
step, and `dpa4spin/kk` reads only the compressed archive, so `pair_style` has
to point at that archive. Run device-resident with:

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

By default the pair style stores the precession magnetic force that the LAMMPS
SPIN package expects, the negative energy gradient `-dE/dS` scaled by the
moment magnitude over `hbar`. The `fix nve/spin` of a stock LAMMPS build does
not use this force: the fix evaluates the magnetic force itself, through the
pair styles whose names begin with `spin`, and `dpa4spin` and `dpa4spin/kk` are
not among them. With no such pair style, the fix runs silently on a single MPI
rank and stops with a sectoring error on several. Spin dynamics therefore needs
a LAMMPS build whose integrator reads the magnetic force of the pair style.

A Newtonian integrator of the Cartesian spin vector needs `-dE/dS` itself
rather than the precession force. Appending `magnetic_force cartesian` to the
`pair_style` line selects `-dE/dS` and requires `units metal`; the pair style
declares the selected convention through
`Pair::extract("fm_cartesian_force", dim)`.
