---
name: lammps-deepmd
description: >
  A tool and knowledge base for running molecular dynamics (MD) simulations in LAMMPS with the DeePMD-kit plugin. It handles input script preparation, ensemble selection (NVE/NVT/NPT), and job execution via `uv` or offline binaries.
  USE WHEN you need to set up, write, explain, or execute a LAMMPS molecular dynamics simulation using a DeePMD machine learning potential (e.g., `graph.pb`).
compatibility: Requires LAMMPS with DeePMD-kit support. Online mode prefers `uvx --from lammps --with deepmd-kit[gpu,torch,lmp] lmp`; offline mode requires a user-provided LAMMPS executable or module.
license: LGPL-3.0-or-later
metadata:
  author: OpenClaw
  version: '1.0'
  repository: https://github.com/deepmodeling/deepmd-kit
  lammps_docs: https://docs.lammps.org/
---

# LAMMPS + DeePMD-kit

Use this skill when the user wants to run molecular dynamics in LAMMPS with a DeePMD-kit potential, prepare or explain an `input.lammps` file, or switch between common ensembles such as NVE, NVT, and NPT.

## Agent responsibilities

1. Confirm the available execution mode:
   - **Online mode**: if internet access is available and `uv` is installed, prefer
     `uvx --from lammps --with deepmd-kit[gpu,torch,lmp] lmp ...`
   - **Offline mode**: do **not** guess the executable. Ask the user which LAMMPS command, module, or container should be used.
1. Confirm the minimum simulation inputs:
   - structure/data file (for example `data.system`)
   - DeePMD model file (for example `graph.pb` or compressed model)
   - atom type to element mapping, including required per-type masses if the data file does not define them
   - target ensemble (NVE, NVT, NPT, or another explicitly requested setup)
   - temperature, pressure if applicable, timestep, and total number of steps
1. Write the LAMMPS input script yourself instead of asking the user to hand-write it.
1. Keep the example readable and fully explained. If you include an example input script, explain what **every command** does.
1. When possible, validate command availability against the LAMMPS docs or local `lmp -h` output before execution.
1. Report clearly which command was run, which files were used, and where outputs were written.

## Decide the execution mode

### Online mode (preferred when internet access is available)

Use:

```bash
uvx --from lammps --with deepmd-kit[gpu,torch,lmp] lmp -in input.lammps
```

If you need to inspect the local command-line help:

```bash
uvx --from lammps --with deepmd-kit[gpu,torch,lmp] lmp -h | tee /dev/tty
```

Notes:

- This is the preferred path because it can provision LAMMPS and DeePMD-kit on demand.
- The `gpu,torch,lmp` extras match the requested runtime pattern from the user.
- If the environment is slow or the packages are large, warn the user that the first run may take time.

### Offline mode

If internet access is unavailable or the user explicitly wants a site-installed binary, ask a concrete question such as:

- "Which LAMMPS executable should I use, for example `lmp`, `lmp_mpi`, `mpirun -np 8 lmp`, or an HPC module command?"
- "Do you already have a DeePMD-enabled LAMMPS build on this machine or cluster?"

Do not invent a binary name or module name.

## Minimal information to collect

Ask only for what is missing:

- DeePMD model path
- LAMMPS data file path
- ensemble
- target temperature
- target pressure if using NPT
- timestep
- run length in steps
- whether velocities should be generated from scratch
- preferred execution command if offline

## Recommended workflow

1. Inspect available files in the working directory.
1. Draft `input.lammps`.
1. Explain the script to the user if they asked for an explanation or if the script is nontrivial.
1. Run a short smoke test first when reasonable.
1. Run the full simulation.
1. Summarize outputs such as `log.lammps`, dump trajectories, restart files, and thermodynamic data.

## Example: annotated NVT input

The following example is adapted from the user-provided tutorial pattern and slightly generalized. See also `assets/input.nvt.lammps`.

```lammps
variable        NSTEPS          equal 1000000
variable        THERMO_FREQ     equal 1000
variable        DUMP_FREQ       equal 1000
variable        TEMP            equal 300.0
variable        TAU_T           equal 0.1

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

read_data       data.system
mass            1 28.0855
mass            2 15.999
pair_style      deepmd graph_compressed.pb
pair_coeff      * *

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}
dump            1 all custom ${DUMP_FREQ} traj.lammpstrj id type x y z

velocity        all create ${TEMP} 743574
fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}

timestep        0.0005
run             ${NSTEPS}
```

### What every command means

- `variable NSTEPS equal 1000000`

  - Defines a numeric variable called `NSTEPS` with value `1000000`.
  - Used later by `run ${NSTEPS}` so the run length is easy to modify in one place.

- `variable THERMO_FREQ equal 1000`

  - Defines how often LAMMPS prints thermodynamic information.
  - Used by `thermo ${THERMO_FREQ}`.

- `variable DUMP_FREQ equal 1000`

  - Defines how often coordinates are written to the trajectory dump.

- `variable TEMP equal 300.0`

  - Sets the target temperature in the current unit system.
  - Because `units metal` is used below, this temperature is interpreted in kelvin.

- `variable TAU_T equal 0.1`

  - Sets the thermostat damping parameter used by the NVT fix.
  - In `metal` units this is in picoseconds.

- `units metal`

  - Selects the LAMMPS `metal` unit system.
  - This determines the physical meaning of timestep, temperature, pressure, energy, distance, and time.
  - In this unit system, distances are in angstrom, time is in picoseconds, and the timestep should be chosen accordingly.

- `boundary p p p`

  - Applies periodic boundary conditions in x, y, and z.
  - Suitable for bulk condensed-phase simulations.

- `atom_style atomic`

  - Uses the `atomic` atom style, appropriate when atoms have no explicit bonds, angles, or molecular topology in the force field description.
  - Common for DeePMD simulations of condensed phases when the structure is provided as atoms in a box.

- `neighbor 1.0 bin`

  - Sets the neighbor-list skin distance to `1.0` in the current distance unit.
  - Uses the `bin` neighbor-building method.
  - Neighbor lists help LAMMPS efficiently find nearby atoms for force evaluation.

- `read_data data.system`

  - Reads the initial atomic structure, atom types, simulation box, and related information from the LAMMPS data file `data.system`.
  - Replace this filename with the actual user file.

- `mass 1 28.0855`, `mass 2 15.999`

  - Defines per-type atomic masses when the data file does not contain a `Masses` section.
  - These example values correspond to a two-type Si/O mapping; adjust them to the actual atom type to element mapping. LAMMPS velocity creation and thermostats require masses; without them, runs can fail with `Not all per-type masses are set`.

- `pair_style deepmd graph_compressed.pb`

  - Selects the DeePMD pair style.
  - Loads the DeePMD model from `graph_compressed.pb`.
  - Replace the model filename with the actual model path, for example `graph.pb`, `graph-compress.pb`, or another supported exported model.

- `pair_coeff * *`

  - Activates the previously selected pair style for all atom types.
  - For DeePMD this often takes the simple form `* *` because the mapping is embedded in the model workflow rather than through conventional pairwise parameters.

- `thermo_style custom step temp pe ke etotal press vol lx ly lz xy xz yz`

  - Chooses exactly which thermodynamic quantities to print.
  - `step`: timestep index.
  - `temp`: instantaneous temperature.
  - `pe`: potential energy.
  - `ke`: kinetic energy.
  - `etotal`: total energy.
  - `press`: pressure.
  - `vol`: box volume.
  - `lx ly lz`: box lengths.
  - `xy xz yz`: triclinic tilt factors, which are harmless to print even for an orthogonal box.

- `thermo ${THERMO_FREQ}`

  - Prints the thermo block every `THERMO_FREQ` timesteps.

- `dump 1 all custom ${DUMP_FREQ} traj.lammpstrj id type x y z`

  - Creates dump ID `1`.
  - Dumps atoms from group `all`.
  - Uses the `custom` dump format.
  - Writes every `DUMP_FREQ` steps.
  - Saves to `traj.lammpstrj`.
  - Outputs per-atom columns `id type x y z`.

- `velocity all create ${TEMP} 743574`

  - Assigns random initial velocities to all atoms.
  - The target temperature is `TEMP`.
  - `743574` is the random seed.
  - Use this when starting a fresh MD trajectory. If restarting from a previous equilibrated state, this command may be unnecessary.

- `fix 1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}`

  - Creates fix ID `1` on group `all`.
  - Applies the Nose-Hoover NVT thermostat.
  - The target temperature is ramped from `${TEMP}` to `${TEMP}`, meaning constant temperature here.
  - `${TAU_T}` is the thermostat damping constant.

- `timestep 0.0005`

  - Sets the MD timestep.
  - In `metal` units, `0.0005` means `0.0005 ps = 0.5 fs`.
  - The safe choice depends on the system and model quality.

- `run ${NSTEPS}`

  - Runs molecular dynamics for `NSTEPS` timesteps.

## Common ensemble modifications

### NVE

Replace the NVT thermostat line with:

```lammps
fix 1 all nve
```

Meaning:

- integrates Newton's equations in the microcanonical ensemble
- no thermostat or barostat is applied
- useful for short stability checks or production runs after equilibration

### NPT

A typical isotropic NPT alternative is:

```lammps
variable        PRESS           equal 1.0
variable        TAU_P           equal 1.0
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRESS} ${PRESS} ${TAU_P}
```

Meaning:

- `PRESS` is the target pressure
- `TAU_P` is the barostat damping constant
- `iso` applies isotropic pressure control to the simulation box
- this simultaneously thermostats and barostats the system

When using NPT, it is often useful to keep `vol`, `lx`, `ly`, and `lz` in the thermo output.

## Execution templates

### Online run

```bash
uvx --from lammps --with deepmd-kit[gpu,torch,lmp] lmp -in input.lammps
```

### Online help

```bash
uvx --from lammps --with deepmd-kit[gpu,torch,lmp] lmp -h | tee /dev/tty
```

### Offline run

Only after the user specifies the executable, use a command such as one of these exact patterns:

```bash
lmp -in input.lammps
mpirun -np 8 lmp_mpi -in input.lammps
srun lmp -in input.lammps
```

The agent must not choose one of these on its own without user guidance in offline mode.

## Output checklist

After a run, report at least:

- executed command
- input script path
- data file path
- model path
- main log path
- trajectory path if any
- whether the run completed successfully
- any obvious warnings or errors from the log

## References

- LAMMPS command categories: https://docs.lammps.org/Commands_category.html
- LAMMPS command index: https://docs.lammps.org/Commands_all.html
- DeePMD-kit: https://github.com/deepmodeling/deepmd-kit
- User-provided tutorial reference: https://github.com/tongzhugroup/Chapter13-tutorial/blob/master/input.lammps
- Detailed notes: `references/commands-and-workflow.md`
