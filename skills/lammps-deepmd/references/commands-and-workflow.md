# LAMMPS + DeePMD-kit Reference Notes

This reference expands the main skill with practical operating guidance.

## Practical rules for agents

1. Prefer small, explicit input scripts over clever but opaque templates.
1. Explain every command in the example script, because many users treat the example as a starting point for their own production run.
1. If the user asks to run a simulation, always confirm the structure file and DeePMD model file before execution.
1. Ask which exact LAMMPS command, module, container, or source-built runtime should be used instead of guessing or installing one silently.
1. Keep shell environment variables and LAMMPS variables distinct; pass values
   with an explicit LAMMPS mechanism instead of copying shell syntax into input.
1. Keep a vacuum or nonperiodic slab axis fixed; do not barostat that direction
   unless the scientific task explicitly requires changing it.
1. If the user only asks for a template, do not overcomplicate it with advanced computes or fixes unless they are needed.

## Suggested canary strategy

Before a long production run, stage validation as:

```text
run 0 -> short NVE when physically appropriate -> short requested ensemble -> production
```

Check the complete LAMMPS exit code and log at every stage. A canary passes only
when the model loads, thermodynamic values are finite, the atom count is stable,
and early temperature, pressure, and controlled variables are physically
compatible with the initial state. Exit code zero alone is insufficient.

This catches obvious issues such as:

- unsupported model artifact or pair style in the selected runtime;
- malformed data headers, boxes, coordinates, or triclinic tilt factors;
- missing masses or inconsistent element/type mapping;
- immediate numerical instability or lost atoms.

## Typical files in a DeePMD-LAMMPS job

- `input.lammps`: input script
- `data.system`: atomic structure and box
- a supported DeePMD deployment artifact such as `.pb`, `.pth`, or DPA4/SeZM `.pt2`; see `model-deployment.md`
- `log.lammps`: main textual log
- `traj.lammpstrj`: trajectory output

## Caution points

- The correct timestep depends on the physical system and the DeePMD model quality.
- The first line of a LAMMPS data file is a skipped title; put header counts after it.
- Ensure every atom type has a mass, either in the LAMMPS data file `Masses` section or via explicit `mass` commands after `read_data`.
- `velocity ... create ...` should usually not be repeated when continuing from a restart.
- NPT settings need physically sensible damping constants; avoid copying values blindly.
- Some local LAMMPS builds may support DeePMD under slightly different package configurations. Check `lmp -h` if unsure.
