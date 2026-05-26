# LAMMPS + DeePMD-kit Reference Notes

This reference expands the main skill with practical operating guidance.

## Practical rules for agents

1. Prefer small, explicit input scripts over clever but opaque templates.
1. Explain every command in the example script, because many users treat the example as a starting point for their own production run.
1. If the user asks to run a simulation, always confirm the structure file and DeePMD model file before execution.
1. If the user asks for offline execution, ask which exact LAMMPS command should be used instead of guessing.
1. If the user only asks for a template, do not overcomplicate it with advanced computes or fixes unless they are needed.

## Suggested smoke test strategy

Before a long production run, consider a short test such as:

```lammps
run 100
```

This helps catch obvious issues such as:

- missing model file
- unsupported pair style in the local LAMMPS build
- malformed data file
- missing per-type masses in the data file or input script
- immediate numerical instability

Then replace the short run with the intended production length.

## Typical files in a DeePMD-LAMMPS job

- `input.lammps`: input script
- `data.system`: atomic structure and box
- `graph.pb` or `graph_compressed.pb`: DeePMD model
- `log.lammps`: main textual log
- `traj.lammpstrj`: trajectory output

## Caution points

- The correct timestep depends on the physical system and the DeePMD model quality.
- Ensure every atom type has a mass, either in the LAMMPS data file `Masses` section or via explicit `mass` commands after `read_data`.
- `velocity ... create ...` should usually not be repeated when continuing from a restart.
- NPT settings need physically sensible damping constants; avoid copying values blindly.
- Some local LAMMPS builds may support DeePMD under slightly different package configurations. Check `lmp -h` if unsure.
