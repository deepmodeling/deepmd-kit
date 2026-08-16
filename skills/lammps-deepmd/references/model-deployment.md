# DeePMD model deployment in LAMMPS

Read this reference when choosing a model artifact, exporting a checkpoint, or
mapping LAMMPS atom types to model elements. General simulation setup and
execution remain in `commands-and-workflow.md`.

## Choose the model artifact

A training checkpoint is not automatically a LAMMPS deployment artifact. The
`.pt` suffix also does not distinguish DPA3 from DPA4. Inspect an unfamiliar
PyTorch checkpoint before choosing an export path:

```bash
dp --pt show model.pt descriptor fitting-net type-map
```

| Model artifact                     | Deployment route                                                                  |
| ---------------------------------- | --------------------------------------------------------------------------------- |
| TensorFlow frozen `.pb`            | Use directly with a compatible `pair_style deepmd`.                               |
| Conventional PyTorch frozen `.pth` | Use directly with a compatible DeePMD-enabled LAMMPS build.                       |
| PyTorch checkpoint `.pt`           | Inspect the stored model configuration and freeze using its model-specific route. |
| AOTInductor archive `.pt2`         | Inspect its metadata and use only with a compatible DeePMD-enabled LAMMPS build.  |

Do not call a DPA4 `.pt2` archive a compressed model: DPA4 does not support
`dp compress`.

## DPA4/SeZM deployment

Before export, read
`../../deepmd-python-inference/references/dpa4-freeze-policy.md` and explicitly
choose the freeze-time inference environment. Then freeze a DPA4/SeZM checkpoint
with the standard PyTorch command:

```bash
dp --pt freeze -c model.ckpt.pt -o frozen_model
```

The backend detects DPA4/SeZM and writes `frozen_model.pt2`. Validate the archive
in the actual target environment. For a multi-task checkpoint, select the head
during export:

```bash
dp --pt freeze -c model.ckpt.pt -o frozen_model --head SELECTED_BRANCH
```

Create and consume the archive on the same target physical compute node and
allocation: inspect the native checkpoint -> freeze `.pt2` -> `run 0` -> bounded
MD -> production. Do not freeze in job or node A and move the archive to B unless
portability has been independently validated for that exact device and toolchain.

Two DPA4 `.pt2` export contracts exist. `dp --pt freeze` uses the DPA4-specific
`edge_vec` ABI. `dp --pt_expt freeze --lower-kind graph` uses the NeighborGraph
ABI. They share a suffix but are not interchangeable contracts, and a `.pt2`
suffix alone does not prove multi-rank support. A multi-rank archive must report
`has_comm_artifact=true` and contain
`model/extra/forward_lower_with_comm.pt2`.

A basic energy-model input uses:

```lammps
atom_style      atomic
atom_modify     map yes
read_data       data.system

pair_style      deepmd frozen_model.pt2
pair_coeff      * * O H
```

`atom_modify map yes` must appear before `read_data` for the documented DPA4
route. Ordinary DPA4 energy models use `pair_style deepmd`; spin models may
require a different documented route and must not be treated as ordinary energy
models without inspection.

Single-rank DPA4 execution is covered for supported `edge_vec`, graph, and
dense/nlist archives. Multi-rank execution is supported only when the archive
contains the with-communication artifact required by its ABI; fail closed when
that metadata or nested artifact is absent.

## Atom-type mapping

LAMMPS atom types, dataset type indices, and model types are separate namespaces.
Inspect the artifact's ordered type map, for example with
`dp --pt show model.pt type-map`, and treat element identity as the bridge.
For DeePMD data with `type_map.raw`, decode each zero-based `type.raw` index
through that ordered map. Without `type_map.raw`, require provenance that dataset
indices already follow the candidate model's ordered type map. Fail closed when
neither contract is established; do not reuse a dataset integer as a LAMMPS type.

Use compact one-based LAMMPS types for the elements present in the structure and
write the same element order in masses, `pair_coeff`, and dump metadata:

```lammps
mass            1 15.999
mass            2 1.008
pair_coeff      * * O H
dump            1 all custom 100 traj.lammpstrj id type element x y z
dump_modify     1 element O H sort id
```

Here LAMMPS type 1 maps to `O` and type 2 maps to `H`. `dump_modify ... element`
labels each local type with that same mapping, while `sort id` gives a stable
per-frame atom order. Do not sort atoms by element or model type-map position.
Require that:

- every LAMMPS atom type has a mass;
- every mapped element is supported by the inspected model type map;
- `pair_coeff`, masses, and dump element labels share one LAMMPS type order;
- the structure's atom and species counts are unchanged during conversion.

An implicit `pair_coeff * *` is acceptable only when the model and LAMMPS type
orders have been verified to match. Prefer explicit element names for auditable
workflows.

## LAMMPS data and box checks

- The first line of a LAMMPS data file is a title and is skipped by `read_data`;
  place the actual header counts after it.
- Put `atom_modify map yes` before `read_data` for the documented DPA4 route.
- For a restricted triclinic box, preserve the tilt mapping `xy = b_x`,
  `xz = c_x`, and `yz = c_y`; never write `c_z` or `lz` into `yz`.
- After conversion, compare atom count, species counts, and box volume with the
  source structure before running dynamics.

## Pre-production validation

1. Confirm the model loads without format or backend errors.
1. Keep DPA4 freeze, `run 0`, canary, and production on the same target physical
   compute node and allocation unless exact artifact portability is proven.
1. For multi-rank execution, verify the archive's communication metadata and
   nested with-comm artifact before launching MPI.
1. Stage `run 0` -> short NVE when physically appropriate -> short requested
   ensemble -> production; do not jump from a successful load to a long run.
1. Require finite thermodynamics, stable atom count, and no mapping, box, or
   lost-atom errors. Exit code zero alone is not a passed canary.
1. Require early temperature, pressure, and controlled variables to remain
   physically compatible with the initial state and requested ensemble.
1. Preserve the generated data and input files, model path and SHA256, runtime
   identity, command, complete log, and true exit code.

## References

- [DPA4 export and LAMMPS](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html)
- [DeePMD-kit LAMMPS commands](https://docs.deepmodeling.com/projects/deepmd/en/latest/third-party/lammps-command.html)
