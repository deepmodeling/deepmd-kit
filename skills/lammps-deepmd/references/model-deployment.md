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

Freeze a DPA4/SeZM checkpoint with the standard PyTorch command:

```bash
dp --pt freeze -c model.ckpt.pt -o frozen_model
```

The backend detects DPA4/SeZM and writes `frozen_model.pt2`. Validate the archive
in the actual target environment. For a multi-task checkpoint, select the head
during export:

```bash
dp --pt freeze -c model.ckpt.pt -o frozen_model --head SELECTED_BRANCH
```

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

LAMMPS atom types are local integer IDs. Model types are ordered element names.
Build an explicit mapping instead of assuming the numbers are interchangeable.
For example:

```lammps
mass            1 15.999
mass            2 1.008
pair_coeff      * * O H
```

Here LAMMPS type 1 maps to `O` and type 2 maps to `H`. Require that:

- every LAMMPS atom type has a mass;
- every mapped element is supported by the model type map;
- the `pair_coeff` element order matches LAMMPS type order;
- the structure's atom and species counts are unchanged during conversion.

An implicit `pair_coeff * *` is acceptable only when the model and LAMMPS type
orders have been verified to match. Prefer explicit element names for auditable
workflows.

## Pre-production validation

1. Confirm the model loads without format or backend errors.
1. Treat `.pt2` as a device/toolchain-compiled artifact. Verify that the export
   and final LAMMPS runtime use compatible PyTorch, DeePMD-kit, device, and C++
   interfaces.
1. For multi-rank execution, verify the archive's communication metadata and
   nested with-comm artifact before launching MPI.
1. Run `run 0` or a bounded short run before production.
1. Require finite energy, force, pressure, and temperature where applicable.
1. Check atom count, masses, element mapping, box, and units when values are
   anomalous or atoms are lost.
1. Preserve the generated LAMMPS input, model path, command, log, and exit code.

## References

- [DPA4 export and LAMMPS](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html)
- [DeePMD-kit LAMMPS commands](https://docs.deepmodeling.com/projects/deepmd/en/latest/third-party/lammps-command.html)
