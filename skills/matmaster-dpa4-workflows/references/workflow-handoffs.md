# DeepMD workflow handoffs for MatMaster

## Contents

1. Common contract
1. Training and fine-tuning
1. Inference
1. LAMMPS execution
1. Handoff back to the owning skill

## Common contract

Before platform staging, require the owning DeepMD skill to provide:

- workflow kind and owning skill;
- relative entry command and all files or persistent paths it reads;
- input/model/checkpoint checksums and branch/case identity;
- runtime, accelerator, memory, node/rank, and wall-time requirements;
- declared incremental log, checkpoints/restarts, final outputs, and acceptance
  owner;
- smoke-test contract and retry/resume constraints.

MatMaster/Bohrium may choose among currently available resources that satisfy
this contract. It must not alter scientific parameters to fit a resource.
Materialize the exact prepared command in a short fail-fast `run.sh` using
relative package paths and explicit log/output locations. Do not reconstruct or
edit the scientific command while wrapping it for Bohrium.

## Training and fine-tuning

The `deepmd-train` or `deepmd-finetune-dpa4` handoff must establish the complete
training input, data split and paths, starting checkpoint/head when applicable,
entry command, checkpoint cadence, metric/log files, and scientific acceptance.

Platform rules:

- Keep large reusable datasets and checkpoints under durable `/personal` or
  `/share` storage when the job runtime can access those mounts. Otherwise use
  a documented dataset/package mechanism; do not silently copy partial data.
- Record dataset paths plus a manifest/checksum appropriate to their size.
- Run a bounded representative batch before the full job to prove image,
  accelerator, memory, data access, and checkpoint writing.
- Declare the training log, learning-curve/metric files, periodic checkpoints,
  selected/final checkpoint outputs, resolved input, and runtime record as job
  outputs.
- Monitor platform progress and file growth without choosing a best checkpoint
  or interpreting loss curves.
- After download, a platform/file first pass may use
  `python scripts/audit_cases.py ROOT --mode train --require lcurve.out --require 'ckpt/*' --json REPORT.json`;
  adjust required paths to the handoff contract. A pass is not training
  acceptance.
- On interruption or wall time, preserve all checkpoints. Ask the owning skill
  which checkpoint and command form are valid for resume; do not restart from
  scratch or change a branch implicitly.

For sweeps, use one stable case ID per parameter set and record each attempt.
Do not let an automatic reschedule run a non-resume-aware training command from
the beginning over existing outputs.

## Inference

The `deepmd-python-inference` handoff must establish the artifact, input data,
type mapping, entry command, batch/resource requirements, and declared
predictions or test outputs.

Use Bohrium when local/session resources are insufficient or batch isolation is
valuable. Preserve input ordering and case identity. Return predictions, logs,
runtime metadata, and failures without interpreting model accuracy here.

## LAMMPS execution

The `lammps-deepmd` handoff must establish the deployment artifact, type mapping,
LAMMPS inputs, entry command, runtime/rank requirements, output cadence,
restart policy, and simulation acceptance.

Stage and execute that contract without rewriting the ensemble, timestep,
mapping, or restart state. Preserve all terminal outputs, including failed-job
logs and partial restarts, then return them to `lammps-deepmd`.

## Handoff back to the owning skill

Return:

- workflow/case/attempt identity and job/group IDs;
- project, full image address/digest, machine type, node/rank request, and
  runtime versions;
- input and persistent-storage paths plus checksums/manifests;
- scheduler history, exit evidence, logs, and every declared output path;
- checkpoint/restart lineage and the exact platform fields changed on retry;
- platform verdict and unresolved platform risks.

The owning skill determines training, inference, MD, and scientific validity.
