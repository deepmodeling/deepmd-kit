# DeePMD workflow handoffs

## Common contract

Before staging, require the owning DeePMD skill to provide:

- workflow kind, owning skill, case ID, and relative entry command;
- every input file or durable path plus checksums/manifests;
- image/runtime, accelerator, memory, node/rank, and wall-time requirements;
- incremental logs, checkpoints/restarts, final outputs, and output cadence;
- smoke-test contract, retry/resume constraints, and acceptance owner.

Wrap the exact accepted command in `run.sh`. Do not reinterpret scientific
configuration during packaging.

## Training and fine-tuning

`deepmd-train` or `deepmd-finetune-dpa4` owns the training input, data split,
starting checkpoint/head, checkpoint cadence, command, metrics, and scientific
acceptance.

- Reference large durable datasets/checkpoints rather than copying partial data.
- Prove data access, representative batch size, accelerator compatibility, and
  checkpoint writing with the supplied bounded smoke test.
- Preserve resolved inputs, logs, metrics, periodic checkpoints, runtime record,
  and all attempts.
- On interruption, return every checkpoint. The owning skill selects the valid
  resume checkpoint and command.

The platform layer may report progress and file growth, but must not choose the
best checkpoint or interpret convergence.

## Inference

`deepmd-python-inference` owns the model artifact, input ordering, type mapping,
command, batch size, declared predictions, and accuracy acceptance. Preserve
ordering and case identity; return predictions, logs, runtime metadata, and
failures without interpreting accuracy.

## LAMMPS

`lammps-deepmd` owns deployment, type mapping, LAMMPS inputs, ensemble,
timestep, ranks, output cadence, restart policy, and simulation acceptance.
Preserve trajectories, logs, final structures, and partial restarts without
rewriting those scientific choices.

## Return boundary

Return case/attempt/job/group identity, platform selections and versions,
input/output paths, scheduler/exit evidence, logs, retry lineage, and a platform
verdict. The owning skill determines training, inference, MD, and scientific
validity.
