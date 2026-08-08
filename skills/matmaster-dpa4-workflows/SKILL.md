---
name: matmaster-dpa4-workflows
description: Orchestrate DPA4 training, fine-tuning, inference, and LAMMPS molecular-dynamics workflows on MatMaster and Bohrium after the scientific run contract has been prepared with the DeepMD-kit skills. Use when an agent must stage datasets, models, or simulation cases in persistent storage; discover Bohrium projects, images, and GPU resources; generate and validate job specifications; submit or group jobs; monitor and download outputs; recover from checkpoints or restarts; classify platform failures; or produce an auditable run ledger. Includes CLI and read-only OpenAPI fallbacks when MatMaster built-in platform skills are absent.
compatibility: Designed to be installed with the DeepMD-kit skills deepmd-train, deepmd-finetune-dpa4, deepmd-python-inference, and lammps-deepmd. Requires Python 3. Bohrium operations require MatMaster built-ins, a Bohrium CLI compatible with 2.5.17, or BOHR_ACCESS_KEY for read-only discovery.
license: LGPL-3.0-or-later
metadata:
  author: MatMaster
  version: '1.0'
  repository: https://github.com/deepmodeling/deepmd-kit
---

# Orchestrate DPA4 workflows on MatMaster

Treat this as the MatMaster/Bohrium platform layer of the DeepMD-kit skill set.
It owns staging, resource discovery, submission, monitoring, collection,
checkpoint/restart transport, retry lineage, and platform acceptance. It does
not own model architecture, training or fine-tuning parameters, inference API
semantics, LAMMPS input design, ensemble choice, or scientific interpretation.

## Route to sibling skills

- Use `deepmd-train` to prepare training inputs, data splits, commands,
  checkpoint policy, and training acceptance.
- Use `deepmd-finetune-dpa4` to prepare standard or LoRA fine-tuning inputs,
  pretrained-checkpoint/head selection, commands, and acceptance.
- Use `deepmd-python-inference` to prepare inference or `dp test` inputs,
  commands, artifact routing, and result acceptance.
- Use `lammps-deepmd` to prepare model deployment, atom mapping, LAMMPS inputs,
  execution commands, ensembles, and MD/minimization acceptance.

Do not duplicate those instructions here. Accept their output as a prepared
case contract, then orchestrate it on MatMaster/Bohrium.
If the owning sibling skill is unavailable, stop and install/reload the complete
DeepMD-kit skill set instead of inventing the missing scientific workflow.

Read only the platform reference needed for the current action:

- session nodes, persistent storage, resources, jobs, groups, and lifecycle:
  [matmaster-operations.md](references/matmaster-operations.md)
- workflow-specific handoff, staging, outputs, and recovery contracts:
  [workflow-handoffs.md](references/workflow-handoffs.md)
- terminal-state, file, batch, and retry-lineage checks:
  [platform-validation.md](references/platform-validation.md)
- packaged CLI/OpenAPI adapters and job template:
  [bundled-tools.md](references/bundled-tools.md)

## Execute the platform workflow

### 1. Accept the prepared case contract

Classify the request as training, fine-tuning, inference, or LAMMPS execution,
then require one or more prepared case directories with:

- a relative entry command, normally `bash run.sh`;
- every dataset reference, checkpoint/model, structure/input, and auxiliary file
  required by that command;
- declared logs, checkpoints/restarts, metrics, predictions, trajectories,
  final artifacts, and other outputs;
- input checksums, expected case count, and any branch, replica, or sweep
  identity;
- runtime and resource requirements established by the owning sibling skill.

Do not reinterpret model architecture, data split, loss, optimizer, checkpoint
head, inference artifact, atom mapping, ensemble, timestep, or other scientific
parameters. Route an incomplete contract back to the owning sibling skill.
Use [workflow-handoffs.md](references/workflow-handoffs.md) to verify the handoff
and `assets/bohrium-job/manifest.json` as the starting ledger when no manifest
was supplied.

### 2. Stage on MatMaster

- Inspect available MatMaster platform tools, then run
  `python scripts/check_environment.py --case-dir CASE_DIR`.
- Use the session node for staging and post-processing, not as an implicit
  production runtime.
- Keep durable work under `/personal` or `/share`; do not rely on the session
  node system disk for irreplaceable inputs or results.
- Create a manifest and stable directory for every case. Keep large shared
  datasets/checkpoints in persistent storage when the accepted platform
  contract permits it; do not duplicate them into every job package.
- Use relative paths inside uploaded packages because Bohrium job work
  directories are generated dynamically.

### 3. Resolve Bohrium execution resources

- Resolve a real project ID, full image address, current machine type, storage
  destination, wall time, and reschedule policy.
- Require the selected image/resource to satisfy the owning skill's runtime
  contract; prove it with a bounded training, inference, or LAMMPS smoke case.
- Benchmark representative dataset batch size, inference batch, or MD atom
  count before increasing duration, concurrency, or replicas.
- Never promote a historical image, GPU SKU, project, path, or case count to a
  default.

### 4. Submit safely

- Prefer installed MatMaster/Bohrium platform tools for the exact operation.
- If they are absent, generate a spec with `scripts/make_job_spec.py`, use
  `scripts/bohr_cli.py` for CLI operations, and use
  `scripts/bohrium_readonly.py` only for read-only discovery.
- Preview mutations first. For submission, run `--validate` before the explicit
  `--execute` action.
- Stop if neither a submission tool nor the `bohr` CLI is available.
- Record case, job, group, attempt, and `retry_of` identity in the manifest.

### 5. Monitor and collect

- Reconcile the complete job group with the manifest before retrying anything.
- Treat scheduler states as platform states, not proof of a valid training,
  inference, or simulation result.
- Preserve logs and outputs from failed jobs and download every terminal case.
- Run `python scripts/audit_cases.py ROOT --json REPORT.json` as a bulk
  filesystem/log first pass, adding `--require PATTERN` for declared outputs.
- Hand downloaded artifacts back to the owning sibling skill for workflow
  acceptance: training/fine-tuning metrics and checkpoints, inference outputs,
  or LAMMPS logs/trajectories/restarts.

### 6. Recover the smallest subset

Handle packaging, path, output declaration, queue, image-cache, quota, resource,
interruption, wall-time, and scheduler failures here. Preserve a training
checkpoint or LAMMPS restart selected by the owning skill, but do not decide its
scientific validity. Route training divergence/configuration, checkpoint/head,
inference, model export, atom mapping, LAMMPS input, and numerical failures back
to the owning sibling skill. After it supplies a tested correction, resubmit
only invalid cases and preserve attempt lineage.

## Deliver the platform ledger

Return workspace and persistent paths, project/image/machine selections,
group/job/attempt mappings, expected/submitted/downloaded/platform-valid counts,
retry lineage, logs and result paths, the audit report, and unresolved platform
risks. Keep platform acceptance separate from training, inference, simulation,
and scientific acceptance.
