---
name: matmaster-dpa4-workflows
description: Run prepared DPA4 training, fine-tuning, inference, and LAMMPS cases on MatMaster/Bohrium. Use when an agent must stage persistent inputs, resolve Bohrium projects/images/GPU resources, generate and validate a job specification, submit or monitor jobs, collect outputs, recover checkpoints or restarts, and record platform provenance. Delegate scientific setup and acceptance to the installed DeePMD-kit skills.
compatibility: Install with deepmd-train, deepmd-finetune-dpa4, deepmd-python-inference, and lammps-deepmd. Requires Python 3; production submission requires a MatMaster platform tool or a Bohrium CLI compatible with 2.5.17.
license: LGPL-3.0-or-later
metadata:
  author: MatMaster
  version: '1.0'
  repository: https://github.com/deepmodeling/deepmd-kit
---

# Run DPA4 workflows on MatMaster

Act only as the MatMaster/Bohrium platform layer. Own storage, resource
resolution, job packaging, submission, monitoring, collection, and retry
lineage. Do not own model architecture, training parameters, checkpoint/head
selection, inference semantics, LAMMPS input design, or scientific acceptance.

## Route scientific work

- Use `deepmd-train` for a new DPA4 training case.
- Use `deepmd-finetune-dpa4` for standard or LoRA fine-tuning.
- Use `deepmd-python-inference` for Python inference or `dp test`.
- Use `lammps-deepmd` for deployment, minimization, and MD.

Require the owning skill to return a prepared case contract. If that skill is
unavailable, install or reload the complete DeePMD-kit skill set instead of
reconstructing its instructions here.

Read references only when needed:

- [workflow-handoffs.md](references/workflow-handoffs.md): required inputs,
  outputs, and acceptance boundary for each workflow;
- [matmaster-operations.md](references/matmaster-operations.md): storage,
  resources, Bohrium CLI fallback, lifecycle, and recovery.

## Execute

### 1. Accept a prepared case

Require a relative entry command, normally `bash run.sh`, plus all referenced
files or durable paths, checksums, runtime/resource requirements, declared
outputs, smoke-test criteria, and checkpoint/restart policy. Do not change the
scientific command to fit an available resource.

### 2. Inventory and stage

Run:

```bash
python scripts/check_environment.py --case-dir CASE_DIR
```

Use `--require-file`, `--require-deepmd`, and `--require-lammps` as required by
the handoff. Keep durable inputs and results under `/personal` or `/share`.
Use relative paths inside job packages; the Bohrium working directory is not a
stable absolute path.

### 3. Resolve execution resources

Discover a real project ID, full image address, current machine type, wall
time, and output destination. Never reuse a historical project, image tag, GPU
SKU, or case count as a default. Prove image/data/output compatibility with the
bounded smoke test supplied by the owning skill.

### 4. Build and validate the job

Generate a concrete job specification:

```bash
python scripts/make_job_spec.py CASE_DIR \
    --output CASE_DIR/job.json \
    --project-id PROJECT_ID \
    --image FULL_IMAGE_ADDRESS \
    --machine MACHINE_TYPE \
    --name JOB_NAME
```

Preview submission, then use the installed platform tool or the native Bohrium
CLI dry run. Submit only after validation and explicit authorization. Stop if
neither a submission tool nor a compatible `bohr` executable exists.

### 5. Monitor and collect

Reconcile every job/group ID with `assets/bohrium-job/manifest.json`. Download
finished and failed cases, retain partial checkpoints/restarts, and run a
filesystem/log first pass:

```bash
python scripts/audit_cases.py ROOT --json REPORT.json
```

Add `--mode` and repeat `--require` for declared outputs. A scheduler success
or file-audit pass is not scientific acceptance; return artifacts to the owning
DeePMD skill.

### 6. Recover the smallest subset

Classify packaging, path, image, quota, queue, resource, interruption, and
wall-time failures here. Route training divergence/configuration, checkpoint
validity, inference correctness, model export, type mapping, LAMMPS input, and
numerical failures back to the owning skill. Retry only invalid cases and set
`retry_of`; never overwrite evidence from an earlier attempt.

## Return the platform ledger

Return persistent paths, project/image/machine selections, case/job/group and
attempt mappings, expected/submitted/downloaded/platform-valid counts, retry
lineage, logs, declared outputs, audit report, and unresolved platform risks.
Keep the platform verdict separate from training, inference, MD, and scientific
verdicts.
