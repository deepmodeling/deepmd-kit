# MatMaster/Bohrium platform validation

## Contents

1. Acceptance boundary
1. Platform and file checks
1. Batch and attempt checks
1. Failure routing
1. Minimum ledger

## Acceptance boundary

This skill accepts only the platform portion of a case:

1. scheduler identity and terminal state are reconciled with the manifest;
1. submitted inputs and runtime metadata are traceable;
1. declared results and failure evidence are recovered;
1. batch counts and retry lineage are internally consistent.

Training convergence and checkpoint selection, inference accuracy, model
deployment, atom mapping, ensemble behavior, minimization/MD validity, and
scientific interpretation belong to the owning DeepMD skill or a downstream
analysis workflow. Do not mark a workflow scientifically valid from this
platform verdict.

## Platform and file checks

- Reconcile manifest case IDs with unique submitted jobs and downloads.
- Preserve stdout/stderr, application logs, image/runtime record, input
  checksums, job metadata, and all outputs declared by the prepared case.
- Verify declared trajectories, restart/checkpoint files, final structures, and
  analysis inputs were downloaded when the case contract requests them.
- Treat scheduler `Finished` without declared artifacts as invalid.
- Run `audit_cases.py` only as a bulk first pass. Its verdict is neither
  numerical nor MD-quality acceptance.

## Batch and attempt checks

- Expected count = unique prepared count = submitted unique count =
  platform-valid + platform-invalid/excluded.
- Every parameter/replica value appears exactly as declared.
- Every attempt records case ID, job ID, group ID, source attempt, reason, and
  changed platform field.
- Retry groups contain only invalid cases and link to original job IDs.
- Preserve the checkpoint or restart selected by the owning skill; do not
  regenerate, replace, or reinterpret it at the platform layer.
- Write continued outputs to a new attempt directory; never overwrite prior
  logs or downloaded evidence.
- Compare performance only after separating queue, image-pull, and one-time
  initialization overhead from the measured run.

## Failure routing

Handle here:

- missing package files, wrong relative paths, output declaration errors;
- scheduling, quota, image pull/cache, node interruption, wall-time, and
  resource allocation failures;
- job/group identity mismatches and incomplete downloads.

Route to the owning DeepMD skill:

- training input/data/branch, divergence, metric, or checkpoint-validity issues;
- inference artifact/input/output or accuracy issues;
- model export/load, atom mapping, LAMMPS input, or ensemble errors;
- NaN/Inf, lost atoms, nonconvergence, unstable integration, truncated stages,
  or restart-state validity.

## Minimum ledger

| Case    | Attempt | Job/group | Terminal | Files     | Platform verdict     | Evidence                |
| ------- | ------- | --------- | -------- | --------- | -------------------- | ----------------------- |
| case_id | 0       | IDs       | state    | pass/fail | valid/retry/excluded | concise paths/log lines |
