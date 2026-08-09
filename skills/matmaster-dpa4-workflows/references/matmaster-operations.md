# MatMaster and Bohrium operations

## Capability order

Use the first available capability:

1. installed MatMaster/Bohrium skill or platform tool;
1. an installed Bohrium CLI whose help matches the required operation;
1. explicit prerequisite installation or user/platform intervention.

This skill bundles no executable, credential, project, image, or GPU default.
Do not invent an OpenAPI route when a platform tool or compatible CLI is
missing. Use `python scripts/check_environment.py --probe --require-bohr` to
inventory the fallback without exposing secret values.

## Storage and job boundaries

| Layer                  | Use                                | Persistence                       |
| ---------------------- | ---------------------------------- | --------------------------------- |
| MatMaster session node | prepare, inspect, and post-process | system disk is not durable        |
| `/personal`            | user-scoped inputs and results     | durable                           |
| `/share`               | project-shared inputs and results  | durable, permission scoped        |
| Bohrium job workspace  | production execution               | temporary except declared outputs |

Do not assume that a session-node package exists in the job image. Do not save
mounted `/personal` or `/share` data into an image. Confirm project permission
before writing shared storage, and stop unused billable nodes according to the
accepted lifecycle policy.

## Images and resources

- Query current projects, full image addresses, and machine availability.
- Pin the image address and version/digest; a display name is insufficient.
- Select resources that satisfy the prepared handoff, then run its smoke test.
- Treat image-pull/cache startup as distinct from a hung calculation.
- Confirm project, quota, budget, wall time, and reschedule policy before submit.

## Package and submit

Keep the command in a short `run.sh`, use relative paths, write an incremental
log, and declare every result needed after the temporary workspace disappears.
Start from `assets/bohrium-job/job.json` or generate the resolved specification
with `scripts/make_job_spec.py`. Reject every remaining `__PLACEHOLDER__`.

For Bohrium CLI 2.5.17, the validation shape is:

```bash
bohr version
bohr job submit -i job.json \
    --input_directory ./case_dir/ \
    --dry-run --output json
```

Inspect `bohr job submit --help` when the installed interface differs. Remove
`--dry-run` only for the authorized production submission. Prefer a MatMaster
built-in submit tool when present.

Record the case ID, attempt, job/group IDs, project, image, machine, input
checksums, declared outputs, and `retry_of` in the manifest. A reschedule may
restart `run.sh` from the beginning; enable rescheduling only when the wrapper
is safely restart-aware.

## Monitor, collect, and recover

For each job group:

1. reconcile its members with the manifest;
1. inspect representative running and failed logs;
1. distinguish queue/image startup from execution progress;
1. download finished and failed terminal jobs;
1. run the file/log audit and return results to the owning skill;
1. retry only invalid cases in a new attempt directory.

`Finished` is a platform state, not proof of a valid model or simulation.
Preserve partial logs, checkpoints, and restarts before any retry.

Use terminate when recoverable outputs should normally remain. Use kill only
for an unrecoverable job after resolving its exact ID. Delete jobs or persistent
files only when explicitly requested and after verifying the target.

## Authentication

Let platform tools handle credentials when possible. For a CLI fallback, read
the access key from the injected environment (`BOHR_ACCESS_KEY`, or
`ACCESS_KEY` when required by that CLI version). Never print, hard-code, copy
into a job package, or return credentials, storage tokens, or node passwords.
