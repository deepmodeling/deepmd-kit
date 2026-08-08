# MatMaster and Bohrium platform operations

## Contents

1. Platform model
1. Capability priority
1. Session node and storage
1. Images and resources
1. Job packaging and submission
1. Monitoring, collection, and lifecycle
1. Authentication and safety

## Platform model

Keep four layers distinct:

| Layer                  | Purpose                                                | Persistent?                                                     |
| ---------------------- | ------------------------------------------------------ | --------------------------------------------------------------- |
| MatMaster session node | prepare inputs, debug, inspect, and analyze            | system disk: no guarantee; mounted data disks: yes              |
| `/personal`            | user-scoped durable files                              | yes                                                             |
| `/share`               | project-shared durable files                           | yes, project permissions apply                                  |
| Bohrium container job  | isolated production compute with chosen image/resource | job workspace is temporary; declared/downloaded outputs persist |

The session node and submitted job are different resources with different lifecycles and billing. Do not assume a package installed on the session node exists inside the job image.

## Capability priority

Use this order:

1. MatMaster built-in skill/tool for the exact operation;
1. bundled helper scripts over an installed `bohr` CLI;
1. bundled read-only OpenAPI probe for discovery;
1. raw CLI/OpenAPI only when the packaged helpers lack the required operation.

Expected MatMaster platform capability families:

| Need                           | Capability                                              |
| ------------------------------ | ------------------------------------------------------- |
| prepare/inspect files          | workspace shell and file tools                          |
| submit a job or batch          | built-in `bohrium-submit` skill                         |
| list/query jobs and groups     | Bohrium query/list action                               |
| view/download logs and results | Bohrium log/download action                             |
| stop wrong work                | Bohrium terminate/kill action after resolving exact IDs |
| list machine/SKU availability  | Bohrium resource-list action                            |
| track long execution           | planning/todo and monitoring capability                 |

Discover the installed schemas at runtime. Historical names are evidence, not an API contract.

## Session node and storage

- Use the MatMaster node for editing, preprocessing, smoke inputs, and post-processing.
- Store durable models, structures, manifests, and downloaded results in `/personal` or `/share`.
- Use `/share` for project collaboration and `/personal` for user-private durable work.
- Confirm project and permissions before writing `/share`.
- Do not embed `/personal` or `/share` data into a saved image; Bohrium images save the system environment, not mounted data disks.
- Stop an unused development/session node according to the user's lifecycle policy because a started node continues billing.

For direct file API work, use the Bohrium file capability. Personal/share storage uses the v1 file routes; appJob workspace v2 routes are a different namespace.

## Images and resources

- Jobs are container jobs; use the full registry URL returned by image listing.
- Query available public/private images instead of guessing tags.
- Pin an image that satisfies the runtime contract and smoke test supplied by
  the owning DeepMD skill.
- Record image URL and digest/version. A display name is insufficient.
- Expect custom-image cache creation or refresh to take minutes; a long initial wait may be image pulling rather than job failure.
- Container nodes do not support running Docker inside them. Build/save custom images through Bohrium-supported image workflows.
- Query current SKU availability and price when resource choice matters. Do not hard-code a historical GPU SKU as universally available.
- Treat project visibility as discovery, not spending authorization. Require the
  user or accepted workspace contract to select the billing project and confirm
  applicable quota/budget before submission.
- Use the resource class required by the prepared case after compatibility
  proof; use CPU/session-node work for lightweight staging when appropriate.

## Job packaging and submission

Bohrium expands the uploaded input into an unpredictable work directory. Use relative paths and a short entry command:

```json
{
  "job_name": "dpa4-workflow-case-0001",
  "command": "bash run.sh",
  "log_file": "run.log",
  "backward_files": [
    "run.log",
    "logs/",
    "results/",
    "provenance.json"
  ],
  "project_id": 123,
  "machine_type": "CURRENT_VALID_GPU_SKU",
  "image_address": "registry.dp.tech/.../validated-dpa4:tag",
  "job_type": "container",
  "result_path": "/share",
  "max_reschedule_times": 1,
  "max_run_time": 1440,
  "nnode": 1
}
```

Rules:

- never `cd /root/input` or another guessed absolute job path;
- package every input declared by the prepared case contract, its relative run
  wrapper, and provenance with each self-contained case or a documented shared
  mechanism;
- keep commands in `run.sh`; complex inline shell can trigger WAF rejection;
- set `log_file` to a real incrementally written file;
- declare `backward_files`, result path, runtime, reschedule policy, and node count;
- make long-run wrappers checkpoint-aware; `max_reschedule_times` may rerun the command from the beginning unless the wrapper explicitly finds and validates a restart;
- inspect input-package size and hidden files before submission;
- use a manifest as the join key for case ID, job ID, group ID, inputs, outputs, and retry.

Example CLI shape:

```bash
bohr job submit -i job.json --input_directory ./case_dir/ --dry-run --output json
```

This is a validation-only dry run for Bohrium CLI 2.5.17. If calling `bohr` directly, first check the installed command help, and obtain project ID and current image/machine values from the platform rather than copying placeholders.

## Monitoring, collection, and lifecycle

Platform states are `Pending`, `Scheduling`, `Running`, `Finished`, and
`Failed`. `Finished` does not prove a valid training, inference, or simulation
result.

For every group:

1. list all members and compare unique case IDs with the manifest;
1. inspect logs for representative running and failed jobs;
1. wait through expected image-cache/runtime-initialization startup before calling a job hung;
1. download both finished and failed terminal jobs;
1. audit platform/files, then route workflow-quality acceptance to the owning skill;
1. retry only invalid cases.

Actions differ:

| Action    | Outputs             | Record   | Use                                               |
| --------- | ------------------- | -------- | ------------------------------------------------- |
| terminate | normally retained   | retained | stop while preserving recoverable state           |
| kill      | may not be retained | retained | force-stop an unrecoverable run                   |
| delete    | removed             | removed  | destructive cleanup only when explicitly required |

Resolve exact individual job IDs before lifecycle actions. Web UI group IDs and CLI-created group IDs may differ.

## Authentication and safety

- Let MatMaster built-ins handle credentials when possible.
- For CLI fallback, read `BOHR_ACCESS_KEY` from the environment; never print or hard-code it. Some CLI versions require mapping it to `ACCESS_KEY`.
- OpenAPI uses a bearer token and real project/user IDs; use API only when needed.
- Treat upload credentials, node passwords, and storage tokens as secrets.
- Verify paths before move, overwrite, or deletion, especially on `/share`.
