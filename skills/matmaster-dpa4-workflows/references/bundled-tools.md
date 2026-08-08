# Bundled fallbacks

## Contents

1. Compatibility policy
2. Environment inventory
3. Job-spec generation
4. Bohrium CLI adapter
5. Read-only OpenAPI probe
6. Job template
7. Missing prerequisites

## Compatibility policy

The platform adapters are self-contained at the Python/helper-script level.
The scientific run contract comes from the sibling DeepMD-kit skills. This
skill does not bundle the proprietary MatMaster UI, Bohrium credentials, the
`bohr` executable, models, scientific runtimes, or container images.

Use capabilities in this order:

1. installed MatMaster built-in skill/tool;
2. bundled helper using an installed `bohr` CLI;
3. bundled read-only OpenAPI probe;
4. explicit prerequisite installation or user/platform intervention.

Never fabricate a missing submission API. The bundled OpenAPI helper is intentionally read-only.

## Environment inventory

Run:

```bash
python scripts/check_environment.py --case-dir case-0001
python scripts/check_environment.py --case-dir case-0001 --require-file INPUT_FROM_HANDOFF
python scripts/check_environment.py --probe --require-bohr
python scripts/check_environment.py --probe --require-deepmd
python scripts/check_environment.py --probe --require-deepmd --require-lammps
```

It reports command paths, safe version probes, presence—not values—of
credential/project variables, persistent mounts, and required handoff files.
`--model` optionally records a model checksum without interpreting the model.
Use `--require-deepmd` for training/fine-tuning/inference images and add
`--require-lammps` for LAMMPS images. `--require-runtime` remains a shorthand
for requiring both.

## Job-spec generation

Generate a concrete `job.json` without editing a placeholder manually:

```bash
python scripts/make_job_spec.py case-0001 \
  --output case-0001/job.json \
  --project-id 123 \
  --image registry.dp.tech/path/to/validated-dpa4:tag \
  --machine 'CURRENT_VALID_GPU_SKU' \
  --name dpa4-workflow-case-0001 \
  --backward run.log --backward logs/ --backward results/ \
  --max-run-time 1440 --max-reschedule-times 1
```

The script rejects incomplete image addresses, guessed `/root/input` commands, missing case directories, and absent `run.sh` for the default command.

## Bohrium CLI adapter

Use `scripts/bohr_cli.py` instead of reconstructing command syntax:

```bash
python scripts/bohr_cli.py doctor
python scripts/bohr_cli.py project-list --json
python scripts/bohr_cli.py image-list --json
python scripts/bohr_cli.py image-list --type DeePMD-kit --json
python scripts/bohr_cli.py image-list --type LAMMPS --json
python scripts/bohr_cli.py machine-list --kind gpu --scene job --json
python scripts/bohr_cli.py job-list --json
python scripts/bohr_cli.py job-describe JOB_ID --json
python scripts/bohr_cli.py job-log JOB_ID --output logs/
python scripts/bohr_cli.py job-download JOB_ID --output results/
python scripts/bohr_cli.py group-list --json
python scripts/bohr_cli.py group-download GROUP_ID --output results/
```

Mutating actions preview by default:

```bash
python scripts/bohr_cli.py group-create --name run-v1 --project-id 123
python scripts/bohr_cli.py group-create --name run-v1 --project-id 123 --execute

python scripts/bohr_cli.py job-submit \
  --spec case-0001/job.json --input case-0001 --group-id GROUP_ID
python scripts/bohr_cli.py job-submit \
  --spec case-0001/job.json --input case-0001 --group-id GROUP_ID --validate
python scripts/bohr_cli.py job-submit \
  --spec case-0001/job.json --input case-0001 --group-id GROUP_ID --execute
```

The first form only prints a copy-safe command, `--validate` invokes Bohrium's native dry run without submission, and `--execute` performs the submission. Local validation rejects unresolved placeholders in the job spec and default `run.sh`. The sibling skill owns validation of scientific input files referenced by that wrapper. `job-terminate` and `job-kill` also require `--execute`. Resolve exact job IDs and preserve outputs before lifecycle changes.

The adapter command forms are validated against Bohrium CLI 2.5.17. Its preview output includes the CLI-native `--dry-run` flag and does not invoke the command. If the installed CLI has a different major/minor interface, inspect `bohr <family> <action> --help` before execution and update the adapter rather than guessing flags.

## Read-only OpenAPI probe

When no Bohrium skill or CLI is available but `BOHR_ACCESS_KEY` is injected, use:

```bash
python scripts/bohrium_readonly.py identity
python scripts/bohrium_readonly.py projects
python scripts/bohrium_readonly.py jobs --project-id 123 --status 1
python scripts/bohrium_readonly.py node-resources
python scripts/bohrium_readonly.py nodes
python scripts/bohrium_readonly.py image-search dpa4 --limit 10
python scripts/bohrium_readonly.py file-stat personal/path/to/file --project-id 0 --user-id USER_ID
```

The helper uses Python's standard library, never prints the access key, and performs only GET requests. Treat returned storage tokens or node credentials as secrets if upstream responses contain them.

## Job template

Use:

- `assets/bohrium-job/job.json` as the container-job skeleton;
- `assets/bohrium-job/manifest.json` as the case/job/attempt lineage skeleton.

Fill all `__PLACEHOLDER__` tokens. The case `run.sh` and its scientific inputs
must come from the prepared owning-skill handoff.

Prefer `make_job_spec.py` for the final job specification. Do not submit templates with unresolved placeholders.

## Missing prerequisites

If `bohr` is missing and production submission is required, install the current official Bohrium CLI through the platform-supported method, then run `bohr version`. Do not bundle or download an unverified executable inside this skill.

If the prepared case contract is missing, return to the owning DeepMD skill; do
not reconstruct training, inference, model deployment, or LAMMPS semantics in
this platform skill.
