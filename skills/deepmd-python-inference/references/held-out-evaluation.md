# Complete held-out evaluation

Read this reference when labeled DeePMD systems are used to select a checkpoint
or admit a model for production. A bounded smoke test is not a complete
held-out evaluation.

## Admit the data

- Keep held-out systems independent of training and validation by system,
  trajectory, or source family; do not randomly split correlated adjacent frames.
- Enumerate every held-out system and all of its `set.*` directories.
- Set `natoms` to the number of whitespace-separated entries in `type.raw`.
  Require coordinate and force widths of `3 * natoms`, one energy row per frame,
  nine box values per periodic frame, and finite values for every evaluated label.
- When `type_map.raw` is present, interpret `type.raw` as zero-based indices into
  that ordered map and compare model and data types by element identity. When it
  is absent, require provenance that the dataset indices already follow the
  candidate model's ordered type map. Fail closed when neither contract is
  established; never copy dataset indices into an assumed model map.

## Run every system

Use the backend required by the exact candidate artifact. For a DPA4/SeZM native
checkpoint, run one command per held-out system:

```bash
detail_root="details/selected-SHA256"
test ! -e "$detail_root" || exit 1
mkdir -p "$detail_root"
detail_prefix="$detail_root/system.000"
dp --pt test -m selected.pt -s held_out/system.000 -n 0 -d "$detail_prefix"
```

`-n 0` evaluates all frames. Require explicit `-m`, `-s`, and a unique `-d`
prefix for each system. Preserve the command, log, true exit code, checkpoint
SHA256, and dataset identity. Do not overwrite existing detail files silently.

For a native multi-task checkpoint, inspect its branches and pass the admitted
branch during evaluation:

```bash
dp --pt show selected.pt model-branch
dp --pt test -m selected.pt -s held_out/system.000 -n 0 \
    -d "$detail_prefix" --head SELECTED_BRANCH
```

A frozen selected `.pt2` is already single-head; do not pass `--head` to it.

## Validate detail outputs

For an energy model, retain the emitted total-energy (`.e.out`),
energy-per-atom (`.e_peratom.out`), and force (`.f.out`) details. Retain
virial/stress details only when both reference labels and model outputs exist.
Require:

- energy rows equal the evaluated frame count;
- force rows equal `frames * natoms`, with xyz stored as columns;
- all reference and prediction values are finite;
- energy-per-atom errors come directly from `.e_peratom.out`, or total-energy
  errors are divided by `natoms` exactly once.

Reject missing systems, partial frame coverage, reused detail prefixes, and
results tied to another checkpoint hash.

## Report and decide

For every available label, report per-system MAE, RMSE, units, sample count, and
the population standard deviation (`ddof=0`) of the corresponding held-out
reference values. Report an absent label as `N/A`, never zero. Aggregate from all
retained rows; do not average per-system RMSE values.

Build parity plots from unrounded detail rows, with reference on x, prediction on
y, an equal-aspect `y=x` line, system identity, checkpoint hash, units, sample
count, RMSE, and reference-label standard deviation.

Admit the candidate only when every declared held-out system is complete and the
declared thresholds pass. Training logs, a successful freeze, or a LAMMPS
canary cannot replace this evaluation.
