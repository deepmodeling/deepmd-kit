---
name: deepmd-finetune-dpa4
description: Fine-tune a DPA4 model in DeePMD-kit. Use for standard or LoRA fine-tuning from a DPA4/SeZM .pt checkpoint, validation and .pt2 export.
compatibility: Requires deepmd-kit with the PyTorch backend. DPA4/SeZM training is GPU-oriented.
license: LGPL-3.0-or-later
metadata:
  author: SchrodingersCattt
  version: '1.0'
  repository: https://github.com/deepmodeling/deepmd-kit
---

# DeePMD-kit Fine-tuning: DPA4

Fine-tune a DPA4/SeZM checkpoint on downstream DeePMD data. This skill covers
single-task standard and LoRA fine-tuning. Do not infer the model family from a
`.pt` suffix or filename: DPA3 and DPA4 checkpoints use the same suffix.

## Route the checkpoint

If the user has not already established the model family, inspect the stored
configuration:

```bash
dp --pt show pretrained.pt descriptor fitting-net type-map
```

Use this skill only when the descriptor/model configuration identifies DPA4 or
SeZM. If the checkpoint is multi-task, inspect its branches before selecting a
head:

```bash
dp --pt show pretrained.pt model-branch descriptor type-map
```

Do not guess a branch. Use `deepmd-finetune-dpa3` instead when the descriptor is
DPA3, and stop when the family cannot be established.

## Obtain a pretrained checkpoint

Fine-tuning requires a DPA4/SeZM **training checkpoint** (`.pt`), not a compiled
`.pt2` deployment archive. First check the registry exposed by the installed
version:

```bash
dp pretrained download -h
```

Use a built-in model only when an exact DPA4/SeZM name is listed there. Do not
guess `dp pretrained download DPA4`: some releases have no registered DPA4
checkpoint.

For a workflow smoke test, a DeePMD-kit source checkout contains:

```text
examples/water/dpa4/lmp/pretrained.pt
```

Use that file directly:

```bash
cp examples/water/dpa4/lmp/pretrained.pt ./pretrained.pt
dp --pt show pretrained.pt descriptor fitting-net type-map
```

It is a compact O/H smoke-test model, not a general-purpose pretrained
potential. For scientific fine-tuning, obtain a checkpoint from its documented
publisher or train one with the same DeePMD-kit revision that will perform the
fine-tuning. Pin and record the model source, checksum, DeePMD-kit revision,
architecture, task branch, and `type_map`. Reject a checkpoint whose descriptor,
element set/order, or architecture does not match the intended target. Prefer a
bounded one-step compatibility run before a long job.

## Before fine-tuning

1. Confirm the checkpoint exists and can be inspected.
1. Confirm training and validation systems, labels, and element type maps.
1. Keep a held-out test set that is not used for training or model selection.
1. Start from the exact checkpoint architecture. Introducing new element types,
   changing architecture, or combining specialized spin/property/multi-task
   configurations requires separate compatibility validation.
1. Choose standard fine-tuning or LoRA. Do not assume a built-in DPA4 model name;
   check `dp pretrained download -h` for the installed version.

## Decide whether to use LoRA

Use **standard fine-tuning** by default when the target is multi-task, the domain
shift is large, all parameters should adapt, or the workflow combines untested
spin/property/denoising/ZBL changes. Use **LoRA** when the target is single-task,
parameter-efficient adaptation is desired, the downstream domain is reasonably
close to pretraining, and the exact base architecture is known. DPA4 LoRA is
supported by `dp --pt`; do not use it with the exportable training backend.

The pretrained checkpoint does not need to contain LoRA. LoRA is enabled by the
new fine-tuning input through a non-null `model.lora` block. Check an input with:

```bash
python - input.json <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)

model = config.get("model", {})
branches = model.get("model_dict")
lora = model.get("lora")
print("multi_task =", isinstance(branches, dict))
print("lora =", lora)
print("use_lora =", lora is not None)
if isinstance(branches, dict) and lora is not None:
    raise SystemExit("DPA4 LoRA targets must be single-task")
PY
```

Interpretation:

- absent or `null` `model.lora` means standard fine-tuning;
- a `model.lora` mapping means LoRA fine-tuning;
- a multi-task target plus LoRA is unsupported.

To diagnose whether a `.pt` file still contains **active, unmerged** LoRA state,
inspect its saved model parameters and adapter tensors without executing pickled
code:

```bash
python - pretrained.pt <<'PY'
import sys
import torch

raw = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
if not isinstance(state, dict):
    raise TypeError(f"Unsupported checkpoint payload: {type(state).__name__}")

extra = state.get("_extra_state", {})
params = extra.get("model_params", {}) if isinstance(extra, dict) else {}
configured = params.get("lora") if isinstance(params, dict) else None
markers = (".A_by_l", ".B_by_l", ".A_m0", ".B_m0", ".A_m.", ".B_m.", ".lora_scaling")
adapter_keys = sorted(
  key for key in state if any(marker in key for marker in markers)
)

print("configured_lora =", configured)
print("adapter_tensor_count =", len(adapter_keys))
for key in adapter_keys[:20]:
    print("adapter_tensor =", key)

if configured is not None and adapter_keys:
    print("classification = active/unmerged LoRA checkpoint")
elif configured is not None or adapter_keys:
    print("classification = inconsistent or transitional; inspect before use")
else:
    print("classification = plain checkpoint or merged LoRA checkpoint")
PY
```

Do not infer training history from the last classification. Validation-selected
best LoRA checkpoints fold adapter deltas into ordinary DPA4 weights and remove
LoRA metadata/tensors, so they intentionally look plain. Such merged checkpoints
are suitable for evaluation, new fine-tuning, and `.pt2` export, but do not carry
the optimizer/EMA state needed to resume the same LoRA run. Periodic/final LoRA
checkpoints retain active adapters and are the resumable form.

## Standard fine-tuning

The model section in `input.json` must match the checkpoint unless the standard
pretrained-script mechanism is deliberately used:

```bash
dp --pt train input.json --finetune pretrained.pt
```

When fine-tuning a single-task target from a multi-task checkpoint and the
intent is to preserve a particular pretrained fitting head, pass the branch
selected above:

```bash
dp --pt train input.json --finetune pretrained.pt --model-branch SELECTED_BRANCH
```

If `--model-branch` is omitted, the fitting net may be initialized from the
`RANDOM` branch instead. A multi-task target uses `finetune_head` in each target
branch rather than the command-line option.

If the architecture is unknown, `--use-pretrain-script` can inherit the stored
model configuration except for `type_map`:

```bash
dp --pt train input.json --finetune pretrained.pt --use-pretrain-script
```

Inspect the resulting configuration and run a bounded initial segment before a
long training job. Do not combine model-specific additions with
`--use-pretrain-script` unless that combination has been validated.

## LoRA fine-tuning

DPA4/SeZM supports LoRA adapters for single-task fine-tuning. Copy the exact base
architecture into `lora_ft.json`, then add:

```json
{
  "model": {
    "type": "dpa4",
    "lora": {
      "rank": 16,
      "alpha": 16.0
    }
  }
}
```

Run:

```bash
dp --pt train lora_ft.json --finetune pretrained.pt
```

The JSON fragment above is not a complete training input. Adapt the full public
example at `../../examples/water/dpa4/lora_ft.json`, but copy the architecture
from the actual source checkpoint before adding `model.lora`. The compact
`examples/water/dpa4/lmp/pretrained.pt` does not match the larger architecture
in the maintained `lora_ft.json` and must not be paired with it unchanged. Do
not add `--use-pretrain-script` to this LoRA command unless a targeted test
confirms that the intended LoRA configuration is retained.

## Monitor and validate

Monitor `lcurve.out` for non-finite values and train/validation divergence.
Select a checkpoint using validation data, then evaluate the selected checkpoint
on the complete held-out test systems. Report energy and force errors, plus
virial errors when those labels are part of the task.

## Export and test

DPA4/SeZM uses the `.pt2` AOTInductor export path rather than the conventional
PyTorch `.pth` freeze path:

```bash
dp --pt freeze -c ckpt/model.ckpt.pt -o finetuned_model
dp test -m finetuned_model.pt2 -s /path/to/test_system -n 30
```

The freeze command detects DPA4/SeZM and writes `finetuned_model.pt2`. Validate
the exported archive in the target environment before deployment.

For a multi-task checkpoint, freeze the selected head explicitly:

```bash
dp --pt freeze -c ckpt/model.ckpt.pt -o finetuned_model --head SELECTED_BRANCH
```

The resulting `.pt2` contains the selected single head; do not pass a branch
again when loading that archive.

## Checklist

- [ ] The stored descriptor identifies DPA4/SeZM; the `.pt` suffix was not used as proof.
- [ ] The checkpoint source, checksum, DeePMD-kit revision, architecture, and type map are recorded.
- [ ] The intended branch is explicit for a multi-task checkpoint.
- [ ] Training, validation, and held-out test systems are separate.
- [ ] The input architecture is compatible with the checkpoint.
- [ ] Standard fine-tuning versus LoRA was selected from the task layout and domain shift.
- [ ] Active LoRA state versus a merged best checkpoint is interpreted correctly.
- [ ] LoRA uses a complete base configuration and is not silently overwritten.
- [ ] Training and held-out metrics are finite and reported with units.
- [ ] The selected `.pt` checkpoint was exported to and tested as `.pt2`.

## References

- [DPA4 model and LoRA documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html)
- [Fine-tuning documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/train/finetuning.html)
- [Show model information](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/show-model-info.html)
