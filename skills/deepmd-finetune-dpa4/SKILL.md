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

Fine-tuning requires a DPA4/SeZM training checkpoint (`.pt`), not a `.pt2`
deployment archive. Check whether the installed version provides one:

```bash
dp pretrained download -h
```

Use only a listed model or a checkpoint supplied by the user or its publisher.
Record its source and DeePMD-kit version, then verify its descriptor, branch,
architecture, and `type_map` before use.

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

Use standard fine-tuning by default. Use LoRA only for a single-task target when
parameter-efficient adaptation is wanted and the exact base architecture is
known. LoRA is enabled by a non-null `model.lora` block in the new input; the
pretrained checkpoint does not need to contain LoRA. Multi-task LoRA targets are
unsupported.

Periodic LoRA checkpoints retain adapters and can resume training. Best
checkpoints may merge the adapters into ordinary DPA4 weights, so absence of
LoRA metadata does not prove LoRA was never used.

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

When `pretrained.pt` is multi-task, preserve the selected fitting head:

```bash
dp --pt train lora_ft.json --finetune pretrained.pt \
    --model-branch SELECTED_BRANCH
```

Use the shorter command only for a single-task source checkpoint.

The JSON fragment above is not a complete training input. Adapt the full public
example at `../../examples/water/dpa4/lora_ft.json`, but copy the exact
architecture from the source checkpoint before adding `model.lora`. Do not add
`--use-pretrain-script` unless a targeted test confirms that LoRA is retained.

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
- [ ] The checkpoint source, DeePMD-kit revision, architecture, and type map are recorded.
- [ ] The intended branch is explicit for a multi-task checkpoint.
- [ ] Training, validation, and held-out test systems are separate.
- [ ] The input architecture is compatible with the checkpoint.
- [ ] Standard fine-tuning versus LoRA was selected from the task layout and domain shift.
- [ ] A resumable LoRA checkpoint is distinguished from a merged best checkpoint.
- [ ] LoRA uses a complete base configuration and is not silently overwritten.
- [ ] Training and held-out metrics are finite and reported with units.
- [ ] The selected `.pt` checkpoint was exported to and tested as `.pt2`.

## References

- [DPA4 model and LoRA documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html)
- [Fine-tuning documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/train/finetuning.html)
- [Show model information](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/show-model-info.html)
