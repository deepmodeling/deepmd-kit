# DPA4 training reference

Read this file only after the user chooses DPA4/SeZM, or when it is the best fit
for the task. Keep shared data checks and the train/monitor workflow in
`../SKILL.md`; this file records DPA4-specific choices.

## When to choose DPA4

Choose DPA4 when the user explicitly requests DPA4/SeZM or wants its
SO(3)-equivariant message-passing architecture and accepts a GPU-oriented,
PyTorch-only workflow. The aliases `DPA4`, `SeZM`, and `sezm` select the same
implementation.

DPA4 is not selected merely because a checkpoint ends in `.pt`. Inspect an
existing checkpoint with:

```bash
dp --pt show model.pt descriptor fitting-net type-map
```

## Minimal model configuration

Start from the maintained example at `examples/water/dpa4/input.json`. A minimal
model section is:

```json
{
  "model": {
    "type": "dpa4",
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "rcut": 6.0
    }
  }
}
```

`model.type: "dpa4"` selects the DPA4/SeZM descriptor and its default energy
fitting network. DPA4 defaults to `float32`; double precision is unnecessary and
not recommended for the normal workflow.

## Parameters to choose deliberately

- `rcut` sets the local environment cutoff.
- On the conservative energy path, `sel` is an initial neighbor-search capacity
  that grows on demand; it does not truncate the neighbor list. It may also be
  set to `auto` or `auto:factor` from training data.
- `lmax`/`l_schedule` and `mmax`/`m_schedule` control angular resolution and are
  primary accuracy-cost levers.
- `n_blocks` controls depth; `channels` and `n_radial` control width.
- `n_focus` and `n_atten_head` control aggregation.

Use documented defaults or a maintained example unless the user has evidence for
changing these parameters. Do not copy DPA3 descriptor parameters into DPA4.

## Train and monitor

Use the PyTorch backend:

```bash
dp --pt train input.json
```

Monitor `lcurve.out`, validation metrics, checkpoint creation, and non-finite
values. DPA4 also supports advanced property, spin, denoising, ZBL, multitask,
and LoRA configurations; follow the DPA4 documentation and examples rather than
combining those features from memory. For checkpoint adaptation and LoRA, use
the `deepmd-finetune-dpa4` skill.

## Freeze and test

DPA4 checkpoints are `.pt`, but deployment uses an AOTInductor `.pt2` archive:

```bash
dp --pt freeze -c model.ckpt.pt -o frozen_model
dp test -m frozen_model.pt2 -s /path/to/test_system -n 30
```

The command detects DPA4/SeZM and appends `.pt2`. DPA4 does not use the ordinary
TorchScript `.pth` freeze path and does not support model compression. Validate
the exported archive in the target inference or LAMMPS environment.

If the checkpoint is multi-task, inspect its branches and pass the selected
head during export:

```bash
dp --pt show model.ckpt.pt model-branch descriptor type-map
dp --pt freeze -c model.ckpt.pt -o frozen_model --head SELECTED_BRANCH
```

The frozen `.pt2` is a selected single-head artifact.

## DPA4 checklist

- [ ] The PyTorch backend is available.
- [ ] `model.type` is `dpa4`/`sezm`, or the stored checkpoint configuration proves it.
- [ ] `type_map`, data labels, and train/validation systems are consistent.
- [ ] Parameter changes are based on DPA4 documentation, not DPA3 defaults.
- [ ] Training and validation metrics are finite.
- [ ] The selected checkpoint is exported to `.pt2` and tested.
- [ ] `dp compress` is not used for DPA4.

## References

- [DPA4 model documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html)
- [DPA4 training example](../../../examples/water/dpa4/input.json)
- [Energy model training](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/train-energy.html)
