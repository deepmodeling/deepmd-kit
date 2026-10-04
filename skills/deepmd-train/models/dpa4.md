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


## Recommended preset workflow

For a new DPA4/SeZM configuration, choose the family and grade first, then use
the v20260911 preset for that grade. Available DPA4 grades are nano, mini, neo,
air, plus, pro, max, and ultra; the name format is
dpa4-<grade>-v20260911. The compact nano starting point is used by
examples/water/dpa4/input_preset.json.

The model section of a new energy input should stay small:

```json
{
  "model": {
    "preset": "dpa4-nano-v20260911",
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "use_amp": true,
      "seed": 42
    },
    "fitting_net": {
      "seed": 42
    }
  }
}
```

The preset supplies type, type_map, descriptor, and fitting_net. An explicit
type_map replaces the preset's 118-element map as a whole. Explicit keys inside
descriptor or fitting_net merge over the preset and take precedence; use them
for run-specific settings such as seeds, use_amp, add_chg_spin_ebd,
default_chg_spin, or vacuum_ref. Do not retain a full manual descriptor and
fitting-network block beside preset, because those blocks would override the
architecture the preset is meant to select.

Keep learning_rate, loss, and the complete training section outside this model
simplification. Start from examples/water/dpa4/input_preset.json and run:

```bash
cd examples/water/dpa4
dp --pt train input_preset.json
```

model.preset expands before argument validation and fine-tuning rules. The
expanded configuration is the one recorded in out.json.

### Vacuum reference and charge/spin conditioning

fitting_net.vacuum_ref is an explicit modeling choice. It only remains enabled
when the model has an assigned energy bias through model.preset_out_bias.energy.
The entries must be isolated-atom energies from the same reference calculation
and must follow the configured type_map; use the maintained
examples/water/dpa4/input_e0.json as the shape of this workflow. If no bias is
assigned, DeePMD-kit disables vacuum_ref.

descriptor.add_chg_spin_ebd and descriptor.default_chg_spin are also explicit
conditioning choices. They describe charge/spin inputs and defaults; they are
independent of execution controls such as model.use_compile, model.enable_tf32,
and freeze-time inference environment variables.

### Advanced architecture overrides

Use manual descriptor and fitting_net architecture blocks only when designing
an architecture outside the released presets or reproducing an existing
checkpoint exactly:

```json
{
  "model": {
    "type": "dpa4",
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "type": "dpa4",
      "rcut": 6.0,
      "channels": 32,
      "lmax": 2
    },
    "fitting_net": {
      "type": "dpa4_ener",
      "neuron": [
        192,
        192,
        192
      ]
    }
  }
}
```

Do not combine this full architecture form with a preset. For fine-tuning,
preserve the checkpoint's stored descriptor and fitting-network structure;
do not replace it blindly with a newer v20260911 preset.

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

DPA4 checkpoints are `.pt`, but deployment uses an AOTInductor `.pt2` archive.
Read `../../deepmd-python-inference/references/dpa4-freeze-policy.md` and choose
the freeze-time inference policy before exporting:

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
- [DPA4 training example](../../../examples/water/dpa4/input_preset.json)
- [Energy model training](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/train-energy.html)
