# DPA4C training reference

Read this file only after the user chooses DPA4C, or when it is the best fit for
the task. Keep shared data checks and the train/monitor workflow in
`../SKILL.md`; this file records DPA4C-specific choices.

## Backend contract

DPA4C uses the PyTorch Exportable backend. Train it with:

```bash
dp --pt-expt train input.json
```

Do not substitute `dp --pt`. DPA4/SeZM uses the conventional PyTorch backend,
whereas DPA4C is implemented for `--pt-expt`.

## Recommended preset workflow

For a new DPA4C configuration, choose the family and grade first, then use a
v20260911 preset. Available grades are nano, mini, neo, air, and plus; the name
format is dpa4c-<grade>-v20260911. The maintained
examples/water/dpa4c/input.json uses dpa4c-nano-v20260911.

The model section of a new energy input should contain the preset, the dataset's
type_map, and task-specific additions only:

```json
{
  "model": {
    "preset": "dpa4c-nano-v20260911",
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "use_amp": false,
      "seed": 42
    },
    "fitting_net": {
      "seed": 42
    }
  }
}
```

The preset supplies type_map, descriptor, and fitting_net. An explicit type_map
replaces the preset's 118-element map as a whole. Explicit descriptor and
fitting_net keys merge over the preset and take precedence. Do not leave the old
full architecture blocks beside preset, because those keys would override the
selected grade.

Keep learning_rate, loss, and training outside the architecture simplification.
DPA4C uses dp --pt-expt train; execution controls training.enable_compile and
training.enable_tf32 stay in training and are independent of model conditioning.
Do not put `use_compile` under `model` for DPA4C.

### Vacuum reference and charge/spin conditioning

fitting_net.vacuum_ref is an explicit modeling choice. It requires an assigned
energy bias through model.preset_out_bias.energy, with isolated-atom energies
from the same reference calculation and entries aligned with the configured
type_map. Without an assigned bias, DeePMD-kit disables vacuum_ref. The
maintained isolated-atom examples show the preset_out_bias shape and reference
data contract.

If the task needs charge/spin conditioning, add descriptor.add_chg_spin_ebd and,
when a default state is required, descriptor.default_chg_spin. These are
model-input choices and are not interchangeable with compile, TF32, AMP, or
other execution flags.

### Advanced architecture overrides

Use explicit descriptor and fitting-network architecture blocks only when
reproducing a compatible checkpoint or intentionally designing a configuration
outside the released presets:

```json
{
  "model": {
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "type": "dpa4c",
      "rcut": 6.0,
      "channels": 32,
      "lmax": 2,
      "radial_modes": 0
    },
    "fitting_net": {
      "type": "ener",
      "neuron": [
        192,
        192,
        192
      ]
    }
  }
}
```

Do not combine this full architecture form with a preset. During fine-tuning,
preserve the checkpoint's architecture and task-specific fields instead of
blindly replacing them with a newer preset.

## Parameters to choose deliberately

- `channels` is the primary width and memory/throughput control.
- `lmax` controls angular resolution.
- `radial_modes` adds type-pair radial flexibility without widening the
  per-atom state.
- `use_amp` is the descriptor's CUDA bf16 policy and is independent of
  `training.enable_compile` and `training.enable_tf32`.
- DPA4C has no `sel`; it consumes a carry-all neighbor graph within `rcut`.

Use a compression-supported configuration when compressed deployment is
required: `channels` in `{8, 16, 32, 64, 128}`, `lmax` in `{2, 3, 4}`, and
`radial_modes` in `{0, 2, 4, 8}`.

## Freeze, compress, and test

Choose the inference policy before export. The three values below are captured
in the archive rather than re-evaluated when LAMMPS loads it. For
molecular-dynamics production, a conservative accelerated policy is:

```bash
export DP_CUDA_INFER=1
export DP_TF32_INFER=0
export DP_AMP_INFER=0
```

A compressed archive needs `DP_CUDA_INFER` of at least `1` to use the fused
path. Validate more aggressive precision settings against the target system.

```bash
dp --pt-expt freeze -c model.ckpt.pt -o frozen_model --lower-kind graph
dp --pt-expt compress -i frozen_model.pt2 -o compressed_model.pt2
dp test -m compressed_model.pt2 -s /path/to/test_system -n 30
```

Keep the same target physical compute node and allocation for freeze,
compression, validation, and production unless portability has been independently
validated for that exact device and toolchain. Validate both the plain `.pt2`
archive and, when used, the compressed archive before production MD.

## DPA4C checklist

- [ ] The command uses `dp --pt-expt`.
- [ ] `model.descriptor.type` is `dpa4c`.
- [ ] `training.enable_compile` and `training.enable_tf32` are set deliberately.
- [ ] No DPA4-only `model.use_compile` or `model.enable_tf32` was copied in.
- [ ] The first compiled step is allowed extra warm-up time.
- [ ] The checkpoint is exported to `.pt2` and tested.
- [ ] Compression constraints are satisfied when `dp --pt-expt compress` is used.
- [ ] Freeze-time inference variables and the target runtime are recorded.
- [ ] Export and production stay on the target node unless portability is proven.

## References

- [DPA4C model documentation](../../../doc/model/dpa4c.md)
- [DPA4C training example](../../../examples/water/dpa4c/input.json)
- [Energy model training](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/train-energy.html)
