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

## Start from the maintained example

Start from `examples/water/dpa4c/input.json`. DPA4C is selected as a descriptor,
not as a separate model scaffold:

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
      "radial_modes": 4,
      "precision": "float32"
    },
    "fitting_net": {
      "neuron": [
        192,
        192,
        192
      ],
      "activation_function": "silu",
      "precision": "float32"
    }
  },
  "training": {
    "training_data": {
      "systems": [
        "../data/data_0"
      ],
      "batch_size": 1
    },
    "numb_steps": 1000000,
    "enable_compile": true,
    "enable_tf32": true
  }
}
```

The acceleration controls are backend-specific and their location is part of
the input contract:

- DPA4C / `--pt-expt`: `training.enable_compile` and
  `training.enable_tf32`.
- DPA4 / `--pt`: `model.use_compile` and `model.enable_tf32`.

Do not put `use_compile` under `model` for DPA4C. Strict argument validation
rejects `model.use_compile`, and copying DPA4's key placement silently defeats
the intended workflow until validation fails. Before launching a long job,
normalize or run a bounded smoke test and confirm that the generated training
input still contains both DPA4C keys under `training`.

`training.enable_compile=true` uses `make_fx` plus `torch.compile`/Inductor for
training. The first step is slower because compilation is one-time work.
`training.enable_tf32=true` controls CUDA training matmuls independently of the
compiled path. Both choices affect numerical policy and should be recorded with
the run configuration.

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
