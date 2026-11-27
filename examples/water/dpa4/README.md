# DPA4/SeZM water examples

This directory contains input files for training DPA4/SeZM on the
water example dataset. The recommended model and descriptor type is `DPA4`;
`dpa4`, `SeZM`, and `sezm` are accepted aliases for the same implementation.

Input files:

- `input_preset.json`: recommended new-training path using the
  `dpa4-nano-v20260911` preset.
- `input_dpa45.json`: the `dpa4.5-nano-v20261111` preset with ZBL zone bridging
  and the recommended optimizer setting; train it with
  `dp --pt-expt train input_dpa45.json`.
- `input_dpa45_moe_multitask.json`: the `dpa4.5-nano-v20261111` preset with a
  mixture of experts (`moe`) in the descriptor and in the fitting network,
  trained on two tasks that share both; every branch is a task. Train it with
  `dp --pt-expt train input_dpa45_moe_multitask.json`, then merge the experts
  of one branch with
  `dp --pt-expt moe-merge model.ckpt.pt --task water_1 -o plain.pt`, or keep
  the weights of the elements with `--elements all`, and freeze the merged
  checkpoint as usual.
- `input.json`: advanced/manual architecture override example.
- `input-zbl.json`: energy training with ZBL zone bridging.
- `input_dens.json`: direct-force denoising training.
- `input_multitask.json`: multitask training with a shared descriptor and
  case-conditioned shared fitting network.
- `input_multitask_preset.json`: the same multitask training with the shared
  descriptor and fitting network taken from the named preset
  `dpa4-nano-v20260911`.
- `lora_ft.json`: LoRA fine-tuning.
- `lmp/`: compact checkpoint and LAMMPS smoke-test files.

Run with the recommended PyTorch Exportable backend:

```bash
cd examples/water/dpa4
dp --pt-expt train input.json
```

`dp --pt train` trains the same model with the PyTorch backend, which
`input_dens.json` and `lora_ft.json` require.
