# DPA4/SeZM water examples

This directory contains PyTorch input files for training DPA4/SeZM on the
water example dataset. The recommended model and descriptor type is `DPA4`;
`dpa4`, `SeZM`, and `sezm` are accepted aliases for the same implementation.

Input files:

- `input.json`: baseline conservative energy training, using a compact
  DPA4-Neo-style parameter set.
- `input-zbl.json`: energy training with ZBL zone bridging.
- `input-spin.json`: spin-energy training with the DeePMD spin convention.
- `input_dens.json`: direct-force denoising training.
- `input_multitask.json`: multitask training with a shared descriptor and
  case-conditioned shared fitting network.
- `lora_ft.json`: LoRA fine-tuning.
- `lmp/`: compact checkpoint and LAMMPS smoke-test files.

Run:

```bash
cd examples/water/dpa4
dp --pt train input.json
```
