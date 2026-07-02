# DPA4/SeZM water examples

This directory contains PyTorch input files for training DPA4/SeZM on the
water example dataset. The recommended model and descriptor type is `DPA4`;
`dpa4`, `SeZM`, and `sezm` are accepted aliases for the same implementation.

## Model architecture variants

The `input.json` uses a DPA4-Neo configuration. To switch to a
heavier or lighter variant, recommended changes only involve the following 
`descriptor` keys:

| Key in `descriptor` | DPA4-Neo (default) | DPA4-Air | DPA4-Mini |
| ------------------- | ------------------ | -------- | --------- |
| `channels`          | 32                 | 64       | 32        |
| `lmax`              | 3                  | 3        | 2         |
| `n_blocks`          | 2                  | 3        | 2         |
| `mixing_layers`     | 3                  | 4        | 3         |
| `n_focus`           | 2                  | 1        | 1         |

Ready-to-use input files for each variant are provided in
[`arc_variants/`](arc_variants/).

## Input files

- `input.json`: baseline conservative energy training, using a compact
  DPA4-Neo-style parameter set.

- `input-zbl.json`: energy training with ZBL zone bridging.

- `input-spin.json`: spin-energy training with the DeePMD spin convention.

- `input_dens.json`: direct-force denoising training.

- `input_multitask.json`: multitask training with a shared descriptor and
  case-conditioned shared fitting network.

- `lora_ft.json`: LoRA fine-tuning.

- `lmp/`: compact checkpoint and LAMMPS smoke-test files.

- `arc_variants/`: input files for DPA4-Neo, DPA4-Air, and DPA4-Mini
  architectures.

Run:

```bash
cd examples/water/dpa4
dp --pt train input.json
```
