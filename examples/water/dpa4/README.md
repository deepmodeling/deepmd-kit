# DPA4/SeZM water examples

This directory contains PyTorch input files for training DPA4/SeZM on the
water example dataset. The recommended model and descriptor type is `DPA4`;
`dpa4`, `SeZM`, and `sezm` are accepted aliases for the same implementation.

## DPA4 model variants

DPA4 offers three pretrained model variants from the
[DPA4-MatPES-v20260628](https://www.aissquare.com/models/detail?pageType=models&name=DPA4-MatPES-v20260628&id=424)
release, trained on the MatPES R2SCAN 2025.2 dataset covering the full periodic
table (H–Og) with a 6.0 Å cutoff radius:

| Model     | Parameters          | Energy MAE    | Force MAE     | Training hours (H20) |
|-----------|---------------------|---------------|---------------|-----------------------|
| DPA4-Mini | 655,504 (0.655 M)   | 21.99 meV/atom | 110.23 meV/Å  | 9.9 h                 |
| DPA4-Neo  | 1,125,372 (1.125 M) | 17.70 meV/atom | 95.91 meV/Å   | 14.6 h                |
| DPA4-Air  | 5,148,611 (5.149 M) | 15.34 meV/atom | 93.27 meV/Å   | 19.5 h                |

### Architecture comparison

The three variants share the same DPA4/SeZM architecture but differ in
descriptor capacity. The table below shows the key `descriptor` parameters
that control model size and expressiveness:

| Parameter (`descriptor`)   | DPA4-Mini | DPA4-Neo | DPA4-Air | Description                                          |
|----------------------------|-----------|----------|----------|------------------------------------------------------|
| `channels`                 | 32        | 32       | 64       | Feature channels per (l, m) coefficient              |
| `lmax`                     | 2         | 3        | 3        | Maximum angular degree of equivariant representation |
| `n_blocks`                 | 2         | 2        | 3        | Number of message-passing interaction blocks         |
| `so2_layers`               | 3         | 3        | 4        | SO(2) mixing layers per interaction block            |
| `n_focus`                  | 1         | 2        | 1        | Parallel focus streams in SO(2) convolution          |
| `ffn_blocks`               | 1         | 1        | 1        | FFN sublayers per interaction block                  |

All variants share these settings: `rcut = 6.0`, `n_radial = 16`, `mmax = 1`,
`n_atten_head = 1`, `radial_so2_mode = "degree_channel"`, `radial_so2_rank = 1`,
`so3_readout = "mlp"`, `use_env_seed = true`, `message_node_so3 = true`,
`ffn_so3_grid = true`, `ffn_neurons = 0` (auto), `grid_branch = [0, 0, 1]`,
`precision = "float32"`, `use_amp = true`, `enable_tf32 = true`,
`use_compile = true`.

### Training hyperparameters

| Parameter                | DPA4-Mini    | DPA4-Neo     | DPA4-Air     |
|--------------------------|--------------|--------------|--------------|
| `start_lr`               | 0.001        | 0.0005       | 0.00035      |
| `stop_lr`                | 1e-6         | 1e-6         | 1e-6         |
| `batch_size`             | `filter:12000` | `filter:3000` | `filter:2000` |
| `num_epochs`             | 300          | 250          | 200          |
| Optimizer                | HybridMuon   | HybridMuon   | HybridMuon   |
| `weight_decay`           | 0.001        | 0.001        | 0.001        |
| `decay_phase_ratio`      | 0.65         | 0.65         | 0.65         |

### How to switch between variants in `input.json`

To reproduce a specific variant, modify these sections in your `input.json`:

**1. Descriptor section** — set the architecture-defining keys:

```json
"descriptor": {
    "channels": 32,      // Mini=32, Neo=32, Air=64
    "lmax": 2,           // Mini=2, Neo=3, Air=3
    "n_blocks": 2,       // Mini=2, Neo=2, Air=3
    "so2_layers": 3,     // Mini=3, Neo=3, Air=4
    "n_focus": 1,        // Mini=1, Neo=2, Air=1
    "ffn_blocks": 1      // all variants use 1
}
```

**2. Learning rate section** — match the per-variant schedule:

```json
"learning_rate": {
    "start_lr": 0.001,   // Mini=0.001, Neo=0.0005, Air=0.00035
    "stop_lr": 1e-6
}
```

**3. Training section** — set batch size and epochs:

```json
"training": {
    "training_data": {
        "batch_size": "filter:12000"  // Mini=12000, Neo=3000, Air=2000
    },
    "num_epochs": 300                 // Mini=300, Neo=250, Air=200
}
```

The `type_map` in the pretrained models covers all 118 elements (H–Og). When
training on a smaller system (like water), replace it with only the elements
present in your dataset, e.g. `["O", "H"]`.

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

## Run

```bash
cd examples/water/dpa4
dp --pt train input.json
```

## Using pretrained checkpoints

Download pretrained checkpoints from
[AIS Square](https://www.aissquare.com/models/detail?pageType=models&name=DPA4-MatPES-v20260628&id=424):

```bash
# Evaluate
dp --pt test -m DPA4-<Mini|Neo|Air>-MatPES-v20260628.pt -s /path/to/test/system -n 1000

# Freeze for deployment (produces frozen_model.pt2)
dp --pt freeze -c DPA4-<Mini|Neo|Air>-MatPES-v20260628.pt -o frozen_model

# Fine-tune from a pretrained checkpoint
dp --pt train input_finetune.json --finetune DPA4-<Mini|Neo|Air>-MatPES-v20260628.pt
```

## References

- [DPA4 paper](https://arxiv.org/abs/2606.02419): Li et al., *DPA4: Pushing the Accuracy-Cost Frontier of Interatomic Potentials with EMFA SO(2) Convolution*, arXiv:2606.02419 (2026).
- [DeePMD-kit DPA4 documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html)
- [MatPES R2SCAN dataset](https://arxiv.org/abs/2503.04070): Kaplan et al., arXiv:2503.04070 (2025).
