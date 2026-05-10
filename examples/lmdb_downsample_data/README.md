# LMDB Example Data (Downsampled)

**WARNING: This data is heavily downsampled and intended ONLY for testing
the LMDB data loading pipeline. Do NOT use it for accuracy benchmarks or
comparisons with the standard npy data format.**

## Contents

- `water_training.lmdb` - 80 frames downsampled from `water/data/data_0`
- `water_validation.lmdb` - 20 frames downsampled from `water/data/data_2`
- `input_lmdb.json` - Example training config using LMDB data

## Usage

```bash
cd examples/lmdb_downsample_data
dp --pt train input_lmdb.json
```
