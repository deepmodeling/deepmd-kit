# XAS Spectrum Fitting with DeePMD-kit

This example shows how to train a model to predict X-ray absorption spectra (XAS)
from atomic structure using DeePMD-kit's `property` fitting net.

## Concept

- The model predicts a 102-dimensional output per atom: `[E_min, E_max, I_0, …, I_99]`
- During training, per-atom outputs are averaged over atoms of the **absorbing element**
  (identified by `sel_type.npy` in each training system)
- The edge type (K, L1, L2, …) is provided as a frame-level parameter `fparam`
- One training system per `(element, edge)` pair

## Quick Start

**1. Generate example training data**

```bash
python gen_data.py
```

This creates `data/Fe_K/` and `data/O_K/` with 50 frames each.

**2. Train the model**

```bash
dp train input.json
```

**3. Freeze the model**

```bash
dp freeze -o model.pb
```

**4. Test the model**

```bash
dp test -m model.pb -s data/Fe_K -n 10
dp test -m model.pb -s data/O_K  -n 10
```

`dp test` automatically detects `sel_type.npy` and applies element-wise averaging
before computing the error metrics.

## Data Format

Each system directory must contain:

```
data/Fe_K/
├── type.raw          # atom type indices, one per line (int)
├── type_map.raw      # element symbols, one per line
└── set.000/
    ├── coord.npy     # [nframes, natoms*3]   Cartesian coordinates (Å)
    ├── box.npy       # [nframes, 9]          cell vectors (Å), row-major
    ├── fparam.npy    # [nframes, nfparam]    edge one-hot encoding
    ├── sel_type.npy  # [nframes, 1]          absorbing element type index (float64)
    └── xas.npy       # [nframes, 102]        XAS label: [E_min, E_max, I_0..I_99]
```

### `sel_type.npy`

The type index of the absorbing element, stored as float64, constant per system.

```
Fe is type 0  →  sel_type.npy filled with 0.0
O  is type 1  →  sel_type.npy filled with 1.0
```

### `xas.npy` label layout (`task_dim = 102`)

| Column    | Meaning                                     |
|-----------|---------------------------------------------|
| `xas[i,0]` | `E_min` (eV) — lower bound of energy grid  |
| `xas[i,1]` | `E_max` (eV) — upper bound of energy grid  |
| `xas[i,2:]`| `I[0..99]` — 100 intensity values on `linspace(E_min, E_max, 100)` |

### `fparam.npy` edge encoding (`nfparam = 3`)

| Edge | Encoding  |
|------|-----------|
| K    | `[1,0,0]` |
| L1   | `[0,1,0]` |
| L2   | `[0,0,1]` |

Extend with more entries for additional edges and set `numb_fparam` accordingly.

## Input Parameters

Key fields in `input.json`:

| Parameter | Description |
|-----------|-------------|
| `fitting_net.type` | Must be `"property"` |
| `fitting_net.task_dim` | `102` (2 energy bounds + 100 intensities) |
| `fitting_net.intensive` | `true` — per-atom outputs are **averaged**, not summed |
| `fitting_net.numb_fparam` | Number of edge-type features (3 for K/L1/L2) |
| `loss.type` | `"xas"` — uses `sel_type.npy` for element-selective averaging |
| `loss.loss_func` | `"smooth_mae"` (recommended) or `"mse"` |

## Extending to More Elements / Edges

- Add a new system directory per `(element, edge)` pair
- Set `sel_type.npy` to the type index of the absorbing element in that system
- Set `fparam.npy` to the one-hot vector for the corresponding edge
- List all system paths under `training.training_data.systems`
- Increase `numb_fparam` if adding new edge types
