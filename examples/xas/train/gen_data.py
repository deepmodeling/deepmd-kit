"""Generate example XAS training data for a Fe-O system.

This script shows the required data format for XAS spectrum fitting.

Data layout
-----------
One training system per (element, edge) pair:

  data/Fe_K/          — Fe K-edge XAS
  data/O_K/           — O  K-edge XAS

Each system directory contains:

  type.raw            — atom type indices (int, one per line)
  type_map.raw        — element symbols, one per line
  set.000/
    coord.npy         — [nframes, natoms*3]  Cartesian coordinates (Å)
    box.npy           — [nframes, 9]         cell vectors (Å), row-major
    fparam.npy        — [nframes, nfparam]   edge encoding (one-hot or continuous)
    sel_type.npy      — [nframes, 1]         type index of absorbing element (float)
    xas.npy           — [nframes, task_dim]  XAS label: [E_min, E_max, I_0..I_99]

Label format (task_dim = 102)
------------------------------
  xas[i, 0]    = E_min  (eV) — lower bound of the energy grid for frame i
  xas[i, 1]    = E_max  (eV) — upper bound of the energy grid for frame i
  xas[i, 2:]   = I      (arb. units) — 100 equally-spaced intensity values
                  on the grid linspace(E_min, E_max, 100)

fparam encoding (nfparam = 3 for K/L1/L2 edges)
-------------------------------------------------
  K-edge  → [1, 0, 0]
  L1-edge → [0, 1, 0]
  L2-edge → [0, 0, 1]
  (extend as needed; use numb_fparam in input.json accordingly)

sel_type.npy
------------
  Integer type index of the absorbing element, stored as float64.
  All frames in a system must share the same value (it is constant per system).
  Example: Fe is type 0 → sel_type.npy filled with 0.0
           O  is type 1 → sel_type.npy filled with 1.0
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
nframes   = 50        # number of frames per system
numb_pts  = 100       # energy grid points
task_dim  = numb_pts + 2   # E_min + E_max + 100 intensities
nfparam   = 3         # K / L1 / L2 one-hot
natoms    = 8         # 4 Fe (type 0) + 4 O (type 1)
box_size  = 4.0       # Å

rng = np.random.default_rng(42)

# Equilibrium positions: simple rock-salt-like arrangement
base_pos = np.array([
    [0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [2.0, 0.0, 2.0], [0.0, 2.0, 2.0],  # Fe
    [1.0, 1.0, 1.0], [3.0, 3.0, 1.0], [3.0, 1.0, 3.0], [1.0, 3.0, 3.0],  # O
])

coords = base_pos[None] + rng.normal(0, 0.1, (nframes, natoms, 3))
box    = np.tile(np.diag([box_size] * 3).reshape(9), (nframes, 1))

type_arr = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)  # Fe Fe Fe Fe O O O O


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gaussian_spectrum(peak_eV, e_min, e_max, npts=100, width_frac=0.10):
    grid  = np.linspace(e_min, e_max, npts)
    width = (e_max - e_min) * width_frac
    return np.exp(-0.5 * ((grid - peak_eV) / width) ** 2)


def write_system(
    path: str,
    sel_type_idx: int,
    atom_slice,       # slice object selecting absorbing atoms
    e_min: float,
    e_max: float,
    peak_center: float,
    peak_shift_scale: float,
    fparam_vec,       # 1-D array of length nfparam (one-hot edge encoding)
):
    os.makedirs(f"{path}/set.000", exist_ok=True)

    # --- structure ---
    np.savetxt(f"{path}/type.raw", type_arr, fmt="%d")
    with open(f"{path}/type_map.raw", "w") as f:
        f.write("Fe\nO\n")
    np.save(f"{path}/set.000/box.npy",   box.astype(np.float64))
    np.save(f"{path}/set.000/coord.npy",
            coords.reshape(nframes, natoms * 3).astype(np.float64))

    # --- fparam: same edge for all frames ---
    fparam = np.tile(fparam_vec, (nframes, 1)).astype(np.float64)
    np.save(f"{path}/set.000/fparam.npy", fparam)

    # --- sel_type: constant per system ---
    sel = np.full((nframes, 1), float(sel_type_idx), dtype=np.float64)
    np.save(f"{path}/set.000/sel_type.npy", sel)

    # --- xas labels ---
    labels = np.zeros((nframes, task_dim), dtype=np.float64)
    for i in range(nframes):
        # peak position shifts slightly with mean x-coordinate of absorbing atoms
        mean_x  = coords[i, atom_slice, 0].mean()
        peak    = peak_center + mean_x * peak_shift_scale
        spectrum = gaussian_spectrum(peak, e_min, e_max)
        labels[i, 0]  = e_min
        labels[i, 1]  = e_max
        labels[i, 2:] = spectrum
    np.save(f"{path}/set.000/xas.npy", labels)

    print(f"  {path}:")
    print(f"    sel_type = {sel_type_idx}  fparam = {fparam_vec.tolist()}")
    print(f"    xas.npy  shape = {labels.shape}")


# ---------------------------------------------------------------------------
# Generate Fe K-edge and O K-edge systems
# ---------------------------------------------------------------------------
print("Generating example XAS training data...")

write_system(
    path             = "data/Fe_K",
    sel_type_idx     = 0,           # Fe is type 0
    atom_slice       = slice(0, 4), # first 4 atoms are Fe
    e_min            = 7100.0,      # Fe K-edge region (eV)
    e_max            = 7250.0,
    peak_center      = 7112.0,      # Fe K-edge energy
    peak_shift_scale = 2.0,         # chemical shift ∝ local environment
    fparam_vec       = np.array([1.0, 0.0, 0.0]),  # K-edge one-hot
)

write_system(
    path             = "data/O_K",
    sel_type_idx     = 1,           # O is type 1
    atom_slice       = slice(4, 8), # last 4 atoms are O
    e_min            = 525.0,       # O K-edge region (eV)
    e_max            = 560.0,
    peak_center      = 535.0,       # O K-edge energy
    peak_shift_scale = 0.5,
    fparam_vec       = np.array([1.0, 0.0, 0.0]),  # also K-edge
)

print(f"\nDone. {nframes} frames per system, task_dim={task_dim}, nfparam={nfparam}")
print("Data written to ./data/Fe_K/ and ./data/O_K/")
