# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single-point energy, forces, and stress with a DPA-4 / SeZM model.

This example loads a trained DPA-4 / SeZM checkpoint, wraps it with
:class:`deepmd.pt.nvalchemi.DPA4Wrapper`, builds an ``nvalchemi`` batch from a
DeePMD-kit data frame, computes a neighbour list, and evaluates the potential
energy, atomic forces, and the Cauchy stress tensor for one configuration.

It is the smallest complete example of the nvalchemi inference path and a good
starting point before running molecular dynamics.

Usage
-----
::

    python single_point.py \
        --model ../lmp/pretrained.pt \
        --data ../../data/data_0
"""

from __future__ import (
    annotations,
)

import argparse
from pathlib import (
    Path,
)

import numpy as np
import torch
from nvalchemi.data import (
    AtomicData,
    Batch,
)
from nvalchemi.neighbors import (
    compute_neighbors,
)

from deepmd.pt.model.model.sezm_model import (
    ELEMENT_TO_Z,
)
from deepmd.pt.nvalchemi import (
    DPA4Wrapper,
)

# 1 eV/A^3 expressed in GPa, for reporting the pressure in familiar units.
_EV_PER_A3_TO_GPA = 160.21766208


def load_frame(
    data_dir: str | Path,
    frame: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one frame from a DeePMD-kit ``npy`` data system.

    Parameters
    ----------
    data_dir
        Directory holding ``type.raw``, ``type_map.raw``, and a ``set.*``
        sub-directory with ``coord.npy`` and ``box.npy``.
    frame
        Frame index to load.
    dtype
        Floating-point dtype for positions and cell.
    device
        Target device.

    Returns
    -------
    atomic_numbers
        ``(N,)`` long tensor of atomic numbers.
    positions
        ``(N, 3)`` Cartesian coordinates in Angstrom.
    cell
        ``(1, 3, 3)`` lattice vectors (rows) in Angstrom.
    """
    data_dir = Path(data_dir)
    set_dir = sorted(data_dir.glob("set.*"))[0]
    coord = np.load(set_dir / "coord.npy")[frame].reshape(-1, 3)
    box = np.load(set_dir / "box.npy")[frame].reshape(3, 3)
    type_index = np.loadtxt(data_dir / "type.raw", dtype=int).reshape(-1)
    type_map = (data_dir / "type_map.raw").read_text().split()
    z = np.array([ELEMENT_TO_Z[type_map[t]] for t in type_index], dtype=np.int64)

    atomic_numbers = torch.tensor(z, dtype=torch.long, device=device)
    positions = torch.tensor(coord, dtype=dtype, device=device)
    cell = torch.tensor(box, dtype=dtype, device=device).reshape(1, 3, 3)
    return atomic_numbers, positions, cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="../lmp/pretrained.pt", help="DPA-4 / SeZM checkpoint (.pt)"
    )
    parser.add_argument(
        "--data", default="../../data/data_0", help="DeePMD-kit data system directory"
    )
    parser.add_argument("--frame", type=int, default=0, help="frame index to evaluate")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    # Wrap the trained model. ``compute_stress=True`` adds the Cauchy stress to
    # the active outputs (it requires a periodic cell).
    model = DPA4Wrapper.from_checkpoint(args.model, device=device, compute_stress=True)
    model.eval()

    atomic_numbers, positions, cell = load_frame(
        args.data, frame=args.frame, device=device
    )
    n_atoms = atomic_numbers.shape[0]
    data = AtomicData(
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
    )
    batch = Batch.from_data_list([data], device=device)

    # Populate ``batch.neighbor_list`` / ``batch.neighbor_list_shifts`` with the
    # cutoff the model declares in its ModelConfig.
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    out = model(batch)

    energy = out["energy"].item()
    forces = out["forces"]
    stress = out["stress"][0]
    pressure = -torch.diagonal(stress).mean().item()  # -tr(sigma)/3

    print(f"model           : {args.model}  (rcut={model.rcut} A)")
    print(f"system          : {n_atoms} atoms, {batch.num_edges} edges")
    print(f"energy          : {energy:.6f} eV  ({energy / n_atoms:.6f} eV/atom)")
    print(f"max |force|     : {forces.norm(dim=-1).max().item():.6f} eV/A")
    print(
        f"rms |force|     : {forces.norm(dim=-1).pow(2).mean().sqrt().item():.6f} eV/A"
    )
    print(
        f"pressure        : {pressure:.6e} eV/A^3  "
        f"({pressure * _EV_PER_A3_TO_GPA:.4f} GPa)"
    )
    print("stress (eV/A^3) :")
    for row in stress.tolist():
        print("  " + "  ".join(f"{v: .6e}" for v in row))


if __name__ == "__main__":
    main()
