# SPDX-License-Identifier: LGPL-3.0-or-later
import pathlib
from typing import (
    Optional,
    Union,
)

import numpy as np
import torch

from deepmd.common import j_loader as dp_j_loader
from deepmd.main import (
    main,
)
from deepmd.pt.utils.env import (
    DEVICE,
    GLOBAL_PT_FLOAT_PRECISION,
)

tests_path = pathlib.Path(__file__).parent.absolute()


def j_loader(filename):
    return dp_j_loader(tests_path / filename)


def run_dp(cmd: str) -> int:
    """Run DP directly from the entry point instead of the subprocess.

    It is quite slow to start DeePMD-kit with subprocess.

    Parameters
    ----------
    cmd : str
        The command to run.

    Returns
    -------
    int
        Always returns 0.
    """
    cmds = cmd.split()
    if cmds[0] == "dp":
        cmds = cmds[1:]
    else:
        raise RuntimeError("The command is not dp")

    main(cmds)
    return 0


def eval_model(
    model,
    coords: Union[np.ndarray, torch.Tensor],
    cells: Optional[Union[np.ndarray, torch.Tensor]],
    atom_types: Union[np.ndarray, torch.Tensor, list[int]],
    spins: Optional[Union[np.ndarray, torch.Tensor]] = None,
    atomic: bool = False,
    infer_batch_size: int = 2,
    denoise: bool = False,
):
    model = model.to(DEVICE)
    energy_out = []
    atomic_energy_out = []
    force_out = []
    force_mag_out = []
    virial_out = []
    atomic_virial_out = []
    updated_coord_out = []
    logits_out = []
    err_msg = (
        f"All inputs should be the same format, "
        f"but found {type(coords)}, {type(cells)}, {type(atom_types)} instead! "
    )
    return_tensor = True
    if isinstance(coords, torch.Tensor):
        if cells is not None:
            assert isinstance(cells, torch.Tensor), err_msg
        if spins is not None:
            assert isinstance(spins, torch.Tensor), err_msg
        assert isinstance(atom_types, torch.Tensor) or isinstance(atom_types, list)
        atom_types = torch.tensor(atom_types, dtype=torch.int32, device=DEVICE)
    elif isinstance(coords, np.ndarray):
        if cells is not None:
            assert isinstance(cells, np.ndarray), err_msg
        if spins is not None:
            assert isinstance(spins, np.ndarray), err_msg
        assert isinstance(atom_types, np.ndarray) or isinstance(atom_types, list)
        atom_types = np.array(atom_types, dtype=np.int32)
        return_tensor = False

    nframes = coords.shape[0]
    if len(atom_types.shape) == 1:
        natoms = len(atom_types)
        if isinstance(atom_types, torch.Tensor):
            atom_types = torch.tile(atom_types.unsqueeze(0), [nframes, 1]).reshape(
                nframes, -1
            )
        else:
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
    else:
        natoms = len(atom_types[0])

    coord_input = torch.tensor(
        coords.reshape([-1, natoms, 3]), dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
    )
    spin_input = None
    if spins is not None:
        spin_input = torch.tensor(
            spins.reshape([-1, natoms, 3]),
            dtype=GLOBAL_PT_FLOAT_PRECISION,
            device=DEVICE,
        )
    has_spin = getattr(model, "has_spin", False)
    if callable(has_spin):
        has_spin = has_spin()
    type_input = torch.tensor(atom_types, dtype=torch.long, device=DEVICE)
    box_input = None
    if cells is None:
        pbc = False
    else:
        pbc = True
        box_input = torch.tensor(
            cells.reshape([-1, 3, 3]), dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
        )
    num_iter = int((nframes + infer_batch_size - 1) / infer_batch_size)

    for ii in range(num_iter):
        batch_coord = coord_input[ii * infer_batch_size : (ii + 1) * infer_batch_size]
        batch_atype = type_input[ii * infer_batch_size : (ii + 1) * infer_batch_size]
        batch_box = None
        batch_spin = None
        if spin_input is not None:
            batch_spin = spin_input[ii * infer_batch_size : (ii + 1) * infer_batch_size]
        if pbc:
            batch_box = box_input[ii * infer_batch_size : (ii + 1) * infer_batch_size]
        input_dict = {
            "coord": batch_coord,
            "atype": batch_atype,
            "box": batch_box,
            "do_atomic_virial": atomic,
        }
        if has_spin:
            input_dict["spin"] = batch_spin
        batch_output = model(**input_dict)
        if isinstance(batch_output, tuple):
            batch_output = batch_output[0]
        if not return_tensor:
            if "energy" in batch_output:
                energy_out.append(batch_output["energy"].detach().cpu().numpy())
            if "atom_energy" in batch_output:
                atomic_energy_out.append(
                    batch_output["atom_energy"].detach().cpu().numpy()
                )
            if "force" in batch_output:
                force_out.append(batch_output["force"].detach().cpu().numpy())
            if "force_mag" in batch_output:
                force_mag_out.append(batch_output["force_mag"].detach().cpu().numpy())
            if "virial" in batch_output:
                virial_out.append(batch_output["virial"].detach().cpu().numpy())
            if "atom_virial" in batch_output:
                atomic_virial_out.append(
                    batch_output["atom_virial"].detach().cpu().numpy()
                )
            if "updated_coord" in batch_output:
                updated_coord_out.append(
                    batch_output["updated_coord"].detach().cpu().numpy()
                )
            if "logits" in batch_output:
                logits_out.append(batch_output["logits"].detach().cpu().numpy())
        else:
            if "energy" in batch_output:
                energy_out.append(batch_output["energy"])
            if "atom_energy" in batch_output:
                atomic_energy_out.append(batch_output["atom_energy"])
            if "force" in batch_output:
                force_out.append(batch_output["force"])
            if "force_mag" in batch_output:
                force_mag_out.append(batch_output["force_mag"])
            if "virial" in batch_output:
                virial_out.append(batch_output["virial"])
            if "atom_virial" in batch_output:
                atomic_virial_out.append(batch_output["atom_virial"])
            if "updated_coord" in batch_output:
                updated_coord_out.append(batch_output["updated_coord"])
            if "logits" in batch_output:
                logits_out.append(batch_output["logits"])
    if not return_tensor:
        energy_out = (
            np.concatenate(energy_out) if energy_out else np.zeros([nframes, 1])  # pylint: disable=no-explicit-dtype
        )
        atomic_energy_out = (
            np.concatenate(atomic_energy_out)
            if atomic_energy_out
            else np.zeros([nframes, natoms, 1])  # pylint: disable=no-explicit-dtype
        )
        force_out = (
            np.concatenate(force_out) if force_out else np.zeros([nframes, natoms, 3])  # pylint: disable=no-explicit-dtype
        )
        force_mag_out = (
            np.concatenate(force_mag_out)
            if force_mag_out
            else np.zeros([nframes, natoms, 3])  # pylint: disable=no-explicit-dtype
        )
        virial_out = (
            np.concatenate(virial_out) if virial_out else np.zeros([nframes, 3, 3])  # pylint: disable=no-explicit-dtype
        )
        atomic_virial_out = (
            np.concatenate(atomic_virial_out)
            if atomic_virial_out
            else np.zeros([nframes, natoms, 3, 3])  # pylint: disable=no-explicit-dtype
        )
        updated_coord_out = (
            np.concatenate(updated_coord_out) if updated_coord_out else None
        )
        logits_out = np.concatenate(logits_out) if logits_out else None
    else:
        energy_out = (
            torch.cat(energy_out)
            if energy_out
            else torch.zeros(
                [nframes, 1], dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
            )
        )
        atomic_energy_out = (
            torch.cat(atomic_energy_out)
            if atomic_energy_out
            else torch.zeros(
                [nframes, natoms, 1], dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
            )
        )
        force_out = (
            torch.cat(force_out)
            if force_out
            else torch.zeros(
                [nframes, natoms, 3], dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
            )
        )
        force_mag_out = (
            torch.cat(force_mag_out)
            if force_mag_out
            else torch.zeros(
                [nframes, natoms, 3], dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
            )
        )
        virial_out = (
            torch.cat(virial_out)
            if virial_out
            else torch.zeros(
                [nframes, 3, 3], dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
            )
        )
        atomic_virial_out = (
            torch.cat(atomic_virial_out)
            if atomic_virial_out
            else torch.zeros(
                [nframes, natoms, 3, 3], dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
            )
        )
        updated_coord_out = torch.cat(updated_coord_out) if updated_coord_out else None
        logits_out = torch.cat(logits_out) if logits_out else None
    if denoise:
        return updated_coord_out, logits_out
    else:
        results_dict = {
            "energy": energy_out,
            "force": force_out,
            "virial": virial_out,
        }
        if has_spin:
            results_dict["force_mag"] = force_mag_out
        if atomic:
            results_dict["atom_energy"] = atomic_energy_out
            results_dict["atom_virial"] = atomic_virial_out
        return results_dict
