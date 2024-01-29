# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch

from deepmd.infer.deep_pot import DeepPot as DeepPotBase
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.pt.utils.env import (
    DEVICE,
    GLOBAL_PT_FLOAT_PRECISION,
)


class DeepEval:
    def __init__(
        self,
        model_file: "Path",
        auto_batch_size: Union[bool, int, AutoBatchSize] = True,
    ):
        self.model_path = model_file
        state_dict = torch.load(model_file, map_location=env.DEVICE)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.input_param = state_dict["_extra_state"]["model_params"]
        self.input_param["resuming"] = True
        self.multi_task = "model_dict" in self.input_param
        assert not self.multi_task, "multitask mode currently not supported!"
        self.type_split = self.input_param["descriptor"]["type"] in ["se_e2_a"]
        self.type_map = self.input_param["type_map"]
        self.dp = ModelWrapper(get_model(self.input_param, None).to(DEVICE))
        self.dp.load_state_dict(state_dict)
        self.rcut = self.dp.model["Default"].descriptor.get_rcut()
        self.sec = np.cumsum(self.dp.model["Default"].descriptor.get_sel())
        if isinstance(auto_batch_size, bool):
            if auto_batch_size:
                self.auto_batch_size = AutoBatchSize()
            else:
                self.auto_batch_size = None
        elif isinstance(auto_batch_size, int):
            self.auto_batch_size = AutoBatchSize(auto_batch_size)
        elif isinstance(auto_batch_size, AutoBatchSize):
            self.auto_batch_size = auto_batch_size
        else:
            raise TypeError("auto_batch_size should be bool, int, or AutoBatchSize")

    def eval(
        self,
        coords: Union[np.ndarray, torch.Tensor],
        cells: Optional[Union[np.ndarray, torch.Tensor]],
        atom_types: Union[np.ndarray, torch.Tensor, List[int]],
        atomic: bool = False,
    ):
        raise NotImplementedError


class DeepPot(DeepEval, DeepPotBase):
    def __init__(
        self,
        model_file: "Path",
        auto_batch_size: Union[bool, int, AutoBatchSize] = True,
        neighbor_list=None,
    ):
        if neighbor_list is not None:
            raise NotImplementedError
        super().__init__(
            model_file,
            auto_batch_size=auto_batch_size,
        )

    def eval(
        self,
        coords: np.ndarray,
        cells: np.ndarray,
        atom_types: List[int],
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        efield: Optional[np.ndarray] = None,
        mixed_type: bool = False,
    ):
        if fparam is not None or aparam is not None or efield is not None:
            raise NotImplementedError
        # convert all of the input to numpy array
        atom_types = np.array(atom_types, dtype=np.int32)
        coords = np.array(coords)
        if cells is not None:
            cells = np.array(cells)
        natoms, numb_test = self._get_natoms_and_nframes(
            coords, atom_types, len(atom_types.shape) > 1
        )
        return self._eval_func(self._eval_model, numb_test, natoms)(
            coords, cells, atom_types, atomic
        )

    def _eval_func(self, inner_func: Callable, numb_test: int, natoms: int) -> Callable:
        """Wrapper method with auto batch size.

        Parameters
        ----------
        inner_func : Callable
            the method to be wrapped
        numb_test : int
            number of tests
        natoms : int
            number of atoms

        Returns
        -------
        Callable
            the wrapper
        """
        if self.auto_batch_size is not None:

            def eval_func(*args, **kwargs):
                return self.auto_batch_size.execute_all(
                    inner_func, numb_test, natoms, *args, **kwargs
                )

        else:
            eval_func = inner_func
        return eval_func

    def _get_natoms_and_nframes(
        self,
        coords: np.ndarray,
        atom_types: Union[List[int], np.ndarray],
        mixed_type: bool = False,
    ) -> Tuple[int, int]:
        if mixed_type:
            natoms = len(atom_types[0])
        else:
            natoms = len(atom_types)
        if natoms == 0:
            assert coords.size == 0
        else:
            coords = np.reshape(np.array(coords), [-1, natoms * 3])
        nframes = coords.shape[0]
        return natoms, nframes

    def _eval_model(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        atomic: bool = False,
    ):
        model = self.dp.to(DEVICE)
        energy_out = None
        atomic_energy_out = None
        force_out = None
        virial_out = None
        atomic_virial_out = None

        nframes = coords.shape[0]
        if len(atom_types.shape) == 1:
            natoms = len(atom_types)
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        else:
            natoms = len(atom_types[0])

        coord_input = torch.tensor(
            coords.reshape([-1, natoms, 3]), dtype=GLOBAL_PT_FLOAT_PRECISION
        ).to(DEVICE)
        type_input = torch.tensor(atom_types, dtype=torch.long).to(DEVICE)
        if cells is not None:
            box_input = torch.tensor(
                cells.reshape([-1, 3, 3]), dtype=GLOBAL_PT_FLOAT_PRECISION
            ).to(DEVICE)
        else:
            box_input = None

        batch_output = model(
            coord_input, type_input, box=box_input, do_atomic_virial=atomic
        )
        if isinstance(batch_output, tuple):
            batch_output = batch_output[0]
        energy_out = batch_output["energy"].reshape(nframes, 1).detach().cpu().numpy()
        if "atom_energy" in batch_output:
            atomic_energy_out = (
                batch_output["atom_energy"]
                .reshape(nframes, natoms, 1)
                .detach()
                .cpu()
                .numpy()
            )
        force_out = (
            batch_output["force"].reshape(nframes, natoms, 3).detach().cpu().numpy()
        )
        virial_out = batch_output["virial"].reshape(nframes, 9).detach().cpu().numpy()
        if "atomic_virial" in batch_output:
            atomic_virial_out = (
                batch_output["atomic_virial"]
                .reshape(nframes, natoms, 9)
                .detach()
                .cpu()
                .numpy()
            )

        if not atomic:
            return energy_out, force_out, virial_out
        else:
            return (
                energy_out,
                force_out,
                virial_out,
                atomic_energy_out,
                atomic_virial_out,
            )

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        return len(self.type_map)

    def get_type_map(self) -> List[str]:
        """Get the type map (element name of the atom types) of this model."""
        return self.type_map

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this DP."""
        return 0

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        return 0


# For tests only
def eval_model(
    model,
    coords: Union[np.ndarray, torch.Tensor],
    cells: Optional[Union[np.ndarray, torch.Tensor]],
    atom_types: Union[np.ndarray, torch.Tensor, List[int]],
    atomic: bool = False,
    infer_batch_size: int = 2,
    denoise: bool = False,
):
    model = model.to(DEVICE)
    energy_out = []
    atomic_energy_out = []
    force_out = []
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
        assert isinstance(atom_types, torch.Tensor) or isinstance(atom_types, list)
        atom_types = torch.tensor(atom_types, dtype=torch.long).to(DEVICE)
    elif isinstance(coords, np.ndarray):
        if cells is not None:
            assert isinstance(cells, np.ndarray), err_msg
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
        coords.reshape([-1, natoms, 3]), dtype=GLOBAL_PT_FLOAT_PRECISION
    ).to(DEVICE)
    type_input = torch.tensor(atom_types, dtype=torch.long).to(DEVICE)
    box_input = None
    if cells is None:
        pbc = False
    else:
        pbc = True
        box_input = torch.tensor(
            cells.reshape([-1, 3, 3]), dtype=GLOBAL_PT_FLOAT_PRECISION
        ).to(DEVICE)
    num_iter = int((nframes + infer_batch_size - 1) / infer_batch_size)

    for ii in range(num_iter):
        batch_coord = coord_input[ii * infer_batch_size : (ii + 1) * infer_batch_size]
        batch_atype = type_input[ii * infer_batch_size : (ii + 1) * infer_batch_size]
        batch_box = None
        if pbc:
            batch_box = box_input[ii * infer_batch_size : (ii + 1) * infer_batch_size]
        batch_output = model(batch_coord, batch_atype, box=batch_box)
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
            if "virial" in batch_output:
                virial_out.append(batch_output["virial"].detach().cpu().numpy())
            if "atomic_virial" in batch_output:
                atomic_virial_out.append(
                    batch_output["atomic_virial"].detach().cpu().numpy()
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
            if "virial" in batch_output:
                virial_out.append(batch_output["virial"])
            if "atomic_virial" in batch_output:
                atomic_virial_out.append(batch_output["atomic_virial"])
            if "updated_coord" in batch_output:
                updated_coord_out.append(batch_output["updated_coord"])
            if "logits" in batch_output:
                logits_out.append(batch_output["logits"])
    if not return_tensor:
        energy_out = (
            np.concatenate(energy_out) if energy_out else np.zeros([nframes, 1])
        )
        atomic_energy_out = (
            np.concatenate(atomic_energy_out)
            if atomic_energy_out
            else np.zeros([nframes, natoms, 1])
        )
        force_out = (
            np.concatenate(force_out) if force_out else np.zeros([nframes, natoms, 3])
        )
        virial_out = (
            np.concatenate(virial_out) if virial_out else np.zeros([nframes, 3, 3])
        )
        atomic_virial_out = (
            np.concatenate(atomic_virial_out)
            if atomic_virial_out
            else np.zeros([nframes, natoms, 3, 3])
        )
        updated_coord_out = (
            np.concatenate(updated_coord_out) if updated_coord_out else None
        )
        logits_out = np.concatenate(logits_out) if logits_out else None
    else:
        energy_out = (
            torch.cat(energy_out)
            if energy_out
            else torch.zeros([nframes, 1], dtype=GLOBAL_PT_FLOAT_PRECISION).to(DEVICE)
        )
        atomic_energy_out = (
            torch.cat(atomic_energy_out)
            if atomic_energy_out
            else torch.zeros([nframes, natoms, 1], dtype=GLOBAL_PT_FLOAT_PRECISION).to(
                DEVICE
            )
        )
        force_out = (
            torch.cat(force_out)
            if force_out
            else torch.zeros([nframes, natoms, 3], dtype=GLOBAL_PT_FLOAT_PRECISION).to(
                DEVICE
            )
        )
        virial_out = (
            torch.cat(virial_out)
            if virial_out
            else torch.zeros([nframes, 3, 3], dtype=GLOBAL_PT_FLOAT_PRECISION).to(
                DEVICE
            )
        )
        atomic_virial_out = (
            torch.cat(atomic_virial_out)
            if atomic_virial_out
            else torch.zeros(
                [nframes, natoms, 3, 3], dtype=GLOBAL_PT_FLOAT_PRECISION
            ).to(DEVICE)
        )
        updated_coord_out = torch.cat(updated_coord_out) if updated_coord_out else None
        logits_out = torch.cat(logits_out) if logits_out else None
    if denoise:
        return updated_coord_out, logits_out
    else:
        if not atomic:
            return energy_out, force_out, virial_out
        else:
            return (
                energy_out,
                force_out,
                virial_out,
                atomic_energy_out,
                atomic_virial_out,
            )
