# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch

from deepmd.dpmodel.output_def import (
    ModelOutputDef,
    OutputVariableDef,
)
from deepmd.infer.deep_eval import (
    DeepEvalBackend,
)
from deepmd.infer.deep_pot import (
    DeepPot,
)
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

if TYPE_CHECKING:
    import ase.neighborlist

    from deepmd.infer.deep_eval import DeepEval as DeepEvalWrapper


class DeepEval(DeepEvalBackend):
    """PyTorch backend implementaion of DeepEval.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    output_def : ModelOutputDef
        The output definition of the model.
    *args : list
        Positional arguments.
    auto_batch_size : bool or int or AutomaticBatchSize, default: False
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.
    neighbor_list : ase.neighborlist.NewPrimitiveNeighborList, optional
        The ASE neighbor list class to produce the neighbor list. If None, the
        neighbor list will be built natively in the model.
    **kwargs : dict
        Keyword arguments.
    """

    def __init__(
        self,
        model_file: str,
        output_def: ModelOutputDef,
        *args: List[Any],
        auto_batch_size: Union[bool, int, AutoBatchSize] = True,
        neighbor_list: Optional["ase.neighborlist.NewPrimitiveNeighborList"] = None,
        **kwargs: Dict[str, Any],
    ):
        self.output_def = output_def
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

    def get_rcut(self) -> float:
        """Get the cutoff radius of this model."""
        return self.rcut

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

    @property
    def model_type(self) -> "DeepEvalWrapper":
        """The the evaluator of the model type."""
        return DeepPot

    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model."""
        return []

    def get_numb_dos(self) -> int:
        """Get the number of DOS."""
        return 0

    def get_has_efield(self):
        """Check if the model has efield."""
        return False

    def get_ntypes_spin(self):
        """Get the number of spin atom types of this model."""
        return 0

    _OUTDEF_DP2PT: ClassVar[dict] = {
        "energy": "atom_energy",
        "energy_redu": "energy",
        "energy_derv_r": "force",
        # not same as TF...
        "energy_derv_c": "atomic_virial",
        "energy_derv_c_redu": "virial",
    }

    def eval(
        self,
        coords: np.ndarray,
        cells: np.ndarray,
        atom_types: np.ndarray,
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Evaluate the energy, force and virial by using this DP.

        Parameters
        ----------
        coords
            The coordinates of atoms.
            The array should be of size nframes x natoms x 3
        cells
            The cell of the region.
            If None then non-PBC is assumed, otherwise using PBC.
            The array should be of size nframes x 9
        atom_types
            The atom types
            The list should contain natoms ints
        atomic
            Calculate the atomic energy and virial
        fparam
            The frame parameter.
            The array can be of size :
            - nframes x dim_fparam.
            - dim_fparam. Then all frames are assumed to be provided with the same fparam.
        aparam
            The atomic parameter
            The array can be of size :
            - nframes x natoms x dim_aparam.
            - natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
            - dim_aparam. Then all frames and atoms are provided with the same aparam.
        **kwargs
            Other parameters

        Returns
        -------
        output_dict : dict
            The output of the evaluation. The keys are the names of the output
            variables, and the values are the corresponding output arrays.
        """
        if fparam is not None or aparam is not None:
            raise NotImplementedError
        # convert all of the input to numpy array
        atom_types = np.array(atom_types, dtype=np.int32)
        coords = np.array(coords)
        if cells is not None:
            cells = np.array(cells)
        natoms, numb_test = self._get_natoms_and_nframes(
            coords, atom_types, len(atom_types.shape) > 1
        )
        request_defs = self._get_request_defs(atomic)
        out = self._eval_func(self._eval_model, numb_test, natoms)(
            coords, cells, atom_types, request_defs
        )
        return dict(
            zip(
                [x.name for x in request_defs],
                out,
            )
        )

    def _get_request_defs(self, atomic: bool) -> List[OutputVariableDef]:
        """Get the requested output definitions.

        When atomic is True, all output_def are requested.
        When atomic is False, only energy (tensor), force, and virial
        are requested.

        Parameters
        ----------
        atomic : bool
            Whether to request the atomic output.

        Returns
        -------
        list[OutputVariableDef]
            The requested output definitions.
        """
        if atomic:
            return list(self.output_def.var_defs.values())
        else:
            return [
                x
                for x in self.output_def.var_defs.values()
                if x.name.endswith("_redu") or x.name.endswith("_derv_r")
            ]

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
        atom_types: np.ndarray,
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
        request_defs: List[OutputVariableDef],
    ):
        model = self.dp.to(DEVICE)

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

        do_atomic_virial = any(x.name.endswith("_derv_c") for x in request_defs)
        batch_output = model(
            coord_input, type_input, box=box_input, do_atomic_virial=do_atomic_virial
        )
        if isinstance(batch_output, tuple):
            batch_output = batch_output[0]

        results = []
        for odef in request_defs:
            pt_name = self._OUTDEF_DP2PT[odef.name]
            if pt_name in batch_output:
                shape = self._get_output_shape(odef.name, nframes, natoms, odef.shape)
                out = batch_output[pt_name].reshape(shape).detach().cpu().numpy()
                results.append(out)
        return tuple(results)

    def _get_output_shape(self, name, nframes, natoms, shape):
        if "_redu" in name:
            if "_derv_c" in name:
                # virial
                return [nframes, *shape[:-2], 9]
            else:
                # energy
                return [nframes, *shape, 1]
        else:
            if "_derv_c" in name:
                # atom_virial
                return [nframes, *shape[:-2], natoms, 9]
            elif "_derv_r" in name:
                # force
                return [nframes, *shape[:-1], natoms, 3]
            else:
                # atom_energy, atom_tensor
                # Something wrong here?
                # return [nframes, *shape, natoms, 1]
                return [nframes, natoms, *shape, 1]


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
