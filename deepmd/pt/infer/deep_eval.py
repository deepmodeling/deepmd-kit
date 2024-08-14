# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch

from deepmd.dpmodel.output_def import (
    ModelOutputDef,
    OutputVariableCategory,
    OutputVariableDef,
)
from deepmd.infer.deep_dipole import (
    DeepDipole,
)
from deepmd.infer.deep_dos import (
    DeepDOS,
)
from deepmd.infer.deep_eval import DeepEval as DeepEvalWrapper
from deepmd.infer.deep_eval import (
    DeepEvalBackend,
)
from deepmd.infer.deep_polar import (
    DeepGlobalPolar,
    DeepPolar,
)
from deepmd.infer.deep_pot import (
    DeepPot,
)
from deepmd.infer.deep_wfc import (
    DeepWFC,
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
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)

if TYPE_CHECKING:
    import ase.neighborlist


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
        output_def: ModelOutputDef,  # anchor noted: not callable
        *args: Any,
        auto_batch_size: Union[bool, int, AutoBatchSize] = True,
        neighbor_list: Optional["ase.neighborlist.NewPrimitiveNeighborList"] = None,
        head: Optional[str] = None,
        **kwargs: Any,
    ):
        self.output_def = output_def
        # print(f"--output_def.keys in __init__ in pt/deep_eval.py: {self.output_def.var_defs.keys()}")  # anchor added
        # if "energy" in list(self.output_def.var_defs.keys()):  # anchor added
        #     self.output_def.var_defs["energy"].r_hessian = True  # if model type is EHM: useless
        #     print(f"--output_def.var_defs['energy'].r_hessian is set to True in __init__ in pt/deep_eval.py")
        #     print(f"--output_def.keys in __init__ in pt/deep_eval.py with r_h True: {self.output_def.var_defs.keys()}")
        # print(f"--output_def.values in __init__ in pt/deep_eval.py: "
        #       f"{type(list(self.output_def.var_defs.values())[0])}")  # anchor added: OutputVariableDef
        # print(f"--output_def.def_outp in __init__ in pt/deep_eval.py of {type(self.output_def.def_outp)}: "
        #       f"{self.output_def.def_outp}")  # anchor added: FittingOutputDef
        # print(f"--output_def.def_derv_r in __init__ in pt/deep_eval.py of {type(self.output_def.def_derv_r)}: "
        #       f"{self.output_def.def_derv_r}")  # anchor added: dict: {energy_derv_r: OutputVariableDef}
        # print(f"--output_def.def_hess_r in __init__ in pt/deep_eval.py of {type(self.output_def.def_hess_r)}: "
        #       f"{self.output_def.def_hess_r}")  # anchor added: dict: {}
        self.model_path = model_file
        if str(self.model_path).endswith(".pt"):
            state_dict = torch.load(model_file, map_location=env.DEVICE)
            # print(f"--state_dict of {type(state_dict)} in pt/deep_eval.py: {state_dict.keys()}")  # anchor added
            if "model" in state_dict:
                state_dict = state_dict["model"]
                # print(f"--state_dict['model'] of {type(state_dict)} in pt/deep_eval.py: {state_dict.keys()}")  # anchor added
            self.input_param = state_dict["_extra_state"]["model_params"]
            # print(f"--input param of {type(state_dict)} in pt/deep_eval.py: {self.input_param.keys()}")  # anchor added
            # print(f"--state_dict['_extra_state'] of {type(state_dict['_extra_state'])} in pt/deep_eval.py: "
            #       f"{state_dict['_extra_state'].keys()}")  # anchor added
            self.multi_task = "model_dict" in self.input_param
            if self.multi_task:
                model_keys = list(self.input_param["model_dict"].keys())
                assert (
                    head is not None
                ), f"Head must be set for multitask model! Available heads are: {model_keys}"
                assert (
                    head in model_keys
                ), f"No head named {head} in model! Available heads are: {model_keys}"
                self.input_param = self.input_param["model_dict"][head]
                state_dict_head = {"_extra_state": state_dict["_extra_state"]}
                for item in state_dict:
                    if f"model.{head}." in item:
                        state_dict_head[
                            item.replace(f"model.{head}.", "model.Default.")
                        ] = state_dict[item].clone()
                state_dict = state_dict_head
            model = get_model(self.input_param).to(DEVICE)
            print(f"--type of model in __init__ in pt/deep_eval: {type(model)}")  # anchor added -- EHM
            # model = torch.jit.script(model)  # anchor commented out
            # print(f"--type of model in pt/infer/deep_eval after torch.jit: {type(model)}")  # anchor added
            self.dp = ModelWrapper(model)
            # print(f"--dp in pt/infer/deep_eval of {type(self.dp)}: {self.dp}")  # anchor added
            self.dp.load_state_dict(state_dict)
            # print(f"--dp in pt/infer/deep_eval after load of {type(self.dp)}: {self.dp}")  # anchor added
            print(f"--dp.model in pt/deep_eval.py of {type(self.dp.model)}: {self.dp.model.keys()}")  # anchor added
            print(f"--dp.model.default in pt/deep_eval.py of {type(self.dp.model['Default'])}")  # anchor added -- EHM
        elif str(self.model_path).endswith(".pth"):
            model = torch.jit.load(model_file, map_location=env.DEVICE)
            self.dp = ModelWrapper(model)
        else:
            raise ValueError("Unknown model file format!")
        self.rcut = self.dp.model["Default"].get_rcut()
        self.type_map = self.dp.model["Default"].get_type_map()
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
        self._has_spin = getattr(self.dp.model["Default"], "has_spin", False)
        if callable(self._has_spin):
            self._has_spin = self._has_spin()
        print(f"--class DeepEval in pt/deep_eval.py is activated")  # anchor added

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
        return self.dp.model["Default"].get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        return self.dp.model["Default"].get_dim_aparam()

    @property
    def model_type(self) -> Type["DeepEvalWrapper"]:
        """The the evaluator of the model type."""
        model_output_type = self.dp.model["Default"].model_output_type()
        print(f"--model_output_type in model_type in pt/deep_eval.py of {type(model_output_type)}: "
              f"{model_output_type} of {type(model_output_type[0])}")  # anchor added -- list ['energy', 'mask'] str
        if "energy" in model_output_type:
            print(f"--model_output_type of DeepPot is returned in model_type_in pt/deep_eval.py")  # anchor added
            return DeepPot
        elif "dos" in model_output_type:
            return DeepDOS
        elif "dipole" in model_output_type:
            return DeepDipole
        elif "polar" in model_output_type:
            return DeepPolar
        elif "global_polar" in model_output_type:
            return DeepGlobalPolar
        elif "wfc" in model_output_type:
            return DeepWFC
        else:
            raise RuntimeError("Unknown model type")

    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.dp.model["Default"].get_sel_type()

    def get_numb_dos(self) -> int:
        """Get the number of DOS."""
        return self.dp.model["Default"].get_numb_dos()

    def get_has_efield(self):
        """Check if the model has efield."""
        return False

    def get_ntypes_spin(self):
        """Get the number of spin atom types of this model. Only used in old implement."""
        return 0

    def get_has_spin(self):
        """Check if the model has spin atom types."""
        return self._has_spin

    def eval(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        **kwargs: Any,
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
        # convert all of the input to numpy array
        print(f"--eval in pt/deep_eval.py is activated")  # anchor added
        atom_types = np.array(atom_types, dtype=np.int32)
        coords = np.array(coords)
        if cells is not None:
            cells = np.array(cells)
        natoms, numb_test = self._get_natoms_and_nframes(
            coords, atom_types, len(atom_types.shape) > 1
        )
        request_defs = self._get_request_defs(atomic)
        print(f"--request_defs via _get_request_defs in eval in pt/deep_eval.py: {len(request_defs)} {type(request_defs[0])}")  # anchor added
        if "spin" not in kwargs or kwargs["spin"] is None:
            print(f"--type of _eval_model in eval in pt/deep_eval.py: {type(self._eval_model)}")  # anchor added -- <class 'method'>
            out = self._eval_func(self._eval_model, numb_test, natoms)(
                coords, cells, atom_types, fparam, aparam, request_defs
            )
        else:
            out = self._eval_func(self._eval_model_spin, numb_test, natoms)(
                coords,
                cells,
                atom_types,
                np.array(kwargs["spin"]),
                fparam,
                aparam,
                request_defs,
            )
        # anchor noted: out here are only of 5 contents without 'energy_derv_c_derv_c'
        print(f"--request_defs in eval in pt/deep_eval.py: {[x.name for x in request_defs]}")  # anchor added
        print(f"--out in eval in pt/deep_eval.py: {[o.shape for o in out]}")  # anchor added
        # anchor noted: request_defs: ['energy', 'energy_redu', 'energy_derv_r', 'energy_derv_c_redu', 'mask']
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
        print(f"--output_def in _get_request_defs in pt/deep_eval.py: {self.output_def.var_defs.keys()}")  # anchor added
        if atomic:
            print(f"--atomic in _get_request_defs in pt/deep_eval.py is called")  # anchor added
            return list(self.output_def.var_defs.values())
        else:
            print(f"--atomic in _get_request_defs in pt/deep_eval.py is not called")  # anchor added
            print(f"--self.output_def.var_defs.keys in _get_request_defs in pt/deep_eval.py: {self.output_def.var_defs.keys()}")  # anchor added
            print(f"--self.output_def.var_defs.value.category in _get_request_defs in pt/deep_eval.py: {[x.category for x in self.output_def.var_defs.values()]}")  # anchor added
            # print(f"--self.output_def.var_defs.values in _get_request_defs in pt/deep_eval.py: {self.output_def.var_defs.values()}")  # anchor added
            return [
                x
                for x in self.output_def.var_defs.values()
                if x.category
                in (
                    OutputVariableCategory.OUT,
                    OutputVariableCategory.REDU,
                    OutputVariableCategory.DERV_R,
                    OutputVariableCategory.DERV_C_REDU,
                    OutputVariableCategory.DERV_R_DERV_R,  # anchor added
                )
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
        print(f"--type of eval_func in _eval_func in pt/deep_eval.py: {type(eval_func)}")  # anchor added -- <class 'function'>
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
        fparam: Optional[np.ndarray],
        aparam: Optional[np.ndarray],
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
            coords.reshape([-1, natoms, 3]),
            dtype=GLOBAL_PT_FLOAT_PRECISION,
            device=DEVICE,
        )
        type_input = torch.tensor(atom_types, dtype=torch.long, device=DEVICE)
        if cells is not None:
            box_input = torch.tensor(
                cells.reshape([-1, 3, 3]),
                dtype=GLOBAL_PT_FLOAT_PRECISION,
                device=DEVICE,
            )
        else:
            box_input = None
        if fparam is not None:
            fparam_input = to_torch_tensor(fparam.reshape(-1, self.get_dim_fparam()))
        else:
            fparam_input = None
        if aparam is not None:
            aparam_input = to_torch_tensor(
                aparam.reshape(-1, natoms, self.get_dim_aparam())
            )
        else:
            aparam_input = None
        do_atomic_virial = any(
            x.category == OutputVariableCategory.DERV_C for x in request_defs
        )
        batch_output = model(
            coord_input,
            type_input,
            box=box_input,
            do_atomic_virial=do_atomic_virial,
            fparam=fparam_input,
            aparam=aparam_input,
        )
        if isinstance(batch_output, tuple):
            batch_output = batch_output[0]
        print(f"--batch_output in _eval_model in pt/deep_eval.py of {type(batch_output)}: "
              f"{batch_output.keys()}")  # anchor added: dict with hessian in

        results = []
        print(f"--request_defs in _eval_model in pt/deep_eval.py of {type(request_defs)}: {request_defs}")  # anchor added
        for odef in request_defs:
            pt_name = self._OUTDEF_DP2BACKEND[odef.name]
            # print(f"--pt_name in _eval_model in pt/deep_eval.py: {pt_name}")  # anchor added
            if pt_name in batch_output:
                shape = self._get_output_shape(odef, nframes, natoms)
                print(f"--pt {pt_name} in _eval_model in pt/deep_eval.py shape: {shape}, odef: {odef}")  # anchor added
                out = batch_output[pt_name].reshape(shape).detach().cpu().numpy()
                results.append(out)
            else:
                shape = self._get_output_shape(odef, nframes, natoms)
                results.append(np.full(np.abs(shape), np.nan))  # this is kinda hacky
        return tuple(results)

    def _eval_model_spin(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        spins: np.ndarray,
        fparam: Optional[np.ndarray],
        aparam: Optional[np.ndarray],
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
            coords.reshape([-1, natoms, 3]),
            dtype=GLOBAL_PT_FLOAT_PRECISION,
            device=DEVICE,
        )
        type_input = torch.tensor(atom_types, dtype=torch.long, device=DEVICE)
        spin_input = torch.tensor(
            spins.reshape([-1, natoms, 3]),
            dtype=GLOBAL_PT_FLOAT_PRECISION,
            device=DEVICE,
        )
        if cells is not None:
            box_input = torch.tensor(
                cells.reshape([-1, 3, 3]),
                dtype=GLOBAL_PT_FLOAT_PRECISION,
                device=DEVICE,
            )
        else:
            box_input = None
        if fparam is not None:
            fparam_input = to_torch_tensor(fparam.reshape(-1, self.get_dim_fparam()))
        else:
            fparam_input = None
        if aparam is not None:
            aparam_input = to_torch_tensor(
                aparam.reshape(-1, natoms, self.get_dim_aparam())
            )
        else:
            aparam_input = None

        do_atomic_virial = any(
            x.category == OutputVariableCategory.DERV_C_REDU for x in request_defs
        )
        batch_output = model(
            coord_input,
            type_input,
            spin=spin_input,
            box=box_input,
            do_atomic_virial=do_atomic_virial,
            fparam=fparam_input,
            aparam=aparam_input,
        )
        if isinstance(batch_output, tuple):
            batch_output = batch_output[0]

        results = []
        for odef in request_defs:
            pt_name = self._OUTDEF_DP2BACKEND[odef.name]
            if pt_name in batch_output:
                shape = self._get_output_shape(odef, nframes, natoms)
                out = batch_output[pt_name].reshape(shape).detach().cpu().numpy()
                results.append(out)
            else:
                shape = self._get_output_shape(odef, nframes, natoms)
                results.append(np.full(np.abs(shape), np.nan))  # this is kinda hacky
        return tuple(results)

    def _get_output_shape(self, odef, nframes, natoms):
        if odef.category == OutputVariableCategory.DERV_C_REDU:
            # virial
            return [nframes, *odef.shape[:-1], 9]
        elif odef.category == OutputVariableCategory.REDU:
            # energy
            return [nframes, *odef.shape, 1]
        elif odef.category == OutputVariableCategory.DERV_C:
            # atom_virial
            return [nframes, *odef.shape[:-1], natoms, 9]
        elif odef.category == OutputVariableCategory.DERV_R:
            # force
            return [nframes, *odef.shape[:-1], natoms, 3]
        elif odef.category == OutputVariableCategory.OUT:
            # atom_energy, atom_tensor
            # Something wrong here?
            # return [nframes, *shape, natoms, 1]
            return [nframes, natoms, *odef.shape, 1]
        elif odef.category == OutputVariableCategory.DERV_R_DERV_R:  # anchor added
            return [nframes, 3 * natoms, 3 * natoms]
            # return [nframes, *odef.shape, 3 * natoms, 3 * natoms]
        else:
            raise RuntimeError("unknown category")


# For tests only
def eval_model(
    model,
    coords: Union[np.ndarray, torch.Tensor],
    cells: Optional[Union[np.ndarray, torch.Tensor]],
    atom_types: Union[np.ndarray, torch.Tensor, List[int]],
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
        atom_types = torch.tensor(atom_types, dtype=torch.long, device=DEVICE)
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
        force_mag_out = (
            np.concatenate(force_mag_out)
            if force_mag_out
            else np.zeros([nframes, natoms, 3])
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

