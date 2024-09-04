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
import paddle

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
from deepmd.pd.model.model import (
    get_model,
)
from deepmd.pd.train.wrapper import (
    ModelWrapper,
)
from deepmd.pd.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.pd.utils.env import (
    DEVICE,
    GLOBAL_PD_FLOAT_PRECISION,
)
from deepmd.pd.utils.utils import (
    to_paddle_tensor,
)

if TYPE_CHECKING:
    import ase.neighborlist


class DeepEval(DeepEvalBackend):
    """Paddle backend implementaion of DeepEval.

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
        *args: Any,
        auto_batch_size: Union[bool, int, AutoBatchSize] = True,
        neighbor_list: Optional["ase.neighborlist.NewPrimitiveNeighborList"] = None,
        head: Optional[str] = None,
        **kwargs: Any,
    ):
        paddle.core.set_prim_eager_enabled(True)
        paddle.core._set_prim_all_enabled(True)
        self.output_def = output_def
        self.model_path = model_file
        if str(self.model_path).endswith(".pd"):
            state_dict = paddle.load(model_file)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            # TODO: fix there
            state_dict["_extra_state"] = eval(
                "{'model_params': {'type_map': ['O', 'H'], 'descriptor': {'type': 'se_e2_a', 'sel': [46, 92], 'rcut_smth': 0.5, 'rcut': 6.0, 'neuron': [25, 50, 100], 'resnet_dt': False, 'axis_neuron': 16, 'type_one_side': True, 'seed': 1, 'activation_function': 'tanh', 'precision': 'default', 'trainable': True, 'exclude_types': [], 'env_protection': 0.0, 'set_davg_zero': False}, 'fitting_net': {'neuron': [240, 240, 240], 'resnet_dt': True, 'seed': 1, 'type': 'ener', 'numb_fparam': 0, 'numb_aparam': 0, 'activation_function': 'tanh', 'precision': 'default', 'trainable': True, 'rcond': None, 'atom_ener': [], 'use_aparam_as_mask': False}, 'data_stat_nbatch': 20, 'data_stat_protect': 0.01, 'data_bias_nsample': 10, 'pair_exclude_types': [], 'atom_exclude_types': [], 'srtab_add_bias': True, 'type': 'standard'}, 'train_infos': {'lr': 5.861945287651712e-08, 'step': 99999}}"
            )
            self.input_param = state_dict["_extra_state"]["model_params"]
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
            # model = paddle.jit.to_static(model)
            self.dp = ModelWrapper(model)
            self.dp.set_state_dict(state_dict)
        elif str(self.model_path).endswith(".pdmodel"):
            model = paddle.jit.load(model_file)
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
        if "energy" in model_output_type:
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
        atom_types = np.array(atom_types, dtype=np.int32)
        coords = np.array(coords)
        if cells is not None:
            cells = np.array(cells)
        natoms, numb_test = self._get_natoms_and_nframes(
            coords, atom_types, len(atom_types.shape) > 1
        )
        request_defs = self._get_request_defs(atomic)
        if "spin" not in kwargs or kwargs["spin"] is None:
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
                if x.category
                in (
                    OutputVariableCategory.OUT,
                    OutputVariableCategory.REDU,
                    OutputVariableCategory.DERV_R,
                    OutputVariableCategory.DERV_C_REDU,
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

        coord_input = paddle.to_tensor(
            coords.reshape([nframes, natoms, 3]),
            dtype=GLOBAL_PD_FLOAT_PRECISION,
        ).to(DEVICE)
        type_input = paddle.to_tensor(atom_types, dtype=paddle.int64).to(DEVICE)
        if cells is not None:
            box_input = paddle.to_tensor(
                cells.reshape([nframes, 3, 3]),
                dtype=GLOBAL_PD_FLOAT_PRECISION,
            ).to(DEVICE)
        else:
            box_input = None
        if fparam is not None:
            fparam_input = to_paddle_tensor(
                fparam.reshape([nframes, self.get_dim_fparam()])
            )
        else:
            fparam_input = None
        if aparam is not None:
            aparam_input = to_paddle_tensor(
                aparam.reshape([nframes, natoms, self.get_dim_aparam()])
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

        results = []
        for odef in request_defs:
            pd_name = self._OUTDEF_DP2BACKEND[odef.name]
            if pd_name in batch_output:
                shape = self._get_output_shape(odef, nframes, natoms)
                out = batch_output[pd_name].reshape(shape).numpy()
                results.append(out)
            else:
                shape = self._get_output_shape(odef, nframes, natoms)
                results.append(
                    np.full(np.abs(shape), np.nan)  # pylint: disable=no-explicit-dtype
                )  # this is kinda hacky
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

        coord_input = paddle.to_tensor(
            coords.reshape([nframes, natoms, 3]),
            dtype=GLOBAL_PD_FLOAT_PRECISION,
        ).to(DEVICE)
        type_input = paddle.to_tensor(atom_types, dtype=paddle.int64).to(DEVICE)
        spin_input = paddle.to_tensor(
            spins.reshape([nframes, natoms, 3]),
            dtype=GLOBAL_PD_FLOAT_PRECISION,
        ).to(DEVICE)
        if cells is not None:
            box_input = paddle.to_tensor(
                cells.reshape([nframes, 3, 3]),
                dtype=GLOBAL_PD_FLOAT_PRECISION,
            ).to(DEVICE)
        else:
            box_input = None
        if fparam is not None:
            fparam_input = to_paddle_tensor(
                fparam.reshape(nframes, self.get_dim_fparam())
            )
        else:
            fparam_input = None
        if aparam is not None:
            aparam_input = to_paddle_tensor(
                aparam.reshape(nframes, natoms, self.get_dim_aparam())
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
            pd_name = self._OUTDEF_DP2BACKEND[odef.name]
            if pd_name in batch_output:
                shape = self._get_output_shape(odef, nframes, natoms)
                out = batch_output[pd_name].reshape(shape).numpy()
                results.append(out)
            else:
                shape = self._get_output_shape(odef, nframes, natoms)
                results.append(
                    np.full(np.abs(shape), np.nan)  # pylint: disable=no-explicit-dtype
                )  # this is kinda hacky
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
        else:
            raise RuntimeError("unknown category")


# For tests only
def eval_model(
    model,
    coords: Union[np.ndarray, paddle.Tensor],
    cells: Optional[Union[np.ndarray, paddle.Tensor]],
    atom_types: Union[np.ndarray, paddle.to_tensor, List[int]],
    spins: Optional[Union[np.ndarray, paddle.Tensor]] = None,
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
    if isinstance(coords, paddle.Tensor):
        if cells is not None:
            assert isinstance(cells, paddle.Tensor), err_msg
        if spins is not None:
            assert isinstance(spins, paddle.Tensor), err_msg
        assert isinstance(atom_types, paddle.Tensor) or isinstance(atom_types, list)
        atom_types = paddle.to_tensor(atom_types, dtype=paddle.int64).to(DEVICE)
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
        if isinstance(atom_types, paddle.Tensor):
            atom_types = paddle.tile(atom_types.unsqueeze(0), [nframes, 1]).reshape(
                nframes, -1
            )
        else:
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
    else:
        natoms = len(atom_types[0])

    coord_input = paddle.to_tensor(
        coords.reshape([-1, natoms, 3]), dtype=GLOBAL_PD_FLOAT_PRECISION
    ).to(DEVICE)
    spin_input = None
    if spins is not None:
        spin_input = paddle.to_tensor(
            spins.reshape([-1, natoms, 3]),
            dtype=GLOBAL_PD_FLOAT_PRECISION,
        ).to(DEVICE)
    has_spin = getattr(model, "has_spin", False)
    if callable(has_spin):
        has_spin = has_spin()
    type_input = paddle.to_tensor(atom_types, dtype=paddle.int64).to(DEVICE)
    box_input = None
    if cells is None:
        pbc = False
    else:
        pbc = True
        box_input = paddle.to_tensor(
            cells.reshape([-1, 3, 3]), dtype=GLOBAL_PD_FLOAT_PRECISION
        ).to(DEVICE)
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
                energy_out.append(batch_output["energy"].numpy())
            if "atom_energy" in batch_output:
                atomic_energy_out.append(batch_output["atom_energy"].numpy())
            if "force" in batch_output:
                force_out.append(batch_output["force"].numpy())
            if "force_mag" in batch_output:
                force_mag_out.append(batch_output["force_mag"].numpy())
            if "virial" in batch_output:
                virial_out.append(batch_output["virial"].numpy())
            if "atom_virial" in batch_output:
                atomic_virial_out.append(batch_output["atom_virial"].numpy())
            if "updated_coord" in batch_output:
                updated_coord_out.append(batch_output["updated_coord"].numpy())
            if "logits" in batch_output:
                logits_out.append(batch_output["logits"].numpy())
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
            paddle.concat(energy_out)
            if energy_out
            else paddle.zeros([nframes, 1], dtype=GLOBAL_PD_FLOAT_PRECISION).to(DEVICE)
        )
        atomic_energy_out = (
            paddle.concat(atomic_energy_out)
            if atomic_energy_out
            else paddle.zeros([nframes, natoms, 1], dtype=GLOBAL_PD_FLOAT_PRECISION).to(
                DEVICE
            )
        )
        force_out = (
            paddle.concat(force_out)
            if force_out
            else paddle.zeros([nframes, natoms, 3], dtype=GLOBAL_PD_FLOAT_PRECISION).to(
                DEVICE
            )
        )
        force_mag_out = (
            paddle.concat(force_mag_out)
            if force_mag_out
            else paddle.zeros([nframes, natoms, 3], dtype=GLOBAL_PD_FLOAT_PRECISION).to(
                DEVICE
            )
        )
        virial_out = (
            paddle.concat(virial_out)
            if virial_out
            else paddle.zeros([nframes, 3, 3], dtype=GLOBAL_PD_FLOAT_PRECISION).to(
                DEVICE
            )
        )
        atomic_virial_out = (
            paddle.concat(atomic_virial_out)
            if atomic_virial_out
            else paddle.zeros(
                [nframes, natoms, 3, 3], dtype=GLOBAL_PD_FLOAT_PRECISION
            ).to(DEVICE)
        )
        updated_coord_out = (
            paddle.concat(updated_coord_out) if updated_coord_out else None
        )
        logits_out = paddle.concat(logits_out) if logits_out else None
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
