# SPDX-License-Identifier: LGPL-3.0-or-later
import io
import json
import logging
from collections.abc import (
    Callable,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

import numpy as np
import torch

from deepmd.dpmodel.common import PRECISION_DICT as NP_PRECISION_DICT
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
from deepmd.infer.deep_property import (
    DeepProperty,
)
from deepmd.infer.deep_wfc import (
    DeepWFC,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNetConsistent,
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
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    VesinNeighborList,
    is_vesin_torch_available,
)
from deepmd.utils.batch_size import (
    RetrySignal,
)
from deepmd.utils.econf_embd import (
    sort_element_type,
)
from deepmd.utils.model_branch_dict import (
    get_model_dict,
)

if TYPE_CHECKING:
    import ase.neighborlist

    from deepmd.pt.model.model.model import (
        BaseModel,
    )

log = logging.getLogger(__name__)


def _is_sezm_model_params(model_params: dict[str, Any]) -> bool:
    """Return whether the params describe a SeZM / DPA4 model."""
    model_type = str(model_params.get("type", "")).lower()
    if model_type in {"sezm", "dpa4", "sezm_spin"}:
        return True
    descriptor = model_params.get("descriptor")
    if isinstance(descriptor, dict):
        descriptor_type = str(descriptor.get("type", "")).lower()
        return descriptor_type in {"sezm", "dpa4"}
    return False


class DeepEval(DeepEvalBackend):
    """PyTorch backend implementation of DeepEval.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    output_def : ModelOutputDef
        The output definition of the model.
    *args : list
        Positional arguments.
    auto_batch_size : bool or int or AutomaticBatchSize, default: True
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
        auto_batch_size: bool | int | AutoBatchSize = True,
        neighbor_list: Optional["ase.neighborlist.NewPrimitiveNeighborList"] = None,
        head: str | int | None = None,
        no_jit: bool = False,
        nlist_backend: str = "auto",
        **kwargs: Any,
    ) -> None:
        self.output_def = output_def
        self.model_path = model_file
        self.neighbor_list = neighbor_list
        # data modifier, populated only for frozen .pth models that carry one
        self.modifier = None
        if str(self.model_path).endswith(".pt"):
            state_dict = torch.load(
                model_file, map_location=env.DEVICE, weights_only=True
            )
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.input_param = state_dict["_extra_state"]["model_params"]
            self.model_def_script = self.input_param
            self.multi_task = "model_dict" in self.input_param
            if self.multi_task:
                model_alias_dict, model_branch_dict = get_model_dict(
                    self.input_param["model_dict"]
                )
                model_keys = list(self.input_param["model_dict"].keys())
                if head is None and "Default" in model_alias_dict:
                    head = "Default"
                    log.info(
                        f"Using default head {model_alias_dict[head]} for multitask model."
                    )
                if isinstance(head, int):
                    head = model_keys[0]
                assert head is not None, (
                    f"Head must be set for multitask model! Available heads are: {model_keys}, "
                    f"use `dp --pt show your_model.pt model-branch` to show detail information."
                )
                if head not in model_alias_dict:
                    # preprocess with potentially case-insensitive input
                    head_lower = head.lower()
                    for mk in model_alias_dict:
                        if mk.lower() == head_lower:
                            # mapped the first matched head
                            head = mk
                            break
                # replace with alias
                assert head in model_alias_dict, (
                    f"No head or alias named {head} in model! Available heads are: {model_keys},"
                    f"use `dp --pt show your_model.pt model-branch` to show detail information."
                )
                head = model_alias_dict[head]

                self.input_param = self.input_param["model_dict"][head]
                state_dict_head = {"_extra_state": state_dict["_extra_state"]}
                for item in state_dict:
                    if f"model.{head}." in item:
                        state_dict_head[
                            item.replace(f"model.{head}.", "model.Default.")
                        ] = state_dict[item].clone()
                state_dict = state_dict_head
            model = get_model(self.input_param).to(DEVICE)
            disable_jit = no_jit or _is_sezm_model_params(self.input_param)
            if not self.input_param.get("hessian_mode") and not disable_jit:
                model = torch.jit.script(model)
            self.dp = ModelWrapper(model)
            missing, unexpected = self.dp.load_state_dict(state_dict, strict=False)
            if missing:
                log.warning(
                    "Checkpoint loaded with missing keys (likely from an older "
                    "version): %s",
                    missing,
                )
            if unexpected:
                log.warning(
                    "Checkpoint loaded with unexpected keys: %s",
                    unexpected,
                )
        elif str(self.model_path).endswith(".pth"):
            extra_files = {"data_modifier.pth": ""}
            model = torch.jit.load(
                model_file, map_location=env.DEVICE, _extra_files=extra_files
            )
            modifier = None
            # Load modifier if it exists in extra_files
            if len(extra_files["data_modifier.pth"]) > 0:
                # Create a file-like object from the in-memory data
                modifier_data = extra_files["data_modifier.pth"]
                if isinstance(modifier_data, bytes):
                    modifier_data = io.BytesIO(modifier_data)
                # Load the modifier directly from the file-like object
                modifier = torch.jit.load(modifier_data, map_location=env.DEVICE)
            self.dp = ModelWrapper(model, modifier=modifier)
            self.modifier = modifier
            model_def_script = self.dp.model["Default"].get_model_def_script()
            if model_def_script:
                self.model_def_script = json.loads(model_def_script)
            else:
                self.model_def_script = {}
        else:
            raise ValueError("Unknown model file format!")
        self.dp.eval()
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
        self._has_hessian = self.model_def_script.get("hessian_mode", False)
        self._setup_nlist_backend(nlist_backend)

    def _setup_nlist_backend(self, nlist_backend: str) -> None:
        """Resolve the neighbor-list construction strategy from a user choice.

        ``"native"`` uses the dense all-pairs builder; ``"vesin"`` forces the
        O(N) ``vesin.torch`` cell list (raising if it is unavailable or the
        model/inputs are unsupported); ``"auto"`` uses vesin when applicable and
        silently falls back to the native builder otherwise.  Results are
        unchanged either way -- only the neighbor-search cost differs.
        """
        if nlist_backend not in ("auto", "vesin", "native"):
            raise ValueError(
                f"Unknown nlist_backend '{nlist_backend}'; "
                "expected 'auto', 'vesin', or 'native'."
            )
        # reason vesin cannot be used (None means it can)
        unsupported = None
        if self._has_spin:
            unsupported = "spin models"
        elif self._has_hessian:
            unsupported = "hessian models"
        elif self.modifier is not None:
            # the vesin path runs forward_common_lower directly, bypassing
            # ModelWrapper.forward (which applies the data modifier); fall back
            # to the native path so the modifier is still applied.
            unsupported = "models with a data modifier"
        elif "energy" not in self.dp.model["Default"].model_output_type():
            # _eval_lower_vesin reconstructs the backend output from the
            # forward_common_lower / communicate keys via _OUTDEF_DP2BACKEND,
            # which matches the model's own translation only for the energy
            # model (e.g. the polar fitting key is "polarizability" but the
            # backend output is "polar").  Restrict vesin to energy models --
            # the large-system inference target -- and fall back to native
            # for the other fitting types.
            unsupported = "non-energy models"
        ase_provided = self.neighbor_list is not None
        if nlist_backend == "native":
            self._use_vesin = False
        elif nlist_backend == "vesin":
            if not is_vesin_torch_available():
                raise ImportError(
                    "nlist_backend='vesin' was requested but 'vesin.torch' is "
                    "not installed. Install it (`pip install vesin[torch]`) or "
                    "use nlist_backend='native' (or 'auto')."
                )
            if unsupported is not None:
                raise ValueError(
                    f"nlist_backend='vesin' is not supported for {unsupported}; "
                    "use nlist_backend='native' (or 'auto')."
                )
            if ase_provided:
                raise ValueError(
                    "nlist_backend='vesin' conflicts with an explicitly "
                    "supplied ASE neighbor_list; pass only one."
                )
            self._use_vesin = True
        else:  # auto: use vesin when possible, otherwise fall back silently
            self._use_vesin = (
                is_vesin_torch_available() and unsupported is None and not ase_provided
            )
        self._nlist_builder = VesinNeighborList() if self._use_vesin else None

    def get_rcut(self) -> float:
        """Get the cutoff radius of this model."""
        return self.rcut

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        return len(self.type_map)

    def get_type_map(self) -> list[str]:
        """Get the type map (element name of the atom types) of this model."""
        return self.type_map

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this DP."""
        return self.dp.model["Default"].get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        return self.dp.model["Default"].get_dim_aparam()

    def has_default_fparam(self) -> bool:
        """Check if the model has default frame parameters."""
        try:
            return self.dp.model["Default"].has_default_fparam()
        except AttributeError:
            # for compatibility with old models
            return False

    def has_chg_spin_ebd(self) -> bool:
        """Check if the model has charge spin embedding."""
        try:
            return self.dp.model["Default"].has_chg_spin_ebd()
        except AttributeError:
            return False

    def has_default_chg_spin(self) -> bool:
        """Check if the model has default charge_spin values."""
        try:
            return self.dp.model["Default"].has_default_chg_spin()
        except AttributeError:
            return False

    def get_intensive(self) -> bool:
        return self.dp.model["Default"].get_intensive()

    def get_var_name(self) -> str:
        """Get the name of the property."""
        if hasattr(self.dp.model["Default"], "get_var_name") and callable(
            getattr(self.dp.model["Default"], "get_var_name")
        ):
            return self.dp.model["Default"].get_var_name()
        else:
            raise NotImplementedError

    @property
    def model_type(self) -> type["DeepEvalWrapper"]:
        """The the evaluator of the model type."""
        model_output_type = self.dp.model["Default"].model_output_type()
        if "energy" in model_output_type:
            return DeepPot
        elif "dos" in model_output_type:
            return DeepDOS
        elif "dipole" in model_output_type:
            return DeepDipole
        elif "polar" in model_output_type or "polarizability" in model_output_type:
            return DeepPolar
        elif "global_polar" in model_output_type:
            return DeepGlobalPolar
        elif "wfc" in model_output_type:
            return DeepWFC
        elif self.get_var_name() in model_output_type:
            return DeepProperty
        else:
            raise RuntimeError("Unknown model type")

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.dp.model["Default"].get_sel_type()

    def get_numb_dos(self) -> int:
        """Get the number of DOS."""
        return self.dp.model["Default"].get_numb_dos()

    def get_task_dim(self) -> int:
        """Get the output dimension."""
        return self.dp.model["Default"].get_task_dim()

    def get_has_efield(self) -> bool:
        """Check if the model has efield."""
        return False

    def get_ntypes_spin(self) -> int:
        """Get the number of spin atom types of this model. Only used in old implement."""
        return 0

    def get_has_spin(self) -> bool:
        """Check if the model has spin atom types."""
        return self._has_spin

    def get_use_spin(self) -> list[bool]:
        """Get the per-type spin usage of this model."""
        if self._has_spin:
            model = self.dp.model["Default"]
            return model.spin.use_spin.tolist()
        return []

    def get_has_hessian(self) -> bool:
        """Check if the model has hessian."""
        return self._has_hessian

    def get_model_branch(self) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
        """Get the model branch information."""
        if "model_dict" in self.model_def_script:
            model_alias_dict, model_branch_dict = get_model_dict(
                self.model_def_script["model_dict"]
            )
            return model_alias_dict, model_branch_dict
        else:
            # single-task model
            return {"Default": "Default"}, {"Default": {"alias": [], "info": {}}}

    def eval(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        atomic: bool = False,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        charge_spin: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
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
                coords, cells, atom_types, fparam, aparam, request_defs, charge_spin
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
                charge_spin,
            )
        return dict(
            zip(
                [x.name for x in request_defs],
                out,
            )
        )

    def _get_request_defs(self, atomic: bool) -> list[OutputVariableDef]:
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
            output_defs = list(self.output_def.var_defs.values())
        else:
            output_defs = [
                x
                for x in self.output_def.var_defs.values()
                if x.category
                in (
                    OutputVariableCategory.OUT,
                    OutputVariableCategory.REDU,
                    OutputVariableCategory.DERV_R,
                    OutputVariableCategory.DERV_C_REDU,
                    OutputVariableCategory.DERV_R_DERV_R,
                )
            ]
        if not self.get_has_hessian():
            output_defs = [
                x
                for x in output_defs
                if x.category != OutputVariableCategory.DERV_R_DERV_R
            ]
        return output_defs

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

            def eval_func(*args: Any, **kwargs: Any) -> Any:
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
    ) -> tuple[int, int]:
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
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        request_defs: list[OutputVariableDef],
        charge_spin: np.ndarray | None,
    ) -> tuple[np.ndarray, ...]:
        model = self.dp.to(DEVICE)
        prec = NP_PRECISION_DICT[RESERVED_PRECISION_DICT[GLOBAL_PT_FLOAT_PRECISION]]

        nframes = coords.shape[0]
        if len(atom_types.shape) == 1:
            natoms = len(atom_types)
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        else:
            natoms = len(atom_types[0])

        coord_input = torch.tensor(
            coords.reshape([nframes, natoms, 3]).astype(prec),
            dtype=GLOBAL_PT_FLOAT_PRECISION,
            device=DEVICE,
        )
        type_input = torch.tensor(
            atom_types.astype(NP_PRECISION_DICT[RESERVED_PRECISION_DICT[torch.long]]),
            dtype=torch.long,
            device=DEVICE,
        )
        if cells is not None:
            box_input = torch.tensor(
                cells.reshape([nframes, 3, 3]).astype(prec),
                dtype=GLOBAL_PT_FLOAT_PRECISION,
                device=DEVICE,
            )
        else:
            box_input = None
        if fparam is not None:
            fparam_input = to_torch_tensor(
                fparam.reshape(nframes, self.get_dim_fparam())
            )
        else:
            fparam_input = None
        if aparam is not None:
            aparam_input = to_torch_tensor(
                aparam.reshape(nframes, natoms, self.get_dim_aparam())
            )
        else:
            aparam_input = None
        if charge_spin is not None:
            charge_spin_input = to_torch_tensor(charge_spin.reshape(nframes, 2))
        else:
            charge_spin_input = None
        do_atomic_virial = any(
            x.category == OutputVariableCategory.DERV_C for x in request_defs
        )
        if self._use_vesin:
            batch_output = self._eval_lower_vesin(
                coord_input,
                type_input,
                box_input,
                fparam_input,
                aparam_input,
                charge_spin_input,
                do_atomic_virial,
            )
        else:
            batch_output = model(
                coord_input,
                type_input,
                box=box_input,
                do_atomic_virial=do_atomic_virial,
                fparam=fparam_input,
                aparam=aparam_input,
                charge_spin=charge_spin_input,
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
                results.append(
                    np.full(np.abs(shape), np.nan, dtype=prec)
                )  # this is kinda hacky
        return tuple(results)

    def _eval_lower_vesin(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None,
        fparam: torch.Tensor | None,
        aparam: torch.Tensor | None,
        charge_spin: torch.Tensor | None,
        do_atomic_virial: bool,
    ) -> dict[str, torch.Tensor]:
        """Evaluate via the O(N) vesin-built ``(i,j,S)`` extended neighbor list.

        Builds the extended representation with the vesin cell list, runs the
        model's ``forward_common_lower``, and maps the extended outputs back to
        local atoms with ``communicate_extended_output``.  Returns a dict keyed
        by backend names, matching the normal ``model()`` output so the caller's
        extraction is unchanged.  ``forward_common_atomic`` sets
        ``requires_grad`` on the extended coordinates internally, exactly as on
        the native path, so forces/virials are produced identically.
        """
        inner = self.dp.model["Default"]
        ext_coord, ext_atype, nlist, mapping = self._nlist_builder.build(
            coord, atype, box, self.rcut, list(inner.get_sel())
        )
        model_lower = inner.forward_common_lower(
            ext_coord,
            ext_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
        )
        predict = communicate_extended_output(
            model_lower,
            inner.model_output_def(),
            mapping,
            do_atomic_virial=do_atomic_virial,
        )
        return {
            backend: predict[internal]
            for internal, backend in self._OUTDEF_DP2BACKEND.items()
            if predict.get(internal) is not None
        }

    def _eval_model_spin(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        spins: np.ndarray,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        request_defs: list[OutputVariableDef],
        charge_spin: np.ndarray | None,
    ) -> tuple[np.ndarray, ...]:
        model = self.dp.to(DEVICE)

        nframes = coords.shape[0]
        if len(atom_types.shape) == 1:
            natoms = len(atom_types)
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        else:
            natoms = len(atom_types[0])

        coord_input = torch.tensor(
            coords.reshape([nframes, natoms, 3]),
            dtype=GLOBAL_PT_FLOAT_PRECISION,
            device=DEVICE,
        )
        type_input = torch.tensor(atom_types, dtype=torch.long, device=DEVICE)
        spin_input = torch.tensor(
            spins.reshape([nframes, natoms, 3]),
            dtype=GLOBAL_PT_FLOAT_PRECISION,
            device=DEVICE,
        )
        if cells is not None:
            box_input = torch.tensor(
                cells.reshape([nframes, 3, 3]),
                dtype=GLOBAL_PT_FLOAT_PRECISION,
                device=DEVICE,
            )
        else:
            box_input = None
        if fparam is not None:
            fparam_input = to_torch_tensor(
                fparam.reshape(nframes, self.get_dim_fparam())
            )
        else:
            fparam_input = None
        if aparam is not None:
            aparam_input = to_torch_tensor(
                aparam.reshape(nframes, natoms, self.get_dim_aparam())
            )
        else:
            aparam_input = None
        if charge_spin is not None:
            charge_spin_input = to_torch_tensor(charge_spin.reshape(nframes, 2))
        else:
            charge_spin_input = None

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
            charge_spin=charge_spin_input,
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
                results.append(
                    np.full(
                        np.abs(shape),
                        np.nan,
                        dtype=NP_PRECISION_DICT[
                            RESERVED_PRECISION_DICT[GLOBAL_PT_FLOAT_PRECISION]
                        ],
                    )
                )  # this is kinda hacky
        return tuple(results)

    def _get_output_shape(
        self, odef: OutputVariableDef, nframes: int, natoms: int
    ) -> list[int]:
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
        elif odef.category == OutputVariableCategory.DERV_R_DERV_R:
            return [nframes, 3 * natoms, 3 * natoms]
            # return [nframes, *odef.shape, 3 * natoms, 3 * natoms]
        else:
            raise RuntimeError("unknown category")

    def eval_typeebd(self) -> np.ndarray:
        """Evaluate output of type embedding network by using this model.

        Returns
        -------
        np.ndarray
            The output of type embedding network. The shape is [ntypes, o_size] or [ntypes + 1, o_size],
            where ntypes is the number of types, and o_size is the number of nodes
            in the output layer. If there are multiple type embedding networks,
            these outputs will be concatenated along the second axis.

        Raises
        ------
        KeyError
            If the model does not enable type embedding.

        See Also
        --------
        deepmd.pt.model.network.network.TypeEmbedNetConsistent :
            The type embedding network.
        """
        out = []
        for mm in self.dp.model["Default"].modules():
            if (
                getattr(mm, "original_name", type(mm).__name__)
                == TypeEmbedNetConsistent.__name__
            ):
                out.append(mm(DEVICE))
        if not out:
            raise KeyError("The model has no type embedding networks.")
        typeebd = torch.cat(out, dim=1)
        return to_numpy_array(typeebd)

    def get_model_def_script(self) -> dict:
        """Get model definition script."""
        return self.model_def_script

    def get_model_size(self) -> dict:
        """Get model parameter count.

        Returns
        -------
        dict
            A dictionary containing the number of parameters in the model.
            The keys are 'descriptor', 'fitting_net', and 'total'.
        """
        params_dict = dict(self.dp.named_parameters())
        sum_param_des = sum(
            params_dict[k].numel() for k in params_dict.keys() if "descriptor" in k
        )
        sum_param_fit = sum(
            params_dict[k].numel()
            for k in params_dict.keys()
            if "fitting" in k and "_networks" not in k
        )
        return {
            "descriptor": sum_param_des,
            "fitting-net": sum_param_fit,
            "total": sum_param_des + sum_param_fit,
        }

    def get_observed_types(self) -> dict:
        """Get observed types (elements) of the model during data statistics.

        Returns
        -------
        dict
            A dictionary containing the information of observed type in the model:
            - 'type_num': the total number of observed types in this model.
            - 'observed_type': a list of the observed types in this model.
        """
        # Try metadata first (from model_def_script, already a dict)
        observed_type_list = self.model_def_script.get("info", {}).get("observed_type")
        if observed_type_list is not None:
            return {
                "type_num": len(observed_type_list),
                "observed_type": observed_type_list,
            }
        # Fallback: bias-based approach for old models
        observed_type_list = self.dp.model["Default"].get_observed_type_list()
        return {
            "type_num": len(observed_type_list),
            "observed_type": sort_element_type(observed_type_list),
        }

    def get_model(self) -> "BaseModel":
        """Get the PyTorch model.

        Returns
        -------
        BaseModel
            The PyTorch model instance.
        """
        return self.dp.model["Default"]

    def eval_descriptor(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Evaluate descriptors by using this DP.

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

        Returns
        -------
        descriptor
            Descriptors.
        """
        model = self.dp.model["Default"]
        while True:
            if self.auto_batch_size is not None:
                self.auto_batch_size.set_oom_retry_mode(True)
            model.set_eval_descriptor_hook(True)
            retry = False
            try:
                self.eval(
                    coords,
                    cells,
                    atom_types,
                    atomic=False,
                    fparam=fparam,
                    aparam=aparam,
                    **kwargs,
                )
                descriptor = model.eval_descriptor()
            except RetrySignal:
                retry = True
            finally:
                model.set_eval_descriptor_hook(False)
                if self.auto_batch_size is not None:
                    self.auto_batch_size.set_oom_retry_mode(False)
            if not retry:
                return to_numpy_array(descriptor)

    def eval_fitting_last_layer(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Evaluate fitting before last layer by using this DP.

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

        Returns
        -------
        fitting
            Fitting output before last layer.
        """
        model = self.dp.model["Default"]
        while True:
            if self.auto_batch_size is not None:
                self.auto_batch_size.set_oom_retry_mode(True)
            model.set_eval_fitting_last_layer_hook(True)
            retry = False
            try:
                self.eval(
                    coords,
                    cells,
                    atom_types,
                    atomic=False,
                    fparam=fparam,
                    aparam=aparam,
                    **kwargs,
                )
                fitting_net = model.eval_fitting_last_layer()
            except RetrySignal:
                retry = True
            finally:
                model.set_eval_fitting_last_layer_hook(False)
                if self.auto_batch_size is not None:
                    self.auto_batch_size.set_oom_retry_mode(False)
            if not retry:
                return to_numpy_array(fitting_net)
