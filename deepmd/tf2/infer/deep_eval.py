# SPDX-License-Identifier: LGPL-3.0-or-later
import json
from collections.abc import (
    Callable,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

import numpy as np
import tensorflow as tf

from deepmd.dpmodel.output_def import (
    ModelOutputDef,
    OutputVariableCategory,
    OutputVariableDef,
)
from deepmd.dpmodel.utils.batch_size import (
    AutoBatchSize,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
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

if TYPE_CHECKING:
    import ase.neighborlist


def _decode_list_of_bytes(list_of_bytes: list[bytes]) -> list[str]:
    return [item.decode() for item in list_of_bytes]


def _to_numpy_dict(ret: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: value.numpy() if isinstance(value, tf.Tensor) else value
        for key, value in ret.items()
    }


class TF2SavedModelWrapper(tf.Module):
    """Small Python wrapper around the exported TensorFlow SavedModel."""

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = tf.saved_model.load(model)
        self.type_map = _decode_list_of_bytes(
            self.model.get_type_map().numpy().tolist()
        )
        self.rcut = self.model.get_rcut().numpy().item()
        self.dim_fparam = self.model.get_dim_fparam().numpy().item()
        self.dim_aparam = self.model.get_dim_aparam().numpy().item()
        self.sel_type = self.model.get_sel_type().numpy().tolist()
        self._is_aparam_nall = self.model.is_aparam_nall().numpy().item()
        self._model_output_type = _decode_list_of_bytes(
            self.model.model_output_type().numpy().tolist()
        )
        self._mixed_types = self.model.mixed_types().numpy().item()
        self.min_nbor_dist = (
            self.model.get_min_nbor_dist().numpy().item()
            if hasattr(self.model, "get_min_nbor_dist")
            else None
        )
        self.sel = self.model.get_sel().numpy().tolist()
        self.model_def_script = self.model.get_model_def_script().numpy().decode()
        self._has_default_fparam = (
            self.model.has_default_fparam().numpy().item()
            if hasattr(self.model, "has_default_fparam")
            else False
        )
        self.default_fparam = (
            self.model.get_default_fparam().numpy().tolist()
            if hasattr(self.model, "get_default_fparam")
            else None
        )
        # property models only (absent for other model types).
        self._var_name = (
            self.model.get_var_name().numpy().decode()
            if hasattr(self.model, "get_var_name")
            else None
        )
        self._task_dim = (
            self.model.get_task_dim().numpy().item()
            if hasattr(self.model, "get_task_dim")
            else None
        )
        self._intensive = (
            self.model.get_intensive().numpy().item()
            if hasattr(self.model, "get_intensive")
            else False
        )

    def __call__(
        self,
        coord: np.ndarray,
        atype: np.ndarray,
        box: np.ndarray | None = None,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, np.ndarray]:
        call = self.model.call_atomic_virial if do_atomic_virial else self.model.call
        coord = tf.convert_to_tensor(coord, dtype=tf.float64)
        atype = tf.convert_to_tensor(atype, dtype=tf.int32)
        if box is None:
            box = np.empty((coord.shape[0], 0, 0), dtype=np.float64)
        if fparam is None:
            fparam = np.empty((coord.shape[0], self.get_dim_fparam()), dtype=np.float64)
        if aparam is None:
            aparam = np.empty(
                (coord.shape[0], coord.shape[1], self.get_dim_aparam()),
                dtype=np.float64,
            )
        ret = call(
            coord,
            atype,
            tf.convert_to_tensor(box, dtype=tf.float64),
            tf.convert_to_tensor(fparam, dtype=tf.float64),
            tf.convert_to_tensor(aparam, dtype=tf.float64),
        )
        return _to_numpy_dict(ret)

    def get_type_map(self) -> list[str]:
        return self.type_map

    def get_rcut(self) -> float:
        return self.rcut

    def get_dim_fparam(self) -> int:
        return self.dim_fparam

    def get_dim_aparam(self) -> int:
        return self.dim_aparam

    def get_sel_type(self) -> list[int]:
        return self.sel_type

    def is_aparam_nall(self) -> bool:
        return self._is_aparam_nall

    def model_output_type(self) -> list[str]:
        return self._model_output_type

    def mixed_types(self) -> bool:
        return self._mixed_types

    def get_min_nbor_dist(self) -> float | None:
        return self.min_nbor_dist

    def get_sel(self) -> list[int]:
        return self.sel

    def get_model_def_script(self) -> str:
        return self.model_def_script

    def has_default_fparam(self) -> bool:
        return self._has_default_fparam

    def get_default_fparam(self) -> list[float] | None:
        return self.default_fparam

    def get_var_name(self) -> str:
        """Get the name of the property (property models only)."""
        if self._var_name is None:
            raise NotImplementedError
        return self._var_name

    def get_task_dim(self) -> int:
        """Get the output dimension of the property (property models only)."""
        if self._task_dim is None:
            raise NotImplementedError
        return self._task_dim

    def get_intensive(self) -> bool:
        """Whether the property is intensive (property models only)."""
        return self._intensive


class DeepEval(DeepEvalBackend):
    """TensorFlow 2 SavedModel backend implementation of DeepEval."""

    def __init__(
        self,
        model_file: str,
        output_def: ModelOutputDef,
        *args: Any,
        auto_batch_size: bool | int | AutoBatchSize = True,
        neighbor_list: Optional["ase.neighborlist.NewPrimitiveNeighborList"] = None,
        **kwargs: Any,
    ) -> None:
        if not model_file.endswith(".savedmodeltf"):
            raise ValueError("TF2 backend only supports .savedmodeltf files")
        self.output_def = output_def
        self.model_path = model_file
        self.dp = TF2SavedModelWrapper(model_file)
        self.rcut = self.dp.get_rcut()
        self.type_map = self.dp.get_type_map()
        if isinstance(auto_batch_size, bool):
            self.auto_batch_size = AutoBatchSize() if auto_batch_size else None
        elif isinstance(auto_batch_size, int):
            self.auto_batch_size = AutoBatchSize(auto_batch_size)
        elif isinstance(auto_batch_size, AutoBatchSize):
            self.auto_batch_size = auto_batch_size
        else:
            raise TypeError("auto_batch_size should be bool, int, or AutoBatchSize")

    def get_rcut(self) -> float:
        return self.rcut

    def get_ntypes(self) -> int:
        return len(self.type_map)

    def get_type_map(self) -> list[str]:
        return self.type_map

    def get_dim_fparam(self) -> int:
        return self.dp.get_dim_fparam()

    def get_dim_aparam(self) -> int:
        return self.dp.get_dim_aparam()

    def has_default_fparam(self) -> bool:
        return self.dp.has_default_fparam()

    @property
    def model_type(self) -> type["DeepEvalWrapper"]:
        model = self.get_model()
        model_output_type = model.model_output_type()
        if "energy" in model_output_type:
            return DeepPot
        if "dos" in model_output_type:
            return DeepDOS
        if "dipole" in model_output_type:
            return DeepDipole
        if "polar" in model_output_type or "polarizability" in model_output_type:
            return DeepPolar
        if "wfc" in model_output_type:
            return DeepWFC
        if self._get_property_var_name(model) in model_output_type:
            return DeepProperty
        raise RuntimeError("Unknown model type")

    def get_sel_type(self) -> list[int]:
        return self.dp.get_sel_type()

    def get_numb_dos(self) -> int:
        return 0

    def get_has_efield(self) -> bool:
        return False

    def get_ntypes_spin(self) -> int:
        return 0

    def eval(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        atomic: bool = False,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        atom_types = np.array(atom_types, dtype=np.int32)
        coords = np.array(coords)
        if cells is not None:
            cells = np.array(cells)
        natoms, numb_test = self._get_natoms_and_nframes(
            coords, atom_types, len(atom_types.shape) > 1
        )
        request_defs = self._get_request_defs(atomic)
        out = self._eval_func(self._eval_model, numb_test, natoms)(
            coords, cells, atom_types, fparam, aparam, request_defs
        )
        # ``AutoBatchSize.execute_all`` unwraps a single-output result out of
        # its tuple, which would make ``zip`` iterate over the array's frame
        # axis. Re-wrap so the request-def names line up (a single request def
        # arises for global-only DOS/property inference at atomic=False).
        if not isinstance(out, tuple):
            out = (out,)
        return dict(zip([x.name for x in request_defs], out, strict=True))

    def _get_request_defs(self, atomic: bool) -> list[OutputVariableDef]:
        if atomic:
            return list(self.output_def.var_defs.values())
        return [
            x
            for x in self.output_def.var_defs.values()
            if x.category
            in (
                OutputVariableCategory.REDU,
                OutputVariableCategory.DERV_R,
                OutputVariableCategory.DERV_C_REDU,
            )
        ]

    def _eval_func(self, inner_func: Callable, numb_test: int, natoms: int) -> Callable:
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
        return natoms, coords.shape[0]

    def _eval_model(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        request_defs: list[OutputVariableDef],
    ) -> tuple[np.ndarray, ...]:
        nframes = coords.shape[0]
        if len(atom_types.shape) == 1:
            natoms = len(atom_types)
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        else:
            natoms = len(atom_types[0])

        coord_input = coords.reshape([-1, natoms, 3])
        type_input = atom_types
        box_input = cells.reshape([-1, 3, 3]) if cells is not None else None
        if fparam is not None:
            fparam_input = fparam.reshape(nframes, self.get_dim_fparam())
        elif self.dp.has_default_fparam():
            default_fparam = self.dp.get_default_fparam()
            assert default_fparam is not None
            fparam_input = np.tile(
                np.array(default_fparam, dtype=GLOBAL_NP_FLOAT_PRECISION),
                (nframes, 1),
            )
        else:
            fparam_input = None
        aparam_input = (
            aparam.reshape(nframes, natoms, self.get_dim_aparam())
            if aparam is not None
            else None
        )

        do_atomic_virial = any(
            x.category == OutputVariableCategory.DERV_C_REDU for x in request_defs
        )
        batch_output = self.dp(
            coord_input,
            type_input,
            box=box_input,
            fparam=fparam_input,
            aparam=aparam_input,
            do_atomic_virial=do_atomic_virial,
        )

        results = []
        for odef in request_defs:
            dp_name = odef.name
            shape = self._get_output_shape(odef, nframes, natoms)
            if dp_name in batch_output and batch_output[dp_name] is not None:
                results.append(batch_output[dp_name].reshape(shape))
            else:
                results.append(
                    np.full(np.abs(shape), np.nan, dtype=GLOBAL_NP_FLOAT_PRECISION)
                )
        return tuple(results)

    def _get_output_shape(
        self, odef: OutputVariableDef, nframes: int, natoms: int
    ) -> list[int]:
        if odef.category == OutputVariableCategory.DERV_C_REDU:
            return [nframes, *odef.shape[:-1], 9]
        if odef.category == OutputVariableCategory.REDU:
            return [nframes, *odef.shape, 1]
        if odef.category == OutputVariableCategory.DERV_C:
            return [nframes, *odef.shape[:-1], natoms, 9]
        if odef.category == OutputVariableCategory.DERV_R:
            return [nframes, *odef.shape[:-1], natoms, 3]
        if odef.category == OutputVariableCategory.OUT:
            return [nframes, natoms, *odef.shape, 1]
        if odef.category == OutputVariableCategory.DERV_R_DERV_R:
            return [nframes, 3 * natoms, 3 * natoms]
        raise RuntimeError("unknown category")

    def get_model_def_script(self) -> dict:
        return json.loads(self.dp.get_model_def_script())

    def get_model(self) -> TF2SavedModelWrapper:
        return self.dp
