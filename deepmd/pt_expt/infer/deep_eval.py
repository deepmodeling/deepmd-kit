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
import torch

from deepmd.dpmodel.model.transform_output import (
    communicate_extended_output,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableCategory,
    OutputVariableDef,
)
from deepmd.dpmodel.utils.batch_size import (
    AutoBatchSize,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
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
from deepmd.infer.deep_wfc import (
    DeepWFC,
)

if TYPE_CHECKING:
    import ase.neighborlist


def _reconstruct_model_output_def(metadata: dict) -> ModelOutputDef:
    """Reconstruct ModelOutputDef from stored fitting_output_defs metadata."""
    var_defs = []
    for vd in metadata["fitting_output_defs"]:
        var_defs.append(
            OutputVariableDef(
                name=vd["name"],
                shape=vd["shape"],
                reducible=vd["reducible"],
                r_differentiable=vd["r_differentiable"],
                c_differentiable=vd["c_differentiable"],
                atomic=vd["atomic"],
                category=vd["category"],
                r_hessian=vd["r_hessian"],
                magnetic=vd["magnetic"],
                intensive=vd["intensive"],
            )
        )
    fitting_output_def = FittingOutputDef(var_defs)
    return ModelOutputDef(fitting_output_def)


class DeepEval(DeepEvalBackend):
    """PyTorch Exportable backend implementation of DeepEval.

    Loads a .pte file containing a torch.export-ed model and evaluates
    it using pre-built neighbor lists.

    Parameters
    ----------
    model_file : Path
        The name of the .pte model file.
    output_def : ModelOutputDef
        The output definition of the model.
    *args : list
        Positional arguments.
    auto_batch_size : bool or int or AutoBatchSize, default: True
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
        **kwargs: Any,
    ) -> None:
        self.output_def = output_def
        self.model_path = model_file

        # Load the exported program with metadata
        extra_files = {"model_def_script.json": ""}
        exported = torch.export.load(model_file, extra_files=extra_files)
        self.exported_module = exported.module()

        # Parse metadata
        self.metadata = json.loads(extra_files["model_def_script.json"])
        self.rcut = self.metadata["rcut"]
        self.type_map = self.metadata["type_map"]

        # Reconstruct the model output def from stored fitting output defs
        self._model_output_def = _reconstruct_model_output_def(self.metadata)

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

    def get_type_map(self) -> list[str]:
        """Get the type map (element name of the atom types) of this model."""
        return self.type_map

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this DP."""
        return self.metadata["dim_fparam"]

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        return self.metadata["dim_aparam"]

    @property
    def model_type(self) -> type["DeepEvalWrapper"]:
        """The the evaluator of the model type."""
        model_output_type = self.metadata["model_output_type"]
        if "energy" in model_output_type:
            return DeepPot
        elif "dos" in model_output_type:
            return DeepDOS
        elif "dipole" in model_output_type:
            return DeepDipole
        elif "polar" in model_output_type or "polarizability" in model_output_type:
            return DeepPolar
        elif "wfc" in model_output_type:
            return DeepWFC
        else:
            raise RuntimeError("Unknown model type")

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.metadata["sel_type"]

    def get_numb_dos(self) -> int:
        """Get the number of DOS."""
        return 0

    def get_has_efield(self) -> bool:
        """Check if the model has efield."""
        return False

    def get_ntypes_spin(self) -> int:
        """Get the number of spin atom types of this model."""
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
        return dict(
            zip(
                [x.name for x in request_defs],
                out,
                strict=True,
            )
        )

    def _get_request_defs(self, atomic: bool) -> list[OutputVariableDef]:
        """Get the requested output definitions."""
        if atomic:
            return list(self.output_def.var_defs.values())
        else:
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
        """Wrapper method with auto batch size."""
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
    ) -> tuple[np.ndarray, ...]:
        nframes = coords.shape[0]
        if len(atom_types.shape) == 1:
            natoms = len(atom_types)
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        else:
            natoms = len(atom_types[0])

        rcut = self.rcut
        sel = self.metadata["sel"]
        mixed_types = self.metadata["mixed_types"]

        coord_input = coords.reshape(nframes, natoms, 3)
        if cells is not None:
            box_input = cells.reshape(nframes, 3, 3)
        else:
            box_input = None

        if box_input is not None:
            coord_normalized = normalize_coord(coord_input, box_input)
        else:
            coord_normalized = coord_input

        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            coord_normalized,
            atom_types,
            cells,
            rcut,
        )
        nlist = build_neighbor_list(
            extended_coord,
            extended_atype,
            natoms,
            rcut,
            sel,
            distinguish_types=not mixed_types,
        )
        extended_coord = extended_coord.reshape(nframes, -1, 3)

        # Convert to torch tensors
        from deepmd.pt_expt.utils.env import (
            DEVICE,
        )

        ext_coord_t = torch.tensor(extended_coord, dtype=torch.float64, device=DEVICE)
        ext_atype_t = torch.tensor(extended_atype, dtype=torch.int64, device=DEVICE)
        nlist_t = torch.tensor(nlist, dtype=torch.int64, device=DEVICE)
        mapping_t = torch.tensor(mapping, dtype=torch.int64, device=DEVICE)

        if fparam is not None:
            fparam_t = torch.tensor(
                fparam.reshape(nframes, self.get_dim_fparam()),
                dtype=torch.float64,
                device=DEVICE,
            )
        else:
            fparam_t = None

        if aparam is not None:
            aparam_t = torch.tensor(
                aparam.reshape(nframes, natoms, self.get_dim_aparam()),
                dtype=torch.float64,
                device=DEVICE,
            )
        else:
            aparam_t = None

        # Call the exported module (forward_common_lower interface, internal keys)
        model_ret = self.exported_module(
            ext_coord_t, ext_atype_t, nlist_t, mapping_t, fparam_t, aparam_t
        )

        # Apply communicate_extended_output to map extended atoms → local atoms
        do_atomic_virial = any(
            x.category == OutputVariableCategory.DERV_C for x in request_defs
        )
        model_predict = communicate_extended_output(
            model_ret,
            self._model_output_def,
            mapping_t,
            do_atomic_virial=do_atomic_virial,
        )

        # Translate internal keys to backend names and collect results
        results = []
        for odef in request_defs:
            # odef.name is the internal key (e.g. "energy_derv_r")
            # _OUTDEF_DP2BACKEND maps it to backend name (e.g. "force")
            # but model_predict uses internal keys from communicate_extended_output
            if odef.name in model_predict:
                shape = self._get_output_shape(odef, nframes, natoms)
                if model_predict[odef.name] is not None:
                    out = model_predict[odef.name].detach().numpy().reshape(shape)
                else:
                    out = np.full(shape, np.nan, dtype=GLOBAL_NP_FLOAT_PRECISION)
                results.append(out)
            else:
                shape = self._get_output_shape(odef, nframes, natoms)
                results.append(
                    np.full(np.abs(shape), np.nan, dtype=GLOBAL_NP_FLOAT_PRECISION)
                )
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
            return [nframes, natoms, *odef.shape, 1]
        elif odef.category == OutputVariableCategory.DERV_R_DERV_R:
            # hessian
            return [nframes, 3 * natoms, 3 * natoms]
        else:
            raise RuntimeError("unknown category")

    def get_model_def_script(self) -> dict:
        """Get model definition script."""
        return self.metadata

    def get_model(self) -> torch.nn.Module:
        """Get the exported model module.

        Returns
        -------
        torch.nn.Module
            The exported model module.
        """
        return self.exported_module
