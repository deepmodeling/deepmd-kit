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
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
    nlist_distinguish_types,
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
from deepmd.pt.utils.auto_batch_size import (
    AutoBatchSize,
)

if TYPE_CHECKING:
    import ase.neighborlist


class DeepEval(DeepEvalBackend):
    """PyTorch Exportable backend implementation of DeepEval.

    Loads a .pte or .pt2 file containing a torch.export-ed model and evaluates
    it using pre-built neighbor lists.

    Parameters
    ----------
    model_file : Path
        The name of the .pte or .pt2 model file.
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
        self.neighbor_list = neighbor_list
        self._is_pt2 = model_file.endswith(".pt2")

        if self._is_pt2:
            self._load_pt2(model_file)
        elif model_file.endswith(".pte"):
            self._load_pte(model_file)
        elif model_file.endswith(".pt"):
            self._load_pt(model_file, head=kwargs.get("head"))
        else:
            raise ValueError(
                f"Unsupported model file '{model_file}' for the pt_expt "
                "backend: expected `.pt2` / `.pte` (deployable archives) or "
                "`.pt` (training checkpoint)."
            )

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

    def _init_from_model_json(self, model_json_str: str) -> None:
        """Deserialize model.json and derive model API from the dpmodel instance."""
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )
        from deepmd.pt_expt.utils.serialization import (
            _json_to_numpy,
        )

        model_dict = json.loads(model_json_str)
        model_dict = _json_to_numpy(model_dict)
        model_data = model_dict["model"]

        if model_data.get("type") == "spin_ener":
            from deepmd.pt_expt.model.spin_model import (
                SpinModel,
            )

            self._dpmodel = SpinModel.deserialize(model_data)
            self._is_spin = True
        else:
            self._dpmodel = BaseModel.deserialize(model_data)
            self._is_spin = False

        self.rcut = self._dpmodel.get_rcut()
        self.type_map = self._dpmodel.get_type_map()
        if self._is_spin:
            self._model_output_def = ModelOutputDef(
                FittingOutputDef(
                    [
                        OutputVariableDef(
                            "energy",
                            shape=[1],
                            reducible=True,
                            r_differentiable=True,
                            c_differentiable=True,
                            atomic=True,
                            magnetic=True,
                        )
                    ]
                )
            )
        else:
            self._model_output_def = ModelOutputDef(self._dpmodel.atomic_output_def())

    def _load_pte(self, model_file: str) -> None:
        """Load a .pte (torch.export) model file."""
        extra_files = {
            "model.json": "",
            "model_def_script.json": "",
            "metadata.json": "",
        }
        exported = torch.export.load(model_file, extra_files=extra_files)
        self.exported_module = exported.module()
        self._init_from_model_json(extra_files["model.json"])
        mds = extra_files["model_def_script.json"]
        self._model_def_script = json.loads(mds) if mds else {}
        md = extra_files["metadata.json"]
        self.metadata = json.loads(md) if md else {}

    def _load_pt2(self, model_file: str) -> None:
        """Load a .pt2 (AOTInductor) model file."""
        import zipfile

        from torch._inductor import (
            aoti_load_package,
        )

        # Read metadata from the .pt2 ZIP archive
        with zipfile.ZipFile(model_file, "r") as zf:
            names = zf.namelist()
            if "extra/model.json" not in names:
                raise ValueError(
                    f"Invalid .pt2 file '{model_file}': missing 'extra/model.json'"
                )
            model_json_str = zf.read("extra/model.json").decode("utf-8")
            mds = ""
            if "extra/model_def_script.json" in names:
                mds = zf.read("extra/model_def_script.json").decode("utf-8")
            md = ""
            if "extra/metadata.json" in names:
                md = zf.read("extra/metadata.json").decode("utf-8")

        self._init_from_model_json(model_json_str)
        self._model_def_script = json.loads(mds) if mds else {}
        self.metadata = json.loads(md) if md else {}

        # Load the AOTInductor model package (.pt2 ZIP archive).
        # Uses torch._inductor.aoti_load_package (private API, stable since PyTorch 2.6).
        self._pt2_runner = aoti_load_package(model_file)
        self.exported_module = None

    def _load_pt(self, model_file: str, head: str | None = None) -> None:
        """Load a `.pt` training checkpoint (eager mode, no torch.export)."""
        from copy import (
            deepcopy,
        )

        from deepmd.pt.utils.env import (
            DEVICE,
        )
        from deepmd.pt_expt.model import (
            get_model,
        )

        state_dict = torch.load(model_file, map_location=DEVICE, weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model_params = deepcopy(state_dict["_extra_state"]["model_params"])

        if "model_dict" in model_params:
            # Multi-task: pick the requested head (defaults to "Default" if present).
            heads = list(model_params["model_dict"].keys())
            if head is None:
                if "Default" in heads:
                    head = "Default"
                else:
                    raise ValueError(
                        f"Multi-task checkpoint '{model_file}' has heads "
                        f"{heads}; pass --head to select one."
                    )
            if head not in heads:
                raise ValueError(
                    f"Head '{head}' not found in checkpoint '{model_file}'. "
                    f"Available heads: {heads}."
                )
            head_params = model_params["model_dict"][head]
            # Restrict state_dict to the chosen head and rename to "Default".
            head_state = {"_extra_state": state_dict["_extra_state"]}
            for key, value in state_dict.items():
                prefix = f"model.{head}."
                if key.startswith(prefix):
                    head_state[key.replace(prefix, "model.Default.")] = (
                        value.clone() if torch.is_tensor(value) else value
                    )
            state_dict = head_state
            model_params = head_params

        model = get_model(deepcopy(model_params)).to(DEVICE)

        # Load weights into a {"Default": model} wrapper to match the
        # `model.Default.*` key prefix used in the saved state_dict.
        from deepmd.pt_expt.train.wrapper import (
            ModelWrapper,
        )

        wrapper = ModelWrapper(model)
        wrapper.load_state_dict(state_dict)
        model = wrapper.model["Default"].eval()

        self._dpmodel = model
        self._is_spin = (
            model_params.get("type") == "spin_ener" or "spin" in model_params
        )
        self.rcut = model.get_rcut()
        self.type_map = model.get_type_map()
        if self._is_spin:
            self._model_output_def = ModelOutputDef(
                FittingOutputDef(
                    [
                        OutputVariableDef(
                            "energy",
                            shape=[1],
                            reducible=True,
                            r_differentiable=True,
                            c_differentiable=True,
                            atomic=True,
                            magnetic=True,
                        )
                    ]
                )
            )
        else:
            self._model_output_def = ModelOutputDef(model.atomic_output_def())
        self._model_def_script = model_params
        # Populate metadata so eval helpers (e.g. default_fparam fallback)
        # behave the same as the .pt2/.pte path.  Mirrors the fields that
        # `_collect_metadata` writes into metadata.json.
        self.metadata = {
            "type_map": model.get_type_map(),
            "rcut": model.get_rcut(),
            "sel": model.get_sel(),
            "dim_fparam": model.get_dim_fparam(),
            "dim_aparam": model.get_dim_aparam(),
            "mixed_types": model.mixed_types(),
            "has_default_fparam": model.has_default_fparam(),
            "default_fparam": model.get_default_fparam(),
            "is_spin": self._is_spin,
        }
        if self._is_spin:
            self.metadata["ntypes_spin"] = model.spin.get_ntypes_spin()
            self.metadata["use_spin"] = [bool(v) for v in model.spin.use_spin]

        # Eager runner with the same signature as the .pt2/.pte exported module.
        # Use forward_common_lower (not forward_lower) to match the export-time
        # output keys ("energy", "energy_redu", "energy_derv_r", ...) that
        # communicate_extended_output downstream consumes.
        # Non-spin: (ext_coord, ext_atype, nlist, mapping, fparam, aparam)
        # Spin:     (ext_coord, ext_atype, ext_spin, nlist, mapping, fparam, aparam)
        if self._is_spin:

            def _eager_runner_spin(
                ext_coord: torch.Tensor,
                ext_atype: torch.Tensor,
                ext_spin: torch.Tensor,
                nlist: torch.Tensor,
                mapping: torch.Tensor | None,
                fparam: torch.Tensor | None,
                aparam: torch.Tensor | None,
            ) -> dict[str, torch.Tensor]:
                ext_coord = ext_coord.detach().requires_grad_(True)
                return model.forward_common_lower(
                    ext_coord,
                    ext_atype,
                    ext_spin,
                    nlist,
                    mapping,
                    fparam=fparam,
                    aparam=aparam,
                    do_atomic_virial=True,
                )

            self.exported_module = _eager_runner_spin
        else:

            def _eager_runner(
                ext_coord: torch.Tensor,
                ext_atype: torch.Tensor,
                nlist: torch.Tensor,
                mapping: torch.Tensor | None,
                fparam: torch.Tensor | None,
                aparam: torch.Tensor | None,
            ) -> dict[str, torch.Tensor]:
                ext_coord = ext_coord.detach().requires_grad_(True)
                return model.forward_common_lower(
                    ext_coord,
                    ext_atype,
                    nlist,
                    mapping,
                    fparam=fparam,
                    aparam=aparam,
                    do_atomic_virial=True,
                )

            self.exported_module = _eager_runner

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
        return self._dpmodel.get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        return self._dpmodel.get_dim_aparam()

    @property
    def model_type(self) -> type["DeepEvalWrapper"]:
        """The the evaluator of the model type."""
        model_output_type = self._dpmodel.model_output_type()
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
        return self._dpmodel.get_sel_type()

    def get_numb_dos(self) -> int:
        """Get the number of DOS."""
        return 0

    def get_has_efield(self) -> bool:
        """Check if the model has efield."""
        return False

    def get_has_spin(self) -> bool:
        """Check if the model has spin atom types."""
        return getattr(self, "_is_spin", False)

    def get_use_spin(self) -> list[bool]:
        """Get the per-type spin usage of this model."""
        if getattr(self, "_is_spin", False):
            return self._dpmodel.spin.use_spin.tolist()
        return []

    def get_ntypes_spin(self) -> int:
        """Get the number of spin atom types of this model. Only used in old implement."""
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
            The array should be of size nframes x dim_fparam.
        aparam
            The atomic parameter.
            The array should be of size nframes x natoms x dim_aparam.
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
        spins = kwargs.get("spin")
        if self._is_spin and spins is None:
            raise ValueError(
                "This is a spin model but no `spin` argument was provided. "
                "Please call eval(..., spin=spin_array)."
            )
        if not self._is_spin and spins is not None:
            raise ValueError(
                "This is not a spin model but a `spin` argument was provided. "
                "Please call eval(...) without the `spin` argument."
            )
        if spins is not None:
            spins = np.array(spins)
            out = self._eval_func(self._eval_model_spin, numb_test, natoms)(
                coords, cells, atom_types, spins, fparam, aparam, request_defs
            )
        else:
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
                    OutputVariableCategory.OUT,
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

    def _build_nlist_native(
        self,
        coords: torch.Tensor,
        cells: torch.Tensor | None,
        atom_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build extended coords, atype, nlist, mapping using native nlist.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates, shape (nframes, natoms, 3).
        cells : torch.Tensor or None
            Cell vectors, shape (nframes, 9). None for non-PBC.
        atom_types : torch.Tensor
            Atom types, shape (nframes, natoms).

        Returns
        -------
        extended_coord, extended_atype, nlist, mapping
            All as torch.Tensor on the same device as inputs.
        """
        nframes = coords.shape[0]
        natoms = coords.shape[1]
        rcut = self.rcut
        sel = self._dpmodel.get_sel()
        mixed_types = self._dpmodel.mixed_types()

        if cells is not None:
            box_input = cells.reshape(nframes, 3, 3)
            coord_normalized = normalize_coord(coords, box_input)
        else:
            coord_normalized = coords

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
        return extended_coord, extended_atype, nlist, mapping

    def _build_nlist_ase(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build extended coords, atype, nlist, mapping using ASE neighbor list.

        Handles multiple frames by building per frame and padding to
        a common nall.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates, shape (nframes, natoms, 3).
        cells : np.ndarray or None
            Cell vectors, shape (nframes, 9). None for non-PBC.
        atom_types : np.ndarray
            Atom types, shape (nframes, natoms).

        Returns
        -------
        extended_coord, extended_atype, nlist, mapping
        """
        nframes = coords.shape[0]
        frame_results = []
        for ff in range(nframes):
            ec, ea, nl, mp = self._build_nlist_ase_single(
                coords[ff],
                cells[ff] if cells is not None else None,
                atom_types[ff],
            )
            frame_results.append((ec, ea, nl, mp))
        # Pad to max nall across frames
        max_nall = max(ec.shape[0] for ec, _, _, _ in frame_results)
        ext_coords, ext_atypes, nlists, mappings = [], [], [], []
        for ec, ea, nl, mp in frame_results:
            pad = max_nall - ec.shape[0]
            if pad > 0:
                ec = np.concatenate(
                    [ec, np.zeros((pad, 3), dtype=ec.dtype)],
                    axis=0,
                )
                ea = np.concatenate(
                    [ea, np.full(pad, -1, dtype=ea.dtype)],
                    axis=0,
                )
                mp = np.concatenate(
                    [mp, np.zeros(pad, dtype=mp.dtype)],
                    axis=0,
                )
            ext_coords.append(ec)
            ext_atypes.append(ea)
            nlists.append(nl)
            mappings.append(mp)
        return (
            np.stack(ext_coords, axis=0),
            np.stack(ext_atypes, axis=0),
            np.stack(nlists, axis=0),
            np.stack(mappings, axis=0),
        )

    def _build_nlist_ase_single(
        self,
        positions: np.ndarray,
        cell: np.ndarray | None,
        atype: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build extended coords, atype, nlist, mapping for a single frame.

        Parameters
        ----------
        positions : np.ndarray
            Atom positions, shape (natoms, 3).
        cell : np.ndarray or None
            Cell vector, shape (9,). None for non-PBC.
        atype : np.ndarray
            Atom types, shape (natoms,).

        Returns
        -------
        extended_coord : np.ndarray, shape (nall, 3)
        extended_atype : np.ndarray, shape (nall,)
        nlist : np.ndarray, shape (nloc, nsel)
        mapping : np.ndarray, shape (nall,)
        """
        sel = self._dpmodel.get_sel()
        mixed_types = self._dpmodel.mixed_types()
        nsel = sum(sel)

        natoms = positions.shape[0]
        cell_3x3 = (
            cell.reshape(3, 3)
            if cell is not None
            else np.zeros((3, 3), dtype=np.float64)
        )
        pbc = np.repeat(cell is not None, 3)

        nl = self.neighbor_list
        nl.bothways = True
        nl.self_interaction = False
        if nl.update(pbc, cell_3x3, positions):
            nl.build(pbc, cell_3x3, positions)

        first_neigh = nl.first_neigh.copy()
        pair_second = nl.pair_second.copy()
        offset_vec = nl.offset_vec.copy()

        # Identify ghost atoms (out-of-box neighbors)
        out_mask = np.any(offset_vec != 0, axis=1)
        out_idx = pair_second[out_mask]
        out_offset = offset_vec[out_mask]
        out_coords = positions[out_idx] + out_offset.dot(cell_3x3)
        out_atype = atype[out_idx]

        nloc = natoms
        nghost = out_idx.size

        # Extended arrays (no leading frame dimension)
        extended_coord = np.concatenate((positions, out_coords), axis=0)
        extended_atype = np.concatenate((atype, out_atype))
        mapping = np.concatenate(
            (np.arange(nloc, dtype=np.int32), out_idx.astype(np.int32))
        )

        # Remap neighbor indices: ghost atoms get new indices [nloc, nloc+nghost)
        ghost_remap = pair_second.copy()
        ghost_remap[out_mask] = np.arange(nloc, nloc + nghost, dtype=np.int64)

        # Build nlist: vectorized CSR-to-dense conversion
        rcut = self.rcut
        counts = np.diff(first_neigh)
        max_nn = int(counts.max()) if counts.size > 0 else 0

        # CSR to dense: (nloc, max_nn) neighbor index array, padded with -1
        col_idx = np.arange(len(ghost_remap), dtype=np.int64) - np.repeat(
            first_neigh[:-1], counts
        )
        row_idx = np.repeat(np.arange(nloc, dtype=np.int64), counts)
        dense_idx = np.full((nloc, max_nn), -1, dtype=np.int64)
        dense_idx[row_idx, col_idx] = ghost_remap

        # Compute all distances at once
        valid = dense_idx >= 0
        lookup = np.where(valid, dense_idx, 0)
        neigh_coords = extended_coord[lookup]  # (nloc, max_nn, 3)
        dists = np.linalg.norm(
            neigh_coords - positions[:, None, :], axis=-1
        )  # (nloc, max_nn)

        # Mask invalid and out-of-range, sort by distance
        valid &= dists <= rcut
        dists = np.where(valid, dists, np.inf)
        order = np.argsort(dists, axis=-1)
        sorted_idx = np.take_along_axis(dense_idx, order, axis=-1)
        sorted_valid = np.take_along_axis(valid, order, axis=-1)

        # Take first nsel neighbors, pad if fewer than nsel
        if max_nn >= nsel:
            nlist = sorted_idx[:, :nsel]
            nlist = np.where(sorted_valid[:, :nsel], nlist, -1)
        else:
            nlist = np.full((nloc, nsel), -1, dtype=np.int64)
            nlist[:, :max_nn] = np.where(sorted_valid, sorted_idx, -1)

        if not mixed_types:
            # nlist_distinguish_types expects (nframes, nloc, nsel)
            nlist = nlist_distinguish_types(
                nlist[None],
                extended_atype[None],
                sel,
            )[0]

        return extended_coord, extended_atype, nlist, mapping

    def _prepare_inputs(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
    ) -> tuple:
        """Prepare tensor inputs for model evaluation.

        Returns
        -------
        tuple
            (ext_coord_t, ext_atype_t, nlist_t, mapping_t,
             fparam_t, aparam_t, nframes, natoms)
        """
        nframes = coords.shape[0]
        if len(atom_types.shape) == 1:
            natoms = len(atom_types)
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        else:
            natoms = len(atom_types[0])

        from deepmd.pt_expt.utils.env import (
            DEVICE,
        )

        coord_input = coords.reshape(nframes, natoms, 3)
        if self.neighbor_list is not None:
            # ASE path: builds nlist in numpy, then convert to tensors
            extended_coord, extended_atype, nlist, mapping = self._build_nlist_ase(
                coord_input,
                cells,
                atom_types,
            )
            ext_coord_t = torch.tensor(
                extended_coord, dtype=torch.float64, device=DEVICE
            )
            ext_atype_t = torch.tensor(extended_atype, dtype=torch.int64, device=DEVICE)
            nlist_t = torch.tensor(nlist, dtype=torch.int64, device=DEVICE)
            mapping_t = torch.tensor(mapping, dtype=torch.int64, device=DEVICE)
        else:
            # Native path: convert to tensors first so array-API functions
            # use the torch backend (runs on DEVICE).
            coord_t = torch.tensor(coord_input, dtype=torch.float64, device=DEVICE)
            atype_t = torch.tensor(atom_types, dtype=torch.int64, device=DEVICE)
            cells_t = (
                torch.tensor(cells, dtype=torch.float64, device=DEVICE)
                if cells is not None
                else None
            )
            ext_coord_t, ext_atype_t, nlist_t, mapping_t = self._build_nlist_native(
                coord_t,
                cells_t,
                atype_t,
            )

        if fparam is not None:
            fparam_t = torch.tensor(
                fparam.reshape(nframes, self.get_dim_fparam()),
                dtype=torch.float64,
                device=DEVICE,
            )
        elif self.get_dim_fparam() > 0:
            # Exported models (.pt2/.pte) are compiled with fparam as a
            # required input.  Fill with default values from metadata.
            default_fp = self.metadata.get("default_fparam")
            if default_fp is not None:
                fparam_t = (
                    torch.tensor(default_fp, dtype=torch.float64, device=DEVICE)
                    .unsqueeze(0)
                    .expand(nframes, -1)
                    .contiguous()
                )
            else:
                raise ValueError(
                    f"fparam is required for this model (dim_fparam={self.get_dim_fparam()}) "
                    "but was not provided, and no default_fparam is stored in the model."
                )
        else:
            fparam_t = None

        if aparam is not None:
            aparam_t = torch.tensor(
                aparam.reshape(nframes, natoms, self.get_dim_aparam()),
                dtype=torch.float64,
                device=DEVICE,
            )
        elif self.get_dim_aparam() > 0:
            # Exported models (.pt2/.pte) are compiled with aparam as a
            # required positional input.  Unlike fparam, there is no default.
            raise ValueError(
                f"aparam is required for this model (dim_aparam={self.get_dim_aparam()}) "
                "but was not provided."
            )
        else:
            aparam_t = None

        return (
            ext_coord_t,
            ext_atype_t,
            nlist_t,
            mapping_t,
            fparam_t,
            aparam_t,
            nframes,
            natoms,
        )

    def _eval_model(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        request_defs: list[OutputVariableDef],
    ) -> tuple[np.ndarray, ...]:
        (
            ext_coord_t,
            ext_atype_t,
            nlist_t,
            mapping_t,
            fparam_t,
            aparam_t,
            nframes,
            natoms,
        ) = self._prepare_inputs(coords, cells, atom_types, fparam, aparam)

        # Call the model (forward_common_lower interface, internal keys)
        if self._is_pt2:
            # AOTInductor's __call__ unflattens output using stored out_spec,
            # returning a dict just like the .pte module.
            # It also filters non-tensor args automatically, matching the
            # export-time signature where None args were excluded.
            model_ret = self._pt2_runner(
                ext_coord_t, ext_atype_t, nlist_t, mapping_t, fparam_t, aparam_t
            )
        else:
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
                    out = model_predict[odef.name].detach().cpu().numpy().reshape(shape)
                else:
                    out = np.full(shape, np.nan, dtype=GLOBAL_NP_FLOAT_PRECISION)
                results.append(out)
            else:
                shape = self._get_output_shape(odef, nframes, natoms)
                results.append(
                    np.full(np.abs(shape), np.nan, dtype=GLOBAL_NP_FLOAT_PRECISION)
                )
        return tuple(results)

    def _eval_model_spin(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        spins: np.ndarray,
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

        from deepmd.pt_expt.utils.env import (
            DEVICE,
        )

        coord_input = coords.reshape(nframes, natoms, 3)
        if self.neighbor_list is not None:
            extended_coord, extended_atype, nlist, mapping = self._build_nlist_ase(
                coord_input,
                cells,
                atom_types,
            )
            ext_coord_t = torch.tensor(
                extended_coord, dtype=torch.float64, device=DEVICE
            )
            ext_atype_t = torch.tensor(extended_atype, dtype=torch.int64, device=DEVICE)
            nlist_t = torch.tensor(nlist, dtype=torch.int64, device=DEVICE)
            mapping_t = torch.tensor(mapping, dtype=torch.int64, device=DEVICE)
        else:
            coord_t = torch.tensor(coord_input, dtype=torch.float64, device=DEVICE)
            atype_t = torch.tensor(atom_types, dtype=torch.int64, device=DEVICE)
            cells_t = (
                torch.tensor(cells, dtype=torch.float64, device=DEVICE)
                if cells is not None
                else None
            )
            ext_coord_t, ext_atype_t, nlist_t, mapping_t = self._build_nlist_native(
                coord_t,
                cells_t,
                atype_t,
            )

        # Extend spin to ghost atoms using mapping
        spin_t = torch.tensor(
            spins.reshape(nframes, natoms, 3), dtype=torch.float64, device=DEVICE
        )
        batch_idx = (
            torch.arange(nframes, dtype=torch.long, device=DEVICE)
            .unsqueeze(1)
            .expand_as(mapping_t)
        )
        ext_spin_t = spin_t[batch_idx, mapping_t]

        if fparam is not None:
            fparam_t = torch.tensor(
                fparam.reshape(nframes, self.get_dim_fparam()),
                dtype=torch.float64,
                device=DEVICE,
            )
        elif self.get_dim_fparam() > 0:
            # Exported models (.pt2/.pte) are compiled with fparam as a
            # required input.  Fill with default values from metadata.
            default_fp = self.metadata.get("default_fparam")
            if default_fp is not None:
                fparam_t = (
                    torch.tensor(default_fp, dtype=torch.float64, device=DEVICE)
                    .unsqueeze(0)
                    .expand(nframes, -1)
                    .contiguous()
                )
            else:
                raise ValueError(
                    f"fparam is required for this model (dim_fparam={self.get_dim_fparam()}) "
                    "but was not provided, and no default_fparam is stored in the model."
                )
        else:
            fparam_t = None

        if aparam is not None:
            aparam_t = torch.tensor(
                aparam.reshape(nframes, natoms, self.get_dim_aparam()),
                dtype=torch.float64,
                device=DEVICE,
            )
        elif self.get_dim_aparam() > 0:
            raise ValueError(
                f"aparam is required for this model (dim_aparam={self.get_dim_aparam()}) "
                "but was not provided."
            )
        else:
            aparam_t = None

        # Call the model with spin (7 args)
        if self._is_pt2:
            model_ret = self._pt2_runner(
                ext_coord_t,
                ext_atype_t,
                ext_spin_t,
                nlist_t,
                mapping_t,
                fparam_t,
                aparam_t,
            )
        else:
            model_ret = self.exported_module(
                ext_coord_t,
                ext_atype_t,
                ext_spin_t,
                nlist_t,
                mapping_t,
                fparam_t,
                aparam_t,
            )

        # Apply communicate_extended_output to map extended atoms → local atoms
        do_atomic_virial = any(
            x.category == OutputVariableCategory.DERV_C for x in request_defs
        )

        # Save pre-computed reduced virial: it includes both real and virtual
        # atom contributions.  communicate_extended_output would recompute it
        # from only the real-atom per-atom virial, losing the virtual part.
        saved_virial_redu = model_ret.get("energy_derv_c_redu")

        model_predict = communicate_extended_output(
            model_ret,
            self._model_output_def,
            mapping_t,
            do_atomic_virial=do_atomic_virial,
        )

        # Restore the correct reduced virial (includes virtual atom contribution)
        if saved_virial_redu is not None:
            model_predict["energy_derv_c_redu"] = saved_virial_redu

        # Translate internal keys to backend names and collect results
        results = []
        for odef in request_defs:
            if odef.name in model_predict:
                shape = self._get_output_shape(odef, nframes, natoms)
                if model_predict[odef.name] is not None:
                    out = model_predict[odef.name].detach().cpu().numpy().reshape(shape)
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
        """Get model definition script (training config)."""
        return self._model_def_script

    def get_model(self) -> torch.nn.Module:
        """Get the exported model module.

        Returns
        -------
        torch.nn.Module
            The exported model module.
        """
        return self.exported_module

    def _is_spin_model(self) -> bool:
        """Check if the underlying dpmodel is a SpinModel."""
        from deepmd.dpmodel.model.spin_model import (
            SpinModel,
        )

        return isinstance(self._dpmodel, SpinModel)

    def eval_typeebd(self) -> np.ndarray:
        """Evaluate type embedding.

        Returns
        -------
        np.ndarray
            Type embedding array of shape ``(ntypes, tebd_dim)``.

        Raises
        ------
        KeyError
            If the model has no type embedding networks.
        """
        from deepmd.dpmodel.utils.type_embed import TypeEmbedNet as TypeEmbedNetDP

        model = self._dpmodel
        if self._is_spin_model():
            model = model.backbone_model
        out = []
        for mm in model.modules():
            if isinstance(mm, TypeEmbedNetDP):
                out.append(mm())
        if not out:
            raise KeyError("The model has no type embedding networks.")
        typeebd = torch.cat(out, dim=1)
        return typeebd.detach().cpu().numpy()

    def eval_descriptor(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Evaluate descriptor.

        Parameters
        ----------
        coords
            Coordinates, shape ``(nframes, natoms, 3)``.
        cells
            Cell vectors, shape ``(nframes, 3, 3)`` or ``None``.
        atom_types
            Atom types, shape ``(natoms,)`` or ``(nframes, natoms)``.
        fparam
            Frame parameters, optional.
        aparam
            Atom parameters, optional.

        Returns
        -------
        np.ndarray
            Descriptor output, shape ``(nframes, nloc, dim_descrpt)``.
        """
        coords = np.array(coords)
        atom_types = np.array(atom_types, dtype=np.int32)
        if cells is not None:
            cells = np.array(cells)
        if self._is_spin_model():
            raise NotImplementedError(
                "eval_descriptor is not supported for spin models."
            )
        dp_am = self._dpmodel.get_dp_atomic_model()
        if dp_am is None:
            raise NotImplementedError(
                "eval_descriptor is not supported for this model type "
                f"({type(self._dpmodel).__name__})."
            )
        (
            ext_coord_t,
            ext_atype_t,
            nlist_t,
            mapping_t,
            fparam_t,
            _aparam_t,
            _nframes,
            _natoms,
        ) = self._prepare_inputs(coords, cells, atom_types, fparam, aparam)
        with torch.no_grad():
            fparam_for_des = (
                fparam_t if getattr(dp_am, "add_chg_spin_ebd", False) else None
            )
            descriptor, *_ = dp_am.descriptor(
                ext_coord_t,
                ext_atype_t,
                nlist_t,
                mapping=mapping_t,
                fparam=fparam_for_des,
            )
        return descriptor.detach().cpu().numpy()

    def eval_fitting_last_layer(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Evaluate the last hidden layer of the fitting network.

        Parameters
        ----------
        coords
            Coordinates, shape ``(nframes, natoms, 3)``.
        cells
            Cell vectors, shape ``(nframes, 3, 3)`` or ``None``.
        atom_types
            Atom types, shape ``(natoms,)`` or ``(nframes, natoms)``.
        fparam
            Frame parameters, optional.
        aparam
            Atom parameters, optional.

        Returns
        -------
        np.ndarray
            Middle-layer output, shape ``(nframes, nloc, neuron[-1])``.
        """
        coords = np.array(coords)
        atom_types = np.array(atom_types, dtype=np.int32)
        if cells is not None:
            cells = np.array(cells)
        if self._is_spin_model():
            raise NotImplementedError(
                "eval_fitting_last_layer is not supported for spin models."
            )
        dp_am = self._dpmodel.get_dp_atomic_model()
        if dp_am is None:
            raise NotImplementedError(
                "eval_fitting_last_layer is not supported for this model type "
                f"({type(self._dpmodel).__name__})."
            )
        (
            ext_coord_t,
            ext_atype_t,
            nlist_t,
            mapping_t,
            fparam_t,
            aparam_t,
            _nframes,
            natoms,
        ) = self._prepare_inputs(coords, cells, atom_types, fparam, aparam)
        with torch.no_grad():
            fparam_for_des = (
                fparam_t if getattr(dp_am, "add_chg_spin_ebd", False) else None
            )
            descriptor, rot_mat, g2, h2, _sw = dp_am.descriptor(
                ext_coord_t,
                ext_atype_t,
                nlist_t,
                mapping=mapping_t,
                fparam=fparam_for_des,
            )
            atype = ext_atype_t[:, :natoms]
            fitting_net = dp_am.fitting_net
            fitting_net.set_return_middle_output(True)
            try:
                ret = fitting_net(
                    descriptor,
                    atype,
                    gr=rot_mat,
                    g2=g2,
                    h2=h2,
                    fparam=fparam_t,
                    aparam=aparam_t,
                )
            finally:
                fitting_net.set_return_middle_output(False)
        return ret["middle_output"].detach().cpu().numpy()
