# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import warnings
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
from deepmd.infer.deep_property import (
    DeepProperty,
)
from deepmd.infer.deep_wfc import (
    DeepWFC,
)
from deepmd.pt.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.pt_expt.utils.edge_schema import (
    edge_schema_from_extended,
)
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    VesinNeighborList,
    is_vesin_torch_available,
)

if TYPE_CHECKING:
    import ase.neighborlist

    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )
    from deepmd.dpmodel.utils.neighbor_graph import (
        NeighborGraph,
    )


# Public output keys emitted by the graph-form AOTI forward
# (``forward_lower_graph_exportable``) keyed by the output-variable category that
# ``request_defs`` carries.  The graph path is LOCAL-only (``N == sum(n_node)``
# nodes, no ghosts), so its outputs are already at local-atom resolution -- no
# ``communicate_extended_output`` fold-back is needed.
_GRAPH_CATEGORY_TO_KEY = {
    OutputVariableCategory.OUT: "atom_energy",
    OutputVariableCategory.REDU: "energy",
    OutputVariableCategory.DERV_R: "force",
    OutputVariableCategory.DERV_C_REDU: "virial",
    OutputVariableCategory.DERV_C: "atom_virial",
}


def _graph_spin_output_key(odef: "OutputVariableDef") -> str | None:
    """Map a native-spin request def to its graph-spin public model key.

    Twin of ``_GRAPH_CATEGORY_TO_KEY`` for
    ``NativeSpinEnergyModel.forward_lower_graph_exportable``'s output dict
    (``_translate_spin_energy_keys``: ``atom_energy``, ``energy``,
    ``force``, ``force_mag``, ``virial``, ``atom_virial``).  Category alone
    is NOT enough here: ``do_derivative`` (``deepmd.dpmodel.output_def``)
    gives the magnetic derivative def (``energy_derv_r_mag``) the SAME
    category as the physical one (``energy_derv_r``) -- only ``.magnetic``
    tells them apart -- so this checks ``magnetic`` before falling back to
    the category table.  Returns ``None`` for outputs the graph-spin ABI
    does not export: ``mask_mag`` is synthesized by the caller
    (``_eval_model_graph_spin``) from the artifact's ``use_spin`` metadata
    and the input atom types, so it never reaches the placeholder path;
    ``mask`` (real-atom mask, a virtual-atom-scheme concept) falls back to
    the NaN-filled placeholder, same as any other unavailable output.
    """
    if odef.name in ("mask", "mask_mag"):
        return None
    if odef.magnetic and odef.category == OutputVariableCategory.DERV_R:
        return "force_mag"
    return _GRAPH_CATEGORY_TO_KEY.get(odef.category)


def _reshape_charge_spin(
    charge_spin: np.ndarray, nframes: int, dim_chg_spin: int
) -> np.ndarray:
    charge_spin_arr = np.asarray(charge_spin)
    try:
        return charge_spin_arr.reshape(nframes, dim_chg_spin)
    except ValueError as err:
        raise ValueError(
            f"charge_spin must be reshape-compatible with ({nframes}, {dim_chg_spin}), "
            f"got shape {charge_spin_arr.shape}."
        ) from err


def _is_pt_backend_dpa4_params(model_params: dict[str, Any]) -> bool:
    """Return whether a training checkpoint should be loaded by the pt backend."""
    model_type = str(model_params.get("type", "")).lower()
    if model_type in {"sezm", "dpa4", "sezm_spin"}:
        return True
    descriptor = model_params.get("descriptor")
    if isinstance(descriptor, dict):
        descriptor_type = str(descriptor.get("type", "")).lower()
        return descriptor_type in {"sezm", "dpa4"}
    return False


def _warn_legacy_edge_vec(metadata: dict) -> None:
    """Warn once per model load when an edge_vec-schema artifact is opened.

    The ``edge_vec`` lower schema is produced only by the pt backend's
    SeZM/DPA4 freeze and is superseded by the NeighborGraph lower. Support
    will be removed in a future release; energy SeZM checkpoints can be
    refrozen through the pt_expt backend (graph schema) instead.

    Parameters
    ----------
    metadata
        The ``metadata.json`` dict of the opened ``.pt2`` archive.
    """
    if metadata.get("lower_input_kind") == "edge_vec":
        warnings.warn(
            "This .pt2 uses the deprecated edge_vec lower schema (pt-backend "
            "SeZM/DPA4 freeze). Support will be removed in a future release; "
            "refreeze the checkpoint with the pt_expt backend (graph schema).",
            DeprecationWarning,
            stacklevel=3,
        )


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
    nlist_backend : str, default: "auto"
        Neighbor-list builder for the NLIST/extended lower path (``.pte`` and
        nlist-form ``.pt2``): ``"auto"`` / ``"vesin"`` / ``"native"``. Not
        used by graph-form ``.pt2`` artifacts.
    neighbor_graph_method : str, default: "dense"
        Carry-all graph builder for GRAPH-FORM ``.pt2`` artifacts ONLY
        (``metadata["lower_input_kind"] == "graph"``): ``"dense"`` / ``"ase"``
        (backend-agnostic) or ``"vesin"`` / ``"nv"`` (on-device O(N)). A
        non-default value on any other artifact raises at construction — the
        knob would silently do nothing there; use ``nlist_backend`` for the
        nlist path instead. All builders emit the same neighbor set, so the
        choice is performance-only. Consolidating the two knobs into a single
        backend-selection API is deferred to the dense-nlist deprecation.
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
        nlist_backend: str = "auto",
        neighbor_graph_method: str = "dense",
        **kwargs: Any,
    ) -> None:
        self.output_def = output_def
        self.model_path = model_file
        self.neighbor_list = neighbor_list
        # World-2 graph-form ``.pt2`` (lower_input_kind == "graph") builder select:
        # "dense"/"ase" (backend-agnostic) or "vesin"/"nv" (on-device O(N)).
        self._neighbor_graph_method = neighbor_graph_method
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

        # Single choke point: self.metadata is set by all three loaders
        # above (_load_pt2 / _load_pte / _load_pt), so this is the one
        # place that sees every model load regardless of archive kind.
        _warn_legacy_edge_vec(self.metadata)

        # neighbor_graph_method is consumed ONLY by graph-form .pt2 eval
        # (_eval_model_graph); fail fast instead of silently ignoring it on
        # nlist-form artifacts (there, the builder knob is nlist_backend).
        if neighbor_graph_method != "dense" and getattr(self, "metadata", {}).get(
            "lower_input_kind"
        ) not in ("graph", "dpa1_canonical"):
            raise ValueError(
                f"neighbor_graph_method={neighbor_graph_method!r} only applies to "
                "graph-form .pt2 artifacts (lower_input_kind == 'graph'); this "
                f"model is not graph-form. Use nlist_backend to select the "
                "neighbor-list builder for the nlist path."
            )

        self._setup_nlist_backend(nlist_backend)

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
        is_spin = bool(getattr(self, "_is_spin", False))
        # Native-spin (NeighborGraph route) graph-form artifacts never touch
        # this NLIST builder at all -- graph-form eval uses
        # ``neighbor_graph_method`` instead (see ``_eval_model_graph_spin``).
        # Only the virtual-atom (dense/nlist) spin scheme actually needs the
        # vesin restriction below.
        is_native_spin_graph = is_spin and (
            getattr(self, "metadata", {}).get("lower_input_kind") == "graph"
        )
        ase_provided = self.neighbor_list is not None
        # reason vesin cannot be used (None means it can)
        unsupported = "spin models" if is_spin and not is_native_spin_graph else None
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

        model_type = model_data.get("type")
        if model_type == "spin_ener":
            from deepmd.pt_expt.model.spin_model import (
                SpinModel,
            )

            self._dpmodel = SpinModel.deserialize(model_data)
            self._is_spin = True
        else:
            # Registry-dispatched: wrapper classes registered in the pt_expt
            # BaseModel registry (e.g. the native-spin models, type
            # "native_spin") come back as their pt_expt torch classes and
            # declare spin via the base-model capability method.
            self._dpmodel = BaseModel.deserialize(model_data)
            self._is_spin = self._dpmodel.has_spin()

        self._rcut = self._dpmodel.get_rcut()
        self._type_map = self._dpmodel.get_type_map()
        self._sel = list(self._dpmodel.get_sel())
        self._mixed_types = bool(self._dpmodel.mixed_types())
        if self._is_spin:
            spin_fitting_defs = self._dpmodel.model_output_def().def_outp.get_data()
            # Keep only physical fitting outputs; mask is derived by ModelOutputDef.
            fitting_defs = [
                vdef for name, vdef in spin_fitting_defs.items() if name != "mask"
            ]
            self._model_output_def = ModelOutputDef(FittingOutputDef(fitting_defs))
        else:
            self._model_output_def = ModelOutputDef(self._dpmodel.atomic_output_def())

    def _init_from_metadata(self) -> None:
        """Initialize DeepEval from ``metadata.json`` alone.

        Used when the ``.pt2`` / ``.pte`` archive ships no ``model.json``
        (e.g. for backends that do not travel through the dpmodel round-trip).
        The metadata contract is the same one the C++ ``DeepPotPTExpt``
        reader consumes, so anything that validates against the C++ side
        automatically validates here.

        ``self._dpmodel`` is left as ``None`` to signal the metadata-only
        mode.  Inference does not need it: it runs through
        ``aoti_load_package`` / the exported module and uses plain
        attributes (``self._rcut``, ``self._sel``, ``self._mixed_types``,
        ``self._model_output_def``) for all metadata-level queries.
        """
        self._dpmodel = None
        self._is_spin = bool(self.metadata.get("is_spin", False))
        self._rcut = float(self.metadata["rcut"])
        self._type_map = list(self.metadata["type_map"])
        self._sel = [int(s) for s in self.metadata["sel"]]
        self._mixed_types = bool(self.metadata["mixed_types"])

        fitting_defs = []
        for vdef in self.metadata["fitting_output_defs"]:
            fitting_defs.append(
                OutputVariableDef(
                    name=vdef["name"],
                    shape=list(vdef["shape"]),
                    reducible=vdef.get("reducible", False),
                    r_differentiable=vdef.get("r_differentiable", False),
                    c_differentiable=vdef.get("c_differentiable", False),
                    atomic=vdef.get("atomic", True),
                    category=int(
                        vdef.get("category", OutputVariableCategory.OUT.value)
                    ),
                    r_hessian=vdef.get("r_hessian", False),
                    magnetic=vdef.get("magnetic", False),
                    intensive=vdef.get("intensive", False),
                )
            )
        self._model_output_def = ModelOutputDef(FittingOutputDef(fitting_defs))

    def _load_pte(self, model_file: str) -> None:
        """Load a .pte (torch.export) model file.

        ``model.json`` is optional: when present it is used to reconstruct
        the dpmodel instance (enabling dpmodel-level introspection such as
        ``eval_descriptor``); when absent we fall back to pure metadata
        mode via :meth:`_init_from_metadata`.  ``metadata.json`` is the
        only contract the inference path actually requires.
        """
        extra_files = {
            "model.json": "",
            "model_def_script.json": "",
            "metadata.json": "",
        }
        exported = torch.export.load(model_file, extra_files=extra_files)
        self.exported_module = exported.module()
        mds = extra_files["model_def_script.json"]
        self._model_def_script = json.loads(mds) if mds else {}
        md = extra_files["metadata.json"]
        if not md:
            raise ValueError(
                f"Invalid .pte file '{model_file}': missing 'metadata.json'"
            )
        self.metadata = json.loads(md)

        model_json_str = extra_files["model.json"]
        if model_json_str:
            self._init_from_model_json(model_json_str)
        else:
            self._init_from_metadata()

    def _load_pt2(self, model_file: str) -> None:
        """Load a .pt2 (AOTInductor) model file.

        ``model.json`` is optional — it only enables the dpmodel
        round-trip (used by ``eval_descriptor``, ``eval_typeebd``, etc.).
        Pure AOTI inference (``DeepPot.eval`` / ``dp test`` / ASE
        calculator) only needs ``metadata.json``, matching the contract
        the C++ ``DeepPotPTExpt`` reader enforces.

        Archive entries are located under ``model/extra/`` so that the
        PyTorch 2.11 ``load_pt2`` loader accepts the archive without the
        "outdated pt2 file" fallback warning.
        """
        import zipfile

        from torch._inductor import (
            aoti_load_package,
        )

        from deepmd.pt_expt.utils.serialization import (
            PT2_EXTRA_PREFIX,
        )

        md_entry = PT2_EXTRA_PREFIX + "metadata.json"
        model_json_entry = PT2_EXTRA_PREFIX + "model.json"
        mds_entry = PT2_EXTRA_PREFIX + "model_def_script.json"

        # Read metadata from the .pt2 ZIP archive
        with zipfile.ZipFile(model_file, "r") as zf:
            names = zf.namelist()
            if md_entry not in names:
                raise ValueError(
                    f"Invalid .pt2 file '{model_file}': missing '{md_entry}'"
                )
            md = zf.read(md_entry).decode("utf-8")
            model_json_str = ""
            if model_json_entry in names:
                model_json_str = zf.read(model_json_entry).decode("utf-8")
            mds = ""
            if mds_entry in names:
                mds = zf.read(mds_entry).decode("utf-8")

        self.metadata = json.loads(md)
        self._model_def_script = json.loads(mds) if mds else {}
        if model_json_str:
            self._init_from_model_json(model_json_str)
        else:
            self._init_from_metadata()

        # Load the AOTInductor model package (.pt2 ZIP archive).
        # Uses torch._inductor.aoti_load_package (private API, stable since PyTorch 2.6).
        self._pt2_runner = aoti_load_package(model_file)
        self.exported_module = None

    def _load_pt(self, model_file: str, head: str | None = None) -> None:
        """Load a `.pt` training checkpoint (eager mode, no torch.export)."""
        from copy import (
            deepcopy,
        )

        from deepmd.pt_expt.model import (
            get_model,
        )
        from deepmd.pt_expt.utils.env import (
            DEVICE,
        )

        # Match the training resume path (training.py:712) — weights_only=True
        # avoids unpickling arbitrary code from untrusted checkpoints.
        state_dict = torch.load(model_file, map_location=DEVICE, weights_only=True)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        extra = state_dict.get("_extra_state") if isinstance(state_dict, dict) else None
        if not (isinstance(extra, dict) and "model_params" in extra):
            raise ValueError(
                f"Invalid .pt file '{model_file}': expected a pt_expt training "
                "checkpoint containing '_extra_state' with nested "
                "'model_params'. If this is a legacy pt-trained checkpoint, "
                "load it with `dp --pt` instead. If this is an exported model, "
                "use a `.pte` or `.pt2` artifact."
            )
        model_params = deepcopy(extra["model_params"])

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
            # No tensor cloning needed: load_state_dict copies into the
            # destination parameters and does not mutate the input dict.
            head_state = {"_extra_state": state_dict["_extra_state"]}
            prefix = f"model.{head}."
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    head_state[key.replace(prefix, "model.Default.")] = value
            state_dict = head_state
            model_params = head_params

        if _is_pt_backend_dpa4_params(model_params):
            raise ValueError(
                "DPA4/SeZM `.pt` checkpoints belong to the regular `pt` backend. "
                "Use the `pt` backend for eager checkpoint inference, or export "
                "the checkpoint to `.pt2` / `.pte` before loading it with `pt_expt`."
            )

        model = get_model(deepcopy(model_params)).to(DEVICE)

        # Strip the `_CompiledModel` wrapper that pt_expt training applies
        # after compilation (training.py:996).  The saved state_dict has
        # `model.Default.original_model.X` keys (the real weights) plus
        # `model.Default.compiled_forward_lower._orig_mod._param_constant*`
        # / `_tensor_constant*` keys (graph constants baked into the
        # compiled forward — duplicates of the real weights, useless for
        # eager inference).  Drop the latter and unwrap the former.
        cleaned: dict[str, Any] = {}
        compiled_marker = ".compiled_forward_lower."
        # Per-task buffer copies registered on _CompiledModel (bias_atom_e,
        # case_embd) — real values live on the original model's fitting net.
        task_buf_marker = "._task_"
        wrapper_infix = ".original_model."
        for key, value in state_dict.items():
            if compiled_marker in key:
                continue
            if task_buf_marker in key:
                continue
            if wrapper_infix in key:
                key = key.replace(wrapper_infix, ".", 1)
            cleaned[key] = value
        state_dict = cleaned

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
        self._rcut = model.get_rcut()
        self._type_map = model.get_type_map()
        self._sel = list(model.get_sel())
        self._mixed_types = bool(model.mixed_types())
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
            "ntypes": model.get_descriptor().get_ntypes(),
            "rcut": model.get_rcut(),
            "sel": model.get_sel(),
            "dim_fparam": model.get_dim_fparam(),
            "dim_aparam": model.get_dim_aparam(),
            "dim_chg_spin": model.get_dim_chg_spin()
            if hasattr(model, "get_dim_chg_spin")
            else 0,
            "mixed_types": model.mixed_types(),
            "has_default_fparam": model.has_default_fparam(),
            "default_fparam": model.get_default_fparam(),
            "has_chg_spin_ebd": (
                model.has_chg_spin_ebd()
                if hasattr(model, "has_chg_spin_ebd")
                else False
            ),
            "has_default_chg_spin": (
                model.has_default_chg_spin()
                if hasattr(model, "has_default_chg_spin")
                else False
            ),
            "default_chg_spin": (
                model.get_default_chg_spin()
                if hasattr(model, "get_default_chg_spin")
                else None
            ),
            "is_spin": self._is_spin,
            "lower_input_kind": "nlist",
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
                charge_spin: torch.Tensor | None = None,
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
                    charge_spin=charge_spin,
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
                charge_spin: torch.Tensor | None = None,
            ) -> dict[str, torch.Tensor]:
                ext_coord = ext_coord.detach().requires_grad_(True)
                return model.forward_common_lower(
                    ext_coord,
                    ext_atype,
                    nlist,
                    mapping,
                    fparam=fparam,
                    aparam=aparam,
                    charge_spin=charge_spin,
                    do_atomic_virial=True,
                )

            self.exported_module = _eager_runner

    def get_rcut(self) -> float:
        """Get the cutoff radius of this model."""
        return self._rcut

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        if self._type_map:
            return len(self._type_map)
        return int(self.metadata.get("ntypes", 0))

    def get_type_map(self) -> list[str]:
        """Get the type map (element name of the atom types) of this model."""
        return self._type_map

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this DP."""
        if self._dpmodel is not None:
            return self._dpmodel.get_dim_fparam()
        return int(self.metadata["dim_fparam"])

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        if self._dpmodel is not None:
            return self._dpmodel.get_dim_aparam()
        return int(self.metadata["dim_aparam"])

    def has_chg_spin_ebd(self) -> bool:
        """Check whether the model uses a dedicated charge_spin input."""
        if self._dpmodel is not None and hasattr(self._dpmodel, "has_chg_spin_ebd"):
            return bool(self._dpmodel.has_chg_spin_ebd())
        return bool(self.metadata.get("has_chg_spin_ebd", self.get_dim_chg_spin() > 0))

    def has_default_chg_spin(self) -> bool:
        """Check whether the model has a default charge_spin fallback."""
        if self._dpmodel is not None and hasattr(self._dpmodel, "has_default_chg_spin"):
            return bool(self._dpmodel.has_default_chg_spin())
        return bool(
            self.metadata.get(
                "has_default_chg_spin",
                self.metadata.get("default_chg_spin") is not None,
            )
        )

    def get_dim_chg_spin(self) -> int:
        """Get the width of charge/spin condition inputs."""
        if self._dpmodel is not None and hasattr(self._dpmodel, "get_dim_chg_spin"):
            return self._dpmodel.get_dim_chg_spin()
        return int(self.metadata.get("dim_chg_spin", 0))

    def _make_charge_spin_input(
        self, nframes: int, charge_spin: np.ndarray | None = None
    ) -> torch.Tensor | None:
        """Build the fixed charge/spin tensor used by exported SeZM models."""
        from deepmd.pt_expt.utils.env import (
            DEVICE,
        )

        dim_chg_spin = self.get_dim_chg_spin()
        if dim_chg_spin == 0:
            return None
        if charge_spin is not None:
            return torch.tensor(
                _reshape_charge_spin(charge_spin, nframes, dim_chg_spin),
                dtype=torch.float64,
                device=DEVICE,
            )
        default_chg_spin = self.metadata.get("default_chg_spin")
        if default_chg_spin is None:
            raise ValueError(
                "charge_spin is required for this model but was not provided, "
                "and no default_chg_spin is stored in the model."
            )
        if hasattr(default_chg_spin, "cpu"):
            default_chg_spin = default_chg_spin.cpu().numpy()
        return (
            torch.tensor(default_chg_spin, dtype=torch.float64, device=DEVICE)
            .view(1, dim_chg_spin)
            .expand(nframes, -1)
            .contiguous()
        )

    @property
    def model_type(self) -> type["DeepEvalWrapper"]:
        """The evaluator of the model type."""
        if self._dpmodel is not None:
            model_output_type = self._dpmodel.model_output_type()
        else:
            # Metadata-only mode: derive the output-type set from the
            # fitting_output_defs names.  `model_output_type()` on a
            # dpmodel is the same set — just the base output names, not
            # their derived `*_redu` / `*_derv_*` twins.
            model_output_type = [
                d.name for d in self._model_output_def.def_outp.get_data().values()
            ]
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
        elif (
            self._dpmodel is not None
            and hasattr(self._dpmodel, "get_var_name")
            and self._dpmodel.get_var_name() in model_output_type
        ):
            return DeepProperty
        else:
            raise RuntimeError("Unknown model type")

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        if self._dpmodel is not None:
            return self._dpmodel.get_sel_type()
        # Metadata-only mode: read the `sel_type` field populated by
        # `_collect_metadata`.  Missing field → `[]` (every type
        # selected), matching the dpmodel default for energy models.
        return [int(t) for t in self.metadata.get("sel_type", [])]

    def get_numb_dos(self) -> int:
        """Get the number of DOS."""
        return 0

    def get_var_name(self) -> str:
        """Get the name of the property (property models only)."""
        if self._dpmodel is not None and hasattr(self._dpmodel, "get_var_name"):
            return self._dpmodel.get_var_name()
        raise NotImplementedError(
            "get_var_name is only available for property models with the "
            "reconstructed dpmodel (not in metadata-only mode)."
        )

    def get_task_dim(self) -> int:
        """Get the output dimension of the property (property models only)."""
        if self._dpmodel is not None and hasattr(self._dpmodel, "get_task_dim"):
            return self._dpmodel.get_task_dim()
        raise NotImplementedError(
            "get_task_dim is only available for property models with the "
            "reconstructed dpmodel (not in metadata-only mode)."
        )

    def get_intensive(self) -> bool:
        """Whether the property is intensive (property models only)."""
        if self._dpmodel is not None and hasattr(self._dpmodel, "get_intensive"):
            return self._dpmodel.get_intensive()
        raise NotImplementedError(
            "get_intensive is only available for property models with the "
            "reconstructed dpmodel (not in metadata-only mode)."
        )

    def get_has_efield(self) -> bool:
        """Check if the model has efield."""
        return False

    def get_has_spin(self) -> bool:
        """Check if the model has spin atom types."""
        return self._is_spin

    def get_use_spin(self) -> list[bool]:
        """Get the per-type spin usage of this model."""
        if not self._is_spin:
            return []
        if self._dpmodel is not None:
            return self._dpmodel.spin.use_spin.tolist()
        return [bool(v) for v in self.metadata.get("use_spin", [])]

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
            The array should be of size nframes x dim_fparam.
        aparam
            The atomic parameter.
            The array should be of size nframes x natoms x dim_aparam.
        charge_spin
            The charge and spin values for each frame.
            The array should be reshape-compatible with nframes x 2, where the first
            column is charge and the second column is spin. If the model has
            add_chg_spin_ebd=True and no default_chg_spin is set, this parameter is
            required. If default_chg_spin is configured, this parameter is optional
            and will override the default.
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
                coords,
                cells,
                atom_types,
                spins,
                fparam,
                aparam,
                request_defs,
                charge_spin,
            )
        else:
            out = self._eval_func(self._eval_model, numb_test, natoms)(
                coords,
                cells,
                atom_types,
                fparam,
                aparam,
                request_defs,
                charge_spin,
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
        rcut = self._rcut
        sel = self._sel
        mixed_types = self._mixed_types

        # Model-level pair exclusion is a nlist-BUILD transform (decision
        # #18/A4): fold it in here; the exported dense lower consumes a
        # pre-excluded nlist and never re-applies it.
        pair_excl = self._model_pair_excl()
        if self._nlist_builder is not None:
            # O(N) cell-list strategy (e.g. vesin): builds the same extended
            # representation.  Match the native builder's type handling
            # (``distinguish_types=not mixed_types``) so consumers that bypass
            # ``forward_common_lower``'s ``format_nlist`` -- e.g.
            # ``eval_descriptor`` calling the descriptor directly -- receive the
            # type-distinguished nlist a non-mixed-type descriptor expects.  The
            # main eval path is unaffected (its ``format_nlist`` re-formats).
            extended_coord, extended_atype, nlist, mapping = self._nlist_builder.build(
                coords, atom_types, cells, rcut, sel, pair_excl=pair_excl
            )
            if not mixed_types:
                nlist = nlist_distinguish_types(nlist, extended_atype, sel)
            return extended_coord, extended_atype, nlist, mapping

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
            pair_excl=pair_excl,
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
        extended_atype = np.stack(ext_atypes, axis=0)
        nlist = np.stack(nlists, axis=0)
        # Model-level pair exclusion is a nlist-BUILD transform (decision
        # #18/A4): fold it in here, like the native builder path.
        from deepmd.dpmodel.utils.nlist import (
            apply_pair_exclusion_nlist,
        )

        nlist = apply_pair_exclusion_nlist(
            nlist, extended_atype, self._model_pair_excl()
        )
        return (
            np.stack(ext_coords, axis=0),
            extended_atype,
            nlist,
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
        sel = self._sel
        mixed_types = self._mixed_types
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
        rcut = self._rcut
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

    @staticmethod
    def _build_edge_inputs_from_nlist(
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a padded neighbor list into the compact-edge schema."""
        nloc = nlist.shape[1]
        schema = edge_schema_from_extended(
            extended_coord,
            extended_atype[:, :nloc],
            nlist,
            mapping,
        )
        return (
            schema.edge_index,
            schema.edge_vec,
            schema.edge_scatter_index,
            schema.edge_mask,
        )

    def _prepare_nlist_inputs(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        charge_spin: np.ndarray | None = None,
    ) -> tuple:
        """Prepare the extended-coordinate and padded-neighbor-list inputs.

        Returns
        -------
        tuple
            (ext_coord_t, ext_atype_t, nlist_t, mapping_t,
             fparam_t, aparam_t, charge_spin_t, nframes, natoms)
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

        charge_spin_t = self._make_charge_spin_input(nframes, charge_spin)

        return (
            ext_coord_t,
            ext_atype_t,
            nlist_t,
            mapping_t,
            fparam_t,
            aparam_t,
            charge_spin_t,
            nframes,
            natoms,
        )

    def _prepare_inputs(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        charge_spin: np.ndarray | None = None,
    ) -> tuple[tuple[torch.Tensor | None, ...], torch.Tensor, int, int]:
        """Prepare lower-interface inputs and the output fold-back mapping."""
        if (
            self.metadata.get("lower_input_kind") == "edge_vec"
            and self._nlist_builder is not None
            and self.neighbor_list is None
        ):
            from deepmd.pt_expt.utils.env import (
                DEVICE,
            )

            nframes = coords.shape[0]
            if len(atom_types.shape) == 1:
                natoms = len(atom_types)
                atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
            else:
                natoms = len(atom_types[0])
            coord_t = torch.tensor(
                coords.reshape(nframes, natoms, 3),
                dtype=torch.float64,
                device=DEVICE,
            )
            atype_t = torch.tensor(atom_types, dtype=torch.int64, device=DEVICE)
            cells_t = (
                torch.tensor(cells, dtype=torch.float64, device=DEVICE)
                if cells is not None
                else None
            )
            edge_schema = self._nlist_builder.build(
                coord_t,
                atype_t,
                cells_t,
                self._rcut,
                self._sel,
                return_mode="edges",
            )
            fparam_t, aparam_t = self._prepare_optional_lower_inputs(
                fparam,
                aparam,
                nframes,
                natoms,
                DEVICE,
            )
            charge_spin_t = self._make_charge_spin_input(nframes, charge_spin)
            model_inputs = (
                edge_schema.coord,
                edge_schema.atype,
                edge_schema.edge_index,
                edge_schema.edge_vec,
                edge_schema.edge_scatter_index,
                edge_schema.edge_mask,
                fparam_t,
                aparam_t,
                charge_spin_t,
            )
            mapping_t = torch.arange(natoms, dtype=torch.int64, device=DEVICE).reshape(
                1, natoms
            )
            mapping_t = mapping_t.expand(nframes, -1).contiguous()
            return model_inputs, mapping_t, nframes, natoms

        (
            ext_coord_t,
            ext_atype_t,
            nlist_t,
            mapping_t,
            fparam_t,
            aparam_t,
            charge_spin_t,
            nframes,
            natoms,
        ) = self._prepare_nlist_inputs(
            coords, cells, atom_types, fparam, aparam, charge_spin
        )
        if self.metadata.get("lower_input_kind") == "edge_vec":
            edge_index_t, edge_vec_t, edge_scatter_t, edge_mask_t = (
                self._build_edge_inputs_from_nlist(
                    ext_coord_t,
                    ext_atype_t,
                    nlist_t,
                    mapping_t,
                )
            )
            model_inputs = (
                ext_coord_t,
                ext_atype_t[:, :natoms],
                edge_index_t,
                edge_vec_t,
                edge_scatter_t,
                edge_mask_t,
                fparam_t,
                aparam_t,
                charge_spin_t,
            )
        else:
            model_inputs = (
                ext_coord_t,
                ext_atype_t,
                nlist_t,
                mapping_t,
                fparam_t,
                aparam_t,
                charge_spin_t,
            )
        return model_inputs, mapping_t, nframes, natoms

    def _prepare_optional_lower_inputs(
        self,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        nframes: int,
        natoms: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Prepare optional frame and atomic parameters for lower interfaces."""
        if fparam is not None:
            fparam_t = torch.tensor(
                fparam.reshape(nframes, self.get_dim_fparam()),
                dtype=torch.float64,
                device=device,
            )
        elif self.get_dim_fparam() > 0:
            default_fp = self.metadata.get("default_fparam")
            if default_fp is None:
                raise ValueError(
                    f"fparam is required for this model (dim_fparam={self.get_dim_fparam()}) "
                    "but was not provided, and no default_fparam is stored in the model."
                )
            fparam_t = (
                torch.tensor(default_fp, dtype=torch.float64, device=device)
                .unsqueeze(0)
                .expand(nframes, -1)
                .contiguous()
            )
        else:
            fparam_t = None

        if aparam is not None:
            aparam_t = torch.tensor(
                aparam.reshape(nframes, natoms, self.get_dim_aparam()),
                dtype=torch.float64,
                device=device,
            )
        elif self.get_dim_aparam() > 0:
            raise ValueError(
                f"aparam is required for this model (dim_aparam={self.get_dim_aparam()}) "
                "but was not provided."
            )
        else:
            aparam_t = None
        return fparam_t, aparam_t

    def _eval_model(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        request_defs: list[OutputVariableDef],
        charge_spin: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...]:
        if self.metadata.get("lower_input_kind") in ("graph", "dpa1_canonical"):
            return self._eval_model_graph(
                coords, cells, atom_types, fparam, aparam, request_defs, charge_spin
            )
        model_inputs, mapping_t, nframes, natoms = self._prepare_inputs(
            coords, cells, atom_types, fparam, aparam, charge_spin
        )
        if self._is_pt2:
            # AOTInductor's __call__ unflattens output using stored out_spec,
            # returning a dict just like the .pte module.
            # It also filters non-tensor args automatically, matching the
            # export-time signature where None args were excluded.
            model_ret = self._pt2_runner(*model_inputs)
        else:
            model_ret = self.exported_module(*model_inputs)

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
        charge_spin: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...]:
        if self.metadata.get("lower_input_kind") == "graph":
            # Native-spin (NeighborGraph route): no virtual atoms and no
            # extended/nlist ABI at all -- dispatch to the graph-native fast
            # path (mirrors _eval_model's dispatch to _eval_model_graph for
            # the non-spin case). charge_spin rides the conditional slot-13
            # tail (see NativeSpinEnergyModel.forward_lower_graph_exportable).
            return self._eval_model_graph_spin(
                coords,
                cells,
                atom_types,
                spins,
                fparam,
                aparam,
                request_defs,
                charge_spin=charge_spin,
            )
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

        charge_spin_t = self._make_charge_spin_input(nframes, charge_spin)

        # Build the lower inputs for the model's spin ABI. The native scheme
        # shares the energy edge contract and feeds the owned-atom spins (the
        # descriptor only needs local spins; ghost neighbours resolve to their
        # local owners). The deepspin scheme keeps the extended nlist contract.
        if self.metadata.get("lower_input_kind") == "edge_vec":
            edge_schema = edge_schema_from_extended(
                ext_coord_t,
                ext_atype_t[:, :natoms],
                nlist_t,
                mapping_t,
            )
            model_inputs = (
                edge_schema.coord,
                edge_schema.atype,
                edge_schema.edge_index,
                edge_schema.edge_vec,
                edge_schema.edge_scatter_index,
                edge_schema.edge_mask,
                spin_t,
                fparam_t,
                aparam_t,
                charge_spin_t,
            )
        else:
            model_inputs = (
                ext_coord_t,
                ext_atype_t,
                ext_spin_t,
                nlist_t,
                mapping_t,
                fparam_t,
                aparam_t,
                charge_spin_t,
            )
        if self._is_pt2:
            model_ret = self._pt2_runner(*model_inputs)
        else:
            model_ret = self.exported_module(*model_inputs)

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

    def _eval_model_graph_spin(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        spins: np.ndarray,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        request_defs: list[OutputVariableDef],
        charge_spin: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Evaluate a graph-form native-spin ``.pt2`` (``lower_input_kind ==
        "graph"`` and ``is_spin``).

        Mirrors :meth:`_eval_model_graph`'s carry-all
        :class:`~deepmd.dpmodel.utils.neighbor_graph.NeighborGraph`
        construction (SAME builder, SAME positional ABI up through
        ``source_row_ptr``), then inserts the owned-atom ``spin`` tensor
        ``(N, 3)`` at positional index 10 of
        ``NativeSpinEnergyModel.forward_lower_graph_exportable`` -- the node
        axis IS the owned-local-atom axis for single-rank eval (no ghost
        nodes), so ``spin`` needs no extension/mapping, unlike the dense
        spin path's ``ext_spin_t``. ``charge_spin`` rides the conditional
        slot-13 tail (combined native-spin + charge-spin FiLM models; the
        slot is dropped from the exported signature otherwise). The forward
        returns LOCAL public keys directly (``atom_energy``,
        ``energy``, ``force``, ``force_mag``, ``virial``, ``atom_virial``),
        so results are reshaped without ``communicate_extended_output``,
        same as the non-spin graph path.
        """
        from deepmd.pt_expt.utils.env import (
            DEVICE,
        )

        nframes = coords.shape[0]
        if len(atom_types.shape) == 1:
            natoms = len(atom_types)
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        else:
            natoms = len(atom_types[0])

        coord_input = coords.reshape(nframes, natoms, 3)
        box_input = cells.reshape(nframes, 9) if cells is not None else None
        graph = self._build_eval_graph(coord_input, atom_types, box_input, DEVICE)

        atype_t = torch.tensor(
            np.asarray(atom_types).reshape(-1), dtype=torch.int64, device=DEVICE
        )
        n_node_t = torch.as_tensor(graph.n_node, dtype=torch.int64, device=DEVICE)
        edge_index_t = torch.as_tensor(
            graph.edge_index, dtype=torch.int64, device=DEVICE
        )
        edge_dtype = (
            torch.float32
            if self.metadata.get("graph_edge_dtype") == "float32"
            else torch.float64
        )
        edge_vec_t = torch.as_tensor(
            graph.edge_vec,
            dtype=edge_dtype,
            device=DEVICE,
        )
        edge_mask_t = torch.as_tensor(graph.edge_mask, dtype=torch.bool, device=DEVICE)
        destination_order_t = torch.as_tensor(
            graph.destination_order,
            device=DEVICE,
        )
        destination_row_ptr_t = torch.as_tensor(
            graph.destination_row_ptr,
            dtype=torch.int64,
            device=DEVICE,
        )
        source_order_t = torch.as_tensor(
            graph.source_order,
            device=DEVICE,
        )
        source_row_ptr_t = torch.as_tensor(
            graph.source_row_ptr,
            dtype=torch.int64,
            device=DEVICE,
        )

        spin_t = torch.tensor(
            np.asarray(spins).reshape(nframes * natoms, 3),
            dtype=torch.float64,
            device=DEVICE,
        )

        fparam_t, aparam_t = self._prepare_optional_lower_inputs(
            fparam, aparam, nframes, natoms, DEVICE
        )
        if aparam_t is not None:
            # graph-lower ABI: aparam is FLAT on the node axis, (N, nda) --
            # the same axis as ``atype``/``spin`` (mirrors _eval_model_graph).
            aparam_t = aparam_t.reshape(nframes * natoms, -1)

        model_inputs = (
            atype_t,
            n_node_t,
            n_node_t,
            edge_index_t,
            edge_vec_t,
            edge_mask_t,
            destination_order_t,
            destination_row_ptr_t,
            source_order_t,
            source_row_ptr_t,
            spin_t,
            fparam_t,
            aparam_t,
            self._make_charge_spin_input(nframes, charge_spin),
        )
        if self._is_pt2:
            model_ret = self._pt2_runner(*model_inputs)
        else:
            model_ret = self.exported_module(*model_inputs)

        results = []
        for odef in request_defs:
            shape = self._get_output_shape(odef, nframes, natoms)
            if odef.name == "mask_mag":
                # The graph ABI does not export mask_mag; it is a pure
                # function of the artifact's ``use_spin`` metadata and the
                # input atom types (``use_spin[atype]`` -- same math as the
                # eager model's ``spin_mask[atype] > 0``), so the adapter
                # synthesizes it here to preserve the public DeepPot
                # semantics (``results["mask_mag"]`` is always valid for a
                # spin model).
                use_spin = np.asarray(self.get_use_spin(), dtype=bool)
                mask_mag = use_spin[np.asarray(atom_types)][..., None]
                results.append(mask_mag.reshape(shape))
                continue
            gkey = _graph_spin_output_key(odef)
            val = model_ret.get(gkey) if gkey is not None else None
            if val is not None:
                results.append(val.detach().cpu().numpy().reshape(shape))
            else:
                results.append(
                    np.full(np.abs(shape), np.nan, dtype=GLOBAL_NP_FLOAT_PRECISION)
                )
        return tuple(results)

    def _eval_model_graph(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        request_defs: list[OutputVariableDef],
        charge_spin: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Evaluate a graph-form ``.pt2`` (``lower_input_kind == "graph"``).

        Builds a carry-all :class:`~deepmd.dpmodel.utils.neighbor_graph.NeighborGraph`
        from the eval system at its exact (tight) edge count and feeds the
        positional schema
        ``(atype, n_node, n_local, edge_index, edge_vec, edge_mask,
        destination_order, destination_row_ptr, source_order, source_row_ptr,
        fparam, aparam, charge_spin)`` to the exported forward. The AOTI
        artifact's edge axis is dynamic, so no ``edge_capacity`` padding is needed. The
        ``graph_edge_dtype`` metadata selects float32 geometry for compressed
        DPA1 and float64 for generic graph descriptors. The forward returns the
        LOCAL public keys directly, so results are reshaped without
        ``communicate_extended_output``.
        """
        from deepmd.pt_expt.utils.env import (
            DEVICE,
        )

        nframes = coords.shape[0]
        if len(atom_types.shape) == 1:
            natoms = len(atom_types)
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        else:
            natoms = len(atom_types[0])

        coord_input = coords.reshape(nframes, natoms, 3)
        box_input = cells.reshape(nframes, 9) if cells is not None else None
        # Build the carry-all graph at its exact edge count; the exported edge
        # axis is dynamic.
        graph = self._build_eval_graph(coord_input, atom_types, box_input, DEVICE)

        atype_t = torch.tensor(
            np.asarray(atom_types).reshape(-1), dtype=torch.int64, device=DEVICE
        )
        # graph fields may be numpy (dense/ase) or torch, possibly on CUDA
        # (vesin/nv) -- torch.as_tensor handles both and moves to DEVICE.
        n_node_t = torch.as_tensor(graph.n_node, dtype=torch.int64, device=DEVICE)
        edge_index_t = torch.as_tensor(
            graph.edge_index, dtype=torch.int64, device=DEVICE
        )
        edge_dtype = (
            torch.float32
            if self.metadata.get("graph_edge_dtype") == "float32"
            else torch.float64
        )
        edge_vec_t = torch.as_tensor(
            graph.edge_vec,
            dtype=edge_dtype,
            device=DEVICE,
        )
        edge_mask_t = torch.as_tensor(graph.edge_mask, dtype=torch.bool, device=DEVICE)
        destination_order_t = torch.as_tensor(
            graph.destination_order,
            device=DEVICE,
        )
        destination_row_ptr_t = torch.as_tensor(
            graph.destination_row_ptr,
            dtype=torch.int64,
            device=DEVICE,
        )
        source_order_t = torch.as_tensor(
            graph.source_order,
            device=DEVICE,
        )
        source_row_ptr_t = torch.as_tensor(
            graph.source_row_ptr,
            dtype=torch.int64,
            device=DEVICE,
        )

        if self.metadata.get("lower_input_kind") == "dpa1_canonical":
            # The canonical ABI has NO fparam/aparam/charge_spin slots; the
            # export gate (fitting_eligible) rejects such models today, so
            # this is unreachable -- assert it loudly so a future loosening
            # of the eligibility fails here instead of silently dropping
            # conditioning inputs at inference.
            if (
                self.get_dim_fparam() > 0
                or self.get_dim_aparam() > 0
                or int(self.metadata.get("dim_chg_spin", 0) or 0) > 0
            ):
                raise NotImplementedError(
                    "dpa1_canonical artifacts carry no fparam/aparam/"
                    "charge_spin inputs; a model requiring them must not be "
                    "frozen with lower_kind='dpa1_canonical' (the export "
                    "eligibility gate should have rejected it)."
                )
            from deepmd.dpmodel.utils.neighbor_graph import (
                NeighborGraph,
            )
            from deepmd.pt_expt.utils.canonical_graph import (
                canonical_graph_from_neighbor_graph,
            )

            generic_graph = NeighborGraph(
                n_node=n_node_t,
                edge_index=edge_index_t,
                edge_vec=edge_vec_t,
                edge_mask=edge_mask_t,
                n_local=n_node_t,
                destination_order=destination_order_t,
                destination_row_ptr=destination_row_ptr_t,
                source_order=source_order_t,
                source_row_ptr=source_row_ptr_t,
                destination_sorted=bool(graph.destination_sorted),
            )
            compact = canonical_graph_from_neighbor_graph(generic_graph)
            model_inputs = (
                atype_t,
                compact.n_node,
                compact.n_local,
                compact.source,
                compact.edge_vec,
                compact.destination_row_ptr,
                compact.source_row_ptr,
                compact.source_order,
            )
        else:
            fparam_t, aparam_t = self._prepare_optional_lower_inputs(
                fparam, aparam, nframes, natoms, DEVICE
            )
            if aparam_t is not None:
                # graph-lower ABI: aparam is FLAT on the node axis, (N, nda)
                # -- the same axis as ``atype`` (the shared helper above
                # returns the dense rectangular (nf, natoms, nda) layout).
                aparam_t = aparam_t.reshape(nframes * natoms, -1)
            charge_spin_t = self._make_charge_spin_input(nframes, charge_spin)
            model_inputs = (
                atype_t,
                n_node_t,
                n_node_t,
                edge_index_t,
                edge_vec_t,
                edge_mask_t,
                destination_order_t,
                destination_row_ptr_t,
                source_order_t,
                source_row_ptr_t,
                fparam_t,
                aparam_t,
                charge_spin_t,
            )
        if self._is_pt2:
            model_ret = self._pt2_runner(*model_inputs)
        else:
            model_ret = self.exported_module(*model_inputs)

        results = []
        for odef in request_defs:
            shape = self._get_output_shape(odef, nframes, natoms)
            gkey = _GRAPH_CATEGORY_TO_KEY.get(odef.category)
            val = model_ret.get(gkey) if gkey is not None else None
            if val is not None:
                results.append(val.detach().cpu().numpy().reshape(shape))
            else:
                results.append(
                    np.full(np.abs(shape), np.nan, dtype=GLOBAL_NP_FLOAT_PRECISION)
                )
        return tuple(results)

    def _build_eval_graph(
        self,
        coord_input: np.ndarray,
        atom_types: np.ndarray,
        box_input: np.ndarray | None,
        device: "torch.device",
    ) -> "NeighborGraph":
        """Build the carry-all NeighborGraph for graph-form ``.pt2`` inference.

        Dispatches on ``self._neighbor_graph_method``: ``dense``/``ase`` run
        backend-agnostic (numpy); ``vesin``/``nv`` run on-device (torch, O(N)).
        All backends emit the SAME neighbor set (carry-all, sel-free), so the
        selection is a pure performance choice and results are unchanged. The
        result is canonicalized to the destination-major graph-form ``.pt2``
        ABI after construction.
        """
        method = self._neighbor_graph_method
        # Model-level ``pair_exclude_types`` is a graph-BUILD transform
        # (decision #18): apply it here so the exported ``.pt2`` lower consumes a
        # pre-excluded ``edge_mask`` and never re-applies it (mirrors the C++
        # ``applyPairExclusion`` and the eager dpmodel/pt_expt build path).
        pair_excl = self._model_pair_excl()
        if method == "dense":
            from deepmd.dpmodel.utils.neighbor_graph import (
                build_neighbor_graph,
            )

            return build_neighbor_graph(
                coord_input,
                atom_types,
                box_input,
                self._rcut,
                canonicalize=True,
                pair_excl=pair_excl,
            )
        if method == "ase":
            from deepmd.dpmodel.utils.neighbor_graph import (
                build_neighbor_graph_ase,
            )

            return build_neighbor_graph_ase(
                coord_input,
                atom_types,
                box_input,
                self._rcut,
                canonicalize=True,
                pair_excl=pair_excl,
            )
        if method in ("vesin", "nv"):
            cc = torch.as_tensor(coord_input, dtype=torch.float64, device=device)
            aa = torch.as_tensor(
                np.asarray(atom_types), dtype=torch.int64, device=device
            )
            bb = (
                torch.as_tensor(box_input, dtype=torch.float64, device=device)
                if box_input is not None
                else None
            )
            if method == "vesin":
                from deepmd.pt_expt.utils.vesin_graph_builder import (
                    build_neighbor_graph_vesin,
                )

                return build_neighbor_graph_vesin(
                    cc,
                    aa,
                    bb,
                    self._rcut,
                    canonicalize=True,
                    pair_excl=pair_excl,
                )
            from deepmd.pt_expt.utils.nv_graph_builder import (
                build_neighbor_graph_nv,
            )

            return build_neighbor_graph_nv(
                cc,
                aa,
                bb,
                self._rcut,
                canonicalize=True,
                pair_excl=pair_excl,
            )
        raise ValueError(
            f"unknown neighbor_graph_method {method!r}; "
            "use 'dense', 'ase', 'vesin', or 'nv'"
        )

    def _model_pair_excl(self) -> "PairExcludeMask | None":
        """Model-level ``pair_exclude_types`` as a ``PairExcludeMask`` (or None).

        Applied at graph BUILD time (decision #18), NOT inside the exported
        ``.pt2`` lower. Reads the excluded pairs from the loaded dpmodel (if any)
        or the ``pair_exclude_types`` field in ``metadata.json``, and returns a
        FRESH numpy-backed mask.

        A numpy ``type_mask`` converts cleanly onto whichever namespace/device the
        builder's ``atype`` uses (dense/ase pass numpy; vesin/nv pass torch). The
        dpmodel's own ``pair_excl`` is NOT reused: as a pt_expt module attribute
        its ``type_mask`` is a torch (possibly CUDA) buffer, which cannot convert
        to a numpy ``atype`` on the dense/ase build path.

        Returns
        -------
        PairExcludeMask | None
            The exclusion mask, or ``None`` when the model excludes no pairs.
        """
        from deepmd.dpmodel.utils.exclude_mask import (
            PairExcludeMask,
        )

        if self._dpmodel is not None:
            pe = getattr(self._dpmodel.atomic_model, "pair_excl", None)
            pet = pe.get_exclude_types() if pe is not None else []
        else:
            pet = self.metadata.get("pair_exclude_types", [])
        if not pet:
            return None
        return PairExcludeMask(len(self._type_map), [tuple(p) for p in pet])

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

    def serialize(self) -> dict[str, Any]:
        from deepmd.pt_expt.utils.serialization import (
            serialize_from_file,
        )

        data = serialize_from_file(self.model_path)
        return data["model"] if isinstance(data, dict) and "model" in data else data

    def get_model(self) -> torch.nn.Module:
        """Get the exported model module.

        Returns
        -------
        torch.nn.Module
            The exported model module.
        """
        return self.exported_module

    def _is_spin_model(self) -> bool:
        """Check if the underlying model is a SpinModel.

        Primary path: the :attr:`_is_spin` attribute set by the loaders
        — this works for both ``model.json`` and metadata-only archives
        (a spin ``.pt2`` carries ``is_spin=true`` in its metadata).

        Legacy path: ``isinstance(_dpmodel, SpinModel)`` — retained for
        tests that construct a non-spin archive and then swap
        :attr:`_dpmodel` to a :class:`SpinModel` instance after load.
        """
        if self._is_spin:
            return True
        if self._dpmodel is None:
            return False
        from deepmd.dpmodel.model.spin_model import (
            SpinModel,
        )

        return isinstance(self._dpmodel, SpinModel)

    def _require_dpmodel(self, feature: str) -> None:
        """Guard for features that need a deserialised dpmodel instance.

        ``eval_descriptor`` / ``eval_typeebd`` / ``eval_fitting_last_layer``
        all introspect the dpmodel's internal sub-modules, which requires
        ``model.json`` to have been present at load time.  Archives
        shipped without ``model.json`` (metadata-only mode) can still run
        the main ``eval`` inference path but cannot expose these hooks.
        """
        if self._dpmodel is None:
            raise NotImplementedError(
                f"{feature} requires the dpmodel instance, which is only "
                "available when the .pt2 / .pte archive contains "
                "'model.json'. The loaded archive is metadata-only; "
                "re-export with the full dpmodel serialisation to enable "
                "this feature."
            )

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
        NotImplementedError
            If the archive was loaded in metadata-only mode.
        """
        self._require_dpmodel("eval_typeebd")

        from deepmd.dpmodel.utils.type_embed import TypeEmbedNet as TypeEmbedNetDP
        from deepmd.pt_expt.model.spin_model import (
            SpinModel,
        )

        model = self._dpmodel
        if isinstance(model, SpinModel):
            # Virtual-atom wrapper: type-embed nets live on the backbone.
            # Native-spin models ARE the model (is-a); no unwrap.
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
        charge_spin: np.ndarray | None = None,
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
        self._require_dpmodel("eval_descriptor")

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
            charge_spin_t,
            _nframes,
            _natoms,
        ) = self._prepare_nlist_inputs(
            coords, cells, atom_types, fparam, aparam, charge_spin
        )
        with torch.no_grad():
            descriptor, *_ = dp_am.descriptor(
                ext_coord_t,
                ext_atype_t,
                nlist_t,
                mapping=mapping_t,
                charge_spin=charge_spin_t
                if getattr(dp_am, "add_chg_spin_ebd", False)
                else None,
            )
        return descriptor.detach().cpu().numpy()

    def eval_fitting_last_layer(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        charge_spin: np.ndarray | None = None,
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
        self._require_dpmodel("eval_fitting_last_layer")

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
            charge_spin_t,
            _nframes,
            natoms,
        ) = self._prepare_nlist_inputs(
            coords, cells, atom_types, fparam, aparam, charge_spin
        )
        with torch.no_grad():
            descriptor, rot_mat, g2, h2, _sw = dp_am.descriptor(
                ext_coord_t,
                ext_atype_t,
                nlist_t,
                mapping=mapping_t,
                charge_spin=charge_spin_t
                if getattr(dp_am, "add_chg_spin_ebd", False)
                else None,
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
