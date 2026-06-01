# dpa_tools/finetuner.py
#
# Path B architecture: frozen DPA descriptor → sklearn predictor
# DPA checkpoint is used purely as a feature extractor (no dp train).

import os
from pathlib import Path
from typing import List, Optional, Union

import dpdata
import numpy as np

from deepmd.dpa_tools._backend import (
    _DescriptorExtraction,
    build_model_from_config,
    get_torch_device,
    load_torch_file,
    resolve_model_branch,
)
from deepmd.dpa_tools.conditions import ConditionManager, DPAConditionError
from deepmd.dpa_tools.data.errors import DPADataError
from deepmd.dpa_tools.data.loader import load_data, _resolve_label_key, _get_source
from deepmd.dpa_tools.utils.dotdict import DotDict


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _load_labels(
    systems: List[dpdata.System],
    target_key: str,
) -> np.ndarray:
    """Load and concatenate labels from dpdata systems.

    *target_key* is resolved through ``_LABEL_KEY_ALIASES`` so that
    ``"energy"`` → ``"energies"`` for backward compatibility.

    When the resolved key is not present in ``system.data`` (dpdata only
    loads standard DeepMD keys), this function falls back to reading
    ``set.*/{key}.npy`` directly from the system source directory.
    """
    resolved = _resolve_label_key(target_key)
    all_labels = []
    for system in systems:
        if resolved in system.data:
            all_labels.append(np.asarray(system.data[resolved]))
            continue

        # Fallback: load set.*/key.npy directly from the system directory.
        source = _get_source(system)
        if source is not None:
            source_path = Path(source)
            set_dirs = sorted(source_path.glob("set.*"))
            npy_labels = []
            for sd in set_dirs:
                npy_path = sd / f"{resolved}.npy"
                if npy_path.exists():
                    npy_labels.append(np.load(npy_path))
            if npy_labels:
                all_labels.append(np.concatenate(npy_labels, axis=0))
                continue

        # Neither dpdata nor direct .npy found — build a clear error.
        available = sorted(system.data.keys())
        if source is not None:
            set_dirs = sorted(Path(source).glob("set.*"))
            available_npy = sorted(set(
                p.name for sd in set_dirs for p in sd.glob("*.npy")
            ))
        else:
            available_npy = []
        msg = (
            f"Label key {resolved!r} not found. "
            f"Checked system.data keys: {available}."
        )
        if available_npy:
            msg += f" Checked set.*/npy files: {available_npy}."
        else:
            msg += " No system source path for direct .npy fallback."
        msg += f" (target_key={target_key!r})."
        raise DPADataError(msg)

    return np.concatenate(all_labels, axis=0)


def _read_data_type_map(system) -> list[str]:
    """Read element symbols from a dpdata System's ``atom_names``.

    Returns an empty list when the names are dpdata's auto-generated
    ``Type_0`` / ``Type_1`` placeholders (which appear when the source
    data had no ``type_map.raw``).
    """
    names = list(system.data.get("atom_names", []))
    if not names:
        return []
    # dpdata generates "Type_0", "Type_1", ... when no type_map.raw was present.
    if all(n.startswith("Type_") for n in names):
        return []
    return names


def _load_npy_system(system: dpdata.System):
    """Extract (coords, boxes, atom_types) from a dpdata System.

    Adapts dpdata's native shapes to the format expected by
    ``_extract_features``:

    - coords     : (n_frames, n_atoms*3)  (flattened)
    - boxes      : (n_frames, 9) or None for non-periodic
    - atom_types : (n_atoms,) int

    Returns
    -------
    coords     : np.ndarray, shape (n_frames, n_atoms*3)
    boxes      : np.ndarray, shape (n_frames, 9), or None
    atom_types : np.ndarray, shape (n_atoms,)
    """
    d = system.data
    coords = np.asarray(d["coords"])       # (n_frames, n_atoms, 3)
    n_atoms = coords.shape[1]
    coords = coords.reshape(coords.shape[0], n_atoms * 3)

    cells = np.asarray(d["cells"])          # (n_frames, 3, 3)
    boxes = cells.reshape(cells.shape[0], 9)

    atom_types = np.asarray(d["atom_types"])  # (n_atoms,)

    if d.get("nopbc", False) or np.allclose(boxes, 0):
        boxes = None

    return coords, boxes, atom_types


# ---------------------------------------------------------------------------
# Public descriptor extraction
# ---------------------------------------------------------------------------

def extract_descriptors(
    data,
    pretrained: str,
    model_branch: str = None,
    pooling: str = "mean",
    cache: bool = True,
) -> np.ndarray:
    """
    Extract pooled DPA descriptors for one or more deepmd/npy systems.

    This is the same feature extraction pipeline ``DPAFineTuner.fit()`` uses
    internally, exposed as a standalone function so downstream tools (e.g.
    multi-task fine-tuning, auxiliary-data selection) can share it without
    constructing a finetuner.

    Parameters
    ----------
    data : str | list[str]
        Path(s) to deepmd/npy system directories.
    pretrained : str
        Path to the pretrained DPA checkpoint (.pt).
    model_branch : str, optional
        Branch name for multi-task checkpoints (e.g. ``"Omat24"``).
    pooling : str
        Pooling strategy. One of ``"mean"``, ``"sum"``, ``"mean+std"``,
        ``"mean+std+max+min"``.
    cache : bool
        If True (default), cache the extracted descriptors on disk so
        repeated calls with the same data + checkpoint + pooling are
        instant.  The cache is invalidated when any ``coord.npy`` or the
        checkpoint changes (mtime-based fingerprint).

    Returns
    -------
    np.ndarray
        Pooled descriptor features, shape ``(n_frames_total, feat_dim)``.
        ``feat_dim`` depends on the pooling strategy.
    """
    from deepmd.dpa_tools.data.desc_cache import load_or_extract

    systems = load_data(data)
    return load_or_extract(
        systems=systems,
        pretrained=pretrained,
        model_branch=model_branch,
        pooling=pooling,
        cache=cache,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DPAFineTuner:
    """Frozen DPA descriptor + sklearn head (Path B) or single-task training.

    Two modes, selected by *strategy*:

    ==================  ======================================================
    ``frozen_sklearn``  (default) Encode each system once with the pretrained
                        DPA descriptor, pool, and train a lightweight sklearn
                        regressor (Ridge / KRR / MLP) on top.
    ``linear_probe``    Freeze the DPA backbone, train only a neural property
                        fitting net via ``dp --pt train --finetune``.
    ``finetune``        Load the pretrained backbone and fine-tune the full
                        network (descriptor + fitting net).
    ``scratch``         (known limitation) Random-initialize and train from
                        scratch — type_map is auto-inferred correctly but
                        ``dp --pt train`` exits before writing train.log;
                        descriptor config likely missing required fields.
                        Not recommended for small-data regimes.
    ==================  ======================================================

    .. note::

       ``strategy="scratch"`` is a known limitation as of Phase 2 closeout.
       The entry point and auto-type_map logic are retained, but the emitted
       ``input.json`` does not yet produce a successful ``dp --pt train`` run
       (exit 1 before train.log).  Scratch training on 19-formula small data
       has negligible practical value; completing it is deferred to a future
       phase when larger datasets make random-init training meaningful.

    Parameters
    ----------
    pretrained : str
        Path to the pretrained DPA checkpoint (.pt).  Set to ``None`` for
        ``scratch`` strategy.
    model_branch : str, optional
        Branch name for multi-task checkpoints (e.g. ``"Omat24"``).  Used
        by ``frozen_sklearn`` for descriptor extraction.
    predictor : str
        sklearn head type (``frozen_sklearn`` only): ``"rf"``,
        ``"linear"`` / ``"ridge"``, or ``"mlp"``.
    pooling : str
        Descriptor pooling (``frozen_sklearn`` only): ``"mean"``, ``"sum"``,
        ``"mean+std"``, ``"mean+std+max+min"``.
    seed : int
        Random seed for the sklearn predictor or training.
    strategy : str
        ``"frozen_sklearn"`` (default), ``"linear_probe"``, ``"finetune"``,
        or ``"scratch"``.
    property_name : str
        Property label filename under ``set.*/`` (training paradigms).
    task_dim : int
        Output dimensionality of the property head.
    intensive : bool
        Whether the property is intensive (mean-pool) or extensive (sum).
    init_branch : str
        Checkpoint branch for descriptor init (LP/FT only).
    learning_rate, stop_lr : float
        Exp-decay LR endpoints (training paradigms).
    max_steps : int
        Total training steps.
    batch_size : str or int
        DeepMD-kit batch_size spec.
    loss_function : str
        ``"mse"`` or ``"smooth_mae"``.
    output_dir : str
        Directory for checkpoints, input.json, and logs.
    save_freq, disp_freq : int
        DeepMD-kit save/display intervals.
    """

    _VALID_POOLING = {"mean", "sum", "mean+std", "mean+std+max+min"}
    _VALID_STRATEGIES = {
        "frozen_sklearn", "linear_probe", "finetune", "mft", "scratch",
    }

    def __init__(
        self,
        pretrained="DPA-3.1-3M",
        model_branch=None,
        predictor="rf",
        pooling="mean",
        seed=42,
        # ---- training paradigms ----
        strategy="frozen_sklearn",
        property_name="property",
        task_dim=1,
        intensive=True,
        init_branch="SPICE2",
        learning_rate=1e-3,
        stop_lr=1e-5,
        max_steps=100_000,
        batch_size="auto:512",
        loss_function="mse",
        output_dir="./dpa_output",
        save_freq=10_000,
        disp_freq=1_000,
        # ---- mft-only ----
        aux_branch="MP_traj_v024_alldata_mixu",
        aux_prob: float = 0.5,
        aux_type_map: list[str] | None = None,
        downstream_type_map: list[str] | None = None,
        fitting_net_params: dict | None = None,
        downstream_task_type: str = "property",
        aux_batch_size: str | None = None,
        downstream_batch_size: int | None = None,
    ):
        if pooling not in self._VALID_POOLING:
            raise ValueError(
                f"pooling must be one of {sorted(self._VALID_POOLING)}, "
                f"got {pooling!r}"
            )
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(self._VALID_STRATEGIES)}; "
                f"got {strategy!r}"
            )

        self.strategy = strategy
        # Scratch forces pretrained=None (random init, no ckpt).
        if strategy == "scratch":
            pretrained = None

        self.pretrained      = pretrained
        self.model_branch    = model_branch
        self._predictor_type = predictor
        self.pooling         = pooling
        self.seed            = seed

        # Training-paradigm params (unused by frozen_sklearn).
        self.property_name   = property_name
        self.task_dim        = task_dim
        self.intensive       = intensive
        self.init_branch     = init_branch
        self.learning_rate   = learning_rate
        self.stop_lr         = stop_lr
        self.max_steps       = max_steps
        self.batch_size      = batch_size
        self.loss_function   = loss_function
        self.output_dir      = output_dir
        self.save_freq       = save_freq
        self.disp_freq       = disp_freq

        # MFT-only parameters.
        self.aux_branch            = aux_branch
        self.aux_prob              = aux_prob
        self.aux_type_map          = aux_type_map
        self.downstream_type_map   = downstream_type_map
        self.fitting_net_params    = fitting_net_params
        self.downstream_task_type  = downstream_task_type
        self.aux_batch_size        = aux_batch_size
        self.downstream_batch_size = downstream_batch_size

        if strategy == "mft":
            if not isinstance(property_name, str) or not property_name.isidentifier():
                raise ValueError(
                    "property_name is required when strategy='mft' and must be a "
                    f"valid Python identifier; got {property_name!r}."
                )

        # populated by fit()
        self.type_map           = []
        self._target_key        = None
        self._task_dim          = 1
        self.predictor          = None   # sklearn object after fit()
        self._fitted            = False
        self._model             = None   # lazy-loaded descriptor model (cached)
        self._device            = None   # set when model is first loaded
        self._checkpoint_type_map = []   # set by _load_descriptor_model
        self._condition_manager = None

    # -----------------------------------------------------------------------
    # Internal: descriptor feature extraction
    # -----------------------------------------------------------------------

    def _load_descriptor_model(self):
        """Load the pretrained DPA checkpoint and return a (non-JIT) ModelWrapper."""
        import torch

        state_dict = load_torch_file(self.pretrained)
        if "model" in state_dict:
            state_dict = state_dict["model"]

        input_param = state_dict["_extra_state"]["model_params"]

        if "model_dict" in input_param:
            # Multi-task checkpoint: select the right branch
            model_alias_dict, _ = resolve_model_branch(input_param["model_dict"])
            head = self.model_branch or "Omat24"

            # Case-insensitive fallback
            if head not in model_alias_dict:
                head_lower = head.lower()
                for mk in model_alias_dict:
                    if mk.lower() == head_lower:
                        head = mk
                        break
            assert head in model_alias_dict, (
                f"Branch '{head}' not found. "
                f"Available: {list(model_alias_dict)}"
            )
            head = model_alias_dict[head]

            # Build single-task input_param from the selected branch
            input_param = input_param["model_dict"][head]

            # Remap state dict keys: model.{head}.xxx → model.Default.xxx
            new_sd = {"_extra_state": state_dict["_extra_state"]}
            for key, val in state_dict.items():
                prefix = f"model.{head}."
                if key.startswith(prefix):
                    new_sd[key.replace(prefix, "model.Default.", 1)] = val
            state_dict = new_sd

        self._checkpoint_type_map = list(input_param.get("type_map", []))

        # Build model WITHOUT JIT so that eval_descriptor_hook works
        wrapper = build_model_from_config(input_param)
        wrapper.load_state_dict(state_dict)
        wrapper.eval()

        device = get_torch_device()
        wrapper = wrapper.to(device)
        self._device = device
        return wrapper

    def _validate_type_map(
        self, user_type_map: list[str], systems: list
    ) -> None:
        """Raise DPADataError if any data element is not in the checkpoint type_map.

        The data type_map can be any subset of the checkpoint's type_map — order
        and contiguity are irrelevant. Local indices are remapped to checkpoint
        global indices in ``_extract_features``.
        """
        ckpt = self._checkpoint_type_map
        if not ckpt:
            return  # checkpoint has no type_map metadata → skip

        ckpt_set = set(ckpt)

        def _check(candidate: list[str], source: str) -> None:
            unsupported = [e for e in candidate if e not in ckpt_set]
            if unsupported:
                ckpt_repr = (
                    f"{ckpt[:3] + ['...'] + ckpt[-1:]} ({len(ckpt)} elements)"
                    if len(ckpt) > 8 else str(ckpt)
                )
                raise DPADataError(
                    f"Element(s) in {source} not supported by this checkpoint.\n"
                    f"  Data type_map     : {candidate}\n"
                    f"  Checkpoint covers : {ckpt_repr}\n"
                    f"  Unsupported       : {unsupported}\n"
                    "Please re-convert your data with a supported element set."
                )

        if user_type_map:
            _check(user_type_map, "user-provided type_map")

        for system in systems:
            data_tm = _read_data_type_map(system)
            if data_tm:
                identifier = system.orig if hasattr(system, "orig") else "system"
                _check(data_tm, f"atom_names of {identifier}")

    def _remap_atom_types(
        self, atom_types: np.ndarray, system
    ) -> np.ndarray:
        """Map local atom-type indices to checkpoint-global indices.

        ``atom_types`` are 0-based indices into the system's type_map.
        The model expects indices into the checkpoint's ``type_map``.
        """
        ckpt = self._checkpoint_type_map

        data_tm = _read_data_type_map(system) or list(self.type_map)

        identifier = system.orig if hasattr(system, "orig") else "system"

        if not data_tm:
            if ckpt and atom_types.size and int(atom_types.max()) >= len(ckpt):
                raise DPADataError(
                    f"No atom_names in system and no type_map provided, "
                    f"but atom type index {int(atom_types.max())} "
                    f"is out of range for the checkpoint type_map "
                    f"(size {len(ckpt)}). "
                    "Pass type_map=[...] to fit()."
                )
            return atom_types

        if not ckpt:
            return atom_types

        try:
            local_to_global = np.array(
                [ckpt.index(elem) for elem in data_tm], dtype=np.int64,
            )
        except ValueError as e:
            unsupported = [e for e in data_tm if e not in set(ckpt)]
            raise DPADataError(
                f"Element(s) in data type_map for {identifier!r} not "
                f"supported by this checkpoint.\n"
                f"  Data type_map : {data_tm}\n"
                f"  Unsupported   : {unsupported}"
            ) from e

        if atom_types.size and int(atom_types.max()) >= len(local_to_global):
            raise DPADataError(
                f"atom type index {int(atom_types.max())} in {identifier!r} "
                f"exceeds the data type_map size ({len(local_to_global)}). "
                "Check that type_map and atom_types are consistent."
            )

        return local_to_global[atom_types]

    def _extract_features_cached(self, systems: list) -> np.ndarray:
        """Call ``_extract_features`` with descriptor-cache lookup.

        Uses the same cache-key scheme as ``load_or_extract()``.  Falls
        back to direct extraction when the cache key cannot be computed
        (e.g. the pretrained file does not exist on disk).
        """
        try:
            from deepmd.dpa_tools.data.desc_cache import _cache_key, _cache_dir

            key = _cache_key(systems, self.pretrained, self.pooling)
            cache_path = _cache_dir() / f"{key}.npy"
            if cache_path.is_file():
                return np.load(cache_path)
        except Exception:
            pass

        features = self._extract_features(systems)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, features)
        except Exception:
            pass
        return features

    def _extract_features(self, systems: list) -> np.ndarray:
        """Extract per-structure descriptor features by pooling over atoms.

        The pooling strategy is controlled by ``self.pooling``:
        - ``"mean"``             → shape (n_frames, feat_dim)
        - ``"sum"``              → shape (n_frames, feat_dim)
        - ``"mean+std"``         → shape (n_frames, feat_dim*2)
        - ``"mean+std+max+min"`` → shape (n_frames, feat_dim*4)

        Parameters
        ----------
        systems : list[dpdata.System]
            dpdata systems to extract descriptors from.

        Returns
        -------
        np.ndarray, shape (n_frames_total, feature_dim)
        """
        import torch

        if self._model is None:
            self._model = self._load_descriptor_model()

        extractor = _DescriptorExtraction(self._model)
        extractor._enable_hook()

        all_features = []

        for system in systems:
            coords, boxes, atom_types = _load_npy_system(system)
            n_frames = coords.shape[0]
            n_atoms = len(atom_types)

            # Remap local atom-type indices to checkpoint-global indices.
            atom_types_global = self._remap_atom_types(atom_types, system)

            # Non-periodic structures must NOT use all-zero box:
            # the descriptor produces NaN in that case.
            # Use a large 100 Å cubic box instead.
            if boxes is None:
                boxes = np.tile(np.eye(3) * 100.0, (n_frames, 1)).reshape(n_frames, 9)

            # coord requires grad: forward_common calls autograd.grad
            # internally to compute forces, which fails under no_grad.
            coord_t = torch.tensor(
                coords.reshape(n_frames, n_atoms * 3), dtype=torch.float64,
                device=self._device,
            ).requires_grad_(True)
            atype_t = torch.tensor(
                np.tile(atom_types_global, (n_frames, 1)), dtype=torch.long,
                device=self._device,
            )
            box_t = torch.tensor(boxes, dtype=torch.float64, device=self._device)

            # Shape: (n_frames, n_atoms, feat_dim)
            descrpt = extractor._run_forward(coord_t, atype_t, box_t)
            if self.pooling == "mean":
                feat = descrpt.mean(dim=1)
            elif self.pooling == "sum":
                feat = descrpt.sum(dim=1)
            elif self.pooling == "mean+std":
                mean = descrpt.mean(dim=1)
                std = torch.nan_to_num(descrpt.std(dim=1), nan=0.0)
                feat = torch.cat([mean, std], dim=-1)
            elif self.pooling == "mean+std+max+min":
                mean = descrpt.mean(dim=1)
                std = torch.nan_to_num(descrpt.std(dim=1), nan=0.0)
                feat = torch.cat([
                    mean, std,
                    descrpt.max(dim=1).values, descrpt.min(dim=1).values,
                ], dim=-1)
            feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            all_features.append(feat.cpu().numpy())

        extractor._disable_hook()
        return np.concatenate(all_features, axis=0)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Type-map auto-inference (shared with MFTFineTuner via data/type_map.py)
    # -------------------------------------------------------------------

    def _resolve_type_maps(self, train_data) -> list[str]:
        """Auto-infer the global type_map from the checkpoint and validate
        *train_data* element set is a subset.

        Returns the checkpoint's type_map (e.g. 118-element full periodic
        table for DPA-3.1-3M).  For scratch (``pretrained=None``) there is no
        checkpoint — the type_map is the union of data ``atom_names``.
        """
        from deepmd.dpa_tools.data.type_map import (
            read_checkpoint_type_map,
            read_data_type_map_union,
            validate_type_map_subset,
        )

        try:
            systems = load_data(train_data)
        except DPADataError:
            # Data paths may not exist during testing; fall back gracefully.
            if self.pretrained is None:
                raise ValueError(
                    "strategy='scratch' requires valid data paths or "
                    "pass type_map=[...] explicitly."
                )
            return read_checkpoint_type_map(
                self.pretrained, branch=self.init_branch,
            )

        if self.pretrained is None:
            try:
                tm = read_data_type_map_union(systems)
            except ValueError:
                raise ValueError(
                    "strategy='scratch' requires atom_names in data "
                    "systems, or pass type_map=[...] explicitly. "
                    "Without a checkpoint, the global type_map cannot be "
                    "auto-inferred."
                )
            return tm

        tm = read_checkpoint_type_map(
            self.pretrained, branch=self.init_branch,
        )

        try:
            elements = read_data_type_map_union(systems)
            validate_type_map_subset(elements, tm, label="train data")
        except ValueError:
            pass  # no atom_names — deepmd uses raw atom indices

        return tm

    # -------------------------------------------------------------------
    # Training-paradigm fit (linear_probe / finetune / scratch)
    # -------------------------------------------------------------------

    def _fit_training(self, train_data, valid_data, type_map):
        """Delegate to DPATrainer for single-task ``dp --pt train``."""
        from deepmd.dpa_tools.trainer import DPATrainer

        freeze = self.strategy == "linear_probe"
        trainer = DPATrainer(
            pretrained=self.pretrained,
            init_branch=self.init_branch,
            freeze_backbone=freeze,
            property_name=self.property_name,
            task_dim=self.task_dim,
            intensive=self.intensive,
            train_systems=train_data,
            valid_systems=valid_data,
            type_map=type_map,
            learning_rate=self.learning_rate,
            stop_lr=self.stop_lr,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            loss_function=self.loss_function,
            seed=self.seed,
            output_dir=self.output_dir,
            save_freq=self.save_freq,
            disp_freq=self.disp_freq,
        )
        ckpt_path = trainer.fit()
        self._fitted = True
        return ckpt_path

    # -------------------------------------------------------------------
    # fit (dispatch)
    # -------------------------------------------------------------------

    def fit(
        self,
        train_data,
        valid_data=None,
        type_map=None,
        target_key=None,
        labels=None,
        fmt=None,
        conditions=None,
        aux_data=None,
    ):
        """Train the model.

        *frozen_sklearn* (default): extract descriptors, fit sklearn head.
        *linear_probe* / *finetune* / *scratch*: run ``dp --pt train``.
        *mft*: multi-task fine-tuning (property head + force-field head).

        Parameters
        ----------
        train_data : str | list[str]
            Path(s) to deepmd/npy system directories.
        valid_data : str | list[str], optional
            Validation system directories.  Required for training paradigms;
            ignored by ``frozen_sklearn``.
        type_map : list[str], optional
            Element symbols.  Auto-inferred from the checkpoint and data
            ``type_map.raw`` when not provided.
        target_key : str, optional
            (frozen_sklearn) Label key, e.g. ``"energy"``.
        labels : np.ndarray, optional
            (frozen_sklearn) Pre-computed labels.
        fmt : str, optional
            Reserved for future format support.
        conditions : dict[str, np.ndarray], optional
            (frozen_sklearn) Named condition arrays.
        aux_data : str | list[str], optional
            (mft only) Auxiliary training system directories.  Required when
            ``strategy='mft'``; must be absent otherwise.
        """
        if self.strategy == "frozen_sklearn":
            return self._fit_sklearn(train_data, type_map, target_key, labels, fmt,
                                     conditions)

        if self.strategy == "mft":
            if aux_data is None:
                raise ValueError(
                    "strategy='mft' requires aux_data. "
                    "Provide auxiliary system directories for the force-field head."
                )
            return self._fit_mft(train_data, aux_data, valid_data)

        # ---- single-task training paradigms ----
        if aux_data is not None:
            raise ValueError(
                f"aux_data is only valid when strategy='mft'; "
                f"got strategy={self.strategy!r}."
            )

        if type_map is None:
            type_map = self._resolve_type_maps(train_data)

        self.type_map = type_map
        return self._fit_training(train_data, valid_data, type_map)

    def _fit_mft(self, train_data, aux_data, valid_data=None):
        """Delegate to MFTFineTuner for multi-task fine-tuning."""
        from deepmd.dpa_tools.mft import MFTFineTuner

        mft = MFTFineTuner(
            pretrained=self.pretrained,
            aux_branch=self.aux_branch,
            aux_prob=self.aux_prob,
            aux_type_map=self.aux_type_map,
            downstream_type_map=self.downstream_type_map,
            fitting_net_params=self.fitting_net_params,
            downstream_task_type=self.downstream_task_type,
            property_name=self.property_name,
            task_dim=self.task_dim,
            intensive=self.intensive,
            learning_rate=self.learning_rate,
            stop_lr=self.stop_lr,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            aux_batch_size=self.aux_batch_size,
            downstream_batch_size=self.downstream_batch_size,
            seed=self.seed,
            output_dir=self.output_dir,
            save_freq=self.save_freq,
            disp_freq=self.disp_freq,
        )
        mft.fit(train_data=train_data, aux_data=aux_data, valid_data=valid_data)
        self._fitted = True
        return self.output_dir

    def _fit_sklearn(
        self,
        data,
        type_map=None,
        target_key=None,
        labels=None,
        fmt=None,
        conditions=None,
    ):
        """Original frozen_sklearn fit (unchanged logic)."""
        if target_key is not None and labels is not None:
            raise ValueError(
                "target_key and labels are mutually exclusive; provide only one."
            )
        if target_key is None and labels is None:
            raise ValueError("Either target_key or labels must be provided.")

        self.type_map    = type_map or []
        self._target_key = target_key if target_key is not None else "property"

        systems = load_data(data, fmt=fmt)
        if self._model is None:
            self._model = self._load_descriptor_model()
        self._validate_type_map(type_map or [], systems)

        features = self._extract_features_cached(systems)

        self._condition_manager = None
        if conditions is not None:
            self._condition_manager = ConditionManager()
            X_cond = self._condition_manager.fit_transform(conditions)
            features = np.concatenate([features, X_cond], axis=1)

        if labels is not None:
            y = np.asarray(labels)
        else:
            y = _load_labels(systems, self._target_key)

        self._task_dim = 1 if y.ndim == 1 else y.shape[-1]
        y_flat = y.ravel() if self._task_dim == 1 else y

        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from deepmd.dpa_tools.utils.sklearn_heads import build_sklearn_head

        head = build_sklearn_head(self._predictor_type, seed=self.seed)
        self.predictor = make_pipeline(StandardScaler(), head)
        self.predictor.fit(features, y_flat)
        self._fitted = True

    def predict(self, data, fmt=None, conditions=None) -> DotDict:
        """
        Extract features and run the fitted sklearn predictor.

        Parameters
        ----------
        data : str | list[str]
            Path(s) to deepmd/npy system directories.
        fmt : str, optional
            Reserved for future format support.
        conditions : dict[str, np.ndarray], optional
            Named condition arrays.  Required when the model was fit with
            conditions; must be absent otherwise.

        Returns
        -------
        DotDict
            ``predictions`` : np.ndarray, shape (n_frames, task_dim)
        """
        if not self._fitted:
            raise RuntimeError(
                "predict() was called before fit(). "
                "Train the model with fit() first."
            )

        systems     = load_data(data, fmt=fmt)
        features    = self._extract_features(systems)

        if self._condition_manager is not None:
            if conditions is None:
                raise DPAConditionError(
                    "This model was fit with conditions. "
                    "Pass conditions= to predict()."
                )
            X_cond = self._condition_manager.transform(conditions)
            features = np.concatenate([features, X_cond], axis=1)
        elif conditions is not None:
            raise DPAConditionError(
                "This model was fit without conditions."
            )

        raw         = self.predictor.predict(features)
        predictions = np.asarray(raw).reshape(-1, self._task_dim)
        return DotDict({"predictions": predictions})

    def evaluate(self, data, fmt=None, conditions=None) -> DotDict:
        """
        Predict on ``data`` and compute evaluation metrics against stored labels.

        Parameters
        ----------
        data : str | list[str]
            Path(s) to deepmd/npy system directories with label files.
        fmt : str, optional
            Reserved for future format support.
        conditions : dict[str, np.ndarray], optional
            Named condition arrays.  Required when the model was fit with
            conditions; must be absent otherwise.

        Returns
        -------
        DotDict
            mae, rmse, r2 : float
            predictions   : np.ndarray, shape (n_frames, task_dim)
            labels        : np.ndarray, shape (n_frames, task_dim)
        """
        result      = self.predict(data, fmt=fmt, conditions=conditions)
        predictions = result.predictions

        systems = load_data(data, fmt=fmt)
        labels  = _load_labels(systems, self._target_key)
        labels  = labels.reshape(predictions.shape)

        if predictions.shape != labels.shape:
            raise DPADataError(
                f"Shape mismatch: predictions {predictions.shape} vs "
                f"labels {labels.shape}."
            )

        err    = predictions - labels
        mae    = float(np.mean(np.abs(err)))
        rmse   = float(np.sqrt(np.mean(err ** 2)))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((labels - labels.mean()) ** 2)
        r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        return DotDict({
            "mae":         mae,
            "rmse":        rmse,
            "r2":          r2,
            "predictions": predictions,
            "labels":      labels,
        })

    def freeze(self, output_path="frozen_model.pth") -> str:
        """
        Serialize the fitted model bundle to a single file via ``torch.save``.

        The bundle contains the sklearn predictor object, the DPA checkpoint
        path, and metadata needed to reconstruct predictions.

        Parameters
        ----------
        output_path : str
            Destination file path.

        Returns
        -------
        str
            The resolved ``output_path``.
        """
        if not self._fitted:
            raise RuntimeError(
                "freeze() was called before fit(). "
                "Train the model with fit() first."
            )

        bundle = {
            "format_version":  1,
            "pretrained":      self.pretrained,
            "model_branch":    self.model_branch,
            "predictor":       self.predictor,
            "target_key":      self._target_key,
            "type_map":        self.type_map,
            "task_dim":        self._task_dim,
            "predictor_type":  self._predictor_type,
            "pooling":         self.pooling,
            "condition_manager": self._condition_manager,
        }

        output_path = str(output_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        import torch
        torch.save(bundle, output_path)
        print(f"Frozen model saved to: {output_path}")
        return output_path
