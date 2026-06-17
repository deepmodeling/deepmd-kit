# SPDX-License-Identifier: LGPL-3.0-or-later
# dpa_adapt/finetuner.py
#
# frozen_sklearn architecture: frozen DPA descriptor → sklearn predictor
# DPA checkpoint is used purely as a feature extractor (no dp train).

import logging
import os
import re
import subprocess
from pathlib import (
    Path,
)

import dpdata
import numpy as np

from dpa_adapt._backend import (
    _DescriptorExtraction,
    build_model_from_config,
    get_torch_device,
    load_torch_file,
    resolve_model_branch,
    resolve_pretrained_path,
)
from dpa_adapt.conditions import (
    ConditionManager,
    DPAConditionError,
)
from dpa_adapt.data.errors import (
    DPADataError,
)
from dpa_adapt.data.loader import (
    _get_source,
    _resolve_label_key,
    load_data,
)
from dpa_adapt.utils.dotdict import (
    DotDict,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _load_labels(
    systems: list[dpdata.System],
    target_key,  # str | list[str] — union type omitted for runtime simplicity
) -> np.ndarray:
    """Load and concatenate labels from dpdata systems.

    *target_key* may be a single string (existing behaviour) or a list of
    strings (new: multi-property).  When a list is given each key is loaded
    independently and the results are stacked column-wise into a 2-D array
    of shape ``(n_frames, len(target_key))``.

    Each key is resolved through ``_LABEL_KEY_ALIASES`` so that
    ``"energy"`` → ``"energies"`` for backward compatibility.

    When a resolved key is not present in ``system.data`` (dpdata only
    loads standard DeepMD keys), this function falls back to reading
    ``set.*/{key}.npy`` directly from the system source directory.
    """
    keys = [target_key] if isinstance(target_key, str) else list(target_key)
    columns = []

    for key in keys:
        resolved = _resolve_label_key(key)
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
                available_npy = sorted(
                    set(p.name for sd in set_dirs for p in sd.glob("*.npy"))
                )
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
            msg += f" (target_key={key!r})."
            raise DPADataError(msg)

        columns.append(np.concatenate(all_labels, axis=0))

    if len(columns) == 1:
        return columns[0]
    return np.column_stack(columns)


def _read_fparam_from_systems(
    systems: list[dpdata.System],
) -> dict[str, np.ndarray] | None:
    """Auto-read fparam.npy from each system's ``set.*/`` directories.

    Returns a dict mapping ``"fparam_0"``, ``"fparam_1"``, ... to 1-D
    arrays of length ``n_frames_total``, suitable for passing as
    ``conditions=`` to :meth:`ConditionManager.fit_transform`.

    Returns ``None`` when no system has a ``set.*/fparam.npy`` file.
    """
    all_fparams = []
    for system in systems:
        source = _get_source(system)
        if source is None:
            continue
        fps = sorted(Path(source).glob("set.*/fparam.npy"))
        if not fps:
            continue
        arrs = [np.load(str(fp)) for fp in fps]
        all_fparams.append(np.concatenate(arrs, axis=0))
    if not all_fparams:
        return None
    combined = np.concatenate(all_fparams, axis=0)  # (n_frames, fparam_dim)
    return {f"fparam_{i}": combined[:, i] for i in range(combined.shape[1])}


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
    coords = np.asarray(d["coords"])  # (n_frames, n_atoms, 3)
    n_atoms = coords.shape[1]
    coords = coords.reshape(coords.shape[0], n_atoms * 3)

    cells = np.asarray(d["cells"])  # (n_frames, 3, 3)
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
    from dpa_adapt.data.desc_cache import (
        load_or_extract,
    )

    systems = load_data(data)
    return load_or_extract(
        systems=systems,
        pretrained=pretrained,
        model_branch=model_branch,
        pooling=pooling,
        cache=cache,
    )


# ---------------------------------------------------------------------------
# Internal: frozen-sklearn pipeline (extracted from DPAFineTuner)
#
# Refactored: all descriptor-loading, feature-extraction, and sklearn-fitting
# logic moved into this helper so DPAFineTuner is a thin dispatcher.
# ---------------------------------------------------------------------------


class _FrozenSklearnPipeline:
    """Internal helper: frozen DPA descriptor → sklearn predictor pipeline.

    Encapsulates descriptor model loading, feature extraction (with
    caching), type-map validation / remapping, and sklearn fitting /
    prediction / evaluation / freeze.  DPAFineTuner holds one of these
    when ``strategy='frozen_sklearn'`` and delegates public API calls to it.

    Refactored: extracted from ``DPAFineTuner`` to separate the sklearn
    code path from the training-paradigm and MFT dispatch logic.
    """

    _VALID_POOLING = {"mean", "sum", "mean+std", "mean+std+max+min"}

    def __init__(self, pretrained, model_branch, predictor_type, pooling, seed):
        self.pretrained = pretrained
        self.model_branch = model_branch
        self._predictor_type = predictor_type
        self.pooling = pooling
        self.seed = seed

        # Populated during fit / extraction
        self._model = None
        self._device = None
        self._checkpoint_type_map = []
        self.predictor = None
        self._task_dim = 1
        self._target_key = None
        self._condition_manager = None
        self._fitted = False
        self.type_map = []

    # ------------------------------------------------------------------
    # Descriptor model loading
    # ------------------------------------------------------------------

    def load_descriptor_model(self):
        """Load the pretrained DPA checkpoint and return a (non-JIT) ModelWrapper.

        If *pretrained* is a built-in model name (e.g. ``"DPA-3.1-3M"``)
        rather than a local path, it is automatically downloaded.
        """
        resolved = resolve_pretrained_path(self.pretrained)
        state_dict = load_torch_file(resolved)
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
                f"Branch '{head}' not found. Available: {list(model_alias_dict)}"
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

    # ------------------------------------------------------------------
    # Type-map helpers
    # ------------------------------------------------------------------

    def validate_type_map(self, user_type_map, systems):
        """Raise DPADataError if any data element is not in the checkpoint type_map.

        The data type_map can be any subset of the checkpoint's type_map — order
        and contiguity are irrelevant. Local indices are remapped to checkpoint
        global indices in ``extract_features``.
        """
        ckpt = self._checkpoint_type_map
        if not ckpt:
            return  # checkpoint has no type_map metadata → skip

        ckpt_set = set(ckpt)

        def _check(candidate, source):
            unsupported = [e for e in candidate if e not in ckpt_set]
            if unsupported:
                ckpt_repr = (
                    f"{ckpt[:3] + ['...'] + ckpt[-1:]} ({len(ckpt)} elements)"
                    if len(ckpt) > 8
                    else str(ckpt)
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

    def remap_atom_types(self, atom_types, system):
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
                [ckpt.index(elem) for elem in data_tm],
                dtype=np.int64,
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

    # ------------------------------------------------------------------
    # Feature extraction  (extract_features_cached is on DPAFineTuner
    # so that patches on DPAFineTuner._extract_features are honoured)
    # ------------------------------------------------------------------

    def extract_features(self, systems):
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
            self._model = self.load_descriptor_model()

        extractor = _DescriptorExtraction(self._model)
        extractor._enable_hook()

        all_features = []

        for system in systems:
            coords, boxes, atom_types = _load_npy_system(system)
            n_frames = coords.shape[0]
            n_atoms = len(atom_types)

            # Remap local atom-type indices to checkpoint-global indices.
            atom_types_global = self.remap_atom_types(atom_types, system)

            # Non-periodic structures must NOT use all-zero box:
            # the descriptor produces NaN in that case.
            # Use a large 100 Å cubic box instead.
            if boxes is None:
                boxes = np.tile(np.eye(3) * 100.0, (n_frames, 1)).reshape(n_frames, 9)

            # coord requires grad: forward_common calls autograd.grad
            # internally to compute forces, which fails under no_grad.
            coord_t = torch.tensor(
                coords.reshape(n_frames, n_atoms * 3),
                dtype=torch.float64,
                device=self._device,
            ).requires_grad_(True)
            atype_t = torch.tensor(
                np.tile(atom_types_global, (n_frames, 1)),
                dtype=torch.long,
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
                feat = torch.cat(
                    [
                        mean,
                        std,
                        descrpt.max(dim=1).values,
                        descrpt.min(dim=1).values,
                    ],
                    dim=-1,
                )
            feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            all_features.append(feat.cpu().numpy())

        extractor._disable_hook()
        return np.concatenate(all_features, axis=0)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DPAFineTuner:
    """Adapt a pretrained DPA model to a downstream property via transfer learning.

    Four strategies, selected by *strategy*.

    ====================  =====================================================
    ``frozen_sklearn``    (default, CPU) Freeze the DPA backbone, extract
                          descriptors once, pool, and fit a scikit-learn
                          regressor (Ridge, KRR, or MLP).  No GPU needed;
                          fastest for small datasets.
    ``frozen_head``      Freeze the backbone, train only a neural property
                          fitting net via ``dp --pt train``.
    ``finetune``          Fine-tune the full network (descriptor + fitting
                          net) end-to-end via ``dp --pt train``.
    ``mft``               Multi-task fine-tuning: a downstream property head
                          is trained jointly with an auxiliary force/energy
                          head to regularise the representation.  Requires
                          *aux_data* at ``fit()`` time.
    ====================  =====================================================

    Parameters
    ----------
    pretrained : str
        Path to the pretrained DPA checkpoint (``.pt``), or a built-in name
        such as ``"DPA-3.1-3M"`` that could be auto-downloaded.
    model_branch : str or None
        Multi-task branch for descriptor extraction (e.g. ``"Domains_Drug"``).
    predictor : str
        (frozen_sklearn only) scikit-learn head: ``"rf"``, ``"linear"`` /
        ``"ridge"``, or ``"mlp"``.
    pooling : str
        (frozen_sklearn only) Descriptor pooling: ``"mean"`` (default),
        ``"sum"``, ``"mean+std"``, or ``"mean+std+max+min"``.
    seed : int
        Random seed for the head or for full training.
    strategy : str
        ``"frozen_sklearn"`` (default), ``"frozen_head"``, ``"finetune"``,
        or ``"mft"``.

    property_name : str
        Label key written under ``set.*/`` (e.g. ``"bandgap"``).  Used by
        all non-``frozen_sklearn`` strategies, and by ``frozen_sklearn``
        when *target_key* is not passed explicitly to ``fit()``.
    task_dim : int
        Output dimensionality of the property fitting net.
    intensive : bool
        If True (default), the property is intensive and frame-averaged;
        if False it is extensive (summed).
    init_branch : str
        Checkpoint branch used to initialise the descriptor (LP / FT only).
    learning_rate, stop_lr : float
        Start and end points of the exponential learning-rate schedule
        (training paradigms).
    decay_steps : int or None
        Steps between LR decays for the ``exp`` scheduler (deepmd-kit
        native).  ``None`` (default) auto-selects: 1000 for
        ``frozen_head``/``finetune``; 1000 for MFT property mode,
        5000 for MFT ener mode.
    warmup_steps : int
        Linear LR warmup steps (deepmd-kit native).  0 = disabled.
    max_steps : int
        Total training steps (LP / FT / MFT).
    batch_size : str or int
        DeepMD-kit batch-size spec (e.g. ``"auto:512"`` or 128).
    loss_function : str
        ``"mse"`` or ``"smooth_mae"`` (training paradigms).
    fitting_net_params : dict or None
        Extra kwargs merged into the fitting-net config (e.g.
        ``{"neuron": [128, 128]}``).  Applies to ``frozen_head``,
        ``finetune``, and ``mft`` strategies.
    fparam_dim : int
        Dimension of per-frame context features (e.g. temperature,
        humidity).  When > 0, ``set.*/fparam.npy`` of shape
        ``(n_frames, fparam_dim)`` is read automatically for all
        strategies.  For ``frozen_sklearn``, fparam columns are
        standardized and concatenated to the descriptor via
        ``ConditionManager``.  Default 0 (disabled).
    output_dir : str
        Directory for ``input.json``, checkpoints, and logs.
    save_freq, disp_freq : int
        Checkpoint save and log-display intervals (steps).

    aux_branch : str
        (MFT only) Pre-trained branch for the auxiliary force/energy head.
    aux_prob : float
        (MFT only) Probability of sampling an auxiliary batch at each step.
    type_map : list[str] or None
        (MFT only) The global (shared) type map. Both branches share a single
        descriptor, so this must be the union of elements in both datasets.
        Auto-detected from the checkpoint if not provided.
    downstream_task_type : str
        (MFT only) Task type of the downstream head (``"property"`` etc.).
    aux_batch_size : str or None
        (MFT only) Batch-size spec for the auxiliary head.
    downstream_batch_size : int or None
        (MFT only) Batch size for the downstream head.
    """

    _VALID_POOLING = {"mean", "sum", "mean+std", "mean+std+max+min"}
    _VALID_STRATEGIES = {
        "frozen_sklearn",
        "frozen_head",
        "finetune",
        "mft",
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
        decay_steps: int | None = None,  # None → auto: 1000 for training, MFT auto-detect
        warmup_steps: int = 0,
        max_steps=100_000,
        batch_size="auto:512",
        loss_function="mse",
        fitting_net_params: dict | None = None,
        fparam_dim: int = 0,
        output_dir="./dpa_output",
        save_freq=10_000,
        disp_freq=1_000,
        # ---- mft-only ----
        aux_branch="MP_traj_v024_alldata_mixu",
        aux_prob: float = 0.5,
        type_map: list[str] | None = None,
        downstream_task_type: str = "property",
        aux_batch_size: str | None = None,
        downstream_batch_size: int | None = None,
    ):
        if pooling not in self._VALID_POOLING:
            raise ValueError(
                f"pooling must be one of {sorted(self._VALID_POOLING)}, got {pooling!r}"
            )
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(self._VALID_STRATEGIES)}; "
                f"got {strategy!r}"
            )

        self.strategy = strategy

        self.pretrained = pretrained
        self.model_branch = model_branch
        self._predictor_type = predictor
        self.pooling = pooling
        self.seed = seed

        # Training-paradigm params (unused by frozen_sklearn).
        self.property_name = property_name
        self.task_dim = task_dim
        self.intensive = intensive
        self.init_branch = init_branch
        self.learning_rate = learning_rate
        self.stop_lr = stop_lr
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.fitting_net_params = fitting_net_params
        self.fparam_dim = fparam_dim
        self.output_dir = output_dir
        self.save_freq = save_freq
        self.disp_freq = disp_freq

        # MFT-only parameters.
        self.aux_branch = aux_branch
        self.aux_prob = aux_prob
        self.type_map = type_map
        self.downstream_task_type = downstream_task_type
        self.aux_batch_size = aux_batch_size
        self.downstream_batch_size = downstream_batch_size

        if strategy == "mft":
            if not isinstance(property_name, str) or not property_name.isidentifier():
                raise ValueError(
                    "property_name is required when strategy='mft' and must be a "
                    f"valid Python identifier; got {property_name!r}."
                )

        # ---- frozen_sklearn pipeline (created lazily by fit()) ----
        self._sklearn: _FrozenSklearnPipeline | None = None
        self._mft = None

        # ---- backward-compat state mirrors (delegated to pipeline) ----
        if self.type_map is None:
            self.type_map = []
        self._target_key = None
        self._task_dim = 1
        self.predictor = None  # sklearn object after fit()
        self._fitted = False
        self._model = None  # lazy-loaded descriptor model (cached)
        self._device = None  # set when model is first loaded
        self._checkpoint_type_map = []  # set by _load_descriptor_model
        self._condition_manager = None

    # ------------------------------------------------------------------
    # Frozen-sklearn pipeline helpers (thin delegators)
    #
    # Each method forwards to the corresponding method on
    # ``_FrozenSklearnPipeline``.  State set directly on DPAFineTuner
    # (e.g. ``_checkpoint_type_map`` by tests) is propagated into the
    # pipeline on each call so that direct setters continue to work.
    # ------------------------------------------------------------------

    def _ensure_sklearn(self):
        """Create the pipeline on first use if it doesn't exist yet."""
        if self._sklearn is None:
            self._sklearn = _FrozenSklearnPipeline(
                pretrained=self.pretrained,
                model_branch=self.model_branch,
                predictor_type=self._predictor_type,
                pooling=self.pooling,
                seed=self.seed,
            )
        # Sync state that external code may have set on DPAFineTuner directly.
        self._sklearn._model = self._model
        self._sklearn._device = self._device
        self._sklearn._checkpoint_type_map = self._checkpoint_type_map
        self._sklearn.type_map = self.type_map
        return self._sklearn

    def _load_descriptor_model(self):
        return self._ensure_sklearn().load_descriptor_model()

    def _validate_type_map(self, user_type_map, systems):
        return self._ensure_sklearn().validate_type_map(user_type_map, systems)

    def _remap_atom_types(self, atom_types, system):
        return self._ensure_sklearn().remap_atom_types(atom_types, system)

    def _extract_features_cached(self, systems):
        """Call ``_extract_features`` with descriptor-cache lookup.

        Kept on DPAFineTuner (not delegated) so that patches on
        ``DPAFineTuner._extract_features`` are honoured through the
        ``self._extract_features()`` call below.
        """
        try:
            from dpa_adapt.data.desc_cache import (
                _cache_dir,
                _cache_key,
            )

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

    def _extract_features(self, systems):
        return self._ensure_sklearn().extract_features(systems)

    # ------------------------------------------------------------------
    # The heavy implementations of the following methods now live in
    # _FrozenSklearnPipeline (see class docstring above).  The thin
    # delegators at the top of this class forward calls to the pipeline.
    # ------------------------------------------------------------------

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
        table for DPA-3.1-3M).
        """
        from dpa_adapt.data.type_map import (
            read_checkpoint_type_map,
            read_data_type_map_union,
            validate_type_map_subset,
        )

        try:
            systems = load_data(train_data)
        except DPADataError:
            # Data paths may not exist during testing; fall back gracefully.
            return read_checkpoint_type_map(
                self.pretrained,
                branch=self.init_branch,
            )

        tm = read_checkpoint_type_map(
            self.pretrained,
            branch=self.init_branch,
        )

        try:
            elements = read_data_type_map_union(systems)
            validate_type_map_subset(elements, tm, label="train data")
        except ValueError:
            pass  # no atom_names — deepmd uses raw atom indices

        return tm

    # -------------------------------------------------------------------
    # Training-paradigm fit (frozen_head / finetune)
    # -------------------------------------------------------------------

    def _fit_training(self, train_data, valid_data, type_map):
        """Delegate to DPATrainer for single-task ``dp --pt train``."""
        from dpa_adapt.trainer import (
            DPATrainer,
        )

        freeze = self.strategy == "frozen_head"
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
            fitting_net_params=self.fitting_net_params,
            learning_rate=self.learning_rate,
            stop_lr=self.stop_lr,
            decay_steps=self.decay_steps if self.decay_steps is not None else 1000,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            loss_function=self.loss_function,
            fparam_dim=self.fparam_dim,
            seed=self.seed,
            output_dir=self.output_dir,
            save_freq=self.save_freq,
            disp_freq=self.disp_freq,
        )
        ckpt_path = trainer.fit()
        self._fitted = True
        return ckpt_path

    def _latest_training_checkpoint(self) -> str:
        ckpts = list(Path(self.output_dir).glob("model.ckpt-*.pt"))
        if not ckpts:
            raise RuntimeError(
                f"No model.ckpt-*.pt found in {self.output_dir}; call fit() first."
            )

        def step_of(path):
            return int(path.stem.split("-")[-1])

        return str(max(ckpts, key=step_of))

    @staticmethod
    def _expand_system_specs(data) -> list[str]:
        import glob

        patterns = [data] if isinstance(data, str) else list(data)
        systems = []
        for pattern in patterns:
            matches = sorted(glob.glob(str(pattern)))
            systems.extend(matches or [str(pattern)])

        seen = set()
        systems = [s for s in systems if not (s in seen or seen.add(s))]
        if not systems:
            raise DPADataError(f"No systems matched {data!r}.")
        return systems

    def _run_training_predict(self, data, fmt=None) -> DotDict:
        """Run ``dp --pt test`` and parse property predictions from detail files."""
        from dpa_adapt.trainer import (
            DPATrainer,
        )

        if fmt is not None:
            raise ValueError(
                "fmt is not supported for frozen_head/finetune predict(); "
                "provide deepmd/npy system directories."
            )

        ckpt = self._latest_training_checkpoint()
        systems = self._expand_system_specs(data)

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        datafile = output_dir / "predict_systems.txt"
        datafile.write_text("\n".join(systems) + "\n")

        detail_prefix = output_dir / "predict_detail"
        for old in output_dir.glob(f"{detail_prefix.name}.property.out.*"):
            old.unlink()

        cmd = [
            "dp",
            "--pt",
            "test",
            "-m",
            ckpt,
            "-f",
            str(datafile),
            "-n",
            "999999",
            "-d",
            str(detail_prefix),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        combined = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            raise RuntimeError(
                f"dp --pt test failed (return code {result.returncode}).\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        detail_files = sorted(
            output_dir.glob(f"{detail_prefix.name}.property.out.*"),
            key=lambda p: int(p.name.rsplit(".", 1)[-1]),
        )
        if not detail_files:
            raise RuntimeError(
                "dp --pt test completed but no property detail files were written. "
                f"Command was: {' '.join(cmd)}"
            )

        rows = []
        for path in detail_files:
            arr = np.loadtxt(path)
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] < 2:
                raise RuntimeError(
                    f"Expected at least two columns in {path}, got shape {arr.shape}."
                )
            rows.append(arr[:, :2])

        values = np.concatenate(rows, axis=0)
        if values.shape[0] % self.task_dim != 0:
            raise RuntimeError(
                f"Could not reshape property detail rows {values.shape[0]} "
                f"into task_dim={self.task_dim}."
            )

        values = values.reshape(-1, self.task_dim, 2)
        labels = values[:, :, 0]
        predictions = values[:, :, 1]
        if self.task_dim == 1:
            labels = labels.reshape(-1, 1)
            predictions = predictions.reshape(-1, 1)

        metrics = DPATrainer._parse_test_output(combined)
        n_sys_match = re.search(
            r"number of systems\s*[:=]?\s*(\d+)", combined, re.IGNORECASE
        )
        n_systems = int(n_sys_match.group(1)) if n_sys_match else len(systems)
        return DotDict(
            {
                "predictions": predictions,
                "labels": labels,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "n_frames": metrics["n_frames"],
                "n_systems": n_systems,
                "detail_prefix": str(detail_prefix),
                "_raw_stdout": combined,
            }
        )

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
        aux_data=None,
    ):
        """Train the model.

        *frozen_sklearn* (default): extract descriptors, fit sklearn head.
        *frozen_head* / *finetune*: run ``dp --pt train``.
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
        aux_data : str | list[str], optional
            (mft only) Auxiliary training system directories.  Required when
            ``strategy='mft'``; must be absent otherwise.
        """
        if self.strategy == "frozen_sklearn":
            return self._fit_sklearn(
                train_data, type_map, target_key, labels, fmt
            )

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
        mft = self._ensure_mft()
        mft.fit(train_data=train_data, aux_data=aux_data, valid_data=valid_data)
        self._fitted = True
        return self.output_dir

    def _ensure_mft(self):
        """Create the MFT delegate on first use."""
        from dpa_adapt.mft import (
            MFTFineTuner,
        )

        if self._mft is None:
            self._mft = MFTFineTuner(
                pretrained=self.pretrained,
                aux_branch=self.aux_branch,
                aux_prob=self.aux_prob,
                type_map=self.type_map,
                fitting_net_params=self.fitting_net_params,
                downstream_task_type=self.downstream_task_type,
                property_name=self.property_name,
                task_dim=self.task_dim,
                intensive=self.intensive,
                learning_rate=self.learning_rate,
                stop_lr=self.stop_lr,
                decay_steps=self.decay_steps,
                warmup_steps=self.warmup_steps,
                max_steps=self.max_steps,
                batch_size=self.batch_size,
                aux_batch_size=self.aux_batch_size,
                downstream_batch_size=self.downstream_batch_size,
                seed=self.seed,
                fparam_dim=self.fparam_dim,
                output_dir=self.output_dir,
                save_freq=self.save_freq,
                disp_freq=self.disp_freq,
            )
        return self._mft

    def _fit_sklearn(
        self,
        data,
        type_map=None,
        target_key=None,
        labels=None,
        fmt=None,
    ):
        """Fit the frozen-sklearn pipeline (delegates to ``_FrozenSklearnPipeline``).

        Refactored: logic extracted to ``_FrozenSklearnPipeline``; this method
        now orchestrates the pipeline and mirrors its state for backward compat.
        """
        if target_key is not None and labels is not None:
            raise ValueError(
                "target_key and labels are mutually exclusive; provide only one."
            )
        if target_key is None and labels is None:
            raise ValueError("Either target_key or labels must be provided.")

        p = self._ensure_sklearn()

        self.type_map = type_map or []
        self._target_key = target_key if target_key is not None else "property"

        systems = load_data(data, fmt=fmt)
        if self._model is None:
            self._model = self._load_descriptor_model()
        self._validate_type_map(type_map or [], systems)

        features = self._extract_features_cached(systems)

        self._condition_manager = None
        if self.fparam_dim > 0:
            conditions = _read_fparam_from_systems(systems)
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

        from sklearn.pipeline import (
            make_pipeline,
        )
        from sklearn.preprocessing import (
            StandardScaler,
        )

        from dpa_adapt.utils.sklearn_heads import (
            build_sklearn_head,
        )

        head = build_sklearn_head(
            self._predictor_type,
            seed=self.seed,
            n_outputs=self._task_dim,
        )
        self.predictor = make_pipeline(StandardScaler(), head)
        self.predictor.fit(features, y_flat)
        self._fitted = True

        # Mirror pipeline state for backward compat.
        p.predictor = self.predictor
        p.type_map = self.type_map
        p._target_key = self._target_key
        p._task_dim = self._task_dim
        p._condition_manager = self._condition_manager
        p._fitted = True

    def predict(self, data, fmt=None) -> DotDict:
        """
        Predict with the adapted model.

        ``frozen_sklearn`` extracts features and runs the fitted sklearn
        predictor. Training strategies run ``dp --pt test`` and parse the
        property predictions from DeepMD's detail files.

        Parameters
        ----------
        data : str | list[str]
            Path(s) to deepmd/npy system directories.
        fmt : str, optional
            Reserved for future format support.

        Returns
        -------
        DotDict
            ``predictions`` : np.ndarray, shape (n_frames, task_dim)
        """
        if self.strategy in {"frozen_head", "finetune"}:
            return self._run_training_predict(data, fmt=fmt)
        if self.strategy == "mft":
            if fmt is not None:
                raise ValueError(
                    "fmt is not supported for mft predict(); "
                    "provide deepmd/npy system directories."
                )
            return self._ensure_mft().predict(data)

        if not self._fitted:
            raise RuntimeError(
                "predict() was called before fit(). Train the model with fit() first."
            )

        systems = load_data(data, fmt=fmt)
        features = self._extract_features(systems)

        if self._condition_manager is not None:
            conditions = _read_fparam_from_systems(systems)
            if conditions is None:
                raise DPAConditionError(
                    "This model was fit with fparam but set.*/fparam.npy "
                    "was not found in the test data."
                )
            X_cond = self._condition_manager.transform(conditions)
            features = np.concatenate([features, X_cond], axis=1)

        raw = self.predictor.predict(features)
        predictions = np.asarray(raw).reshape(-1, self._task_dim)
        return DotDict({"predictions": predictions})

    def evaluate(self, data, fmt=None) -> DotDict:
        """
        Predict on ``data`` and compute evaluation metrics against stored labels.

        Parameters
        ----------
        data : str | list[str]
            Path(s) to deepmd/npy system directories with label files.
        fmt : str, optional
            Reserved for future format support.

        Returns
        -------
        DotDict
            mae, rmse, r2 : float
            predictions   : np.ndarray, shape (n_frames, task_dim)
            labels        : np.ndarray, shape (n_frames, task_dim)
        """
        if self.strategy in {"frozen_head", "finetune"}:
            result = self._run_training_predict(data, fmt=fmt)
            labels = result.labels
            predictions = result.predictions
            err = predictions - labels
            ss_res = np.sum(err**2)
            ss_tot = np.sum((labels - labels.mean()) ** 2)
            result["r2"] = (
                float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            )
            return result
        if self.strategy == "mft":
            if fmt is not None:
                raise ValueError(
                    "fmt is not supported for mft evaluate(); "
                    "provide deepmd/npy system directories."
                )
            result = self._ensure_mft().predict(data)
            labels = result.labels
            predictions = result.predictions
            err = predictions - labels
            ss_res = np.sum(err**2)
            ss_tot = np.sum((labels - labels.mean()) ** 2)
            result["r2"] = (
                float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            )
            return result

        result = self.predict(data, fmt=fmt)
        predictions = result.predictions

        systems = load_data(data, fmt=fmt)
        labels = _load_labels(systems, self._target_key)
        labels = labels.reshape(predictions.shape)

        if predictions.shape != labels.shape:
            raise DPADataError(
                f"Shape mismatch: predictions {predictions.shape} vs "
                f"labels {labels.shape}."
            )

        err = predictions - labels
        if isinstance(self._target_key, list):
            # Per-property metrics
            keys = self._target_key
            mae, rmse, r2 = {}, {}, {}
            for i, key in enumerate(keys):
                e_i = err[:, i]
                mae[key] = float(np.mean(np.abs(e_i)))
                rmse[key] = float(np.sqrt(np.mean(e_i**2)))
                ss_res_i = np.sum(e_i**2)
                ss_tot_i = np.sum((labels[:, i] - labels[:, i].mean()) ** 2)
                r2[key] = (
                    float(1.0 - ss_res_i / ss_tot_i) if ss_tot_i > 0 else float("nan")
                )
        else:
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            ss_res = np.sum(err**2)
            ss_tot = np.sum((labels - labels.mean()) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        return DotDict(
            {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "predictions": predictions,
                "labels": labels,
            }
        )

    def freeze(self, output_path="frozen_model.pth") -> str:
        """
        Serialize the fitted model bundle to a single file via ``torch.save``.

        The bundle contains the sklearn predictor object, the DPA checkpoint
        path, and metadata needed to reconstruct predictions.

        ``target_key`` is stored as-is (``str`` or ``list[str]``).  Loading a
        bundle with a ``list`` target_key requires dpa_adapt >= 0.2.

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
                "freeze() was called before fit(). Train the model with fit() first."
            )

        bundle = {
            "format_version": 1,
            "pretrained": self.pretrained,
            "model_branch": self.model_branch,
            "predictor": self.predictor,
            "target_key": self._target_key,
            "type_map": self.type_map,
            "task_dim": self._task_dim,
            "predictor_type": self._predictor_type,
            "pooling": self.pooling,
            "condition_manager": self._condition_manager,
        }

        output_path = str(output_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        import torch

        torch.save(bundle, output_path)
        _LOG = logging.getLogger("dpa_adapt")
        _LOG.info("Frozen model saved to: %s", output_path)
        return output_path
