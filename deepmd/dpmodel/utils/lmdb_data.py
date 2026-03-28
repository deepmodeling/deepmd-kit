# SPDX-License-Identifier: LGPL-3.0-or-later
"""Framework-agnostic LMDB data utilities for DeePMD-kit.

All code here is pure Python/NumPy/lmdb/msgpack — no framework dependency.
Backend-specific wrappers (PyTorch Dataset, JAX, etc.) import from here.
"""

import logging
import math
from collections.abc import (
    Iterator,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import lmdb
import msgpack
import numpy as np

from deepmd.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

log = logging.getLogger(__name__)

# LMDB key → DeePMD convention
_KEY_REMAP = {
    "coords": "coord",
    "cells": "box",
    "energies": "energy",
    "forces": "force",
    "atom_types": "atype",
    "virials": "virial",
}

# Keys whose high_prec is always True in the standard pipeline
# (energy is set by Loss DataRequirementItem; reduce() also sets high_prec=True)
_HIGH_PREC_KEYS = frozenset({"energy"})


def _open_lmdb(path: str) -> lmdb.Environment:
    """Open LMDB environment readonly."""
    return lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)


def _read_metadata(txn: lmdb.Transaction) -> dict:
    """Read and decode __metadata__ from LMDB transaction."""
    raw = txn.get(b"__metadata__")
    if raw is None:
        raise ValueError("LMDB file missing __metadata__ key")
    return msgpack.unpackb(raw, raw=False)


def _decode_array(obj: dict) -> np.ndarray:
    """Reconstruct ndarray from msgpack-encoded dict with {type, shape, data}.

    Handles both string keys ("type", "data") and byte keys (b"type", b"data").
    """
    dtype_key = "type" if "type" in obj else b"type"
    data_key = "data" if "data" in obj else b"data"
    shape_key = "shape" if "shape" in obj else b"shape"
    dtype = np.dtype(obj[dtype_key])
    data = obj[data_key]
    if shape_key in obj:
        shape = tuple(obj[shape_key])
    else:
        shape = (len(data) // dtype.itemsize,)
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def _is_encoded_array(val: Any) -> bool:
    """Check if a value is a msgpack-encoded ndarray dict."""
    if not isinstance(val, dict):
        return False
    return ("data" in val and "type" in val) or (b"data" in val and b"type" in val)


def _decode_value(val: Any) -> Any:
    """Decode a value: encoded array -> ndarray, list of encoded -> list of ndarray, else pass through."""
    if _is_encoded_array(val):
        return _decode_array(val)
    elif isinstance(val, list) and len(val) > 0 and _is_encoded_array(val[0]):
        return [_decode_array(item) for item in val]
    return val


def _decode_frame(raw_bytes: bytes) -> dict[str, Any]:
    """Decode a msgpack-serialized frame into a dict of numpy arrays / scalars."""
    frame = msgpack.unpackb(raw_bytes, raw=False)
    result = {}
    for key, val in frame.items():
        result[key] = _decode_value(val)
    return result


def _remap_keys(frame: dict[str, Any]) -> dict[str, Any]:
    """Remap LMDB key names to DeePMD convention, pass through unknown keys."""
    out = {}
    for k, v in frame.items():
        out[_KEY_REMAP.get(k, k)] = v
    return out


def is_lmdb(systems: Any) -> bool:
    """Check if systems points to an LMDB dataset."""
    if not isinstance(systems, str):
        return False
    return systems.endswith(".lmdb") or Path(systems, "data.mdb").exists()


def _parse_metadata(meta: dict) -> tuple[int, str, list[int]]:
    """Parse LMDB metadata into (nframes, frame_fmt, natoms_per_type).

    Handles system_info as list or dict, and natoms as plain ints or encoded arrays.
    """
    nframes = meta["nframes"]
    frame_fmt = meta.get("frame_idx_fmt", "012d")
    raw_sys_info = meta.get("system_info", {})

    if isinstance(raw_sys_info, list):
        sys_info = raw_sys_info[0] if raw_sys_info else {}
    else:
        sys_info = raw_sys_info

    raw_natoms = sys_info.get("natoms", [])
    natoms_per_type = []
    for item in raw_natoms:
        if _is_encoded_array(item):
            natoms_per_type.append(int(_decode_array(item).item()))
        else:
            natoms_per_type.append(int(item))

    return nframes, frame_fmt, natoms_per_type


def _scan_frame_nlocs(
    env: lmdb.Environment, nframes: int, frame_fmt: str, fallback_natoms: int
) -> list[int]:
    """Scan all frames to get per-frame atom count.

    Reads only the atom_types shape from msgpack without decoding array data.
    """
    nlocs = []
    with env.begin() as txn:
        for i in range(nframes):
            key = format(i, frame_fmt).encode()
            raw = txn.get(key)
            if raw is not None:
                frame_raw = msgpack.unpackb(raw, raw=False)
                atype_raw = frame_raw.get("atom_types")
                if isinstance(atype_raw, dict):
                    shape = atype_raw.get("shape") or atype_raw.get(b"shape")
                    if shape:
                        nlocs.append(int(shape[0]))
                        continue
            nlocs.append(fallback_natoms)
    return nlocs


def _compute_batch_size(nloc: int, rule: int) -> int:
    """Compute batch_size for a given nloc using the auto rule."""
    bsi = rule // max(nloc, 1)
    if bsi * nloc < rule:
        bsi += 1
    return max(bsi, 1)


class LmdbDataReader:
    """Framework-agnostic LMDB dataset reader.

    Reads LMDB frames and returns dicts of numpy arrays.
    Backend-specific Dataset classes (PyTorch, JAX, etc.) wrap this.

    Datasets are typically mixed-nloc (frames with different atom counts).
    The ``mixed_batch`` flag controls batching strategy:

    - ``mixed_batch=False`` (default, old format): each batch contains only
      frames with the same nloc. A ``SameNlocBatchSampler`` groups frames
      by nloc and yields same-nloc batches. Auto batch_size is computed
      per-nloc-group.
    - ``mixed_batch=True`` (new format): frames with different nloc can
      coexist in one batch (requires padding + mask in collate_fn).
      Currently raises ``NotImplementedError`` at collation time.

    Parameters
    ----------
    lmdb_path : str
        Path to the LMDB directory.
    type_map : list[str]
        Global type map from model config.
    batch_size : int or str
        Batch size. Supports int, "auto", "auto:N".
    mixed_batch : bool
        If True, allow different nloc in the same batch (future).
        If False (default), enforce same-nloc-per-batch.
    """

    def __init__(
        self,
        lmdb_path: str,
        type_map: list[str],
        batch_size: int | str = "auto",
        mixed_batch: bool = False,
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self._type_map = type_map
        self._env = _open_lmdb(self.lmdb_path)
        self.mixed_batch = mixed_batch

        with self._env.begin() as txn:
            meta = _read_metadata(txn)

        self.nframes, self._frame_fmt, self._natoms_per_type = _parse_metadata(meta)
        self._natoms = sum(self._natoms_per_type)
        self._ntypes = len(type_map)

        # Build type remapping if LMDB's type_map differs from model's type_map
        lmdb_type_map = meta.get("type_map")
        self._lmdb_type_map = lmdb_type_map
        self._type_remap: np.ndarray | None = None
        if lmdb_type_map is not None and list(lmdb_type_map) != list(type_map):
            # Build remap: lmdb_type_idx -> model_type_idx
            remap = np.empty(len(lmdb_type_map), dtype=np.int32)
            for i, name in enumerate(lmdb_type_map):
                if name not in type_map:
                    raise ValueError(
                        f"Element '{name}' in LMDB type_map {lmdb_type_map} "
                        f"not found in model type_map {type_map}"
                    )
                remap[i] = type_map.index(name)
            self._type_remap = remap
            log.info(
                f"Type remapping: LMDB {lmdb_type_map} -> model {type_map}, "
                f"remap={remap}"
            )

        # Persistent read-only transaction for __getitem__ (avoids per-read overhead).
        # Safe because we use num_workers=0 in DataLoader.
        self._txn = self._env.begin()

        # Scan per-frame nloc only when needed for same-nloc batching.
        # For mixed_batch=True, skip the scan entirely (future: padding handles it).
        if not mixed_batch:
            # Fast path: use pre-computed frame_nlocs from metadata if available.
            # Falls back to scanning each frame's atom_types shape (~10 us/frame).
            meta_nlocs = meta.get("frame_nlocs")
            if meta_nlocs is not None:
                self._frame_nlocs = [int(n) for n in meta_nlocs]
            else:
                self._frame_nlocs = _scan_frame_nlocs(
                    self._env, self.nframes, self._frame_fmt, self._natoms
                )
            self._nloc_groups: dict[int, list[int]] = {}
            for idx, nloc in enumerate(self._frame_nlocs):
                self._nloc_groups.setdefault(nloc, []).append(idx)
        else:
            self._frame_nlocs = []
            self._nloc_groups = {}

        # Parse frame_system_ids for auto_prob support
        meta_sys_ids = meta.get("frame_system_ids")
        if meta_sys_ids is not None:
            self._frame_system_ids: list[int] | None = [int(s) for s in meta_sys_ids]
            self._nsystems = max(self._frame_system_ids) + 1
            self._system_groups: dict[int, list[int]] = {}
            for idx, sid in enumerate(self._frame_system_ids):
                self._system_groups.setdefault(sid, []).append(idx)
            self._system_nframes: list[int] = [
                len(self._system_groups.get(i, [])) for i in range(self._nsystems)
            ]
        else:
            self._frame_system_ids = None
            self._nsystems = 1
            self._system_groups = {0: list(range(self.nframes))}
            self._system_nframes = [self.nframes]

        # Parse batch_size spec
        self._auto_rule: int | None = None
        if isinstance(batch_size, str):
            if batch_size == "auto":
                self._auto_rule = 32
            elif batch_size.startswith("auto:"):
                self._auto_rule = int(batch_size.split(":")[1])
            else:
                self._auto_rule = 32
            # Default batch_size uses first frame's nloc (for total_batch estimate)
            self.batch_size = _compute_batch_size(self._natoms, self._auto_rule)
        else:
            self.batch_size = int(batch_size)

        # Data requirements tracking
        self._data_requirements: dict[str, DataRequirementItem] = {}

    def _compute_natoms_vec(self, atype: np.ndarray) -> np.ndarray:
        """Compute natoms_vec from a frame's atype array.

        Returns [nloc, nloc, count_type0, count_type1, ...] with length ntypes+2.
        """
        nloc = len(atype)
        counts = np.bincount(atype, minlength=self._ntypes)[: self._ntypes]
        vec = np.empty(self._ntypes + 2, dtype=np.int64)
        vec[0] = nloc
        vec[1] = nloc
        vec[2:] = counts
        return vec

    def _resolve_dtype(self, key: str) -> np.dtype:
        """Resolve the target numpy dtype for a given key.

        Priority: DataRequirementItem.dtype > DataRequirementItem.high_prec >
        built-in defaults (energy=high, others=normal).
        """
        if key in self._data_requirements:
            req = self._data_requirements[key]
            # Support both DataRequirementItem objects and plain dicts
            if isinstance(req, dict):
                dtype = req.get("dtype")
                if dtype is not None:
                    return dtype
                if req.get("high_prec", False):
                    return GLOBAL_ENER_FLOAT_PRECISION
                return GLOBAL_NP_FLOAT_PRECISION
            else:
                # DataRequirementItem object
                if hasattr(req, "dtype") and req.dtype is not None:
                    return req.dtype
                if hasattr(req, "high_prec") and req.high_prec:
                    return GLOBAL_ENER_FLOAT_PRECISION
                return GLOBAL_NP_FLOAT_PRECISION
        # Fall back to built-in defaults
        if key in _HIGH_PREC_KEYS:
            return GLOBAL_ENER_FLOAT_PRECISION
        return GLOBAL_NP_FLOAT_PRECISION

    def get_batch_size_for_nloc(self, nloc: int) -> int:
        """Get batch_size for a given nloc. Uses auto rule if configured."""
        if self._auto_rule is not None:
            return _compute_batch_size(nloc, self._auto_rule)
        return self.batch_size

    def __len__(self) -> int:
        return self.nframes

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Read frame from LMDB, decode, remap keys, return dict of numpy arrays."""
        key = format(index, self._frame_fmt).encode()
        raw = self._txn.get(key)
        if raw is None:
            raise IndexError(f"Frame {index} not found in LMDB")
        frame = _decode_frame(raw)
        frame = _remap_keys(frame)

        # Remove LMDB-specific metadata keys not needed by trainer
        for meta_key in ("atom_numbs", "atom_names", "orig"):
            frame.pop(meta_key, None)

        # Flatten arrays to match DeePMD convention
        if "coord" in frame and isinstance(frame["coord"], np.ndarray):
            frame["coord"] = (
                frame["coord"].reshape(-1, 3).astype(self._resolve_dtype("coord"))
            )
        if "box" in frame and isinstance(frame["box"], np.ndarray):
            frame["box"] = frame["box"].reshape(9).astype(self._resolve_dtype("box"))
        if "energy" in frame:
            val = frame["energy"]
            if isinstance(val, np.ndarray):
                frame["energy"] = val.reshape(1).astype(self._resolve_dtype("energy"))
            else:
                frame["energy"] = np.array(
                    [float(val)], dtype=self._resolve_dtype("energy")
                )
        if "force" in frame and isinstance(frame["force"], np.ndarray):
            frame["force"] = (
                frame["force"].reshape(-1, 3).astype(self._resolve_dtype("force"))
            )
        if "atype" in frame and isinstance(frame["atype"], np.ndarray):
            frame["atype"] = frame["atype"].reshape(-1).astype(np.int64)
            # Remap atom types from LMDB's type_map to model's type_map
            if self._type_remap is not None:
                frame["atype"] = self._type_remap[frame["atype"]].astype(np.int64)
        if "virial" in frame and isinstance(frame["virial"], np.ndarray):
            frame["virial"] = (
                frame["virial"].reshape(9).astype(self._resolve_dtype("virial"))
            )

        # Per-frame natoms_vec from atype
        atype = frame.get("atype")
        if atype is not None:
            frame_natoms = len(atype)
            natoms_vec = self._compute_natoms_vec(atype)
            frame["natoms"] = natoms_vec
            frame["real_natoms_vec"] = natoms_vec
        else:
            frame_natoms = self._natoms
            fallback = np.array(
                [self._natoms, self._natoms] + [0] * self._ntypes, dtype=np.int64
            )
            frame["natoms"] = fallback
            frame["real_natoms_vec"] = fallback

        # Add find_* flags for known label keys
        label_keys = [
            "energy",
            "force",
            "virial",
            "atom_ener",
            "atom_pref",
            "drdq",
            "atom_ener_coeff",
            "hessian",
        ]
        for lk in label_keys:
            frame[f"find_{lk}"] = np.float32(1.0) if lk in frame else np.float32(0.0)

        # Handle registered data requirements: fill defaults for missing keys
        for req_key, req_item in self._data_requirements.items():
            if req_key not in frame:
                frame[f"find_{req_key}"] = np.float32(0.0)
                # Support both dict and DataRequirementItem object
                if isinstance(req_item, dict):
                    ndof = req_item["ndof"]
                    default = req_item["default"]
                    atomic = req_item["atomic"]
                    req_dtype = req_item.get("dtype")
                    if req_dtype is None:
                        req_dtype = (
                            GLOBAL_ENER_FLOAT_PRECISION
                            if req_item.get("high_prec", False)
                            else GLOBAL_NP_FLOAT_PRECISION
                        )
                else:
                    ndof = req_item.ndof
                    default = req_item.default
                    atomic = req_item.atomic
                    req_dtype = req_item.dtype
                    if req_dtype is None:
                        req_dtype = (
                            GLOBAL_ENER_FLOAT_PRECISION
                            if req_item.high_prec
                            else GLOBAL_NP_FLOAT_PRECISION
                        )
                if atomic:
                    shape = (frame_natoms, ndof)
                else:
                    shape = (ndof,)
                frame[req_key] = np.full(shape, default, dtype=req_dtype)
            elif f"find_{req_key}" not in frame:
                frame[f"find_{req_key}"] = np.float32(1.0)

        # Add find_* for fparam/aparam/spin if not already set
        for extra_key in ["fparam", "aparam", "spin"]:
            if f"find_{extra_key}" not in frame:
                frame[f"find_{extra_key}"] = (
                    np.float32(1.0) if extra_key in frame else np.float32(0.0)
                )

        frame["fid"] = index

        return frame

    # --- Data requirement interface ---

    def add_data_requirement(self, data_requirement: list[DataRequirementItem]) -> None:
        """Register expected keys; missing keys get default fill + find_key=0.0."""
        for item in data_requirement:
            self._data_requirements[item["key"]] = item

    def print_summary(self, name: str, prob: Any) -> None:
        """Print basic dataset info."""
        n_groups = len(self._nloc_groups)

        log.info(
            f"LMDB {name}: {self.lmdb_path}, "
            f"{self.nframes} frames, {n_groups} nloc groups, "
            f"batch_size={'auto' if self._auto_rule else self.batch_size}, "
            f"mixed_batch={self.mixed_batch}"
        )
        # Print nloc groups in rows of ~10 for readability
        items = [
            f"{nloc}({len(idxs)})" for nloc, idxs in sorted(self._nloc_groups.items())
        ]
        per_row = 10
        for i in range(0, len(items), per_row):
            row = ", ".join(items[i : i + per_row])
            log.info(f"  nloc groups: {row}")

    def set_noise(self, noise_settings: dict[str, Any]) -> None:
        """No-op for now."""

    # --- Properties ---

    @property
    def index(self) -> list[int]:
        """Number of batches per system (single system)."""
        return [max(1, self.nframes // self.batch_size)]

    @property
    def total_batch(self) -> int:
        return self.index[0]

    @property
    def batch_sizes(self) -> list[int]:
        return [self.batch_size]

    @property
    def mixed_type(self) -> bool:
        """LMDB datasets are always mixed_type (frames may have different compositions)."""
        return True

    @property
    def nloc_groups(self) -> dict[int, list[int]]:
        """Nloc → list of frame indices."""
        return self._nloc_groups

    @property
    def frame_nlocs(self) -> list[int]:
        """Per-frame atom count."""
        return self._frame_nlocs

    @property
    def nsystems(self) -> int:
        """Number of original systems merged into this LMDB."""
        return self._nsystems

    @property
    def frame_system_ids(self) -> list[int] | None:
        """Per-frame system index, or None if not available."""
        return self._frame_system_ids

    @property
    def system_groups(self) -> dict[int, list[int]]:
        """System index → list of frame indices."""
        return self._system_groups

    @property
    def system_nframes(self) -> list[int]:
        """Number of frames per system."""
        return self._system_nframes


def compute_block_targets(
    auto_prob_style: str,
    nsystems: int,
    system_nframes: list[int],
) -> list[tuple[list[int], int]]:
    """Compute target frame count per block from auto_prob config.

    Uses the same ``prob_sys_size_ext`` logic as the npy pipeline to parse
    the ``auto_prob`` string, then converts per-system probabilities into
    per-block target frame counts using the "max(frames/prob)" strategy.

    Parameters
    ----------
    auto_prob_style : str
        e.g. ``"prob_sys_size;0:3:0.5;3:10:0.5"``
    nsystems : int
        Total number of systems in the LMDB.
    system_nframes : list[int]
        Number of frames per system.

    Returns
    -------
    list[tuple[list[int], int]]
        Each element is ``(system_indices_in_block, target_frame_count)``.
        Returns empty list if no expansion is needed (all targets == actual).
    """
    from deepmd.utils.data_system import (
        prob_sys_size_ext,
    )

    # Parse block definitions from the auto_prob string
    # Format: "prob_sys_size;stt:end:weight;stt:end:weight;..."
    block_str = auto_prob_style.split(";")[1:]
    blocks: list[tuple[int, int, float]] = []
    for part in block_str:
        stt, end, weight = part.split(":")
        blocks.append((int(stt), int(end), float(weight)))

    # Compute per-system probabilities using the standard function
    sys_probs = prob_sys_size_ext(auto_prob_style, nsystems, system_nframes)

    # Group systems by block, compute block-level frames and prob
    block_info: list[tuple[list[int], int, float]] = []  # (sys_ids, frames, prob)
    for stt, end, _weight in blocks:
        sys_ids = list(range(stt, end))
        block_frames = sum(system_nframes[i] for i in sys_ids)
        block_prob = sum(sys_probs[i] for i in sys_ids)
        block_info.append((sys_ids, block_frames, block_prob))

    # Step 1-2: total_target = ceil(max(block_frames / block_prob))
    ratios = []
    for sys_ids, block_frames, block_prob in block_info:
        if block_prob > 0:
            ratios.append(block_frames / block_prob)
        else:
            ratios.append(0.0)
    total_target = math.ceil(max(ratios)) if ratios else 0

    # Step 3: per-block target = round(total_target * block_prob)
    result: list[tuple[list[int], int]] = []
    needs_expansion = False
    for sys_ids, block_frames, block_prob in block_info:
        target = round(total_target * block_prob)
        target = max(target, block_frames)  # never shrink
        if target > block_frames:
            needs_expansion = True
        result.append((sys_ids, target))

    if not needs_expansion:
        return []

    return result


def _expand_indices_by_blocks(
    indices: list[int],
    frame_system_ids: np.ndarray,
    block_targets: list[tuple[list[int], int]],
    rng: np.random.Generator,
    _block_total_actual: list[int] | None = None,
    _sid_to_blk_arr: np.ndarray | None = None,
) -> list[int]:
    """Expand frame indices according to block targets.

    For each block, computes the proportional target for the subset of
    indices belonging to that block (within the current nloc group),
    then applies full-copy + remainder sampling.

    Parameters
    ----------
    indices : list[int]
        Frame indices in the current nloc group.
    frame_system_ids : np.ndarray
        Per-frame system id for the entire dataset (int64 array).
    block_targets : list[tuple[list[int], int]]
        Per-block (system_ids, total_target_frames).
    rng : np.random.Generator
        RNG for remainder sampling.
    _block_total_actual : list[int] or None
        Pre-computed total actual frame count per block (across all nloc
        groups).  When provided, avoids an O(N) scan of frame_system_ids.
    _sid_to_blk_arr : np.ndarray or None
        Pre-computed system-id to block-index lookup array.  When provided,
        avoids rebuilding the mapping for each call.

    Returns
    -------
    list[int]
        Expanded indices.
    """
    n_blocks = len(block_targets)

    # Build sys_id -> block_idx lookup array
    if _sid_to_blk_arr is None:
        sys_to_block: dict[int, int] = {}
        for blk_idx, (sys_ids, _target) in enumerate(block_targets):
            for sid in sys_ids:
                sys_to_block[sid] = blk_idx
        max_sid = max(sys_to_block.keys()) + 1 if sys_to_block else 0
        _sid_to_blk_arr = np.full(max_sid, -1, dtype=np.int32)
        for sid, blk in sys_to_block.items():
            _sid_to_blk_arr[sid] = blk

    # Partition indices by block using numpy for speed
    idx_arr = np.asarray(indices, dtype=np.int64)
    sid_arr = np.asarray(frame_system_ids, dtype=np.int64)
    # Vectorized lookup: get block id for each index
    idx_sids = sid_arr[idx_arr]
    idx_blks = _sid_to_blk_arr[idx_sids]

    # Pre-compute block_total_actual if not provided
    if _block_total_actual is None:
        _block_total_actual = []
        for sys_ids, _ in block_targets:
            total = sum(int(np.sum(sid_arr == sid)) for sid in sys_ids)
            _block_total_actual.append(total)

    expanded_parts: list[np.ndarray] = []

    # Unassigned indices
    unassigned_mask = idx_blks == -1
    if np.any(unassigned_mask):
        expanded_parts.append(idx_arr[unassigned_mask])

    for blk_idx in range(n_blocks):
        blk_mask = idx_blks == blk_idx
        blk_idxs = idx_arr[blk_mask]
        n_actual = len(blk_idxs)
        if n_actual == 0:
            continue

        _, block_total_target = block_targets[blk_idx]
        block_total_act = _block_total_actual[blk_idx]

        # Proportional target for this nloc subset
        if block_total_act > 0:
            target = round(block_total_target * n_actual / block_total_act)
        else:
            target = n_actual
        target = max(target, n_actual)  # never shrink

        # Full copies + remainder
        deficit = target - n_actual
        if deficit <= 0:
            expanded_parts.append(blk_idxs)
        else:
            full_copies = deficit // n_actual
            remainder = deficit % n_actual
            # Original + full copies
            if full_copies > 0:
                expanded_parts.append(np.tile(blk_idxs, 1 + full_copies))
            else:
                expanded_parts.append(blk_idxs)
            # Remainder: sample without replacement
            if remainder > 0:
                sampled = rng.choice(blk_idxs, size=remainder, replace=False)
                expanded_parts.append(sampled)

    if expanded_parts:
        return np.concatenate(expanded_parts).tolist()
    return []


def _build_all_batches(
    reader: "LmdbDataReader",
    shuffle: bool,
    rng: np.random.Generator,
    block_targets: list[tuple[list[int], int]] | None = None,
) -> list[list[int]]:
    """Build the full list of same-nloc batches from the reader.

    This is the shared batch-construction logic used by both
    SameNlocBatchSampler (single-GPU) and DistributedSameNlocBatchSampler.

    Parameters
    ----------
    reader : LmdbDataReader
        Provides nloc_groups and get_batch_size_for_nloc.
    shuffle : bool
        Whether to shuffle indices within each nloc group and
        shuffle the final batch order.
    rng : np.random.Generator
        Random number generator (deterministic for reproducibility).
    block_targets : list[tuple[list[int], int]] or None
        Per-block (system_ids, target_frame_count) from compute_block_targets.
        When provided, indices are expanded via full-copy + remainder sampling.

    Returns
    -------
    list[list[int]]
        Each inner list is a batch of frame indices, all with the same nloc.
    """
    # Build per-group batches
    group_batches: list[list[list[int]]] = []

    # Pre-compute expensive objects once (avoids O(N) work per nloc group)
    block_total_actual: list[int] | None = None
    sid_arr: np.ndarray | None = None
    sid_to_blk_arr: np.ndarray | None = None
    if block_targets and reader.frame_system_ids is not None:
        block_total_actual = []
        for sys_ids, _ in block_targets:
            total = sum(reader.system_nframes[s] for s in sys_ids)
            block_total_actual.append(total)
        # Convert frame_system_ids to numpy once
        sid_arr = np.array(reader.frame_system_ids, dtype=np.int64)
        # Build sys_id -> block_idx lookup array once
        sys_to_block: dict[int, int] = {}
        for blk_idx, (sys_ids, _target) in enumerate(block_targets):
            for sid in sys_ids:
                sys_to_block[sid] = blk_idx
        max_sid = max(sys_to_block.keys()) + 1 if sys_to_block else 0
        sid_to_blk_arr = np.full(max_sid, -1, dtype=np.int32)
        for sid, blk in sys_to_block.items():
            sid_to_blk_arr[sid] = blk

    for nloc in sorted(reader.nloc_groups.keys()):
        indices = list(reader.nloc_groups[nloc])
        # Expand indices by block targets if provided
        if block_targets and sid_arr is not None:
            indices = _expand_indices_by_blocks(
                indices,
                sid_arr,
                block_targets,
                rng,
                _block_total_actual=block_total_actual,
                _sid_to_blk_arr=sid_to_blk_arr,
            )
        if shuffle:
            rng.shuffle(indices)
        bs = reader.get_batch_size_for_nloc(nloc)
        batches = []
        for start in range(0, len(indices), bs):
            batches.append(indices[start : start + bs])
        group_batches.append(batches)

    # Interleave groups round-robin
    all_batches: list[list[int]] = []
    max_len = max(len(gb) for gb in group_batches) if group_batches else 0
    for i in range(max_len):
        for gb in group_batches:
            if i < len(gb):
                all_batches.append(gb[i])

    # Optionally shuffle the interleaved order
    if shuffle:
        rng.shuffle(all_batches)

    return all_batches


class SameNlocBatchSampler:
    """Batch sampler that groups frames by nloc.

    For mixed-nloc datasets with mixed_batch=False: each batch contains only
    frames with the same nloc. Within each nloc group, frames are shuffled.
    Groups are interleaved round-robin so training sees diverse nloc values.

    When auto batch_size is used, batch_size is computed per-nloc-group.

    The sampler is deterministic: given the same seed, repeated calls to
    ``__iter__`` produce the same batch sequence.

    Parameters
    ----------
    reader : LmdbDataReader
        The dataset reader (provides nloc_groups, get_batch_size_for_nloc).
    shuffle : bool
        Whether to shuffle within each nloc group each epoch.
    seed : int or None
        Random seed for reproducibility.
    block_targets : list[tuple[list[int], int]] or None
        Per-block expansion targets from compute_block_targets.
    """

    def __init__(
        self,
        reader: LmdbDataReader,
        shuffle: bool = True,
        seed: int | None = None,
        block_targets: list[tuple[list[int], int]] | None = None,
    ) -> None:
        self._reader = reader
        self._shuffle = shuffle
        self._seed = seed
        self._block_targets = block_targets

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of frame indices, all with the same nloc."""
        rng = np.random.default_rng(self._seed)
        yield from _build_all_batches(
            self._reader, self._shuffle, rng, self._block_targets
        )

    def __len__(self) -> int:
        """Total number of batches across all nloc groups (estimated)."""
        total = 0
        for nloc, indices in self._reader.nloc_groups.items():
            n = len(indices)
            if self._block_targets and self._reader.frame_system_ids is not None:
                # Estimate expanded count for this nloc group
                n = self._estimate_expanded_count(indices)
            bs = self._reader.get_batch_size_for_nloc(nloc)
            total += (n + bs - 1) // bs
        return total

    def _estimate_expanded_count(self, indices: list[int]) -> int:
        """Estimate expanded index count for __len__ without RNG."""
        if not self._block_targets or self._reader.frame_system_ids is None:
            return len(indices)
        sys_ids = self._reader.frame_system_ids
        total = 0
        for blk_idx, (blk_sys_ids, block_target) in enumerate(self._block_targets):
            blk_sys_set = set(blk_sys_ids)
            n_in_nloc = sum(1 for i in indices if sys_ids[i] in blk_sys_set)
            if n_in_nloc == 0:
                continue
            block_total_actual = sum(1 for sid in sys_ids if sid in blk_sys_set)
            if block_total_actual > 0:
                target = round(block_target * n_in_nloc / block_total_actual)
            else:
                target = n_in_nloc
            total += max(target, n_in_nloc)
        # Add unassigned
        all_sys = set()
        for blk_sys_ids, _ in self._block_targets:
            all_sys.update(blk_sys_ids)
        total += sum(1 for i in indices if sys_ids[i] not in all_sys)
        return total


class DistributedSameNlocBatchSampler:
    """Distributed wrapper for same-nloc batch sampling.

    All ranks build the same deterministic global batch list (using
    ``seed + epoch``), then each rank takes a disjoint subset via
    :meth:`_partition_batches`.

    Override :meth:`_partition_batches` for custom load-balancing strategies.
    The default uses strided partitioning which gives good nloc diversity per
    rank.

    Parameters
    ----------
    reader : LmdbDataReader
        The dataset reader (provides nloc_groups, get_batch_size_for_nloc,
        frame_nlocs).
    rank : int
        Rank of the current process.
    world_size : int
        Total number of processes.
    shuffle : bool
        Whether to shuffle batches.
    seed : int or None
        Base seed for deterministic RNG. All ranks must use the same seed.
    block_targets : list[tuple[list[int], int]] or None
        Per-block expansion targets from compute_block_targets.
    """

    def __init__(
        self,
        reader: LmdbDataReader,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        seed: int | None = None,
        block_targets: list[tuple[list[int], int]] | None = None,
    ) -> None:
        self._reader = reader
        self._rank = rank
        self._world_size = world_size
        self._shuffle = shuffle
        self._seed = seed if seed is not None else 0
        self._epoch = 0
        self._block_targets = block_targets

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic cross-rank shuffling.

        Call this before each training epoch/cycle to get different but
        reproducible batch orderings across epochs.
        """
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        """Yield this rank's partition of the global batch list."""
        # All ranks build the same global batch list deterministically
        rng = np.random.default_rng(self._seed + self._epoch)
        all_batches = _build_all_batches(
            self._reader, self._shuffle, rng, self._block_targets
        )
        # Partition to this rank
        yield from self._partition_batches(all_batches)

    def _partition_batches(self, all_batches: list[list[int]]) -> list[list[int]]:
        """Partition global batches to this rank.

        Default: strided partition ``all_batches[rank::world_size]``.
        This gives good nloc diversity per rank since batches are
        interleaved across nloc groups before shuffling.

        Override this method for custom load-balancing. For example, a
        greedy algorithm could assign batches to ranks based on estimated
        compute cost (``reader.frame_nlocs[batch[0]]`` gives the nloc of
        each batch).
        """
        return all_batches[self._rank :: self._world_size]

    def __len__(self) -> int:
        """Number of batches for this rank."""
        total = 0
        for nloc, indices in self._reader.nloc_groups.items():
            bs = self._reader.get_batch_size_for_nloc(nloc)
            total += (len(indices) + bs - 1) // bs
        return math.ceil(total / self._world_size)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size


class LmdbTestData:
    """LMDB-backed data reader for dp test.

    Mimics the DeepmdData interface used by test_ener():
    .add(), .get_test(), .mixed_type, .pbc

    For mixed-nloc datasets, frames are grouped by nloc.
    get_test(nloc=...) returns data for a specific group.
    """

    def __init__(
        self,
        lmdb_path: str,
        type_map: list[str] | None = None,
        shuffle_test: bool = True,
        **kwargs: Any,
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self._type_map = type_map or []
        self._env = _open_lmdb(self.lmdb_path)

        with self._env.begin() as txn:
            meta = _read_metadata(txn)

        self.nframes, self._frame_fmt, self._natoms_per_type = _parse_metadata(meta)
        self._natoms = sum(self._natoms_per_type)

        # Build type remapping if LMDB's type_map differs from model's type_map
        lmdb_type_map = meta.get("type_map")
        self._lmdb_type_map = lmdb_type_map
        self._type_remap: np.ndarray | None = None
        if (
            lmdb_type_map is not None
            and self._type_map
            and list(lmdb_type_map) != list(self._type_map)
        ):
            remap = np.empty(len(lmdb_type_map), dtype=np.int32)
            for i, name in enumerate(lmdb_type_map):
                if name not in self._type_map:
                    raise ValueError(
                        f"Element '{name}' in LMDB type_map {lmdb_type_map} "
                        f"not found in model type_map {self._type_map}"
                    )
                remap[i] = self._type_map.index(name)
            self._type_remap = remap
            log.info(
                f"LmdbTestData type remapping: LMDB {lmdb_type_map} -> "
                f"model {self._type_map}, remap={remap}"
            )

        # Read all frames
        self._frames: list[dict[str, Any]] = []
        with self._env.begin() as txn:
            for i in range(self.nframes):
                key = format(i, self._frame_fmt).encode()
                raw = txn.get(key)
                if raw is not None:
                    frame = _remap_keys(_decode_frame(raw))
                    # Apply type remapping to atype
                    if (
                        self._type_remap is not None
                        and "atype" in frame
                        and isinstance(frame["atype"], np.ndarray)
                    ):
                        frame["atype"] = self._type_remap[
                            frame["atype"].reshape(-1)
                        ].astype(np.int64)
                    self._frames.append(frame)

        # Shuffle if requested
        if shuffle_test:
            rng = np.random.default_rng()
            indices = rng.permutation(len(self._frames))
            self._frames = [self._frames[i] for i in indices]

        # Group frames by nloc
        self._nloc_groups: dict[int, list[int]] = {}
        for idx, frame in enumerate(self._frames):
            atype = frame.get("atype")
            nloc = len(atype) if isinstance(atype, np.ndarray) else self._natoms
            self._nloc_groups.setdefault(nloc, []).append(idx)

        # Data requirements
        self._requirements: dict[str, dict[str, Any]] = {}

        # Detect PBC: if any frame has a non-zero box
        self.pbc = True
        if len(self._frames) > 0:
            f0 = self._frames[0]
            if "box" not in f0:
                self.pbc = False
            elif isinstance(f0["box"], np.ndarray) and np.allclose(f0["box"], 0.0):
                self.pbc = False

        self.mixed_type = True

    @property
    def nloc_groups(self) -> dict[int, list[int]]:
        """Nloc → list of frame indices in self._frames."""
        return self._nloc_groups

    def add(
        self,
        key: str,
        ndof: int,
        atomic: bool = False,
        must: bool = True,
        high_prec: bool = False,
        repeat: int = 1,
        default: float = 0.0,
        dtype: np.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        """Register a data requirement (mirrors DeepmdData.add)."""
        self._requirements[key] = {
            "ndof": ndof,
            "atomic": atomic,
            "must": must,
            "high_prec": high_prec,
            "repeat": repeat,
            "default": default,
            "dtype": dtype,
        }

    def _resolve_dtype(self, key: str) -> np.dtype:
        """Resolve target dtype for a key using registered requirements."""
        if key in self._requirements:
            req = self._requirements[key]
            dtype = req.get("dtype")
            if dtype is not None:
                return dtype
            if req.get("high_prec", False):
                return GLOBAL_ENER_FLOAT_PRECISION
            return GLOBAL_NP_FLOAT_PRECISION
        if key in _HIGH_PREC_KEYS:
            return GLOBAL_ENER_FLOAT_PRECISION
        return GLOBAL_NP_FLOAT_PRECISION

    def get_test(self, nloc: int | None = None) -> dict[str, Any]:
        """Return frames stacked as numpy arrays.

        Parameters
        ----------
        nloc : int or None
            If specified, return only frames with this atom count.
            If None and all frames have the same nloc, return all.
            If None and mixed nloc, return the largest group and log a warning.
        Returns dict matching DeepmdData.get_test() format:
        """
        if nloc is not None:
            if nloc not in self._nloc_groups:
                raise ValueError(
                    f"No frames with nloc={nloc}. Available: {sorted(self._nloc_groups.keys())}"
                )
            frame_indices = self._nloc_groups[nloc]
            natoms = nloc
        elif len(self._nloc_groups) == 1:
            # Uniform nloc — use all frames
            natoms = next(iter(self._nloc_groups))
            frame_indices = list(range(len(self._frames)))
        else:
            # Mixed nloc — use the largest group
            natoms = max(self._nloc_groups, key=lambda k: len(self._nloc_groups[k]))
            frame_indices = self._nloc_groups[natoms]
            group_summary = {k: len(v) for k, v in sorted(self._nloc_groups.items())}
            log.warning(
                f"Mixed-nloc LMDB for dp test: using nloc={natoms} group "
                f"({len(frame_indices)} frames). "
                f"Available groups: {group_summary}"
            )

        frames = [self._frames[i] for i in frame_indices]
        return self._stack_frames(frames, natoms)

    def _stack_frames(
        self, frames: list[dict[str, Any]], natoms: int
    ) -> dict[str, Any]:
        """Stack a list of same-nloc frames into numpy arrays."""
        nframes = len(frames)
        result: dict[str, Any] = {}

        # Core arrays
        coords = []
        boxes = []
        atypes = []

        for frame in frames:
            if "coord" in frame and isinstance(frame["coord"], np.ndarray):
                coords.append(
                    frame["coord"]
                    .reshape(natoms * 3)
                    .astype(self._resolve_dtype("coord"))
                )
            if "box" in frame and isinstance(frame["box"], np.ndarray):
                boxes.append(frame["box"].reshape(9).astype(self._resolve_dtype("box")))
            else:
                boxes.append(np.zeros(9, dtype=self._resolve_dtype("box")))
            if "atype" in frame and isinstance(frame["atype"], np.ndarray):
                atypes.append(frame["atype"].reshape(natoms).astype(np.int64))

        result["coord"] = (
            np.stack(coords)
            if coords
            else np.zeros((0, natoms * 3), dtype=self._resolve_dtype("coord"))
        )
        result["box"] = (
            np.stack(boxes)
            if boxes
            else np.zeros((0, 9), dtype=self._resolve_dtype("box"))
        )
        result["type"] = (
            np.stack(atypes) if atypes else np.zeros((0, natoms), dtype=np.int64)
        )

        # Label keys and registered requirements
        all_keys: dict[str, dict[str, Any]] = {}
        for key in [
            "energy",
            "force",
            "virial",
            "atom_ener",
            "atom_pref",
            "force_mag",
            "spin",
            "fparam",
            "aparam",
            "hessian",
            "efield",
        ]:
            all_keys[key] = {"ndof": None, "atomic": False, "default": 0.0}
        for key, req in self._requirements.items():
            all_keys[key] = req

        for key, req_info in all_keys.items():
            has_key = any(
                key in f and isinstance(f.get(key), np.ndarray) for f in frames
            )
            result[f"find_{key}"] = 1.0 if has_key else 0.0

            if has_key:
                arrays = []
                for frame in frames:
                    val = frame.get(key)
                    if isinstance(val, np.ndarray):
                        arrays.append(val.astype(self._resolve_dtype(key)).ravel())
                    elif val is not None:
                        arrays.append(
                            np.array([float(val)], dtype=self._resolve_dtype(key))
                        )
                    else:
                        ref = next(
                            (
                                f[key]
                                for f in frames
                                if isinstance(f.get(key), np.ndarray)
                            ),
                            None,
                        )
                        if ref is not None:
                            arrays.append(
                                np.zeros(ref.size, dtype=self._resolve_dtype(key))
                            )
                        else:
                            arrays.append(np.zeros(1, dtype=self._resolve_dtype(key)))
                result[key] = np.stack(arrays)
            elif key in self._requirements:
                ndof = self._requirements[key]["ndof"]
                atomic = self._requirements[key]["atomic"]
                default = self._requirements[key]["default"]
                if atomic:
                    shape = (nframes, natoms * ndof)
                else:
                    shape = (nframes, ndof)
                result[key] = np.full(shape, default, dtype=self._resolve_dtype(key))

        return result


def merge_lmdb(
    src_paths: list[str],
    dst_path: str,
    *,
    map_size: int = 1024**4,  # 1 TB default
) -> str:
    """Merge multiple LMDB datasets into one.

    Frames are concatenated in order. The output metadata includes a
    ``frame_nlocs`` list for fast init (skips per-frame scan).

    Parameters
    ----------
    src_paths : list[str]
        Paths to source LMDB directories.
    dst_path : str
        Path for the merged LMDB output.
    map_size : int
        Maximum size of the output LMDB (default 1 TB).

    Returns
    -------
    str
        Path to the created LMDB.
    """
    import os
    import shutil

    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)

    dst_env = lmdb.open(dst_path, map_size=map_size)
    frame_idx = 0
    fmt = "012d"
    frame_nlocs: list[int] = []
    frame_system_ids: list[int] = []
    first_system_info: dict | None = None
    first_type_map: list[str] | None = None
    sys_id_offset = 0

    for src_path in src_paths:
        src_env = _open_lmdb(src_path)
        with src_env.begin() as txn:
            meta = _read_metadata(txn)
        nframes, src_fmt, natoms_per_type = _parse_metadata(meta)
        fallback_natoms = sum(natoms_per_type)

        if first_system_info is None:
            first_system_info = meta.get("system_info", {})
        if first_type_map is None:
            first_type_map = meta.get("type_map")

        # Check for pre-computed frame_nlocs in source
        src_nlocs = meta.get("frame_nlocs")
        # Check for frame_system_ids in source
        src_sys_ids = meta.get("frame_system_ids")

        with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
            for i in range(nframes):
                src_key = format(i, src_fmt).encode()
                raw = src_txn.get(src_key)
                if raw is None:
                    continue
                dst_key = format(frame_idx, fmt).encode()
                dst_txn.put(dst_key, raw)

                # Get nloc for this frame
                if src_nlocs is not None:
                    frame_nlocs.append(int(src_nlocs[i]))
                else:
                    frame_raw = msgpack.unpackb(raw, raw=False)
                    atype_raw = frame_raw.get("atom_types")
                    if isinstance(atype_raw, dict):
                        shape = atype_raw.get("shape") or atype_raw.get(b"shape")
                        if shape:
                            frame_nlocs.append(int(shape[0]))
                        else:
                            frame_nlocs.append(fallback_natoms)
                    else:
                        frame_nlocs.append(fallback_natoms)

                # Propagate system IDs with offset
                if src_sys_ids is not None and i < len(src_sys_ids):
                    frame_system_ids.append(int(src_sys_ids[i]) + sys_id_offset)
                else:
                    frame_system_ids.append(sys_id_offset)

                frame_idx += 1

        # Update sys_id_offset for next source
        if src_sys_ids is not None and len(src_sys_ids) > 0:
            sys_id_offset += max(int(s) for s in src_sys_ids) + 1
        else:
            sys_id_offset += 1

        src_env.close()

    # Write merged metadata with frame_nlocs for fast init
    merged_meta = {
        "nframes": frame_idx,
        "frame_idx_fmt": fmt,
        "system_info": first_system_info or {},
        "frame_nlocs": frame_nlocs,
        "frame_system_ids": frame_system_ids,
    }
    if first_type_map is not None:
        merged_meta["type_map"] = first_type_map
    with dst_env.begin(write=True) as txn:
        txn.put(b"__metadata__", msgpack.packb(merged_meta, use_bin_type=True))
    dst_env.close()

    nloc_counts: dict[int, int] = {}
    for n in frame_nlocs:
        nloc_counts[n] = nloc_counts.get(n, 0) + 1
    log.info(
        f"Merged {len(src_paths)} LMDBs → {dst_path}: "
        f"{frame_idx} frames, nloc groups: {dict(sorted(nloc_counts.items()))}"
    )
    return dst_path
