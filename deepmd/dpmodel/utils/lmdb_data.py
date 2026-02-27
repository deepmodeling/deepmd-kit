# SPDX-License-Identifier: LGPL-3.0-or-later
"""Framework-agnostic LMDB data utilities for DeePMD-kit.

All code here is pure Python/NumPy/lmdb/msgpack — no framework dependency.
Backend-specific wrappers (PyTorch Dataset, JAX, etc.) import from here.
"""

import logging
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
}


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
            frame["coord"] = frame["coord"].reshape(-1, 3).astype(np.float64)
        if "box" in frame and isinstance(frame["box"], np.ndarray):
            frame["box"] = frame["box"].reshape(9).astype(np.float64)
        if "energy" in frame:
            val = frame["energy"]
            if isinstance(val, np.ndarray):
                frame["energy"] = val.reshape(1).astype(np.float64)
            else:
                frame["energy"] = np.array([float(val)], dtype=np.float64)
        if "force" in frame and isinstance(frame["force"], np.ndarray):
            frame["force"] = frame["force"].reshape(-1, 3).astype(np.float64)
        if "atype" in frame and isinstance(frame["atype"], np.ndarray):
            frame["atype"] = frame["atype"].reshape(-1).astype(np.int64)
        if "virial" in frame and isinstance(frame["virial"], np.ndarray):
            frame["virial"] = frame["virial"].reshape(9).astype(np.float64)

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
                ndof = req_item["ndof"]
                default = req_item["default"]
                atomic = req_item["atomic"]
                if atomic:
                    shape = (frame_natoms, ndof)
                else:
                    shape = (ndof,)
                frame[req_key] = np.full(shape, default, dtype=np.float64)
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
        unique_nlocs = sorted(self._nloc_groups.keys())
        nloc_info = ", ".join(
            f"{nloc}({len(idxs)})" for nloc, idxs in sorted(self._nloc_groups.items())
        )
        log.info(
            f"LMDB {name}: {self.lmdb_path}, "
            f"{self.nframes} frames, nloc groups: [{nloc_info}], "
            f"batch_size={'auto' if self._auto_rule else self.batch_size}, "
            f"mixed_batch={self.mixed_batch}"
        )

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
    def nloc_groups(self) -> dict[int, list[int]]:
        """Nloc → list of frame indices."""
        return self._nloc_groups

    @property
    def frame_nlocs(self) -> list[int]:
        """Per-frame atom count."""
        return self._frame_nlocs


class SameNlocBatchSampler:
    """Batch sampler that groups frames by nloc.

    For mixed-nloc datasets with mixed_batch=False: each batch contains only
    frames with the same nloc. Within each nloc group, frames are shuffled.
    Groups are interleaved round-robin so training sees diverse nloc values.

    When auto batch_size is used, batch_size is computed per-nloc-group.

    Parameters
    ----------
    reader : LmdbDataReader
        The dataset reader (provides nloc_groups, get_batch_size_for_nloc).
    shuffle : bool
        Whether to shuffle within each nloc group each epoch.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        reader: LmdbDataReader,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        self._reader = reader
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of frame indices, all with the same nloc."""
        # Build per-group batches
        group_batches: list[list[list[int]]] = []
        for nloc in sorted(self._reader.nloc_groups.keys()):
            indices = list(self._reader.nloc_groups[nloc])
            if self._shuffle:
                self._rng.shuffle(indices)
            bs = self._reader.get_batch_size_for_nloc(nloc)
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
        if self._shuffle:
            self._rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        """Total number of batches across all nloc groups."""
        total = 0
        for nloc, indices in self._reader.nloc_groups.items():
            bs = self._reader.get_batch_size_for_nloc(nloc)
            total += (len(indices) + bs - 1) // bs
        return total


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

        # Read all frames
        self._frames: list[dict[str, Any]] = []
        with self._env.begin() as txn:
            for i in range(self.nframes):
                key = format(i, self._frame_fmt).encode()
                raw = txn.get(key)
                if raw is not None:
                    self._frames.append(_remap_keys(_decode_frame(raw)))

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

        self.mixed_type = False

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
        }

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
                coords.append(frame["coord"].reshape(natoms * 3).astype(np.float64))
            if "box" in frame and isinstance(frame["box"], np.ndarray):
                boxes.append(frame["box"].reshape(9).astype(np.float64))
            else:
                boxes.append(np.zeros(9, dtype=np.float64))
            if "atype" in frame and isinstance(frame["atype"], np.ndarray):
                atypes.append(frame["atype"].reshape(natoms).astype(np.int64))

        result["coord"] = (
            np.stack(coords) if coords else np.zeros((0, natoms * 3), dtype=np.float64)
        )
        result["box"] = np.stack(boxes) if boxes else np.zeros((0, 9), dtype=np.float64)
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
                        arrays.append(val.astype(np.float64).ravel())
                    elif val is not None:
                        arrays.append(np.array([float(val)], dtype=np.float64))
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
                            arrays.append(np.zeros(ref.size, dtype=np.float64))
                        else:
                            arrays.append(np.zeros(1, dtype=np.float64))
                result[key] = np.stack(arrays)
            elif key in self._requirements:
                ndof = self._requirements[key]["ndof"]
                atomic = self._requirements[key]["atomic"]
                default = self._requirements[key]["default"]
                if atomic:
                    shape = (nframes, natoms * ndof)
                else:
                    shape = (nframes, ndof)
                result[key] = np.full(shape, default, dtype=np.float64)

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
    first_system_info: dict | None = None

    for src_path in src_paths:
        src_env = _open_lmdb(src_path)
        with src_env.begin() as txn:
            meta = _read_metadata(txn)
        nframes, src_fmt, natoms_per_type = _parse_metadata(meta)
        fallback_natoms = sum(natoms_per_type)

        if first_system_info is None:
            first_system_info = meta.get("system_info", {})

        # Check for pre-computed frame_nlocs in source
        src_nlocs = meta.get("frame_nlocs")

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

                frame_idx += 1
        src_env.close()

    # Write merged metadata with frame_nlocs for fast init
    merged_meta = {
        "nframes": frame_idx,
        "frame_idx_fmt": fmt,
        "system_info": first_system_info or {},
        "frame_nlocs": frame_nlocs,
    }
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
