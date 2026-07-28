# SPDX-License-Identifier: LGPL-3.0-or-later
"""Framework-agnostic LMDB data utilities for DeePMD-kit.

All code here is pure Python/NumPy/lmdb/msgpack — no framework dependency.
Backend-specific wrappers (PyTorch Dataset, JAX, etc.) import from here.
"""

import logging
import math
import multiprocessing
import threading
from collections.abc import (
    Iterator,
    Sequence,
)
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
)
from dataclasses import (
    dataclass,
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

from deepmd.dpmodel.utils.dist_check import (
    compute_min_pair_dist_single,
)
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

# Process-level cache: python-lmdb does not allow opening the same path twice
# in one process.  We ref-count so the Environment is closed (and freed from
# the cache) once every reader that shares it is garbage-collected.
_ENV_CACHE: dict[str, tuple[lmdb.Environment, int]] = {}


def _open_lmdb(path: str) -> lmdb.Environment:
    """Open (or reuse) a readonly LMDB environment with reference counting.

    The python-lmdb binding raises ``lmdb.Error`` if the same path is opened
    more than once in a single process.  We cache by resolved absolute path
    and bump a reference count.  Call :func:`_close_lmdb` when done to
    decrement the count; when it reaches zero the environment is closed and
    removed from the cache.
    """
    resolved = str(Path(path).resolve())
    entry = _ENV_CACHE.get(resolved)
    if entry is not None:
        env, refcount = entry
        _ENV_CACHE[resolved] = (env, refcount + 1)
        return env
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    _ENV_CACHE[resolved] = (env, 1)
    return env


def _close_lmdb(path: str) -> None:
    """Decrement the ref-count for *path* and close the env when it hits zero."""
    resolved = str(Path(path).resolve())
    entry = _ENV_CACHE.get(resolved)
    if entry is None:
        return
    env, refcount = entry
    if refcount <= 1:
        del _ENV_CACHE[resolved]
        try:
            env.close()
        except Exception:
            pass
    else:
        _ENV_CACHE[resolved] = (env, refcount - 1)


def _read_metadata(txn: lmdb.Transaction) -> dict:
    """Read and decode __metadata__ from LMDB transaction."""
    raw = txn.get(b"__metadata__")
    if raw is None:
        raise ValueError("LMDB file missing __metadata__ key")
    return msgpack.unpackb(raw, raw=False)


def _decode_array(obj: dict, *, copy: bool = True) -> np.ndarray:
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
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return array.copy() if copy else array


def _is_encoded_array(val: Any) -> bool:
    """Check if a value is a msgpack-encoded ndarray dict."""
    if not isinstance(val, dict):
        return False
    return ("data" in val and "type" in val) or (b"data" in val and b"type" in val)


def _decode_value(val: Any, *, copy_arrays: bool = True) -> Any:
    """Decode a value: encoded array -> ndarray, list of encoded -> list of ndarray, else pass through."""
    if _is_encoded_array(val):
        return _decode_array(val, copy=copy_arrays)
    elif isinstance(val, list) and len(val) > 0 and _is_encoded_array(val[0]):
        return [_decode_array(item, copy=copy_arrays) for item in val]
    return val


def _decode_frame(
    raw_bytes: bytes,
    *,
    copy_arrays: bool = True,
) -> dict[str, Any]:
    """Decode a msgpack-serialized frame into a dict of numpy arrays / scalars."""
    frame = msgpack.unpackb(raw_bytes, raw=False)
    result = {}
    for key, val in frame.items():
        result[key] = _decode_value(val, copy_arrays=copy_arrays)
    return result


def _remap_keys(frame: dict[str, Any]) -> dict[str, Any]:
    """Remap LMDB key names to DeePMD convention, pass through unknown keys."""
    out = {}
    for k, v in frame.items():
        out[_KEY_REMAP.get(k, k)] = v
    return out


def _remap_atom_types(atype: np.ndarray, type_remap: np.ndarray) -> np.ndarray:
    """Remap real atom types while preserving negative virtual sentinels.

    Positive indices retain NumPy's normal bounds checking, so malformed LMDB
    data cannot be silently reinterpreted as a different species.
    """
    remapped_atype = atype.astype(np.int64, copy=True)
    real_atom_mask = remapped_atype >= 0
    remapped_atype[real_atom_mask] = type_remap[remapped_atype[real_atom_mask]]
    return remapped_atype


@dataclass
class LmdbDecodeConfig:
    """Serializable state required to decode one LMDB frame.

    The configuration deliberately excludes the LMDB environment and the
    dataset-wide index tables. It can therefore be sent to worker processes
    without duplicating the potentially very large metadata owned by
    :class:`LmdbDataReader`.

    Parameters
    ----------
    ntypes
        Number of model atom types.
    natoms
        Fallback atom count for records without ``atom_types``.
    type_remap
        Optional LMDB-type to model-type lookup table.
    data_requirements
        Registered data requirements keyed by field name.
    """

    ntypes: int
    natoms: int
    type_remap: np.ndarray | None
    data_requirements: dict[str, Any]


def _requirement_dtype(requirement: Any) -> np.dtype:
    """Resolve the NumPy dtype associated with a data requirement."""
    if isinstance(requirement, dict):
        dtype = requirement.get("dtype")
        high_precision = requirement.get("high_prec", False)
    else:
        dtype = getattr(requirement, "dtype", None)
        high_precision = getattr(requirement, "high_prec", False)
    if dtype is not None:
        return np.dtype(dtype)
    return np.dtype(
        GLOBAL_ENER_FLOAT_PRECISION if high_precision else GLOBAL_NP_FLOAT_PRECISION
    )


def _resolve_frame_dtype(config: LmdbDecodeConfig, key: str) -> np.dtype:
    """Resolve one decoded field's output dtype."""
    requirement = config.data_requirements.get(key)
    if requirement is not None:
        return _requirement_dtype(requirement)
    if key in _HIGH_PREC_KEYS:
        return np.dtype(GLOBAL_ENER_FLOAT_PRECISION)
    return np.dtype(GLOBAL_NP_FLOAT_PRECISION)


def _compute_frame_natoms(atype: np.ndarray, ntypes: int) -> np.ndarray:
    """Build ``[nloc, nloc, count(type_0), ...]`` for one frame.

    Negative virtual types are excluded from the per-type counts, matching
    mixed-type NPY data handling, as are positive indices outside the
    configured type map. The leading ``nloc`` entries still count every atom
    slot.
    """
    nloc = len(atype)
    real_atype = atype[(atype >= 0) & (atype < ntypes)]
    counts = np.bincount(real_atype, minlength=ntypes)
    natoms = np.empty(ntypes + 2, dtype=np.int64)
    natoms[0] = nloc
    natoms[1] = nloc
    natoms[2:] = counts
    return natoms


def decode_lmdb_frame(
    raw: bytes,
    original_key: int,
    config: LmdbDecodeConfig,
    *,
    copy_arrays: bool,
) -> dict[str, Any]:
    """Decode and normalize one LMDB record.

    Parameters
    ----------
    raw
        Msgpack-encoded frame payload.
    original_key
        Integer LMDB frame key.
    config
        Decoder state independent of the LMDB environment.
    copy_arrays
        Whether encoded arrays are copied while unpacking. Batch decoding sets
        this to ``False`` because every value is copied exactly once into its
        preallocated batch destination.

    Returns
    -------
    dict[str, Any]
        One normalized frame in DeePMD data-system convention.
    """
    frame = _remap_keys(_decode_frame(raw, copy_arrays=copy_arrays))

    for metadata_key in ("atom_numbs", "atom_names", "orig"):
        frame.pop(metadata_key, None)

    if "coord" in frame and isinstance(frame["coord"], np.ndarray):
        frame["coord"] = (
            frame["coord"]
            .reshape(-1, 3)
            .astype(_resolve_frame_dtype(config, "coord"), copy=False)
        )
    if "box" in frame and isinstance(frame["box"], np.ndarray):
        frame["box"] = (
            frame["box"]
            .reshape(9)
            .astype(_resolve_frame_dtype(config, "box"), copy=False)
        )
    if "energy" in frame:
        value = frame["energy"]
        if isinstance(value, np.ndarray):
            frame["energy"] = value.reshape(1).astype(
                _resolve_frame_dtype(config, "energy"), copy=False
            )
        else:
            frame["energy"] = np.array(
                [float(value)], dtype=_resolve_frame_dtype(config, "energy")
            )
    if "force" in frame and isinstance(frame["force"], np.ndarray):
        frame["force"] = (
            frame["force"]
            .reshape(-1, 3)
            .astype(_resolve_frame_dtype(config, "force"), copy=False)
        )
    if "atype" in frame and isinstance(frame["atype"], np.ndarray):
        frame["atype"] = frame["atype"].reshape(-1).astype(np.int64, copy=False)
        if config.type_remap is not None:
            frame["atype"] = _remap_atom_types(frame["atype"], config.type_remap)
    if "virial" in frame and isinstance(frame["virial"], np.ndarray):
        frame["virial"] = (
            frame["virial"]
            .reshape(9)
            .astype(_resolve_frame_dtype(config, "virial"), copy=False)
        )

    atype = frame.get("atype")
    if atype is not None:
        frame_natoms = len(atype)
        natoms = _compute_frame_natoms(atype, config.ntypes)
    else:
        frame_natoms = config.natoms
        natoms = np.array(
            [config.natoms, config.natoms] + [0] * config.ntypes,
            dtype=np.int64,
        )
    frame["natoms"] = natoms
    frame["real_natoms_vec"] = natoms

    requirements = config.data_requirements
    coord = frame.get("coord")
    if (
        "min_pair_dist" in requirements
        and "min_pair_dist" not in frame
        and isinstance(coord, np.ndarray)
        and isinstance(atype, np.ndarray)
    ):
        box = frame.get("box")
        if box is not None and np.allclose(box, 0.0):
            box = None
        requirement = requirements["min_pair_dist"]
        default = (
            requirement.get("default", 0.0)
            if isinstance(requirement, dict)
            else getattr(requirement, "default", 0.0)
        )
        frame["find_min_pair_dist"] = np.float32(1.0)
        frame["min_pair_dist"] = np.array(
            [
                compute_min_pair_dist_single(
                    coord,
                    box,
                    atype,
                    stop_below=float(default),
                )
            ],
            dtype=_resolve_frame_dtype(config, "min_pair_dist"),
        )

    structural_keys = frozenset(
        {
            "coord",
            "box",
            "atype",
            "natoms",
            "real_natoms_vec",
            "fid",
        }
    )
    for key in list(frame):
        if key.startswith("find_") or key in structural_keys or key in requirements:
            continue
        frame.setdefault(f"find_{key}", np.float32(1.0))

    for key, requirement in requirements.items():
        if isinstance(requirement, dict):
            ndof = requirement["ndof"]
            default = requirement["default"]
            atomic = requirement["atomic"]
            repeat = requirement.get("repeat", 1)
        else:
            ndof = requirement.ndof
            default = requirement.default
            atomic = requirement.atomic
            repeat = getattr(requirement, "repeat", 1)
        dtype = _requirement_dtype(requirement)

        if key not in frame:
            frame[f"find_{key}"] = np.float32(0.0)
            shape = (frame_natoms, ndof) if atomic else (ndof,)
            data = np.full(shape, default, dtype=dtype)
            if repeat != 1:
                data = np.repeat(data, repeat).reshape(-1)
            frame[key] = data
        else:
            frame.setdefault(f"find_{key}", np.float32(1.0))
            if repeat != 1 and isinstance(frame[key], np.ndarray):
                frame[key] = (
                    np.repeat(frame[key], repeat).reshape(-1).astype(dtype, copy=False)
                )

    for key in ("fparam", "aparam", "spin", "charge_spin"):
        frame.setdefault(
            f"find_{key}",
            np.float32(1.0 if key in frame else 0.0),
        )

    frame["fid"] = original_key
    return frame


def _allocate_lmdb_batch(
    frame: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """Allocate a contiguous NumPy batch from the first decoded frame."""
    batch: dict[str, Any] = {}
    for key, value in frame.items():
        if key.startswith("find_"):
            batch[key] = value
        elif key == "fid":
            batch[key] = [None] * batch_size
            batch[key][0] = value
        elif key == "type":
            continue
        elif value is None:
            batch[key] = None
        else:
            array = np.asarray(value)
            destination = np.empty((batch_size, *array.shape), dtype=array.dtype)
            destination[0] = array
            batch[key] = destination
    return batch


def decode_lmdb_batch(
    transaction: lmdb.Transaction,
    original_keys: Sequence[int],
    frame_format: str,
    config: LmdbDecodeConfig,
) -> dict[str, Any]:
    """Decode LMDB records directly into preallocated contiguous arrays.

    The function keeps at most one temporary frame alive. It avoids the
    decode-copy, dtype-copy, Python frame-list, and final ``numpy.stack``
    sequence used by generic collation.
    """
    if not original_keys:
        raise ValueError("decode_lmdb_batch requires at least one frame key")

    batch: dict[str, Any] | None = None
    batch_size = len(original_keys)
    expected_fields: frozenset[str] | None = None
    for row, original_key in enumerate(original_keys):
        key = format(int(original_key), frame_format).encode()
        raw = transaction.get(key)
        if raw is None:
            raise IndexError(f"Frame {original_key} not found in LMDB")
        frame = decode_lmdb_frame(
            raw,
            int(original_key),
            config,
            copy_arrays=False,
        )
        if batch is None:
            batch = _allocate_lmdb_batch(frame, batch_size)
            expected_fields = frozenset(frame)
            continue

        frame_fields = frozenset(frame)
        if frame_fields != expected_fields:
            raise ValueError(
                "LMDB frames in one same-nloc batch expose inconsistent fields: "
                f"frame {original_keys[0]} has {sorted(expected_fields)}, while "
                f"frame {original_key} has {sorted(frame_fields)}"
            )
        for field, value in frame.items():
            if field.startswith("find_"):
                if not np.array_equal(batch[field], value):
                    raise ValueError(
                        f"LMDB field availability changes within one batch: "
                        f"{field!r} differs at frame {original_key}"
                    )
                continue
            if field == "type" or value is None:
                continue
            if field == "fid":
                batch[field][row] = value
            else:
                destination = batch[field]
                array = np.asarray(value)
                if destination.shape[1:] != array.shape:
                    raise ValueError(
                        f"LMDB field {field!r} changes shape within one batch: "
                        f"expected {destination.shape[1:]}, got {array.shape} "
                        f"for frame {original_key}"
                    )
                result_dtype = np.result_type(destination.dtype, array.dtype)
                if result_dtype != destination.dtype:
                    promoted = np.empty(destination.shape, dtype=result_dtype)
                    promoted[:row] = destination[:row]
                    batch[field] = destination = promoted
                destination[row] = array

    assert batch is not None
    batch["sid"] = np.asarray([0], dtype=np.int64)
    return batch


_WORKER_LMDB_READERS: dict[
    str,
    tuple[lmdb.Environment, lmdb.Transaction],
] = {}


def _decode_lmdb_worker_chunk(
    lmdb_path: str,
    frame_format: str,
    config: LmdbDecodeConfig,
    original_keys: list[int],
) -> dict[str, Any]:
    """Decode one chunk using process-local LMDB state."""
    reader = _WORKER_LMDB_READERS.get(lmdb_path)
    if reader is None:
        environment = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        reader = (environment, environment.begin())
        _WORKER_LMDB_READERS[lmdb_path] = reader
    return decode_lmdb_batch(
        reader[1],
        original_keys,
        frame_format,
        config,
    )


def _merge_lmdb_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge ordered worker chunks into one contiguous batch."""
    if not chunks:
        raise ValueError("cannot merge an empty LMDB chunk list")
    if len(chunks) == 1:
        return chunks[0]

    first = chunks[0]
    expected_fields = frozenset(first)
    for chunk_index, chunk in enumerate(chunks[1:], start=1):
        chunk_fields = frozenset(chunk)
        if chunk_fields != expected_fields:
            raise ValueError(
                "LMDB worker chunks expose inconsistent fields: "
                f"chunk 0 has {sorted(expected_fields)}, while chunk "
                f"{chunk_index} has {sorted(chunk_fields)}"
            )

    merged: dict[str, Any] = {}
    for key, value in first.items():
        if key.startswith("find_"):
            for chunk_index, chunk in enumerate(chunks[1:], start=1):
                if not np.array_equal(value, chunk[key]):
                    raise ValueError(
                        "LMDB field availability changes across worker chunks: "
                        f"{key!r} differs in chunk {chunk_index}"
                    )
            merged[key] = value
        elif key == "sid" or value is None:
            merged[key] = value
        elif key == "fid":
            merged[key] = [frame_id for chunk in chunks for frame_id in chunk[key]]
        else:
            merged[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    return merged


@dataclass
class _LmdbPoolEntry:
    """Reference-counted process pool shared by data tasks in one rank."""

    executor: ProcessPoolExecutor
    users: int


_LMDB_POOL_LOCK = threading.Lock()
_LMDB_POOLS: dict[int, _LmdbPoolEntry] = {}


def _create_lmdb_executor(num_workers: int) -> ProcessPoolExecutor:
    """Create a CUDA-safe LMDB decoder process pool."""
    start_methods = multiprocessing.get_all_start_methods()
    start_method = "forkserver" if "forkserver" in start_methods else "spawn"
    return ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing.get_context(start_method),
    )


def _acquire_lmdb_executor(num_workers: int) -> ProcessPoolExecutor:
    """Acquire the process-wide pool for one worker count."""
    with _LMDB_POOL_LOCK:
        entry = _LMDB_POOLS.get(num_workers)
        if entry is None:
            entry = _LmdbPoolEntry(
                executor=_create_lmdb_executor(num_workers),
                users=0,
            )
            _LMDB_POOLS[num_workers] = entry
        entry.users += 1
        return entry.executor


def _release_lmdb_executor(num_workers: int) -> None:
    """Release one pool user and stop the pool after its final user."""
    executor: ProcessPoolExecutor | None = None
    with _LMDB_POOL_LOCK:
        entry = _LMDB_POOLS.get(num_workers)
        if entry is None:
            return
        entry.users -= 1
        if entry.users == 0:
            executor = entry.executor
            del _LMDB_POOLS[num_workers]
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


class LmdbBatchIterator:
    """Deterministic same-nloc batches with parallel decode and one prefetch.

    The sampler remains in the parent process. Worker tasks receive only the
    selected integer frame keys and compact decoder state, so the dataset-wide
    metadata is never duplicated. Data tasks in the same rank share one
    process pool while retaining independent samplers and pending batches.

    Parameters
    ----------
    reader
        LMDB reader that owns metadata and the synchronous transaction.
    sampler
        Finite iterator yielding same-nloc dataset-index batches.
    num_workers
        Decoder process count. Zero or one selects synchronous decoding.
    """

    def __init__(
        self,
        reader: "LmdbDataReader",
        sampler: Any,
        num_workers: int,
    ) -> None:
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}")
        self._reader = reader
        self._sampler = sampler
        self._iterator = iter(sampler)
        self._num_workers = num_workers
        self._executor: ProcessPoolExecutor | None = None
        self._pending: list[Future[dict[str, Any]]] | None = None
        self._deferred_indices: list[int] | None = None
        self._closed = False

    def __iter__(self) -> "LmdbBatchIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("cannot read from a closed LMDB batch iterator")

        if self._pending is not None:
            chunks = [future.result() for future in self._pending]
            batch = _merge_lmdb_chunks(chunks)
        elif self._deferred_indices is not None:
            indices = self._deferred_indices
            self._deferred_indices = None
            batch = self._reader.decode_batch(indices)
        else:
            indices = self._next_indices()
            if self._parallel_eligible(indices):
                chunks = [future.result() for future in self._submit(indices)]
                batch = _merge_lmdb_chunks(chunks)
            else:
                batch = self._reader.decode_batch(indices)

        self._schedule(self._next_indices())
        return batch

    def _next_indices(self) -> list[int]:
        """Return the next sampler batch and restart after exhaustion."""
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._sampler)
            return next(self._iterator)

    def _submit(self, indices: list[int]) -> list[Future[dict[str, Any]]]:
        """Submit one batch as balanced contiguous chunks."""
        original_keys = self._reader.original_keys(indices)
        workers = min(self._num_workers, len(original_keys))
        base_size, remainder = divmod(len(original_keys), workers)
        chunks: list[list[int]] = []
        start = 0
        for worker_index in range(workers):
            chunk_size = base_size + int(worker_index < remainder)
            stop = start + chunk_size
            chunks.append(original_keys[start:stop])
            start = stop
        if self._executor is None:
            self._executor = _acquire_lmdb_executor(self._num_workers)
        decode_config = self._reader.worker_decode_config()
        return [
            self._executor.submit(
                _decode_lmdb_worker_chunk,
                self._reader.lmdb_path,
                self._reader.frame_format,
                decode_config,
                chunk,
            )
            for chunk in chunks
        ]

    def _parallel_eligible(self, indices: list[int]) -> bool:
        """Whether process decoding amortizes its scheduling and IPC cost."""
        return self._num_workers > 1 and len(indices) >= self._num_workers

    def _schedule(self, indices: list[int]) -> None:
        """Schedule a large successor or defer a small one to the caller."""
        if self._parallel_eligible(indices):
            self._pending = self._submit(indices)
            self._deferred_indices = None
        else:
            self._pending = None
            self._deferred_indices = indices

    @property
    def started(self) -> bool:
        """Whether this iterator has acquired the shared process pool."""
        return self._executor is not None

    @property
    def closed(self) -> bool:
        """Whether this iterator has released its resources."""
        return self._closed

    def close(self) -> None:
        """Cancel prefetched work and release the shared decoder pool."""
        if self._closed:
            return
        self._closed = True
        if self._pending is not None:
            for future in self._pending:
                future.cancel()
            self._pending = None
        self._deferred_indices = None
        if self._executor is not None:
            self._executor = None
            _release_lmdb_executor(self._num_workers)


def is_lmdb(systems: str) -> bool:
    """Check if systems points to an LMDB dataset."""
    return systems.endswith(".lmdb") or Path(systems, "data.mdb").is_file()


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


def _parse_positive_rule(spec: str, prefix: str) -> int:
    """Parse the ``N`` in ``<prefix>N`` and require ``N > 0``.

    Rejects missing/non-integer/non-positive ``N`` up front so that
    misconfigurations (``"filter:"``, ``"filter:0"``, ``"max:-5"``) fail at
    construction time instead of silently producing an empty dataset or a
    batch_size=1 fallback downstream.
    """
    _, _, raw = spec.partition(":")
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(
            f"Unsupported batch_size {spec!r}. "
            f"Expected '{prefix}N' with N a positive integer."
        ) from None
    if n <= 0:
        raise ValueError(
            f"Unsupported batch_size {spec!r}: N must be a positive integer, got {n}."
        )
    return n


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
        Batch size rule used to derive per-nloc batch sizes. Supports:

        - ``int``: fixed, identical batch size for every nloc group.
        - ``"auto"`` / ``"auto:N"``: ``ceil(N / nloc)`` per nloc group
          (``N=32`` for bare ``"auto"``). Acts as a *lower* bound —
          each batch has at least ``N`` atoms, but may exceed ``N``
          by up to ``nloc - 1``.
        - ``"max:N"``: ``max(1, floor(N / nloc))`` per nloc group.
          Acts as an *upper* bound for groups with ``nloc <= N``
          (batch has at most ``N`` atoms). For groups with
          ``nloc > N`` the ``max(1, ...)`` floor kicks in: ``bsi=1``
          and a single-frame batch still carries ``nloc`` atoms,
          which exceeds ``N``.
        - ``"filter:N"``: same per-nloc formula as ``"max:N"`` **and**
          drops every frame whose ``nloc > N`` from the dataset. By
          construction every retained batch has at most ``N`` atoms.
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
        self.lmdb_path = str(Path(lmdb_path).resolve())
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

        # The parent transaction serves synchronous reads. Decoder workers open
        # independent process-local environments and transactions.
        self._txn = self._env.begin()
        self._closed = False

        # Scan per-frame nloc only when needed for same-nloc batching.
        # For mixed_batch=True, skip the scan entirely (future: padding handles it).
        # ``orig_frame_nlocs`` / ``orig_frame_system_ids`` are indexed by the
        # *original* LMDB frame index. After a potential ``filter:N`` drop we
        # rebuild ``self._frame_nlocs`` / ``self._frame_system_ids`` so they
        # are parallel arrays over the *dataset* index space (0..len(self));
        # the dataset-to-original mapping lives in ``self._retained_keys``.
        if not mixed_batch:
            # Fast path: use pre-computed frame_nlocs from metadata if available.
            # Falls back to scanning each frame's atom_types shape (~10 us/frame).
            meta_nlocs = meta.get("frame_nlocs")
            if meta_nlocs is not None:
                orig_frame_nlocs = [int(n) for n in meta_nlocs]
            else:
                orig_frame_nlocs = _scan_frame_nlocs(
                    self._env, self.nframes, self._frame_fmt, self._natoms
                )
        else:
            orig_frame_nlocs = []

        # Parse frame_system_ids for auto_prob support. ``_nsystems`` must stay
        # at ``max(original_sid) + 1`` even after filter:N so that user-facing
        # auto_prob block slicing (e.g. ``prob_sys_size;0:284:0.5;284:842:0.5``)
        # keeps its meaning across filter thresholds.
        meta_sys_ids = meta.get("frame_system_ids")
        if meta_sys_ids is not None:
            orig_frame_system_ids: list[int] | None = [int(s) for s in meta_sys_ids]
            self._nsystems = max(orig_frame_system_ids) + 1
        else:
            orig_frame_system_ids = None
            self._nsystems = 1

        # Parse batch_size spec. ``auto_rule`` and ``max_rule`` are mutually
        # exclusive; ``filter_rule`` implies ``max_rule`` plus dropping frames
        # whose nloc exceeds the threshold.
        self._auto_rule: int | None = None
        self._max_rule: int | None = None
        self._filter_rule: int | None = None
        if isinstance(batch_size, str):
            if batch_size == "auto":
                self._auto_rule = 32
            elif batch_size.startswith("auto:"):
                self._auto_rule = _parse_positive_rule(batch_size, "auto:")
            elif batch_size.startswith("max:"):
                self._max_rule = _parse_positive_rule(batch_size, "max:")
            elif batch_size.startswith("filter:"):
                self._filter_rule = _parse_positive_rule(batch_size, "filter:")
                self._max_rule = self._filter_rule
            else:
                raise ValueError(
                    f"Unsupported batch_size {batch_size!r}. "
                    "Expected int, 'auto', 'auto:N', 'max:N', or 'filter:N'."
                )

        # ``filter:N`` needs per-frame nloc to drop oversized frames; the
        # ``mixed_batch=True`` fast path skips the nloc scan entirely, so the
        # two options are incompatible. Fail fast rather than silently
        # retaining every frame and breaking the documented contract.
        if self._filter_rule is not None and mixed_batch:
            raise ValueError(
                "batch_size='filter:N' is incompatible with mixed_batch=True: "
                "per-frame nloc is unavailable in the mixed-batch fast path. "
                "Use mixed_batch=False, or switch to 'max:N' / a fixed int."
            )

        # Determine which original-index frames survive the filter. Without
        # ``filter:N`` every frame is retained.
        if self._filter_rule is not None:
            retained_keys = [
                i for i, n in enumerate(orig_frame_nlocs) if n <= self._filter_rule
            ]
            n_dropped = self.nframes - len(retained_keys)
            if n_dropped > 0:
                log.info(
                    f"LMDB filter:{self._filter_rule} drops {n_dropped}/"
                    f"{self.nframes} frames with nloc > {self._filter_rule} "
                    f"({self.lmdb_path})."
                )
        else:
            retained_keys = list(range(self.nframes))

        # Dataset-index → original LMDB frame key. ``__getitem__`` looks up
        # this table so that ``reader[i]`` is a valid LMDB read for every
        # ``0 <= i < len(reader)``, no matter how many frames were filtered.
        self._retained_keys: list[int] = retained_keys

        # Re-key _frame_nlocs / _frame_system_ids into the dataset-index
        # space so that every downstream consumer (nloc_groups, system_groups,
        # SameNlocBatchSampler, _expand_indices_by_blocks) operates in a
        # single, self-consistent indexing scheme.
        if not mixed_batch:
            self._frame_nlocs = [orig_frame_nlocs[k] for k in retained_keys]
        else:
            self._frame_nlocs = []

        if orig_frame_system_ids is not None:
            self._frame_system_ids: list[int] | None = [
                orig_frame_system_ids[k] for k in retained_keys
            ]
        else:
            self._frame_system_ids = None

        # Group retained frames by nloc using dataset indices (0..len-1).
        if not mixed_batch:
            self._nloc_groups: dict[int, list[int]] = {}
            for ds_idx, nloc in enumerate(self._frame_nlocs):
                self._nloc_groups.setdefault(nloc, []).append(ds_idx)
        else:
            self._nloc_groups = {}

        # Group retained frames by original system id; the sid numbering is
        # preserved (no compression) so user-facing auto_prob slices stay
        # meaningful across filter thresholds. Fully-dropped systems appear
        # as zero-frame entries in ``_system_nframes``.
        if self._frame_system_ids is not None:
            self._system_groups: dict[int, list[int]] = {}
            for ds_idx, sid in enumerate(self._frame_system_ids):
                self._system_groups.setdefault(sid, []).append(ds_idx)
            self._system_nframes: list[int] = [
                len(self._system_groups.get(i, [])) for i in range(self._nsystems)
            ]
        else:
            self._system_groups = {0: list(range(len(retained_keys)))}
            self._system_nframes = [len(retained_keys)]

        # nframes now reflects retained frames; __len__ returns this and the
        # valid index domain for __getitem__ is [0, self.nframes).
        self.nframes = len(retained_keys)

        # Default batch_size used only by the index/total_batch estimate. The
        # sampler always goes through get_batch_size_for_nloc for real batches.
        if self._auto_rule is not None:
            self.batch_size = _compute_batch_size(self._natoms, self._auto_rule)
        elif self._max_rule is not None:
            self.batch_size = max(1, self._max_rule // max(self._natoms, 1))
        else:
            self.batch_size = int(batch_size)

        # Data requirements tracking
        self._data_requirements: dict[str, DataRequirementItem] = {}
        self._data_requirements_frozen = False
        self._decode_config = LmdbDecodeConfig(
            ntypes=self._ntypes,
            natoms=self._natoms,
            type_remap=self._type_remap,
            data_requirements=self._data_requirements,
        )

    def _resolve_dtype(self, key: str) -> np.dtype:
        """Resolve the target numpy dtype for a given key.

        Priority: DataRequirementItem.dtype > DataRequirementItem.high_prec >
        built-in defaults (energy=high, others=normal).
        """
        return _resolve_frame_dtype(self._decode_config, key)

    def __del__(self) -> None:
        """Release the parent LMDB transaction and environment."""
        self.close()

    def close(self) -> None:
        """Release parent-process LMDB resources idempotently."""
        if getattr(self, "_closed", False):
            return
        transaction = getattr(self, "_txn", None)
        if transaction is not None:
            transaction.abort()
            self._txn = None
        environment = getattr(self, "_env", None)
        if environment is not None:
            self._env = None
            _close_lmdb(self.lmdb_path)
        self._closed = True

    def _transaction(self) -> lmdb.Transaction:
        """Return the active parent transaction or fail after closure."""
        transaction = self._txn
        if transaction is None:
            raise RuntimeError("cannot read from a closed LMDB reader")
        return transaction

    def get_batch_size_for_nloc(self, nloc: int) -> int:
        """Return the per-nloc batch size for the configured rule.

        - ``auto`` / ``auto:N``: ``ceil(N / nloc)`` — may overshoot the
          atom budget by up to ``nloc - 1`` atoms.
        - ``max:N``: ``max(1, floor(N / nloc))``. Acts as an upper bound
          for groups with ``nloc <= N`` (batch has at most ``N`` atoms).
          For groups with ``nloc > N`` the floor clamps to 1 and the
          single-frame batch still carries ``nloc`` atoms, exceeding ``N``.
        - ``filter:N``: same per-nloc formula as ``max:N``; by
          construction every retained group satisfies ``nloc <= N`` so
          no overshoot occurs.
        - fixed int: the same value for every nloc group.
        """
        if self._auto_rule is not None:
            return _compute_batch_size(nloc, self._auto_rule)
        if self._max_rule is not None:
            return max(1, self._max_rule // max(nloc, 1))
        return self.batch_size

    def __len__(self) -> int:
        return self.nframes

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Read frame from LMDB, decode, remap keys, return dict of numpy arrays.

        ``index`` is a dataset-level index in ``[0, len(self))``. Under
        ``filter:N`` the LMDB key space may have gaps (dropped frames), so
        we translate through ``self._retained_keys`` before hitting LMDB.
        """
        self._data_requirements_frozen = True
        if index < 0 or index >= self.nframes:
            raise IndexError(f"dataset index {index} out of range [0, {self.nframes})")
        original_key = self._retained_keys[index]
        key = format(original_key, self._frame_fmt).encode()
        raw = self._transaction().get(key)
        if raw is None:
            raise IndexError(
                f"Frame {original_key} not found in LMDB (dataset index {index})"
            )
        return decode_lmdb_frame(
            raw,
            original_key,
            self._decode_config,
            copy_arrays=True,
        )

    def original_keys(self, indices: Sequence[int]) -> list[int]:
        """Translate dataset indices to original integer LMDB keys."""
        keys: list[int] = []
        for index in indices:
            index = int(index)
            if index < 0 or index >= self.nframes:
                raise IndexError(
                    f"dataset index {index} out of range [0, {self.nframes})"
                )
            keys.append(self._retained_keys[index])
        return keys

    def decode_batch(self, indices: Sequence[int]) -> dict[str, Any]:
        """Decode a same-nloc batch directly into contiguous NumPy arrays."""
        self._data_requirements_frozen = True
        return decode_lmdb_batch(
            self._transaction(),
            self.original_keys(indices),
            self._frame_fmt,
            self._decode_config,
        )

    @property
    def frame_format(self) -> str:
        """Format specification used for integer LMDB frame keys."""
        return self._frame_fmt

    def worker_decode_config(self) -> LmdbDecodeConfig:
        """Freeze and return decoder state for worker serialization."""
        self._data_requirements_frozen = True
        return self._decode_config

    @property
    def closed(self) -> bool:
        """Whether parent-process LMDB resources have been released."""
        return self._closed

    # --- Data requirement interface ---

    def add_data_requirement(self, data_requirement: list[DataRequirementItem]) -> None:
        """Register expected keys; missing keys get default fill + find_key=0.0."""
        if self._data_requirements_frozen:
            raise RuntimeError(
                "LMDB data requirements must be registered before reading any frame"
            )
        for item in data_requirement:
            self._data_requirements[item["key"]] = item

    @property
    def data_requirements(self) -> list[DataRequirementItem]:
        """Registered data requirements in insertion order."""
        return list(self._data_requirements.values())

    def print_summary(self, name: str, prob: Any) -> None:
        """Print basic dataset info."""
        n_groups = len(self._nloc_groups)
        if self._auto_rule is not None:
            bs_str = f"auto:{self._auto_rule}"
        elif self._filter_rule is not None:
            bs_str = f"filter:{self._filter_rule}"
        elif self._max_rule is not None:
            bs_str = f"max:{self._max_rule}"
        else:
            bs_str = str(self.batch_size)

        log.info(
            f"LMDB {name}: {self.lmdb_path}, "
            f"{self.nframes} frames, {n_groups} nloc groups, "
            f"batch_size={bs_str}, "
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
        return [self.total_batch]

    @property
    def total_batch(self) -> int:
        if self.mixed_batch:
            return math.ceil(self.nframes / self.batch_size) if self.nframes else 0
        total = 0
        for nloc, indices in self._nloc_groups.items():
            bs = self.get_batch_size_for_nloc(nloc)
            total += (len(indices) + bs - 1) // bs
        return total

    @property
    def batch_sizes(self) -> list[int]:
        return [self.batch_size]

    @property
    def mixed_type(self) -> bool:
        """LMDB datasets are always mixed_type (frames may have different compositions)."""
        return True

    @property
    def type_map(self) -> list[str]:
        """Model-side type map used when constructing the reader."""
        return self._type_map

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


def collate_lmdb_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of per-frame dicts into a single batch dict.

    Backend-agnostic via ``array_api_compat``: works for numpy, torch, jax,
    etc. The array library is inferred from the first frame's ``coord``.

    Conventions match :func:`deepmd.dpmodel.utils.batch.normalize_batch`:
    ``find_*`` flags are taken from the first frame (constant within a
    batch); ``fid`` is collected as a list; ``type`` is dropped (callers
    should already use ``atype``); other arrays are stacked along axis 0.
    A ``sid`` placeholder is appended.
    """
    import array_api_compat

    if not frames:
        raise ValueError("collate_lmdb_frames requires at least one frame")

    xp = array_api_compat.array_namespace(frames[0]["coord"])
    dev = array_api_compat.device(frames[0]["coord"])
    out: dict[str, Any] = {}
    for key in frames[0]:
        if key.startswith("find_"):
            out[key] = frames[0][key]
        elif key == "fid":
            out[key] = [f[key] for f in frames]
        elif key == "type":
            continue
        elif frames[0][key] is None:
            out[key] = None
        else:
            out[key] = xp.stack([f[key] for f in frames])
    out["sid"] = xp.asarray([0], dtype=xp.int64, device=dev)
    return out


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

    # Drop blocks that retain zero frames (can happen when ``filter:N``
    # eliminates every system in a block). prob_sys_size_ext's per-block
    # ``nbatch_block / sum(nbatch_block)`` would otherwise propagate NaN
    # when the whole block sums to zero. An all-zero dataset yields no
    # targets at all.
    nonempty = [
        (stt, end, weight)
        for stt, end, weight in blocks
        if sum(system_nframes[stt:end]) > 0
    ]
    if not nonempty:
        log.info(
            "compute_block_targets: all blocks are empty in "
            f"{auto_prob_style!r}; dataset has no retained frames."
        )
        return []
    if len(nonempty) < len(blocks):
        # Rewriting auto_prob_style silently re-normalises the remaining
        # weights so they sum to 1.0 — e.g. ``0:3:0.8;3:10:0.2`` with block
        # ``0:3`` empty becomes effectively weight 1.0 on block ``3:10``.
        # Surface this reweighting so operators can correlate it with the
        # preceding ``filter:N`` log line.
        dropped = [
            f"{stt}:{end}:{weight}"
            for (stt, end, weight) in blocks
            if (stt, end, weight) not in nonempty
        ]
        log.info(
            "compute_block_targets: dropping empty blocks (all systems have "
            f"0 frames, likely after filter:N): {dropped}. Remaining block "
            "weights will be renormalised to sum to 1.0."
        )
        auto_prob_style = "prob_sys_size;" + ";".join(
            f"{stt}:{end}:{weight}" for stt, end, weight in nonempty
        )
        blocks = nonempty

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
        self._total_batches = len(
            SameNlocBatchSampler(
                self._reader,
                shuffle=False,
                block_targets=self._block_targets,
            )
        )

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
        return max(
            0,
            (self._total_batches + self._world_size - 1 - self._rank)
            // self._world_size,
        )

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size


def make_neighbor_stat_data(
    lmdb_path: str,
    type_map: list[str] | None,
    max_frames: int = 2000,
) -> Any:
    """Create a duck-typed DeepmdDataSystem-like object for neighbor stat from LMDB.

    Samples up to *max_frames* frames, groups them by nloc, and returns an
    object whose attributes satisfy the interface expected by
    ``NeighborStat.iterator()`` and ``UpdateSel.get_nbor_stat()``.
    """
    from types import (
        SimpleNamespace,
    )

    reader = LmdbDataReader(lmdb_path, type_map=type_map)
    nframes = len(reader)
    rng = np.random.RandomState(42)
    if nframes > max_frames:
        indices = np.sort(rng.choice(nframes, max_frames, replace=False))
    else:
        indices = np.arange(nframes, dtype=np.int64)

    # Read sampled frames, group by nloc
    nloc_frames: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]] = {}
    for idx in indices:
        frame = reader[int(idx)]
        atype = frame["atype"]
        nloc = len(atype)
        nloc_frames.setdefault(nloc, []).append(
            (frame["coord"], atype, frame.get("box"))
        )

    # Build per-nloc data_system proxies
    data_systems = []
    system_dirs: list[str] = []
    for nloc, frames in nloc_frames.items():
        coords = np.stack([c.reshape(nloc * 3) for c, _, _ in frames])
        types = np.stack([a.reshape(nloc) for _, a, _ in frames])
        has_box = frames[0][2] is not None
        boxes = np.stack([b.reshape(9) for _, _, b in frames]) if has_box else None
        set_data = {"coord": coords, "type": types, "box": boxes}
        label = f"lmdb:{nloc}atoms"
        proxy = SimpleNamespace(
            dirs=[label],
            pbc=has_box,
            mixed_type=True,
            get_natoms=lambda _nloc=nloc: _nloc,
            _load_set=lambda _d, _sd=set_data: _sd,
        )
        data_systems.append(proxy)
        system_dirs.append(label)

    ntypes = len(type_map) if type_map else reader._ntypes
    return SimpleNamespace(
        system_dirs=system_dirs,
        data_systems=data_systems,
        get_batch=lambda: None,
        get_ntypes=lambda: ntypes,
        mixed_type=True,
    )


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
                        frame["atype"] = _remap_atom_types(
                            frame["atype"].reshape(-1), self._type_remap
                        )
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

    def __del__(self) -> None:
        """Release the LMDB environment ref-count on garbage collection."""
        path = getattr(self, "lmdb_path", None)
        if path is not None:
            _close_lmdb(path)

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

    def add_data_requirement(self, data_requirement: list[DataRequirementItem]) -> None:
        """Register expected keys from ``DataRequirementItem`` objects.

        Mirrors :meth:`LmdbDataReader.add_data_requirement` so the same
        requirement list can be forwarded to both the training reader and
        the full-validation test data.
        """
        for item in data_requirement:
            self.add(
                item["key"],
                ndof=item["ndof"],
                atomic=item["atomic"],
                must=item["must"],
                high_prec=item["high_prec"],
                repeat=item["repeat"],
                default=item["default"],
                dtype=item["dtype"],
            )

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

        # Dynamically discover all data keys from the first frame, plus
        # any registered requirements.  Structural keys (coord, box, type)
        # are excluded — they are already handled above.
        _structural_keys = frozenset({"coord", "box", "atype"})
        all_keys: dict[str, dict[str, Any]] = {}
        if frames:
            for fk in frames[0]:
                if fk in _structural_keys or fk.startswith("find_"):
                    continue
                if fk not in all_keys:
                    all_keys[fk] = {"ndof": None, "atomic": False, "default": 0.0}
        for key, req in self._requirements.items():
            all_keys[key] = req

        for key, req_info in all_keys.items():
            has_key = any(
                key in f and isinstance(f.get(key), np.ndarray) for f in frames
            )
            result[f"find_{key}"] = 1.0 if has_key else 0.0

            # Get repeat factor from registered requirements
            repeat = 1
            if key in self._requirements:
                repeat = self._requirements[key].get("repeat", 1)

            if has_key:
                arrays = []
                for frame in frames:
                    val = frame.get(key)
                    if isinstance(val, np.ndarray):
                        arr = val.astype(self._resolve_dtype(key)).ravel()
                        if repeat != 1:
                            arr = np.repeat(arr, repeat)
                        arrays.append(arr)
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
                            size = ref.size * repeat if repeat != 1 else ref.size
                            arrays.append(
                                np.zeros(size, dtype=self._resolve_dtype(key))
                            )
                        else:
                            arrays.append(np.zeros(1, dtype=self._resolve_dtype(key)))
                result[key] = np.stack(arrays)
            elif key in self._requirements:
                ndof = self._requirements[key]["ndof"]
                atomic = self._requirements[key]["atomic"]
                default = self._requirements[key]["default"]
                if atomic:
                    shape = (nframes, natoms * ndof * repeat)
                else:
                    shape = (nframes, ndof * repeat)
                result[key] = np.full(shape, default, dtype=self._resolve_dtype(key))

        return result


class LmdbTestDataNlocView:
    """Thin wrapper exposing a fixed-``nloc`` view of :class:`LmdbTestData`.

    The underlying :class:`LmdbTestData` groups frames by atom count. This
    view fixes one ``nloc`` group, so ``get_test()`` returns only the frames
    with that atom count and all other attributes (``pbc``, ``mixed_type``,
    …) are forwarded to the underlying object. It lets downstream consumers
    that expect a ``DeepmdData``-style system (one fixed natoms per
    ``get_test()``) work on mixed-nloc LMDB datasets without changes.
    """

    def __init__(self, lmdb_test_data: "LmdbTestData", nloc: int) -> None:
        self._inner = lmdb_test_data
        self._nloc = nloc

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def get_test(self) -> dict[str, Any]:
        return self._inner.get_test(nloc=self._nloc)


def _copy_lmdb_source(
    src_path: str,
    dst_env: lmdb.Environment,
    dst_format: str,
    frame_idx: int,
    frame_nlocs: list[int],
    frame_system_ids: list[int],
    system_id_offset: int,
) -> tuple[int, dict, list[str] | None, int]:
    """Copy one source under a ref-counted environment lease."""
    src_env = _open_lmdb(src_path)
    try:
        with src_env.begin() as transaction:
            metadata = _read_metadata(transaction)
        nframes, src_format, natoms_per_type = _parse_metadata(metadata)
        fallback_natoms = sum(natoms_per_type)
        source_nlocs = metadata.get("frame_nlocs")
        source_system_ids = metadata.get("frame_system_ids")

        with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
            for source_index in range(nframes):
                source_key = format(source_index, src_format).encode()
                raw = src_txn.get(source_key)
                if raw is None:
                    continue
                destination_key = format(frame_idx, dst_format).encode()
                dst_txn.put(destination_key, raw)

                if source_nlocs is not None:
                    frame_nlocs.append(int(source_nlocs[source_index]))
                else:
                    frame_raw = msgpack.unpackb(raw, raw=False)
                    atype_raw = frame_raw.get("atom_types")
                    if isinstance(atype_raw, dict):
                        shape = atype_raw.get("shape") or atype_raw.get(b"shape")
                        frame_nlocs.append(int(shape[0]) if shape else fallback_natoms)
                    else:
                        frame_nlocs.append(fallback_natoms)

                if source_system_ids is not None and source_index < len(
                    source_system_ids
                ):
                    frame_system_ids.append(
                        int(source_system_ids[source_index]) + system_id_offset
                    )
                else:
                    frame_system_ids.append(system_id_offset)
                frame_idx += 1

        if source_system_ids is not None and len(source_system_ids) > 0:
            system_id_offset += max(int(value) for value in source_system_ids) + 1
        else:
            system_id_offset += 1
        return (
            frame_idx,
            metadata.get("system_info", {}),
            metadata.get("type_map"),
            system_id_offset,
        )
    finally:
        _close_lmdb(src_path)


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
    try:
        for src_path in src_paths:
            (
                frame_idx,
                source_system_info,
                source_type_map,
                sys_id_offset,
            ) = _copy_lmdb_source(
                src_path,
                dst_env,
                fmt,
                frame_idx,
                frame_nlocs,
                frame_system_ids,
                sys_id_offset,
            )
            if first_system_info is None:
                first_system_info = source_system_info
            if first_type_map is None:
                first_type_map = source_type_map

        merged_meta = {
            "nframes": frame_idx,
            "frame_idx_fmt": fmt,
            "system_info": first_system_info or {},
            "frame_nlocs": frame_nlocs,
            "frame_system_ids": frame_system_ids,
        }
        if first_type_map is not None:
            merged_meta["type_map"] = first_type_map
        with dst_env.begin(write=True) as transaction:
            transaction.put(
                b"__metadata__",
                msgpack.packb(merged_meta, use_bin_type=True),
            )
    finally:
        dst_env.close()

    nloc_counts: dict[int, int] = {}
    for n in frame_nlocs:
        nloc_counts[n] = nloc_counts.get(n, 0) + 1
    log.info(
        f"Merged {len(src_paths)} LMDBs → {dst_path}: "
        f"{frame_idx} frames, nloc groups: {dict(sorted(nloc_counts.items()))}"
    )
    return dst_path
