# SPDX-License-Identifier: LGPL-3.0-or-later
"""Framework-agnostic LMDB data utilities for DeePMD-kit.

All code here is pure Python/NumPy/lmdb/msgpack — no framework dependency.
Backend-specific wrappers (PyTorch Dataset, JAX, etc.) import from here.
"""

import dataclasses
import logging
import math
import multiprocessing
import os
import signal
import threading
import time
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
)
from concurrent.futures import wait as futures_wait
from concurrent.futures.process import (
    BrokenProcessPool,
)
from dataclasses import (
    dataclass,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
    cast,
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
from deepmd.utils import random as dp_random
from deepmd.utils.data import (
    DataRequirementItem,
    DataRequirementSourcePolicy,
)

log = logging.getLogger(__name__)


def _is_local_rank_zero() -> bool:
    """Whether this process owns node-local operational logging."""
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


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

# Frames probed to decide whether availability-sensitive labels partition a
# dataset. Requirements are registered after reader construction, so the probe
# is deferred until a consumer first requests sampling groups.
#
# The sample makes the partition probabilistic, and a dataset whose odd frame
# out falls between two probes is grouped as if it were uniform. What such a
# frame costs is bounded: the batch it lands in reports the disputed label
# unavailable, so at most ``batch_size - 1`` frames lose that label for that
# batch, and a reshuffle moves it elsewhere next epoch. No default fill is
# ever mistaken for a real label, which is the property that matters -- see
# :func:`_batch_find_flags`.
#
# Raising this bound trades startup reads for a tighter guarantee, but never
# reaches certainty. A dataset that really does interleave labels is better
# fixed where it is written: recording a per-frame availability id in
# ``__metadata__``, alongside the ``frame_nlocs`` and ``frame_system_ids``
# already there, would make the partition exact and free.
_AVAILABILITY_PROBE_FRAMES = 256
_AVAILABILITY_SCAN_CHUNK = 4096
_AVAILABILITY_LOG_SECONDS = 60.0

# Atom type written into the padded slots of a mixed-nloc batch. A phantom
# atom occupies a tensor slot but no physical site: the neighbor list gives it
# no neighbors, the atomic model zeroes its output, and the loss masks it out.
PHANTOM_ATOM_TYPE = -1

# Fields whose leading axis is the atom axis, and which a mixed-nloc batch must
# therefore pad to the batch-wide atom count. Membership cannot be inferred
# from array shapes: a frame with ``nloc == 9`` makes ``virial`` (shape ``(9,)``)
# indistinguishable from a per-atom field, and ``nloc == 2`` does the same for
# a two-component ``fparam``. The registered data requirements carry the
# authoritative ``atomic`` flag; these two sets cover the fields that exist
# without one.
_STRUCTURAL_PER_ATOM_KEYS = frozenset({"coord", "atype"})
_OPTIONAL_PER_ATOM_KEYS = frozenset({"aparam", "spin"})

# Frame-level fields that a decoded frame may carry without a registered
# requirement. They anchor the shape-based fallback below, which classifies any
# remaining unrecognized field by comparing its leading axis to the frame's
# atom count.
_FRAME_LEVEL_KEYS = frozenset(
    {
        "box",
        "energy",
        "virial",
        "fparam",
        "charge_spin",
        "natoms",
        "real_natoms_vec",
        "min_pair_dist",
    }
)

# Process-level cache: python-lmdb does not allow opening the same path twice
# in one process.  We ref-count so the Environment is closed (and freed from
# the cache) once every reader that shares it is garbage-collected.
_ENV_CACHE: dict[str, tuple[lmdb.Environment, int]] = {}


def _open_lmdb(path: str, *, sequential: bool = False) -> lmdb.Environment:
    """Open (or reuse) a readonly LMDB environment with reference counting.

    The python-lmdb binding raises ``lmdb.Error`` if the same path is opened
    more than once in a single process.  We cache by resolved absolute path
    and bump a reference count.  Call :func:`_close_lmdb` when done to
    decrement the count; when it reaches zero the environment is closed and
    removed from the cache.

    Parameters
    ----------
    path : str
        Path to the LMDB directory.

    sequential : bool, optional
        Whether the caller reads frames in ascending key order. Kernel
        readahead is then left on, which turns the many small faults such a
        walk would make into few large reads. Measured against NFS: 45 000
        frames/s with readahead against 1 300 without. A caller reading in
        shuffled order leaves this false, where readahead instead fetches
        neighbours it will not use and costs about 12% of the throughput.

        One path admits one environment, so a path already open keeps the
        setting its first caller asked for.

    Returns
    -------
    lmdb.Environment
        The shared read-only environment for ``path``.
    """
    resolved = str(Path(path).resolve())
    entry = _ENV_CACHE.get(resolved)
    if entry is not None:
        env, refcount = entry
        _ENV_CACHE[resolved] = (env, refcount + 1)
        return env
    env = lmdb.open(
        path, readonly=True, lock=False, readahead=sequential, meminit=False
    )
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


def _read_metadata_of(path: str) -> dict:
    """Read ``__metadata__`` through an environment that keeps readahead on.

    The value is a single contiguous run of overflow pages -- hundreds of
    megabytes once the writer records a per-frame table. ``MDB_NORDAHEAD``,
    which a reader of shuffled frames asks :func:`_open_lmdb` for, advises
    the kernel ``MADV_RANDOM`` over the whole map, and that turns this
    sequential run into one synchronous fault per 4 KiB page. Measured
    against NFS on a 665 MiB table: 379 s without readahead, 8.2 s with it.

    The environment is closed before the caller opens its own, because
    python-lmdb refuses a second open of one path. A path already open for
    frame serving is read through that environment instead, its pages being
    warm by then in the only case that matters.

    Parameters
    ----------
    path : str
        Path to the LMDB directory.

    Returns
    -------
    dict
        The decoded metadata mapping.
    """
    entry = _ENV_CACHE.get(str(Path(path).resolve()))
    if entry is not None:
        with entry[0].begin() as transaction:
            return _read_metadata(transaction)
    env = lmdb.open(path, readonly=True, lock=False, readahead=True, meminit=False)
    try:
        with env.begin() as transaction:
            return _read_metadata(transaction)
    finally:
        env.close()


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
    """Remap LMDB key names to the canonical in-memory DeePMD convention."""
    out = {}
    for k, v in frame.items():
        key = _KEY_REMAP.get(k, k)
        if key.startswith("find_atomic_"):
            key = "find_atom_" + key.removeprefix("find_atomic_")
        elif key.startswith("atomic_"):
            key = "atom_" + key.removeprefix("atomic_")
        out[key] = v
    return out


def _requirement_value(requirement: Any, key: str, default: Any) -> Any:
    """Read one requirement attribute from object or dictionary form."""
    if isinstance(requirement, dict):
        return requirement.get(key, default)
    return getattr(requirement, key, default)


def _requirement_source_policy(requirement: Any) -> DataRequirementSourcePolicy:
    """Return the normalized source-presence policy of one requirement."""
    policy = str(_requirement_value(requirement, "source_policy", "tracked"))
    if policy not in {"tracked", "default", "derived"}:
        raise ValueError(f"Unsupported data requirement source policy {policy!r}")
    return cast("DataRequirementSourcePolicy", policy)


def _requirement_is_mandatory(requirement: Any) -> bool:
    """Whether unavailable source data violates one requirement."""
    return bool(_requirement_value(requirement, "must", False))


def _availability_requirement_keys(
    requirements: dict[str, Any],
) -> tuple[str, ...]:
    """Return requirements whose source presence affects consumer behavior.

    Only optional ``tracked`` fields participate. Mandatory fields fail at
    decode time rather than forming a default-filled group. ``default`` and
    ``derived`` fields have valid per-frame fallbacks and therefore combine
    safely regardless of source presence.

    Parameters
    ----------
    requirements : dict[str, Any]
        Data requirements keyed by normalized field name.

    Returns
    -------
    tuple[str, ...]
        Sorted availability-sensitive field names.
    """
    return tuple(
        sorted(
            key
            for key, requirement in requirements.items()
            if _requirement_source_policy(requirement) == "tracked"
            and not _requirement_is_mandatory(requirement)
        )
    )


def _raw_frame_availability(
    raw: bytes,
    key_bits: dict[str, int],
) -> int:
    """Return one undecoded frame's selected availability bit mask.

    A selected field is available when the frame carries it unless an
    explicit ``find_*`` flag says otherwise. Fields outside
    ``key_bits`` are deliberately ignored, so auxiliary raw data cannot alter
    a training run's batch partition.

    Only the msgpack map is walked; the arrays it describes stay encoded.

    Parameters
    ----------
    raw : bytes
        The msgpack payload of one LMDB frame.
    key_bits : dict[str, int]
        Bit assigned to each normalized availability-sensitive field.

    Returns
    -------
    int
        Availability mask in ``key_bits`` bit positions.
    """
    frame = msgpack.unpackb(raw, raw=False)
    present = 0
    explicit_known = 0
    explicit_true = 0
    for key, value in frame.items():
        name = _KEY_REMAP.get(key, key)
        if name.startswith("find_"):
            label = name.removeprefix("find_")
            bit = key_bits.get(label)
            if bit is not None:
                explicit_known |= bit
                if float(np.asarray(_decode_value(value)).item()) != 0.0:
                    explicit_true |= bit
        else:
            bit = key_bits.get(name)
            if bit is not None:
                present |= bit
    return present & (~explicit_known | explicit_true)


def _evenly_spaced(keys: Sequence[int], count: int) -> list[int]:
    """Pick up to ``count`` entries spread evenly over ``keys``.

    Spreading rather than truncating matters for a dataset assembled by
    concatenating sources: the tail of the key space is as likely to hold
    the odd one out as the head.

    Parameters
    ----------
    keys : Sequence[int]
        Integer LMDB frame keys to sample from.
    count : int
        Upper bound on the number of keys returned.

    Returns
    -------
    list[int]
        The sampled keys, in ascending position order.
    """
    total = len(keys)
    if total <= count:
        return [int(key) for key in keys]
    return [int(keys[total * position // count]) for position in range(count)]


def _probe_uniform_availability(
    transaction: lmdb.Transaction,
    keys: Iterable[int],
    frame_format: str,
    availability_keys: Sequence[str],
) -> bool:
    """Whether the sampled frames all supply the same labels.

    Parameters
    ----------
    transaction : lmdb.Transaction
        Open read transaction on the LMDB environment.
    keys : Iterable[int]
        Integer LMDB frame keys to probe, already reduced to a bounded
        sample by :func:`_evenly_spaced`.
    frame_format : str
        Format specification for integer LMDB frame keys.
    availability_keys : Sequence[str]
        Normalized field names used by the active consumer.

    Returns
    -------
    bool
        True when every probed frame supplies the same labels. A dataset
        larger than the sample may still be mixed, in which case the batch
        decode reports the disputed label unavailable rather than mixing it.
    """
    key_bits = {key: 1 << position for position, key in enumerate(availability_keys)}
    reference: int | None = None
    for key in keys:
        raw = transaction.get(format(int(key), frame_format).encode())
        if raw is None:
            continue
        availability = _raw_frame_availability(raw, key_bits)
        if reference is None:
            reference = availability
        elif availability != reference:
            return False
    return True


@dataclass(frozen=True)
class _AvailabilityIndex:
    """Compact availability signature ID aligned with a frame index domain."""

    ids: np.ndarray
    signature_count: int

    def groups(
        self,
        indices: np.ndarray,
        *,
        positions: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Partition indices by cached signature IDs without source reads."""
        index_array = np.asarray(indices)
        signature_ids = (
            self.ids
            if positions is None
            else self.ids[np.asarray(positions, dtype=np.int64)]
        )
        if len(index_array) != len(signature_ids):
            raise ValueError(
                "availability index and frame indices have different lengths: "
                f"{len(signature_ids)} != {len(index_array)}"
            )
        if len(index_array) == 0:
            return []
        if self.signature_count == 1:
            return [index_array]

        if self.signature_count <= 8:
            groups: list[np.ndarray] = []
            for signature_id in range(self.signature_count):
                mask = signature_ids == signature_id
                if np.any(mask):
                    groups.append(index_array[mask])
            return groups

        order = np.argsort(signature_ids, kind="stable")
        ordered_ids = signature_ids[order]
        cuts = np.flatnonzero(ordered_ids[1:] != ordered_ids[:-1]) + 1
        ordered_indices = index_array[order]
        return list(np.split(ordered_indices, cuts))


def _widen_signature_ids(ids: np.ndarray, written: int) -> np.ndarray:
    """Return the next wider unsigned ID buffer, preserving written entries."""
    widths = {
        np.dtype(np.uint8): np.dtype(np.uint16),
        np.dtype(np.uint16): np.dtype(np.uint32),
        np.dtype(np.uint32): np.dtype(np.uint64),
    }
    target_dtype = widths.get(ids.dtype)
    if target_dtype is None:
        raise OverflowError("LMDB availability signatures exceed uint64 capacity")
    widened = np.empty(ids.shape, dtype=target_dtype)
    widened[:written] = ids[:written]
    return widened


def _scan_availability_index(
    frame_count: int,
    read_raw: Callable[[int], bytes | None],
    availability_keys: Sequence[str],
    dataset: str,
) -> _AvailabilityIndex:
    """Build compact signature IDs with one exact source scan.

    The scan stores one unsigned integer per frame and one dictionary entry per
    distinct signature. It never accumulates frame indices as Python objects.

    Parameters
    ----------
    frame_count : int
        Number of positions in the index domain.
    read_raw : Callable[[int], bytes or None]
        Function returning the encoded frame for one domain position.
    availability_keys : Sequence[str]
        Normalized field names defining the partition.
    dataset : str
        Dataset path included in progress messages.

    Returns
    -------
    _AvailabilityIndex
        Compact signature IDs aligned with domain positions.
    """
    report_progress = _is_local_rank_zero()
    if report_progress:
        log.info(
            "LMDB label-availability scan started: dataset=%s, frames=%d, labels=%s",
            dataset,
            frame_count,
            list(availability_keys),
        )

    key_bits = {key: 1 << position for position, key in enumerate(availability_keys)}
    signature_ids = np.empty(frame_count, dtype=np.uint8)
    signature_map: dict[int, int] = {}
    next_log = time.monotonic() + _AVAILABILITY_LOG_SECONDS if report_progress else 0.0
    for start in range(0, frame_count, _AVAILABILITY_SCAN_CHUNK):
        stop = min(start + _AVAILABILITY_SCAN_CHUNK, frame_count)
        for position in range(start, stop):
            raw = read_raw(position)
            if raw is None:
                raise RuntimeError(
                    f"LMDB frame at position {position} is missing from {dataset}"
                )
            signature = _raw_frame_availability(raw, key_bits)
            signature_id = signature_map.get(signature)
            if signature_id is None:
                signature_id = len(signature_map)
                signature_map[signature] = signature_id
                if signature_id > np.iinfo(signature_ids.dtype).max:
                    signature_ids = _widen_signature_ids(signature_ids, position)
            signature_ids[position] = signature_id

        if report_progress:
            now = time.monotonic()
            if stop < frame_count and now >= next_log:
                log.info(
                    "LMDB label-availability scan progress: dataset=%s, "
                    "frames=%d/%d (%.1f%%)",
                    dataset,
                    stop,
                    frame_count,
                    100.0 * stop / frame_count,
                )
                next_log = now + _AVAILABILITY_LOG_SECONDS

    if report_progress:
        log.info(
            "LMDB label-availability scan completed: dataset=%s, frames=%d, "
            "groups=%d, index_dtype=%s",
            dataset,
            frame_count,
            len(signature_map),
            signature_ids.dtype,
        )
    return _AvailabilityIndex(signature_ids, len(signature_map))


def _scan_lmdb_path_sequential(
    lmdb_path: str,
    availability_keys: Sequence[str],
    log_level: int,
) -> _AvailabilityIndex:
    """Scan a complete LMDB under a sequential-readahead environment."""
    logging.basicConfig(level=log_level)
    logging.getLogger().setLevel(log_level)
    environment = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
    )
    try:
        with environment.begin() as transaction:
            metadata = _read_metadata(transaction)
        frame_count, frame_format, _natoms_per_type = _parse_metadata(metadata)
        with environment.begin() as transaction:

            def read_raw(position: int) -> bytes | None:
                return transaction.get(format(position, frame_format).encode())

            return _scan_availability_index(
                frame_count,
                read_raw,
                availability_keys,
                lmdb_path,
            )
    finally:
        environment.close()


def _scan_lmdb_path_in_worker(
    lmdb_path: str,
    availability_keys: Sequence[str],
) -> _AvailabilityIndex:
    """Run a sequential scan outside a process holding this LMDB open."""
    executor = _create_lmdb_executor(1)
    try:
        return executor.submit(
            _scan_lmdb_path_sequential,
            lmdb_path,
            availability_keys,
            log.getEffectiveLevel(),
        ).result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


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
    dataset
        Dataset identifier used in frame-level diagnostics.
    """

    ntypes: int
    natoms: int
    type_remap: np.ndarray | None
    data_requirements: dict[str, Any]
    dataset: str = "<unknown LMDB>"


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


def _requirement_is_atomic(requirement: Any) -> bool:
    """Whether a data requirement describes a per-atom quantity."""
    if isinstance(requirement, dict):
        return bool(requirement.get("atomic", False))
    return bool(getattr(requirement, "atomic", False))


def _frame_source_available(frame: dict[str, Any], key: str) -> bool:
    """Resolve one frame's explicit or inferred source availability."""
    value = frame.get(key)
    source_present = _is_encoded_array(value) or isinstance(
        value, (np.ndarray, np.generic, int, float, bool)
    )
    find_key = f"find_{key}"
    if find_key in frame:
        return source_present and bool(
            float(np.asarray(_decode_value(frame[find_key])).item())
        )
    return source_present


def _raise_if_mandatory_unavailable(
    frame: dict[str, Any],
    key: str,
    requirement: Any,
    source_available: bool,
    *,
    dataset: str,
    frame_index: int,
) -> None:
    """Reject unavailable mandatory data at the shared frame boundary."""
    if not _requirement_is_mandatory(requirement) or source_available:
        return
    find_key = f"find_{key}"
    if find_key in frame:
        find_value = float(np.asarray(_decode_value(frame[find_key])).item())
        reason = (
            f"explicit {find_key}=0"
            if find_value == 0.0
            else f"field is absent or invalid despite {find_key}={find_value:g}"
        )
    else:
        reason = "field is absent or invalid"
    raise RuntimeError(
        f"Required LMDB field {key!r} is unavailable in frame {frame_index} "
        f"of {dataset}: {reason}."
    )


def resolve_per_atom_keys(
    frame: dict[str, Any],
    config: LmdbDecodeConfig,
) -> frozenset[str]:
    """Return the fields of one frame whose leading axis is the atom axis.

    Classification is authoritative wherever possible: coordinates and atom
    types are per-atom by construction, and every registered data requirement
    declares whether it is ``atomic``. Only fields the loader has never been
    told about fall back to comparing their leading axis against the atom
    count, and the frame-level fields that DeePMD itself produces are excluded
    from that fallback so a coincidental shape match cannot misclassify them.

    Parameters
    ----------
    frame : dict[str, Any]
        One decoded frame in DeePMD data-system convention. ``coord`` anchors
        the atom axis: it is the one field present in every frame whose
        leading axis is the atom count.
    config : LmdbDecodeConfig
        Decoder state holding the registered data requirements.

    Returns
    -------
    frozenset[str]
        Names of the fields to pad along their leading axis.
    """
    nloc = frame["coord"].shape[0]
    keys = set(_STRUCTURAL_PER_ATOM_KEYS)
    keys |= {
        key
        for key, requirement in config.data_requirements.items()
        if _requirement_is_atomic(requirement)
    }
    keys |= _OPTIONAL_PER_ATOM_KEYS
    for key, value in frame.items():
        if (
            key in keys
            or key in _FRAME_LEVEL_KEYS
            or key in config.data_requirements
            or key.startswith("find_")
            or key == "fid"
        ):
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == nloc:
            keys.add(key)
    return frozenset(keys & frame.keys())


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


def _resolve_derived_requirement(
    frame: dict[str, Any],
    key: str,
    requirement: Any,
    config: LmdbDecodeConfig,
) -> None:
    """Resolve one derived field from normalized structural frame data."""
    if key != "min_pair_dist":
        raise ValueError(f"Unsupported derived LMDB field {key!r}")

    coord = frame.get("coord")
    atype = frame.get("atype")
    if not isinstance(coord, np.ndarray) or not isinstance(atype, np.ndarray):
        frame.pop(key, None)
        frame[f"find_{key}"] = np.float32(0.0)
        return

    box = frame.get("box")
    if box is not None and np.allclose(box, 0.0):
        box = None
    threshold = float(_requirement_value(requirement, "default", 0.0))
    frame[key] = np.array(
        [compute_min_pair_dist_single(coord, box, atype, stop_below=threshold)],
        dtype=_resolve_frame_dtype(config, key),
    )
    frame[f"find_{key}"] = np.float32(1.0)


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
    for key, requirement in requirements.items():
        if _requirement_source_policy(requirement) == "derived":
            _resolve_derived_requirement(frame, key, requirement, config)

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
        find_key = f"find_{key}"
        source_available = _frame_source_available(frame, key)
        _raise_if_mandatory_unavailable(
            frame,
            key,
            requirement,
            source_available,
            dataset=config.dataset,
            frame_index=original_key,
        )

        if not source_available:
            frame[find_key] = np.float32(
                1.0 if _requirement_source_policy(requirement) == "default" else 0.0
            )
            shape = (frame_natoms, ndof) if atomic else (ndof,)
            data = np.full(shape, default, dtype=dtype)
            if repeat != 1:
                data = np.repeat(data, repeat).reshape(-1)
            frame[key] = data
        else:
            frame.setdefault(find_key, np.float32(1.0))
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


def _pad_fill_value(key: str) -> int:
    """Return the value written into the padded tail of one per-atom field.

    Atom types use the phantom sentinel so that downstream code recognizes the
    slot as unoccupied; every other per-atom field is zeroed, which keeps
    padded rows neutral in the sums and masked reductions that consume them.
    """
    return PHANTOM_ATOM_TYPE if key == "atype" else 0


def per_atom_strides(
    frame: dict[str, Any],
    per_atom_keys: frozenset[str],
) -> dict[str, int]:
    """Return how many leading-axis entries each per-atom field spends per atom.

    Most per-atom fields carry one row per atom, so their leading axis is the
    atom count itself. A data requirement declared with ``repeat != 1`` is
    instead stored flat and atom-major, giving a leading axis of
    ``nloc * repeat``. Padding widens that axis by whole atoms either way, so
    the factor is all the padding logic needs to tell the two layouts apart.

    Parameters
    ----------
    frame : dict[str, Any]
        One decoded frame; ``coord`` anchors the atom count.
    per_atom_keys : frozenset[str]
        Fields to measure, as resolved by :func:`resolve_per_atom_keys`.

    Returns
    -------
    dict[str, int]
        Leading-axis entries per atom, keyed by field name.

    Raises
    ------
    ValueError
        If a field's leading axis is not a whole multiple of the atom count.
    """
    nloc = frame["coord"].shape[0]
    strides: dict[str, int] = {}
    for key in per_atom_keys:
        length = np.asarray(frame[key]).shape[0]
        if nloc == 0 or length % nloc:
            raise ValueError(
                f"LMDB field {key!r} has a leading axis of {length}, which is "
                f"not a whole number of entries per atom in a {nloc}-atom frame"
            )
        strides[key] = length // nloc
    return strides


@dataclass(frozen=True)
class BatchLayout:
    """Where each frame's per-atom rows sit on a decoded batch's leading axis.

    Two layouts serve the two shapes a model's node axis can take.

    The **rectangular** layout gives every frame a row of the batch-wide atom
    count and pads the tail of the shorter ones with phantom atoms, which is
    what a model reading an ``(nf, nloc, ...)`` node axis requires. The
    **ragged** layout concatenates the frames instead, so nothing is padded and
    the leading axis is the batch's real atom count; a model reading a flat
    node axis consumes that directly, paired with ``n_node``. Frame-level
    fields are stacked on the frame axis either way.

    Attributes
    ----------
    n_node : numpy.ndarray
        Real atom count of each frame of this decode, with shape ``(nf,)``.
    strides : dict[str, int]
        Leading-axis entries per atom of each per-atom field, as resolved by
        :func:`per_atom_strides`.
    ragged : bool
        Whether frames are concatenated rather than padded to a common width.
    width : int
        Atoms each frame occupies under the rectangular layout. It stays the
        width of the whole batch even where ``n_node`` covers one chunk of it,
        since chunks that padded to their own widths would not concatenate.
    """

    n_node: np.ndarray
    strides: dict[str, int]
    ragged: bool
    width: int

    def __post_init__(self) -> None:
        # Frame offsets on the ragged axis, in atoms. Held rather than summed
        # per lookup, since every field of every frame asks for one.
        object.__setattr__(
            self, "_offset", np.concatenate([[0], np.cumsum(self.n_node)])
        )

    @classmethod
    def over(
        cls, n_node: np.ndarray, strides: dict[str, int], *, ragged: bool
    ) -> "BatchLayout":
        """Return the layout of a batch holding the given per-frame counts."""
        return cls(
            n_node=n_node,
            strides=strides,
            ragged=ragged,
            width=int(n_node.max()) if n_node.size else 0,
        )

    def chunk(self, start: int, stop: int) -> "BatchLayout":
        """Return the layout of a contiguous run of this batch's frames."""
        return dataclasses.replace(self, n_node=self.n_node[start:stop])

    def field_length(self, key: str) -> int:
        """Return the leading-axis length one per-atom field is allocated."""
        stride = self.strides[key]
        if self.ragged:
            return int(self.n_node.sum()) * stride
        return self.width * stride

    def frame_index(self, row: int, key: str) -> Any:
        """Return the index selecting one frame's rows of a per-atom field.

        Rectangular batches carry a frame axis, so the rows of frame ``row``
        are the head of its own row; ragged batches carry none, and the rows
        are a run at the frame's offset.
        """
        stride = self.strides[key]
        length = int(self.n_node[row]) * stride
        if not self.ragged:
            return (row, slice(0, length))
        start = int(self._offset[row]) * stride
        return slice(start, start + length)


def _allocate_lmdb_batch(
    frame: dict[str, Any],
    batch_size: int,
    layout: BatchLayout,
) -> dict[str, Any]:
    """Allocate a contiguous NumPy batch from the first decoded frame.

    Per-atom fields are allocated at the length :class:`BatchLayout` gives
    them: one padded row per frame, or one concatenated run over the batch.
    """
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
        elif key in layout.strides:
            array = np.asarray(value)
            head = (
                (layout.field_length(key),)
                if layout.ragged
                else (batch_size, layout.field_length(key))
            )
            destination = np.full(
                (*head, *array.shape[1:]),
                _pad_fill_value(key),
                dtype=array.dtype,
            )
            destination[layout.frame_index(0, key)] = array
            batch[key] = destination
        else:
            array = np.asarray(value)
            destination = np.empty((batch_size, *array.shape), dtype=array.dtype)
            destination[0] = array
            batch[key] = destination
    return batch


def _promote_batch_field(
    batch: dict[str, Any],
    field: str,
    layout: BatchLayout,
    frames_written: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Widen one batch field's dtype in place, preserving what was written.

    The replacement is prefilled with the field's padding value rather than
    zeroed, so that entries not yet written keep the marker the allocation gave
    them; for ``atype`` under the rectangular layout that marker is what
    identifies a phantom atom.
    """
    destination = batch[field]
    promoted = np.full(destination.shape, _pad_fill_value(field), dtype=dtype)
    # The leading axis counts frames, except for a per-atom field of a ragged
    # batch, where it counts the atom rows those frames have filled.
    if field in layout.strides and layout.ragged:
        written = int(layout._offset[frames_written]) * layout.strides[field]
    else:
        written = frames_written
    promoted[:written] = destination[:written]
    batch[field] = promoted
    return promoted


def decode_lmdb_batch(
    transaction: lmdb.Transaction,
    original_keys: Sequence[int],
    frame_format: str,
    config: LmdbDecodeConfig,
    layout: BatchLayout | None = None,
) -> dict[str, Any]:
    """Decode LMDB records directly into preallocated contiguous arrays.

    The function keeps at most one temporary frame alive. It avoids the
    decode-copy, dtype-copy, Python frame-list, and final ``numpy.stack``
    sequence used by generic collation.

    Parameters
    ----------
    transaction : lmdb.Transaction
        Open read transaction on the LMDB environment.
    original_keys : Sequence[int]
        Integer LMDB frame keys in batch order.
    frame_format : str
        Format specification for integer LMDB frame keys.
    config : LmdbDecodeConfig
        Decoder state independent of the LMDB environment.
    layout : BatchLayout, optional
        Where each frame's per-atom rows belong. Defaults to a rectangular
        layout at the atom count of the first frame, which leaves a batch of
        uniform atom count untouched.

    Returns
    -------
    dict[str, Any]
        One collated batch of contiguous NumPy arrays. A ragged layout adds
        ``n_node``, the per-frame atom count its flat axis is read with.

    Notes
    -----
    The layout fixes the shape of every field of the result, so a chunked
    decode must pass each chunk the layout of its own frames, cut from the
    batch-wide one; :meth:`LmdbDataReader.batch_layout` resolves that once.

    Frames need not expose the same optional fields. A field only some of
    them carry is left out of the batch, and the matching ``find_*`` flag
    reports the label unavailable, which matches what
    :func:`collate_lmdb_frames` produces for the same frames. Registered
    data requirements are exempt: :func:`decode_lmdb_frame` puts them on
    every frame.
    """
    if not original_keys:
        raise ValueError("decode_lmdb_batch requires at least one frame key")

    batch: dict[str, Any] | None = None
    batch_size = len(original_keys)
    # Frames need not expose the same optional fields. Counting the frames
    # each field appears on settles, once the batch is read, which fields
    # have a complete column to stack and which labels the batch may report
    # as available; see :func:`_batch_find_flags` for the same rule applied
    # to the generic collation path.
    field_counts: dict[str, int] = {}
    find_flags: dict[str, bool] = {}
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
        frame_nloc = frame["coord"].shape[0]
        if layout is None:
            layout = BatchLayout.over(
                np.full(batch_size, frame_nloc, dtype=np.int64),
                per_atom_strides(frame, resolve_per_atom_keys(frame, config)),
                ragged=False,
            )
        if int(layout.n_node[row]) != frame_nloc:
            raise ValueError(
                f"the batch layout gives frame {original_key} "
                f"{int(layout.n_node[row])} atoms, but it holds {frame_nloc}"
            )
        for field in frame:
            field_counts[field] = field_counts.get(field, 0) + 1

        if batch is None:
            batch = _allocate_lmdb_batch(frame, batch_size, layout)
            find_flags = {
                field: float(value) != 0.0
                for field, value in frame.items()
                if field.startswith("find_")
            }
            continue

        for field, value in frame.items():
            if field.startswith("find_"):
                find_flags[field] = find_flags.get(field, False) and float(value) != 0.0
                continue
            if field == "type" or value is None or field not in batch:
                continue
            if field == "fid":
                batch[field][row] = value
                continue
            destination = batch[field]
            array = np.asarray(value)
            stride = layout.strides.get(field)
            # A per-atom field may differ in its leading axis, which the layout
            # absorbs as long as the axis stays a whole number of atoms; every
            # remaining axis must match exactly.
            if stride is not None:
                lead_axes = 1 if layout.ragged else 2
                expected_tail: tuple[int, ...] = destination.shape[lead_axes:]
                actual_tail = array.shape[1:]
                if array.shape[0] != frame_nloc * stride:
                    raise ValueError(
                        f"LMDB field {field!r} spends {array.shape[0]} leading "
                        f"entries on {frame_nloc} atoms in frame {original_key}, "
                        f"against {stride} per atom in frame {original_keys[0]}"
                    )
            else:
                expected_tail = destination.shape[1:]
                actual_tail = array.shape
            if expected_tail != actual_tail:
                raise ValueError(
                    f"LMDB field {field!r} changes shape within one batch: "
                    f"expected {expected_tail}, got {actual_tail} "
                    f"for frame {original_key}"
                )
            result_dtype = np.result_type(destination.dtype, array.dtype)
            if result_dtype != destination.dtype:
                destination = _promote_batch_field(
                    batch, field, layout, row, result_dtype
                )
            if stride is not None:
                destination[layout.frame_index(row, field)] = array
            else:
                destination[row] = array

    assert batch is not None and layout is not None
    for field, present in find_flags.items():
        available = present and field_counts.get(field, 0) == batch_size
        batch[field] = np.float32(1.0 if available else 0.0)
    for field, count in field_counts.items():
        if count < batch_size and not field.startswith("find_"):
            batch.pop(field, None)
    if layout.ragged:
        batch["n_node"] = layout.n_node
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
    layout: BatchLayout,
) -> dict[str, Any]:
    """Decode one chunk using process-local LMDB state.

    ``layout`` is this chunk's slice of the batch layout decided by the parent
    process. Deriving it there rather than per chunk is what lets
    :func:`_merge_lmdb_chunks` concatenate the results.
    """
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
        layout,
    )


def _merge_lmdb_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge ordered worker chunks into one contiguous batch.

    Each chunk has already reduced its own frames, so merging repeats that
    reduction one level up: a label is available to the merged batch only
    where every chunk reports it, and a field only some chunks carry has no
    complete column to concatenate.
    """
    if not chunks:
        raise ValueError("cannot merge an empty LMDB chunk list")
    if len(chunks) == 1:
        return chunks[0]

    first = chunks[0]
    merged: dict[str, Any] = {}
    for key, value in first.items():
        if key.startswith("find_"):
            available = float(value) != 0.0 and all(
                key in chunk and float(chunk[key]) != 0.0 for chunk in chunks[1:]
            )
            merged[key] = np.float32(1.0 if available else 0.0)
        elif key == "sid" or value is None:
            merged[key] = value
        elif any(key not in chunk for chunk in chunks[1:]):
            continue
        elif key == "fid":
            merged[key] = [frame_id for chunk in chunks for frame_id in chunk[key]]
        else:
            merged[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    for chunk in chunks[1:]:
        for key in chunk:
            if key.startswith("find_"):
                merged.setdefault(key, np.float32(0.0))
    return merged


@dataclass
class _LmdbPoolEntry:
    """Reference-counted process pool shared by data tasks in one rank.

    Attributes
    ----------
    executor : ProcessPoolExecutor
        The pool itself.
    users : int
        Number of iterators holding the pool, which is retired by its last one.
    healthy : bool
        Whether the pool still decodes. Losing a decoder disables the pool for
        every iterator sharing it, and marks it as one that must not be waited
        on: a pool stuck reading the partial result of a decoder killed
        mid-write never finishes shutting down.
    """

    executor: ProcessPoolExecutor
    users: int
    healthy: bool = True


_LMDB_POOL_LOCK = threading.Lock()
_LMDB_POOLS: dict[int, _LmdbPoolEntry] = {}


def _detach_decoder_from_session() -> None:
    """Shield a decoder from the hangup that ends its launching session.

    A decoder is a background helper of the training process and owns no
    terminal, so the ``SIGHUP`` delivered when the session a run was launched
    from goes away carries no meaning for it, while the default disposition
    makes it fatal. The signals by which a run is actually stopped, ``SIGINT``
    and ``SIGTERM``, keep their disposition.

    This protects the decoders alone. It is effective because the pool is
    built on the ``spawn`` start method, whose workers are direct children of
    the training process with no intermediary of their own to lose.
    """
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)


def _create_lmdb_executor(num_workers: int) -> ProcessPoolExecutor:
    """Create a CUDA-safe LMDB decoder process pool.

    The ``spawn`` start method is chosen over ``forkserver`` for the sake of
    the shielding above: a fork server is an unshielded intermediary whose own
    death is reported as the death of every decoder it started, which breaks
    the pool however well the decoders themselves are protected.
    """
    return ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_detach_decoder_from_session,
    )


def _acquire_lmdb_executor(num_workers: int) -> _LmdbPoolEntry:
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
        return entry


def _release_lmdb_executor(num_workers: int) -> None:
    """Release one pool user and stop the pool after its final user."""
    retired: _LmdbPoolEntry | None = None
    with _LMDB_POOL_LOCK:
        entry = _LMDB_POOLS.get(num_workers)
        if entry is None:
            return
        entry.users -= 1
        if entry.users == 0:
            retired = entry
            del _LMDB_POOLS[num_workers]
    if retired is not None:
        if not retired.healthy:
            _dismantle_lmdb_executor(retired.executor)
        retired.executor.shutdown(wait=retired.healthy, cancel_futures=True)


def _dismantle_lmdb_executor(executor: ProcessPoolExecutor) -> None:
    """Force a pool that stopped delivering to finish shutting down.

    A pool reading the partial result of a decoder killed mid-write waits for
    bytes that never arrive, because the training process itself holds the
    last write end of that queue. The interpreter joins every pool manager
    thread before it exits, so a run would complete its training and then
    never terminate. Closing the write end delivers the awaited end of file;
    the surviving decoders are stopped first so that none of them writes into
    a queue about to close. The pool offers no public way to release a manager
    thread already committed to a read.
    """
    for process in list(getattr(executor, "_processes", {}).values()):
        if process.exitcode is None:
            process.kill()
    writer = getattr(getattr(executor, "_result_queue", None), "_writer", None)
    if writer is not None:
        writer.close()


#: Seconds between two liveness checks while waiting on the decoder pool. The
#: wait ends as soon as the chunks arrive, so this bounds only how long a pool
#: that has stopped delivering goes unnoticed.
_DECODER_LIVENESS_INTERVAL = 5.0


@dataclass
class _PendingBatch:
    """A batch handed to the decoder pool, and the indices that produced it.

    The indices are retained so that the batch can still be decoded in this
    process should the pool fail to deliver it.
    """

    indices: list[int]
    futures: list[Future[dict[str, Any]]]


class LmdbBatchIterator:
    """Deterministic same-nloc batches with parallel decode and one prefetch.

    The sampler remains in the parent process. Worker tasks receive only the
    selected integer frame keys and compact decoder state, so the dataset-wide
    metadata is never duplicated. Data tasks in the same rank share one
    process pool while retaining independent samplers and pending batches.
    Before a new pass is prefetched, samplers exposing ``set_epoch`` are
    advanced to the next deterministic shuffle state.

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
        self._epoch = 0
        self._iterator = self._iter_epoch()
        self._num_workers = num_workers
        self._pool: _LmdbPoolEntry | None = None
        self._pending: _PendingBatch | None = None
        self._deferred_indices: list[int] | None = None
        self._closed = False

    def __iter__(self) -> "LmdbBatchIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("cannot read from a closed LMDB batch iterator")

        if self._pending is not None:
            pending, self._pending = self._pending, None
            batch = self._collect(pending)
        elif self._deferred_indices is not None:
            indices, self._deferred_indices = self._deferred_indices, None
            batch = self._reader.decode_batch(indices)
        else:
            batch = self._decode(self._next_indices())

        self._schedule(self._next_indices())
        return batch

    def _decode(self, indices: list[int]) -> dict[str, Any]:
        """Decode one batch, in the pool when that is worthwhile and possible."""
        futures = self._offer(indices)
        if futures is None:
            return self._reader.decode_batch(indices)
        return self._collect(_PendingBatch(indices, futures))

    def _offer(self, indices: list[int]) -> list[Future[dict[str, Any]]] | None:
        """Hand a batch to the pool, or ``None`` if it will not take it."""
        if not self._worth_decoding_in_parallel(indices):
            return None
        if self._pool is None:
            self._pool = _acquire_lmdb_executor(self._num_workers)
        if not self._pool.healthy:
            return None
        try:
            return self._submit(indices)
        except BrokenProcessPool:
            self._lose_the_pool()
            return None

    def _collect(self, pending: _PendingBatch) -> dict[str, Any]:
        """Return a batch from the pool, decoding it here if the pool cannot.

        A decoder killed from outside -- by the kernel under memory pressure,
        or by a signal aimed at the session the run was launched from -- ends
        the batch one of two ways. Usually the pool notices and fails every
        future it holds. Should the decoder die midway through writing a
        result, however, the pool reads that partial result forever instead of
        reporting itself broken, and the run stops with no diagnosis and no
        error; punctuating the wait with a liveness check covers that case.
        """
        try:
            if self._wait_for_decoder(pending.futures):
                return _merge_lmdb_chunks(
                    [future.result() for future in pending.futures]
                )
        except BrokenProcessPool:
            pass
        self._lose_the_pool(pending.futures)
        return self._reader.decode_batch(pending.indices)

    def _wait_for_decoder(
        self,
        futures: "Sequence[Future[dict[str, Any]]]",
    ) -> bool:
        """Wait for submitted work while checking that its decoder survives."""
        remaining = set(futures)
        while remaining:
            _, remaining = futures_wait(remaining, timeout=_DECODER_LIVENESS_INTERVAL)
            if remaining and self._decoder_exited():
                return False
        return not any(
            isinstance(future.exception(), BrokenProcessPool) for future in futures
        )

    def _decoder_exited(self) -> bool:
        """Whether any decoder has exited, which the pool reports nowhere else."""
        processes = getattr(self._pool.executor, "_processes", None) or {}
        return any(process.exitcode is not None for process in processes.values())

    def _lose_the_pool(self, futures: "Sequence[Future[dict[str, Any]]]" = ()) -> None:
        """Disable the pool for every iterator sharing it, reporting it once."""
        for future in futures:
            future.cancel()
        if self._pool is None or not self._pool.healthy:
            return
        self._pool.healthy = False
        log.warning(
            "An LMDB decoder process exited unexpectedly; decoding continues "
            "in the training process. Throughput may drop. Set "
            "DP_LMDB_NUM_WORKERS=1 to select in-process decoding from the "
            "start, and launch the run under nohup or setsid so that the "
            "decoders outlive the session that started it."
        )

    def _next_indices(self) -> list[int]:
        """Return the next sampler batch and restart after exhaustion."""
        try:
            return next(self._iterator)
        except StopIteration:
            self._epoch += 1
            self._iterator = self._iter_epoch()
            return next(self._iterator)

    def _iter_epoch(self) -> Iterator[list[int]]:
        """Create a sampler iterator for the current epoch."""
        set_epoch = getattr(self._sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(self._epoch)
        return iter(self._sampler)

    def _submit(self, indices: list[int]) -> list[Future[dict[str, Any]]]:
        """Submit one batch as balanced contiguous chunks.

        The batch layout is resolved here rather than per chunk, so that every
        chunk decodes to the same field shapes and the results concatenate.
        """
        original_keys = self._reader.original_keys(indices)
        layout = self._reader.batch_layout(indices)
        workers = min(self._num_workers, len(original_keys))
        base_size, remainder = divmod(len(original_keys), workers)
        chunks: list[tuple[list[int], BatchLayout]] = []
        start = 0
        for worker_index in range(workers):
            stop = start + base_size + int(worker_index < remainder)
            chunks.append((original_keys[start:stop], layout.chunk(start, stop)))
            start = stop
        decode_config = self._reader.worker_decode_config()
        return [
            self._pool.executor.submit(
                _decode_lmdb_worker_chunk,
                self._reader.lmdb_path,
                self._reader.frame_format,
                decode_config,
                chunk,
                chunk_layout,
            )
            for chunk, chunk_layout in chunks
        ]

    def _worth_decoding_in_parallel(self, indices: list[int]) -> bool:
        """Whether process decoding amortizes its scheduling and IPC cost."""
        return self._num_workers > 1 and len(indices) >= self._num_workers

    def _schedule(self, indices: list[int]) -> None:
        """Prefetch the next batch in the pool, or leave it to the caller."""
        futures = self._offer(indices)
        self._pending = _PendingBatch(indices, futures) if futures else None
        self._deferred_indices = None if futures else indices

    @property
    def started(self) -> bool:
        """Whether this iterator has acquired the shared process pool."""
        return self._pool is not None

    @property
    def closed(self) -> bool:
        """Whether this iterator has released its resources."""
        return self._closed

    def close(self) -> None:
        """Finish or cancel prefetched work and release the decoder pool."""
        if self._closed:
            return
        self._closed = True
        if self._pending is not None:
            running = [
                future for future in self._pending.futures if not future.cancel()
            ]
            if running and not self._wait_for_decoder(running):
                self._lose_the_pool(running)
            self._pending = None
        self._deferred_indices = None
        if self._pool is not None:
            self._pool = None
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


def _group_positions_by_value(values: np.ndarray) -> dict[int, np.ndarray]:
    """Group the positions of ``values`` by the value each holds.

    Sorting once and cutting at the value boundaries keeps the work inside
    NumPy. Accumulating Python lists instead costs one interpreter iteration
    and one boxed integer per element, which at the 10^8 frames a large LMDB
    holds is minutes of runtime and tens of gigabytes of resident memory.

    Parameters
    ----------
    values : numpy.ndarray
        Integer values to group by, with shape ``(n,)``.

    Returns
    -------
    dict[int, numpy.ndarray]
        Value → the ascending positions holding it. Each array is a slice of
        one shared buffer, so the mapping costs one extra array in total.
    """
    if values.size == 0:
        return {}
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    cuts = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    stops = np.concatenate((cuts, [values.size]))
    return {
        int(ordered[start]): order[start:stop]
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=True)
    }


def _compute_batch_size(nloc: int, rule: int) -> int:
    """Compute batch_size for a given nloc using the auto rule."""
    bsi = rule // max(nloc, 1)
    if bsi * nloc < rule:
        bsi += 1
    return max(bsi, 1)


#: Seed of the representative shuffle behind :attr:`LmdbDataReader.total_batch`.
#: Fixed so that the reported count is reproducible across calls and processes.
_TOTAL_BATCH_SEED = 0


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

    An LMDB typically holds frames of many different atom counts. The
    ``batch_size`` rule decides how those frames are grouped:

    - Every rule except ``"mix:N"`` keeps a batch homogeneous in atom count,
      so no padding is ever needed, whichever layout it is decoded in.
    - ``"mix:N"`` allows one batch to span several atom counts. A consumer
      reading a flat node axis takes such a batch concatenated; one reading
      an ``(nf, nloc, ...)`` axis takes it padded to the batch-wide maximum,
      the padded rows carrying ``atype = -1`` so that the neighbor list, the
      atomic model and the loss all skip them. The choice is
      :meth:`use_ragged_batches`.

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
        - ``"mix:N"``: mixed-nloc batching with an atom-axis budget of
          ``N``. Frames of different atom counts share a batch, whose atom
          axis holds at most ``N`` entries: the total atom count under the
          ragged layout, the padded ``nframes * max_nloc`` under the
          rectangular one. This is the natural extension of ``"max:N"``:
          the bound is on the decoded batch, and a lone frame with
          ``nloc > N`` still forms a batch of its own. Filling batches
          rather than cutting them at atom-count boundaries also keeps the
          frame-level loss terms closer to the weighting an atom budget asks
          for than ``"max:N"`` manages; see :func:`_chop_mixed_nloc`.
    """

    def __init__(
        self,
        lmdb_path: str,
        type_map: list[str],
        batch_size: int | str = "auto",
    ) -> None:
        self.lmdb_path = str(Path(lmdb_path).resolve())
        self._type_map = type_map
        # Read before opening the frame-serving environment, which disables
        # the readahead this one large sequential value depends on. Training
        # draws frames in shuffled order and so leaves it disabled.
        meta = _read_metadata_of(self.lmdb_path)
        self._env = _open_lmdb(self.lmdb_path)

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

        # Per-frame atom counts drive every batching rule: same-nloc grouping,
        # the ``filter:N`` drop, and the atom-axis layout of a ``mix:N`` batch.
        # ``orig_frame_nlocs`` / ``orig_frame_system_ids`` are indexed by the
        # *original* LMDB frame index. After a potential ``filter:N`` drop we
        # rebuild ``self._frame_nlocs`` / ``self._frame_system_ids`` so they
        # are parallel arrays over the *dataset* index space (0..len(self));
        # the dataset-to-original mapping lives in ``self._retained_keys``.
        # Metadata carries the counts when the writer recorded them; otherwise
        # each frame's atom_types shape is scanned (~10 us/frame).
        # Both tables are held as NumPy arrays rather than Python lists: at
        # 10^8 frames a list of boxed integers costs tens of gigabytes, while
        # the arrays cost 4 bytes an entry and let the grouping below run as
        # array operations. The list msgpack unpacks is released immediately.
        meta_nlocs = meta.get("frame_nlocs")
        if meta_nlocs is not None:
            orig_frame_nlocs = np.fromiter(
                meta_nlocs, dtype=np.int32, count=len(meta_nlocs)
            )
        else:
            orig_frame_nlocs = np.asarray(
                _scan_frame_nlocs(
                    self._env, self.nframes, self._frame_fmt, self._natoms
                ),
                dtype=np.int32,
            )
        del meta_nlocs

        # Parse frame_system_ids for auto_prob support. ``_nsystems`` must stay
        # at ``max(original_sid) + 1`` even after filter:N so that user-facing
        # auto_prob block slicing (e.g. ``prob_sys_size;0:284:0.5;284:842:0.5``)
        # keeps its meaning across filter thresholds.
        meta_sys_ids = meta.get("frame_system_ids")
        if meta_sys_ids is not None:
            orig_frame_system_ids: np.ndarray | None = np.fromiter(
                meta_sys_ids, dtype=np.int32, count=len(meta_sys_ids)
            )
            self._nsystems = int(orig_frame_system_ids.max()) + 1
        else:
            orig_frame_system_ids = None
            self._nsystems = 1
        del meta_sys_ids, meta

        # Parse batch_size spec. ``auto_rule``, ``max_rule`` and ``mix_rule``
        # are mutually exclusive; ``filter_rule`` implies ``max_rule`` plus
        # dropping frames whose nloc exceeds the threshold.
        self._auto_rule: int | None = None
        self._max_rule: int | None = None
        self._filter_rule: int | None = None
        self._mix_rule: int | None = None
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
            elif batch_size.startswith("mix:"):
                self._mix_rule = _parse_positive_rule(batch_size, "mix:")
            else:
                raise ValueError(
                    f"Unsupported batch_size {batch_size!r}. Expected int, "
                    "'auto', 'auto:N', 'max:N', 'filter:N', or 'mix:N'."
                )

        # Determine which original-index frames survive the filter. Without
        # ``filter:N`` every frame is retained, and the identity mapping is
        # left as the ``arange`` so no frame is copied through a mask.
        retained_keys = np.arange(self.nframes, dtype=np.int64)
        if self._filter_rule is not None:
            n_dropped = int(np.count_nonzero(orig_frame_nlocs > self._filter_rule))
            if n_dropped > 0:
                retained_keys = retained_keys[orig_frame_nlocs <= self._filter_rule]
                log.info(
                    f"LMDB filter:{self._filter_rule} drops {n_dropped}/"
                    f"{self.nframes} frames with nloc > {self._filter_rule} "
                    f"({self.lmdb_path})."
                )

        # Dataset-index → original LMDB frame key. ``__getitem__`` looks up
        # this table so that ``reader[i]`` is a valid LMDB read for every
        # ``0 <= i < len(reader)``, no matter how many frames were filtered.
        self._retained_keys: np.ndarray = retained_keys

        # Re-key _frame_nlocs / _frame_system_ids into the dataset-index
        # space so that every downstream consumer (nloc_groups, system_groups,
        # LmdbBatchSampler, _expand_indices_by_blocks) operates in a single,
        # self-consistent indexing scheme.
        keys_are_identity = retained_keys.size == self.nframes
        self._keys_are_identity = keys_are_identity
        self._frame_nlocs = (
            orig_frame_nlocs if keys_are_identity else orig_frame_nlocs[retained_keys]
        )

        if orig_frame_system_ids is None:
            self._frame_system_ids: np.ndarray | None = None
        elif keys_are_identity:
            self._frame_system_ids = orig_frame_system_ids
        else:
            self._frame_system_ids = orig_frame_system_ids[retained_keys]

        # nframes now reflects retained frames; __len__ returns this and the
        # valid index domain for __getitem__ is [0, self.nframes).
        self.nframes = int(retained_keys.size)

        # Group retained frames by nloc using dataset indices (0..len-1).
        # Statistics collection consumes these groups in every batching mode,
        # because per-nloc groups are the largest units that stack without
        # padding.
        self._nloc_groups = _group_positions_by_value(self._frame_nlocs)

        # Frames per original system id; the sid numbering is preserved (no
        # compression) so user-facing auto_prob slices stay meaningful across
        # filter thresholds. Fully-dropped systems count zero. The per-system
        # index lists themselves are built only if asked for; see
        # :attr:`system_groups`.
        self._system_groups: dict[int, np.ndarray] | None = None
        if self._frame_system_ids is not None:
            self._system_nframes: list[int] = np.bincount(
                self._frame_system_ids, minlength=self._nsystems
            ).tolist()
        else:
            self._system_nframes = [self.nframes]

        # Nominal batch size, reported to callers that want a single number.
        # The sampler never uses it: same-nloc modes go through
        # get_batch_size_for_nloc, and ``mix:N`` sizes each batch by budget.
        mean_nloc = (
            float(self._frame_nlocs.mean()) if self._frame_nlocs.size else self._natoms
        )
        if self._auto_rule is not None:
            self.batch_size = _compute_batch_size(self._natoms, self._auto_rule)
        elif self._max_rule is not None:
            self.batch_size = max(1, self._max_rule // max(self._natoms, 1))
        elif self._mix_rule is not None:
            self.batch_size = max(1, int(self._mix_rule / max(mean_nloc, 1.0)))
        else:
            self.batch_size = int(batch_size)

        # Data requirements tracking
        self._data_requirements: dict[str, DataRequirementItem] = {}
        self._data_requirements_frozen = False
        self._data_requirements_revision = 0
        # Requirements arrive after reader construction. Availability remains
        # unresolved until the first grouping request so raw fields unused by
        # the run cause no I/O and cannot influence the partition.
        self._uniform_availability: bool | None = None
        self._availability_index: _AvailabilityIndex | None = None
        self._decode_config = LmdbDecodeConfig(
            ntypes=self._ntypes,
            natoms=self._natoms,
            type_remap=self._type_remap,
            data_requirements=self._data_requirements,
            dataset=self.lmdb_path,
        )
        # Which fields carry an atom axis follows from the registered
        # requirements, so this cache is invalidated when they change.
        self._per_atom_strides: dict[str, int] | None = None
        # Batches are rectangular until a consumer that reads a flat node axis
        # asks otherwise; see :meth:`use_ragged_batches`.
        self._ragged_batches = False

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
        - ``mix:N``: ``max(1, floor(N / nloc))``, the count a batch would
          hold were every one of its frames this size. Training batches are
          sized by budget instead; this value serves the per-nloc statistics
          groups, which stack without padding and therefore batch like
          ``max:N``.
        - fixed int: the same value for every nloc group.
        """
        if self._auto_rule is not None:
            return _compute_batch_size(nloc, self._auto_rule)
        atom_budget = self._max_rule if self._max_rule is not None else self._mix_rule
        if atom_budget is not None:
            return max(1, atom_budget // max(nloc, 1))
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
        original_key = int(self._retained_keys[index])
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
            keys.append(int(self._retained_keys[index]))
        return keys

    def batch_pad_nloc(self, indices: Sequence[int]) -> int:
        """Return the atom count every frame of one batch is padded to.

        Parameters
        ----------
        indices : Sequence[int]
            Dataset indices forming one batch.

        Returns
        -------
        int
            The largest atom count in the batch. Batches drawn from a single
            nloc group return that group's atom count, so padding is a no-op.
        """
        return int(self._frame_nlocs[np.asarray(indices, dtype=np.int64)].max())

    def batch_layout(
        self, indices: Sequence[int], *, ragged: bool | None = None
    ) -> BatchLayout:
        """Return where one batch's per-atom rows belong once decoded.

        The layout fixes the shape of every field of the decoded batch, so a
        decode split across worker processes resolves it here once and cuts a
        chunk's share from it. Resolving it per chunk would let two chunks
        disagree, both on the padded width and on the field classification,
        which falls back to comparing a leading axis against the atom count of
        whichever frame the chunk happens to start with.

        Parameters
        ----------
        indices : Sequence[int]
            Dataset indices forming one batch.
        ragged : bool, optional
            Layout to use, overriding the one configured for training batches.
            Consumers with a layout of their own, such as statistics, name it
            rather than inherit it.

        Returns
        -------
        BatchLayout
            The per-frame atom counts, the per-atom strides, and whether the
            frames are concatenated or padded to a common width.
        """
        # Widened after gathering, and only to the batch's own size: a layout
        # carries int64 counts wherever it is built, including in the decode
        # workers a chunk of it is shipped to.
        return BatchLayout.over(
            self._frame_nlocs[np.asarray(indices, dtype=np.int64)].astype(np.int64),
            self.per_atom_strides(),
            ragged=self._ragged_batches if ragged is None else ragged,
        )

    @property
    def ragged_batches(self) -> bool:
        """Whether decoded batches concatenate their frames rather than pad."""
        return self._ragged_batches

    def use_ragged_batches(self, ragged: bool) -> None:
        """Select the layout mixed-nloc training batches are delivered in.

        The choice belongs to both the batching rule and the model. Only
        ``mix:N`` may place different atom counts in one batch and therefore
        needs a layout choice: a model reading a flat node axis takes those
        frames concatenated, while one reading an ``(nf, nloc, ...)`` axis
        needs them padded to a common width. Every other batching rule keeps
        the established rectangular layout. Only the trainer sees both the
        model and the data, so it requests the model-compatible layout once,
        before training starts. Consumers with a layout of their own --
        statistics, validation -- name theirs at the point of use and are
        unaffected.

        The layout also decides how the sampler packs frames, since padding is
        what makes a batch's cost depend on its widest frame.

        Parameters
        ----------
        ragged : bool
            Whether the consumer accepts concatenated mixed-nloc frames.
        """
        self._ragged_batches = ragged and self.mixed_nloc

    def per_atom_strides(self) -> dict[str, int]:
        """Return the leading-axis entries per atom of each per-atom field.

        Every frame of one LMDB exposes the same fields, so the classification
        is a property of the dataset and its registered requirements rather
        than of a batch, and is resolved from the first frame once.

        Returns
        -------
        dict[str, int]
            Entries per atom, keyed by field name.
        """
        if self._per_atom_strides is None:
            frame = self[0]
            self._per_atom_strides = per_atom_strides(
                frame, resolve_per_atom_keys(frame, self._decode_config)
            )
        return self._per_atom_strides

    def decode_batch(
        self, indices: Sequence[int], *, ragged: bool | None = None
    ) -> dict[str, Any]:
        """Decode one batch directly into contiguous NumPy arrays.

        Parameters
        ----------
        indices : Sequence[int]
            Dataset indices forming one batch.
        ragged : bool, optional
            Layout to decode into, overriding the one configured for training
            batches. See :meth:`batch_layout`.

        Returns
        -------
        dict[str, Any]
            One collated batch of contiguous NumPy arrays.
        """
        self._data_requirements_frozen = True
        return decode_lmdb_batch(
            self._transaction(),
            self.original_keys(indices),
            self._frame_fmt,
            self._decode_config,
            self.batch_layout(indices, ragged=ragged),
        )

    @property
    def frame_format(self) -> str:
        """Format specification used for integer LMDB frame keys."""
        return self._frame_fmt

    @property
    def decode_config(self) -> LmdbDecodeConfig:
        """Decoder state for in-process consumers.

        The returned object shares the reader's live requirement mapping and
        reading it does not freeze registration, unlike
        :meth:`worker_decode_config`, which hands the state to another process
        and so must fix it first.
        """
        return self._decode_config

    def worker_decode_config(self) -> LmdbDecodeConfig:
        """Freeze and return decoder state for worker serialization."""
        self._data_requirements_frozen = True
        return self._decode_config

    @property
    def closed(self) -> bool:
        """Whether parent-process LMDB resources have been released."""
        return self._closed

    # --- Data requirement interface ---

    def _scan_exact_availability(
        self,
        availability_keys: Sequence[str],
    ) -> _AvailabilityIndex:
        """Build the exact index under sequential kernel readahead."""

        def scan(transaction: lmdb.Transaction) -> _AvailabilityIndex:
            def read_raw(position: int) -> bytes | None:
                original_key = int(self._retained_keys[position])
                return transaction.get(format(original_key, self._frame_fmt).encode())

            return _scan_availability_index(
                self.nframes,
                read_raw,
                availability_keys,
                self.lmdb_path,
            )

        environment = self._env
        if environment is None:
            raise RuntimeError("cannot scan a closed LMDB reader")
        if bool(environment.flags().get("readahead", False)):
            return scan(self._transaction())

        resolved = str(Path(self.lmdb_path).resolve())
        cache_entry = _ENV_CACHE.get(resolved)
        owns_environment_exclusively = (
            cache_entry is not None
            and cache_entry[0] is environment
            and cache_entry[1] == 1
        )
        if not owns_environment_exclusively:
            if _is_local_rank_zero():
                log.info(
                    "LMDB label-availability scan uses an isolated sequential "
                    "reader because the random-read environment is shared: %s",
                    self.lmdb_path,
                )
            source_index = _scan_lmdb_path_in_worker(
                self.lmdb_path,
                availability_keys,
            )
            if self._keys_are_identity:
                return source_index
            return _AvailabilityIndex(
                source_index.ids[self._retained_keys],
                source_index.signature_count,
            )

        transaction = self._txn
        if transaction is not None:
            transaction.abort()
        self._txn = None
        self._env = None
        _close_lmdb(self.lmdb_path)
        try:
            sequential_environment = _open_lmdb(self.lmdb_path, sequential=True)
            with sequential_environment.begin() as sequential_transaction:
                return scan(sequential_transaction)
        finally:
            _close_lmdb(self.lmdb_path)
            self._env = _open_lmdb(self.lmdb_path, sequential=False)
            self._txn = self._env.begin()

    def add_data_requirement(self, data_requirement: list[DataRequirementItem]) -> None:
        """Register the consumer data contract before resolving any frame."""
        if self._data_requirements_frozen:
            raise RuntimeError(
                "LMDB data requirements must be registered before reading any frame"
            )
        for item in data_requirement:
            self._data_requirements[item["key"]] = item
        self._data_requirements_revision += 1
        self._uniform_availability = None
        self._availability_index = None
        self._per_atom_strides = None

    def availability_groups(self, indices: np.ndarray) -> list[np.ndarray]:
        """Partition dataset indices into label-compatible groups.

        A ``find_*`` flag is one scalar per batch. Optional tracked labels
        therefore remain homogeneous so present labels are not discarded.
        Mandatory labels fail during frame decoding; default-backed and
        derived fields do not participate.

        The bounded uniformity probe is deferred until requirements have been
        registered. A dataset that probes mixed builds one compact exact index
        for the entire retained dataset. Statistics and every sampler epoch
        reuse that index without further LMDB reads.

        The probe reads a bounded sample, so "uniform" is a finding and not a
        proof; :data:`_AVAILABILITY_PROBE_FRAMES` states what a missed frame
        costs and how a dataset can settle the question exactly.

        Parameters
        ----------
        indices : numpy.ndarray
            Dataset indices to partition.

        Returns
        -------
        list[numpy.ndarray]
            Non-empty index groups in a stable order, together covering
            ``indices``.
        """
        if len(indices) == 0:
            return []
        availability_keys = _availability_requirement_keys(self._data_requirements)
        if not availability_keys:
            self._uniform_availability = True
            return [indices]
        if self._uniform_availability is None:
            self._uniform_availability = _probe_uniform_availability(
                self._transaction(),
                _evenly_spaced(self._retained_keys, _AVAILABILITY_PROBE_FRAMES),
                self._frame_fmt,
                availability_keys,
            )
        if self._uniform_availability:
            return [indices]
        if self._availability_index is None:
            self._availability_index = self._scan_exact_availability(
                availability_keys,
            )
        index_array = np.asarray(indices)
        return self._availability_index.groups(
            index_array,
            positions=index_array,
        )

    @property
    def data_requirements(self) -> list[DataRequirementItem]:
        """Registered data requirements in insertion order."""
        return list(self._data_requirements.values())

    @property
    def data_requirements_revision(self) -> int:
        """Monotonic revision used to invalidate dependent sampling plans."""
        return self._data_requirements_revision

    def print_summary(self, name: str, prob: Any) -> None:
        """Print basic dataset info."""
        n_groups = len(self._nloc_groups)
        if self._auto_rule is not None:
            bs_str = f"auto:{self._auto_rule}"
        elif self._filter_rule is not None:
            bs_str = f"filter:{self._filter_rule}"
        elif self._max_rule is not None:
            bs_str = f"max:{self._max_rule}"
        elif self._mix_rule is not None:
            bs_str = f"mix:{self._mix_rule}"
        else:
            bs_str = str(self.batch_size)

        log.info(
            f"LMDB {name}: {self.lmdb_path}, "
            f"{self.nframes} frames, {n_groups} nloc groups, "
            f"batch_size={bs_str}, "
            f"mixed_nloc={self.mixed_nloc}"
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
        """Number of batches in one pass over the dataset.

        Every batching rule but ``mix:N`` fixes the count independently of the
        order frames are visited in. ``mix:N`` fills batches to an atom budget
        instead, so its count follows the shuffle: a pass in dataset order
        groups frames of one original system together, which are already close
        in atom count, and needs measurably fewer batches than the shuffled
        pass training actually performs. The count is therefore taken from a
        shuffled pass under a fixed seed, which is drawn from the same
        distribution as a training pass while staying reproducible. It remains
        an estimate: consult the sampler that will actually be iterated when
        the exact count matters, as the trainers do to derive an epoch length.
        """
        return len(LmdbBatchSampler(self, shuffle=True, seed=_TOTAL_BATCH_SEED))

    @property
    def batch_sizes(self) -> list[int]:
        return [self.batch_size]

    @property
    def mixed_nloc(self) -> bool:
        """Whether one batch may span several atom counts."""
        return self._mix_rule is not None

    @property
    def atom_budget(self) -> int | None:
        """Atom-axis budget of a ``mix:N`` batch, or ``None`` in other modes.

        The axis is measured in the layout the batch will be decoded in: real
        atoms when the frames are concatenated, padded slots when they are not.
        """
        return self._mix_rule

    @property
    def mixed_type(self) -> bool:
        """LMDB datasets are always mixed_type (frames may have different compositions)."""
        return True

    @property
    def type_map(self) -> list[str]:
        """Model-side type map used when constructing the reader."""
        return self._type_map

    @property
    def nloc_groups(self) -> dict[int, np.ndarray]:
        """Atom count → the ascending dataset indices holding it."""
        return self._nloc_groups

    @property
    def frame_nlocs(self) -> np.ndarray:
        """Per-frame atom count, indexed by dataset index."""
        return self._frame_nlocs

    @property
    def nsystems(self) -> int:
        """Number of original systems merged into this LMDB."""
        return self._nsystems

    @property
    def frame_system_ids(self) -> np.ndarray | None:
        """Per-frame system index, or None when the metadata omits it."""
        return self._frame_system_ids

    @property
    def system_groups(self) -> dict[int, np.ndarray]:
        """System index → the ascending dataset indices of its frames.

        Built on demand: the per-system index lists cost an array the size of
        the dataset, which the training and statistics paths never need --
        they consult :attr:`system_nframes` instead.
        """
        if self._system_groups is None:
            self._system_groups = (
                _group_positions_by_value(self._frame_system_ids)
                if self._frame_system_ids is not None
                else {0: np.arange(self.nframes, dtype=np.int64)}
            )
        return self._system_groups

    @property
    def system_nframes(self) -> list[int]:
        """Number of frames per system."""
        return self._system_nframes


def _batch_find_flags(frames: list[dict[str, Any]]) -> dict[str, np.float32]:
    """Reduce the per-frame ``find_*`` flags to the one scalar a batch carries.

    A ``find_*`` flag is a single scalar for the whole batch, so a label
    counts as available only where every frame supplies it. A frame that
    lacks the label carries a default fill in its place, and reporting the
    label as unavailable is what keeps that fill out of the loss.

    Registered data requirements are default-filled on every frame by
    :func:`decode_lmdb_frame`. Availability-sensitive labels are normally
    partitioned before collation to preserve their usable frames; this
    reduction remains the correctness boundary when the bounded probe misses
    a rare frame. Default-backed and derived requirements may mix
    deliberately, while mandatory requirements fail during frame resolution.

    Parameters
    ----------
    frames : list[dict[str, Any]]
        Per-frame dicts about to be stacked.

    Returns
    -------
    dict[str, numpy.float32]
        One scalar flag for every ``find_*`` key that any frame carries.
    """
    find_keys = sorted(
        {key for frame in frames for key in frame if key.startswith("find_")}
    )
    return {
        key: np.float32(
            1.0
            if all(key in frame and float(frame[key]) != 0.0 for frame in frames)
            else 0.0
        )
        for key in find_keys
    }


def _pad_atom_axis(xp: Any, array: Any, length: int, fill: int, device: Any) -> Any:
    """Widen one per-atom array's leading axis to ``length``."""
    if array.shape[0] == length:
        return array
    tail = xp.full(
        (length - array.shape[0], *array.shape[1:]),
        fill,
        dtype=array.dtype,
        device=device,
    )
    return xp.concat([array, tail], axis=0)


def collate_lmdb_frames(
    frames: list[dict[str, Any]],
    per_atom_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Stack a list of per-frame dicts into a single batch dict.

    Backend-agnostic via ``array_api_compat``: works for numpy, torch, jax,
    etc. The array library is inferred from the first frame's ``coord``.

    Conventions match :func:`deepmd.dpmodel.utils.batch.normalize_batch`:
    ``find_*`` flags remain scalar; ``fid`` is collected as a list; ``type``
    is dropped (callers should already use ``atype``); other arrays are
    stacked along axis 0. A ``sid`` placeholder is appended.

    Frames need not agree on label availability. A label only some of them
    carry is reported unavailable for the whole batch, as
    :func:`_batch_find_flags` describes, and a field only some of them carry
    is left out of the batch entirely because it cannot be stacked. Neither
    can happen to a registered data requirement, which
    :func:`decode_lmdb_frame` puts on every frame.

    The batch keeps the key order of its frames, which is the order
    :func:`decode_lmdb_batch` also produces, so a batch is the same mapping
    whichever of the two decode paths built it.

    Parameters
    ----------
    frames : list[dict[str, Any]]
        Per-frame dicts to stack.
    per_atom_keys : frozenset[str], optional
        Fields whose leading axis is the atom axis. When the frames differ in
        atom count these are padded to the batch maximum, with ``atype``
        filled by :data:`PHANTOM_ATOM_TYPE` and the rest zeroed. Leave empty
        for uniform batches, where padding would be a no-op anyway. Resolve
        the set with :func:`resolve_per_atom_keys`.

    Returns
    -------
    dict[str, Any]
        One collated batch.
    """
    import array_api_compat

    if not frames:
        raise ValueError("collate_lmdb_frames requires at least one frame")

    xp = array_api_compat.array_namespace(frames[0]["coord"])
    dev = array_api_compat.device(frames[0]["coord"])

    find_flags = _batch_find_flags(frames)

    strides = per_atom_strides(frames[0], per_atom_keys) if per_atom_keys else {}
    pad_nloc = max(frame["coord"].shape[0] for frame in frames) if strides else 0

    out: dict[str, Any] = {}
    for key in frames[0]:
        if key.startswith("find_"):
            out[key] = find_flags[key]
        elif key == "fid":
            out[key] = [f[key] for f in frames]
        elif key == "type":
            continue
        elif frames[0][key] is None:
            out[key] = None
        elif any(key not in frame for frame in frames):
            # A field only some frames carry cannot be stacked. Its ``find_``
            # flag is false by the same token, so the batch stays coherent
            # without it.
            continue
        elif key in strides:
            length = pad_nloc * strides[key]
            fill = _pad_fill_value(key)
            out[key] = xp.stack(
                [_pad_atom_axis(xp, f[key], length, fill, dev) for f in frames]
            )
        else:
            out[key] = xp.stack([f[key] for f in frames])
    # A flag a later frame raised that the first one never had still belongs
    # to the batch, reporting the label as unavailable.
    for key, flag in find_flags.items():
        out.setdefault(key, flag)
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

    # A bare ``prob_sys_size`` names no blocks: it asks for a probability
    # proportional to system size, which is what sampling the merged frames
    # uniformly already gives. There is nothing to reweight.
    if not blocks:
        return []

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
            "compute_block_targets: every block of "
            f"{auto_prob_style!r} is empty; the dataset retains no frames in "
            "any of them, so no reweighting is applied."
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
    _group_block_targets: list[int] | None = None,
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
        Per-frame system id for the entire dataset.
    block_targets : list[tuple[list[int], int]]
        Per-block (system_ids, total_target_frames).
    rng : np.random.Generator
        RNG for remainder sampling.
    _block_total_actual : list[int] or None
        Pre-computed total actual frame count per block (across all nloc
        groups).  When provided, avoids an O(N) scan of frame_system_ids.
    _sid_to_blk_arr : np.ndarray or None
        Pre-computed lookup from :func:`system_block_lookup`. When provided,
        avoids rebuilding the mapping for each call.
    _group_block_targets : list[int] or None
        Exact target for each block in this group. Production samplers
        allocate these targets globally across all groups so independent
        rounding cannot change a block's total size.

    Returns
    -------
    list[int]
        Expanded indices.
    """
    n_blocks = len(block_targets)

    if _sid_to_blk_arr is None:
        _sid_to_blk_arr = system_block_lookup(block_targets)

    sid_arr = np.asarray(frame_system_ids)
    idx_arr = np.asarray(indices, dtype=np.int64)
    idx_blks = resolve_frame_blocks(idx_arr, sid_arr, _sid_to_blk_arr)

    # Pre-compute block_total_actual if not provided
    if _block_total_actual is None and _group_block_targets is None:
        _block_total_actual = count_group_blocks(
            np.arange(sid_arr.size, dtype=np.int64),
            sid_arr,
            _sid_to_blk_arr,
            n_blocks,
        ).tolist()

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

        if _group_block_targets is not None:
            target = _group_block_targets[blk_idx]
        else:
            _, block_total_target = block_targets[blk_idx]
            block_total_act = _block_total_actual[blk_idx]

            # Backward-compatible fallback for direct callers. Samplers pass
            # exact group targets allocated by _allocate_group_block_targets.
            if block_total_act > 0:
                target = round(block_total_target * n_actual / block_total_act)
            else:
                target = n_actual
            target = max(target, n_actual)  # never shrink

        if target < n_actual:
            raise ValueError(
                "Per-group auto-probability target cannot shrink original frames: "
                f"target={target}, actual={n_actual}, block={blk_idx}"
            )

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


def collect_lmdb_sampling_groups(
    reader: "LmdbDataReader",
) -> list[tuple[int, np.ndarray]]:
    """Collect homogeneous LMDB groups shared by training and statistics.

    Atom count comes from metadata and costs no read. Availability-sensitive
    requirements subdivide a group only when the bounded probe detects mixed
    source presence; see :meth:`LmdbDataReader.availability_groups`.

    Parameters
    ----------
    reader : LmdbDataReader
        Reader whose frames are grouped by atom count and label availability.

    Returns
    -------
    list[tuple[int, numpy.ndarray]]
        Stable ``(nloc, frame indices)`` groups compatible with collation.
    """
    return [
        (nloc, group)
        for nloc in sorted(reader.nloc_groups)
        for group in reader.availability_groups(reader.nloc_groups[nloc])
    ]


def _collect_batch_groups(reader: "LmdbDataReader") -> list[np.ndarray]:
    """Collect the groups a training batch may be drawn from.

    A group is the largest set of frames one batch may span. Atom count
    partitions the frames in every mode but ``mix:N``, whose decoded batch
    accommodates unequal counts and so is bounded only by label availability.

    Parameters
    ----------
    reader : LmdbDataReader
        Reader providing the frame grouping and the batching rule.

    Returns
    -------
    list[numpy.ndarray]
        Frame indices per group, in the stable order shared by iteration and
        length.
    """
    if reader.mixed_nloc:
        return reader.availability_groups(np.arange(len(reader), dtype=np.int64))
    return [indices for _nloc, indices in collect_lmdb_sampling_groups(reader)]


def system_block_lookup(
    block_targets: list[tuple[list[int], int]],
) -> np.ndarray:
    """Build the system-id to block-index lookup a whole group indexes at once.

    The table carries one row beyond the highest system id named by a block,
    holding the "no block" marker. Clamping a system id into the table then
    maps everything a block does not name onto that row, which keeps the
    lookup total without a per-frame membership test.

    Parameters
    ----------
    block_targets : list[tuple[list[int], int]]
        Per-block ``(system_ids, target_frame_count)``.

    Returns
    -------
    numpy.ndarray
        Block index of each system id, ``-1`` where no block names it, with
        shape ``(max_system_id + 2,)``.
    """
    max_sid = max(
        (sid for system_ids, _target in block_targets for sid in system_ids),
        default=-1,
    )
    lookup = np.full(max_sid + 2, -1, dtype=np.int64)
    for block_index, (system_ids, _target) in enumerate(block_targets):
        if system_ids:
            lookup[np.asarray(system_ids, dtype=np.int64)] = block_index
    return lookup


def resolve_frame_blocks(
    indices: np.ndarray | list[int],
    frame_system_ids: np.ndarray,
    lookup: np.ndarray,
) -> np.ndarray:
    """Map each frame of one group to the block it belongs to.

    Parameters
    ----------
    indices : numpy.ndarray or list[int]
        Dataset indices of one group.
    frame_system_ids : numpy.ndarray
        Per-frame system id of the whole dataset. Gathered from, never
        widened: it holds one entry per frame, so converting its dtype would
        copy the entire dataset for the sake of a single group.
    lookup : numpy.ndarray
        System-id to block-index table from :func:`system_block_lookup`.

    Returns
    -------
    numpy.ndarray
        Block index of each frame, ``-1`` where no block claims it.
    """
    sids = np.asarray(frame_system_ids)[np.asarray(indices, dtype=np.int64)]
    return lookup[np.minimum(sids, lookup.size - 1)]


def count_group_blocks(
    indices: np.ndarray | list[int],
    frame_system_ids: np.ndarray,
    lookup: np.ndarray,
    n_blocks: int,
) -> np.ndarray:
    """Count how many of one group's frames fall in each block.

    Parameters
    ----------
    indices : numpy.ndarray or list[int]
        Dataset indices of one group.
    frame_system_ids : numpy.ndarray
        Per-frame system id of the whole dataset.
    lookup : numpy.ndarray
        System-id to block-index table from :func:`system_block_lookup`.
    n_blocks : int
        Number of blocks.

    Returns
    -------
    numpy.ndarray
        Frame count per block, with shape ``(n_blocks,)``. Frames belonging
        to no block are excluded, so the counts need not sum to the group
        size.
    """
    blocks = resolve_frame_blocks(indices, frame_system_ids, lookup)
    # Shift by one so that the "no block" marker lands in bin zero.
    return np.bincount(blocks + 1, minlength=n_blocks + 1)[1:]


def _allocate_group_block_targets(
    groups: list[np.ndarray],
    frame_system_ids: np.ndarray,
    block_targets: list[tuple[list[int], int]],
) -> list[list[int]]:
    """Allocate every block target exactly across homogeneous groups.

    Original frames form the non-shrinking baseline. Each block's expansion
    deficit is apportioned by actual group size using the integer largest-
    remainder method. Stable group order breaks equal-remainder ties, which
    keeps distributed ranks deterministic without floating-point rounding.
    """
    lookup = system_block_lookup(block_targets)
    group_actual = [
        count_group_blocks(
            indices, frame_system_ids, lookup, len(block_targets)
        ).tolist()
        for indices in groups
    ]

    group_targets = [counts.copy() for counts in group_actual]
    for block_index, (_system_ids, block_target) in enumerate(block_targets):
        block_actual = sum(counts[block_index] for counts in group_actual)
        if block_target < block_actual:
            raise ValueError(
                "Auto-probability block target cannot shrink original frames: "
                f"target={block_target}, actual={block_actual}, block={block_index}"
            )
        if block_actual == 0:
            if block_target != 0:
                raise ValueError(
                    "Cannot allocate a nonzero auto-probability target to an "
                    f"empty block: target={block_target}, block={block_index}"
                )
            continue

        deficit = block_target - block_actual
        remainders: list[tuple[int, int]] = []
        allocated = 0
        for group_index, counts in enumerate(group_actual):
            actual = counts[block_index]
            quotient, remainder = divmod(deficit * actual, block_actual)
            group_targets[group_index][block_index] += quotient
            allocated += quotient
            if actual > 0:
                remainders.append((remainder, group_index))

        # The largest-remainder method leaves fewer units than nonempty
        # groups, so each selected group receives at most one final unit.
        remainder_units = deficit - allocated
        remainders.sort(key=lambda item: (-item[0], item[1]))
        for _remainder, group_index in remainders[:remainder_units]:
            group_targets[group_index][block_index] += 1

    return group_targets


def _chop_same_nloc(
    reader: "LmdbDataReader", indices: Sequence[int]
) -> list[list[int]]:
    """Split one homogeneous group into fixed-size batches."""
    batch_size = reader.get_batch_size_for_nloc(int(reader.frame_nlocs[indices[0]]))
    index_list = np.asarray(indices).tolist()
    return [
        index_list[start : start + batch_size]
        for start in range(0, len(index_list), batch_size)
    ]


def _chop_mixed_nloc(reader: "LmdbDataReader", indices: list[int]) -> list[list[int]]:
    """Split one group into batches under an atom-axis budget.

    ``mix:N`` budgets the length of a batch's atom axis, and the layout the
    batch will be decoded in decides both how that length is measured and the
    order the frames are visited in.

    **Ragged.** The frames are concatenated, so the axis is simply their total
    atom count and nothing is padded. No frame's cost depends on its
    neighbours, so the group is taken in the caller's order, which training has
    already shuffled. Sorting would only make each batch homogeneous in system
    size, correlating the frames of an optimizer step for no gain.

    **Rectangular.** Every frame is padded to the widest one, so the axis is
    ``nframes * max_nloc`` and a batch pays for atoms it does not hold. Sorting
    by atom count is what keeps that overhead small: a batch is then a run of
    frames adjacent in the sorted order, so its padding is bounded by the
    atom-count spread across that run alone. Ties fall back to the caller's
    order, so frames of equal atom count still mix freely across epochs.

    Either way the group is cut only where the next frame would push the axis
    past the budget, which is the fewest batches obtainable **without
    reordering**: a batch's cost does not fall when frames are dropped from its
    front, so a batch that starts later can always extend at least as far as
    one that starts earlier, and by induction on the batch count, cutting as
    late as possible covers the longest prefix for every count.

    Under the rectangular layout the sort is part of the algorithm and the
    result is optimal outright, since an exchange argument turns any packing
    into contiguous runs of the sorted order. Under the ragged layout the order
    is the caller's, so the count is optimal only for that order; reordering
    could pack tighter -- the general problem is bin packing -- and is declined
    to keep the frames of an optimizer step decorrelated in system size.
    Padding, where it exists, is not separately minimized either.

    That the batches are full is what keeps the gradient weighting faithful. A
    batch is one optimizer step; its per-atom loss terms pool over the real
    labels, so a frame's weight there follows its atom count whatever the
    packing, while its frame-level terms (energy, virial) weigh frames equally,
    giving a frame the weight ``1 / k_b``. An atom budget makes ``k_b`` follow
    the atom count, exactly so under the ragged layout and up to the padding
    under the rectangular one, and an under-filled batch raises the weight of
    every frame it holds.

    A frame larger than the budget forms a batch of its own, matching how
    ``max:N`` treats an oversized nloc group.

    Parameters
    ----------
    reader : LmdbDataReader
        Provides the per-frame atom counts, the atom budget and the layout.
    indices : list[int]
        Dataset indices of one group.

    Returns
    -------
    list[list[int]]
        Batches whose union is ``indices``.
    """
    budget = reader.atom_budget
    if budget is None:
        raise ValueError("mixed-nloc batching requires a batch_size of 'mix:N'")

    index_array = np.asarray(indices, dtype=np.int64)
    # Gathered, not widened: the atom counts are one entry per frame, so a
    # dtype conversion here would copy the whole dataset for one group.
    nloc_array = reader.frame_nlocs[index_array]
    if not reader.ragged_batches:
        order = np.argsort(nloc_array, kind="stable")
        index_array, nloc_array = index_array[order], nloc_array[order]

    batches: list[list[int]] = []
    batch_start = 0
    real_atoms = 0
    for position, nloc in enumerate(nloc_array.tolist()):
        count = position - batch_start + 1
        # Length of the atom axis this run would occupy once decoded. Under the
        # rectangular layout the ascending sort makes the current frame the
        # widest, so it alone sets the padded width.
        axis = real_atoms + nloc if reader.ragged_batches else count * nloc
        if count > 1 and axis > budget:
            batches.append(index_array[batch_start:position].tolist())
            batch_start, real_atoms = position, 0
        real_atoms += nloc
    batches.append(index_array[batch_start:].tolist())
    return batches


def _build_all_batches(
    reader: "LmdbDataReader",
    shuffle: bool,
    rng: np.random.Generator,
    block_targets: list[tuple[list[int], int]] | None = None,
) -> list[list[int]]:
    """Build the batches of one pass over the dataset.

    Groups are chopped into batches, then interleaved round-robin so that
    consecutive batches come from different groups. Under ``mix:N`` a batch
    also spans several atom counts; every other mode keeps it uniform.

    Parameters
    ----------
    reader : LmdbDataReader
        Provides the frame grouping and the batching rule.
    shuffle : bool
        Whether to shuffle indices within each group and shuffle the final
        batch order.
    rng : np.random.Generator
        Random number generator (deterministic for reproducibility).
    block_targets : list[tuple[list[int], int]] or None
        Per-block (system_ids, target_frame_count) from compute_block_targets.
        When provided, indices are expanded via full-copy + remainder sampling.

    Returns
    -------
    list[list[int]]
        Dataset indices grouped into batches.
    """
    groups = _collect_batch_groups(reader)
    chop = _chop_mixed_nloc if reader.mixed_nloc else _chop_same_nloc

    # Build per-group batches
    group_batches: list[list[list[int]]] = []

    # Pre-compute expensive objects once (avoids O(N) work per nloc group)
    sid_arr: np.ndarray | None = None
    sid_to_blk_arr: np.ndarray | None = None
    group_block_targets: list[list[int]] | None = None
    if block_targets and reader.frame_system_ids is not None:
        sid_arr = reader.frame_system_ids
        group_block_targets = _allocate_group_block_targets(
            groups, sid_arr, block_targets
        )
        sid_to_blk_arr = system_block_lookup(block_targets)

    for group_index, original_indices in enumerate(groups):
        indices = original_indices
        # Expand each group independently using targets that were allocated
        # globally, so that a block's total size is preserved exactly.
        if block_targets and sid_arr is not None and group_block_targets is not None:
            indices = _expand_indices_by_blocks(
                indices,
                sid_arr,
                block_targets,
                rng,
                _sid_to_blk_arr=sid_to_blk_arr,
                _group_block_targets=group_block_targets[group_index],
            )
        if shuffle:
            # ``permutation`` returns a copy, which matters because a group
            # may alias one of the reader's own index tables.
            indices = rng.permutation(indices)
        group_batches.append(chop(reader, indices) if len(indices) else [])

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


class LmdbBatchSampler:
    """Batch sampler over an LMDB, grouped by atom count.

    Atom count is handled by the reader's batching rule: all rules but
    ``mix:N`` keep a batch uniform in atom count, while ``mix:N`` fills
    batches to the atom budget its decoded layout is measured against.
    Groups are interleaved round-robin and the batch order is then shuffled,
    so training sees a varied mix.

    Optional tracked labels subdivide these groups when a bounded probe
    detects mixed presence. Default-backed and derived fields may mix freely;
    mandatory fields fail when an unavailable frame is decoded.

    The sampler serves one pass at a time, drawn from ``seed + epoch``. The
    pending pass is materialized before it is served, which is what lets
    ``__len__`` report exactly what ``__iter__`` will yield: under ``mix:N``
    the batch count follows the shuffle, because batches are filled to an atom
    budget rather than to a fixed frame count. Serving a pass advances the
    epoch, so a caller that just re-iterates sees a different shuffle every
    time, and :meth:`set_epoch` repositions that progression for a caller --
    a distributed run, a resumed one -- that needs to name the pass instead.

    Parameters
    ----------
    reader : LmdbDataReader
        The dataset reader providing the frame grouping and batching rule.
    shuffle : bool
        Whether to shuffle within each group and shuffle the batch order.
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
        self._epoch = 0
        self._block_targets = block_targets
        self._batches: list[list[int]] | None = None
        self._data_requirements_revision = -1

    def batches(self) -> list[list[int]]:
        """Return the batch list of the pending pass, building it if needed.

        Returns
        -------
        list[list[int]]
            Dataset indices grouped into batches.
        """
        revision = self._reader.data_requirements_revision
        if self._batches is None or self._data_requirements_revision != revision:
            seed = None if self._seed is None else self._seed + self._epoch
            self._batches = _build_all_batches(
                self._reader,
                self._shuffle,
                np.random.default_rng(seed),
                self._block_targets,
            )
            self._data_requirements_revision = revision
        return self._batches

    def set_epoch(self, epoch: int) -> None:
        """Select the pass to serve, discarding any pass still pending.

        Parameters
        ----------
        epoch : int
            Zero-based training epoch.
        """
        if epoch != self._epoch:
            self._epoch = epoch
            self._batches = None

    def __iter__(self) -> Iterator[list[int]]:
        """Yield the pending pass, and move the epoch on to its successor."""
        batches = self.batches()
        self.set_epoch(self._epoch + 1)
        yield from batches

    def __len__(self) -> int:
        """Number of batches the pending pass holds."""
        return len(self.batches())

    @property
    def total_batches(self) -> int:
        """Number of batches the pending pass holds over the whole dataset.

        The same count as ``len(self)`` here, and the two part company only in
        the distributed sampler, so a caller after a global figure need not
        know which of the two it holds.
        """
        return len(self)


class DistributedLmdbBatchSampler:
    """Distributed wrapper for LMDB batch sampling.

    All ranks build the same deterministic global batch list (using
    ``seed + epoch``). The list is padded deterministically when its length is
    not divisible by the number of ranks, then each rank takes a strided
    subset via :meth:`_partition_batches`. This keeps every rank on the same
    sampler epoch while duplicating at most ``world_size - 1`` batches.

    Override :meth:`_partition_batches` for custom load-balancing strategies.
    The default uses strided partitioning which gives good nloc diversity per
    rank.

    Parameters
    ----------
    reader : LmdbDataReader
        The dataset reader providing the frame grouping and batching rule.
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
        self._global: LmdbBatchSampler | None = None

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic cross-rank shuffling.

        Call this before each training epoch/cycle to get different but
        reproducible batch orderings across epochs.
        """
        if epoch != self._epoch:
            self._epoch = epoch
            self._global = None

    def _global_batches(self) -> list[list[int]]:
        """Return the batch list every rank builds identically."""
        if self._global is None:
            self._global = LmdbBatchSampler(
                self._reader,
                shuffle=self._shuffle,
                seed=self._seed + self._epoch,
                block_targets=self._block_targets,
            )
        return self._global.batches()

    def __iter__(self) -> Iterator[list[int]]:
        """Yield this rank's partition of the global batch list."""
        yield from self._partition_batches(self._global_batches())

    def _partition_batches(self, all_batches: list[list[int]]) -> list[list[int]]:
        """Partition global batches to this rank.

        The default pads the global list to a multiple of ``world_size`` and
        then takes ``all_batches[rank::world_size]``. This gives good nloc
        diversity per rank since batches are interleaved across groups before
        shuffling, while ensuring that every rank yields the same number of
        batches.

        Override this method for custom load-balancing. For example, a
        greedy algorithm could assign batches to ranks based on estimated
        compute cost (``reader.batch_pad_nloc(batch) * len(batch)`` gives the
        padded cost of each batch).
        """
        if not all_batches:
            return []
        batches_per_rank = (len(all_batches) + self._world_size - 1) // self._world_size
        total_size = batches_per_rank * self._world_size
        padding_size = total_size - len(all_batches)
        if padding_size:
            repetitions = (padding_size + len(all_batches) - 1) // len(all_batches)
            all_batches = [
                *all_batches,
                *(all_batches * repetitions)[:padding_size],
            ]
        return all_batches[self._rank :: self._world_size]

    def __len__(self) -> int:
        """Number of batches for this rank."""
        return len(self._partition_batches(self._global_batches()))

    @property
    def total_batches(self) -> int:
        """Number of batches one full pass holds, before the per-rank split."""
        return len(self._global_batches())

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
        max_frames: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self._type_map = type_map or []
        meta = _read_metadata_of(self.lmdb_path)
        # Every read this class serves walks a group in ascending key order,
        # which is the pattern kernel readahead exists for.
        self._env = _open_lmdb(self.lmdb_path, sequential=True)

        self.nframes, self._frame_fmt, self._natoms_per_type = _parse_metadata(meta)
        self._natoms = sum(self._natoms_per_type)
        self._ntypes = (
            len(self._type_map) if self._type_map else len(self._natoms_per_type)
        )

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

        # Select the frames to test, without reading any of them. Atom counts
        # come from the metadata, so grouping, shuffling and truncation are
        # index arithmetic; only the retained frames are ever decoded.
        self._nloc_groups = self._select_frames(meta, shuffle_test, max_frames)

        # Data requirements
        self._requirements: dict[str, dict[str, Any]] = {}
        self._uniform_availability: bool | None = None
        self._availability_indices: dict[
            int, tuple[np.ndarray, _AvailabilityIndex]
        ] = {}
        self._full_availability_index: _AvailabilityIndex | None = None
        self._decode_config = LmdbDecodeConfig(
            ntypes=self._ntypes,
            natoms=self._natoms,
            type_remap=self._type_remap,
            data_requirements=self._requirements,
            dataset=self.lmdb_path,
        )

        # Detect PBC from the first retained frame.
        self.pbc = True
        first = next(
            (indices[0] for indices in self._nloc_groups.values() if len(indices)), None
        )
        probe = self._read_frames([first]) if first is not None else []
        if probe:
            box = probe[0].get("box")
            if not isinstance(box, np.ndarray) or np.allclose(box, 0.0):
                self.pbc = False

        self.mixed_type = True

    def _select_frames(
        self,
        meta: dict,
        shuffle_test: bool,
        max_frames: float | None,
    ) -> dict[int, np.ndarray]:
        """Group the frame indices by atom count, then sample each group.

        Parameters
        ----------
        meta : dict
            The LMDB metadata. ``frame_nlocs`` gives the atom count of every
            frame; without it the frames have to be scanned for it.
        shuffle_test : bool
            Whether the frames of a group are drawn in random order.
        max_frames : int or float or None
            Upper bound on the number of frames retained per group. ``None``
            and a non-finite bound retain the whole group.

        Returns
        -------
        dict[int, numpy.ndarray]
            The retained LMDB frame indices of each atom count.
        """
        raw_nlocs = meta.get("frame_nlocs")
        if _is_encoded_array(raw_nlocs):
            nlocs = _decode_array(raw_nlocs).reshape(-1).astype(np.int32)
        elif raw_nlocs is not None:
            nlocs = np.fromiter(raw_nlocs, dtype=np.int32, count=len(raw_nlocs))
        else:
            nlocs = np.asarray(
                _scan_frame_nlocs(
                    self._env, self.nframes, self._frame_fmt, self._natoms
                ),
                dtype=np.int32,
            )

        groups = _group_positions_by_value(nlocs)
        keep = (
            None
            if max_frames is None or not np.isfinite(max_frames)
            else int(max_frames)
        )
        if not shuffle_test and keep is None:
            return groups

        for nloc, indices in groups.items():
            if shuffle_test:
                dp_random.shuffle(indices)
            groups[nloc] = indices if keep is None else indices[:keep]
        return groups

    def _read_frames(self, frame_indices: Sequence[int]) -> list[dict[str, Any]]:
        """Decode the given LMDB frames, applying the type remapping.

        Parameters
        ----------
        frame_indices : Sequence[int]
            Indices of the frames to read, as keyed in the LMDB.

        Returns
        -------
        list[dict[str, Any]]
            One decoded frame per index that the LMDB holds.
        """
        frames: list[dict[str, Any]] = []
        with self._env.begin() as transaction:
            for index in np.asarray(frame_indices).tolist():
                raw = transaction.get(format(index, self._frame_fmt).encode())
                if raw is None:
                    continue
                frames.append(
                    decode_lmdb_frame(
                        raw,
                        int(index),
                        self._decode_config,
                        copy_arrays=True,
                    )
                )
        return frames

    def __del__(self) -> None:
        """Release the LMDB environment ref-count on garbage collection.

        The count is released only once, and only if construction got as far
        as taking it: an instance that failed earlier holds no reference, and
        releasing one it never took would close the environment underneath
        whichever reader does hold it.
        """
        if getattr(self, "_env", None) is None:
            return
        self._env = None
        _close_lmdb(self.lmdb_path)

    @property
    def nloc_groups(self) -> dict[int, np.ndarray]:
        """Nloc → the LMDB frame indices retained for that atom count."""
        return self._nloc_groups

    def availability_groups(self, frame_indices: np.ndarray) -> list[np.ndarray]:
        """Partition LMDB frame indices into label-compatible groups.

        The validation counterpart of
        :meth:`LmdbDataReader.availability_groups`. Only optional tracked
        labels participate. Exact signature IDs are cached per retained nloc
        group and reused by every full-validation pass.

        Parameters
        ----------
        frame_indices : numpy.ndarray
            LMDB frame indices to partition.

        Returns
        -------
        list[numpy.ndarray]
            Non-empty index groups in a stable order, together covering the
            input. An empty group would name no frames to stack, so an empty
            input yields no group at all.
        """
        if not len(frame_indices):
            return []
        availability_keys = _availability_requirement_keys(self._requirements)
        if not availability_keys:
            self._uniform_availability = True
            return [frame_indices]
        if self._uniform_availability is None:
            groups = list(self._nloc_groups.values())
            per_group = max(1, _AVAILABILITY_PROBE_FRAMES // max(len(groups), 1))
            with self._env.begin() as transaction:
                self._uniform_availability = _probe_uniform_availability(
                    transaction,
                    (
                        key
                        for group in groups
                        for key in _evenly_spaced(group, per_group)
                    ),
                    self._frame_fmt,
                    availability_keys,
                )
        if self._uniform_availability:
            return [frame_indices]
        index_array = np.asarray(frame_indices)
        if not bool(self._env.flags().get("readahead", False)):
            if self._full_availability_index is None:
                self._full_availability_index = _scan_lmdb_path_in_worker(
                    self.lmdb_path,
                    availability_keys,
                )
            return self._full_availability_index.groups(
                index_array,
                positions=index_array,
            )

        cache_key = id(frame_indices)
        cached = self._availability_indices.get(cache_key)
        if cached is None or cached[0] is not frame_indices:
            with self._env.begin() as transaction:

                def read_raw(position: int) -> bytes | None:
                    frame_index = int(index_array[position])
                    return transaction.get(
                        format(frame_index, self._frame_fmt).encode()
                    )

                availability_index = _scan_availability_index(
                    len(index_array),
                    read_raw,
                    availability_keys,
                    self.lmdb_path,
                )
            self._availability_indices[cache_key] = (
                frame_indices,
                availability_index,
            )
        else:
            availability_index = cached[1]
        return availability_index.groups(index_array)

    def get_test_by_indices(self, frame_indices: Sequence[int]) -> dict[str, Any]:
        """Stack one homogeneous validation group selected by frame index."""
        if not len(frame_indices):
            raise ValueError("frame_indices must contain at least one frame")
        frames = self._read_frames(frame_indices)
        nlocs = {
            len(frame["atype"])
            for frame in frames
            if isinstance(frame.get("atype"), np.ndarray)
        }
        if len(nlocs) != 1:
            raise ValueError(
                "LMDB validation group must contain exactly one atom count, "
                f"got {sorted(nlocs)}"
            )
        return self._stack_frames(frames, nlocs.pop())

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
        source_policy: DataRequirementSourcePolicy = "tracked",
        **kwargs: Any,
    ) -> None:
        """Register a data requirement (mirrors DeepmdData.add)."""
        requirement = DataRequirementItem(
            key,
            ndof,
            atomic=atomic,
            must=must,
            high_prec=high_prec,
            repeat=repeat,
            default=default,
            dtype=dtype,
            source_policy=source_policy,
        )
        self._requirements[key] = requirement.dict
        self._uniform_availability = None
        self._availability_indices.clear()
        self._full_availability_index = None

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
                source_policy=item["source_policy"],
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
        frame_indices, natoms = self._resolve_group(nloc)
        return self._stack_frames(self._read_frames(frame_indices), natoms)

    def iter_test(
        self,
        *,
        chunk_atoms: int,
        numb_test: float = float("inf"),
        nloc: int | None = None,
        frame_indices: Sequence[int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield the test frames in chunks, reading each chunk on demand.

        Only the frames of the chunk being yielded are held, which is what
        makes a dataset larger than memory testable.

        Parameters
        ----------
        chunk_atoms : int
            Upper bound on the number of atoms per chunk. A chunk always
            carries at least one frame, however many atoms it has.
        numb_test : float, optional
            Upper bound on the number of frames served. A non-finite bound
            serves every frame of the group.
        nloc : int or None, optional
            Atom count selecting the group, resolved as in :meth:`get_test`.
        frame_indices : Sequence[int] or None, optional
            Frames to serve in place of the whole group, which is how a
            label-compatible subgroup names itself. They must all have the
            atom count ``nloc`` selects.

        Yields
        ------
        dict[str, Any]
            One chunk of frames, stacked as :meth:`get_test` stacks them.
        """
        group_indices, natoms = self._resolve_group(nloc)
        frame_indices = group_indices if frame_indices is None else frame_indices
        if np.isfinite(numb_test):
            frame_indices = frame_indices[: int(numb_test)]
        step = max(1, int(chunk_atoms) // max(1, natoms))
        for begin in range(0, len(frame_indices), step):
            chunk = frame_indices[begin : begin + step]
            yield self._stack_frames(self._read_frames(chunk), natoms)

    def _resolve_group(self, nloc: int | None) -> tuple[np.ndarray, int]:
        """Return the retained frame indices and atom count of one group.

        Parameters
        ----------
        nloc : int or None
            The atom count to select. ``None`` selects the only group when the
            dataset is uniform, and the largest group otherwise.

        Returns
        -------
        tuple[numpy.ndarray, int]
            The LMDB frame indices of the group and its atom count.

        Raises
        ------
        ValueError
            If no frame has the requested atom count.
        """
        if nloc is not None:
            if nloc not in self._nloc_groups:
                raise ValueError(
                    f"No frames with nloc={nloc}. Available: {sorted(self._nloc_groups.keys())}"
                )
            return self._nloc_groups[nloc], nloc
        if len(self._nloc_groups) == 1:
            natoms = next(iter(self._nloc_groups))
            return self._nloc_groups[natoms], natoms
        natoms = max(self._nloc_groups, key=lambda k: len(self._nloc_groups[k]))
        group_summary = {k: len(v) for k, v in sorted(self._nloc_groups.items())}
        log.warning(
            f"Mixed-nloc LMDB for dp test: using nloc={natoms} group "
            f"({len(self._nloc_groups[natoms])} frames). "
            f"Available groups: {group_summary}"
        )
        return self._nloc_groups[natoms], natoms

    def _stack_frames(
        self, frames: list[dict[str, Any]], natoms: int
    ) -> dict[str, Any]:
        """Stack a list of same-nloc frames into numpy arrays.

        Frames need not agree on label availability; a label only some of
        them carry is reported unavailable for the whole group, mirroring
        :func:`_batch_find_flags` on the training path.
        """
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
        _structural_keys = frozenset(
            {"coord", "box", "atype", "natoms", "real_natoms_vec", "fid"}
        )
        all_keys: dict[str, dict[str, Any]] = {}
        if frames:
            for fk in frames[0]:
                if fk in _structural_keys or fk.startswith("find_"):
                    continue
                if fk not in all_keys:
                    all_keys[fk] = {"ndof": None, "atomic": False, "default": 0.0}
        for key, req in self._requirements.items():
            all_keys[key] = req

        for key in all_keys:
            has_key = all(_frame_source_available(frame, key) for frame in frames)
            requirement = self._requirements.get(key)
            if requirement is None and not has_key:
                # An unregistered field that only part of the group carries
                # has no complete column and remains outside the result.
                continue
            arrays: list[np.ndarray] = []
            for frame in frames:
                value = frame.get(key)
                if isinstance(value, np.ndarray):
                    arrays.append(value.astype(self._resolve_dtype(key)).ravel())
                elif value is not None:
                    arrays.append(
                        np.array([float(value)], dtype=self._resolve_dtype(key))
                    )
                else:
                    raise RuntimeError(
                        f"Resolved LMDB field {key!r} is absent in frame "
                        f"{frame.get('fid', '<unknown>')} of {self.lmdb_path}"
                    )
            result[key] = np.stack(arrays)
            result[f"find_{key}"] = 1.0 if has_key else 0.0

        return result


class LmdbTestDataNlocView:
    """Expose one stack-compatible subset of :class:`LmdbTestData`.

    The underlying :class:`LmdbTestData` groups frames by atom count. This
    view fixes one ``nloc`` and can additionally select a label-compatible
    subgroup within it. All other attributes (``pbc``, ``mixed_type``, …)
    are forwarded to the underlying object. It lets downstream consumers
    that expect a ``DeepmdData``-style system work on mixed-nloc or
    partially labeled LMDB datasets without vector find flags.
    """

    def __init__(
        self,
        lmdb_test_data: "LmdbTestData",
        nloc: int,
        frame_indices: Sequence[int] | None = None,
    ) -> None:
        self._inner = lmdb_test_data
        self._nloc = nloc
        self._frame_indices = frame_indices

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def get_test(self) -> dict[str, Any]:
        if self._frame_indices is not None:
            return self._inner.get_test_by_indices(self._frame_indices)
        return self._inner.get_test(nloc=self._nloc)

    def iter_test(
        self,
        *,
        chunk_atoms: int,
        numb_test: float = float("inf"),
    ) -> Iterator[dict[str, Any]]:
        """Yield this group's frames in chunks, as :meth:`get_test` selects them."""
        return self._inner.iter_test(
            chunk_atoms=chunk_atoms,
            numb_test=numb_test,
            nloc=self._nloc,
            frame_indices=self._frame_indices,
        )


def _validate_merge_type_maps(
    source_metadata: list[tuple[str, dict[str, Any]]],
) -> list[str] | None:
    """Return the shared type map required for byte-for-byte frame merging.

    ``merge_lmdb`` does not decode and rewrite atom-type arrays, so every
    source must use exactly the same index-to-species mapping. All-missing
    legacy metadata remains supported, but mixing explicit and missing maps is
    rejected because compatibility cannot be established.
    """
    source_type_maps = [(path, meta.get("type_map")) for path, meta in source_metadata]
    explicit_type_maps = [
        (path, list(type_map))
        for path, type_map in source_type_maps
        if type_map is not None
    ]
    if not explicit_type_maps:
        return None

    formatted_maps = ", ".join(
        f"{path}: {list(type_map)!r}" if type_map is not None else f"{path}: missing"
        for path, type_map in source_type_maps
    )
    if len(explicit_type_maps) != len(source_type_maps):
        raise ValueError(
            "Cannot merge LMDB datasets with mixed type_map metadata because "
            f"raw atom-type indices cannot be validated ({formatted_maps})"
        )

    canonical_type_map = explicit_type_maps[0][1]
    if any(type_map != canonical_type_map for _, type_map in explicit_type_maps[1:]):
        raise ValueError(
            "Cannot merge LMDB datasets with incompatible type_map values "
            f"because frames are copied without remapping ({formatted_maps})"
        )
    return canonical_type_map


def _copy_lmdb_source(
    src_path: str,
    metadata: dict[str, Any],
    dst_env: lmdb.Environment,
    dst_format: str,
    frame_idx: int,
    frame_nlocs: list[int],
    frame_system_ids: list[int],
    system_id_offset: int,
) -> tuple[int, dict, int]:
    """Copy one validated source under a ref-counted environment lease.

    The caller supplies metadata collected during the validation preflight so
    this copy pass does not read and decode it a second time.
    """
    src_env = _open_lmdb(src_path)
    try:
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

    Raises
    ------
    ValueError
        If sources use different explicit type maps, or mix explicit type-map
        metadata with legacy metadata where the mapping is missing.
    """
    import os
    import shutil

    # Validate every source before replacing or creating the destination. A
    # type-map validation failure must not destroy an existing dataset or
    # leave a partial output.
    source_metadata: list[tuple[str, dict[str, Any]]] = []
    for src_path in src_paths:
        src_env = _open_lmdb(src_path)
        try:
            with src_env.begin() as txn:
                source_metadata.append((src_path, _read_metadata(txn)))
        finally:
            _close_lmdb(src_path)
    merged_type_map = _validate_merge_type_maps(source_metadata)

    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    dst_env = lmdb.open(dst_path, map_size=map_size)
    frame_idx = 0
    fmt = "012d"
    frame_nlocs: list[int] = []
    frame_system_ids: list[int] = []
    first_system_info: dict | None = None
    sys_id_offset = 0
    try:
        for src_path, metadata in source_metadata:
            (
                frame_idx,
                source_system_info,
                sys_id_offset,
            ) = _copy_lmdb_source(
                src_path,
                metadata,
                dst_env,
                fmt,
                frame_idx,
                frame_nlocs,
                frame_system_ids,
                sys_id_offset,
            )
            if first_system_info is None:
                first_system_info = source_system_info

        merged_meta = {
            "nframes": frame_idx,
            "frame_idx_fmt": fmt,
            "system_info": first_system_info or {},
            "frame_nlocs": frame_nlocs,
            "frame_system_ids": frame_system_ids,
        }
        if merged_type_map is not None:
            merged_meta["type_map"] = merged_type_map
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
