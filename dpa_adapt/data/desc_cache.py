# SPDX-License-Identifier: LGPL-3.0-or-later
# data/desc_cache.py
#
# Transparent on-disk cache for extracted DPA descriptors.
# Two-tier: (1) per-system cache keyed by lightweight content hash,
# (2) bulk cache under ``~/.cache/dpa_adapt/desc_cache/`` keyed by
# (aggregate data fingerprint, checkpoint identity, branch, pooling).
#
# Systems are ``dpdata.System`` objects; cache keys are computed from
# data fingerprints and resolved checkpoint metadata.
#
# Note: ``load_or_extract()`` and ``ensure_per_system_cache()`` live in
# ``dpa_adapt.finetuner`` to avoid an import cycle (those functions need
# ``DPAFineTuner``, while ``finetuner`` imports cache helpers from here).

from __future__ import (
    annotations,
)

import hashlib
import os
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
)

import numpy as np

from dpa_adapt._backend import (
    resolve_pretrained_path,
)

if TYPE_CHECKING:
    import dpdata


# ---------------------------------------------------------------------------
# cache directory
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME", os.path.join(str(Path.home()), ".cache"))
    return Path(base) / "dpa_adapt" / "desc_cache"


# ---------------------------------------------------------------------------
# system fingerprint (O(n) over the full descriptor-relevant arrays)
# ---------------------------------------------------------------------------


def _hash_array(h: hashlib._Hash, arr: np.ndarray) -> None:
    """Fold an array's shape, dtype, and full byte content into *h*.

    The contiguous buffer is fed to :meth:`hashlib._Hash.update` directly via
    the buffer protocol, so no large intermediate ``bytes`` copy is made.
    """
    arr = np.ascontiguousarray(arr)
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr)


def _system_fingerprint(system: dpdata.System) -> str:
    """Return a hex fingerprint for a dpdata System.

    Hashes the *full* contents of the descriptor-relevant arrays — ``coords``,
    ``cells`` and ``atom_types`` — together with ``atom_names``.  Sampling
    only the first/last few entries (as an earlier version did) let any change
    in the middle of a long trajectory keep the same key, so the cache could
    return descriptors extracted from a different structure.  Hashing every
    element costs O(total array size), but that is negligible next to the
    descriptor extraction the cache guards, and it makes the key collision-safe
    for changed systems.
    """
    d = system.data

    h = hashlib.sha1()
    # atom-type identity
    _hash_array(h, np.asarray(d["atom_types"]))
    # atom_names (if present)
    names = d.get("atom_names", [])
    h.update("|".join(str(n) for n in names).encode())
    # full geometry
    _hash_array(h, np.asarray(d["coords"]))
    if "cells" in d:
        _hash_array(h, np.asarray(d["cells"]))
    return h.hexdigest()[:16]


def _data_fingerprint(systems: list) -> str:
    """Aggregate fingerprint for a list of systems in request order."""
    fps = [_system_fingerprint(s) for s in systems]
    h = hashlib.sha1()
    for fp in fps:
        h.update(fp.encode())
    return h.hexdigest()


def _checkpoint_fingerprint(pretrained: str) -> str:
    resolved = Path(resolve_pretrained_path(pretrained)).resolve()
    stat = resolved.stat()
    payload = f"{resolved}|{stat.st_mtime_ns}|{stat.st_size}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _type_map_payload(type_map: list[str] | tuple[str, ...] | None) -> str:
    if not type_map:
        return ""
    return "\x1f".join(str(item) for item in type_map)


def _cache_key(
    systems: list,
    pretrained: str,
    model_branch: str | None,
    pooling: str,
    *,
    type_map: list[str] | tuple[str, ...] | None = None,
) -> str:
    fp = _data_fingerprint(systems)
    ckpt_fp = _checkpoint_fingerprint(pretrained)
    tm = _type_map_payload(type_map)
    payload = f"{fp}|{ckpt_fp}|{model_branch or ''}|{pooling}|{tm}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# per-system cache path helpers
# ---------------------------------------------------------------------------


def _per_system_cache_path(
    system: dpdata.System,
    pretrained: str,
    model_branch: str | None = None,
    pooling: str = "mean",
    type_map: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Return the cache path for one system under a descriptor identity."""
    system_fp = _system_fingerprint(system)
    ckpt_fp = _checkpoint_fingerprint(pretrained)
    tm = _type_map_payload(type_map)
    payload = f"{system_fp}|{ckpt_fp}|{model_branch or ''}|{pooling}|{tm}"
    fp = hashlib.sha1(payload.encode()).hexdigest()[:16]
    return _cache_dir() / "per_system" / f"{fp}.npy"


def get_per_system_descriptor(
    system: dpdata.System,
    pretrained: str,
    model_branch: str | None = None,
    pooling: str = "mean",
    type_map: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    """Read cached descriptors for one system and descriptor identity.

    Raises ``FileNotFoundError`` if the cache file does not exist.
    """
    cache_path = _per_system_cache_path(
        system,
        pretrained,
        model_branch,
        pooling,
        type_map,
    )
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Per-system descriptor cache not found: {cache_path}\n"
            f"Run ensure_per_system_cache() first."
        )
    return np.load(cache_path)
