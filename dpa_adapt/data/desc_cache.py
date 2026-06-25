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

from __future__ import (
    annotations,
)

import hashlib
import logging
import os
from pathlib import (
    Path,
)

import numpy as np

from dpa_adapt._backend import (
    resolve_pretrained_path,
)

_LOG = logging.getLogger("dpa_adapt.data.desc_cache")


# ---------------------------------------------------------------------------
# cache directory
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME", os.path.join(str(Path.home()), ".cache"))
    return Path(base) / "dpa_adapt" / "desc_cache"


# ---------------------------------------------------------------------------
# lightweight system fingerprint (O(1) on array size, O(n) on atom count)
# ---------------------------------------------------------------------------


def _system_fingerprint(system) -> str:
    """Return a short hex fingerprint for a dpdata System.

    Uses only metadata and a tiny sample of coordinate data so it is fast
    even for large (10⁵+ frame) systems.  Collisions are possible in
    principle but vanishingly unlikely in practice given the combination of
    shape, dtype, atom_types, and first/last bytes.
    """
    d = system.data
    coords = np.asarray(d["coords"])
    atom_types = np.asarray(d["atom_types"])

    h = hashlib.sha1()
    # structural identity
    h.update(str(coords.shape).encode())
    h.update(str(coords.dtype).encode())
    h.update(atom_types.tobytes())
    # atom_names (if present)
    names = d.get("atom_names", [])
    h.update("|".join(str(n) for n in names).encode())
    # first / last 64 bytes of coords (captures actual content without
    # hashing the entire array)
    if coords.size > 0:
        flat = coords.ravel()
        h.update(flat[: min(64, len(flat))].tobytes())
        h.update(flat[-min(64, len(flat)) :].tobytes())
    # same for cells, if present
    if "cells" in d:
        cells = np.asarray(d["cells"])
        h.update(str(cells.shape).encode())
        if cells.size > 0:
            fc = cells.ravel()
            h.update(fc[: min(64, len(fc))].tobytes())
            h.update(fc[-min(64, len(fc)) :].tobytes())
    return h.hexdigest()[:16]


def _data_fingerprint(systems: list) -> str:
    """Aggregate fingerprint for a list of systems (order-independent)."""
    fps = sorted(_system_fingerprint(s) for s in systems)
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
# bulk cache
# ---------------------------------------------------------------------------


def load_or_extract(
    systems: list,
    pretrained: str,
    model_branch: str = None,
    pooling: str = "mean",
    cache: bool = True,
    type_map: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    """Return descriptors for *systems*, using the cache when possible.

    Parameters
    ----------
    systems : list[dpdata.System]
        Systems to extract descriptors from.
    pretrained : str
        Path to the DPA checkpoint.
    model_branch : str, optional
        Branch name.
    pooling : str
        Pooling strategy.
    cache : bool
        If False the cache is bypassed entirely.

    Returns
    -------
    np.ndarray, shape ``(n_frames_total, feat_dim)``
    """
    if cache:
        key = _cache_key(
            systems,
            pretrained,
            model_branch,
            pooling,
            type_map=type_map,
        )
        cache_path = _cache_dir() / f"{key}.npy"
        if cache_path.is_file():
            _LOG.info("Descriptor cache hit: %s", cache_path.name)
            return np.load(cache_path)
        _LOG.info("Descriptor cache miss; extracting...")
    else:
        _LOG.info("Descriptor cache bypassed (cache=False).")

    from dpa_adapt.finetuner import (
        DPAFineTuner,
    )

    extractor = DPAFineTuner(
        pretrained=pretrained,
        model_branch=model_branch,
        predictor="linear",
        pooling=pooling,
        type_map=list(type_map) if type_map else None,
    )
    descriptors = extractor._extract_features(systems)

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, descriptors)
        _LOG.info("Cached descriptors to %s", cache_path)

    return descriptors


# ---------------------------------------------------------------------------
# per-system cache — used by cross_validate to avoid OOM
# ---------------------------------------------------------------------------


def _per_system_cache_path(
    system,
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


def ensure_per_system_cache(
    systems: list,
    pretrained: str,
    model_branch: str = None,
    pooling: str = "mean",
    type_map: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Ensure every system has its descriptors cached to disk.

    Existing cache files are reused as-is.  Missing ones are extracted one
    system at a time for low peak memory.
    """
    missing: list = []
    for system in systems:
        if not _per_system_cache_path(
            system,
            pretrained,
            model_branch,
            pooling,
            type_map,
        ).is_file():
            missing.append(system)

    if not missing:
        _LOG.info(
            "All %d systems have per-system cache; nothing to extract.", len(systems)
        )
        return

    import torch

    from dpa_adapt.finetuner import (
        DPAFineTuner,
    )

    _LOG.info(
        "%d/%d systems missing per-system cache; extracting one by one...",
        len(missing),
        len(systems),
    )

    extractor = DPAFineTuner(
        pretrained=pretrained,
        model_branch=model_branch,
        predictor="linear",
        pooling=pooling,
        type_map=list(type_map) if type_map else None,
    )

    for i, system in enumerate(missing):
        cache_path = _per_system_cache_path(
            system,
            pretrained,
            model_branch,
            pooling,
            type_map,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        desc = extractor._extract_features([system])
        np.save(cache_path, desc)
        if extractor._device is not None and extractor._device.type == "cuda":
            torch.cuda.empty_cache()
        if i > 0 and i % 50 == 0:
            _LOG.info("  per-system cache: %d/%d done", i, len(missing))

    _LOG.info("Per-system cache ready (%d systems).", len(systems))


def get_per_system_descriptor(
    system,
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
