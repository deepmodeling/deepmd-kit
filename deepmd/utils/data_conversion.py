# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent conversion of external training data."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import shutil
import tempfile
import time
import uuid
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
)

import dpdata

log = logging.getLogger(__name__)

_EXTXYZ_SUFFIXES = frozenset({".xyz", ".extxyz"})
_CACHE_ENV = "DEEPMD_EXTXYZ_CACHE"
_CACHE_MANIFEST = ".deepmd_extxyz_cache.json"
_CONVERTER_SCHEMA_VERSION = 1
_SET_SIZE = 2000
_STRESS_SIGN = -1
_LOCK_TIMEOUT = 600.0
_STALE_LOCK_AGE = 24 * 60 * 60


def is_extxyz_path(path: str | os.PathLike[str]) -> bool:
    """Return whether an explicit path names an extxyz input file."""
    source = Path(path)
    return source.suffix.lower() in _EXTXYZ_SUFFIXES and not source.is_dir()


def _is_lmdb_path(path: str | os.PathLike[str]) -> bool:
    """Match the existing LMDB path predicate without importing dpmodel."""
    source = Path(path)
    return str(path).endswith(".lmdb") or (source / "data.mdb").is_file()


def _cache_root() -> Path:
    configured = os.environ.get(_CACHE_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(tempfile.gettempdir()) / "deepmd-kit" / "extxyz").resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(source: Path, dpdata_version: str) -> dict[str, Any]:
    return {
        "converter_schema_version": _CONVERTER_SCHEMA_VERSION,
        "conversion_settings": {
            "format": "extxyz",
            "output_format": "deepmd/npy",
            "set_size": _SET_SIZE,
            "stress_sign": _STRESS_SIGN,
        },
        "dpdata_version": dpdata_version,
        "source_path": str(source),
        "source_sha256": _file_sha256(source),
    }


def _fingerprint_digest(fingerprint: dict[str, Any]) -> str:
    encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_manifest(root: Path) -> dict[str, Any] | None:
    try:
        with (root / _CACHE_MANIFEST).open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(manifest, dict):
        return None
    return manifest


def _manifest_system_paths(root: Path, manifest: dict[str, Any]) -> list[str] | None:
    entries = manifest.get("systems")
    if not isinstance(entries, list) or not entries:
        return None

    resolved_root = root.resolve()
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            return None
        system = (root / entry["path"]).resolve()
        if not system.is_relative_to(resolved_root):
            return None
        if not (system / "type.raw").is_file():
            return None
        if not any(path.is_dir() for path in system.glob("set.*")):
            return None
        result.append(str(system))
    return result


def expand_extxyz_cache(path: str | os.PathLike[str]) -> list[str] | None:
    """Expand a materialized extxyz cache in deterministic manifest order."""
    root = Path(path)
    manifest = _read_manifest(root)
    if manifest is None:
        return None
    return _manifest_system_paths(root, manifest)


def _valid_cache(
    target: Path, fingerprint: dict[str, Any]
) -> tuple[dict[str, Any], list[str]] | None:
    manifest = _read_manifest(target)
    if manifest is None or manifest.get("fingerprint") != fingerprint:
        return None
    systems = _manifest_system_paths(target, manifest)
    if systems is None:
        return None
    return manifest, systems


def _labels(system: Any) -> list[str]:
    """Return DeePMD label names for fields produced by dpdata."""
    labels = []
    if "energies" in system.data:
        labels.append("energy")
    # dpdata uses plural dictionary keys. The singular names recorded here
    # are the canonical DeePMD dataset names (energy.npy, force.npy, etc.).
    if "forces" in system.data:
        labels.append("force")
    if "virials" in system.data:
        labels.append("virial")
    return labels


def _frame_error(source: Path, frame_index: int, exc: Exception) -> ValueError:
    detail = str(exc)
    lowered = detail.lower()
    if "energy" in lowered or "energies" in lowered:
        problem = "is missing a usable total-energy label"
    elif "force" in lowered or "forces" in lowered:
        problem = "is missing usable atomic-force labels"
    else:
        problem = "is not a valid labeled extended-XYZ frame"
    return ValueError(
        f"Frame {frame_index + 1} in extxyz file '{source}' {problem}. "
        "Each frame must contain atomic species, positions, total energy, and "
        f"atomic forces. dpdata reported: {detail}"
    )


def _validate_representable_pbc(source: Path) -> None:
    """Reject per-axis PBC that traditional DeePMD systems cannot encode."""
    false_values = {"f", "false", "0"}
    true_values = {"t", "true", "1"}
    frame_index = 0
    with source.open(encoding="utf-8") as stream:
        while line := stream.readline():
            try:
                natoms = int(line.strip())
            except ValueError:
                continue
            header = stream.readline()
            try:
                fields = shlex.split(header)
            except ValueError:
                fields = []  # Let dpdata report malformed extxyz syntax.
            for field in fields:
                key, separator, value = field.partition("=")
                if not separator or key.casefold() != "pbc":
                    continue
                flags = value.split()
                normalized = [flag.casefold() for flag in flags]
                recognized = all(
                    flag in false_values or flag in true_values for flag in normalized
                )
                if not recognized or len(flags) not in {1, 3}:
                    raise ValueError(
                        f"Frame {frame_index + 1} in extxyz file '{source}' has "
                        f"an invalid pbc field: '{value}'. Use one or three "
                        "boolean values."
                    )
                periodic = [flag in true_values for flag in normalized]
                if any(periodic) and not all(periodic):
                    raise ValueError(
                        f"Frame {frame_index + 1} in extxyz file '{source}' is "
                        f"partially periodic (pbc='{value}'). Traditional DeePMD "
                        "NumPy systems can represent only all-periodic or "
                        "all-nonperiodic boundary conditions."
                    )
                break
            for _ in range(natoms):
                stream.readline()
            frame_index += 1


def _read_extxyz_systems(source: Path) -> list[dpdata.LabeledSystem]:
    """Read every frame through dpdata and form fixed-shape DeePMD systems."""
    _validate_representable_pbc(source)

    # dpdata exposes formats through its plugin registry. Using the registered
    # extxyz reader here keeps aliases, units, and stress conversion in dpdata.
    from dpdata.format import (
        Format,
    )

    format_class = Format.get_formats().get("extxyz")
    if format_class is None:  # pragma: no cover - guaranteed by the dependency
        raise RuntimeError(
            f"dpdata {dpdata.__version__} does not provide its extxyz reader"
        )

    groups: OrderedDict[tuple[Any, ...], dpdata.LabeledSystem] = OrderedDict()
    frames = iter(
        format_class().from_multi_systems(str(source), stress_sign=_STRESS_SIGN)
    )
    frame_index = 0
    while True:
        try:
            frame_data = next(frames)
        except StopIteration:
            break
        except Exception as exc:
            raise _frame_error(source, frame_index, exc) from exc

        try:
            frame = dpdata.LabeledSystem(data=frame_data)
        except Exception as exc:
            raise _frame_error(source, frame_index, exc) from exc

        labels = tuple(_labels(frame))
        if "energy" not in labels:
            raise _frame_error(
                source, frame_index, ValueError("energies not found in data")
            )
        if "force" not in labels:
            raise _frame_error(
                source, frame_index, ValueError("forces not found in data")
            )

        # Canonicalizing type-map names lets dpdata append compatible
        # compositions while retaining the first frame's atom order.
        frame.sort_atom_names()
        key = (
            frame.uniq_formula,
            bool(frame.data.get("nopbc", False)),
            labels,
        )
        if key not in groups:
            groups[key] = frame
        else:
            try:
                groups[key].append(frame)
            except Exception as exc:
                raise ValueError(
                    f"Frame {frame_index + 1} in extxyz file '{source}' cannot "
                    "be combined safely with frames of the same composition. "
                    f"dpdata reported: {exc}"
                ) from exc
        frame_index += 1

    if frame_index == 0:
        raise ValueError(f"Extxyz file '{source}' contains no frames.")
    return list(groups.values())


def _write_cache(
    temporary: Path,
    source: Path,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    systems = _read_extxyz_systems(source)
    entries = []
    for index, system in enumerate(systems):
        relative = f"system.{index:03d}"
        system.to("deepmd/npy", str(temporary / relative), set_size=_SET_SIZE)
        entries.append(
            {
                "atom_names": list(system.data["atom_names"]),
                "atom_numbs": [int(value) for value in system.data["atom_numbs"]],
                "frames": int(system.get_nframes()),
                "labels": _labels(system),
                "nopbc": bool(system.data.get("nopbc", False)),
                "path": relative,
            }
        )

    manifest = {
        "fingerprint": fingerprint,
        "source": str(source),
        "systems": entries,
    }
    with (temporary / _CACHE_MANIFEST).open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def _remove_invalid_cache(path: Path, cache_root: Path) -> None:
    if not path.exists():
        return
    if path.resolve().parent != cache_root.resolve():
        raise RuntimeError(f"Refusing to remove cache outside '{cache_root}': {path}")
    shutil.rmtree(path)


def materialize_extxyz(
    path: str | os.PathLike[str],
) -> tuple[str, dict[str, Any]]:
    """Convert an extxyz file to an atomically published DeePMD NumPy cache."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Extxyz training-data file does not exist: '{source}'")

    fingerprint = _fingerprint(source, dpdata.__version__)
    digest = _fingerprint_digest(fingerprint)
    cache_root = _cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / digest

    cached = _valid_cache(target, fingerprint)
    if cached is not None:
        return str(target), cached[0]

    lock = cache_root / f".{digest}.lock"
    deadline = time.monotonic() + _LOCK_TIMEOUT
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            cached = _valid_cache(target, fingerprint)
            if cached is not None:
                return str(target), cached[0]
            try:
                lock_age = time.time() - lock.stat().st_mtime
                if lock_age > _STALE_LOCK_AGE:
                    lock.rmdir()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for extxyz cache creation for '{source}'. "
                    f"If no conversion is running, remove stale lock '{lock}'."
                ) from None
            time.sleep(0.1)

    temporary = cache_root / f".{digest}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        cached = _valid_cache(target, fingerprint)
        if cached is not None:
            return str(target), cached[0]

        _remove_invalid_cache(target, cache_root)
        temporary.mkdir()
        log.info("Converting extxyz training data %s with dpdata", source)
        manifest = _write_cache(temporary, source, fingerprint)
        if _file_sha256(source) != fingerprint["source_sha256"]:
            raise RuntimeError(
                f"Extxyz training-data file '{source}' changed while it was "
                "being converted. Retry after the file is no longer being written."
            )
        temporary.replace(target)
        return str(target), manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        try:
            lock.rmdir()
        except FileNotFoundError:
            pass


def _active(loss: dict[str, Any], *keys: str) -> bool:
    return any(float(loss.get(key, 0.0)) != 0.0 for key in keys)


def _validate_loss_labels(
    source: str, manifest: dict[str, Any], loss: dict[str, Any]
) -> None:
    loss_type = loss.get("type", "ener")
    if loss_type not in {"ener", "dens"}:
        raise ValueError(
            f"Extxyz training data '{source}' currently supports energy-model "
            f"losses only; configured loss type is '{loss_type}'."
        )

    unsupported = {
        "atomic energy": ("start_pref_ae", "limit_pref_ae"),
        "atomic preference": ("start_pref_pf", "limit_pref_pf"),
        "Hessian": ("start_pref_h", "limit_pref_h"),
        "generalized force": ("start_pref_gf", "limit_pref_gf"),
    }
    for label, keys in unsupported.items():
        if _active(loss, *keys):
            raise ValueError(
                f"The configured loss requires {label} labels, but dpdata's "
                f"extxyz reader does not convert them for '{source}'."
            )

    required = {"energy", "force"}
    if _active(loss, "start_pref_v", "limit_pref_v"):
        required.add("virial")
    for index, system in enumerate(manifest["systems"]):
        missing = sorted(required.difference(system["labels"]))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                f"Extxyz file '{source}' is missing required {missing_text} "
                f"label(s) in converted system {index}; the configured loss "
                "uses those labels. Supply virial or ASE-style stress when "
                "virial loss is enabled."
            )


def _normalize_dataset(
    dataset: dict[str, Any], loss: dict[str, Any], location: str
) -> None:
    systems_value = dataset.get("systems")
    if isinstance(systems_value, str):
        systems = [systems_value]
        scalar = True
    elif isinstance(systems_value, list):
        systems = systems_value
        scalar = False
    else:
        return

    extxyz_indices = [
        index for index, path in enumerate(systems) if is_extxyz_path(path)
    ]
    if not extxyz_indices:
        return

    if any(_is_lmdb_path(path) for path in systems):
        raise ValueError(
            f"{location}/systems cannot mix extxyz and LMDB inputs. LMDB is "
            "supported only as a single systems path; convert the inputs to one "
            "representation first."
        )

    normalized = list(systems)
    expands_to_multiple = False
    for index in extxyz_indices:
        source = systems[index]
        cache, manifest = materialize_extxyz(source)
        _validate_loss_labels(source, manifest, loss)
        normalized[index] = cache
        expands_to_multiple |= len(manifest["systems"]) > 1

    if expands_to_multiple:
        if isinstance(dataset.get("batch_size"), list):
            raise ValueError(
                f"{location}/batch_size cannot be a list when one extxyz file "
                "expands into multiple fixed-shape systems. Use a scalar or "
                "'auto' batch size."
            )
        if dataset.get("sys_probs") is not None:
            raise ValueError(
                f"{location}/sys_probs is ambiguous when one extxyz file expands "
                "into multiple fixed-shape systems. Omit it and use auto_prob."
            )
        if ";" in dataset.get("auto_prob", ""):
            raise ValueError(
                f"{location}/auto_prob cannot use indexed blocks when one extxyz "
                "file expands into multiple fixed-shape systems."
            )

    dataset["systems"] = normalized[0] if scalar else normalized


def normalize_extxyz_training_data(
    data: dict[str, Any], *, multi_task: bool = False
) -> dict[str, Any]:
    """Materialize explicit extxyz paths on a normalized configuration copy."""
    training = data.get("training")
    if not isinstance(training, dict):
        return data

    if multi_task:
        data_dict = training.get("data_dict", {})
        bindings = [
            (
                task_data,
                data.get("loss_dict", {}).get(task, {"type": "ener"}),
                f"training/data_dict/{task}",
            )
            for task, task_data in data_dict.items()
        ]
    else:
        bindings = [(training, data.get("loss", {"type": "ener"}), "training")]

    if not any(
        is_extxyz_path(path)
        for task_data, _, _ in bindings
        for name in ("training_data", "validation_data")
        if isinstance(task_data.get(name), dict)
        for path in (
            [task_data[name]["systems"]]
            if isinstance(task_data[name].get("systems"), str)
            else task_data[name].get("systems", [])
        )
    ):
        return data

    result = deepcopy(data)
    if multi_task:
        result_bindings = [
            (
                task_data,
                result.get("loss_dict", {}).get(task, {"type": "ener"}),
                f"training/data_dict/{task}",
            )
            for task, task_data in result["training"].get("data_dict", {}).items()
        ]
    else:
        result_bindings = [
            (
                result["training"],
                result.get("loss", {"type": "ener"}),
                "training",
            )
        ]

    for task_data, loss, location in result_bindings:
        for name in ("training_data", "validation_data"):
            dataset = task_data.get(name)
            if isinstance(dataset, dict):
                _normalize_dataset(dataset, loss, f"{location}/{name}")
    return result
