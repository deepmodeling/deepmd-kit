# SPDX-License-Identifier: LGPL-3.0-or-later
"""Cached dpdata conversion independent of dataset discovery and loaders.

This layer owns conversion manifests and writer locks. Path routing supplies
native-format discovery after conversion, keeping the dependency graph acyclic.
"""

import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import (
    Callable,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

from filelock import (
    FileLock,
)

log = logging.getLogger(__name__)

_DPDATA_CACHE_DIR = ".deepmd_dpdata_cache"
_DPDATA_DEFAULT_OUT_FORMAT = "deepmd/lmdb"
_DPDATA_CONVERSION_SCHEMA_VERSION = "3"
_DPDATA_CONVERSION_CACHE: dict[tuple[str, str, str, str, str], list[str]] = {}
_DPDATA_SOURCE_HASH_LIMIT = 1024 * 1024
_CONVERSION_LOCK_HEARTBEAT_SECONDS = 5.0
# Allow for shared-filesystem attribute caching before diagnosing a stalled
# heartbeat. Remote owner liveness still cannot be inferred from elapsed time.
_CONVERSION_LOCK_STALE_SECONDS = 300.0


def _is_deepmd_data_format(fmt: str) -> bool:
    """Recognize native formats that bypass the external conversion layer."""
    return fmt in {
        "deepmd",
        "deepmd/raw",
        "deepmd/npy",
        "deepmd/comp",
        "deepmd/npy/mixed",
        "deepmd/hdf5",
        "deepmd/lmdb",
        "lmdb",
    }


def _is_dpdata_lmdb_format(fmt: str) -> bool:
    """Return whether *fmt* names dpdata's DeePMD-compatible LMDB format."""
    return fmt in {"deepmd/lmdb", "lmdb"}


def _canonical_dpdata_out_format(out_fmt: str | None) -> str:
    """Return the canonical dpdata output format used by conversion caches."""
    if out_fmt is None:
        return _DPDATA_DEFAULT_OUT_FORMAT
    out_fmt = out_fmt.lower()
    return _DPDATA_DEFAULT_OUT_FORMAT if out_fmt == "lmdb" else out_fmt


def conversion_will_write_lmdb(data_config: dict[str, Any]) -> bool:
    """Whether a non-DeePMD input config will be converted to LMDB."""
    data_format = data_config.get("format")
    if data_format is None or _is_deepmd_data_format(data_format.lower()):
        return False
    out_format = data_config.get("out_format", data_config.get("output_format"))
    return _is_dpdata_lmdb_format(_canonical_dpdata_out_format(out_format))


def _looks_like_extxyz(path: Path) -> bool:
    """Probe the XYZ comment line without requiring a valid text input."""
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as fp:
            fp.readline()
            comment = fp.readline()
    except (OSError, UnicodeDecodeError):
        return False
    return "Properties=" in comment or "Lattice=" in comment


def _normalize_dpdata_format(fmt: str, source: Path) -> str:
    """Resolve aliases and infer common dpdata formats from the input path."""
    fmt = fmt.lower()
    if fmt == "ase":
        return "ase/structure"
    if fmt != "auto":
        return fmt
    suffix = source.suffix.lower().lstrip(".")
    if suffix == "traj":
        return "ase/traj"
    if suffix == "extxyz" or (suffix == "xyz" and _looks_like_extxyz(source)):
        return "extxyz"
    return suffix or fmt


def _iter_conversion_inputs(path: str, patterns: list[str] | None) -> list[str]:
    if patterns is None:
        return [path]
    root = Path(path)
    if not root.is_dir():
        return [path]
    matches = []
    for pattern in patterns:
        matches.extend(str(match) for match in root.rglob(pattern))
    return sorted(set(matches))


def _conversion_cache_path(source: Path, fmt: str, out_fmt: str) -> Path:
    source_resolved = source.resolve(strict=False)
    try:
        dpdata_version = importlib.metadata.version("dpdata")
    except importlib.metadata.PackageNotFoundError:
        dpdata_version = "unknown"
    digest = hashlib.sha256(
        (
            f"{source_resolved}|{fmt}|{out_fmt}|"
            f"schema={_DPDATA_CONVERSION_SCHEMA_VERSION}|dpdata={dpdata_version}"
        ).encode()
    ).hexdigest()[:16]
    stem = source_resolved.stem or source_resolved.name or "dataset"
    safe_out_fmt = out_fmt.replace("/", "-")
    suffix = ".lmdb" if _is_dpdata_lmdb_format(out_fmt) else ""
    return Path.cwd() / _DPDATA_CACHE_DIR / f"{stem}-{safe_out_fmt}-{digest}{suffix}"


def _source_signature(source: Path, output: Path) -> str:
    """Fingerprint source metadata, including restored files and tree changes.

    Compare nanosecond timestamps and sizes for equality, never ordering.
    ctime also detects a same-size rewrite whose mtime was preserved on Unix.
    Hash small files as well, including on Windows where ctime is creation time,
    without rereading large trajectory files on every routing call. Scan on
    each call: a time-based memo can hide edits to directory children.
    Exclude our cache subtree so publication cannot invalidate its own input.
    """
    digest = hashlib.sha256()
    cache_dir = output.parent.resolve()

    def add_file(path: Path, name: str) -> None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            if not path.is_symlink():
                # A disappearing regular file is an incomplete source scan.
                raise
            # Broken links may be unrelated to the files dpdata reads. Track
            # the link itself, but keep following valid links so target edits
            # still invalidate the converted dataset.
            stat = path.lstat()
            digest.update(
                json.dumps(
                    [
                        "dangling_symlink",
                        name,
                        str(path.readlink()),
                        stat.st_mode,
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_ctime_ns,
                    ]
                ).encode()
            )
            return
        digest.update(
            json.dumps(
                [name, stat.st_mode, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            ).encode()
        )
        if stat.st_size <= _DPDATA_SOURCE_HASH_LIMIT:
            with path.open("rb") as fp:
                # Bound the read even if a concurrent writer grows the input.
                digest.update(
                    hashlib.sha256(fp.read(_DPDATA_SOURCE_HASH_LIMIT + 1)).digest()
                )

    if source.is_file():
        add_file(source, source.name)
    elif source.is_dir():

        def raise_walk_error(error: OSError) -> None:
            # An incomplete scan must never validate an old conversion.
            raise error

        visited: set[tuple[int, int]] = set()
        for root, dirs, files in os.walk(
            source, onerror=raise_walk_error, followlinks=True
        ):
            directory = Path(root)
            stat = directory.stat()
            identity = (stat.st_dev, stat.st_ino)
            relative_root = directory.relative_to(source)
            # Track both retargeting and edits below directory links. Inode
            # deduplication prevents cycles and repeated scans of aliases.
            digest.update(
                json.dumps(
                    [
                        str(relative_root),
                        identity,
                        str(directory.readlink()) if directory.is_symlink() else None,
                    ]
                ).encode()
            )
            if identity in visited:
                dirs[:] = []
                continue
            visited.add(identity)
            dirs[:] = sorted(
                name for name in dirs if (Path(root) / name).resolve() != cache_dir
            )
            digest.update(json.dumps([str(relative_root), dirs]).encode())
            for name in sorted(files):
                add_file(Path(root) / name, str(relative_root / name))
    else:
        raise FileNotFoundError(f"dpdata input does not exist: {source}")
    return digest.hexdigest()


def _conversion_manifest_path(output: Path) -> Path:
    """Return the sidecar recording the source used for a published output."""
    return output.with_name(output.name + ".source.json")


def _is_conversion_current(source: Path, output: Path) -> bool:
    """Reuse only a complete conversion with exactly matching source metadata."""
    if not output.exists():
        return False
    try:
        manifest = json.loads(_conversion_manifest_path(output).read_text())
        return manifest == {"source_signature": _source_signature(source, output)}
    except (OSError, ValueError):
        # Missing, damaged, or legacy manifests require a fresh conversion.
        return False


def _write_conversion_manifest(output: Path, signature: str) -> None:
    """Atomically mark a successfully published conversion as reusable."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as fp:
            temporary = Path(fp.name)
            json.dump({"source_signature": signature}, fp)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temporary, _conversion_manifest_path(output))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _process_start_time(pid: int) -> str | None:
    """Return Linux's stable process start token, if available."""
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields_after_name = stat_text.rsplit(")", 1)[1].split()
    return fields_after_name[19] if len(fields_after_name) > 19 else None


def _same_lock_file(lock_path: Path, expected: os.stat_result) -> bool:
    """Whether *lock_path* still names the inode originally acquired/read."""
    try:
        current = lock_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (current.st_dev, current.st_ino) == (expected.st_dev, expected.st_ino)


def _conversion_lock_guard(lock_path: Path) -> FileLock:
    """Serialize all changes to the owner file with an OS-managed lock.

    Filelock owns the guard's inode lifecycle; callers must never unlink it.
    The OS releases the guard if a process crashes, so owner-file recovery
    runs exclusively without implementing another stale-lock protocol.
    """
    return FileLock(str(lock_path) + ".guard")


class _ConversionLock:
    """Owned conversion lock with serialized creation, recovery and release."""

    def __init__(self, lock_path: Path) -> None:
        self.path = lock_path
        with _conversion_lock_guard(lock_path), lock_path.open("x") as lock_file:
            try:
                self._stat = os.fstat(lock_file.fileno())
                payload = {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "process_start": _process_start_time(os.getpid()),
                    "created": time.time(),
                }
                # Publish a complete owner before a waiter can inspect it.
                json.dump(payload, lock_file)
                lock_file.flush()
                os.fsync(lock_file.fileno())
                self._stop = threading.Event()
                self._thread = threading.Thread(
                    target=self._heartbeat,
                    name="deepmd-dpdata-conversion-lock",
                    daemon=True,
                )
                self._thread.start()
            except BaseException:
                lock_file.close()
                # Creation and cleanup share the guard, so no successor can
                # have acquired this pathname even if initialization failed.
                self.path.unlink(missing_ok=True)
                raise

    def _heartbeat(self) -> None:
        while not self._stop.wait(_CONVERSION_LOCK_HEARTBEAT_SECONDS):
            with _conversion_lock_guard(self.path):
                if not _same_lock_file(self.path, self._stat):
                    return
                os.utime(self.path, None, follow_symlinks=False)

    def release(self) -> None:
        """Stop heartbeating and remove only the lock inode we own."""
        self._stop.set()
        self._thread.join()
        with _conversion_lock_guard(self.path):
            if _same_lock_file(self.path, self._stat):
                self.path.unlink(missing_ok=True)


def _lock_owner_is_alive(payload: dict[str, Any]) -> bool | None:
    """Return owner liveness locally, or None for another host/invalid data."""
    if payload.get("hostname") != socket.gethostname():
        return None
    try:
        pid = int(payload["pid"])
    except (KeyError, TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    expected_start = payload.get("process_start")
    current_start = _process_start_time(pid)
    if expected_start is not None and current_start is not None:
        return str(expected_start) == current_start
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recover_stale_conversion_lock(lock_path: Path) -> bool:
    """Recover dead local owners and abandoned, uninitialized lock files.

    A remote host's heartbeat may look stale because of NFS attribute caching
    or clock skew. Never evict an identified remote writer on that evidence;
    report the stalled lock for owner verification instead of racing it.
    """
    # Hold the same guard as creation/release across both the inode check
    # and unlink. Another recovering waiter cannot publish a live successor
    # into the check-to-unlink window.
    with _conversion_lock_guard(lock_path):
        try:
            lock_stat = lock_path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        try:
            payload = json.loads(lock_path.read_text())
        except (OSError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        owner_alive = _lock_owner_is_alive(payload)
        lease_expired = (
            time.time() - lock_stat.st_mtime > _CONVERSION_LOCK_STALE_SECONDS
        )
        if (
            owner_alive is None
            and lease_expired
            and payload.get("hostname") not in (None, socket.gethostname())
        ):
            raise RuntimeError(
                f"Cannot verify the remote owner of stalled dpdata conversion lock "
                f"{lock_path} on {payload['hostname']}. Check that the owner has "
                "stopped before removing the lock and retrying."
            )
        stale = owner_alive is False or (owner_alive is None and lease_expired)
        if not stale or not _same_lock_file(lock_path, lock_stat):
            return False
        log.warning("Recovering stale dpdata conversion lock %s", lock_path)
        lock_path.unlink(missing_ok=True)
        return True


def _wait_for_conversion(source: Path, output: Path, lock_path: Path) -> bool:
    """Wait without rescanning the source tree while a valid writer owns it."""
    while lock_path.exists():
        if _recover_stale_conversion_lock(lock_path):
            continue
        time.sleep(1.0)
    # Freshness is checked once after publication, not once per waiter-second.
    return _is_conversion_current(source, output)


def _remove_path(path: Path) -> None:
    # Check links before directories: Path.is_dir follows a directory symlink,
    # while cache cleanup must never recurse into a target outside the cache.
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _publish_conversion_output(tmp_output: Path, output: Path) -> None:
    """Publish a non-LMDB conversion without discarding a valid old cache."""
    if not output.exists() or output.is_symlink() or not output.is_dir():
        os.replace(tmp_output, output)
        backup = output.with_name(f".{output.name}.backup")
        _remove_path(backup)
        return

    # POSIX cannot replace a non-empty directory directly. Preserve the old
    # cache under a sibling name and restore it if the second rename fails.
    backup = output.with_name(f".{output.name}.backup")
    _remove_path(backup)
    os.replace(output, backup)
    try:
        os.replace(tmp_output, output)
    except Exception:
        os.replace(backup, output)
        raise
    else:
        _remove_path(backup)


def _write_dpdata_conversion(
    source: Path, fmt: str, out_fmt: str, output: Path
) -> None:
    """Load *source* with dpdata and publish it in a DeePMD format.

    dpdata 1.1.0 makes ``deepmd/lmdb`` writes transactional: it stages and
    validates the complete database before atomically publishing it. Use that
    writer directly so DeePMD-kit does not duplicate or weaken its overwrite
    guarantees. Other dpdata formats do not share that contract, so they keep
    the cache-level temporary output used for failure isolation.
    """
    try:
        import dpdata
    except ImportError as exc:
        raise ImportError(
            "dpdata is required when training_data.format or "
            "validation_data.format is specified. Install dpdata to enable "
            "automatic dataset conversion."
        ) from exc

    multi_systems = dpdata.MultiSystems()
    try:
        multi_systems.load_systems_from_file(str(source), fmt=fmt)
    except (NotImplementedError, ValueError) as labeled_error:
        # dpdata 1.1 exposes an explicit unlabeled path. This matters for
        # structure-only EXTXYZ/ASE inputs, which are valid descriptor data
        # even though a supervised loss may later require labels.
        unlabeled_systems = dpdata.MultiSystems()
        try:
            unlabeled_systems.load_systems_from_file(
                str(source), fmt=fmt, labeled=False
            )
        except (NotImplementedError, TypeError, ValueError):
            try:
                labeled_system = dpdata.LabeledSystem(str(source), fmt=fmt)
            except Exception:
                raise labeled_error from None
            multi_systems = dpdata.MultiSystems(labeled_system)
        else:
            log.info("Loaded unlabeled dpdata input %s using format %s", source, fmt)
            multi_systems = unlabeled_systems
    if len(multi_systems) == 0:
        raise RuntimeError(f"No frames were loaded by dpdata from {source}")

    if _is_dpdata_lmdb_format(out_fmt):
        multi_systems.to(out_fmt, str(output), overwrite=True)
        return

    tmp_output = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    _remove_path(tmp_output)
    try:
        multi_systems.to(out_fmt, str(tmp_output))
        _publish_conversion_output(tmp_output, output)
    except Exception:
        _remove_path(tmp_output)
        raise


def _convert_system_by_dpdata(
    source_path: str,
    fmt: str,
    out_fmt: str | None,
    *,
    expand_systems: Callable[[str], list[str]],
) -> list[str]:
    """Convert one source while keeping native discovery outside this module.

    The caller supplies discovery for HDF5/NPY outputs so conversion does not
    import loaders or their path-routing module, including during startup.
    """
    source = Path(source_path)
    fmt = _normalize_dpdata_format(fmt, source)
    out_fmt = _canonical_dpdata_out_format(out_fmt)
    output = _conversion_cache_path(source, fmt, out_fmt)
    cache_key = (
        str(Path.cwd().resolve(strict=False)),
        str(source.resolve(strict=False)),
        fmt,
        out_fmt,
        str(output),
    )
    cached_systems = _DPDATA_CONVERSION_CACHE.get(cache_key)
    conversion_current = _is_conversion_current(source, output)
    if cached_systems is not None:
        if conversion_current:
            return cached_systems
        # A long-lived training/validation process may observe source files
        # rewritten in place. Drop the fast-path entry so the normal locked
        # conversion flow refreshes the on-disk result before it is reused.
        del _DPDATA_CONVERSION_CACHE[cache_key]

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(output.suffix + ".lock")
    # Reuse only this pre-lock check. Waiting, acquiring the lock and writing
    # each require fresh scans because the source can change between phases.
    if not conversion_current:
        while True:
            try:
                conversion_lock = _ConversionLock(lock_path)
            except FileExistsError:
                if _wait_for_conversion(source, output, lock_path):
                    break
                continue
            else:
                try:
                    if not _is_conversion_current(source, output):
                        log.info(
                            "Converting %s from dpdata format %s to %s at %s",
                            source,
                            fmt,
                            out_fmt,
                            output,
                        )
                        signature = _source_signature(source, output)
                        # Invalidate before replacing output: a crash must never
                        # pair a new dataset with an older source manifest.
                        _conversion_manifest_path(output).unlink(missing_ok=True)
                        _write_dpdata_conversion(source, fmt, out_fmt, output)
                        if signature != _source_signature(source, output):
                            raise RuntimeError(
                                f"dpdata input changed during conversion: {source}. "
                                "Retry with an unchanged input dataset."
                            )
                        _write_conversion_manifest(output, signature)
                finally:
                    conversion_lock.release()
                break

    if _is_dpdata_lmdb_format(out_fmt):
        converted_systems = [str(output)]
    else:
        converted_systems = expand_systems(str(output))
    if not converted_systems:
        raise RuntimeError(f"No DeePMD systems were found in converted file {output}")
    _DPDATA_CONVERSION_CACHE[cache_key] = converted_systems
    return converted_systems
