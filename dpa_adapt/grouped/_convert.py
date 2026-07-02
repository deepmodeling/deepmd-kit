# SPDX-License-Identifier: LGPL-3.0-or-later
"""Materialize grouped auxiliaries for existing deepmd/npy systems.

Several upstream builders (e.g. the OER ``O*/OH*/OOH*`` adsorption datasets)
already write mixed-type ``real_atom_types.npy`` with ``-1`` marking masked /
virtual atoms and one shared label per frame group, but do **not** write the
``group_id`` / ``pool_mask`` files that the grouped training route
(:mod:`deepmd.pt.utils.grouped`, ``strategy="finetune"``) needs:

* without ``group_id.npy`` the finetuner never routes into the grouped path and,
  even if it did, the loss falls back to one group per frame -- so the shared
  label is never aggregated across the group;
* without ``pool_mask.npy`` the model pools the ``-1`` virtual atoms in with the
  real atoms (``pool_mask`` defaults to all-ones), so e.g. ``O*``/``OH*``/``OOH*``
  frames become nearly indistinguishable.

This module fills that gap **in place**, deriving ``pool_mask`` from the
mixed-type ``real_atom_types`` and assigning ``group_id`` by a configurable
policy, without touching the existing coord/box/label data.  The files it writes
match :func:`dpa_adapt.grouped._core._write_group_system` exactly (``group_id``
shape ``(nframes,)`` int64, ``pool_mask`` shape ``(nframes, natoms)`` float64).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from glob import has_magic
from glob import glob as _glob
from pathlib import Path
from collections.abc import Iterable

import numpy as np

from dpa_adapt.data.errors import DPADataError

GROUP_ID_KEY = "group_id"
WEIGHT_KEY = "weight"
POOL_MASK_KEY = "pool_mask"
REAL_ATYPE_KEY = "real_atom_types"


@dataclass
class GroupMarkerResult:
    """Per-system summary of what :func:`add_group_markers` wrote."""

    system: Path
    n_frames: int = 0
    n_groups: int = 0
    wrote_group_id: bool = False
    wrote_pool_mask: bool = False
    wrote_weight: bool = False
    set_dirs: list[Path] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""


def add_group_markers(
    data: str | Path | Iterable[str | Path],
    *,
    group_by: str | int = "system",
    property_name: str = "property",
    weight: float | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[GroupMarkerResult]:
    """Write ``group_id`` / ``pool_mask`` markers into deepmd/npy systems.

    Parameters
    ----------
    data
        A system directory (one that directly contains ``set.*/``), a parent
        directory / glob / arbitrarily deep tree containing many such systems,
        or an iterable mixing any of the above.  The tree is searched
        recursively for every directory that directly holds ``set.*`` subdirs.
    group_by
        How frames are grouped **within one system** (frame order follows the
        DeepMD convention: sorted ``set.*`` directories concatenated):

        * ``"system"`` (default) -- every frame in the system is one group
          (group id ``0``).  Matches "one system directory == one group", the
          layout emitted by the OER writer (one equation directory holds the
          ``O*``/``OH*``/``OOH*`` triplet).
        * ``"label"`` -- frames whose ``property_name`` label rows are equal
          form a group; distinct label rows get ids ``0, 1, 2, ...`` in
          first-appearance order.  Use when several groups were merged into one
          system.
        * ``int`` -- fixed group size: every ``group_by`` consecutive frames
          form a group.  A trailing remainder becomes a smaller final group.
    property_name
        Label key read from ``set.*/{property_name}.npy`` when
        ``group_by="label"``.  Ignored otherwise.
    weight
        When not ``None`` a constant ``weight.npy`` of this value is written for
        every frame.  Rarely needed: the grouped model defaults missing weights
        to ``1.0`` (an unweighted sum over the group).
    overwrite
        When ``False`` (default) an existing ``group_id.npy`` / ``pool_mask.npy``
        / ``weight.npy`` is left untouched; only missing files are written.
        ``True`` regenerates them (the derivation is deterministic).
    dry_run
        Compute and report what would be written without touching disk.

    Returns
    -------
    list[GroupMarkerResult]
        One entry per discovered system.
    """
    systems = _discover_systems(data)
    if not systems:
        raise DPADataError(
            f"No deepmd system directories (containing set.*/) found under {data!r}."
        )
    if isinstance(group_by, bool) or (
        not isinstance(group_by, int) and group_by not in {"system", "label"}
    ):
        raise DPADataError(
            f"group_by must be 'system', 'label', or a positive int; got {group_by!r}."
        )
    if isinstance(group_by, int) and group_by < 1:
        raise DPADataError(f"group_by size must be >= 1; got {group_by}.")

    return [
        _process_system(
            sysdir,
            group_by=group_by,
            property_name=property_name,
            weight=weight,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        for sysdir in systems
    ]


# ---------------------------------------------------------------------------
# system discovery
# ---------------------------------------------------------------------------


def _as_paths(data: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(data, (str, Path)):
        raw = str(data)
        if has_magic(raw):
            return [Path(p) for p in sorted(_glob(raw))]
        return [Path(raw)]
    if isinstance(data, Iterable):
        out: list[Path] = []
        for item in data:
            out.extend(_as_paths(item))
        return out
    raise DPADataError(f"Unsupported data spec: {data!r}.")


def _is_system_dir(path: Path) -> bool:
    return path.is_dir() and any(child.is_dir() for child in path.glob("set.*"))


def _discover_systems(data: str | Path | Iterable[str | Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for root in _as_paths(data):
        for sysdir in _walk_systems(root):
            resolved = sysdir.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(sysdir)
    return found


def _walk_systems(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    if _is_system_dir(root):
        yield root
        return
    if not root.is_dir():
        return
    for dirpath, dirnames, _ in os.walk(root):
        path = Path(dirpath)
        if _is_system_dir(path):
            yield path
            # Do not descend into this system's own set.* directories.
            dirnames[:] = [d for d in dirnames if not d.startswith("set.")]


# ---------------------------------------------------------------------------
# per-system processing
# ---------------------------------------------------------------------------


def _process_system(
    sysdir: Path,
    *,
    group_by: str | int,
    property_name: str,
    weight: float | None,
    overwrite: bool,
    dry_run: bool,
) -> GroupMarkerResult:
    set_dirs = sorted(d for d in sysdir.glob("set.*") if d.is_dir())
    if not set_dirs:
        return GroupMarkerResult(sysdir, skipped=True, reason="no set.* directories")

    per_set: list[tuple[Path, int, int, np.ndarray | None]] = []
    for set_dir in set_dirs:
        nframes, natoms, real_types = _read_set_shape(set_dir)
        if nframes is None:
            return GroupMarkerResult(
                sysdir, skipped=True, reason=f"{set_dir.name} has no coord/real_atom_types"
            )
        per_set.append((set_dir, nframes, natoms, real_types))

    total = sum(n for _, n, _, _ in per_set)
    group_ids = _assign_group_ids(group_by, per_set, property_name)

    result = GroupMarkerResult(
        sysdir, n_frames=total, n_groups=len(np.unique(group_ids))
    )
    offset = 0
    for set_dir, nframes, natoms, real_types in per_set:
        result.set_dirs.append(set_dir)
        gid_slice = group_ids[offset : offset + nframes].astype(np.int64, copy=False)
        if _should_write(set_dir / f"{GROUP_ID_KEY}.npy", overwrite):
            if not dry_run:
                np.save(set_dir / f"{GROUP_ID_KEY}.npy", gid_slice)
            result.wrote_group_id = True

        if real_types is not None and bool((real_types < 0).any()):
            if _should_write(set_dir / f"{POOL_MASK_KEY}.npy", overwrite):
                pool_mask = (real_types >= 0).astype(np.float64)
                if not dry_run:
                    np.save(set_dir / f"{POOL_MASK_KEY}.npy", pool_mask)
                result.wrote_pool_mask = True

        if weight is not None and _should_write(
            set_dir / f"{WEIGHT_KEY}.npy", overwrite
        ):
            if not dry_run:
                np.save(
                    set_dir / f"{WEIGHT_KEY}.npy",
                    np.full((nframes,), float(weight), dtype=np.float64),
                )
            result.wrote_weight = True
        offset += nframes
    return result


def _should_write(path: Path, overwrite: bool) -> bool:
    return overwrite or not path.is_file()


def _read_set_shape(set_dir: Path) -> tuple[int | None, int, np.ndarray | None]:
    """Return ``(nframes, natoms, real_atom_types|None)`` for one set directory."""
    real_path = set_dir / f"{REAL_ATYPE_KEY}.npy"
    if real_path.is_file():
        real_types = np.load(real_path)
        if real_types.ndim != 2:
            raise DPADataError(
                f"{real_path} has shape {real_types.shape}; expected (nframes, natoms)."
            )
        return int(real_types.shape[0]), int(real_types.shape[1]), real_types

    coord_path = set_dir / "coord.npy"
    if coord_path.is_file():
        coord = np.load(coord_path, mmap_mode="r")
        nframes = int(coord.shape[0])
        natoms = int(coord.shape[1] // 3) if coord.ndim == 2 else int(coord.shape[1])
        return nframes, natoms, None
    return None, 0, None


def _assign_group_ids(
    group_by: str | int,
    per_set: list[tuple[Path, int, int, np.ndarray | None]],
    property_name: str,
) -> np.ndarray:
    total = sum(n for _, n, _, _ in per_set)
    if group_by == "system":
        return np.zeros(total, dtype=np.int64)
    if isinstance(group_by, int):
        return np.arange(total, dtype=np.int64) // group_by
    # group_by == "label"
    labels = _load_system_label(per_set, property_name)
    if labels.shape[0] != total:
        raise DPADataError(
            f"label rows ({labels.shape[0]}) do not match frame count ({total})."
        )
    ids = np.empty(total, dtype=np.int64)
    seen: dict[tuple, int] = {}
    nxt = 0
    for i, row in enumerate(labels):
        key = tuple(np.round(np.asarray(row, dtype=float), 8).tolist())
        if key not in seen:
            seen[key] = nxt
            nxt += 1
        ids[i] = seen[key]
    return ids


def _load_system_label(
    per_set: list[tuple[Path, int, int, np.ndarray | None]],
    property_name: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for set_dir, nframes, _, _ in per_set:
        label_path = set_dir / f"{property_name}.npy"
        if not label_path.is_file():
            raise DPADataError(
                f"group_by='label' needs {label_path}, which is missing. "
                f"Pass the correct property_name (got {property_name!r})."
            )
        arr = np.load(label_path)
        chunks.append(arr.reshape(nframes, -1))
    return np.concatenate(chunks, axis=0)


def _main() -> None:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Add group_id / pool_mask markers to mixed-type deepmd/npy systems "
            "so they can be trained via the grouped route (strategy='finetune')."
        )
    )
    parser.add_argument("data", nargs="+", help="System dir(s), parent tree(s), or glob(s).")
    parser.add_argument(
        "--group-by",
        default="system",
        help="'system' (default), 'label', or an integer group size.",
    )
    parser.add_argument("--property-name", default="property")
    parser.add_argument("--weight", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    group_by: str | int = args.group_by
    if isinstance(group_by, str) and group_by.isdigit():
        group_by = int(group_by)

    results = add_group_markers(
        args.data,
        group_by=group_by,
        property_name=args.property_name,
        weight=args.weight,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    n_written = sum(1 for r in results if r.wrote_group_id or r.wrote_pool_mask)
    print(  # noqa: T201
        f"{'[dry-run] ' if args.dry_run else ''}"
        f"{len(results)} systems discovered; "
        f"{n_written} updated; "
        f"{sum(r.n_groups for r in results)} groups total."
    )
    for r in results[:10]:
        print(  # noqa: T201
            f"  {r.system}: frames={r.n_frames} groups={r.n_groups} "
            f"group_id={r.wrote_group_id} pool_mask={r.wrote_pool_mask}"
            + (f" SKIPPED({r.reason})" if r.skipped else "")
        )
    if len(results) > 10:
        print(f"  ... (+{len(results) - 10} more)")  # noqa: T201


if __name__ == "__main__":  # pragma: no cover
    _main()
