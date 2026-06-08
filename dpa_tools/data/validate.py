# data/validate.py
#
# Content-level sanity checks for dpdata systems.
#
# Scope: flag things that are almost certainly bugs (NaN/Inf, degenerate
# cells, misaligned frame counts) plus two coarse magnitude bounds. This is
# NOT anomaly detection — it does not look for statistical outliers.

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, NamedTuple, Union

import numpy as np

from dpa_tools.data.errors import DPADataError

# Magnitude sanity thresholds — values past these are almost never real.
_ENERGY_MAX_EV_PER_ATOM = 1000.0
_FORCE_MAX_EV_PER_ANGSTROM = 100.0

# A box matrix with |det| below this is treated as degenerate.
_BOX_DET_TOLERANCE = 1e-10


class Issue(NamedTuple):
    """A single data-quality finding from check_data()."""

    severity: Literal["warning", "error"]
    system: str          # system identifier (source path or hash)
    set_dir: str         # always "" for dpdata systems (no set.* granularity)
    file: str            # data key the issue concerns, e.g. "energies"
    description: str     # human-readable explanation


def _check_system(
    system, identifier: str, box_det_tol: float,
) -> list[Issue]:
    """Run all content checks on a single dpdata system."""
    issues: list[Issue] = []
    name = identifier

    def _issue(severity: str, file: str, description: str) -> Issue:
        return Issue(severity, name, "", file, description)

    d = system.data
    coords = np.asarray(d.get("coords"))
    cells_raw = d.get("cells")
    energies = d.get("energies")
    forces = d.get("forces")

    # --- normalise cells to (n_frames, 3, 3) ---
    # dpdata versions differ: some return (n_frames, 9), others (n_frames, 3, 3).
    # Reshape explicitly so downstream checks see a uniform layout.
    cells = None
    if cells_raw is not None:
        cells = np.asarray(cells_raw)
        if cells.ndim == 2 and cells.shape[1] == 9:
            try:
                cells = cells.reshape(-1, 3, 3)
            except ValueError as exc:
                raise DPADataError(
                    f"Cannot reshape cells of shape {cells_raw.shape} to "
                    f"(-1, 3, 3): {exc}"
                ) from exc
        elif cells.ndim == 3 and cells.shape[1:] == (3, 3):
            pass  # already canonical
        else:
            raise DPADataError(
                f"Unexpected cells shape {cells_raw.shape!r}. "
                "Expected (n_frames, 9) or (n_frames, 3, 3)."
            )

    # --- NaN / Inf ---
    for key, arr in [("energies", energies), ("forces", forces), ("cells", cells)]:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.count_nonzero(~np.isfinite(arr)))
            issues.append(_issue(
                "error", key,
                f"{key}: contains {n_bad} non-finite value(s) (NaN or Inf).",
            ))

    # --- degenerate box (|det| below tolerance) ---
    if cells is not None and np.all(np.isfinite(cells)):
        dets = np.abs(np.linalg.det(cells))
        for fi in np.where(dets < box_det_tol)[0]:
            issues.append(_issue(
                "error", "cells",
                f"cells: frame {int(fi)} has |det| = {dets[fi]:.2e} "
                f"(< tol {box_det_tol:.0e}), likely degenerate box.",
            ))

    # --- energy magnitude (per atom) ---
    if energies is not None and coords is not None and coords.ndim >= 2:
        energies = np.asarray(energies)
        if np.all(np.isfinite(energies)):
            n_atoms = coords.shape[1]  # dpdata coords: (n_frames, n_atoms, 3)
            if n_atoms > 0:
                per_atom = np.abs(energies) / n_atoms
                for fi in np.where(per_atom > _ENERGY_MAX_EV_PER_ATOM)[0]:
                    issues.append(_issue(
                        "warning", "energies",
                        f"energies: frame {int(fi)} has |E/atom| = "
                        f"{per_atom[fi]:.1f} eV/atom "
                        f"(> {_ENERGY_MAX_EV_PER_ATOM:.0f}); suspicious magnitude.",
                    ))

    # --- force magnitude (per component) ---
    if forces is not None:
        forces = np.asarray(forces)
        if np.all(np.isfinite(forces)):
            abs_f = np.abs(forces)
            per_frame_max = abs_f.max(axis=tuple(range(1, abs_f.ndim)))
            for fi in np.where(per_frame_max > _FORCE_MAX_EV_PER_ANGSTROM)[0]:
                issues.append(_issue(
                    "warning", "forces",
                    f"forces: frame {int(fi)} has a force component of "
                    f"{per_frame_max[fi]:.1f} eV/Ang "
                    f"(> {_FORCE_MAX_EV_PER_ANGSTROM:.0f}); suspicious magnitude.",
                ))

    # --- frame-count alignment ---
    ref = coords.shape[0] if coords.ndim >= 2 else 0
    for key in ("cells", "energies", "forces"):
        arr = d.get(key)
        if arr is not None:
            arr = np.asarray(arr)
            if arr.ndim >= 1 and arr.shape[0] != ref and ref > 0:
                issues.append(_issue(
                    "error", key,
                    f"{key} has {arr.shape[0]} frame(s) but coords has "
                    f"{ref}; frame counts must align.",
                ))

    return issues


def check_data(
    data,
    strict: bool = False,
    box_det_tol: float = _BOX_DET_TOLERANCE,
) -> list[Issue]:
    """
    Content-level sanity check of one or more dpdata systems.

    Checks for NaN/Inf, degenerate (zero-volume) cells, misaligned frame
    counts, and coarse magnitude bounds.

    Parameters
    ----------
    data : dpdata.System | list[dpdata.System]
        Systems to check.
    strict : bool
        If True, raise ``DPADataError`` on the first issue.
    box_det_tol : float
        A cell matrix with ``|det|`` below this is reported as degenerate.

    Returns
    -------
    list[Issue]
    """
    import dpdata

    if isinstance(data, (dpdata.System, dpdata.LabeledSystem)):
        systems = [data]
    elif isinstance(data, (list, tuple)):
        systems = list(data)
    else:
        raise TypeError(
            f"check_data expects dpdata.System or list, got {type(data).__name__}"
        )

    issues: list[Issue] = []

    for i, system in enumerate(systems):
        source = getattr(system, "_dpa_source", None)
        identifier = source if source else f"system[{i}]"
        for issue in _check_system(system, identifier, box_det_tol):
            if strict:
                raise DPADataError(
                    f"check_data (strict): {issue.description}"
                )
            issues.append(issue)

    return issues
