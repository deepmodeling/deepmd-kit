# SPDX-License-Identifier: LGPL-3.0-or-later
"""Normalize raw batches from DeepmdDataSystem into canonical format."""

from typing import (
    Any,
)

import numpy as np

# Keys that are metadata / not needed by models or loss functions.
_DROP_KEYS = {"default_mesh", "sid", "fid"}

# Keys that belong to model input (everything else is label).
_INPUT_KEYS = {"coord", "atype", "spin", "box", "fparam", "aparam"}


def normalize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw batch from :class:`DeepmdDataSystem` to canonical format.

    The following conversions are applied:

    * ``"type"`` is renamed to ``"atype"`` (int64).
    * ``"natoms_vec"`` (1-D) is tiled to 2-D ``[nframes, 2+ntypes]``
      and stored as ``"natoms"``.
    * ``find_*`` flags are converted to ``np.bool_``.
    * Metadata keys (``default_mesh``, ``sid``, ``fid``) are dropped.

    Parameters
    ----------
    batch : dict[str, Any]
        Raw batch dict returned by ``DeepmdDataSystem.get_batch()``.

    Returns
    -------
    dict[str, Any]
        Normalized batch dict (new dict; the input is not mutated).
    """
    out: dict[str, Any] = {}

    for key, val in batch.items():
        if key in _DROP_KEYS:
            continue

        if key == "type":
            out["atype"] = val.astype(np.int64)
        elif key.startswith("find_"):
            out[key] = np.bool_(float(val) > 0.5)
        elif key == "natoms_vec":
            nv = val
            if nv.ndim == 1 and "coord" in batch:
                nframes = batch["coord"].shape[0]
                nv = np.tile(nv, (nframes, 1))
            out["natoms"] = nv
        else:
            out[key] = val

    return out


def split_batch(
    batch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a normalized batch into input and label dicts.

    Parameters
    ----------
    batch : dict[str, Any]
        Normalized batch (output of :func:`normalize_batch`).

    Returns
    -------
    input_dict : dict[str, Any]
        Model inputs (coord, atype, box, fparam, aparam, spin).
    label_dict : dict[str, Any]
        Labels and find flags (energy, force, virial, find_*, natoms, …).
    """
    input_dict: dict[str, Any] = {}
    label_dict: dict[str, Any] = {}

    for key, val in batch.items():
        if key in _INPUT_KEYS:
            input_dict[key] = val
        else:
            label_dict[key] = val

    return input_dict, label_dict
