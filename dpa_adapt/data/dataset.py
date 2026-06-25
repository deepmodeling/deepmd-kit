# SPDX-License-Identifier: LGPL-3.0-or-later
# data/dataset.py
#
# Label-aware data loading for supervised training / fine-tuning.
# Thin layer on top of load_data() that additionally verifies every
# system carries the requested label key (e.g. "energy", "homo").

from __future__ import (
    annotations,
)

import logging
from pathlib import (
    Path,
)

import dpdata

from dpa_adapt.data.errors import (
    DPADataError,
)
from dpa_adapt.data.loader import (
    _resolve_label_key,
    load_data,
)

_LOG = logging.getLogger("dpa_adapt.data.dataset")

_DataInput = (
    str
    | Path
    | dpdata.System
    | dpdata.LabeledSystem
    | list[str | Path | dpdata.System | dpdata.LabeledSystem]
)


def load_dataset(
    data: _DataInput,
    label_key: str = "energy",
) -> list[dpdata.LabeledSystem]:
    """
    Load systems and keep only those that carry *label_key*.

    Internally calls ``load_data()`` to normalise input, then inspects each
    system's ``data`` dict for the requested label.  Systems that lack the
    label are skipped with a warning rather than raising, so a partial
    dataset (e.g. a directory tree where only some systems have energies)
    does not block downstream work.

    Parameters
    ----------
    data : str | Path | dpdata.System | dpdata.LabeledSystem | list
        Any input accepted by ``load_data()`` — single path, glob string,
        dpdata object, or heterogeneous list of the above.
    label_key : str
        Label key to check in each system's ``data`` dict (e.g.
        ``"energy"``, ``"force"``, ``"homo"``).  Default ``"energy"``.

    Returns
    -------
    list[dpdata.LabeledSystem]
        Systems that passed label validation.  May be empty only if
        *every* candidate was skipped, in which case a ``DPADataError``
        is raised (fail-fast for training workflows).
    """
    systems = load_data(data)

    resolved_key = _resolve_label_key(label_key)

    validated: list[dpdata.LabeledSystem] = []
    skipped: list[str] = []

    for i, system in enumerate(systems):
        # dpdata stores everything (coords, energies, forces, ...) in the
        # ``data`` dict; label_key (after alias resolution) presence is the litmus test.
        if resolved_key in system.data:
            validated.append(system)
        else:
            identifier = getattr(system, "_dpa_source", f"system[{i}]")
            skipped.append(f"{identifier} (missing {resolved_key!r})")

    if skipped:
        _LOG.warning(
            "load_dataset: %d system(s) skipped (missing label key %r):\n  %s",
            len(skipped),
            resolved_key,
            "\n  ".join(skipped),
        )

    if not validated:
        raise DPADataError(
            f"load_dataset: no valid systems found with label_key={label_key!r} "
            f"(resolved to {resolved_key!r}). "
            f"Skipped {len(skipped)} candidate(s). "
            "Check that the path and label_key are correct."
        )

    return validated
