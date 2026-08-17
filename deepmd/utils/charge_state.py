# SPDX-License-Identifier: LGPL-3.0-or-later
"""The frame charge and spin-multiplicity condition shared by every descriptor.

A descriptor that accepts ``add_chg_spin_ebd`` embeds the frame condition by
gathering one row of a charge table and one row of a multiplicity table. The
geometry of those tables is a property of the feature rather than of any one
descriptor: DPA3, DPA4 and DPA4C all size them from the constants here and
index them with the same offset, so a state that addresses one model's tables
addresses them all.

Neither the gather nor the compiled kernel bounds-checks the row index, so a
value outside a table reads past it and a fractional value is silently
truncated onto a neighbouring row. Every host-side boundary that accepts a
condition therefore passes it through :func:`validate_charge_state` first. The
per-forward path is deliberately not guarded: its values come from the data
pipeline, which owns their validity exactly as it owns the validity of an atom
type.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import numpy as np

#: Rows of the charge table, covering integer charges in ``[-100, 99]``.
CHARGE_TABLE_ROWS = 200

#: Index of the neutral charge row, so row ``CHARGE_OFFSET + Q`` holds ``Q``.
CHARGE_OFFSET = 100

#: Rows of the spin table, covering integer multiplicities below this bound.
MULTIPLICITY_TABLE_ROWS = 100

#: Half-open range of representable total charges, in units of the elementary
#: charge.
CHARGE_RANGE = (-CHARGE_OFFSET, CHARGE_TABLE_ROWS - CHARGE_OFFSET)

#: Half-open range of representable spin multiplicities.
MULTIPLICITY_RANGE = (0, MULTIPLICITY_TABLE_ROWS)

#: Name of each value of a charge state, in order, for diagnostics.
CHARGE_STATE_FIELDS = ("charge", "multiplicity")

#: Half-open row range addressed by each value of a charge state, in order.
CHARGE_STATE_TABLE_RANGES = (CHARGE_RANGE, MULTIPLICITY_RANGE)

#: Number of values in one charge state.
CHARGE_STATE_WIDTH = len(CHARGE_STATE_FIELDS)


def validate_charge_state(charge_spin: Any) -> list[float]:
    """Check that a frame condition addresses a row of each embedding table.

    Parameters
    ----------
    charge_spin
        A pair ``[charge, multiplicity]``, in any sequence form.

    Returns
    -------
    list[float]
        The same pair, as two floats.

    Raises
    ------
    ValueError
        If the pair does not hold exactly two integral values within the
        representable ranges.
    """
    values = [float(value) for value in np.reshape(np.asarray(charge_spin), (-1,))]
    if len(values) != CHARGE_STATE_WIDTH:
        raise ValueError(
            f"A charge state must be a `[charge, multiplicity]` pair, got "
            f"{len(values)} values"
        )
    for value, name, (low, high) in zip(
        values,
        CHARGE_STATE_FIELDS,
        CHARGE_STATE_TABLE_RANGES,
        strict=True,
    ):
        if not np.isfinite(value) or value != int(value):
            raise ValueError(f"The {name} must be an integer, got {value}")
        if not low <= value < high:
            raise ValueError(
                f"The {name} must lie in [{low}, {high}), got {int(value)}"
            )
    return values
