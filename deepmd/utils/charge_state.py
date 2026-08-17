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
condition therefore checks it here first: a configured default through
:func:`validate_charge_state`, and a batch of frames read from the training
data through :func:`validate_charge_states`. Both share one rule, evaluated
over the whole batch at once so the training loop pays no per-frame cost, and
both run before tensor conversion so no device synchronization is needed.
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


def _as_states(charge_spin: Any) -> np.ndarray:
    """Read a request as ``(n, CHARGE_STATE_WIDTH)`` states."""
    values = np.asarray(charge_spin, dtype=np.float64).reshape(-1)
    if values.size == 0 or values.size % CHARGE_STATE_WIDTH:
        raise ValueError(
            f"A charge state must be a `[charge, multiplicity]` pair, got "
            f"{values.size} values"
        )
    return values.reshape(-1, CHARGE_STATE_WIDTH)


def _check_states(states: np.ndarray) -> None:
    """Reject any state that addresses no row of the embedding tables.

    The whole batch is tested column by column with array operations, so the
    cost does not grow with the number of frames. A non-finite value is
    reported as the integrality failure rather than reaching the range test,
    whose message would have to render it.

    Parameters
    ----------
    states : np.ndarray
        Charge states with shape (n, CHARGE_STATE_WIDTH).

    Raises
    ------
    ValueError
        If any value is not an integer inside its table's row range.
    """
    integral = np.isfinite(states) & (states == np.floor(states))
    for column, (name, (low, high)) in enumerate(
        zip(CHARGE_STATE_FIELDS, CHARGE_STATE_TABLE_RANGES, strict=True)
    ):
        values = states[:, column]
        offending = values[~integral[:, column]]
        if offending.size:
            raise ValueError(f"The {name} must be an integer, got {offending[0]}")
        outside = values[(values < low) | (values >= high)]
        if outside.size:
            raise ValueError(
                f"The {name} must lie in [{low}, {high}), got {outside[0]:.0f}"
            )


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
    states = _as_states(charge_spin)
    if states.shape[0] != 1:
        raise ValueError(
            f"A charge state must be a `[charge, multiplicity]` pair, got "
            f"{states.size} values"
        )
    _check_states(states)
    return states[0].tolist()


def validate_charge_states(charge_spin: Any) -> None:
    """Check every frame condition of a batch against the embedding tables.

    Parameters
    ----------
    charge_spin
        One ``[charge, multiplicity]`` pair per frame, in any shape holding a
        whole number of pairs.

    Raises
    ------
    ValueError
        If any frame names a state that no table row answers.
    """
    _check_states(_as_states(charge_spin))
