# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fixed packed Wigner layout for the Neo ``lmax=3, mmax=1`` K1 path.

The panel stores only rows selected by ``coeff_index_m`` and only columns from
the matching Wigner block. Phase A reads ``D[coeff, degree]`` while Phase C
reads ``Dt[degree, coeff]``; both expressions therefore address the same slot.
"""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )


BLOCK_WIDTHS = (1, 3, 5, 7)
FULL_BLOCK_OFFSETS = (0, 1, 4, 9, 16)

# Rows are ordered m=0, m=-1, m=+1 within each non-scalar block.
SELECTED_LOCAL_ROWS = ((0,), (1, 0, 2), (2, 1, 3), (3, 2, 4))
PANEL_BLOCK_OFFSETS = (0, 1, 10, 25, 46)
PACKED_VALUE_COUNT = PANEL_BLOCK_OFFSETS[-1]

# DeePMD's m-major reduced ordering for lmax=3, mmax=1.
COEFF_INDEX_M = (0, 2, 6, 12, 1, 5, 11, 3, 7, 13)
REDUCED_DEGREES = (0, 1, 2, 3, 1, 2, 3, 1, 2, 3)
REDUCED_PANEL_ROW_OFFSETS = (0, 1, 10, 25, 4, 15, 32, 7, 20, 39)
ZONAL_PANEL_OFFSETS = tuple(range(1, 4)) + tuple(range(10, 15)) + tuple(range(25, 32))


@dataclass(frozen=True)
class PackedWignerEntry:
    offset: int
    degree: int
    reduced: int
    full_row: int
    full_col: int


_REDUCED_BY_FULL_ROW = {
    full_row: reduced for reduced, full_row in enumerate(COEFF_INDEX_M)
}


def d_offset(reduced: int, full_col: int) -> int | None:
    """Map ``D[coeff_index_m[reduced], full_col]`` into the packed panel."""
    degree = REDUCED_DEGREES[reduced]
    block_start = FULL_BLOCK_OFFSETS[degree]
    block_stop = FULL_BLOCK_OFFSETS[degree + 1]
    if full_col < block_start or full_col >= block_stop:
        return None
    return REDUCED_PANEL_ROW_OFFSETS[reduced] + full_col - block_start


def dt_offset(full_row: int, reduced: int) -> int | None:
    """Map ``Dt[full_row, coeff_index_m[reduced]]`` to its shared D slot."""
    return d_offset(reduced, full_row)


def iter_packed_entries() -> Iterator[PackedWignerEntry]:
    """Yield the 46 stored entries in contiguous panel order."""
    for degree, (width, block_start, panel_start, local_rows) in enumerate(
        zip(
            BLOCK_WIDTHS,
            FULL_BLOCK_OFFSETS[:-1],
            PANEL_BLOCK_OFFSETS[:-1],
            SELECTED_LOCAL_ROWS,
            strict=True,
        )
    ):
        for row_slot, local_row in enumerate(local_rows):
            full_row = block_start + local_row
            reduced = _REDUCED_BY_FULL_ROW[full_row]
            row_start = panel_start + row_slot * width
            for local_col in range(width):
                yield PackedWignerEntry(
                    offset=row_start + local_col,
                    degree=degree,
                    reduced=reduced,
                    full_row=full_row,
                    full_col=block_start + local_col,
                )
