# SPDX-License-Identifier: LGPL-3.0-or-later
"""Flat-N ragged-native graph path: nodes on a flat (N,) axis, N = sum(n_node);
per-frame reductions use segment_sum over frame_id. UNEQUAL per-frame node counts
(ragged) -- the case the old rectangular (nf,nloc) path could not represent.
"""

import numpy as np

from deepmd.dpmodel.utils.neighbor_graph import frame_id_from_n_node


def test_frame_id_ragged():
    fid = frame_id_from_n_node(np.array([3, 5, 2], dtype=np.int64))  # N=10
    np.testing.assert_array_equal(
        fid, np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=fid.dtype)
    )


def test_frame_id_rectangular():
    fid = frame_id_from_n_node(np.array([4, 4], dtype=np.int64))
    np.testing.assert_array_equal(
        fid, np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=fid.dtype)
    )
