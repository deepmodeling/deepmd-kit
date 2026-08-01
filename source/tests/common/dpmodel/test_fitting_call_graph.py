# SPDX-License-Identifier: LGPL-3.0-or-later
"""GeneralFitting.call_graph is the graph-native (flat-N) fitting API. Its result
must be bit-identical to the dense __call__ raveled over (nf, nloc) -- it reuses
the dense net via the (N,1,nd) single-atom-frame workaround. fparam is node-level
(N, ndf) (the caller gathers per-frame fparam by frame_id).
"""

import numpy as np
import pytest

from deepmd.dpmodel.fitting import (
    InvarFitting,
)


@pytest.mark.parametrize("ndf", [0, 3])  # numb_fparam: no-fparam AND fparam
def test_call_graph_matches_dense_raveled(ndf):
    rng = np.random.default_rng(0)
    nf, nloc, nd, ntypes, ng = 2, 4, 8, 2, 5
    ft = InvarFitting("energy", ntypes, nd, 1, mixed_types=True, numb_fparam=ndf)
    desc = rng.normal(size=(nf, nloc, nd))
    atype = rng.integers(0, ntypes, size=(nf, nloc))
    gr = rng.normal(size=(nf, nloc, ng, 3))
    fparam = rng.normal(size=(nf, ndf)) if ndf else None
    dense = ft(desc, atype, gr=gr, fparam=fparam)["energy"]  # (nf, nloc, 1)
    N = nf * nloc
    frame_id = np.repeat(np.arange(nf), nloc)
    fparam_node = fparam[frame_id] if ndf else None  # (N, ndf)
    flat = ft.call_graph(
        desc.reshape(N, nd),
        atype.reshape(N),
        gr=gr.reshape(N, ng, 3),
        fparam=fparam_node,
    )["energy"]  # (N, 1)
    assert flat.shape == (N, 1)
    np.testing.assert_allclose(flat, dense.reshape(N, 1), rtol=1e-12, atol=1e-12)
