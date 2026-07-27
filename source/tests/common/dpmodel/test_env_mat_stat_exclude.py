# SPDX-License-Identifier: LGPL-3.0-or-later
"""Piece A: model-level pair exclusion enters the descriptor INPUT stat at
nlist BUILD, matching the model forward.

Model-level ``pair_exclude_types`` is a nlist-BUILD transform (decision
#18/A4). In the forward the descriptor is fed a pre-excluded nlist, so an
excluded pair is treated exactly like an empty slot (env_mat 0, still counted).
The input stat now folds the same exclusion into the neighbor list it builds
(``EnvMatStatSe.iter``) instead of deselecting excluded pairs during
accumulation.

The load-bearing invariant: excluding a type-pair via model-level
``pair_exclude_types`` must give the SAME stat (davg/dstd) as excluding it via
descriptor-level ``exclude_types`` -- both zero-and-count the excluded pairs.
Before this change they diverged (descriptor-level zero-counted, model-level
deselected). A no-exclusion control proves the exclusion is genuinely active.
"""

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)


def _sample() -> dict:
    rng = np.random.default_rng(0)
    nf, nloc = 2, 6
    coord = rng.normal(size=(nf, nloc, 3)) * 2.0
    # both types present so the (0, 1) exclusion actually removes pairs
    atype = np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]], dtype=np.int64)
    box = np.tile((np.eye(3) * 12.0).reshape(1, 9), (nf, 1))
    return {"coord": coord, "atype": atype, "box": box}


def _stats(exclude_types, pair_exclude_types):
    """Compute (davg, dstd) for a se_e2_a descriptor under the given exclusions."""
    ds = DescrptSeA(6.0, 0.5, [10, 10], exclude_types=exclude_types)
    sample = _sample()
    if pair_exclude_types is not None:
        sample["pair_exclude_types"] = pair_exclude_types
    ds.compute_input_stats([sample])
    return np.asarray(ds.davg), np.asarray(ds.dstd)


def test_model_level_equals_descriptor_level_exclusion() -> None:
    """Model-level pair_exclude of (0,1) == descriptor-level exclude_types of (0,1)."""
    davg_model, dstd_model = _stats(exclude_types=[], pair_exclude_types=[[0, 1]])
    davg_desc, dstd_desc = _stats(exclude_types=[[0, 1]], pair_exclude_types=None)
    # Both zero-and-count the same excluded pairs, so the env-matrix
    # distribution -- hence the stat -- is bit-identical.
    np.testing.assert_allclose(davg_model, davg_desc, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(dstd_model, dstd_desc, rtol=1e-14, atol=1e-14)


def test_exclusion_is_active() -> None:
    """Excluding (0,1) shifts the stat vs no exclusion (proves it is applied)."""
    davg_none, dstd_none = _stats(exclude_types=[], pair_exclude_types=None)
    davg_excl, dstd_excl = _stats(exclude_types=[], pair_exclude_types=[[0, 1]])
    assert not np.allclose(dstd_none, dstd_excl)
