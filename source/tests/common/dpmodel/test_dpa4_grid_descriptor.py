# SPDX-License-Identifier: LGPL-3.0-or-later
"""Full-descriptor pt->dpmodel interop and invariance tests for the DPA4 SO(3) grid.

These exercise the flagship ``examples/water/dpa4/input.json`` grid flags
(``ffn_so3_grid`` + ``message_node_so3`` + ``grid_branch``/``grid_mlp``) at the
descriptor level:

* Part B (convert-backend interop): build the pt ``DescrptSeZM`` (the class
  behind ``type: DPA4``) and the dpmodel ``DescrptDPA4`` with a matching,
  example-config-like descriptor block, weight-copy via
  ``DescrptDPA4.deserialize(pt.serialize())`` -- which is exactly the schema
  path ``dp convert-backend`` uses -- and assert descriptor-output parity at
  fp64 1e-10 on CPU. pt weights are deterministically perturbed so the
  comparison is non-trivial.
* Part C (invariance): permuting the neighbor order within the nlist, and
  appending an extra empty (``-1``) neighbor slot, both leave the per-atom
  descriptor output unchanged.

pt imports live inside the test functions because ruff TID253 bans
module-level ``deepmd.pt`` imports under ``source/tests/common``.
"""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4,
)


def build_neighbor_list_np(coord, rcut, nnei):
    """Build a padded, distance-sorted gas-phase neighbor list.

    Parameters
    ----------
    coord
        Coordinates with shape (nf, nloc, 3); no PBC.
    rcut
        Cutoff radius.
    nnei
        Number of neighbor slots; pads with -1.

    Returns
    -------
    np.ndarray
        Neighbor list with shape (nf, nloc, nnei).
    """
    nf, nloc, _ = coord.shape
    nlist = -np.ones((nf, nloc, nnei), dtype=np.int64)
    for f in range(nf):
        dist = np.linalg.norm(coord[f][:, None, :] - coord[f][None, :, :], axis=-1)
        for i in range(nloc):
            neighbors = [
                (dist[i, j], j) for j in range(nloc) if j != i and dist[i, j] < rcut
            ]
            neighbors.sort()
            for slot, (_, j) in enumerate(neighbors[:nnei]):
                nlist[f, i, slot] = j
    return nlist


def make_inputs(seed=5, nf=2, nloc=6, rcut=6.0, nnei=20, ntypes=2):
    rng = np.random.default_rng(seed)
    coord = rng.uniform(0.0, 3.5, size=(nf, nloc, 3))
    atype = rng.integers(0, ntypes, size=(nf, nloc))
    nlist = build_neighbor_list_np(coord, rcut, nnei)
    return coord, atype, nlist


def example_descriptor_kwargs(**overrides) -> dict:
    """Small example-config-like (examples/water/dpa4/input.json) descriptor block.

    Sizes are shrunk (channels=8, sel=20, mixing_layers=2) for fast fp64 parity,
    but the grid-relevant structure (lmax=3, mmax=1, n_focus=2, n_blocks=2,
    grid_branch=[1,1,1], ffn_so3_grid + message_node_so3) mirrors the flagship
    config.
    """
    kwargs = {
        "ntypes": 2,
        "sel": 20,
        "rcut": 6.0,
        "channels": 8,
        "n_radial": 8,
        "lmax": 3,
        "mmax": 1,
        "n_blocks": 2,
        "mixing_layers": 2,
        "n_focus": 2,
        "focus_dim": 0,
        "ffn_so3_grid": True,
        "grid_mlp": [False, False, False],
        "grid_branch": [1, 1, 1],
        "message_node_so3": True,
        "precision": "float64",
        "seed": 42,
    }
    kwargs.update(overrides)
    return kwargs


# Part B: pt -> dpmodel convert-backend interop, one case per grid path so a
# bug in either the FFN SO(3) grid or the post-aggregation SO(3) message is
# isolated.
@pytest.mark.parametrize(
    "grid_flags",
    [
        {"ffn_so3_grid": True, "message_node_so3": False},  # FFN SO(3) grid only
        {"ffn_so3_grid": False, "message_node_so3": True},  # post-agg SO(3) msg only
        {"ffn_so3_grid": True, "message_node_so3": True},  # both (example-config-like)
    ],
)
def test_pt_to_dpmodel_interop(grid_flags) -> None:
    """``DescrptDPA4.deserialize(pt.serialize())`` reproduces the pt descriptor.

    This proves the ``dp convert-backend`` schema interop for the SO(3) grid
    paths at fp64 1e-10 (CPU).
    """
    import torch

    from deepmd.pt.model.descriptor.sezm import (
        DescrptSeZM,
    )

    kwargs = example_descriptor_kwargs(**grid_flags)

    # pin to CPU: torch.from_numpy fp64 inputs and the module must agree under
    # the CUDA-default-device CI configuration
    pt_dd = DescrptSeZM(**kwargs).to("cpu")
    # random init can give near-zero output; perturb deterministically so the
    # comparison is non-trivial (asserted below via the magnitude check)
    rng = np.random.default_rng(1234)
    with torch.no_grad():
        for p in pt_dd.parameters():
            p += torch.from_numpy(0.1 * rng.standard_normal(size=tuple(p.shape)))

    # the convert-backend path: dpmodel reconstructs purely from pt's schema
    dp_dd = DescrptDPA4.deserialize(pt_dd.serialize())

    coord, atype, nlist = make_inputs()
    nf = atype.shape[0]
    coord_ext = coord.reshape(nf, -1)

    pt_out = (
        pt_dd(
            torch.from_numpy(coord_ext),
            torch.from_numpy(atype),
            torch.from_numpy(nlist),
        )[0]
        .detach()
        .cpu()
        .numpy()
    )
    dp_out = np.asarray(dp_dd.call(coord_ext, atype, nlist)[0])

    # non-trivial output (perturbation took effect)
    assert np.abs(dp_out).max() > 1e-6
    np.testing.assert_allclose(dp_out, pt_out, rtol=1e-10, atol=1e-10)


# Part C: invariance of the example-config descriptor (both grid flags on).
def test_permutation_invariance() -> None:
    """Permuting the neighbor order within the nlist leaves the output unchanged."""
    dd = DescrptDPA4(**example_descriptor_kwargs())
    coord, atype, nlist = make_inputs()
    nf = atype.shape[0]
    coord_ext = coord.reshape(nf, -1)
    out = np.asarray(dd.call(coord_ext, atype, nlist)[0])

    rng = np.random.default_rng(7)
    perm = rng.permutation(nlist.shape[-1])
    nlist2 = nlist[:, :, perm]
    out2 = np.asarray(dd.call(coord_ext, atype, nlist2)[0])
    assert np.abs(out).max() > 1e-6
    np.testing.assert_allclose(out2, out, rtol=1e-10, atol=1e-12)


def test_masked_edge_noop() -> None:
    """An extra all-(-1) neighbor slot must not change the descriptor."""
    dd = DescrptDPA4(**example_descriptor_kwargs())
    coord, atype, nlist = make_inputs()
    nf, nloc = atype.shape
    coord_ext = coord.reshape(nf, -1)
    out = np.asarray(dd.call(coord_ext, atype, nlist)[0])
    assert np.abs(out).max() > 1e-6

    pad = -np.ones((nf, nloc, 1), dtype=nlist.dtype)
    nlist2 = np.concatenate([nlist, pad], axis=-1)
    out2 = np.asarray(dd.call(coord_ext, atype, nlist2)[0])
    np.testing.assert_allclose(out2, out, rtol=1e-10, atol=1e-12)
