# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for :func:`apply_pair_exclusion_nlist` and the
``pair_excl`` parameter of :func:`build_neighbor_list` /
``DefaultNeighborList.build``.
"""

import numpy as np
import pytest

from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask
from deepmd.dpmodel.utils.nlist import (
    apply_pair_exclusion_nlist,
    build_neighbor_list,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_nlist_atype():
    """2-frame, 3-local, 4-neighbor test fixture.

    nlist[f, i, k] = index of the k-th neighbor of atom i in frame f.
    -1 = empty slot.

    Frame 0 - atoms 0,1,2 (types 0,1,0); ghost atom 3 (type 1).
    Frame 1 - same layout, different neighbor assignment.
    """
    # shape (nf=2, nloc=3, nnei=4)
    nlist = np.array(
        [
            # frame 0
            [[1, 2, 3, -1], [0, 3, -1, -1], [0, 1, -1, -1]],
            # frame 1
            [[2, 3, -1, -1], [0, -1, -1, -1], [1, 2, 3, -1]],
        ],
        dtype=np.int64,
    )
    # extended atype: local atoms + 1 ghost; shape (nf=2, nall=4)
    # types: atom0=0, atom1=1, atom2=0, ghost3=1
    atype_ext = np.array(
        [[0, 1, 0, 1], [0, 1, 0, 1]],
        dtype=np.int64,
    )
    return nlist, atype_ext


# ---------------------------------------------------------------------------
# apply_pair_exclusion_nlist — unit tests
# ---------------------------------------------------------------------------


def test_none_is_identity() -> None:
    nlist, atype_ext = _make_nlist_atype()
    result = apply_pair_exclusion_nlist(nlist, atype_ext, None)
    assert result is nlist


def test_empty_exclude_list_is_identity() -> None:
    nlist, atype_ext = _make_nlist_atype()
    excl = PairExcludeMask(2, [])
    result = apply_pair_exclusion_nlist(nlist, atype_ext, excl)
    assert result is nlist


def test_excluded_pairs_become_minus_one() -> None:
    """Neighbors of excluded type should become -1."""
    nlist, atype_ext = _make_nlist_atype()
    # Exclude (type0, type1) pairs (symmetric: also type1,type0).
    excl = PairExcludeMask(2, [(0, 1)])
    result = apply_pair_exclusion_nlist(nlist, atype_ext, excl)

    # Frame 0, atom 0 (type 0): neighbors 1(type1), 2(type0), 3(type1), -1
    # (0,1) and (0,3) are excluded; (0,2) kept; -1 stays -1
    np.testing.assert_array_equal(result[0, 0], [-1, 2, -1, -1])
    # Frame 0, atom 1 (type 1): neighbors 0(type0), 3(type1), -1, -1
    # (1,0) excluded; (1,3) kept (type1,type1 not excluded)
    np.testing.assert_array_equal(result[0, 1], [-1, 3, -1, -1])
    # Frame 0, atom 2 (type 0): neighbors 0(type0), 1(type1), -1, -1
    # (2,0) kept (0,0); (2,1) excluded (0,1)
    np.testing.assert_array_equal(result[0, 2], [0, -1, -1, -1])


def test_already_empty_slots_preserved() -> None:
    """Entries that are already -1 must remain -1 after exclusion."""
    nlist, atype_ext = _make_nlist_atype()
    excl = PairExcludeMask(2, [(0, 1)])
    result = apply_pair_exclusion_nlist(nlist, atype_ext, excl)
    # -1 positions in the original nlist must still be -1
    original_minus_one = nlist == -1
    np.testing.assert_array_equal(result[original_minus_one], -1)


def test_ghost_atom_type_respected() -> None:
    """Ghost atoms (index >= nloc) are indexed by atype_ext; their types count."""
    nlist, atype_ext = _make_nlist_atype()
    # atype_ext[*,3] == 1; atom 0 is type 0.  (0,1) is excluded.
    # atom 0 in frame 0 lists neighbor 3 (ghost, type 1) -> should be excluded.
    excl = PairExcludeMask(2, [(0, 1)])
    result = apply_pair_exclusion_nlist(nlist, atype_ext, excl)
    assert result[0, 0, 2] == -1  # was 3 (type 1), now excluded


def test_idempotent_double_application() -> None:
    """Applying exclusion twice must give the same result as applying once."""
    nlist, atype_ext = _make_nlist_atype()
    excl = PairExcludeMask(2, [(0, 1)])
    once = apply_pair_exclusion_nlist(nlist, atype_ext, excl)
    twice = apply_pair_exclusion_nlist(once, atype_ext, excl)
    np.testing.assert_array_equal(once, twice)


def test_no_matching_pairs_leaves_nlist_unchanged() -> None:
    """Exclusion list non-empty but no edge matches — nlist unchanged."""
    nlist, atype_ext = _make_nlist_atype()
    # All atoms are type 0 or 1; exclude (2,3) which doesn't appear.
    excl = PairExcludeMask(4, [(2, 3)])
    result = apply_pair_exclusion_nlist(nlist, atype_ext, excl)
    np.testing.assert_array_equal(result, nlist)


# ---------------------------------------------------------------------------
# apply_pair_exclusion_nlist — torch namespace smoke test
# ---------------------------------------------------------------------------


def test_torch_namespace_smoke() -> None:
    torch = pytest.importorskip("torch")
    nlist, atype_ext = _make_nlist_atype()
    nlist_t = torch.from_numpy(nlist)
    atype_t = torch.from_numpy(atype_ext)
    excl = PairExcludeMask(2, [(0, 1)])
    result = apply_pair_exclusion_nlist(nlist_t, atype_t, excl)
    np.testing.assert_array_equal(
        result.numpy(), apply_pair_exclusion_nlist(nlist, atype_ext, excl)
    )


# ---------------------------------------------------------------------------
# build_neighbor_list pair_excl parameter
# ---------------------------------------------------------------------------


def _simple_extended_system():
    """2-atom local system with 1 ghost to give a nontrivial nlist.

    Atoms: local 0 (type 0) at (0,0,0), local 1 (type 1) at (1,0,0).
    Ghost 2 (type 1) at (-1,0,0) (periodic image of atom 1 across pbc).

    rcut=1.5: atom 0 sees neighbors 1, 2; atom 1 sees neighbor 0.
    """
    # nf=1, nall=3
    coord_ext = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]],
        dtype=np.float64,
    )
    atype_ext = np.array([[0, 1, 1]], dtype=np.int64)
    return coord_ext, atype_ext


def test_build_neighbor_list_pair_excl_equals_post_application() -> None:
    """build_neighbor_list(pair_excl=excl) must equal apply_pair_exclusion_nlist
    applied to the result without pair_excl (oracle equivalence).
    """
    coord_ext, atype_ext = _simple_extended_system()
    nloc = 2
    rcut = 1.5
    sel = [4]
    excl = PairExcludeMask(2, [(0, 1)])

    nlist_plain = build_neighbor_list(
        coord_ext, atype_ext, nloc, rcut, sel, distinguish_types=False
    )
    nlist_excl_builtin = build_neighbor_list(
        coord_ext, atype_ext, nloc, rcut, sel, distinguish_types=False, pair_excl=excl
    )
    nlist_excl_manual = apply_pair_exclusion_nlist(nlist_plain, atype_ext, excl)

    np.testing.assert_array_equal(nlist_excl_builtin, nlist_excl_manual)


def test_build_neighbor_list_none_excl_unchanged() -> None:
    """pair_excl=None must not change the nlist."""
    coord_ext, atype_ext = _simple_extended_system()
    nloc = 2
    rcut = 1.5
    sel = [4]

    nlist_plain = build_neighbor_list(
        coord_ext, atype_ext, nloc, rcut, sel, distinguish_types=False
    )
    nlist_with_none = build_neighbor_list(
        coord_ext, atype_ext, nloc, rcut, sel, distinguish_types=False, pair_excl=None
    )
    np.testing.assert_array_equal(nlist_plain, nlist_with_none)


# ---------------------------------------------------------------------------
# DefaultNeighborList.build pair_excl parameter
# ---------------------------------------------------------------------------


def _local_system():
    """Return a simple local (non-extended) 2-atom system for DefaultNeighborList."""
    # shape (nf=1, nloc=2, 3)
    coord = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    atype = np.array([[0, 1]], dtype=np.int64)
    return coord, atype


def test_default_neighbor_list_pair_excl_equals_seam() -> None:
    """DefaultNeighborList(pair_excl=excl) nlist equals build-then-apply."""
    from deepmd.dpmodel.utils.default_neighbor_list import DefaultNeighborList

    coord, atype = _local_system()
    rcut = 1.5
    sel = [4]
    excl = PairExcludeMask(2, [(0, 1)])

    builder = DefaultNeighborList()
    # Build without exclusion
    ext_coord, ext_atype, nlist_plain, _ = builder.build(
        coord, atype, box=None, rcut=rcut, sel=sel
    )
    nlist_manual = apply_pair_exclusion_nlist(nlist_plain, ext_atype, excl)

    # Build with exclusion at builder level
    _, ext_atype2, nlist_builtin, _ = builder.build(
        coord, atype, box=None, rcut=rcut, sel=sel, pair_excl=excl
    )
    np.testing.assert_array_equal(ext_atype, ext_atype2)
    np.testing.assert_array_equal(nlist_builtin, nlist_manual)


# ---------------------------------------------------------------------------
# VesinNeighborList.build pair_excl (torch only)
# ---------------------------------------------------------------------------


def test_vesin_neighbor_list_pair_excl_equals_seam() -> None:
    """VesinNeighborList(pair_excl=excl) nlist equals build-then-apply."""
    torch = pytest.importorskip("torch")
    from deepmd.pt_expt.utils.vesin_neighbor_list import (
        VesinNeighborList,
        is_vesin_torch_available,
    )

    if not is_vesin_torch_available():
        pytest.skip("vesin.torch not installed")

    coord = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float64)
    atype = torch.tensor([[0, 1]], dtype=torch.int64)
    rcut = 1.5
    sel = [4]
    excl = PairExcludeMask(2, [(0, 1)])

    builder = VesinNeighborList()
    ext_coord, ext_atype, nlist_plain, _ = builder.build(
        coord, atype, box=None, rcut=rcut, sel=sel
    )
    nlist_manual = apply_pair_exclusion_nlist(nlist_plain, ext_atype, excl)

    _, ext_atype2, nlist_builtin, _ = builder.build(
        coord, atype, box=None, rcut=rcut, sel=sel, pair_excl=excl
    )
    np.testing.assert_array_equal(nlist_builtin.numpy(), nlist_manual.numpy())


# ---------------------------------------------------------------------------
# NvNeighborList.build: edges + pair_excl raises
# ---------------------------------------------------------------------------


def test_nv_nlist_edges_pair_excl_raises():
    """NvNeighborList.build raises NotImplementedError for edges+pair_excl.

    The guard fires before any CUDA search, so this test runs on CPU.
    NvNeighborList requires CUDA to produce results, but the early-exit
    raise is device-independent.
    """
    import torch

    from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask
    from deepmd.pt.utils.nv_nlist import NvNeighborList

    coord = torch.zeros((1, 4, 3), dtype=torch.float64)
    atype = torch.zeros((1, 4), dtype=torch.int64)
    pe = PairExcludeMask(2, [(0, 1)])
    nl = NvNeighborList()
    with pytest.raises(NotImplementedError, match="return_mode='edges'"):
        nl.build(coord, atype, None, 2.0, [4], return_mode="edges", pair_excl=pe)
