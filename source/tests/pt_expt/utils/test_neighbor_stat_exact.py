# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run the independent integer oracle on the selected PyTorch device."""

import pytest

from deepmd.pt_expt.utils.neighbor_stat import (
    NeighborStat,
)

from ...common.test_neighbor_stat_exact import (
    SCALE,
    check_cutoff_boundary,
    check_exact_case,
    check_overlap_in_tail,
    check_permutations,
)


@pytest.mark.parametrize("seed", range(24))
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("cap", [1, 3, 8, 100, None])
def test_matches_independent_integer_oracle(monkeypatch, seed, mixed, cap):
    check_exact_case(NeighborStat, monkeypatch, seed, mixed, cap)


@pytest.mark.parametrize("seed", [1, 5, 10, 17, 22])
@pytest.mark.parametrize("mixed", [False, True])
def test_frame_atom_order_and_translation(monkeypatch, seed, mixed):
    check_permutations(NeighborStat, monkeypatch, seed, mixed)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("separation", [SCALE - 1, SCALE, SCALE + 1])
@pytest.mark.parametrize("cap", [1, 8, 100])
def test_cutoff_boundary_and_virtual_overlap(
    monkeypatch, periodic, mixed, separation, cap
):
    check_cutoff_boundary(NeighborStat, monkeypatch, periodic, mixed, separation, cap)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("cap", [1, 8, 100])
def test_real_overlap_in_final_partial_batch(monkeypatch, periodic, cap):
    check_overlap_in_tail(NeighborStat, monkeypatch, periodic, cap)
