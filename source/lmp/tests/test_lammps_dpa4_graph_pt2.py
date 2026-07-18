# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test LAMMPS with the NeighborGraph (graph-schema) .pt2 DPA4 model.

The model ``deeppot_dpa4_graph.pt2`` is a DPA4/SeZM descriptor exported with
``lower_kind="graph"`` (gen_dpa4.py Section B). DPA4 is graph-native
end-to-end (no dense-only sub-block to gate off, unlike DPA2's
``use_three_body``), so the same config used for the dense ``deeppot_dpa4.pt2``
fixture is graph-eligible; the weights are freshly jittered (see gen_dpa4.py
Section B.1) so the fixture is geometry-sensitive rather than the
architecturally edge-independent output of a fresh, untrained DPA4.

DPA4 declares no cross-rank exchange (``has_message_passing_across_ranks() ==
False``, see the Task-6 fix in ``deepmd/dpmodel/descriptor/dpa4.py``): no
lower path implements ghost feature exchange across MPI ranks, so
``deserialize_to_file`` never emits a nested with-comm artifact for DPA4 --
unlike DPA2 (whose repformer participates in per-layer ghost exchange).
Multi-rank LAMMPS inference for a graph-lower, message-passing model
therefore has no with-comm route to fall back on and fails fast in C++
(Task 6); this file intentionally covers SINGLE-RANK only, mirroring the
single-rank half of ``test_lammps_dpa2_graph_pt2.py`` (its multi-rank
with-comm section has no DPA4 counterpart).

Single-rank LAMMPS folds ghosts onto local owners (``fold_to_local=True``,
``N == nloc``) and requires ``atom_modify map yes`` (``has_message_passing_``
== True) so the C++ side can resolve ghost-to-local mapping from the LAMMPS
atom-map.

Reference values come from ``source/tests/infer/gen_dpa4.py`` (the same
``deeppot_dpa4_graph.expected`` the C++ gtest uses). A second, independent
oracle -- ``deeppot_dpa4_graph_nlist_ref.pt2`` (same weights, dense-nlist
lower, NOT graph) -- is also exercised directly through LAMMPS so a
regression that only breaks the *C++* graph ingestion (not the Python
export path already cross-checked at gen-time) still gets caught.
"""

import os
from pathlib import (
    Path,
)

import constants
import numpy as np
import pytest
from expected_ref import (
    read_expected_ref,
)
from lammps import (
    PyLammps,
)
from write_lmp_data import (
    write_lmp_data,
)

pb_file = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa4_graph.pt2"
)
# Independent dense-nlist oracle exported from the SAME (jittered) weights
# (gen_dpa4.py B.1/B.2, lower_kind="nlist"); at non-binding sel graph and
# dense math are equivalent (gen-time cross-check already enforces atomic
# energy / force / total-virial agreement within 1e-8 at the Python level --
# see gen_dpa4.py B.4). Comparing through LAMMPS as well exercises the C++
# graph ingestion path (edge tensors, node atype slicing) independently of
# that Python-level check.
nlist_ref_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "deeppot_dpa4_graph_nlist_ref.pt2"
)
ref_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "deeppot_dpa4_graph.expected"
)

# Reference values written by source/tests/infer/gen_dpa4.py (PBC case).
# Guarded with try/except because gen_dpa4.py only runs when PyTorch is built,
# and the graph section is itself skipped under LeakSanitizer (see
# gen_dpa4.py Section B's module comment) -- either way this file must still
# be collectible, with the affected tests skipping cleanly.
try:
    _ref = read_expected_ref(ref_file)["pbc"]
    expected_e = float(np.sum(_ref["expected_e"]))
    expected_f = _ref["expected_f"].reshape(6, 3)
    # LAMMPS uses the opposite sign convention for virial vs DeepPot.
    expected_v = -_ref["expected_v"].reshape(6, 9)
except FileNotFoundError:
    expected_e = expected_f = expected_v = None

_HAS_REF = expected_e is not None

# Same 6-atom water system as the dense DPA4 fixture
# (source/tests/infer/gen_dpa4.py / test_lammps_dpa4_pt2.py): type_map
# [O, H], box 13x13x13.
box = np.array([0, 13, 0, 13, 0, 13, 0, 0, 0])
coord = np.array(
    [
        [12.83, 2.56, 2.18],
        [12.09, 2.87, 2.74],
        [0.25, 3.32, 1.68],
        [3.36, 3.00, 1.81],
        [3.51, 2.51, 2.60],
        [4.27, 3.22, 1.56],
    ]
)
# Model type_map is ["O", "H"]; gen_dpa4.py atype = [0, 1, 1, 0, 1, 1] ->
# LAMMPS types [1, 2, 2, 1, 2, 2] under identity ``pair_coeff * *``.
type_OH = np.array([1, 2, 2, 1, 2, 2])

data_file = Path(__file__).parent / "data_dpa4_graph_pt2.lmp"


def setup_module() -> None:
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip(
            "Skip test because PyTorch support is not enabled.",
        )
    write_lmp_data(box, coord, type_OH, data_file)


def teardown_module() -> None:
    if data_file.exists():
        os.remove(data_file)


def _lammps(data_file, units="metal", atom_map: str = "yes") -> PyLammps:
    lammps = PyLammps()
    lammps.units(units)
    lammps.boundary("p p p")
    lammps.atom_style("atomic")
    if atom_map != "no":
        lammps.atom_modify(f"map {atom_map}")
    lammps.neighbor("2.0 bin")
    lammps.neigh_modify("every 10 delay 0 check no")
    lammps.read_data(data_file.resolve())
    lammps.mass("1 16")
    lammps.mass("2 2")
    lammps.timestep(0.0005)
    lammps.fix("1 all nve")
    return lammps


@pytest.fixture
def lammps():
    lmp = _lammps(data_file=data_file)
    yield lmp
    lmp.close()


@pytest.mark.skipif(
    not _HAS_REF, reason="gen_dpa4.py graph .expected fixture not generated"
)
def test_pair_deepmd(lammps) -> None:
    """Single-rank serial run (``atom_modify map yes``): the graph .pt2
    folds ghosts onto local owners (``fold_to_local=True``) and must match
    the gen_dpa4.py reference for energy and per-atom force.
    """
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    lammps.run(1)


@pytest.mark.skipif(
    not _HAS_REF, reason="gen_dpa4.py graph .expected fixture not generated"
)
def test_pair_deepmd_virial(lammps) -> None:
    """Single-rank per-atom virial via ``centroid/stress/atom``."""
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.compute("virial all centroid/stress/atom NULL pair")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps.variable(f"virial{jj} atom c_virial[{ii + 1}]")
    lammps.dump(
        "1 all custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    )
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    idx_map = lammps.lmp.numpy.extract_atom("id")[: coord.shape[0]] - 1
    for ii in range(9):
        assert np.array(
            lammps.variables[f"virial{ii}"].value
        ) / constants.nktv2p == pytest.approx(expected_v[idx_map, ii])


@pytest.mark.skipif(
    not nlist_ref_file.exists(),
    reason="gen_dpa4.py deeppot_dpa4_graph_nlist_ref.pt2 fixture not generated",
)
def test_pair_deepmd_graph_matches_nlist_ref() -> None:
    """Single-rank graph .pt2 vs the independent dense-nlist oracle
    (``deeppot_dpa4_graph_nlist_ref.pt2``, same weights, ``lower_kind="nlist"``)
    through LAMMPS, energy/force within 1e-6.

    gen_dpa4.py already cross-checks graph vs nlist at the Python
    ``DeepPot.eval`` level (B.4, 1e-8) at generation time; this test instead
    drives BOTH artifacts through the C++ ``DeepPotPTExpt`` / LAMMPS pair
    style, so a regression confined to the C++ graph-ingestion seam (edge
    tensor construction, node atype slicing) that the Python-level gen-time
    check cannot see is still caught. The per-atom virial is deliberately
    NOT compared here: the graph path assigns each edge's force/virial
    contribution fully to the source atom (edge_force_virial full-to-src), a
    different (equally valid) decomposition than the dense path's -- only
    energy and force (and, at the Python level already, the *total* virial)
    are convention-independent.
    """
    lmp_graph = _lammps(data_file=data_file)
    lmp_graph.pair_style(f"deepmd {pb_file.resolve()}")
    lmp_graph.pair_coeff("* *")
    lmp_graph.run(0)
    e_graph = lmp_graph.eval("pe")
    f_graph = np.array([lmp_graph.atoms[ii].force for ii in range(6)])
    id_graph = [lmp_graph.atoms[ii].id for ii in range(6)]
    lmp_graph.close()

    lmp_nlist = _lammps(data_file=data_file)
    lmp_nlist.pair_style(f"deepmd {nlist_ref_file.resolve()}")
    lmp_nlist.pair_coeff("* *")
    lmp_nlist.run(0)
    e_nlist = lmp_nlist.eval("pe")
    f_nlist = np.array([lmp_nlist.atoms[ii].force for ii in range(6)])
    id_nlist = [lmp_nlist.atoms[ii].id for ii in range(6)]
    lmp_nlist.close()

    # Same data file, single rank -> identical atom-id ordering; assert this
    # rather than silently re-sorting so an ordering change is loud.
    assert id_graph == id_nlist
    assert e_graph == pytest.approx(e_nlist, rel=0, abs=1e-6)
    np.testing.assert_allclose(f_graph, f_nlist, atol=1e-6, rtol=0)
