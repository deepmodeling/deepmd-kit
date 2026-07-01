# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph-form ``.pt2`` DeepEval parity vs the eager dense reference.

A graph-form ``.pt2`` (exported with ``lower_kind="graph"``) carries the
NeighborGraph schema ``(atype, n_node, edge_index, edge_vec, edge_mask, ...)``
in its AOTI forward.  This test verifies that evaluating such an archive
through the pt_expt :class:`DeepPot` reproduces the eager dpa1 energy / force /
virial to ``rtol=atol=1e-10`` (fp64), for both PBC and non-PBC.

The graph path is CARRY-ALL (every neighbor within ``rcut``); the eager dense
reference is sel-capped (``sel=30``, forced via
``neighbor_graph_method="legacy"``).  They coincide only at NON-BINDING ``sel``
(max neighbor count ``< sel``), so the test fixture is a small, sparse cluster
and the non-binding condition is asserted explicitly -- otherwise the parity
would vacuously compare two different neighbor sets.
"""

import copy
import os
import tempfile

import numpy as np
import pytest
import torch

from deepmd.infer import (
    DeepPot,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
)

# dpa1 with attn_layer == 0 -- the energy model exercised by the graph path.
DPA1_CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "se_atten",
        "sel": 30,
        "rcut_smth": 2.0,
        "rcut": 6.0,
        "neuron": [2, 4, 8],
        "axis_neuron": 4,
        "attn": 5,
        "attn_layer": 0,
        "attn_dotr": True,
        "attn_mask": False,
        "activation_function": "tanh",
        "scaling_factor": 1.0,
        "normalize": True,
        "temperature": 1.0,
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [5, 5, 5],
        "resnet_dt": True,
        "seed": 1,
    },
}

RCUT = 6.0
SEL = 30


def _build_system(
    natoms: int = 8, seed: int = 20240626
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small, sparse cluster: ``natoms`` inside a 5 A blob, centered in an 18 A box.

    The blob keeps every atom within ``rcut`` of at most ``natoms - 1`` others
    (<< ``sel``), so the carry-all graph neighbor set equals the sel-capped
    dense one.  Varying ``natoms`` yields a different edge count, exercising the
    DYNAMIC edge axis of the exported ``.pt2`` (B2.0).
    """
    rng = np.random.default_rng(seed)
    box_size = 18.0
    blob = rng.random((natoms, 3)) * 5.0 + box_size * 0.5 - 2.5
    coords = blob.reshape(1, natoms, 3)
    cells = (np.eye(3) * box_size).reshape(1, 9)
    # Alternate O/H types; both species present regardless of natoms.
    atype = np.array([i % 2 for i in range(natoms)], dtype=np.int32)
    return coords, cells, atype


# Two DIFFERENT-size systems evaluated through the SAME exported ``.pt2``.
# Both are sparse, non-binding clusters but with different edge counts, so the
# second size FAILS against a static-``E`` artifact (B1) and PASSES only once
# the edge axis is dynamic (B2.0).
_SYSTEMS = {
    "small_8": {"natoms": 8, "seed": 20240626},
    "large_20": {"natoms": 20, "seed": 20240701},
}


def _max_neighbors(
    coords: np.ndarray, cells: np.ndarray | None, atype: np.ndarray
) -> int:
    """Max carry-all neighbor count per center within ``rcut`` (for the non-binding check)."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        build_neighbor_graph,
    )

    natoms = atype.shape[0]
    graph = build_neighbor_graph(
        coords.reshape(1, natoms, 3),
        atype.reshape(1, natoms),
        cells.reshape(1, 9) if cells is not None else None,
        RCUT,
    )
    real = np.asarray(graph.edge_mask)
    dst = np.asarray(graph.edge_index)[1][real]
    counts = np.bincount(dst, minlength=natoms)
    return int(counts.max())


def _eager_dense_reference(
    model: torch.nn.Module,
    coords: np.ndarray,
    cells: np.ndarray | None,
    atype: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reference energy/force/virial from the eager dense (sel-capped) path."""
    natoms = atype.shape[0]
    coord_t = torch.tensor(
        coords.reshape(1, natoms, 3), dtype=torch.float64, device=DEVICE
    ).requires_grad_(True)
    atype_t = torch.tensor(atype.reshape(1, natoms), dtype=torch.int64, device=DEVICE)
    box_t = (
        torch.tensor(cells.reshape(1, 9), dtype=torch.float64, device=DEVICE)
        if cells is not None
        else None
    )
    ret = model.call_common(
        coord_t,
        atype_t,
        box_t,
        do_atomic_virial=True,
        neighbor_graph_method="legacy",
    )
    out = {
        "atom_energy": ret["energy"],
        "energy": ret["energy_redu"],
        "force": ret["energy_derv_r"].squeeze(-2),
        "virial": ret["energy_derv_c_redu"].squeeze(-2),
        "atom_virial": ret["energy_derv_c"].squeeze(-2),
    }
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


@pytest.fixture(scope="module")
def graph_pt2():
    """Build a dpa1(attn_layer=0) model and export it to a graph-form ``.pt2``.

    The AOTI compile is slow (~90 s), so it is done once per module.  The eager
    pt_expt model is returned alongside the archive path to serve as the dense
    parity reference.
    """
    from deepmd.pt_expt.model import (
        get_model,
    )

    model = get_model(copy.deepcopy(DPA1_CONFIG)).to(torch.float64)
    model.eval()
    data = {"model": model.serialize()}

    tmpdir = tempfile.mkdtemp()
    pt2_path = os.path.join(tmpdir, "deeppot_dpa1_graph.pt2")
    deserialize_to_file(
        pt2_path,
        copy.deepcopy(data),
        do_atomic_virial=True,
        lower_kind="graph",
    )
    yield pt2_path, model
    os.unlink(pt2_path)
    os.rmdir(tmpdir)


@pytest.mark.parametrize("system", list(_SYSTEMS))  # two different edge counts
@pytest.mark.parametrize("pbc", [True, False])  # periodic vs non-periodic
def test_graph_pt2_deepeval_parity(graph_pt2, pbc, system) -> None:
    """Graph ``.pt2`` DeepEval == eager dense dpa1 (energy/force/virial), 1e-10.

    Both ``_SYSTEMS`` are fed through the SAME module-scoped ``.pt2``; the
    differing edge counts prove the exported artifact's edge axis is dynamic
    (a static-``E`` B1 artifact would reject / mis-shape the larger system).
    """
    pt2_path, model = graph_pt2
    coords, cells, atype = _build_system(**_SYSTEMS[system])
    box = cells if pbc else None

    # Anti-vacuity: the carry-all graph and the sel-capped dense reference only
    # agree when no center is sel-bound.  Assert the system is non-binding.
    max_nn = _max_neighbors(coords, box, atype)
    assert max_nn < SEL, (
        f"test system is sel-binding (max neighbors {max_nn} >= sel {SEL}); "
        "carry-all graph and sel-capped dense reference would diverge"
    )

    dp = DeepPot(pt2_path)
    assert dp.deep_eval.metadata["lower_input_kind"] == "graph"

    e, f, v, ae, av = dp.eval(coords, box, atype, atomic=True)
    ref = _eager_dense_reference(model, coords, box, atype)

    np.testing.assert_allclose(
        e.reshape(-1),
        ref["energy"].reshape(-1),
        rtol=1e-10,
        atol=1e-10,
        err_msg="energy",
    )
    np.testing.assert_allclose(
        f.reshape(-1), ref["force"].reshape(-1), rtol=1e-10, atol=1e-10, err_msg="force"
    )
    np.testing.assert_allclose(
        v.reshape(-1),
        ref["virial"].reshape(-1),
        rtol=1e-10,
        atol=1e-10,
        err_msg="virial",
    )
    np.testing.assert_allclose(
        ae.reshape(-1),
        ref["atom_energy"].reshape(-1),
        rtol=1e-10,
        atol=1e-10,
        err_msg="atom_energy",
    )
    np.testing.assert_allclose(
        av.reshape(-1),
        ref["atom_virial"].reshape(-1),
        rtol=1e-10,
        atol=1e-10,
        err_msg="atom_virial",
    )
