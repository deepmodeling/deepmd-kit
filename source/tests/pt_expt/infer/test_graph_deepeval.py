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
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    is_vesin_torch_available,
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
    dense one. Varying ``natoms`` yields a different edge count and exercises
    the dynamic edge axis of the exported ``.pt2``.
    """
    rng = np.random.default_rng(seed)
    box_size = 18.0
    blob = rng.random((natoms, 3)) * 5.0 + box_size * 0.5 - 2.5
    coords = blob.reshape(1, natoms, 3)
    cells = (np.eye(3) * box_size).reshape(1, 9)
    # Alternate O/H types; both species present regardless of natoms.
    atype = np.array([i % 2 for i in range(natoms)], dtype=np.int32)
    return coords, cells, atype


# Two system sizes with different edge counts use the same exported artifact.
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


@pytest.fixture(scope="module", params=[0, 2], ids=["attn0", "attn2"])
def graph_pt2(request):
    """Build a dpa1 model and export it to a graph-form ``.pt2``.

    Parametrized over ``attn_layer``: 0 exercises the factorizable graph lower;
    2 exercises the carry-all ATTENTION graph lower, whose compact pair
    enumeration exports via unbacked SymInts (``xp_hint_dynamic_size``).
    ``smooth_type_embedding`` stays False: the smooth dense reference keeps
    sel-padding in its softmax denominator, so dense==carry-all parity holds
    only for the non-smooth branch.

    The AOTI compile is slow (~90 s), so it is done once per param.  The eager
    pt_expt model is returned alongside the archive path to serve as the dense
    parity reference.
    """
    from deepmd.pt_expt.model import (
        get_model,
    )

    config = copy.deepcopy(DPA1_CONFIG)
    config["descriptor"]["attn_layer"] = request.param
    config["descriptor"]["smooth_type_embedding"] = False
    model = get_model(config).to(torch.float64)
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

    Both systems use the same module-scoped artifact; their different edge
    counts exercise its dynamic edge axis.
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


@pytest.mark.skipif(not is_vesin_torch_available(), reason="vesin[torch] not installed")
@pytest.mark.parametrize("pbc", [True, False])  # periodic vs non-periodic
def test_graph_pt2_deepeval_vesin_matches_dense(graph_pt2, pbc) -> None:
    """Selecting neighbor_graph_method='vesin' at DeepEval yields identical
    energy/force/virial to the default 'dense' builder on the SAME graph ``.pt2``
    (the builder is a pure perf choice; neighbor sets are equal).
    """
    pt2_path, _ = graph_pt2
    coords, cells, atype = _build_system(**_SYSTEMS["small_8"])
    box = cells if pbc else None
    max_nn = _max_neighbors(coords, box, atype)
    assert max_nn < SEL, "test system must be non-binding for carry-all parity"

    dp_dense = DeepPot(pt2_path)  # default neighbor_graph_method == "dense"
    dp_vesin = DeepPot(pt2_path, neighbor_graph_method="vesin")
    assert dp_vesin.deep_eval._neighbor_graph_method == "vesin"

    e_d, f_d, v_d = dp_dense.eval(coords, box, atype)
    e_v, f_v, v_v = dp_vesin.eval(coords, box, atype)
    np.testing.assert_allclose(e_v, e_d, rtol=1e-10, atol=1e-10, err_msg="energy")
    np.testing.assert_allclose(f_v, f_d, rtol=1e-10, atol=1e-10, err_msg="force")
    np.testing.assert_allclose(v_v, v_d, rtol=1e-10, atol=1e-10, err_msg="virial")


def test_graph_pt2_single_atom_no_edges(graph_pt2) -> None:
    """A single isolated atom (zero real edges) evaluates through the ``.pt2``.

    The graph builder emits only masked guard edges here, so at runtime the
    compact pair enumeration sees ``R == 0`` real edges — the empty extreme of
    the unbacked-SymInt sizes the attention export carries.  Energy must match
    the eager dense reference and the force must be (numerically) zero.
    """
    pt2_path, model = graph_pt2
    coords = np.array([[[9.0, 9.0, 9.0]]])
    atype = np.array([0], dtype=np.int32)

    dp = DeepPot(pt2_path)
    e, f, v = dp.eval(coords, None, atype)[:3]
    ref = _eager_dense_reference(model, coords, None, atype)
    np.testing.assert_allclose(
        e.reshape(-1), ref["energy"].reshape(-1), rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(f.reshape(-1), 0.0, atol=1e-12)


def test_graph_pt2_deepeval_aparam(tmp_path) -> None:
    """Graph ``.pt2`` DeepEval with ``numb_aparam > 0``.

    DeepEval receives the user-facing rectangular ``(nf, natoms, nda)``
    aparam and must flatten it to the graph ABI's flat ``(N, nda)`` node
    axis before feeding the artifact (regression for the aparam
    graph-freeze review round). Checks parity vs the eager dense reference
    with the same aparam, and that aparam genuinely reaches the fitting.
    """
    from deepmd.pt_expt.model import (
        get_model,
    )

    config = copy.deepcopy(DPA1_CONFIG)
    config["descriptor"]["smooth_type_embedding"] = False
    config["fitting_net"] = {**config["fitting_net"], "numb_aparam": 1}
    model = get_model(config).to(torch.float64)
    model.eval()
    pt2_path = str(tmp_path / "deeppot_dpa1_graph_aparam.pt2")
    deserialize_to_file(
        pt2_path,
        {"model": model.serialize()},
        do_atomic_virial=True,
        lower_kind="graph",
    )

    coords, cells, atype = _build_system(**_SYSTEMS["small_8"])
    natoms = atype.shape[0]
    aparam = np.linspace(0.1, 0.9, natoms).reshape(1, natoms, 1)

    dp = DeepPot(pt2_path)
    assert dp.deep_eval.metadata["lower_input_kind"] == "graph"
    e, f, v = dp.eval(coords, cells, atype, atomic=False, aparam=aparam)[:3]

    # aparam must genuinely reach the fitting through the flat node axis
    e_bump = dp.eval(coords, cells, atype, atomic=False, aparam=aparam + 1.0)[0]
    assert not np.allclose(e_bump, e), "aparam bump must change the energy"

    # parity vs the eager dense (sel-capped, non-binding) reference
    coord_t = torch.tensor(
        coords.reshape(1, natoms, 3), dtype=torch.float64, device=DEVICE
    ).requires_grad_(True)
    atype_t = torch.tensor(atype.reshape(1, natoms), dtype=torch.int64, device=DEVICE)
    box_t = torch.tensor(cells.reshape(1, 9), dtype=torch.float64, device=DEVICE)
    ap_t = torch.tensor(
        aparam.reshape(1, natoms, 1), dtype=torch.float64, device=DEVICE
    )
    ret = model.call_common(
        coord_t,
        atype_t,
        box_t,
        aparam=ap_t,
        neighbor_graph_method="legacy",
    )
    np.testing.assert_allclose(
        e.reshape(-1),
        ret["energy_redu"].detach().cpu().numpy().reshape(-1),
        rtol=1e-10,
        atol=1e-10,
        err_msg="energy (graph .pt2 + aparam vs eager dense)",
    )
