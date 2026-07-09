# SPDX-License-Identifier: LGPL-3.0-or-later
"""Equivalence of the vesin O(N) ``NeighborList`` strategy with the default builder.

A ``NeighborList`` strategy is injected at ``forward_common``/``call_common``,
replacing the dense all-pairs ghost expansion (~27*N images + an O(N^2) distance
matrix) with vesin's O(N) cell list.  Both strategies hand the *same* extended
representation to the downstream model, so every model output (energy, force,
virial, atomic virial) must match the default builder to fp round-off.

Two layers are covered:

* ``test_builder_*`` -- the builder in isolation, asserting the per-atom
  neighbor *distance multisets* match the default, for the numpy (dpmodel) and
  torch (pt/pt_expt) namespaces, periodic and non-periodic, and that the
  returned tensors live on the input device.
* ``test_dpmodel_equivalence`` / ``test_pt_expt_equivalence`` /
  ``test_default_fallback`` -- full model equivalence across descriptor families
  (non-mixed, attention/mixed-types, message-passing with single and multiple
  cutoffs, repflows, hybrid), for dpmodel (energy/atomic energy) and pt_expt
  (energy/force/virial/atomic virial), periodic and non-periodic, including the
  ``neighbor_list=None`` default falling back to the dense builder
  byte-identically.
"""

import copy

import numpy as np
import pytest
import torch

from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.dpmodel.utils import (
    DefaultNeighborList,
    NeighborList,
)
from deepmd.pt_expt.model import (
    get_model,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    VesinNeighborList,
    is_vesin_torch_available,
)

from ...seed import (
    GLOBAL_SEED,
)

# --- compact model configs (3-type type_map), reduced layers for test speed ---
TYPE_MAP = ["O", "H", "B"]

model_se_e2_a = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "se_e2_a",
        "sel": [20, 20, 8],
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [6, 12],
        "resnet_dt": False,
        "axis_neuron": 4,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_se_r = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "se_e2_r",
        "sel": [20, 20, 8],
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [6, 12],
        "resnet_dt": False,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_se_e3 = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "se_e3",
        "sel": [12, 12, 4],
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [6, 12],
        "resnet_dt": False,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_dpa1 = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "se_atten",
        "sel": 40,
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [6, 12, 24],
        "axis_neuron": 4,
        "attn": 16,
        "attn_layer": 2,
        "attn_dotr": True,
        "attn_mask": False,
        "activation_function": "tanh",
        "scaling_factor": 1.0,
        "normalize": False,
        "temperature": 1.0,
        "set_davg_zero": True,
        "type_one_side": True,
        # smooth attention diverges between the carry-all graph default
        # (neighbor_list=None) and the explicit World-1 builders by design
        # (NeighborGraph PR-D: dense keeps sel-padding in the attention
        # softmax denominator); pin smooth off so all routes are exact.
        "smooth_type_embedding": False,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_dpa1_smooth = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "se_atten",
        "sel": 40,
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [6, 12, 24],
        "axis_neuron": 4,
        "attn": 16,
        "attn_layer": 2,
        "attn_dotr": True,
        "attn_mask": False,
        "activation_function": "tanh",
        "scaling_factor": 1.0,
        "normalize": False,
        "temperature": 1.0,
        "set_davg_zero": True,
        "type_one_side": True,
        # concat's counterpart to model_se_atten_v2 below: smooth attention
        # left ON (unlike model_dpa1 above, which pins it off), so the
        # tebd_input_mode="concat" + attn_layer>0 carry-all-vs-dense
        # divergence (see test_default_fallback's KNOWN_GRAPH_DENSE_DIVERGENT)
        # is exercised for concat too, not just strip (se_atten_v2).
        "smooth_type_embedding": True,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_se_atten_v2 = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "se_atten_v2",
        "sel": 40,
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [6, 12, 24],
        "axis_neuron": 4,
        "attn": 16,
        "attn_layer": 2,
        "attn_dotr": True,
        "attn_mask": False,
        "activation_function": "tanh",
        "set_davg_zero": False,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_dpa2 = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "dpa2",
        "repinit": {
            "rcut": 6.0,
            "rcut_smth": 2.0,
            "nsel": 30,
            "neuron": [2, 4, 8],
            "axis_neuron": 4,
            "activation_function": "tanh",
        },
        "repformer": {
            "rcut": 4.0,
            "rcut_smth": 0.5,
            "nsel": 20,
            "nlayers": 2,
            "g1_dim": 8,
            "g2_dim": 5,
            "attn2_hidden": 3,
            "attn2_nhead": 1,
            "attn1_hidden": 5,
            "attn1_nhead": 1,
            "axis_neuron": 4,
            "update_h2": False,
            "update_g1_has_conv": True,
            "update_g1_has_grrg": True,
            "update_g1_has_drrd": True,
            "update_g1_has_attn": True,
            "update_g2_has_g1g1": True,
            "update_g2_has_attn": True,
            "attn2_has_gate": True,
        },
        "seed": 1,
        "add_tebd_to_repinit_out": False,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_dpa3 = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "dpa3",
        "repflow": {
            "n_dim": 12,
            "e_dim": 8,
            "a_dim": 6,
            "nlayers": 2,
            "e_rcut": 6.0,
            "e_rcut_smth": 3.0,
            "e_sel": 20,
            "a_rcut": 4.0,
            "a_rcut_smth": 2.0,
            "a_sel": 10,
            "axis_neuron": 4,
            "update_angle": True,
            "update_style": "res_residual",
            "update_residual": 0.1,
            "update_residual_init": "const",
        },
        "activation_function": "tanh",
        "use_tebd_bias": False,
        "concat_output_tebd": False,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_hybrid = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "hybrid",
        "list": [
            {
                "type": "se_e2_a",
                "sel": [20, 20, 8],
                "rcut_smth": 0.5,
                "rcut": 4.0,
                "neuron": [6, 12],
                "resnet_dt": False,
                "axis_neuron": 4,
                "seed": 1,
            },
            {
                "type": "se_e2_r",
                "sel": [20, 20, 8],
                "rcut_smth": 0.5,
                "rcut": 4.0,
                "neuron": [6, 12],
                "resnet_dt": False,
                "seed": 1,
            },
        ],
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

ALL_MODELS = {
    "se_e2_a": model_se_e2_a,
    "se_r": model_se_r,
    "se_e3": model_se_e3,
    "dpa1": model_dpa1,
    "dpa1_smooth": model_dpa1_smooth,
    "se_atten_v2": model_se_atten_v2,
    "dpa2": model_dpa2,
    "dpa3": model_dpa3,
    "hybrid": model_hybrid,
}

# tebd_input_mode in {"concat", "strip"} with attn_layer > 0 and
# smooth_type_embedding=True: the carry-all graph default
# (neighbor_list=None) intentionally diverges from the dense route (see
# test_default_fallback's docstring). Both modes hit the same shared
# attention softmax mechanism (dpa1.py's `_graph_attention`, gated only on
# `attn_layer > 0`, entered identically regardless of concat/strip), so one
# tolerance covers both.
KNOWN_GRAPH_DENSE_DIVERGENT = {"dpa1_smooth", "se_atten_v2"}


def _system(natoms: int = 6, box_len: float = 10.0, seed: int = GLOBAL_SEED):
    """A small 3-type periodic system; returns numpy (coord, atype, box)."""
    rng = np.random.default_rng(seed)
    coord = (rng.random((1, natoms, 3)) * box_len).astype(np.float64)
    atype = np.array([[0, 0, 1, 1, 2, 0]], dtype=np.int64)[:, :natoms]
    box = (np.eye(3) * box_len).reshape(1, 9).astype(np.float64)
    return coord, atype, box


def _multiframe_system(nframes: int = 3, natoms: int = 6, seed: int = GLOBAL_SEED):
    """Multi-frame 3-type system whose frames have *different* geometries (and
    box sizes), so the per-frame ghost counts differ and the builder's
    pad-to-common-nall + stack path is exercised.
    """
    rng = np.random.default_rng(seed)
    coords, boxes = [], []
    for ff in range(nframes):
        box_len = 6.0 + 1.5 * ff  # vary box -> vary ghost count per frame
        coords.append((rng.random((natoms, 3)) * box_len).astype(np.float64))
        boxes.append((np.eye(3) * box_len).reshape(9).astype(np.float64))
    coord = np.stack(coords, axis=0)
    atype = np.tile(
        np.array([[0, 0, 1, 1, 2, 0]], dtype=np.int64)[:, :natoms], (nframes, 1)
    )
    box = np.stack(boxes, axis=0)
    return coord, atype, box


def _per_atom_neighbor_dists(ext_coord, nlist, coord):
    """Sorted, rounded valid-neighbor distances for each local atom."""
    ext_coord = np.asarray(ext_coord).reshape(-1, 3)
    coord = np.asarray(coord).reshape(-1, 3)
    out = []
    for i in range(coord.shape[0]):
        ds = [
            round(float(np.linalg.norm(ext_coord[j] - coord[i])), 6)
            for j in np.asarray(nlist)[i]
            if j >= 0
        ]
        out.append(sorted(ds))
    return out


pytestmark = pytest.mark.skipif(
    not is_vesin_torch_available(), reason="vesin.torch is not installed"
)


def _tol(model_dict: dict) -> dict:
    """Equivalence tolerance; loosened only for float32 models."""
    prec = str(model_dict["descriptor"].get("precision", "float64"))
    if "32" in prec:
        return {"rtol": 1e-5, "atol": 1e-5}
    return {"rtol": 1e-9, "atol": 1e-9}


def test_builder_isinstance() -> None:
    assert isinstance(VesinNeighborList(), NeighborList)
    assert isinstance(DefaultNeighborList(), NeighborList)


@pytest.mark.parametrize("periodic", [False, True])  # non-PBC vs PBC
def test_builder_matches_default(periodic: bool) -> None:
    """The vesin builder produces the same per-atom neighbor distance multisets
    as the default all-pairs builder, in both the numpy (dpmodel) and torch
    (pt/pt_expt) namespaces.
    """
    coord_np, atype_np, box_np = _system()
    box_np = box_np if periodic else None
    rcut, sel = 4.0, [20, 20, 8]
    ec_d, _, nl_d, _ = DefaultNeighborList().build(
        coord_np, atype_np, box_np, rcut, sel
    )
    ref = _per_atom_neighbor_dists(ec_d, nl_d[0], coord_np[0])
    # numpy (dpmodel) namespace
    ec_v, _, nl_v, _ = VesinNeighborList().build(coord_np, atype_np, box_np, rcut, sel)
    assert _per_atom_neighbor_dists(ec_v, nl_v[0], coord_np[0]) == ref
    # torch (pt/pt_expt) namespace
    coord_t = torch.tensor(coord_np, dtype=torch.float64)
    atype_t = torch.tensor(atype_np, dtype=torch.int64)
    box_t = None if box_np is None else torch.tensor(box_np, dtype=torch.float64)
    ec_vt, _, nl_vt, _ = VesinNeighborList().build(coord_t, atype_t, box_t, rcut, sel)
    assert torch.is_tensor(ec_vt)
    assert (
        _per_atom_neighbor_dists(
            ec_vt[0].cpu().numpy(), nl_vt[0].cpu().numpy(), coord_np[0]
        )
        == ref
    )


@pytest.mark.parametrize("periodic", [False, True])  # non-PBC vs PBC
def test_builder_empty_system(periodic: bool) -> None:
    """A zero-atom frame must not crash vesin (which rejects empty points); the
    builder returns an empty extended representation, matching the native path.
    """
    coord = np.zeros((1, 0, 3), dtype=np.float64)
    atype = np.zeros((1, 0), dtype=np.int64)
    box = (np.eye(3) * 10.0).reshape(1, 9).astype(np.float64) if periodic else None
    sel = [20, 20, 8]
    ec, ea, nl, mp = VesinNeighborList().build(coord, atype, box, 4.0, sel)
    assert ec.shape == (1, 0, 3)
    assert ea.shape == (1, 0)
    assert nl.shape == (1, 0, sum(sel))
    assert mp.shape == (1, 0)


def test_builder_outputs_on_input_device() -> None:
    coord_np, atype_np, box_np = _system()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coord_t = torch.tensor(coord_np, dtype=torch.float64, device=device)
    atype_t = torch.tensor(atype_np, dtype=torch.int64, device=device)
    box_t = torch.tensor(box_np, dtype=torch.float64, device=device)
    for t in VesinNeighborList().build(coord_t, atype_t, box_t, 4.0, [20, 20, 8]):
        assert t.device.type == device.type


@pytest.mark.parametrize("periodic", [False, True])  # non-PBC vs PBC
def test_builder_multiframe_matches_default(periodic: bool) -> None:
    """Multi-frame build (frames with differing ghost counts) exercises the
    pad-to-common-nall + stack path; every frame's neighbor multiset must still
    match the default builder, in numpy and torch namespaces.
    """
    coord_np, atype_np, box_np = _multiframe_system()
    box_np = box_np if periodic else None
    rcut, sel = 4.0, [20, 20, 8]
    ec_d, _, nl_d, _ = DefaultNeighborList().build(
        coord_np, atype_np, box_np, rcut, sel
    )
    ec_v, _, nl_v, _ = VesinNeighborList().build(coord_np, atype_np, box_np, rcut, sel)
    coord_t = torch.tensor(coord_np, dtype=torch.float64)
    atype_t = torch.tensor(atype_np, dtype=torch.int64)
    box_t = None if box_np is None else torch.tensor(box_np, dtype=torch.float64)
    ec_vt, _, nl_vt, _ = VesinNeighborList().build(coord_t, atype_t, box_t, rcut, sel)
    for ff in range(coord_np.shape[0]):
        ref = _per_atom_neighbor_dists(ec_d[ff], nl_d[ff], coord_np[ff])
        assert _per_atom_neighbor_dists(ec_v[ff], nl_v[ff], coord_np[ff]) == ref
        assert (
            _per_atom_neighbor_dists(
                ec_vt[ff].cpu().numpy(), nl_vt[ff].cpu().numpy(), coord_np[ff]
            )
            == ref
        )


@pytest.mark.parametrize("name", list(ALL_MODELS))  # descriptor family
@pytest.mark.parametrize("periodic", [False, True])  # non-PBC vs PBC
def test_dpmodel_equivalence(name: str, periodic: bool) -> None:
    """Energy / atomic energy (dpmodel) are invariant to the nlist strategy."""
    coord_np, atype_np, box_np = _system()
    box_np = box_np if periodic else None
    model_dict = ALL_MODELS[name]
    md = get_model_dp(copy.deepcopy(model_dict))
    r0 = md.call(coord_np, atype_np, box=box_np, neighbor_list=DefaultNeighborList())
    r1 = md.call(coord_np, atype_np, box=box_np, neighbor_list=VesinNeighborList())
    for k in ("energy", "atom_energy"):
        np.testing.assert_allclose(
            r0[k], r1[k], err_msg=f"{name} {k}", **_tol(model_dict)
        )


@pytest.mark.parametrize("name", list(ALL_MODELS))  # descriptor family
@pytest.mark.parametrize("periodic", [False, True])  # non-PBC vs PBC
def test_pt_expt_equivalence(name: str, periodic: bool) -> None:
    """pt_expt energy / force / virial / atomic virial are invariant to the
    nlist strategy (force/virial come from the existing autograd routines).
    """
    coord_np, atype_np, box_np = _system()
    box_np = box_np if periodic else None
    model_dict = ALL_MODELS[name]
    md = get_model(copy.deepcopy(model_dict)).to(env.DEVICE)
    md.eval()
    box_t = (
        None
        if box_np is None
        else torch.tensor(box_np, dtype=torch.float64, device=env.DEVICE)
    )
    atype_t = torch.tensor(atype_np, dtype=torch.int64, device=env.DEVICE)
    results = {}
    for tag, nl in (("def", DefaultNeighborList()), ("ves", VesinNeighborList())):
        coord_t = torch.tensor(
            coord_np, dtype=torch.float64, device=env.DEVICE
        ).requires_grad_(True)
        results[tag] = md.forward(
            coord_t, atype_t, box=box_t, do_atomic_virial=True, neighbor_list=nl
        )
    for k in ("energy", "atom_energy", "force", "virial", "atom_virial"):
        a, b = results["def"].get(k), results["ves"].get(k)
        if a is None or b is None:
            continue
        np.testing.assert_allclose(
            a.detach().cpu().numpy(),
            b.detach().cpu().numpy(),
            err_msg=f"{name} {k}",
            **_tol(model_dict),
        )


@pytest.mark.parametrize("name", ["se_e2_a", "dpa1"])  # non-mixed + attention
def test_pt_expt_multiframe_equivalence(name: str) -> None:
    """Multi-frame (frames with differing ghost counts) pt_expt outputs are
    invariant to the nlist strategy -- exercises the builder's per-frame pad +
    stack feeding the batched model forward.
    """
    coord_np, atype_np, box_np = _multiframe_system()
    model_dict = ALL_MODELS[name]
    md = get_model(copy.deepcopy(model_dict)).to(env.DEVICE)
    md.eval()
    atype_t = torch.tensor(atype_np, dtype=torch.int64, device=env.DEVICE)
    box_t = torch.tensor(box_np, dtype=torch.float64, device=env.DEVICE)
    results = {}
    for tag, nl in (("def", DefaultNeighborList()), ("ves", VesinNeighborList())):
        coord_t = torch.tensor(
            coord_np, dtype=torch.float64, device=env.DEVICE
        ).requires_grad_(True)
        results[tag] = md.forward(
            coord_t, atype_t, box=box_t, do_atomic_virial=True, neighbor_list=nl
        )
    for k in ("energy", "force", "virial", "atom_virial"):
        np.testing.assert_allclose(
            results["def"][k].detach().cpu().numpy(),
            results["ves"][k].detach().cpu().numpy(),
            err_msg=f"{name} {k}",
            **_tol(model_dict),
        )


@pytest.mark.parametrize("name", list(ALL_MODELS))  # descriptor family
def test_default_fallback(name: str) -> None:
    """``neighbor_list=None`` dispatches to the same DefaultNeighborList builder.

    ``None`` and an explicit ``DefaultNeighborList()`` are the identical builder
    (``call_common`` does ``builder = nl if nl is not None else DefaultNeighborList()``),
    so the two forward passes are the *same computation*; on CPU they are
    bit-identical.  We compare with a tight tolerance rather than exact equality
    because the two passes are independent forward evaluations, and on CUDA the
    GNN message-passing scatter (atomic adds) is not bit-reproducible run-to-run,
    so the virial can differ by ~1 ULP between the passes (a real dispatch bug
    would differ by orders of magnitude more).

    ``KNOWN_GRAPH_DENSE_DIVERGENT`` models (``dpa1_smooth``, ``se_atten_v2``)
    are a special case: that "same builder" premise only holds for the
    DENSE-nlist route.  For a ``mixed_types`` descriptor with
    ``uses_graph_lower() == True``, passing an explicit ``neighbor_list`` forces
    the dense route (``call_common``'s ``neighbor_list is not None`` branch),
    while ``None`` lets pt_expt's default-flip (decision #17) route to the
    carry-all graph instead -- two genuinely different algorithms, not two
    evaluations of one. Both hardcode/pin ``smooth_type_embedding=True``
    (unlike ``model_dpa1`` above, which pins it ``False`` for exactly this
    reason), so graph and dense intentionally diverge (NeighborGraph PR-D:
    dense keeps sel-padding phantom terms in the attention softmax
    denominator, the graph route does not -- see
    ``test_block_compact_graph_smooth_clean_divergence`` in
    ``test_dpa1_graph_attention_parity.py`` for the same invariant at the
    block level). At this test's non-binding ``sel=40`` (vs. <=5 real
    neighbors), the gap is small but non-zero and deterministic (not CUDA
    ULP-style non-determinism): energy differs by ~1e-7, force by ~1e-6,
    virial (a derivative, so it amplifies the softmax-denominator
    perturbation more than the value itself) by up to ~1.3e-5. We assert
    BOUNDED closeness (atol=3e-5, rtol=1e-3 -- individual virial/force
    components can be near-zero, so atol dominates) rather than bit-identity
    for these two, keeping the check meaningful rather than silently dropping
    coverage.
    """
    coord_np, atype_np, box_np = _system()
    md = get_model(copy.deepcopy(ALL_MODELS[name])).to(env.DEVICE)
    md.eval()
    box_t = torch.tensor(box_np, dtype=torch.float64, device=env.DEVICE)
    atype_t = torch.tensor(atype_np, dtype=torch.int64, device=env.DEVICE)
    outs = {}
    for tag, kw in (
        ("none", {}),
        ("explicit", {"neighbor_list": DefaultNeighborList()}),
    ):
        coord_t = torch.tensor(
            coord_np, dtype=torch.float64, device=env.DEVICE
        ).requires_grad_(True)
        outs[tag] = md.forward(coord_t, atype_t, box=box_t, do_atomic_virial=True, **kw)
    tol = (
        {"rtol": 1e-3, "atol": 3e-5}
        if name in KNOWN_GRAPH_DENSE_DIVERGENT
        else {"rtol": 1e-10, "atol": 1e-12}
    )
    for k in ("energy", "force", "virial"):
        np.testing.assert_allclose(
            outs["none"][k].detach().cpu().numpy(),
            outs["explicit"][k].detach().cpu().numpy(),
            err_msg=f"{name} {k}",
            **tol,
        )
