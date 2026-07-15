# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph-lower export: forward_common_lower_graph_exportable traces + torch.export."""

import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt_expt.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt_expt.fitting import (
    InvarFitting,
)
from deepmd.pt_expt.model import (
    EnergyModel,
)

from ...seed import (
    GLOBAL_SEED,
)

_RCUT, _NT = 4.0, 2


def _model():
    ds = DescrptDPA1(
        _RCUT,
        0.5,
        20,
        _NT,
        neuron=[3, 6],
        axis_neuron=2,
        attn_layer=0,
        precision="float64",
        seed=GLOBAL_SEED,
    ).to(env.DEVICE)
    ft = InvarFitting(
        "energy",
        _NT,
        ds.get_dim_out(),
        1,
        mixed_types=ds.mixed_types(),
        precision="float64",
        seed=GLOBAL_SEED,
    ).to(env.DEVICE)
    return EnergyModel(ds, ft, type_map=["A", "B"]).to(env.DEVICE)


def _graph_inputs(model):
    rng = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
    nloc = 6
    coord = (
        torch.rand(1, nloc, 3, dtype=torch.float64, device=env.DEVICE, generator=rng)
        * 3.0
    )
    atype = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.int64, device=env.DEVICE)
    box = torch.eye(3, dtype=torch.float64, device=env.DEVICE).reshape(1, 9) * 20.0
    g = build_neighbor_graph(
        coord,
        atype,
        box,
        model.get_rcut(),
        canonicalize=True,
    )
    edge_mask = g.edge_mask.clone()
    assert bool(edge_mask[0])
    edge_mask[0] = False
    return (
        atype.reshape(-1),
        g.n_node,
        g.n_node,
        g.edge_index,
        g.edge_vec,
        edge_mask,
        g.destination_order,
        g.destination_row_ptr,
        g.source_order,
        g.source_row_ptr,
    )


def test_graph_exportable_traces():
    model = _model().eval()
    graph_inputs = _graph_inputs(model)
    gm = model.forward_common_lower_graph_exportable(
        *graph_inputs,
        do_atomic_virial=False,
        destination_sorted=True,
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
    )
    assert isinstance(gm, torch.nn.Module)
    # the traced module reproduces eager outputs
    eager = model.forward_common_lower_graph(*graph_inputs, do_atomic_virial=False)
    # Optional conditioning inputs remain explicit placeholders.
    traced = gm(*graph_inputs, None, None, None)
    # traced returns a tuple/dict; compare energy_redu
    te = traced["energy_redu"] if isinstance(traced, dict) else traced[1]
    torch.testing.assert_close(te, eager["energy_redu"], rtol=1e-10, atol=1e-10)


def test_graph_export_aparam_flat_node_axis():
    """The regular graph export ABI carries ``aparam`` FLAT on the node axis,
    sharing ``atype``'s dynamic ``N`` symbol.

    Regression (OutisLi review): the trace sample used to build ``aparam``
    rectangular ``(nf, nloc, nda)`` with independent ``nframes``/``nloc``
    dims while ``atype`` is flat ``(N,)``; the graph fitting views ``aparam``
    against the flat node count, so ``torch.export`` derived ``N == nf *
    nloc`` and rejected every ``numb_aparam > 0`` regular graph freeze with
    ``Constraints violated (n_node_total)`` (plus an ``N >= 4`` guard that
    would reject valid 1-to-3-node systems). The flat ``(N, nda)`` ABI must
    export, and the exported program must run BOTH a single-node ``nf=1,
    N=1`` system and a multi-frame ``nf=3, N=15`` system, matching eager.

    ``numb_aparam=2`` also pins the trace-dim selection: traced at the old
    fixed ``nframes=2``, make_fx duck-merged the STATIC ``nda == 2`` with
    the dynamic ``nf`` symbol, baking outputs whose frame axis followed the
    aparam width (silently wrong ``(2, 1)`` energy at ``nf=1``). The
    production trace (``_trace_and_export``, exercised here) now picks
    collision-free prime trace dims like compiled training does.
    """
    from deepmd.pt_expt.model.get_model import (
        get_model,
    )
    from deepmd.pt_expt.utils.serialization import (
        _trace_and_export,
        build_synthetic_graph_inputs,
    )

    config = {
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "se_atten",
            "sel": 20,
            "rcut": _RCUT,
            "rcut_smth": 0.5,
            "neuron": [3, 6],
            "axis_neuron": 2,
            "attn_layer": 0,
            "seed": GLOBAL_SEED,
        },
        "fitting_net": {
            "neuron": [16, 16],
            "numb_aparam": 2,
            "seed": GLOBAL_SEED,
        },
    }
    model = get_model(config).to("cpu")
    model.eval()

    exported, _meta, _dj, _keys = _trace_and_export(
        {"model": model.serialize()},
        model_json_override=None,
        lower_kind="graph",
    )
    loaded = exported.module()

    # nf=1, N=1 (single node: zero real edges, guard rows only) and a
    # multi-frame nf=3, N=15 system: both must pass the input guards and
    # match the eager graph lower.
    for nframes, nloc in ((1, 1), (3, 5)):
        inputs = build_synthetic_graph_inputs(
            model,
            e_max=None,
            nframes=nframes,
            nloc=nloc,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        (
            a2,
            nn2,
            nl2,
            ei2,
            ev2,
            em2,
            do2,
            drp2,
            so2,
            srp2,
            fp2,
            ap2,
            cs2,
        ) = inputs
        # single-rank runtime: every node is owned
        nl2 = nn2.clone()
        # distinct per-row values so the sensitivity check below is real
        ap2 = torch.linspace(0.1, 0.9, ap2.numel(), dtype=torch.float64).reshape(
            ap2.shape
        )
        out = loaded(a2, nn2, nl2, ei2, ev2, em2, do2, drp2, so2, srp2, fp2, ap2, cs2)
        ref = model.forward_common_lower_graph(
            a2,
            nn2,
            nl2,
            ei2,
            ev2,
            em2,
            do2,
            drp2,
            so2,
            srp2,
            fparam=fp2,
            aparam=ap2,
            do_atomic_virial=False,
        )
        torch.testing.assert_close(
            out["energy"], ref["energy_redu"], rtol=1e-10, atol=1e-10
        )
        # aparam must actually reach the fitting: bump it -> energy changes
        out_bump = loaded(
            a2, nn2, nl2, ei2, ev2, em2, do2, drp2, so2, srp2, fp2, ap2 + 1.5, cs2
        )
        assert not torch.allclose(out_bump["energy"], out["energy"]), (
            f"aparam bump must change the energy (nf={nframes}, nloc={nloc})"
        )


def test_graph_export_rejects_false_canonical_claim():
    model = _model().eval()
    graph_inputs = list(_graph_inputs(model))
    graph_inputs[6] = torch.flip(graph_inputs[6], dims=(0,))

    with pytest.raises(ValueError, match="destination_order"):
        model.forward_common_lower_graph_exportable(
            *graph_inputs,
            destination_sorted=True,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )


@pytest.mark.parametrize("do_atomic_virial", [False, True])  # both branches of the bool
def test_forward_lower_graph_exportable_public_keys(do_atomic_virial):
    """EnergyModel.forward_lower_graph_exportable: traces the public-key path and
    reproduces eager energy/force; atom_virial present iff do_atomic_virial.
    """
    model = _model().eval()
    graph_inputs = _graph_inputs(model)
    gm = model.forward_lower_graph_exportable(
        *graph_inputs,
        do_atomic_virial=do_atomic_virial,
        destination_sorted=True,
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
    )
    assert isinstance(gm, torch.nn.Module)
    out = gm(*graph_inputs, None, None, None)

    # public key set (graph path is local-only: force/atom_virial, NOT extended_*)
    assert "atom_energy" in out and "energy" in out and "force" in out
    assert "virial" in out
    assert "extended_force" not in out and "extended_virial" not in out
    # atom_virial appears ONLY when do_atomic_virial=True
    assert ("atom_virial" in out) == do_atomic_virial

    # values match the eager graph lower
    eager = model.forward_common_lower_graph(
        *graph_inputs, do_atomic_virial=do_atomic_virial
    )
    torch.testing.assert_close(
        out["energy"], eager["energy_redu"], rtol=1e-10, atol=1e-10
    )
    torch.testing.assert_close(
        out["force"], eager["energy_derv_r"].squeeze(-2), rtol=1e-10, atol=1e-10
    )
    if do_atomic_virial:
        torch.testing.assert_close(
            out["atom_virial"],
            eager["energy_derv_c"].squeeze(-2),
            rtol=1e-10,
            atol=1e-10,
        )
