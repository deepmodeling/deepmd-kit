# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph-lower export: forward_common_lower_graph_exportable traces + torch.export."""

import torch
from deepmd.pt.utils import env
from deepmd.pt_expt.descriptor.dpa1 import DescrptDPA1
from deepmd.pt_expt.fitting import InvarFitting
from deepmd.pt_expt.model import EnergyModel
from deepmd.dpmodel.utils.neighbor_graph import build_neighbor_graph
from ...seed import GLOBAL_SEED

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
    g = build_neighbor_graph(coord, atype, box, model.get_rcut())
    return (atype.reshape(-1), g.n_node, g.edge_index, g.edge_vec, g.edge_mask)


def test_graph_exportable_traces():
    model = _model().eval()
    atype, n_node, ei, ev, em = _graph_inputs(model)
    gm = model.forward_common_lower_graph_exportable(
        atype,
        n_node,
        ei,
        ev,
        em,
        do_atomic_virial=False,
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
    )
    assert isinstance(gm, torch.nn.Module)
    # the traced module reproduces eager outputs
    eager = model.forward_common_lower_graph(
        atype, n_node, ei, ev, em, do_atomic_virial=False
    )
    # traced module has placeholders for all 8 fn args (fparam/aparam/charge_spin=None)
    traced = gm(atype, n_node, ei, ev, em, None, None, None)
    # traced returns a tuple/dict; compare energy_redu
    te = traced["energy_redu"] if isinstance(traced, dict) else traced[1]
    torch.testing.assert_close(te, eager["energy_redu"], rtol=1e-10, atol=1e-10)
