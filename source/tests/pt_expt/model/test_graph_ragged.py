# SPDX-License-Identifier: LGPL-3.0-or-later
"""Ragged n_node test for forward_common_lower_graph.

Verifies that the flat-N graph transform correctly handles ragged frames
(n_node=[3,2], N=5): energy shape (5,1), energy_redu shape (2,1),
energy_derv_r leading dim 5.  All entries must be finite.
"""

import torch

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

_RCUT = 3.0
_NT = 2


def _make_model() -> EnergyModel:
    ds = DescrptDPA1(
        _RCUT,
        0.5,
        10,
        _NT,
        neuron=[3, 6],
        axis_neuron=2,
        attn=4,
        attn_layer=0,
        attn_dotr=True,
        attn_mask=False,
        activation_function="tanh",
        set_davg_zero=True,
        type_one_side=True,
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


def _make_ragged_graph(device: torch.device) -> tuple:
    """Build a ragged graph with n_node=[3,2] (N=5).

    Frame 0: atoms 0,1,2 — fully connected (6 directed edges within rcut).
    Frame 1: atoms 3,4   — fully connected (2 directed edges within rcut).
    Edge vectors are chosen to be small enough to fall within _RCUT.
    """
    rng = torch.Generator(device=device).manual_seed(GLOBAL_SEED)
    # flat atom types (N=5)
    atype = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int64, device=device)
    # n_node per frame
    n_node = torch.tensor([3, 2], dtype=torch.int64, device=device)
    # edge_index: all pairs within each frame (flat indices into [0,4])
    # frame 0: 0↔1, 0↔2, 1↔2  (both directions = 6 edges)
    # frame 1: 3↔4             (both directions = 2 edges)
    src = torch.tensor([0, 1, 0, 2, 1, 2, 3, 4], dtype=torch.int64, device=device)
    dst = torch.tensor([1, 0, 2, 0, 2, 1, 4, 3], dtype=torch.int64, device=device)
    edge_index = torch.stack([src, dst], dim=0)  # (2, 8)
    # edge_vec: random small vectors well within rcut
    edge_vec = (
        torch.rand(8, 3, dtype=torch.float64, device=device, generator=rng) * 0.5
    ).detach()
    edge_mask = torch.ones(8, dtype=torch.bool, device=device)
    return atype, n_node, edge_index, edge_vec, edge_mask


class TestGraphRagged:
    def setup_method(self) -> None:
        self.model = _make_model()
        self.model.eval()
        self.device = env.DEVICE
        self.atype, self.n_node, self.edge_index, self.edge_vec, self.edge_mask = (
            _make_ragged_graph(self.device)
        )

    def test_flat_energy_shapes(self) -> None:
        """forward_common_lower_graph returns flat (N,1) energy, (nf,1) energy_redu."""
        ret = self.model.forward_common_lower_graph(
            self.atype,
            self.n_node,
            self.edge_index,
            self.edge_vec,
            self.edge_mask,
            do_atomic_virial=False,
        )
        N = int(self.n_node.sum())  # 5
        nf = int(self.n_node.shape[0])  # 2
        # per-atom energy: flat (N, *shap) = (5, 1)
        assert ret["energy"].shape == (N, 1), (
            f"expected (5,1) got {ret['energy'].shape}"
        )
        # reduced energy: per-frame (nf, *shap) = (2, 1)
        assert ret["energy_redu"].shape == (nf, 1), (
            f"expected (2,1) got {ret['energy_redu'].shape}"
        )
        # force: flat leading dim N
        assert ret["energy_derv_r"].shape[0] == N, (
            f"expected leading dim 5 got {ret['energy_derv_r'].shape}"
        )
        # all finite
        assert torch.isfinite(ret["energy"]).all()
        assert torch.isfinite(ret["energy_redu"]).all()
        assert torch.isfinite(ret["energy_derv_r"]).all()

    def test_flat_atom_virial_shapes(self) -> None:
        """With do_atomic_virial=True, atom_virial is also flat (N,1,9)."""
        ret = self.model.forward_common_lower_graph(
            self.atype,
            self.n_node,
            self.edge_index,
            self.edge_vec,
            self.edge_mask,
            do_atomic_virial=True,
        )
        N = int(self.n_node.sum())  # 5
        nf = int(self.n_node.shape[0])  # 2
        assert ret["energy"].shape == (N, 1)
        assert ret["energy_redu"].shape == (nf, 1)
        assert ret["energy_derv_r"].shape[0] == N
        assert ret["energy_derv_c"].shape[0] == N
        assert ret["energy_derv_c_redu"].shape[0] == nf
        assert torch.isfinite(ret["energy_derv_c"]).all()
        assert torch.isfinite(ret["energy_derv_c_redu"]).all()

    def test_invariant_to_charge_spin(self) -> None:
        """dpa1 does NOT consume charge_spin (``get_dim_chg_spin() == 0``);
        forward_common_lower_graph accepts it only for ABI stability with
        charge/spin descriptors (dpa3/dpa4, PR-G), so energy / force / virial /
        atom-virial must be INVARIANT to it.
        """
        assert self.model.get_descriptor().get_dim_chg_spin() == 0  # dpa1
        args = (
            self.atype,
            self.n_node,
            self.edge_index,
            self.edge_vec,
            self.edge_mask,
        )
        base = self.model.forward_common_lower_graph(*args, do_atomic_virial=True)
        nf = int(self.n_node.shape[0])
        # arbitrary non-None charge/spin -> must NOT change any dpa1 graph output
        cs = torch.tensor(
            [[1.0, 2.0]] * nf, dtype=torch.float64, device=self.device
        )
        with_cs = self.model.forward_common_lower_graph(
            *args, do_atomic_virial=True, charge_spin=cs
        )
        assert set(base) == set(with_cs)
        for k, v in base.items():
            if v is None:
                assert with_cs[k] is None
            else:
                torch.testing.assert_close(with_cs[k], v, rtol=1e-12, atol=1e-12)
