# SPDX-License-Identifier: LGPL-3.0-or-later
"""The OUTPUT-AGNOSTIC graph lower supports ANY fitting, not just energy.

A non-energy model with a graph-eligible descriptor (dpa1 ``attn_layer==0``)
routes into the graph path by default.  Before the general output transform this
KeyError'd on ``"energy"``; now every fitting (dos/dipole/polar/property/...)
flows through :func:`fit_output_to_model_output_graph` with no change on the
fitting side.  Each model's graph forward (EXPLICIT ``neighbor_graph_method``
opt-in; non-energy models default to dense since the energy gate)
must match the dense path (``neighbor_graph_method="legacy"``) on every shared
key (carry-all graph at non-binding ``sel`` reproduces the dense neighbor set).
"""

import pytest
import torch

from deepmd.pt.utils import (
    env,
)
from deepmd.pt_expt.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt_expt.fitting import (
    DipoleFitting,
    DOSFittingNet,
    PolarFitting,
    PropertyFittingNet,
)
from deepmd.pt_expt.model import (
    DipoleModel,
    DOSModel,
    PolarModel,
    PropertyModel,
)

from ...seed import (
    GLOBAL_SEED,
)


def _make_descriptor() -> DescrptDPA1:
    return DescrptDPA1(
        4.0,
        0.5,
        20,  # non-binding mixed-type single-int sel -> graph == dense neighbors
        2,
        attn_layer=0,  # graph lower only supports attn_layer == 0
        precision="float64",
        seed=GLOBAL_SEED,
    ).to(env.DEVICE)


def _make_dos(ds: DescrptDPA1):
    return DOSModel(
        ds,
        DOSFittingNet(
            2, ds.get_dim_out(), 5, mixed_types=ds.mixed_types(), seed=GLOBAL_SEED
        ).to(env.DEVICE),
        type_map=["a", "b"],
    ).to(env.DEVICE)


def _make_dipole(ds: DescrptDPA1):
    return DipoleModel(
        ds,
        DipoleFitting(
            2,
            ds.get_dim_out(),
            embedding_width=ds.get_dim_emb(),
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(env.DEVICE),
        type_map=["a", "b"],
    ).to(env.DEVICE)


def _make_polar(ds: DescrptDPA1):
    return PolarModel(
        ds,
        PolarFitting(
            2,
            ds.get_dim_out(),
            embedding_width=ds.get_dim_emb(),
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(env.DEVICE),
        type_map=["a", "b"],
    ).to(env.DEVICE)


def _make_property(ds: DescrptDPA1):
    return PropertyModel(
        ds,
        PropertyFittingNet(
            2,
            ds.get_dim_out(),
            task_dim=3,
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(env.DEVICE),
        type_map=["a", "b"],
    ).to(env.DEVICE)


class TestNonEnergyGraph:
    def setup_method(self) -> None:
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        self.coord = torch.rand(
            1, 5, 3, dtype=torch.float64, device=env.DEVICE, generator=generator
        )
        self.atype = torch.tensor([[0, 1, 0, 1, 0]], device=env.DEVICE)

    def test_dos_repro(self) -> None:
        """The exact bug repro: a DOS model's default forward used to KeyError
        on ``"energy"`` in the graph path; now it succeeds.
        """
        ds = _make_descriptor()
        ft = DOSFittingNet(2, ds.get_dim_out(), 5, mixed_types=ds.mixed_types()).to(
            env.DEVICE
        )
        m = DOSModel(ds, ft, type_map=["a", "b"]).to(env.DEVICE)
        out = m(self.coord, self.atype, box=None)
        # standard DOS model keys (no KeyError)
        assert set(out.keys()) >= {"atom_dos", "dos", "mask"}
        assert out["atom_dos"].shape == (1, 5, 5)
        assert out["dos"].shape == (1, 5)

    @pytest.mark.parametrize(
        "make_model",
        [_make_dos, _make_dipole, _make_polar, _make_property],
    )  # one builder per fitting kind
    def test_graph_matches_dense(self, make_model) -> None:
        """EXPLICIT graph opt-in matches the dense (``legacy``) path on every
        shared key, including derivatives for r/c-differentiable fittings.

        Non-energy models no longer DEFAULT-flip to the graph route (the
        compiled-training trace is energy-specific; see the
        ``_resolve_graph_method`` energy gate), but the eager graph lower
        stays output-agnostic and available via an explicit
        ``neighbor_graph_method`` -- which is what this parity test pins.
        """
        tol = (
            {"rtol": 1e-11, "atol": 1e-11}
            if env.DEVICE.type == "cpu"
            else {"rtol": 1e-9, "atol": 1e-9}
        )
        ds = _make_descriptor()
        m = make_model(ds)
        graph = m.call_common(
            self.coord,
            self.atype,
            None,
            do_atomic_virial=True,
            neighbor_graph_method="dense",
        )
        # the dense path differentiates w.r.t. coord -> needs a coord leaf.
        dense = m.call_common(
            self.coord.detach().requires_grad_(True),
            self.atype,
            None,
            do_atomic_virial=True,
            neighbor_graph_method="legacy",
        )
        shared = [
            k
            for k in graph
            if k in dense and graph[k] is not None and dense[k] is not None
        ]
        # at least the reduced + per-atom output must be present and shared
        assert len(shared) >= 2
        for k in shared:
            torch.testing.assert_close(
                graph[k].to(torch.float64), dense[k].to(torch.float64), **tol
            )
