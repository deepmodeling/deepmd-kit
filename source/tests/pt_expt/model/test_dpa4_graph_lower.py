# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 on the pt_expt NeighborGraph lower: routing, parity, persistence.

The pt_expt model plumbing that routes ``forward_common`` through the
carry-all graph (default-flip ``_resolve_graph_method``, autograd force/
virial via ``forward_common_lower_graph``, the persisted
``graph_lower_disabled`` escape hatch) is GENERIC: it keys off
``descriptor.uses_graph_lower()`` and was implemented once for dpa1/dpa2.
DPA4/SeZM inherits the routing with ZERO new plumbing code once
``DescrptDPA4.uses_graph_lower()`` reports ``True`` -- these tests PROVE
that inheritance (routing/parity/persistence), following the dpa2 pattern
in ``test_dpa2_graph_lower.py``.
"""

import numpy as np
import pytest
import torch

from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.pt_expt.model import (
    EnergyModel,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.model.graph_lower import (
    model_uses_graph_lower,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...common.dpmodel.test_dpa4_call_graph import (
    _jitter_zero_arrays,
)
from ...seed import (
    GLOBAL_SEED,
)

# Small fp64 DPA4/SeZM config -- copied verbatim from the descriptor block of
# ``source/tests/pt_expt/model/test_dpa4_export.py``'s ``_DPA4_CONFIG`` (that
# file documents it as "provably builds"; sel=20/rcut=4.0 is non-binding for
# the 6-atom fixture below -- the real max degree is 11, see the module
# docstring of ``test_dpa2_graph_lower.py`` for the non-binding-sel
# rationale).
_DPA4_CONFIG = {
    "type": "dpa4",
    "type_map": ["foo", "bar"],
    "descriptor": {
        "type": "dpa4",
        "sel": 20,
        "rcut": 4.0,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "precision": "float64",
        "seed": 1,
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [16],
        "precision": "float64",
        "seed": 1,
    },
}


def _make_model(device) -> EnergyModel:
    """Build a graph-eligible pt_expt DPA4/SeZM model from the exported config."""
    model = get_model(_DPA4_CONFIG)
    return model.to(device)


def _make_message_sensitive_model(device, seed: int = 99) -> EnergyModel:
    """A ``_make_model()`` variant with the zero-init residuals jittered.

    DPA4 deliberately zero-initializes several residual output projections
    (see the ``_jitter_zero_arrays`` docstring in
    ``test_dpa4_call_graph.py``) so a freshly constructed, untrained
    descriptor is architecturally edge/message INDEPENDENT: its scalar
    read-out is exactly the type embedding regardless of geometry or
    neighbors. That makes a bare ``_make_model()`` VACUOUS for a
    graph-vs-dense parity check -- the two routes would agree trivially
    because neither route's output depends on the edges at all. This
    jitters those (and only those) zero arrays in the descriptor's
    serialized parameter tree in place, so the model's energy genuinely
    depends on the neighbor edges (pinned by an in-test coordinate-
    perturbation guard in ``test_forward_common_graph_matches_dense``).
    """
    model = _make_model(device)
    ds = model.atomic_model.descriptor
    data = ds.serialize()
    _jitter_zero_arrays(data, np.random.default_rng(seed))
    jittered = DescrptDPA4.deserialize(data).to(device)
    # Standard torch.nn.Module submodule replacement: "descriptor" is
    # already a registered submodule of atomic_model, so this rebinds the
    # ``_modules`` entry in place.
    model.atomic_model.descriptor = jittered
    return model


class TestDpa4GraphLower:
    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.natoms = 6
        self.nt = 2
        self.type_map = ["foo", "bar"]

        generator = torch.Generator(device=self.device).manual_seed(GLOBAL_SEED)
        cell = torch.rand(
            [3, 3], dtype=torch.float64, device=self.device, generator=generator
        )
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device=self.device)
        self.cell = cell.unsqueeze(0)  # [1, 3, 3]
        coord = torch.rand(
            [self.natoms, 3],
            dtype=torch.float64,
            device=self.device,
            generator=generator,
        )
        coord = torch.matmul(coord, cell)
        self.coord = coord.unsqueeze(0).to(self.device)  # [1, natoms, 3]
        self.atype = torch.tensor(
            [[0, 0, 0, 1, 1, 1]], dtype=torch.int64, device=self.device
        )

    def test_model_uses_graph_lower(self) -> None:
        """A graph-eligible DPA4 model default-flips; the escape hatch flips
        it back off.
        """
        model = _make_model(self.device)
        assert model_uses_graph_lower(model) is True
        model.atomic_model.descriptor.disable_graph_lower()
        assert model_uses_graph_lower(model) is False

    def test_graph_lower_disabled_buffer_roundtrip(self) -> None:
        """``disable_graph_lower()`` is persisted state: it round-trips a
        ``state_dict`` save/load (the ``graph_lower_disabled`` buffer), and a
        PRE-knob checkpoint (no buffer key) still strict-loads, defaulting to
        the graph route.
        """
        model = _make_model(self.device)
        model.atomic_model.descriptor.disable_graph_lower()
        sd = model.state_dict()
        assert any("graph_lower_disabled" in k for k in sd), (
            "the hatch must ride the state_dict"
        )

        fresh = _make_model(self.device)
        assert fresh.atomic_model.descriptor.uses_graph_lower() is True
        fresh.load_state_dict(sd)
        assert fresh.atomic_model.descriptor.uses_graph_lower() is False, (
            "restored buffer must re-disable the graph route"
        )

        # back-compat: a checkpoint written before the knob was persisted
        # lacks the buffer key -- the strict load must still succeed and
        # keep the default (graph) route.
        old_sd = {
            k: v
            for k, v in _make_model(self.device).state_dict().items()
            if "graph_lower_disabled" not in k
        }
        fresh2 = _make_model(self.device)
        fresh2.load_state_dict(old_sd)
        assert fresh2.atomic_model.descriptor.uses_graph_lower() is True

    def test_forward_common_graph_matches_dense(self) -> None:
        """Default-flip ``forward_common`` (graph) matches
        ``neighbor_graph_method="legacy"`` (dense) at non-binding sel, on a
        message-sensitive (jittered) model.

        Force tolerance is one decade looser than energy: the two routes
        differentiate through different autograd chains (the graph's
        edge-vec leaf vs the dense route's local-coordinate leaf).
        """
        model = _make_message_sensitive_model(self.device)
        model.eval()
        box = self.cell.reshape(1, 9)

        graph = model.forward_common(
            self.coord.clone().requires_grad_(True), self.atype, box
        )
        dense = model.forward_common(
            self.coord.clone().requires_grad_(True),
            self.atype,
            box,
            neighbor_graph_method="legacy",
        )
        tol_e = {"rtol": 1e-10, "atol": 1e-12}
        tol_f = {"rtol": 1e-8, "atol": 1e-10}
        torch.testing.assert_close(graph["energy_redu"], dense["energy_redu"], **tol_e)
        torch.testing.assert_close(
            graph["energy_derv_r"], dense["energy_derv_r"], **tol_f
        )

        # Anti-vacuity guard: a coordinate perturbation must move the
        # energy. Without the jitter in _make_message_sensitive_model, DPA4's
        # zero-initialized residual projections make the output exactly
        # edge-independent, which would make the parity assertions above
        # trivially (and meaninglessly) true.
        perturbed_coord = self.coord.clone()
        perturbed_coord[0, 0, 0] += 0.1
        perturbed = model.forward_common(
            perturbed_coord.requires_grad_(True), self.atype, box
        )
        e_diff = (perturbed["energy_redu"] - graph["energy_redu"]).abs().max().item()
        assert e_diff > 1e-6, (
            f"expected the message-sensitive model's energy to depend on "
            f"coordinates; got a change of only {e_diff:.3e} (jitter not "
            f"effective -- parity check above would be vacuous)"
        )

    def test_graph_route_actually_taken(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anti-vacuity: the default route genuinely calls
        ``DescrptDPA4.call_graph``, and the disabled route does not (a
        silent fallback to dense on both branches would make the routing
        tests above vacuous).
        """
        model = _make_model(self.device)
        model.eval()
        box = self.cell.reshape(1, 9)

        calls = {"n": 0}
        original = DescrptDPA4.call_graph

        def _spy(self_, *args: object, **kwargs: object) -> object:
            calls["n"] += 1
            return original(self_, *args, **kwargs)

        monkeypatch.setattr(DescrptDPA4, "call_graph", _spy)

        model.forward_common(self.coord.clone().requires_grad_(True), self.atype, box)
        assert calls["n"] > 0, "default route must call DescrptDPA4.call_graph"

        calls["n"] = 0
        model.atomic_model.descriptor.disable_graph_lower()
        model.forward_common(self.coord.clone().requires_grad_(True), self.atype, box)
        assert calls["n"] == 0, "disabled route must not call DescrptDPA4.call_graph"
