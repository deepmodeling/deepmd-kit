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

Task 8 (graph ``.pt2`` export) adds ``test_graph_lower_symbolic_trace`` and
``test_graph_lower_torch_export``: both went GREEN with zero production
changes -- ``forward_lower_graph_exportable``/``forward_common_lower_graph``
and ``_build_graph_dynamic_shapes`` are already output-agnostic over the
descriptor, so DPA4 (channels=16, n_radial=8, lmax=2, mmax=1, n_blocks=2,
SO(3) grid readout) traces and ``torch.export``s through the SAME generic
machinery Task 7 built for dpa1/dpa2, with no ``int()``/``.item()`` calls or
data-dependent ``if`` on a traced tensor anywhere in the DPA4-specific code
path exercised here (``AOTI indirect-indexing`` was already globally
disabled for DPA4 per Task 6). No trap-class fix was needed.
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

    def test_graph_lower_symbolic_trace(self) -> None:
        """``make_fx`` symbolic trace of ``forward_lower_graph_exportable``
        reproduces the eager graph lower bit-tight, on a message-sensitive
        (jittered) model.

        ``forward_common_lower_graph`` computes force/virial via a single
        ``torch.autograd.grad`` backward through its own ``edge_vec`` leaf
        (see ``edge_transform_output.py:106``); tracing the whole exportable
        wrapper with ``make_fx`` therefore traces that backward pass too --
        this is what makes the DPA4 SO(3)/GridMLP compute graph, including
        its analytic force/virial, ``.pt2``-exportable.  ``model.to("cpu")``
        before tracing mirrors the real ``.pt2`` export path (make_fx traces
        on CPU by design, ``serialization.py:924``) and the dpa1 CUDA lesson
        (traced inputs and params must share a device).
        """
        from deepmd.pt_expt.utils.serialization import (
            build_synthetic_graph_inputs,
        )

        model = _make_message_sensitive_model(self.device).to("cpu")
        model.eval()
        sample = build_synthetic_graph_inputs(
            model,
            e_max=175,
            nframes=2,
            nloc=7,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        atype, n_node, n_local, ei, ev, em, do, drp, so, srp, fp, ap, cs = sample
        traced = model.forward_lower_graph_exportable(
            atype,
            n_node,
            n_local,
            ei,
            ev,
            em,
            do,
            drp,
            so,
            srp,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=True,
            charge_spin=cs,
            destination_sorted=True,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )
        out = traced(atype, n_node, n_local, ei, ev, em, do, drp, so, srp, fp, ap, cs)
        ref = model.forward_common_lower_graph(
            atype,
            n_node,
            n_local,
            ei,
            ev,
            em,
            do,
            drp,
            so,
            srp,
            destination_sorted=True,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=True,
        )
        for key in ("energy", "force", "virial"):
            assert torch.isfinite(out[key]).all(), f"non-finite traced {key}"
        tol = {"rtol": 1e-12, "atol": 1e-12}
        torch.testing.assert_close(out["energy"], ref["energy_redu"], **tol)
        torch.testing.assert_close(
            out["force"], ref["energy_derv_r"].reshape(out["force"].shape), **tol
        )
        torch.testing.assert_close(
            out["virial"], ref["energy_derv_c_redu"].reshape(out["virial"].shape), **tol
        )

    def test_graph_lower_torch_export(self) -> None:
        """``torch.export.export`` the traced graph lower with the
        production dynamic shapes (``_build_graph_dynamic_shapes``): the
        edge axis ``E``, the flat node axis ``N``, and the frame axis ``nf``
        are all dynamic. Must export without ``GuardOnDataDependentSymNode``
        and the resulting exported program must reproduce the eager graph
        lower AND generalize to a different (smaller) system size than the
        one it was traced/exported on -- proving the dynamism is real, not
        an artifact baked to the trace-time shapes.
        """
        from deepmd.pt_expt.utils.serialization import (
            _build_graph_dynamic_shapes,
            build_synthetic_graph_inputs,
        )

        model = _make_message_sensitive_model(self.device).to("cpu")
        model.eval()
        sample = build_synthetic_graph_inputs(
            model,
            e_max=175,
            nframes=2,
            nloc=7,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        atype, n_node, n_local, ei, ev, em, do, drp, so, srp, fp, ap, cs = sample
        traced = model.forward_lower_graph_exportable(
            atype,
            n_node,
            n_local,
            ei,
            ev,
            em,
            do,
            drp,
            so,
            srp,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=True,
            charge_spin=cs,
            destination_sorted=True,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )
        dynamic_shapes = _build_graph_dynamic_shapes(
            atype, n_node, n_local, ei, ev, em, do, drp, so, srp, fp, ap, cs
        )
        exported = torch.export.export(
            traced,
            (atype, n_node, n_local, ei, ev, em, do, drp, so, srp, fp, ap, cs),
            dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
        loaded = exported.module()

        # Re-run on a SMALLER system (different nframes/nloc/edge count) to
        # prove the exported program is genuinely dynamic, not specialized
        # to the trace-time shapes.
        small = build_synthetic_graph_inputs(
            model,
            e_max=None,
            nframes=1,
            nloc=3,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        (
            s_atype,
            s_n_node,
            s_n_local,
            s_ei,
            s_ev,
            s_em,
            s_do,
            s_drp,
            s_so,
            s_srp,
            s_fp,
            s_ap,
            s_cs,
        ) = small
        out = loaded(
            s_atype,
            s_n_node,
            s_n_local,
            s_ei,
            s_ev,
            s_em,
            s_do,
            s_drp,
            s_so,
            s_srp,
            s_fp,
            s_ap,
            s_cs,
        )
        ref = model.forward_common_lower_graph(
            s_atype,
            s_n_node,
            s_n_local,
            s_ei,
            s_ev,
            s_em,
            s_do,
            s_drp,
            s_so,
            s_srp,
            destination_sorted=True,
            fparam=s_fp,
            aparam=s_ap,
            do_atomic_virial=True,
        )
        for key in ("energy", "force", "virial"):
            assert torch.isfinite(out[key]).all(), f"non-finite exported {key}"
        tol = {"rtol": 1e-10, "atol": 1e-10}
        torch.testing.assert_close(out["energy"], ref["energy_redu"], **tol)
        torch.testing.assert_close(
            out["force"], ref["energy_derv_r"].reshape(out["force"].shape), **tol
        )
        torch.testing.assert_close(
            out["virial"], ref["energy_derv_c_redu"].reshape(out["virial"].shape), **tol
        )
