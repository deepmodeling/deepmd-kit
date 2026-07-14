# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt DPA2 graph-lower Task 8 tests.

The pt_expt model plumbing that routes ``forward_common`` through the
carry-all graph (default-flip ``_resolve_graph_method``, autograd force/
virial via ``forward_common_lower_graph``, compiled training
``_trace_and_compile_graph``, the freeze gate) is GENERIC: it keys off
``descriptor.uses_graph_lower()`` and was implemented once (Task 7). DPA2
inherits it with ZERO new plumbing code -- these tests PROVE that
inheritance (routing/parity), plus cover the one piece of real code this
task adds: ``DescrptBlockRepformers._exchange_ghosts_graph``, the pt_expt
override that performs the per-layer MPI halo refresh via
``deepmd_export::border_op`` on the graph path (see
``deepmd/pt_expt/descriptor/repformers.py``).
"""

import ctypes
import importlib

import numpy as np
import pytest
import torch

from deepmd.dpmodel.descriptor.dpa2 import (
    RepformerArgs,
    RepinitArgs,
)
from deepmd.pt_expt.descriptor.dpa2 import (
    DescrptDPA2,
)
from deepmd.pt_expt.descriptor.repformers import (
    DescrptBlockRepformers,
)
from deepmd.pt_expt.fitting import (
    InvarFitting,
)
from deepmd.pt_expt.model import (
    EnergyModel,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)

# Trigger registration of the deepmd_export::border_op opaque wrapper
# (importlib form: the module is imported purely for its side effect).
importlib.import_module("deepmd.pt_expt.utils.comm")

# ---------------------------------------------------------------------------
# Self-comm-dict helper (mirrors
# ``source/tests/pt_expt/descriptor/test_repflow_parallel.py``): builds a
# single-rank, self-only MPI ``comm_dict`` whose effect is a plain gather
# from the sendlist-indexed local rows into the ghost slots, so the
# ``border_op`` self-send branch (no MPI runtime needed) can be exercised
# eagerly.


def _addr_of(np_arr: np.ndarray) -> int:
    """Return the raw int address of a numpy array's data buffer."""
    return np_arr.ctypes.data_as(ctypes.c_void_p).value


def _build_self_comm_dict(
    *,
    nloc: int,
    nghost: int,
    sendlist_indices: np.ndarray,
    keepalive: list,
) -> dict:
    """Build a comm_dict for a single-rank self-exchange.

    ``sendlist_indices`` (int32, length ``nghost``) gives the local row to
    copy into each successive ghost slot ``[nloc, nloc + nghost)``.  Control
    tensors are forced to CPU: the C++ ``border_op`` host-side code
    dereferences ``data_ptr<int>()`` directly.
    """
    sendlist_indices = np.ascontiguousarray(sendlist_indices, dtype=np.int32)
    keepalive.append(sendlist_indices)
    addr = _addr_of(sendlist_indices)
    return {
        "send_list": torch.tensor([addr], dtype=torch.int64, device="cpu"),
        "send_proc": torch.zeros(1, dtype=torch.int32, device="cpu"),
        "recv_proc": torch.zeros(1, dtype=torch.int32, device="cpu"),
        "send_num": torch.tensor([nghost], dtype=torch.int32, device="cpu"),
        "recv_num": torch.tensor([nghost], dtype=torch.int32, device="cpu"),
        "communicator": torch.zeros(1, dtype=torch.int64, device="cpu"),
        "nlocal": torch.tensor(nloc, dtype=torch.int32, device="cpu"),
        "nghost": torch.tensor(nghost, dtype=torch.int32, device="cpu"),
    }


def _make_dpa2_descriptor(
    ntypes: int = 2,
    repinit_rcut: float = 4.0,
    repinit_nsel: int = 200,
    repformer_rcut: float = 2.0,
    repformer_nsel: int = 150,
    repformer_attn: bool = False,
    nlayers: int = 2,
    g1_dim: int = 8,
    g2_dim: int = 4,
) -> DescrptDPA2:
    """Small graph-eligible DPA2 descriptor (use_three_body=False, concat
    tebd) with configurable sel and attention toggles.

    ``repformer_attn=False`` (default) keeps ``update_g1_has_attn`` /
    ``update_g2_has_attn`` off: with attention on, the carry-all graph's
    attention is sel-independent BY DESIGN (NeighborGraph PR-D) and
    diverges from the dense body even at non-binding sel, so exact
    graph-vs-dense parity requires attention off (dpa1
    ``test_force_virial_parity_vs_legacy`` precedent, and the dpmodel
    ``TestDPA2ModelEnergyCarryAll`` fixture).
    """
    repinit = RepinitArgs(
        rcut=repinit_rcut,
        rcut_smth=0.5,
        nsel=repinit_nsel,
        neuron=[3, 6],
        axis_neuron=2,
        tebd_dim=4,
    )
    repformer = RepformerArgs(
        rcut=repformer_rcut,
        rcut_smth=0.5,
        nsel=repformer_nsel,
        nlayers=nlayers,
        g1_dim=g1_dim,
        g2_dim=g2_dim,
        axis_neuron=2,
        update_g1_has_attn=repformer_attn,
        update_g2_has_attn=repformer_attn,
        attn1_hidden=4,
        attn1_nhead=2,
        attn2_hidden=4,
        attn2_nhead=2,
    )
    return DescrptDPA2(
        ntypes=ntypes,
        repinit=repinit,
        repformer=repformer,
        precision="float64",
        seed=GLOBAL_SEED,
    )


class TestDpa2GraphLower:
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

    def _make_model(
        self,
        repformer_nsel: int = 150,
        repinit_nsel: int = 200,
        repformer_attn: bool = False,
        numb_fparam: int = 0,
    ) -> EnergyModel:
        ds = _make_dpa2_descriptor(
            ntypes=self.nt,
            repinit_nsel=repinit_nsel,
            repformer_nsel=repformer_nsel,
            repformer_attn=repformer_attn,
        ).to(self.device)
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            numb_fparam=numb_fparam,
            precision="float64",
            seed=GLOBAL_SEED,
        ).to(self.device)
        return EnergyModel(ds, ft, type_map=self.type_map).to(self.device)

    # ------------------------------------------------------------------
    # 1. routing/parity: proves the Task-7 generic plumbing works for DPA2.
    # ------------------------------------------------------------------
    def test_force_virial_parity_vs_legacy(self) -> None:
        """Default-flip ``forward_common`` (graph) matches
        ``neighbor_graph_method="legacy"`` (dense) bit-tight at non-binding
        sel with repformer attention off.
        """
        model = self._make_model()  # non-binding sel, attention off
        model.eval()
        box = self.cell.reshape(1, 9)

        graph = model.forward_common(
            self.coord.clone().requires_grad_(True), self.atype, box
        )
        legacy = model.forward_common(
            self.coord.clone().requires_grad_(True),
            self.atype,
            box,
            neighbor_graph_method="legacy",
        )
        tol = {"rtol": 1e-10, "atol": 1e-10}
        torch.testing.assert_close(graph["energy_redu"], legacy["energy_redu"], **tol)
        torch.testing.assert_close(
            graph["energy_derv_r"], legacy["energy_derv_r"], **tol
        )
        torch.testing.assert_close(
            graph["energy_derv_c_redu"], legacy["energy_derv_c_redu"], **tol
        )

    def test_graph_native_hessian_matches_dense(self) -> None:
        """The graph route computes the Hessian natively and it matches the
        dense route bit-tight.

        A Hessian-enabled graph-eligible DPA2 model default-flips to the graph
        route; ``_call_common_graph`` now runs its own ``_cal_hessian_ext_graph``
        loop (differentiating the reduced energy w.r.t. the LOCAL coords by
        rebuilding the carry-all graph inside the autograd wrapper), so
        ``energy_derv_r_derv_r`` is produced without falling back to dense.
        The result must equal the dense route (forced via the
        ``disable_graph_lower`` escape hatch on the same weights) at fp64
        parity, in the same ``(nf, 1, nloc*3, nloc*3)`` layout.

        A non-binding repformer sel is used so graph and dense agree on the
        first derivatives too (attention off -- the fixture default); the
        Hessian parity would otherwise inherit the binding-sel divergence.
        """
        from deepmd.pt_expt.train.training import (
            _model_uses_graph_lower,
        )

        model = self._make_model()  # non-binding sel, attention off
        model.eval()
        assert model.atomic_model.descriptor.uses_graph_lower() is True
        model.enable_hessian()
        # a Hessian model stays graph-routed: the graph now produces the Hessian.
        assert _model_uses_graph_lower(model) is True

        box = self.cell.reshape(1, 9)
        # ``EnergyModel.forward`` exposes the Hessian under the ``"hessian"``
        # key (translated from ``energy_derv_r_derv_r``); it would KeyError if
        # the graph route failed to produce it.
        graph_out = model.forward(
            self.coord.clone().requires_grad_(True), self.atype, box=box
        )
        assert "hessian" in graph_out
        assert graph_out["hessian"].shape == (1, self.natoms * 3, self.natoms * 3)

        # dense reference on the SAME weights via the escape hatch.
        ref_model = self._make_model()
        ref_model.eval()
        ref_model.load_state_dict(model.state_dict(), strict=False)
        ref_model.atomic_model.descriptor.disable_graph_lower()
        ref_model.enable_hessian()
        assert _model_uses_graph_lower(ref_model) is False
        dense_out = ref_model.forward(
            self.coord.clone().requires_grad_(True), self.atype, box=box
        )
        torch.testing.assert_close(
            graph_out["hessian"], dense_out["hessian"], rtol=1e-9, atol=1e-9
        )
        # Hessian symmetry (a genuine second derivative, not a shape artifact).
        h = graph_out["hessian"][0]
        torch.testing.assert_close(h, h.transpose(-1, -2), rtol=1e-9, atol=1e-9)

    def test_disable_graph_lower_escape_hatch(self) -> None:
        """``descriptor.disable_graph_lower()`` is the documented legacy-dense
        escape hatch: it flips ``uses_graph_lower()`` to ``False`` so the
        default (``neighbor_graph_method=None``) eager forward takes the DENSE
        route -- bit-identical to explicitly requesting ``"legacy"``.

        Uses a BINDING repformer sel so graph and dense genuinely differ: with
        the hatch engaged, the default must match dense (not graph), which a
        binding sel makes an unambiguous (non-bit-tight) distinction.
        """
        generator = torch.Generator(device=self.device).manual_seed(GLOBAL_SEED)
        nloc = 12
        box_size = 3.0
        coord = (
            torch.rand(
                [nloc, 3], dtype=torch.float64, device=self.device, generator=generator
            )
            * box_size
        ).unsqueeze(0)
        atype = torch.tensor(
            [[ii % self.nt for ii in range(nloc)]],
            dtype=torch.int64,
            device=self.device,
        )
        box = (
            torch.eye(3, dtype=torch.float64, device=self.device) * box_size
        ).reshape(1, 9)

        model = self._make_model(repformer_nsel=3, repformer_attn=True)
        model.eval()
        assert model.atomic_model.descriptor.uses_graph_lower() is True

        # engage the escape hatch
        model.atomic_model.descriptor.disable_graph_lower()
        assert model.atomic_model.descriptor.uses_graph_lower() is False

        default_after = model.forward_common(
            coord.clone().requires_grad_(True), atype, box
        )
        legacy = model.forward_common(
            coord.clone().requires_grad_(True),
            atype,
            box,
            neighbor_graph_method="legacy",
        )
        # with the hatch on, default (None) == legacy (dense), bit-identical
        torch.testing.assert_close(
            default_after["energy_redu"], legacy["energy_redu"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            default_after["energy_derv_r"], legacy["energy_derv_r"], rtol=0, atol=0
        )

    def test_binding_sel_diverges(self) -> None:
        """At binding repformer sel, the carry-all graph (sel-independent)
        keeps neighbors the dense body truncates, so the two routes diverge.

        Uses a dense cluster (small box, more atoms) so each atom's real
        neighbor count within ``repformer_rcut`` comfortably exceeds
        ``repformer_nsel`` (dpmodel ``test_dpa2_call_graph.py`` binding-sel
        precedent: box_size=3.0, nloc=12, repformer_nsel=3).
        """
        generator = torch.Generator(device=self.device).manual_seed(GLOBAL_SEED)
        nloc = 12
        box_size = 3.0
        coord = (
            torch.rand(
                [nloc, 3], dtype=torch.float64, device=self.device, generator=generator
            )
            * box_size
        )
        coord = coord.unsqueeze(0)  # [1, nloc, 3]
        atype = torch.tensor(
            [[ii % self.nt for ii in range(nloc)]],
            dtype=torch.int64,
            device=self.device,
        )
        box = (
            torch.eye(3, dtype=torch.float64, device=self.device) * box_size
        ).reshape(1, 9)

        model = self._make_model(
            repformer_nsel=3
        )  # repinit_nsel stays 200 (non-binding)
        model.eval()

        graph = model.forward_common(coord.clone().requires_grad_(True), atype, box)
        legacy = model.forward_common(
            coord.clone().requires_grad_(True),
            atype,
            box,
            neighbor_graph_method="legacy",
        )
        e_diff = (graph["energy_redu"] - legacy["energy_redu"]).abs().max().item()
        e_scale = legacy["energy_redu"].abs().max().item()
        # non-zero (proves the default routed to the GRAPH, not a silent dense
        # fallback -- a fallback would make this exactly 0) AND bounded (the
        # divergence is the carry-all's extra in-cutoff neighbors, not a blow-up)
        assert e_diff > 1e-8, f"expected binding-sel divergence, got {e_diff:.3e}"
        assert e_diff < e_scale, (
            f"binding-sel divergence {e_diff:.3e} must stay below the energy "
            f"scale {e_scale:.3e} (bounded, not a blow-up)"
        )

    def test_binding_sel_diverges_with_attention(self) -> None:
        """Same binding-sel graph-vs-dense divergence as
        :meth:`test_binding_sel_diverges`, but with BOTH repformer attention
        channels ON -- the config iProzd's review asks for.

        With attention enabled and the fixed-phantom-count compensation, the
        two routes are bit-tight at NON-binding sel (see
        ``test_neighbor_list.py::test_default_fallback[dpa2]``). At BINDING sel
        they must still diverge, because the carry-all graph attends over
        neighbors the dense body truncates -- a difference that is both
        non-zero (the default IS the graph route) and bounded (below the energy
        scale). If the default silently fell back to dense, the difference
        would be exactly zero even with attention on.
        """
        generator = torch.Generator(device=self.device).manual_seed(GLOBAL_SEED)
        nloc = 12
        box_size = 3.0
        coord = (
            torch.rand(
                [nloc, 3], dtype=torch.float64, device=self.device, generator=generator
            )
            * box_size
        ).unsqueeze(0)
        atype = torch.tensor(
            [[ii % self.nt for ii in range(nloc)]],
            dtype=torch.int64,
            device=self.device,
        )
        box = (
            torch.eye(3, dtype=torch.float64, device=self.device) * box_size
        ).reshape(1, 9)

        model = self._make_model(repformer_nsel=3, repformer_attn=True)
        model.eval()

        graph = model.forward_common(coord.clone().requires_grad_(True), atype, box)
        legacy = model.forward_common(
            coord.clone().requires_grad_(True),
            atype,
            box,
            neighbor_graph_method="legacy",
        )
        for key in ("energy_redu", "energy_derv_r"):
            diff = (graph[key] - legacy[key]).abs().max().item()
            scale = legacy[key].abs().max().item()
            assert diff > 1e-8, (
                f"expected binding-sel divergence in {key} with attention on, "
                f"got {diff:.3e} (silent dense fallback?)"
            )
            assert diff < max(scale, 1.0), (
                f"{key} binding-sel divergence {diff:.3e} must stay bounded "
                f"below the scale {scale:.3e}"
            )

    def test_graph_lower_symbolic_trace(self) -> None:
        """``make_fx`` symbolic trace of ``forward_common_lower_graph``
        reproduces the eager graph lower bit-tight.  ``model.to("cpu")``
        before tracing mirrors the real ``.pt2`` export path and the dpa1
        CUDA lesson (traced inputs and params must share a device).
        """
        from deepmd.pt_expt.utils.serialization import (
            build_synthetic_graph_inputs,
        )

        model = self._make_model().to("cpu")
        model.eval()
        sample = build_synthetic_graph_inputs(
            model,
            e_max=175,
            nframes=2,
            nloc=7,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        atype, n_node, ei, ev, em, fp, ap, cs = sample
        traced = model.forward_lower_graph_exportable(
            atype,
            n_node,
            ei,
            ev,
            em,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=True,
            charge_spin=cs,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )
        out = traced(atype, n_node, ei, ev, em, fp, ap, cs)
        ref = model.forward_common_lower_graph(
            atype, n_node, ei, ev, em, fparam=fp, aparam=ap, do_atomic_virial=True
        )
        tol = {"rtol": 1e-12, "atol": 1e-12}
        torch.testing.assert_close(out["energy"], ref["energy_redu"], **tol)
        torch.testing.assert_close(
            out["force"], ref["energy_derv_r"].reshape(out["force"].shape), **tol
        )
        torch.testing.assert_close(
            out["virial"], ref["energy_derv_c_redu"].reshape(out["virial"].shape), **tol
        )

    def test_compiled_training_graph_smoke(self) -> None:
        """``_trace_and_compile_graph`` inductor-compiles the graph lower;
        one synthetic batch matches the eager graph lower at fp64 1e-10.

        NOTE: torch 2.11.0+cpu on an AVX2-only box can make inductor's C++
        vectorizer assert on the per-frame virial scatter
        (``AssertionError ... assert index.is_vec``, cpp.py:2980).
        ``_trace_and_compile_graph`` already disables CPU SIMD for the
        graph-lower compile internally (``extra_options={"cpp.simdlen":
        0}``) to route around this, so no test-side workaround is normally
        needed; the accepted fallback (project memory: torch 2.11 AVX2
        inductor bug) is ``torch._inductor.config.cpp.simdlen = 1`` if it
        ever resurfaces here.
        """
        from deepmd.pt_expt.train.training import (
            _trace_and_compile_graph,
        )
        from deepmd.pt_expt.utils.serialization import (
            build_synthetic_graph_inputs,
        )

        model = self._make_model().to("cpu")
        model.eval()

        compiled_lower, buf_order = _trace_and_compile_graph(model, None, None, None)
        assert isinstance(compiled_lower, torch.nn.Module)
        assert buf_order == ()

        sample = build_synthetic_graph_inputs(
            model,
            e_max=97,
            nframes=3,
            nloc=5,
            dtype=torch.float64,
            device=torch.device("cpu"),
            want_fparam=False,
            want_aparam=False,
            want_charge_spin=False,
        )
        atype, n_node, ei, ev, em, fp, ap, cs = sample

        compiled_out = compiled_lower(atype, n_node, ei, ev, em, fp, ap, cs)
        eager = model.forward_common_lower_graph(
            atype,
            n_node,
            ei,
            ev,
            em,
            do_atomic_virial=False,
            fparam=fp,
            aparam=ap,
            charge_spin=cs,
        )
        tol = {"rtol": 1e-10, "atol": 1e-10}
        torch.testing.assert_close(compiled_out["energy"], eager["energy_redu"], **tol)
        torch.testing.assert_close(
            compiled_out["force"],
            eager["energy_derv_r"].reshape(compiled_out["force"].shape),
            **tol,
        )
        torch.testing.assert_close(
            compiled_out["virial"],
            eager["energy_derv_c_redu"].reshape(compiled_out["virial"].shape),
            **tol,
        )

    def test_compiled_training_graph_small_sel(self) -> None:
        """The compiled-training trace capacity derives from the synthetic
        system's REAL edge count, not from ``sel``.

        The carry-all graph builder is sel-free (sel = normalization
        constant only), so a sel-derived static trace capacity
        (``ceil(1.25 * nloc * sum(sel))``) overflows whenever the synthetic
        trace system's actual degree exceeds ``sel``: with repinit/repformer
        ``nsel=10/6`` the trace used to raise ``edge overflow: 106 real
        edges > edge_capacity 89``.  The capacity is now probed from the
        actual unpadded synthetic graph, so the trace must succeed and the
        compiled lower must match the eager graph lower (compiled and eager
        are the SAME route, so parity holds even at binding sel).
        """
        from deepmd.pt_expt.train.training import (
            _trace_and_compile_graph,
        )
        from deepmd.pt_expt.utils.serialization import (
            build_synthetic_graph_inputs,
        )

        model = self._make_model(repinit_nsel=10, repformer_nsel=6).to("cpu")
        model.eval()

        compiled_lower, _ = _trace_and_compile_graph(model, None, None, None)

        sample = build_synthetic_graph_inputs(
            model,
            e_max=97,
            nframes=3,
            nloc=5,
            dtype=torch.float64,
            device=torch.device("cpu"),
            want_fparam=False,
            want_aparam=False,
            want_charge_spin=False,
        )
        atype, n_node, ei, ev, em, fp, ap, cs = sample
        compiled_out = compiled_lower(atype, n_node, ei, ev, em, fp, ap, cs)
        eager = model.forward_common_lower_graph(
            atype,
            n_node,
            ei,
            ev,
            em,
            do_atomic_virial=False,
            fparam=fp,
            aparam=ap,
            charge_spin=cs,
        )
        tol = {"rtol": 1e-10, "atol": 1e-10}
        torch.testing.assert_close(compiled_out["energy"], eager["energy_redu"], **tol)
        torch.testing.assert_close(
            compiled_out["force"],
            eager["energy_derv_r"].reshape(compiled_out["force"].shape),
            **tol,
        )

    def test_graph_lower_fparam_symbolic_trace_and_compile(self) -> None:
        """A graph-eligible DPA2 model with ``numb_fparam > 0`` must export
        (``make_fx`` symbolic) AND inductor-compile.

        Regression for the ``frame_id_from_n_node(graph.n_node)`` call in
        ``dp_atomic_model.forward_atomic_graph``: without a STATIC ``n_total``
        it fell back to ``int(sum(n_node))``, which make_fx / torch.export
        cannot evaluate on a traced tensor (``GuardOnDataDependentSymNode``),
        so both the graph ``.pt2`` export and compiled training failed for any
        fparam model. Both must now trace/compile and match eager bit-tight.
        """
        from deepmd.pt_expt.train.training import (
            _trace_and_compile_graph,
        )
        from deepmd.pt_expt.utils.serialization import (
            build_synthetic_graph_inputs,
        )

        model = self._make_model(numb_fparam=2).to("cpu")
        model.eval()
        assert model.atomic_model.descriptor.uses_graph_lower() is True

        sample = build_synthetic_graph_inputs(
            model,
            e_max=175,
            nframes=2,
            nloc=7,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        atype, n_node, ei, ev, em, fp, ap, cs = sample
        assert fp is not None and fp.shape[-1] == 2, "fparam must be present"

        # (a) symbolic-trace export path
        traced = model.forward_lower_graph_exportable(
            atype,
            n_node,
            ei,
            ev,
            em,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=True,
            charge_spin=cs,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )
        out = traced(atype, n_node, ei, ev, em, fp, ap, cs)
        ref = model.forward_common_lower_graph(
            atype, n_node, ei, ev, em, fparam=fp, aparam=ap, do_atomic_virial=True
        )
        tol = {"rtol": 1e-12, "atol": 1e-12}
        torch.testing.assert_close(out["energy"], ref["energy_redu"], **tol)
        torch.testing.assert_close(
            out["force"], ref["energy_derv_r"].reshape(out["force"].shape), **tol
        )

        # (b) compiled-training path (fparam threaded through the compile)
        compiled_lower, _ = _trace_and_compile_graph(model, fp, None, None)
        compiled_out = compiled_lower(atype, n_node, ei, ev, em, fp, ap, cs)
        ctol = {"rtol": 1e-10, "atol": 1e-10}
        torch.testing.assert_close(compiled_out["energy"], ref["energy_redu"], **ctol)

    # ------------------------------------------------------------------
    # 2. the one piece of real code: the border_op graph exchange override.
    # ------------------------------------------------------------------
    def test_graph_exchange_identity_when_no_comm(self) -> None:
        """``comm_dict is None`` -> identity (ghost-free Python graphs /
        extended single-process graphs need no exchange).
        """
        block = _make_dpa2_descriptor().repformers.to(self.device)
        assert isinstance(block, DescrptBlockRepformers)
        g1 = torch.rand(
            [self.natoms, block.get_dim_out()],
            dtype=torch.float64,
            device=self.device,
        )
        out = block._exchange_ghosts_graph(g1, None, self.natoms)
        assert out is g1

    def test_graph_exchange_border_op_self_communication(self) -> None:
        """A real (non-spin) ``comm_dict`` routes through ``border_op``:
        local rows [0, nlocal) are untouched and halo rows [nlocal, N)
        are overwritten with the owner rows named by the sendlist -- the
        actual new-code path this task adds (the identity/spin-reject
        tests above already hold on the dpmodel base and do not exercise
        ``border_op`` at all).
        """
        block = _make_dpa2_descriptor().repformers.to(self.device)
        nlocal, nghost = 4, 3
        n_total = nlocal + nghost
        dim = block.get_dim_out()
        g1_local = torch.rand([nlocal, dim], dtype=torch.float64, device=self.device)
        g1_ghost_junk = torch.full(
            [nghost, dim], -999.0, dtype=torch.float64, device=self.device
        )
        g1 = torch.cat([g1_local, g1_ghost_junk], dim=0)

        # ghost slot ii (0-indexed within the ghost block) mirrors local
        # owner (ii % nlocal).
        owners = np.array([ii % nlocal for ii in range(nghost)], dtype=np.int32)
        keepalive: list = []
        comm_dict = _build_self_comm_dict(
            nloc=nlocal,
            nghost=nghost,
            sendlist_indices=owners,
            keepalive=keepalive,
        )

        out = block._exchange_ghosts_graph(g1, comm_dict, n_total)

        torch.testing.assert_close(out[:nlocal], g1_local)
        expected_ghosts = g1_local[torch.as_tensor(owners, dtype=torch.int64)]
        torch.testing.assert_close(out[nlocal:], expected_ghosts)

    def test_graph_exchange_rejects_spin_comm(self) -> None:
        """A ``comm_dict`` describing a spin system raises ``NotImplementedError``
        -- spin models never route the graph lower (``disable_graph_lower``),
        so a ``has_spin`` comm_dict reaching the graph exchange seam is a
        programming error, not a supported configuration.
        """
        block = _make_dpa2_descriptor().repformers.to(self.device)
        g1 = torch.rand(
            [self.natoms, block.get_dim_out()],
            dtype=torch.float64,
            device=self.device,
        )
        comm_dict = {"has_spin": torch.ones(1)}
        with pytest.raises(NotImplementedError):
            block._exchange_ghosts_graph(g1, comm_dict, self.natoms)


# ---------------------------------------------------------------------------
# Task 9: owned-node (``n_local``) energy mask in the graph output reduction.
# ---------------------------------------------------------------------------


class TestOwnedNodeMaskEnergyReduction:
    """``n_local`` (owned-node mask) must exclude halo rows from the
    DIFFERENTIATED per-frame energy (each halo atom is owned -- and counted
    -- on another rank), while ``atom_energy`` itself stays FULL and the
    mask is applied BEFORE ``edge_energy_deriv`` differentiates the summed
    energy (so ``grad(energy, edge_vec)`` only carries owned-energy terms).
    """

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

    def _make_model(self) -> EnergyModel:
        ds = _make_dpa2_descriptor(ntypes=self.nt).to(self.device)
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            precision="float64",
            seed=GLOBAL_SEED,
        ).to(self.device)
        return EnergyModel(ds, ft, type_map=self.type_map).to(self.device)

    def _build_graph(self, model: EnergyModel):
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph,
        )

        box = self.cell.reshape(1, 9)
        return build_neighbor_graph(self.coord, self.atype, box, model.get_rcut())

    def test_owned_mask_energy_reduction(self) -> None:
        """``energy_redu`` with ``n_local`` == sum(atom_energy[:n_local]);
        ``atom_energy`` itself stays FULL/unmasked; halo rows [n_local:) of
        the assembled force are NOT all zero (they still receive src-scatter
        contributions from OWNED centers' edges).
        """
        model = self._make_model()
        model.eval()
        ng = self._build_graph(model)
        atype_flat = self.atype.reshape(-1)
        n_local_val = 4

        out_full = model.forward_common_lower_graph(
            atype_flat, ng.n_node, ng.edge_index, ng.edge_vec, ng.edge_mask
        )
        n_local = torch.tensor([n_local_val], dtype=torch.int64, device=self.device)
        out_masked = model.forward_common_lower_graph(
            atype_flat,
            ng.n_node,
            ng.edge_index,
            ng.edge_vec,
            ng.edge_mask,
            n_local=n_local,
        )

        # atom_energy (per-node) is FULL and byte-identical regardless of n_local.
        torch.testing.assert_close(out_masked["energy"], out_full["energy"])

        atom_energy = out_full["energy"].reshape(-1)
        owned_sum = atom_energy[:n_local_val].sum()
        torch.testing.assert_close(
            out_masked["energy_redu"].reshape(-1), owned_sum.reshape(1)
        )

        # halo-row partial forces survive (src-side scatter from owned edges).
        force = out_masked["energy_derv_r"].reshape(self.natoms, 3)
        halo_force = force[n_local_val:]
        assert not torch.allclose(halo_force, torch.zeros_like(halo_force)), (
            "expected nonzero halo-row partial force from owned-center edges"
        )

        # ordering check: the masked force must differ from the unmasked
        # (full-graph) force -- if the mask were applied AFTER
        # edge_energy_deriv (or not at all), both runs would produce the
        # identical force since the autograd leaf (edge_vec) is unchanged.
        force_full = out_full["energy_derv_r"].reshape(self.natoms, 3)
        assert not torch.allclose(force, force_full), (
            "masked force must differ from the unmasked force -- the "
            "owned-node mask must be applied BEFORE edge_energy_deriv"
        )

    def test_n_local_none_is_byte_identical(self) -> None:
        """Regression: omitting ``n_local`` (default ``None``) reproduces the
        pre-Task-9 unmasked reduction exactly.
        """
        model = self._make_model()
        model.eval()
        ng = self._build_graph(model)
        atype_flat = self.atype.reshape(-1)

        out_default = model.forward_common_lower_graph(
            atype_flat, ng.n_node, ng.edge_index, ng.edge_vec, ng.edge_mask
        )
        out_explicit_none = model.forward_common_lower_graph(
            atype_flat,
            ng.n_node,
            ng.edge_index,
            ng.edge_vec,
            ng.edge_mask,
            n_local=None,
        )
        torch.testing.assert_close(
            out_default["energy_redu"], out_explicit_none["energy_redu"]
        )
        torch.testing.assert_close(out_default["energy"], out_explicit_none["energy"])
        torch.testing.assert_close(
            out_default["energy_derv_r"], out_explicit_none["energy_derv_r"]
        )


class TestGraphAparamExtendedAxis:
    """aparam contract on the (extended-region) graph route.

    The graph fitting consumes atomic parameters on the FLAT node axis
    (``N = sum(n_node)``). On extended-region graphs (the multi-rank C++
    routes: ``N == nall_real``, owned prefix first, then halo) the caller
    must therefore supply an extended-axis aparam with the halo rows padded
    -- their fitting outputs are discarded by the owned-node mask, so the
    padded values are inert. This pins the contract the C++ runtime's
    ``extend_graph_aparam`` (DeepPotPTExpt.cc) implements: an owned-only
    (nloc-axis) aparam must fail loudly rather than silently misalign
    parameter rows with nodes.
    """

    def setup_method(self) -> None:
        self.device = env.DEVICE
        self.natoms = 6
        self.n_local_val = 4
        self.nt = 2
        generator = torch.Generator(device=self.device).manual_seed(GLOBAL_SEED)
        cell = torch.rand(
            [3, 3], dtype=torch.float64, device=self.device, generator=generator
        )
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device=self.device)
        self.cell = cell.unsqueeze(0)
        coord = torch.rand(
            [self.natoms, 3],
            dtype=torch.float64,
            device=self.device,
            generator=generator,
        )
        self.coord = torch.matmul(coord, cell).unsqueeze(0).to(self.device)
        self.atype = torch.tensor(
            [[0, 0, 0, 1, 1, 1]], dtype=torch.int64, device=self.device
        )

    def _make_model(self) -> EnergyModel:
        ds = _make_dpa2_descriptor(ntypes=self.nt).to(self.device)
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            numb_aparam=1,
            precision="float64",
            seed=GLOBAL_SEED,
        ).to(self.device)
        return EnergyModel(ds, ft, type_map=["foo", "bar"]).to(self.device)

    def _forward(self, model, aparam):
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph,
        )

        ng = build_neighbor_graph(
            self.coord, self.atype, self.cell.reshape(1, 9), model.get_rcut()
        )
        n_local = torch.tensor(
            [self.n_local_val], dtype=torch.int64, device=self.device
        )
        return model.forward_common_lower_graph(
            self.atype.reshape(-1),
            ng.n_node,
            ng.edge_index,
            ng.edge_vec,
            ng.edge_mask,
            aparam=aparam,
            n_local=n_local,
        )

    def test_extended_axis_accepted_halo_rows_inert(self) -> None:
        """Extended-axis aparam (N rows) runs; halo-row values are inert for
        the owned (masked) energy, owned-row values are not.
        """
        model = self._make_model()
        model.eval()
        base = torch.linspace(
            0.1, 0.6, self.natoms, dtype=torch.float64, device=self.device
        ).reshape(1, self.natoms, 1)
        out = self._forward(model, base)

        # halo rows [n_local:) changed -> owned energy identical
        halo_bump = base.clone()
        halo_bump[:, self.n_local_val :, :] += 7.5
        out_halo = self._forward(model, halo_bump)
        torch.testing.assert_close(out_halo["energy_redu"], out["energy_redu"])

        # an OWNED row changed -> owned energy must change
        owned_bump = base.clone()
        owned_bump[:, 0, :] += 7.5
        out_owned = self._forward(model, owned_bump)
        assert not torch.allclose(out_owned["energy_redu"], out["energy_redu"]), (
            "owned-row aparam change must reach the owned energy"
        )

    def test_owned_only_axis_fails_loudly(self) -> None:
        """An nloc-axis aparam (owned rows only) must raise -- silently
        misaligning parameter rows with the N-node axis is the bug the
        extended-axis contract prevents.
        """
        model = self._make_model()
        model.eval()
        owned_only = torch.linspace(
            0.1, 0.4, self.n_local_val, dtype=torch.float64, device=self.device
        ).reshape(1, self.n_local_val, 1)
        with pytest.raises((RuntimeError, ValueError)):
            self._forward(model, owned_only)
