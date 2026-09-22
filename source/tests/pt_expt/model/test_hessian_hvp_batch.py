# SPDX-License-Identifier: LGPL-3.0-or-later
"""``DP_HESSIAN_HVP_BATCH`` must not change the Hessian, only how it is computed.

The graph route assembles the Hessian from Hessian-vector products and can
evaluate several rows per second-order backward by replicating the structure
along the frame axis.  Frames are independent, so the replicated energy is a
sum of independent terms and its Hessian is block diagonal -- the batched
result is exact, and these tests pin that down in float64.

``test_dpa2_graph_lower`` already reaches the batched helper, but only because
the batch it happens to get exceeds 1: a batch of 1 would turn that coverage
into coverage of the unbatched path alone, silently.  Here the batch is the
object under test, and a counter asserts which branch ran so the coverage
cannot drift away again.

``DP_HESSIAN_HVP_BATCH`` unset means "choose per call from free memory", so the
choice itself and the out-of-memory fallback are tested too.
"""

import pytest
import torch

import deepmd.pt_expt.model.make_model as mm
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
from deepmd.pt_expt.model.graph_lower import (
    model_uses_graph_lower,
)

from ...seed import (
    GLOBAL_SEED,
)

NATOMS = 5
NDOF = 3 * NATOMS  # 15: odd, so most batch sizes leave a partial final chunk
RCUT = 4.0
RCUT_SMTH = 0.5
SEL = 20  # mixed-type single-int sel
NT = 2

# 15 % 2 == 1 and 15 % 8 == 7 exercise the zero-padded final chunk; 15 % 3 == 0
# divides evenly; 16 exceeds NDOF and must be clamped back to a single chunk.
BATCHES = [2, 3, 4, 8, 16]
PADS = {b for b in BATCHES if NDOF % min(b, NDOF)}


@pytest.fixture
def route_counts(monkeypatch):
    """Count which Hessian implementation each forward actually took."""
    counts = {"batched": 0, "graph": 0, "dense": 0}
    originals = {
        "batched": mm._hessian_graph_batched_hvp,
        "graph": mm._cal_hessian_ext_graph,
        "dense": mm._cal_hessian_ext,
    }
    names = {
        "batched": "_hessian_graph_batched_hvp",
        "graph": "_cal_hessian_ext_graph",
        "dense": "_cal_hessian_ext",
    }

    def make(key):
        original = originals[key]

        def counted(*args, **kwargs):
            counts[key] += 1
            return original(*args, **kwargs)

        return counted

    for key, name in names.items():
        monkeypatch.setattr(mm, name, make(key))
    return counts


class TestHessianHvpBatch:
    """Batched and unbatched Hessian-vector products must agree exactly."""

    def setup_method(self) -> None:
        self.device = env.DEVICE
        generator = torch.Generator(device=self.device).manual_seed(GLOBAL_SEED)
        cell = torch.rand(
            [3, 3], dtype=torch.float64, device=self.device, generator=generator
        )
        cell = (cell + cell.T) + 5.0 * torch.eye(
            3, device=self.device, dtype=torch.float64
        )
        self.box = cell.reshape(1, 9)
        coord = torch.rand(
            [NATOMS, 3],
            dtype=torch.float64,
            device=self.device,
            generator=generator,
        )
        self.coord = (coord @ cell).unsqueeze(0)
        self.atype = torch.tensor(
            [[0, 0, 0, 1, 1]], dtype=torch.int64, device=self.device
        )

    def _make_model(self, graph: bool = True) -> EnergyModel:
        ds = DescrptDPA1(
            RCUT,
            RCUT_SMTH,
            SEL,
            NT,
            neuron=[3, 6],
            axis_neuron=2,
            attn=4,
            attn_layer=0,
            attn_dotr=True,
            attn_mask=False,
            # Smooth attention keeps sel-padding in the dense softmax
            # denominator, which the carry-all graph omits; exact graph-vs-dense
            # parity needs it off.
            smooth_type_embedding=False,
            activation_function="tanh",
            set_davg_zero=False,
            type_one_side=True,
            precision="float64",
            seed=GLOBAL_SEED,
        ).to(self.device)
        ft = InvarFitting(
            "energy",
            NT,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            precision="float64",
            seed=GLOBAL_SEED,
        ).to(self.device)
        model = EnergyModel(ds, ft, type_map=["foo", "bar"]).to(self.device)
        model.eval()
        if not graph:
            model.atomic_model.descriptor.disable_graph_lower()
        model.enable_hessian()
        assert model_uses_graph_lower(model) is graph
        return model

    def _hessian(self, model: EnergyModel) -> torch.Tensor:
        out = model.forward(
            self.coord.clone().requires_grad_(True), self.atype, box=self.box
        )
        return out["hessian"].reshape(NDOF, NDOF)

    def test_batch_one_takes_the_unbatched_path(
        self, route_counts, monkeypatch
    ) -> None:
        """1 (and 0) must reach the original one-row-at-a-time implementation."""
        model = self._make_model()
        for batch in (0, 1):
            monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", batch)
            self._hessian(model)
        assert route_counts["graph"] == 2
        assert route_counts["batched"] == 0, "batch<=1 must not take the batched helper"
        assert route_counts["dense"] == 0

    @pytest.mark.parametrize("batch", BATCHES)
    def test_batched_matches_unbatched(self, batch, route_counts, monkeypatch) -> None:
        """Every batch size must reproduce the unbatched Hessian to float64 precision."""
        model = self._make_model()

        monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", 1)
        reference = self._hessian(model)
        assert route_counts["batched"] == 0

        monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", batch)
        batched = self._hessian(model)
        assert route_counts["batched"] == 1, "the batched helper did not run"
        assert route_counts["dense"] == 0

        scale = reference.abs().max()
        torch.testing.assert_close(
            batched, reference, rtol=0.0, atol=float(1e-12 * scale)
        )

    @pytest.mark.parametrize("batch", sorted(PADS))
    def test_padded_final_chunk_is_discarded(self, batch, monkeypatch) -> None:
        """A batch size that does not divide 3*nloc still yields exactly 3*nloc rows.

        The final chunk is zero-padded to keep the retained graph's shape; those
        rows must be dropped, not returned.
        """
        assert NDOF % batch, f"{batch} divides {NDOF}; it exercises no padding"
        model = self._make_model()
        monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", batch)
        hessian = self._hessian(model)
        assert hessian.shape == (NDOF, NDOF)
        # A dropped-row bug shows up as an all-zero row, and a padding leak as a
        # zero row in the middle; neither is possible for a real Hessian here.
        assert (hessian.abs().sum(dim=1) > 0).all()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="the automatic choice needs CUDA"
    )
    def test_probe_prices_one_product_not_the_whole_hessian(self, monkeypatch) -> None:
        """The automatic choice must not pay for a Hessian to decide the batch.

        ``max_rows`` is what keeps the probe to a single Hessian-vector product;
        ignoring it would still give the right batch, just after doing all the
        work the batch was supposed to speed up.
        """
        calls = []
        real = mm._hessian_graph_batched_hvp

        def record(*args, **kwargs):
            out = real(*args, **kwargs)
            calls.append((kwargs.get("max_rows"), out.shape[0]))
            return out

        monkeypatch.setattr(mm, "_hessian_graph_batched_hvp", record)
        monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", None)
        model = self._make_model()
        self._hessian(model)

        assert calls, "the automatic choice never ran the probe"
        probe_max_rows, probe_rows = calls[0]
        assert probe_max_rows == 1
        assert probe_rows == 1, "the probe computed more than one row"

    def test_create_graph_keeps_the_path_to_the_coordinates(self, monkeypatch) -> None:
        """``create_graph`` must leave the Hessian differentiable in the input.

        ``torch.autograd.functional.hessian`` keeps the input in the graph when
        ``create_graph`` is set; detaching instead severs everything upstream of
        the coordinates while leaving the parameter path intact, so the loss of
        signal is silent.
        """
        model = self._make_model()
        model.train()
        monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", 4)
        upstream = self.coord.clone().requires_grad_(True)
        # a non-leaf coordinate, which is what makes the severed path visible
        out = model.forward(upstream * 1.0, self.atype, box=self.box)
        hessian = out["hessian"].reshape(NDOF, NDOF)
        assert hessian.requires_grad, "create_graph produced a detached Hessian"
        (back,) = torch.autograd.grad(
            hessian.sum(), upstream, retain_graph=True, allow_unused=True
        )
        assert back is not None, "the Hessian no longer depends on the coordinates"
        assert torch.isfinite(back).all()

    def test_a_linear_energy_gives_a_zero_hessian_not_a_crash(
        self, monkeypatch
    ) -> None:
        """Constant or linear coordinate dependence must yield zeros.

        ``functional.hessian`` materialises the zero block under its default
        ``strict=False``. Differentiating a constant first derivative again
        instead raises, which turns a legitimate model into a crash.
        """

        class _LinearAtomicModel:
            """Energy exactly linear in the coordinates: d2E/dx2 == 0."""

            def forward_common_atomic_graph(self, graph, atype_flat, **kwargs):
                nb = graph.shape[0]
                energy = (3.0 * graph).sum(-1).reshape(nb * graph.shape[1], 1)
                return {"energy": energy}

        class _Model:
            atomic_model = _LinearAtomicModel()

        monkeypatch.setattr(
            mm,
            "build_neighbor_graph_for_method",
            lambda method, pos, atype, box, rcut, pair_excl: pos,
        )
        nloc = NATOMS
        coord_flat = self.coord.reshape(-1).clone()
        hessian = mm._hessian_graph_batched_hvp(
            model=_Model(),
            kk="energy",
            ci=0,
            nloc=nloc,
            coord_flat=coord_flat,
            atype=self.atype,
            box=self.box,
            method="graph",
            pair_excl=None,
            rcut=RCUT,
            fparam=None,
            aparam=None,
            spin=None,
            charge_spin=None,
            batch=4,
            create_graph=False,
        )
        assert hessian.shape == (NDOF, NDOF)
        assert torch.count_nonzero(hessian) == 0, "a linear energy has no curvature"

    def test_the_probe_does_not_build_a_full_identity(self, monkeypatch) -> None:
        """Pricing one product must not allocate the ``ndof x ndof`` identity.

        The identity is the allocation batching exists to avoid; building it to
        decide the batch can itself be what runs the device out of memory, and
        it inflates the very cost the probe is measuring.
        """
        seen = []
        real_eye = torch.eye

        def record_eye(n, *args, **kwargs):
            seen.append(n)
            return real_eye(n, *args, **kwargs)

        monkeypatch.setattr(torch, "eye", record_eye)
        monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", 4)
        model = self._make_model()
        self._hessian(model)
        assert NDOF not in seen, (
            f"an identity of size {NDOF} was built; sizes seen: {seen}"
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="the automatic choice needs CUDA"
    )
    def test_an_out_of_memory_probe_falls_back_instead_of_escaping(
        self, monkeypatch
    ) -> None:
        """Pricing a product is a product, so it can be the thing that does not fit.

        Letting that escape ends the run before the one-row-at-a-time path --
        which may well have fit -- is ever tried.
        """
        monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", None)

        def always_oom(device, probe):
            raise torch.OutOfMemoryError("probe did not fit")

        monkeypatch.setattr(mm, "_auto_hvp_batch", always_oom)
        model = self._make_model()
        hessian = self._hessian(model)
        assert hessian.shape == (NDOF, NDOF)
        assert torch.isfinite(hessian).all()

    def test_dense_route_is_untouched(self, route_counts, monkeypatch) -> None:
        """The batch size must not reach, or change, the dense Hessian route."""
        dense_model = self._make_model(graph=False)
        monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", 8)
        dense = self._hessian(dense_model)
        assert route_counts["dense"] == 1
        assert route_counts["graph"] == 0
        assert route_counts["batched"] == 0

        graph_model = self._make_model(graph=True)
        graph_batched = self._hessian(graph_model)
        assert route_counts["batched"] == 1

        # An independent implementation agreeing to float64 precision is a
        # stronger statement than batched-vs-unbatched alone: both graph paths
        # share the same wrapper, the dense route does not.
        scale = dense.abs().max()
        torch.testing.assert_close(
            graph_batched, dense, rtol=0.0, atol=float(1e-9 * scale)
        )

    def test_oom_halves_the_batch_and_keeps_the_answer(
        self, route_counts, monkeypatch
    ) -> None:
        model = self._make_model()

        monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", 1)
        reference = self._hessian(model)

        attempted = []
        real = mm._hessian_graph_batched_hvp

        def only_small_batches_fit(*args, **kwargs):
            attempted.append(kwargs["batch"])
            if kwargs["batch"] > 2:
                raise torch.OutOfMemoryError("simulated")
            return real(*args, **kwargs)

        monkeypatch.setattr(mm, "_hessian_graph_batched_hvp", only_small_batches_fit)
        monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", 8)
        recovered = self._hessian(model)

        assert attempted == [8, 4, 2], attempted
        scale = reference.abs().max()
        torch.testing.assert_close(
            recovered, reference, rtol=0.0, atol=float(1e-12 * scale)
        )

    def test_oom_all_the_way_down_lands_on_the_unbatched_path(
        self, route_counts, monkeypatch
    ) -> None:
        """When nothing fits, the fallback bottoms out in the original path."""
        model = self._make_model()

        monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", 1)
        reference = self._hessian(model)
        batched_before = route_counts["batched"]

        def nothing_fits(*args, **kwargs):
            raise torch.OutOfMemoryError("simulated")

        monkeypatch.setattr(mm, "_hessian_graph_batched_hvp", nothing_fits)
        monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", 8)
        recovered = self._hessian(model)

        assert route_counts["batched"] == batched_before, (
            "the counter wraps the real helper, which was replaced"
        )
        scale = reference.abs().max()
        torch.testing.assert_close(
            recovered, reference, rtol=0.0, atol=float(1e-12 * scale)
        )

    def test_hessian_is_symmetric(self, monkeypatch) -> None:
        """Batching changes the summation order; it must not break symmetry."""
        model = self._make_model()
        monkeypatch.setattr(mm, "DP_HESSIAN_HVP_BATCH", 8)
        hessian = self._hessian(model)
        scale = hessian.abs().max()
        torch.testing.assert_close(
            hessian, hessian.T, rtol=0.0, atol=float(1e-12 * scale)
        )


class TestHvpBatchPolicy:
    """The automatic batch: bounded, monotone in free memory, and recoverable."""

    MIB = 1024 * 1024

    def _probe(self, nbytes: int):
        """A stand-in Hessian-vector product that costs a known amount."""

        def probe():
            return torch.empty(nbytes, dtype=torch.uint8, device=env.DEVICE)

        return probe

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="the automatic batch needs CUDA"
    )
    @pytest.mark.parametrize("free_mib", [8, 64, 256, 1024, 4096, 16384, 65536])
    def test_auto_batch_is_bounded(self, free_mib, monkeypatch) -> None:
        """Whatever the free memory, the batch stays within [1, cap]."""
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda *a, **k: (free_mib * self.MIB, 0)
        )
        batch = mm._auto_hvp_batch(env.DEVICE, self._probe(32 * self.MIB))
        assert 1 <= batch <= mm.DP_HESSIAN_HVP_BATCH_CAP

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="the automatic batch needs CUDA"
    )
    def test_auto_batch_rises_with_free_memory(self, monkeypatch) -> None:
        """More memory must never buy a smaller batch."""
        chosen = []
        for free_mib in (8, 32, 128, 512, 2048, 8192):
            monkeypatch.setattr(
                torch.cuda,
                "mem_get_info",
                lambda *a, _f=free_mib, **k: (_f * self.MIB, 0),
            )
            chosen.append(mm._auto_hvp_batch(env.DEVICE, self._probe(32 * self.MIB)))
        assert chosen == sorted(chosen), chosen
        # The ends must actually differ, or monotonicity is vacuous.
        assert chosen[0] == 1
        assert chosen[-1] == mm.DP_HESSIAN_HVP_BATCH_CAP

    def test_auto_batch_is_one_without_cuda(self) -> None:
        """No allocator introspection and no recoverable OOM: stay unbatched."""

        def explode():  # pragma: no cover - must never be called
            raise AssertionError("the probe must not run on a non-CUDA device")

        assert mm._auto_hvp_batch(torch.device("cpu"), explode) == 1
