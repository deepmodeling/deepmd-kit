# SPDX-License-Identifier: LGPL-3.0-or-later
"""``force_mag`` autograd on the DPA4 pt_expt NeighborGraph lower.

Task 3 of the "DPA4 native spin on the NeighborGraph route" plan wires a
SECOND autograd leaf (``spin``) alongside the existing ``edge_vec`` leaf in
``forward_common_lower_graph``/``fit_output_to_model_output_graph``: every
``r_differentiable`` reducible output additionally emits
``<var>_derv_r_mag = -d<var>_redu/dspin``. This exercises the pt_expt
BACKBONE energy model directly (a plain "dpa4" model config with
``use_spin`` set on the descriptor) -- NOT the ``DPA4NativeSpinModel``
wrapper (that is Task 4); ``get_sezm_model`` only rejects a top-level
``"spin"`` key, so setting ``use_spin`` on the descriptor of an otherwise
plain ``"dpa4"`` model config reaches this trunk directly via
``model.call_common(coord, atype, box, spin=...)``.
"""

import copy
import os

import numpy as np
import pytest
import torch

from deepmd.dpmodel.train import (
    DEFAULT_TASK_KEY,
)
from deepmd.pt.model.model import get_model as pt_get_model
from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.pt_expt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt_expt.fitting.dpa4_ener import (
    SeZMEnergyFittingNet,
)
from deepmd.pt_expt.model import (
    EnergyModel,
)
from deepmd.pt_expt.model.dpa4_native_spin_model import (
    DPA4NativeSpinModel,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.utils import env as _env
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

from ...dpa4_fixtures import (
    jitter_zero_arrays,
)
from ...seed import (
    GLOBAL_SEED,
)

# Small fp64 DPA4/SeZM config with native spin enabled on the descriptor
# (``use_spin=[True, False]``: type 0 ("foo") carries spin, type 1 ("bar")
# does not). No top-level "spin" key -> ``get_sezm_model`` builds the plain
# backbone ``EnergyModel``, not the ``DPA4NativeSpinModel`` wrapper.
_DPA4_SPIN_CONFIG = {
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
        "use_spin": [True, False],
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [16],
        "precision": "float64",
        "seed": 1,
    },
}


def _build_jittered_backbone(seed: int = 99) -> EnergyModel:
    """Build the pt_expt DPA4 backbone with ``use_spin`` set, jittered.

    DPA4 deliberately zero-initializes several residual output projections
    (see ``dpa4_fixtures.jitter_zero_arrays``), so a freshly constructed,
    untrained descriptor is architecturally edge/message (and spin)
    INDEPENDENT -- a bare model would make both the finite-difference and
    neutrality checks below vacuous. Jittering the descriptor's zero-init
    weight tree makes the energy genuinely depend on ``spin`` (verified
    in-test by ``TestGraphForceMag.setup_method``'s anti-vacuity guard).
    """
    model = get_model(_DPA4_SPIN_CONFIG)
    ds = model.atomic_model.descriptor
    data = ds.serialize()
    jitter_zero_arrays(data, np.random.default_rng(seed))
    jittered = DescrptDPA4.deserialize(data).to(_env.DEVICE)
    model.atomic_model.descriptor = jittered
    return model.to(_env.DEVICE).eval()


def _finite_diff_mag(model_fn, spin: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central finite difference of ``model_fn`` (scalar) w.r.t. every spin
    component, with the ``force_mag = -dE/dspin`` sign convention baked in.
    """
    fm = np.zeros_like(spin)
    for i in np.ndindex(*spin.shape):
        sp = spin.copy()
        sp[i] += eps
        ep = model_fn(sp)
        sp = spin.copy()
        sp[i] -= eps
        em = model_fn(sp)
        fm[i] = -(ep - em) / (2 * eps)
    return fm


class TestGraphForceMag:
    def setup_method(self) -> None:
        self.device = _env.DEVICE
        self.model = _build_jittered_backbone()

        generator = torch.Generator(device=self.device).manual_seed(GLOBAL_SEED)
        natoms = 6
        cell = torch.rand(
            [3, 3], dtype=torch.float64, device=self.device, generator=generator
        )
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device=self.device)
        coord = torch.rand(
            [natoms, 3], dtype=torch.float64, device=self.device, generator=generator
        )
        coord = torch.matmul(coord, cell)
        self.coord = coord.unsqueeze(0)  # (1, natoms, 3)
        self.atype = torch.tensor(
            [[0, 0, 0, 1, 1, 1]], dtype=torch.int64, device=self.device
        )
        self.box = cell.reshape(1, 9)
        spin = torch.rand(
            [1, natoms, 3], dtype=torch.float64, device=self.device, generator=generator
        )
        # only type 0 ("foo", use_spin=True) carries a magnetic moment; the
        # non-magnetic type's spin input is inert (mirrors DPA4NativeSpinModel's
        # mask_mag convention, dpmodel test_dpa4_native_spin_model.py).
        self.spin = spin * (self.atype == 0)[..., None].to(spin.dtype)

        # Anti-vacuity guard: with the jitter applied, the energy must
        # actually depend on spin (else the FD test below would trivially
        # pass with both sides at zero).
        out0 = self.model.call_common(self.coord, self.atype, self.box, spin=self.spin)
        out1 = self.model.call_common(
            self.coord, self.atype, self.box, spin=2.0 * self.spin
        )
        e_diff = (out1["energy_redu"] - out0["energy_redu"]).abs().max().item()
        assert e_diff > 1e-6, (
            f"expected the jittered model's energy to depend on spin; got a "
            f"change of only {e_diff:.3e} (jitter not effective -- the FD "
            f"test below would be vacuous)"
        )

    def test_force_mag_matches_finite_difference(self) -> None:
        """``energy_derv_r_mag`` from the graph autograd == -dE/dspin by
        central finite difference (atol 1e-6).
        """
        out = self.model.call_common(
            self.coord,
            self.atype,
            self.box,
            spin=self.spin,
            do_atomic_virial=False,
        )
        fm = out["energy_derv_r_mag"]

        def _energy(sp: np.ndarray) -> float:
            sp_t = torch.as_tensor(sp, device=self.device, dtype=self.coord.dtype)
            ret = self.model.call_common(self.coord, self.atype, self.box, spin=sp_t)
            return float(ret["energy_redu"].sum().detach())

        fd = _finite_diff_mag(_energy, self.spin.cpu().numpy())
        np.testing.assert_allclose(
            fm.squeeze(-2).detach().cpu().numpy().reshape(fd.shape),
            fd,
            atol=1e-6,
        )

    def test_force_unchanged_by_spin_leaf_wiring(self) -> None:
        """``call_common`` WITHOUT ``spin`` has no ``energy_derv_r_mag`` key,
        and the spin-less forward is deterministic (the new ``spin is not
        None`` branch is a true no-op when ``spin`` is not supplied).
        """
        out0 = self.model.call_common(self.coord, self.atype, self.box)
        assert "energy_derv_r_mag" not in out0
        out1 = self.model.call_common(self.coord, self.atype, self.box)
        assert "energy_derv_r_mag" not in out1
        # The graph-route reduction (segment_sum) and force assembly
        # (edge_force_virial) scatter through ``index_add``, whose atomicAdd is
        # non-deterministic on CUDA (1-2 fp64 ULP run-to-run); on CPU it is
        # exact. So "the spin-less branch is a no-op" is a bit-exact claim on
        # CPU and an eval-determinism claim (~1e-10) on CUDA. See CLAUDE.md.
        det_rtol, det_atol = (0.0, 0.0) if self.device.type == "cpu" else (1e-10, 1e-12)
        torch.testing.assert_close(
            out0["energy_redu"], out1["energy_redu"], rtol=det_rtol, atol=det_atol
        )
        torch.testing.assert_close(
            out0["energy_derv_r"], out1["energy_derv_r"], rtol=det_rtol, atol=det_atol
        )

    def test_dense_route_spin_raises(self) -> None:
        """Model-level spin rides ONLY the NeighborGraph lower (mirrors
        ``test_dpa4_native_spin_model.py::test_dense_route_spin_raises``).
        """
        with pytest.raises(NotImplementedError, match="NeighborGraph"):
            self.model.call_common(
                self.coord,
                self.atype,
                self.box,
                spin=self.spin,
                neighbor_graph_method="legacy",
            )


# =============================================================================
# Task 4: the ``DPA4NativeSpinModel`` wrapper (top-level "spin" key, scheme
# "native") -- public forward() keys/shapes and weight-copied parity vs the
# pt ``SeZMNativeSpinModel`` reference.
# =============================================================================

# Same shape as dpmodel's NATIVE_SPIN_CONFIG
# (source/tests/common/dpmodel/test_dpa4_native_spin_model.py), plus the
# top-level "type": "dpa4" that pt's ``get_model`` requires to dispatch into
# ``get_sezm_spin_model`` (dpmodel's dispatch keys off `data["spin"]["scheme"]`
# alone and does not need it).
NATIVE_SPIN_CONFIG = {
    "type": "dpa4",
    "type_map": ["Ni", "O"],
    "descriptor": {
        "type": "dpa4",
        "rcut": 4.0,
        "sel": 8,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "precision": "float64",
        "seed": 7,
        "random_gamma": False,
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [8, 8],
        "precision": "float64",
        "seed": 7,
    },
    "spin": {"use_spin": [True, False], "scheme": "native"},
}


def _jittered_wrapper(seed: int = 11) -> DPA4NativeSpinModel:
    """Build the pt_expt ``DPA4NativeSpinModel`` wrapper, jittered.

    Mirrors ``_build_jittered_backbone`` above and
    ``test_dpa4_native_spin_model.py::_jittered_model``: DPA4 zero-initializes
    residual projections, so jittering the descriptor's serialized weight
    tree is needed to make the wrapper's ``force_mag`` genuinely non-trivial.
    """
    model = get_model(NATIVE_SPIN_CONFIG)
    ds = model.backbone_model.atomic_model.descriptor
    data = ds.serialize()
    jitter_zero_arrays(data, np.random.default_rng(seed))
    model.backbone_model.atomic_model.descriptor = DescrptDPA4.deserialize(data).to(
        _env.DEVICE
    )
    return model.to(_env.DEVICE).eval()


class TestDPA4NativeSpinModelPtExpt:
    """Public ``forward()`` contract of the pt_expt ``DPA4NativeSpinModel``."""

    def setup_method(self) -> None:
        self.device = _env.DEVICE
        self.model = _jittered_wrapper(seed=11)

        generator = torch.Generator(device=self.device).manual_seed(GLOBAL_SEED)
        self.nf, self.nloc = 1, 6
        cell = torch.rand(
            [3, 3], dtype=torch.float64, device=self.device, generator=generator
        )
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device=self.device)
        coord = torch.rand(
            [self.nloc, 3], dtype=torch.float64, device=self.device, generator=generator
        )
        coord = torch.matmul(coord, cell)
        self.coord = coord.unsqueeze(0)
        # use_spin=[True, False]: type 0 ("Ni") carries spin, type 1 ("O") does not.
        self.atype = torch.tensor(
            [[0, 0, 0, 1, 1, 1]], dtype=torch.int64, device=self.device
        )
        self.box = cell.reshape(1, 9)
        # Deliberately NOT pre-masked by type: the model's own mask_mag/gating
        # must zero the non-spin rows internally (see test below).
        self.spin = torch.rand(
            [self.nf, self.nloc, 3],
            dtype=torch.float64,
            device=self.device,
            generator=generator,
        )

    def test_forward_keys_and_mask(self) -> None:
        out = self.model.forward(self.coord, self.atype, self.spin, box=self.box)
        for k in ("energy", "atom_energy", "force", "force_mag", "virial", "mask_mag"):
            assert k in out
        assert out["force_mag"].shape == (self.nf, self.nloc, 3)
        assert out["force"].shape == (self.nf, self.nloc, 3)
        assert out["mask_mag"].shape == (self.nf, self.nloc, 1)
        # mask_mag: True only for the spin-active type (0)
        expect_mask = (self.atype == 0).unsqueeze(-1)
        torch.testing.assert_close(out["mask_mag"], expect_mask, rtol=0, atol=0)

    def test_force_mag_zero_on_non_spin_types(self) -> None:
        """Non-spin-type (``atype==1``) rows of ``force_mag`` are exactly
        zero, even though ``self.spin`` feeds nonzero noise there -- the
        descriptor gates the spin embedding by type (docstring in
        ``dpa4_native_spin_model.py``), so no re-masking is applied.
        """
        out = self.model.forward(self.coord, self.atype, self.spin, box=self.box)
        non_spin = self.atype == 1
        assert torch.all(out["force_mag"][non_spin] == 0)
        # anti-vacuity: the spin-active rows must be genuinely nonzero
        spin_active = self.atype == 0
        assert out["force_mag"][spin_active].abs().max().item() > 1e-6

    def test_atomic_virial_key_present_when_requested(self) -> None:
        out = self.model.forward(
            self.coord, self.atype, self.spin, box=self.box, do_atomic_virial=True
        )
        assert "atom_virial" in out
        assert out["atom_virial"].shape == (self.nf, self.nloc, 9)


def _pt_native_spin_model(seed: int = 3):
    """Build the pt (reference) ``SeZMNativeSpinModel``, jittered."""
    model = pt_get_model(copy.deepcopy(NATIVE_SPIN_CONFIG)).to(torch.float64)
    ds = model.atomic_model.descriptor
    data = ds.serialize()
    jitter_zero_arrays(data, np.random.default_rng(seed))
    from deepmd.pt.model.descriptor.sezm import (
        DescrptSeZM,
    )

    model.atomic_model.descriptor = DescrptSeZM.deserialize(data).to(torch.float64)
    return model.eval()


class TestDPA4NativeSpinModelParity:
    """Weight-copied fp64 parity vs ``deepmd.pt``'s ``SeZMNativeSpinModel``.

    Mirrors the component-weight-copy mechanism used throughout
    ``test_dpa4_dpmodel_parity.py`` (build the pt reference, ``.serialize()``
    its descriptor/fitting, ``.deserialize()`` the result into the pt_expt
    counterpart), lifted to the full model level: pt's ``DescrptSeZM``/
    ``SeZMEnergyFittingNet`` serialize to the SAME backend-agnostic dpmodel
    dict schema pt_expt's ``DescrptDPA4``/``SeZMEnergyFittingNet``
    deserialize -- both are ports of one canonical dpmodel architecture.
    Both models run on CPU fp64 (project convention: same-math
    weight-copied fp64 parity ~= rtol/atol 1e-12).
    """

    def setup_method(self) -> None:
        from deepmd.pt.model.task.sezm_ener import (
            SeZMEnergyFittingNet as PtSeZMEnergyFittingNet,
        )

        cpu = torch.device("cpu")

        # --- pt reference (jittered) ---
        self.pt_model = _pt_native_spin_model(seed=3).to(cpu)

        # --- pt_expt model: same architecture, weights copied from pt ---
        pt_expt_model = get_model(NATIVE_SPIN_CONFIG)
        atomic = pt_expt_model.backbone_model.atomic_model
        atomic.descriptor = DescrptDPA4.deserialize(
            self.pt_model.atomic_model.descriptor.serialize()
        )
        atomic.fitting_net = SeZMEnergyFittingNet.deserialize(
            self.pt_model.atomic_model.fitting_net.serialize()
        )
        self.pt_expt_model = pt_expt_model.to(cpu).eval()
        assert isinstance(
            self.pt_model.atomic_model.fitting_net, PtSeZMEnergyFittingNet
        )

        generator = torch.Generator(device=cpu).manual_seed(GLOBAL_SEED + 1)
        self.nf, self.nloc = 1, 6
        cell = torch.rand([3, 3], dtype=torch.float64, generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3)
        coord = torch.rand([self.nloc, 3], dtype=torch.float64, generator=generator)
        coord = torch.matmul(coord, cell)
        self.coord = coord.unsqueeze(0)
        self.atype = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.int64)
        self.box = cell.reshape(1, 9)
        self.spin = torch.rand(
            [self.nf, self.nloc, 3], dtype=torch.float64, generator=generator
        )

    def test_parity_vs_pt_native_spin_model(self) -> None:
        out_pt = self.pt_model.forward(self.coord, self.atype, self.spin, box=self.box)
        out_pte = self.pt_expt_model.forward(
            self.coord, self.atype, self.spin, box=self.box
        )

        # Anti-vacuity: fresh DPA4 zero-initializes residual projections, so
        # a bare model would make force_mag identically zero on both sides
        # and the parity check below vacuous by construction. Guard that the
        # jitter made the magnetic force genuinely non-trivial.
        fm_max = out_pte["force_mag"].abs().max().item()
        assert fm_max > 1e-6, (
            f"expected the jittered model's force_mag to be non-trivial; "
            f"got max |force_mag| = {fm_max:.3e} (jitter not effective -- "
            f"the parity check below would be vacuous)"
        )

        for key in ("energy", "force", "force_mag", "virial"):
            torch.testing.assert_close(
                out_pt[key], out_pte[key], rtol=1e-12, atol=1e-12, msg=key
            )
        torch.testing.assert_close(
            out_pt["mask_mag"], out_pte["mask_mag"], rtol=0, atol=0, msg="mask_mag"
        )


# =============================================================================
# Task 5: ``forward_lower_graph_exportable`` -- the graph-spin ``.pt2``
# exportable ABI (spin at positional index 10). ``make_fx``/``torch.export``
# tracing is CPU-only by design (``serialization.py:924`` moves the model
# to CPU before tracing) -- both tests below build the model AND the sample
# inputs on CPU explicitly, mirroring ``test_dpa4_graph_lower.py``'s
# ``test_graph_lower_symbolic_trace``/``test_graph_lower_torch_export``.
# =============================================================================


def _build_native_spin_model_cpu(seed: int = 21) -> DPA4NativeSpinModel:
    """Build the jittered pt_expt ``DPA4NativeSpinModel`` wrapper on CPU.

    Mirrors ``_jittered_wrapper`` above but pinned to CPU from construction
    (export tracing is CPU-only; the dpa1 CUDA lesson is that traced inputs
    and params must share a device). DPA4 zero-initializes residual
    projections, so jittering is needed to make ``force_mag`` genuinely
    non-trivial -- otherwise the trace-vs-eager comparisons below would be
    vacuous (both sides identically zero).
    """
    cpu = torch.device("cpu")
    model = get_model(NATIVE_SPIN_CONFIG)
    ds = model.backbone_model.atomic_model.descriptor
    data = ds.serialize()
    jitter_zero_arrays(data, np.random.default_rng(seed))
    model.backbone_model.atomic_model.descriptor = DescrptDPA4.deserialize(data).to(cpu)
    return model.to(cpu).eval()


def _build_spin_graph_sample(
    model: DPA4NativeSpinModel,
    *,
    nframes: int = 2,
    nloc: int = 7,
    e_max: int | None = 175,
) -> tuple[torch.Tensor, ...]:
    """Build a small CPU sample matching the graph-spin positional ABI.

    Reuses :func:`build_synthetic_graph_inputs` (canonicalized,
    destination-major) for the shared ``NeighborGraph`` CSR block, then adds
    a ``spin`` tensor at index 10 -- a small NON-ZERO sample (not
    ``torch.zeros``: an all-zero spin leaf can hit degenerate branches in
    the equivariant spin embedding) matching the flat node axis ``N`` that
    ``atype`` shares.

    Returns
    -------
    tuple
        ``(atype, n_node, n_local, edge_index, edge_vec, edge_mask,
        destination_order, destination_row_ptr, source_order,
        source_row_ptr, spin, fparam, aparam)`` -- the exact positional
        order of ``DPA4NativeSpinModel.forward_lower_graph_exportable``.
    """
    from deepmd.pt_expt.utils.serialization import (
        build_synthetic_graph_inputs,
    )

    sample = build_synthetic_graph_inputs(
        model.backbone_model,
        e_max=e_max,
        nframes=nframes,
        nloc=nloc,
        dtype=torch.float64,
        device=torch.device("cpu"),
        want_charge_spin=False,
    )
    (atype, n_node, n_local, ei, ev, em, do, drp, so, srp, fp, ap, _cs) = sample
    generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
    spin = 0.1 + torch.rand(atype.shape[0], 3, dtype=torch.float64, generator=generator)
    return atype, n_node, n_local, ei, ev, em, do, drp, so, srp, spin, fp, ap


class TestDPA4NativeSpinGraphLowerExportable:
    """``forward_lower_graph_exportable`` establishes the positional ``.pt2``
    ABI for graph-spin models (spin at index 10, no ``charge_spin`` slot, no
    with-comm variant) -- Tasks 6/7/9 mirror this ABI.
    """

    def test_graph_lower_exportable_symbolic_trace(self) -> None:
        """``make_fx`` traces the graph-spin closure on CPU; the traced
        output matches the eager ``forward_common_lower_graph`` reference
        (same graph inputs -- the same physical system) bit-tight, and
        includes a genuinely non-trivial ``force_mag``.
        """
        model = _build_native_spin_model_cpu()
        sample = _build_spin_graph_sample(model)
        atype, n_node, n_local, ei, ev, em, do, drp, so, srp, spin, fp, ap = sample

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
            spin,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=True,
            destination_sorted=True,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )
        out = traced(atype, n_node, n_local, ei, ev, em, do, drp, so, srp, spin, fp, ap)
        for key in (
            "atom_energy",
            "energy",
            "force",
            "force_mag",
            "virial",
            "atom_virial",
        ):
            assert key in out, f"missing output key {key}"
            assert torch.isfinite(out[key]).all(), f"non-finite traced {key}"

        # Anti-vacuity: the jittered model's force_mag must be non-trivial,
        # else the parity check below would trivially pass with both sides
        # at zero.
        assert out["force_mag"].abs().max().item() > 1e-6

        ref = model.backbone_model.forward_common_lower_graph(
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
            do_atomic_virial=True,
            fparam=fp,
            aparam=ap,
            spin=spin,
        )
        tol = {"rtol": 1e-12, "atol": 1e-12}
        torch.testing.assert_close(out["atom_energy"], ref["energy"], **tol)
        torch.testing.assert_close(out["energy"], ref["energy_redu"], **tol)
        torch.testing.assert_close(
            out["force"], ref["energy_derv_r"].reshape(out["force"].shape), **tol
        )
        torch.testing.assert_close(
            out["force_mag"],
            ref["energy_derv_r_mag"].reshape(out["force_mag"].shape),
            **tol,
        )
        torch.testing.assert_close(
            out["virial"], ref["energy_derv_c_redu"].reshape(out["virial"].shape), **tol
        )
        torch.testing.assert_close(
            out["atom_virial"],
            ref["energy_derv_c"].reshape(out["atom_virial"].shape),
            **tol,
        )

    def test_graph_lower_exportable_torch_export(self) -> None:
        """``torch.export.export`` succeeds with a dynamic ``nedge`` (and
        ``N``/``nframes``) axis; the exported program reproduces the eager
        graph lower AND generalizes to a different (smaller) system size
        than the one it was traced/exported on.
        """
        model = _build_native_spin_model_cpu()
        sample = _build_spin_graph_sample(model)
        atype, n_node, n_local, ei, ev, em, do, drp, so, srp, spin, fp, ap = sample

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
            spin,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=True,
            destination_sorted=True,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )

        nframes_dim = torch.export.Dim("nframes", min=1)
        n_node_total_dim = torch.export.Dim("n_node_total", min=1)
        nedge_dim = torch.export.Dim("nedge", min=2)
        dynamic_shapes = (
            {0: n_node_total_dim},  # atype
            {0: nframes_dim},  # n_node
            {0: nframes_dim},  # n_local
            {1: nedge_dim},  # edge_index
            {0: nedge_dim},  # edge_vec
            {0: nedge_dim},  # edge_mask
            {0: nedge_dim},  # destination_order
            {0: n_node_total_dim + 1},  # destination_row_ptr
            {0: nedge_dim},  # source_order
            {0: n_node_total_dim + 1},  # source_row_ptr
            {0: n_node_total_dim},  # spin -- shares atype's N axis
            {0: nframes_dim} if fp is not None else None,  # fparam
            {0: n_node_total_dim} if ap is not None else None,  # aparam
        )
        exported = torch.export.export(
            traced,
            (atype, n_node, n_local, ei, ev, em, do, drp, so, srp, spin, fp, ap),
            dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
        loaded = exported.module()

        # Re-run on a SMALLER system to prove the exported program is
        # genuinely dynamic, not specialized to the trace-time shapes.
        small = _build_spin_graph_sample(model, nframes=1, nloc=3, e_max=None)
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
            s_spin,
            s_fp,
            s_ap,
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
            s_spin,
            s_fp,
            s_ap,
        )
        ref = model.backbone_model.forward_common_lower_graph(
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
            do_atomic_virial=True,
            fparam=s_fp,
            aparam=s_ap,
            spin=s_spin,
        )
        for key in ("energy", "force", "force_mag", "virial", "atom_virial"):
            assert torch.isfinite(out[key]).all(), f"non-finite exported {key}"
        tol = {"rtol": 1e-10, "atol": 1e-10}
        torch.testing.assert_close(out["energy"], ref["energy_redu"], **tol)
        torch.testing.assert_close(
            out["force"], ref["energy_derv_r"].reshape(out["force"].shape), **tol
        )
        torch.testing.assert_close(
            out["force_mag"],
            ref["energy_derv_r_mag"].reshape(out["force_mag"].shape),
            **tol,
        )
        torch.testing.assert_close(
            out["virial"], ref["energy_derv_c_redu"].reshape(out["virial"].shape), **tol
        )
        torch.testing.assert_close(
            out["atom_virial"],
            ref["energy_derv_c"].reshape(out["atom_virial"].shape),
            **tol,
        )


# =============================================================================
# Task 11: training smoke -- native-spin DPA4 through the real pt_expt
# trainer (data loading, ``ener_spin`` loss dispatch, ``ModelWrapper``,
# graph-route autograd ``force_mag``), not a hand-rolled forward/backward.
# =============================================================================

# Small fp64 native-spin DPA4/SeZM training config. Reuses the real NiO spin
# dataset (``source/tests/pt/NiO/data``, type_map ["Ni", "O"], 32 atoms: 16
# Ni carrying a magnetic moment, 16 O not) that ``source/tests/pt/
# test_finetune_spin.py``'s ``TestSpinFinetuneSeA`` already exercises for the
# (unrelated) virtual-atom scheme with ``rcut=4.0``, ``sel=[20, 20]`` -- rcut
# reused verbatim here since it is known to see a sane number of neighbors on
# this system; DPA4's ``sel`` is only an initial search-capacity hint (grows
# on demand, never truncates the energy path), so a single scalar sel=40
# (>= the se_e2_a per-type total of 40) is a safe, generous starting point.
# The descriptor is intentionally tiny (channels=8, n_radial=4, lmax=1,
# mmax=1, n_blocks=1) to keep the smoke test fast: this test proves the
# training PLUMBING (data requirement, loss dispatch, wrapper spin
# threading, autograd gradient flow), not model quality.
_TRAIN_MODEL_CONFIG = {
    "type": "dpa4",
    "type_map": ["Ni", "O"],
    "descriptor": {
        "type": "dpa4",
        "rcut": 4.0,
        "sel": 40,
        "channels": 8,
        "n_radial": 4,
        "lmax": 1,
        "mmax": 1,
        "n_blocks": 1,
        "precision": "float64",
        "seed": 1,
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [8],
        "precision": "float64",
        "seed": 1,
    },
    "spin": {"use_spin": [True, False], "scheme": "native"},
}


def _make_train_config(data_dir: str, numb_steps: int = 2) -> dict:
    """Build a minimal native-spin DPA4 training config pointing at *data_dir*.

    Loss prefactors mirror ``source/tests/pt/test_finetune_spin.py``'s
    ``TestSpinFinetuneSeA.setUp`` (the reference pt spin-training config):
    ``ener_spin`` with both real-force (``fr``) and magnetic-force (``fm``)
    terms enabled, so a nonzero magnetic force error actually contributes to
    the loss (and therefore to the backward pass that reaches the
    spin-embedding parameters).
    """
    return {
        "model": copy.deepcopy(_TRAIN_MODEL_CONFIG),
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 3.51e-8,
        },
        "loss": {
            "type": "ener_spin",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_fr": 1000,
            "limit_pref_fr": 1,
            "start_pref_fm": 1000,
            "limit_pref_fm": 1,
        },
        "training": {
            "training_data": {"systems": [data_dir], "batch_size": 1},
            "validation_data": {
                "systems": [data_dir],
                "batch_size": 1,
                "numb_btch": 1,
            },
            "numb_steps": numb_steps,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 1,
            "save_freq": numb_steps,
        },
    }


class TestDPA4NativeSpinTrainingSmoke:
    """End-to-end training smoke for the native-spin DPA4 trainer.

    Exercises the FULL production training path -- data loading, the
    ``ener_spin`` loss dispatch, ``ModelWrapper.forward``, and the
    graph-route backbone's autograd ``force_mag`` -- on the real NiO spin
    dataset, not a hand-rolled forward/backward.

    Writing this test surfaced three real gaps (all in ``deepmd/dpmodel`` or
    ``deepmd/pt_expt``, none in the read-only ``deepmd/pt``), fixed in the
    same commit as this test:

    1. ``ModelWrapper.forward`` (``deepmd/pt_expt/train/wrapper.py``) had no
       ``spin`` parameter, so a spin-labeled batch's ``spin`` key made
       ``self.wrapper(**input_dict, ...)`` raise ``TypeError``. Fixed by
       mirroring ``deepmd.pt.train.wrapper.ModelWrapper.forward``'s
       ``has_spin``-gated threading.
    2. ``get_additional_data_requirement``
       (``deepmd/pt_expt/train/training.py``) never declared ``spin`` as a
       data requirement, so the data loader never learned to read
       ``spin.npy`` in the first place. Fixed by mirroring
       ``deepmd.pt.train.training.get_additional_data_requirement``'s
       ``has_spin`` branch.
    3. ``DPA4NativeSpinModel.forward`` (``deepmd/pt_expt/model/
       dpa4_native_spin_model.py``) accepted no ``charge_spin`` keyword, but
       ``ModelWrapper`` always forwards one -- fixed by accepting (and
       passing through) ``charge_spin``, mirroring pt's
       ``SeZMNativeSpinModel.forward``. Separately, ``EnergySpinLoss.call``
       (``deepmd/dpmodel/loss/ener_spin.py``) diffed the flat
       ``(nf, natoms * 3)`` data-loader label directly against the model's
       ``(nf, natoms, 3)`` prediction for ``force``/``force_mag`` -- unlike
       every sibling atomic-label loss (``dos.py``/``tensor.py``), which
       reshape the label to the canonical atomic shape before use. Fixed by
       adding the same reshape.
    """

    def setup_method(self) -> None:
        self.data_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "pt", "NiO", "data", "single"
        )
        if not os.path.isdir(self.data_dir):
            pytest.skip(f"NiO spin data not found: {self.data_dir}")

    def test_training_smoke(self, tmp_path) -> None:
        """Run 2 training steps; assert finite loss and a live spin-embedding grad."""
        config = _make_train_config(self.data_dir, numb_steps=2)
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            trainer = get_trainer(config)
            assert isinstance(
                trainer.wrapper.model[DEFAULT_TASK_KEY], DPA4NativeSpinModel
            )

            tasks = trainer._make_training_tasks()
            task = trainer.select_task(tasks)

            for step in range(2):
                result = trainer.train_step(task, step)
                loss = result.payload["loss"]
                assert torch.isfinite(loss).all(), f"non-finite loss at step {step}"

            # force_mag gradients flow into training: after the last step's
            # backward(), a spin-embedding parameter carries a nonzero grad.
            # ``optimizer.step()`` reads (but does not clear) ``.grad``, so
            # this checks the SAME gradient the optimizer just consumed --
            # ``train_step`` only calls ``zero_grad()`` at the START of the
            # NEXT call, not after ``optimizer.step()``.
            spin_params = [
                (name, p)
                for name, p in trainer.wrapper.named_parameters()
                if "spin_embedding" in name
            ]
            assert spin_params, "no spin_embedding parameter found in the model"
            nonzero = [
                name
                for name, p in spin_params
                if p.grad is not None and torch.any(p.grad != 0)
            ]
            assert nonzero, (
                "no spin_embedding parameter has a nonzero grad after "
                "training -- force_mag gradients are not flowing into "
                "training"
            )
        finally:
            os.chdir(old_cwd)
