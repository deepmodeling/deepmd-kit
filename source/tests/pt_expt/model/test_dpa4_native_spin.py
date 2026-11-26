# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 physical symmetries, native spin, serialization and training contracts.

The graph backbone and native wrapper are checked through their public
interfaces, including backend parity, export and magnetic-force training.
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
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.model.native_spin_model import (
    NativeSpinEnergyModel,
)
from deepmd.pt_expt.utils import env as _env
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

from ...dpa4_fixtures import (
    four_atom_spin_inputs,
    jitter_zero_arrays,
    tetrahedral_spin_configurations,
)
from ...seed import (
    GLOBAL_SEED,
)

# Small fp64 DPA4/SeZM config with native spin enabled on the descriptor
# (``use_spin=[True, False]``: type 0 ("foo") carries spin, type 1 ("bar")
# does not). No top-level "spin" key -> ``get_sezm_model`` builds the plain
# backbone ``EnergyModel``, not the ``NativeSpinEnergyModel`` wrapper.
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
    data = jitter_zero_arrays(data, np.random.default_rng(seed))
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
        # non-magnetic type's spin input is inert (mirrors NativeSpinEnergyModel's
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

    def test_hessian_uses_spin_conditioning(self) -> None:
        """Coordinate Hessians retain the spin input of the energy surface."""
        self.model.enable_hessian()
        hessian = self.model.call_common(
            self.coord.clone().requires_grad_(True),
            self.atype,
            self.box,
            spin=self.spin,
        )["energy_derv_r_derv_r"]
        scaled_hessian = self.model.call_common(
            self.coord.clone().requires_grad_(True),
            self.atype,
            self.box,
            spin=2.0 * self.spin,
        )["energy_derv_r_derv_r"]

        assert torch.max(torch.abs(scaled_hessian - hessian)).item() > 1e-6

    def test_force_unchanged_by_spin_leaf_wiring(self) -> None:
        """Calls without spin omit magnetic derivatives and remain deterministic."""
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
# Native wrapper outputs and weight-copied parity with the PyTorch model.
# =============================================================================

# The model-level type selects the SeZM factory in PyTorch; the portable
# wrapper uses the same descriptor and fitting configuration.

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


def _jittered_wrapper(seed: int = 11) -> NativeSpinEnergyModel:
    """Build the pt_expt ``NativeSpinEnergyModel`` wrapper, jittered.

    Mirrors ``_build_jittered_backbone`` above and
    ``test_dpa4_native_spin_model.py::_jittered_model``: DPA4 zero-initializes
    residual projections, so jittering the descriptor's serialized weight
    tree is needed to make the wrapper's ``force_mag`` genuinely non-trivial.
    """
    model = get_model(NATIVE_SPIN_CONFIG)
    ds = model.atomic_model.descriptor
    data = ds.serialize()
    data = jitter_zero_arrays(data, np.random.default_rng(seed))
    model.atomic_model.descriptor = DescrptDPA4.deserialize(data).to(_env.DEVICE)
    return model.to(_env.DEVICE).eval()


class TestNativeSpinEnergyModelPtExpt:
    """Public ``forward()`` contract of the pt_expt ``NativeSpinEnergyModel``."""

    def test_adam_route_patterns_match_parameters(self) -> None:
        """The DPA4 backbone and its native-spin model declare the same routed tensors."""
        expected = [
            "atomic_model.descriptor.env_seed_embedding.rbf_proj_layer1.",
            "atomic_model.descriptor.radial_embedding.net.0.",
        ]
        for model in (_build_jittered_backbone(), _jittered_wrapper()):
            patterns = sorted(model.adam_route_patterns())
            assert patterns == expected
            names = [name for name, _ in model.named_parameters()]
            for pattern in patterns:
                assert any(pattern in name for name in names), pattern

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

    def test_padding_is_not_magnetic_when_last_type_is(self) -> None:
        """A phantom type never aliases the last magnetic type."""
        config = copy.deepcopy(NATIVE_SPIN_CONFIG)
        config["spin"]["use_spin"] = [False, True]
        model = get_model(config).to(self.device)
        atype = torch.tensor([[0, -1]], dtype=torch.int64, device=self.device)

        torch.testing.assert_close(
            model._spin_active_mask(atype),
            torch.zeros((1, 2, 1), dtype=torch.bool, device=self.device),
        )

    def test_force_mag_zero_on_non_spin_types(self) -> None:
        """Non-spin-type (``atype==1``) rows of ``force_mag`` are exactly
        zero, even though ``self.spin`` feeds nonzero noise there -- the
        descriptor gates the spin embedding by type (docstring in
        ``native_spin_model.py``), so no re-masking is applied.
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

    def test_ragged_matches_rectangular_mixed_size(self) -> None:
        """Flat and padded node layouts preserve the same spin predictions."""
        coord = self.coord.repeat(2, 1, 1)
        atype = self.atype.repeat(2, 1)
        spin = self.spin.repeat(2, 1, 1)
        box = self.box.repeat(2, 1)
        atype[1, -2:] = -1
        coord[1, -2:] = 0.0
        spin[1, -2:] = 0.0
        real_atom = atype >= 0
        n_node = real_atom.sum(dim=1)

        rectangular = self.model(coord, atype, spin, box=box)
        ragged = self.model.forward_ragged(
            coord[real_atom],
            atype[real_atom],
            n_node,
            spin[real_atom],
            box=box,
        )

        torch.testing.assert_close(rectangular["mask"].bool(), real_atom)
        torch.testing.assert_close(
            ragged["mask"].bool(), torch.ones_like(ragged["mask"], dtype=torch.bool)
        )
        torch.testing.assert_close(ragged["n_node"], n_node)
        for key in ("energy", "virial"):
            torch.testing.assert_close(ragged[key], rectangular[key])
        for key in ("atom_energy", "force", "force_mag", "mask_mag"):
            torch.testing.assert_close(ragged[key], rectangular[key][real_atom])


def _pt_native_spin_model(seed: int = 3):
    """Build the pt (reference) ``SeZMNativeSpinModel``, jittered."""
    model = pt_get_model(copy.deepcopy(NATIVE_SPIN_CONFIG)).to(torch.float64)
    ds = model.atomic_model.descriptor
    data = ds.serialize()
    data = jitter_zero_arrays(data, np.random.default_rng(seed))
    from deepmd.pt.model.descriptor.sezm import (
        DescrptSeZM,
    )

    model.atomic_model.descriptor = DescrptSeZM.deserialize(data).to(torch.float64)
    return model.eval()


class TestNativeSpinEnergyModelParity:
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
        atomic = pt_expt_model.atomic_model
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
# Graph-lower export ABI: spin occupies positional input 10.
# CPU tracing keeps model parameters and inputs on the same device.
# =============================================================================


def _build_native_spin_model_cpu(seed: int = 21) -> NativeSpinEnergyModel:
    """Build a CPU model with nonzero residual and spin projections.

    Nonzero projections ensure that exported magnetic forces depend on spin.
    """
    cpu = torch.device("cpu")
    model = get_model(NATIVE_SPIN_CONFIG)
    ds = model.atomic_model.descriptor
    data = ds.serialize()
    data = jitter_zero_arrays(data, np.random.default_rng(seed))
    model.atomic_model.descriptor = DescrptDPA4.deserialize(data).to(cpu)
    return model.to(cpu).eval()


def _build_spin_graph_sample(
    model: NativeSpinEnergyModel,
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
        order of ``NativeSpinEnergyModel.forward_lower_graph_exportable``.
    """
    from deepmd.pt_expt.utils.serialization import (
        build_synthetic_graph_inputs,
    )

    sample = build_synthetic_graph_inputs(
        model,
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
    return atype, n_node, n_local, ei, ev, em, do, drp, so, srp, spin, fp, ap, _cs


class TestDPA4NativeSpinGraphLowerExportable:
    """``forward_lower_graph_exportable`` establishes the positional ``.pt2``
    ABI for graph-spin models (spin at index 10, conditional ``charge_spin``
    tail at slot 13, then the communicator and the border plan of a model
    with further message-passing blocks) -- the C++/serialization seams
    mirror this ABI.
    """

    def test_graph_lower_exportable_torch_export(self) -> None:
        """``torch.export.export`` succeeds with a dynamic ``nedge`` (and
        ``N``/``nframes``) axis; the exported program reproduces the eager
        graph lower AND generalizes to a different (smaller) system size
        than the one it was traced/exported on.
        """
        model = _build_native_spin_model_cpu()
        sample = _build_spin_graph_sample(model)
        atype, n_node, n_local, ei, ev, em, do, drp, so, srp, spin, fp, ap, cs = sample

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

        # Validate the symbolic trace before exporting the same graph. This
        # keeps same-shape parity and dynamic replay without tracing twice.
        out = traced(
            atype, n_node, n_local, ei, ev, em, do, drp, so, srp, spin, fp, ap, cs
        )
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
            {0: nframes_dim} if cs is not None else None,  # charge_spin
        )
        exported = torch.export.export(
            traced,
            (atype, n_node, n_local, ei, ev, em, do, drp, so, srp, spin, fp, ap, cs),
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
            s_spin,
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


class TestDPA4NativeSpinCompiledTraining:
    """Native-spin data flow through the compiled training lower."""

    def test_eager_eval_and_compiled_magnetic_force_backward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keep eval eager and compiled mixed-size magnetic gradients correct."""
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
            _get_model_structure_key,
        )

        # ``make_fx`` and the rebuilt graph exercise the compile boundary. The
        # identity backend keeps this regression independent of platform
        # toolchains while preserving the traced forward and double backward.
        monkeypatch.setattr(torch, "compile", lambda model, **_: model)

        model = _build_native_spin_model_cpu()
        compiled = _CompiledModel(
            model,
            _get_model_structure_key(model),
            compile_eval=False,
        )
        coord = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float64,
        ).repeat(2, 1, 1)
        atype = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.int64).repeat(2, 1)
        spin = (
            torch.arange(18, dtype=torch.float64).reshape(1, 6, 3) / 20.0 + 0.1
        ).repeat(2, 1, 1)
        box = (8.0 * torch.eye(3, dtype=torch.float64)).reshape(1, 9).repeat(2, 1)
        atype[1, -2:] = -1
        coord[1, -2:] = 0.0
        spin[1, -2:] = 0.0

        eager_result = compiled.eval()(coord, atype, box=box, spin=spin)
        real_atom = atype >= 0
        n_node = real_atom.sum(dim=1)
        eager_ragged = compiled.forward_ragged(
            coord[real_atom],
            atype[real_atom],
            n_node,
            box=box,
            spin=spin[real_atom],
        )
        assert compiled._compiled_lower_by_mode == {}
        assert eager_result["force_mag"].shape == (2, 6, 3)
        torch.testing.assert_close(eager_ragged["n_node"], n_node)

        result = compiled.train()(coord, atype, box=box, spin=spin)

        assert result["force"].shape == (2, 6, 3)
        assert result["force_mag"].shape == (2, 6, 3)
        assert result["mask_mag"].shape == (2, 6, 1)
        assert torch.all(result["force"][1, -2:] == 0.0)
        assert torch.all(result["force_mag"][1, -2:] == 0.0)
        assert torch.all(~result["mask_mag"][1, -2:])
        assert set(compiled._compiled_lower_by_mode) == {True}

        ragged = compiled.forward_ragged(
            coord[real_atom],
            atype[real_atom],
            n_node,
            box=box,
            spin=spin[real_atom],
        )
        torch.testing.assert_close(ragged["n_node"], n_node)
        for key in ("energy", "virial"):
            torch.testing.assert_close(ragged[key], result[key])
        for key in ("atom_energy", "force", "force_mag", "mask", "mask_mag"):
            torch.testing.assert_close(ragged[key], result[key][real_atom])
        ragged["force_mag"].square().sum().backward()

        spin_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if "spin_embedding" in name
        ]
        assert spin_parameters
        assert any(
            parameter.grad is not None and torch.any(parameter.grad != 0)
            for parameter in spin_parameters
        )


# =============================================================================
# Native-spin training through the production data, loss and model wrappers.
# =============================================================================

# The NiO fixture has 32 atoms: magnetic Ni and nonmagnetic O. The small
# float64 descriptor checks training behavior without fitting model accuracy.
# The scalar sel is an initial neighbor-search capacity; the graph retains
# every edge inside the cutoff.
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
    """Exercise native-spin training on the NiO dataset.

    The checks cover spin data requirements, the energy/spin loss, model
    wrapper arguments and magnetic-force parameter gradients.
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
                trainer.wrapper.model[DEFAULT_TASK_KEY], NativeSpinEnergyModel
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


class TestDPA4NativeSpinFinetuneFromSpinFreePretrain:
    """Fine-tune a native-spin DPA4 on top of a pretraining with no magnetic type.

    The production workflow is a large corpus that declares no magnetic species
    followed by a small magnetic one. Both stages are native-spin, so the
    parameter trees agree and transfer wholesale, while ``use_spin`` differs --
    and the per-type spin gate it produces must be rebuilt by the fine-tuned
    model rather than inherited, since the pretraining gate is all zero.
    Fine-tuning also recomputes the output bias from statistics, which evaluates
    the model and therefore requires the spin input to reach it.

    The assertion is the symptom a dead gate produces: a magnetic-force error of
    exactly zero, reported for every step of an otherwise healthy run.
    """

    def setup_method(self) -> None:
        self.data_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "pt", "NiO", "data", "single"
        )
        if not os.path.isdir(self.data_dir):
            pytest.skip(f"NiO spin data not found: {self.data_dir}")

    def _normalized(self, config: dict) -> dict:
        return normalize(update_deepmd_input(copy.deepcopy(config), warning=False))

    def test_magnetic_force_is_live_after_finetune(self, tmp_path) -> None:
        from deepmd.pt_expt.utils.finetune import (
            get_finetune_rules,
        )

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            # Phase 1: pretrain with no type declared magnetic, then checkpoint.
            pretrain_config = _make_train_config(self.data_dir, numb_steps=1)
            pretrain_config["model"]["spin"]["use_spin"] = [False, False]
            get_trainer(self._normalized(pretrain_config)).run()
            checkpoint = tmp_path / "model.ckpt.pt"
            assert checkpoint.is_file(), "pretraining wrote no checkpoint"
            # Phase 2: fine-tune with Ni magnetic, through the production rules.
            finetune_config = self._normalized(
                _make_train_config(self.data_dir, numb_steps=1)
            )
            finetune_config["model"], finetune_links = get_finetune_rules(
                str(checkpoint), finetune_config["model"]
            )
            trainer = get_trainer(
                finetune_config,
                finetune_model=str(checkpoint),
                finetune_links=finetune_links,
            )
            descriptor = trainer.models["Default"].atomic_model.descriptor
            assert torch.all(descriptor.spin_embedding.mag_layer2.w == 0.0)
            assert torch.all(descriptor.spin_embedding.pair_layer.w == 0.0)
            assert torch.all(descriptor.env_seed_embedding.spin_output_proj.w == 0.0)

            task = trainer.select_task(trainer._make_training_tasks())
            more_loss = trainer.train_step(task, 0).payload["more_loss"]
            assert "rmse_fm" in more_loss
            magnetic_error = float(torch.as_tensor(more_loss["rmse_fm"]).detach())
            assert magnetic_error > 0.0, (
                "the magnetic force error is exactly zero after fine-tuning: the "
                "per-type spin gate was inherited from the spin-free pretraining "
                "instead of being rebuilt from `use_spin`"
            )
        finally:
            os.chdir(old_cwd)


class TestDPA4DescriptorVersionPersistence:
    """pt_expt persists the descriptor version so migrations still fire.

    The backend rebuilds a module from its config before loading, so a
    version kept only as a python attribute would come back claiming the
    semantics of the running code and every checkpoint would silently skip
    ``_migrate_version``.
    """

    def _descriptor(self, use_spin: list[bool] | None = None) -> DescrptDPA4:
        config = copy.deepcopy(NATIVE_SPIN_CONFIG)
        if use_spin is not None:
            config["spin"]["use_spin"] = use_spin
        return get_model(config).atomic_model.descriptor

    def test_version_rides_the_state_dict(self) -> None:
        descriptor = self._descriptor()
        assert "version_tensor" in descriptor.state_dict()
        assert float(descriptor.version_tensor.item()) == DescrptDPA4.LATEST_VERSION


class TestNativeSpinConfigFormsPtExpt:
    """pt_expt twin of ``test_dpa4_native_spin_model.py::TestNativeSpinConfigForms``.

    ``spin.use_spin`` index/symbol forms are expanded against ``type_map``
    (``normalize_spin_use_spin``), and ``allow_missing_label`` is forwarded
    into the constructed :class:`Spin` -- observable through the trainer's
    spin data requirement.
    """

    @pytest.mark.parametrize(
        ("use_spin_form", "expected"),
        [
            (["Ni"], [True, False]),  # element-symbol form
            ([0], [True, False]),  # type-index form
            ([0, 1], [True, True]),  # multiple type indices
            ([True, False], [True, False]),  # canonical boolean passthrough
        ],
    )
    def test_use_spin_forms(self, use_spin_form, expected) -> None:
        config = copy.deepcopy(NATIVE_SPIN_CONFIG)
        config["spin"] = {"use_spin": use_spin_form, "scheme": "native"}
        model = get_model(config)
        assert model.spin.use_spin.tolist() == expected
        descriptor = model.atomic_model.descriptor
        assert [bool(flag) for flag in descriptor.use_spin] == expected

    def test_use_spin_unknown_symbol_raises(self) -> None:
        config = copy.deepcopy(NATIVE_SPIN_CONFIG)
        config["spin"] = {"use_spin": ["Fe"], "scheme": "native"}
        with pytest.raises(ValueError, match="absent from type_map"):
            get_model(config)

    @pytest.mark.parametrize(
        ("allow_missing", "expected_must"),
        [
            (True, False),  # relaxed: spin file optional, zero default
            (False, True),  # default: spin file mandatory
        ],
    )
    def test_allow_missing_label_data_requirement(
        self, allow_missing, expected_must
    ) -> None:
        from deepmd.pt_expt.train.training import (
            get_additional_data_requirement,
        )

        config = copy.deepcopy(NATIVE_SPIN_CONFIG)
        if allow_missing:
            config["spin"]["allow_missing_label"] = True
        model = get_model(config)
        assert model.spin.allow_missing_label is allow_missing
        reqs = get_additional_data_requirement(model)
        spin_req = next(rr for rr in reqs if rr.key == "spin")
        assert spin_req.must is expected_must
        assert spin_req.default == 0.0
        assert spin_req.source_policy == ("default" if allow_missing else "tracked")


class TestPublicBaseModelRoundTrip:
    """Review 3638137290: pt_expt must round-trip its own native-spin
    serialization through the PUBLIC BaseModel entry point, returning the
    pt_expt class (a torch.nn.Module with .to() and the graph-export
    machinery), with forward parity.

    Before the registry dispatch, ``BaseModel.deserialize`` entered a
    hard-coded dpmodel branch and returned the numpy class here -- no
    ``.to()`` (the concrete finetune failure: ``training.py`` calls
    ``BaseModel.deserialize(...).to(DEVICE)``), no torch export machinery.
    """

    def test_roundtrip_returns_pt_expt_module(self) -> None:
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )

        model = _jittered_wrapper(seed=11)
        m2 = BaseModel.deserialize(model.serialize())
        assert type(m2) is NativeSpinEnergyModel
        assert isinstance(m2, torch.nn.Module)
        m2 = m2.to(_env.DEVICE)  # the concrete finetune failure: .to(DEVICE)
        assert hasattr(m2, "forward_common_lower_graph_exportable")
        m2 = m2.eval()
        generator = torch.Generator(device=_env.DEVICE).manual_seed(GLOBAL_SEED)
        cell = torch.rand(
            [3, 3], dtype=torch.float64, device=_env.DEVICE, generator=generator
        )
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device=_env.DEVICE)
        coord = torch.matmul(
            torch.rand(
                [6, 3], dtype=torch.float64, device=_env.DEVICE, generator=generator
            ),
            cell,
        ).unsqueeze(0)
        atype = torch.tensor([[0, 0, 1, 0, 1, 1]], device=_env.DEVICE)
        spin = torch.rand(
            [1, 6, 3], dtype=torch.float64, device=_env.DEVICE, generator=generator
        )
        box = cell.unsqueeze(0)
        r1 = model(coord, atype, spin, box=box)
        r2 = m2(coord, atype, spin, box=box)
        for key in ("energy", "force", "force_mag"):
            torch.testing.assert_close(r1[key], r2[key], rtol=1e-12, atol=1e-12)


# =============================================================================
# Combined native spin + charge-spin FiLM (review 3638047227): pt does not
# reject this public configuration -- SeZMNativeSpinModel.forward accepts
# ``charge_spin`` alongside the native ``spin`` input.
# =============================================================================

COMBINED_CHG_SPIN_CONFIG = copy.deepcopy(NATIVE_SPIN_CONFIG)
COMBINED_CHG_SPIN_CONFIG["descriptor"]["add_chg_spin_ebd"] = True


class TestCombinedChargeSpinNativeSpin:
    """Combined config builds, conditions on BOTH inputs, and matches pt.

    Weight-copied fp64 parity mechanics identical to
    :class:`TestNativeSpinEnergyModelParity`. ``charge_spin`` is CATEGORICAL
    (``ChargeSpinEmbedding`` casts the frame ``(charge, spin)`` pair to
    int64 lookup indices), so the probe uses integer-valued changes.
    """

    def setup_method(self) -> None:
        cpu = torch.device("cpu")
        pt_model = pt_get_model(copy.deepcopy(COMBINED_CHG_SPIN_CONFIG)).to(
            torch.float64
        )
        ds = pt_model.atomic_model.descriptor
        data = jitter_zero_arrays(ds.serialize(), np.random.default_rng(3))
        from deepmd.pt.model.descriptor.sezm import (
            DescrptSeZM,
        )

        pt_model.atomic_model.descriptor = DescrptSeZM.deserialize(data).to(
            torch.float64
        )
        self.pt_model = pt_model.eval().to(cpu)

        pt_expt_model = get_model(COMBINED_CHG_SPIN_CONFIG)
        atomic = pt_expt_model.atomic_model
        atomic.descriptor = DescrptDPA4.deserialize(
            self.pt_model.atomic_model.descriptor.serialize()
        )
        from deepmd.pt_expt.fitting.dpa4_ener import SeZMEnergyFittingNet as _FT

        atomic.fitting_net = _FT.deserialize(
            self.pt_model.atomic_model.fitting_net.serialize()
        )
        self.pt_expt_model = pt_expt_model.to(cpu).eval()
        assert self.pt_expt_model.has_chg_spin_ebd()
        assert self.pt_expt_model.has_spin()

        generator = torch.Generator(device=cpu).manual_seed(GLOBAL_SEED + 1)
        self.nf, self.nloc = 1, 6
        cell = torch.rand([3, 3], dtype=torch.float64, generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3)
        coord = torch.rand([self.nloc, 3], dtype=torch.float64, generator=generator)
        self.coord = torch.matmul(coord, cell).unsqueeze(0)
        self.atype = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.int64)
        self.box = cell.reshape(1, 9)
        self.spin = torch.rand(
            [self.nf, self.nloc, 3], dtype=torch.float64, generator=generator
        )
        self.cs0 = torch.zeros([self.nf, 2], dtype=torch.float64)
        self.cs1 = torch.tensor([[1.0, 2.0]], dtype=torch.float64)

    def test_combined_parity_and_sensitivity(self) -> None:
        outs = {}
        for tag, cs in (("cs0", self.cs0), ("cs1", self.cs1)):
            out_pt = self.pt_model.forward(
                self.coord, self.atype, self.spin, box=self.box, charge_spin=cs
            )
            out_pte = self.pt_expt_model.forward(
                self.coord, self.atype, self.spin, box=self.box, charge_spin=cs
            )
            for key in ("energy", "force", "force_mag"):
                torch.testing.assert_close(
                    out_pt[key],
                    out_pte[key],
                    rtol=1e-12,
                    atol=1e-12,
                    msg=f"{key}@{tag}",
                )
            outs[tag] = out_pte
        # charge_spin conditions the energy (reviewer's probe: nonzero
        # response), and BOTH backends agree on the conditioned values above.
        de = (outs["cs1"]["energy"] - outs["cs0"]["energy"]).abs().max().item()
        assert de > 1e-10, f"charge_spin response vanished: {de:.3e}"
        # native spin still conditions the same combined model
        out_spin = self.pt_expt_model.forward(
            self.coord, self.atype, 2.0 * self.spin, box=self.box, charge_spin=self.cs0
        )
        ds_ = (out_spin["energy"] - outs["cs0"]["energy"]).abs().max().item()
        assert ds_ > 1e-10, f"spin response vanished in combined model: {ds_:.3e}"


class TestCombinedChargeSpinTrainingSmoke:
    """Trainer smoke for the COMBINED native-spin + charge-spin FiLM model.

    NiO spin data carries no ``charge_spin`` file, so the run exercises the
    ``default_chg_spin`` metadata path: ``get_additional_data_requirement``
    marks the ``charge_spin`` requirement optional with the descriptor's
    default, and the FiLM conditions every batch on it.
    """

    def setup_method(self) -> None:
        self.data_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "pt", "NiO", "data", "single"
        )
        if not os.path.isdir(self.data_dir):
            pytest.skip(f"NiO spin data not found: {self.data_dir}")

    def test_training_smoke_combined(self, tmp_path) -> None:
        config = _make_train_config(self.data_dir, numb_steps=2)
        config["model"]["descriptor"]["add_chg_spin_ebd"] = True
        config["model"]["descriptor"]["default_chg_spin"] = [0.0, 2.0]
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            trainer = get_trainer(config)
            model = trainer.wrapper.model[DEFAULT_TASK_KEY]
            assert isinstance(model, NativeSpinEnergyModel)
            assert model.has_chg_spin_ebd()
            assert model.get_default_chg_spin() is not None

            tasks = trainer._make_training_tasks()
            task = trainer.select_task(tasks)
            for step in range(2):
                result = trainer.train_step(task, step)
                loss = result.payload["loss"]
                assert torch.isfinite(loss).all(), f"non-finite loss at step {step}"
        finally:
            os.chdir(old_cwd)


class TestCombinedChargeSpinGraphExportable:
    """Combined native-spin + charge-spin FiLM through the graph exportable.

    ``charge_spin`` occupies the conditional slot-13 tail of the graph-spin
    ``.pt2`` ABI; the traced module must (a) match the eager
    ``forward_common_lower_graph`` with the SAME charge_spin, and (b)
    respond to an integer-valued charge_spin change (the FiLM lookup is
    categorical).
    """

    def test_combined_symbolic_trace_parity_and_sensitivity(self) -> None:
        cpu = torch.device("cpu")
        model = get_model(COMBINED_CHG_SPIN_CONFIG)
        ds = model.atomic_model.descriptor
        data = jitter_zero_arrays(ds.serialize(), np.random.default_rng(21))
        model.atomic_model.descriptor = DescrptDPA4.deserialize(data).to(cpu)
        model = model.to(cpu).eval()
        assert model.has_chg_spin_ebd()

        sample = _build_spin_graph_sample(model)
        atype, n_node, n_local, ei, ev, em, do, drp, so, srp, spin, fp, ap, _ = sample
        cs = torch.tensor([[1.0, 2.0]] * int(n_node.shape[0]), dtype=torch.float64)

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
            charge_spin=cs,
            destination_sorted=True,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )
        out = traced(
            atype, n_node, n_local, ei, ev, em, do, drp, so, srp, spin, fp, ap, cs
        )
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
            spin=spin,
            charge_spin=cs,
            destination_sorted=True,
        )
        torch.testing.assert_close(
            out["energy"], ref["energy_redu"], rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            out["force_mag"],
            ref["energy_derv_r_mag"].squeeze(-2),
            rtol=1e-12,
            atol=1e-12,
        )
        # charge_spin conditions the TRACED module (slot 13 is live, not a
        # baked constant): an integer-valued change moves the energy.
        cs0 = torch.zeros_like(cs)
        out0 = traced(
            atype, n_node, n_local, ei, ev, em, do, drp, so, srp, spin, fp, ap, cs0
        )
        de = (out["energy"] - out0["energy"]).abs().max().item()
        assert de > 1e-10, f"charge_spin slot appears baked/dead: {de:.3e}"


class TestNativeSpinModelPairExcludeContract:
    """Model-level ``pair_exclude_types`` on the native-spin graph route.

    Exclusion is a BUILD-time transform owned by the neighbor-graph
    construction (decision #18/A4): the exported lower consumes a pre-excluded
    ``edge_mask``, so every external feeder (Python ``DeepEval``, C++
    ``NativeSpinPTExpt::compute``) must fold it in from the
    ``pair_exclude_types`` metadata field.  These tests pin the two properties
    the C++ regression
    (``source/api_cc/tests/test_deepspin_dpa4_pairexcl_ptexpt.cc``) depends on:
    the field reaches the metadata, and nothing inside the artifact reproduces
    the mask.
    """

    @staticmethod
    def _generic_model(pair_exclude_types: list | None, jitter_seed: int = 11):
        """Build the pt_expt native-spin model the way the .pt2 fixture does.

        dpmodel builder -> serialize (+ inject the model-level exclusion) ->
        pt_expt deserialize, mirroring ``source/tests/infer/gen_dpa4_spin.py``.
        The generic (no ``type: dpa4``) config is what keeps
        ``descriptor.exclude_types`` empty; the DPA4 alias would mirror the
        pairs into it (see ``test_dpa4_alias_mirrors_into_descriptor``).
        """
        from deepmd.dpmodel.model.model import get_model as dp_get_model
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )

        config = copy.deepcopy(NATIVE_SPIN_CONFIG)
        config.pop("type")
        data = dp_get_model(config).serialize()
        # DPA4 zero-initializes its residual projections, so an unjittered
        # model is edge-independent and every exclusion is invisible.
        data = jitter_zero_arrays(data, np.random.default_rng(jitter_seed))
        if pair_exclude_types is not None:
            data["pair_exclude_types"] = copy.deepcopy(pair_exclude_types)
        return BaseModel.deserialize(data).to(_env.DEVICE).eval()

    def test_generic_builder_keeps_descriptor_exclusions_empty(self) -> None:
        """Negative contract: the mask must NOT be baked into the descriptor.

        A descriptor-level copy is compiled INTO the artifact and would make a
        dead external seam invisible -- exactly the failure mode this pins.
        """
        model = self._generic_model([[0, 1]])
        assert [list(p) for p in model.atomic_model.pair_exclude_types] == [[0, 1]]
        assert not (model.atomic_model.descriptor.exclude_types or [])

    def test_dpa4_alias_mirrors_into_descriptor(self) -> None:
        """Contrast: the ``type="dpa4"`` alias DOES mirror the pairs.

        Documented here so nobody rebuilds the C++ fixture from this config:
        with the mirror in place the exclusion is applied inside the compiled
        artifact and the external-seam regression becomes vacuous.
        """
        config = copy.deepcopy(NATIVE_SPIN_CONFIG)  # keeps type="dpa4"
        config["pair_exclude_types"] = [[0, 1]]
        model = get_model(config)
        assert [list(p) for p in model.atomic_model.descriptor.exclude_types] == [
            [0, 1]
        ]

    def test_metadata_carries_pair_exclude_types(self) -> None:
        """The graph-lower metadata is what external feeders rebuild from."""
        from deepmd.pt_expt.utils.serialization import (
            _collect_metadata,
        )

        meta = _collect_metadata(
            self._generic_model([[0, 1]]), spin_scheme="native", lower_kind="graph"
        )
        assert meta["pair_exclude_types"] == [[0, 1]]

        base_meta = _collect_metadata(
            self._generic_model(None), spin_scheme="native", lower_kind="graph"
        )
        assert base_meta["pair_exclude_types"] == []

    def test_exclusion_changes_the_prediction(self) -> None:
        """Anti-vacuity: with identical weights, the mask must move the energy."""
        excluded = self._generic_model([[0, 1]])
        baseline = self._generic_model(None)

        generator = torch.Generator(device=_env.DEVICE).manual_seed(GLOBAL_SEED)
        cell = torch.eye(3, dtype=torch.float64, device=_env.DEVICE) * 6.0
        coord = torch.rand(
            [6, 3], dtype=torch.float64, device=_env.DEVICE, generator=generator
        )
        coord = torch.matmul(coord, cell).unsqueeze(0)
        atype = torch.tensor(
            [[0, 0, 0, 1, 1, 1]], dtype=torch.int64, device=_env.DEVICE
        )
        spin = 0.1 * torch.rand(
            [1, 6, 3], dtype=torch.float64, device=_env.DEVICE, generator=generator
        )
        box = cell.reshape(1, 9)

        e_excl = excluded(coord, atype, spin, box=box)["energy"]
        e_base = baseline(coord, atype, spin, box=box)["energy"]
        de = (e_excl - e_base).abs().max().item()
        assert de > 1e-6, (
            f"pair_exclude_types=[[0, 1]] left the energy unchanged (diff="
            f"{de:.3e}); the exclusion never reached the neighbor-graph build"
        )


@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
@pytest.mark.parametrize("spin_mode", ["exchange", "orbit"])
def test_native_spin_resolves_zero_total_neighbor_correlations(
    backend: str, spin_mode: str
) -> None:
    """Equal first spin moments retain distinct second-order correlations."""
    config = copy.deepcopy(NATIVE_SPIN_CONFIG)
    config["spin"]["scheme"] = f"native:{spin_mode}"
    model = (pt_get_model if backend == "pt" else get_model)(config)
    model = model.to(device="cpu", dtype=torch.float64).eval()
    generator = torch.Generator().manual_seed(37)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                0.1
                * torch.randn(
                    parameter.shape, generator=generator, dtype=parameter.dtype
                )
            )
    coord, atype, spin_a, spin_b = (
        torch.from_numpy(value) for value in tetrahedral_spin_configurations()
    )
    first = model(coord, atype, spin_a)
    second = model(coord, atype, spin_b)
    zero = model(coord, atype, torch.zeros_like(spin_a))
    assert (first["energy"] - zero["energy"]).abs().max() > 1e-8
    assert (first["energy"] - second["energy"]).abs().max() > 1e-8


@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
def test_native_spin_requires_environment_seed(backend: str) -> None:
    """The native model retains its neighbor spin-correlation path."""
    config = copy.deepcopy(NATIVE_SPIN_CONFIG)
    config["descriptor"]["use_env_seed"] = False
    with pytest.raises(ValueError, match=r"Native spin requires.*use_env_seed=True"):
        (pt_get_model if backend == "pt" else get_model)(config)


@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
@pytest.mark.parametrize("spin_mode", ["exchange", "orbit"])
@pytest.mark.parametrize(
    "variant",
    [
        "none",
        "edge",
        "node",
        "node_so3",
        "message_so3",
        "ffn_so3",
        "readout_glu",
        "readout_mlp",
    ],
)
def test_native_spin_physical_symmetries(
    backend: str, spin_mode: str, variant: str
) -> None:
    """Energy and its gradients obey time reversal and axial O(3) transformations."""
    config = copy.deepcopy(NATIVE_SPIN_CONFIG)
    config["spin"]["scheme"] = f"native:{spin_mode}"
    config["descriptor"].update(
        edge_cartesian=variant == "edge",
        node_cartesian="default:1" if variant == "node" else "none",
    )
    grid_option = {
        "node_so3": "node_wise_so3",
        "message_so3": "message_node_so3",
        "ffn_so3": "ffn_so3_grid",
    }.get(variant)
    if grid_option is not None:
        config["descriptor"][grid_option] = True
    if variant.startswith("readout_"):
        config["descriptor"].update(
            n_blocks=1,
            so3_readout=variant.removeprefix("readout_"),
            readout_layers=2,
        )
    model = (pt_get_model if backend == "pt" else get_model)(config)
    descriptor = model.atomic_model.descriptor
    state = jitter_zero_arrays(descriptor.serialize(), np.random.default_rng(517))
    model.atomic_model.descriptor = type(descriptor).deserialize(state)
    model = model.to(device="cpu", dtype=torch.float64).eval()
    coord, atype, spin, _ = (
        torch.from_numpy(value) for value in four_atom_spin_inputs()
    )
    original = model(coord, atype, spin)
    flipped = model(coord, atype, -spin)
    for key in ("energy", "force", "virial"):
        torch.testing.assert_close(flipped[key], original[key], atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        flipped["force_mag"], -original["force_mag"], atol=0.0, rtol=0.0
    )
    zero = model(coord, atype, torch.zeros_like(spin))
    torch.testing.assert_close(
        zero["force_mag"], torch.zeros_like(spin), atol=0.0, rtol=0.0
    )
    assert original["force_mag"].abs().max() > 1e-8
    assert (original["energy"] - zero["energy"]).abs().max() > 1e-8
    torch.testing.assert_close(
        original["force_mag"][:, 2],
        torch.zeros_like(spin[:, 2]),
        atol=0.0,
        rtol=0.0,
    )

    generator = torch.Generator().manual_seed(19)
    rotation, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )
    rotation = rotation * torch.linalg.det(rotation)
    for transform in (rotation, -rotation, -torch.eye(3, dtype=torch.float64)):
        axial = torch.linalg.det(transform) * transform
        transformed = model(coord @ transform.T, atype, spin @ axial.T)
        torch.testing.assert_close(
            transformed["energy"], original["energy"], atol=2e-10, rtol=2e-10
        )
        torch.testing.assert_close(
            transformed["force"],
            original["force"] @ transform.T,
            atol=2e-9,
            rtol=2e-9,
        )
        torch.testing.assert_close(
            transformed["force_mag"],
            original["force_mag"] @ axial.T,
            atol=2e-9,
            rtol=2e-9,
        )
    if spin_mode == "exchange":
        rotated = model(coord, atype, spin @ rotation.T)
        torch.testing.assert_close(
            rotated["energy"], original["energy"], atol=2e-12, rtol=2e-12
        )
        torch.testing.assert_close(
            rotated["force_mag"],
            original["force_mag"] @ rotation.T,
            atol=2e-11,
            rtol=2e-11,
        )
    else:
        rotated = model(coord, atype, spin @ rotation.T)
        assert (rotated["energy"] - original["energy"]).abs().max() > 1e-8


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"use_env_seed": False},
        {"s2_activation": [False, False], "glu_activation": False},
        {"s2_activation": [True, True], "lebedev_quadrature": False},
        {"s2_activation": [True, True], "grid_mlp": True},
        {"s2_activation": [True, True], "grid_branch": [2, 2, 2]},
        {"node_wise_s2": True, "message_node_s2": True},
        {"node_wise_so3": True, "grid_mlp": True},
        {"message_node_so3": True, "grid_branch": [0, 2, 0]},
        {"ffn_so3_grid": True, "kmax": 1, "grid_mlp": True},
        {"ffn_so3_grid": True, "kmax": 2, "grid_branch": [0, 0, 2]},
        {"so3_readout": "glu", "readout_layers": 2, "readout_norm": True},
        {"so3_readout": "mlp", "readout_layers": 2, "kmax": 2},
        {"edge_cartesian": True},
        {"node_cartesian": "default:2"},
        {
            "n_focus": 2,
            "atten_f_mix": True,
            "atten_v_proj": True,
            "atten_o_proj": True,
            "n_atten_head": 2,
        },
        {"radial_so2_mode": "degree", "radial_so2_rank": 0},
        {"radial_so2_mode": "degree_channel", "radial_so2_rank": 0},
        {"so2_attn_res": "dependent", "mixing_layers": 2},
        {
            "full_attn_res": "dependent",
            "extra_node_l": 1,
            "l_schedule": [2, 1],
            "so3_readout": "mlp",
        },
        {"extra_node_l": 1, "l_schedule": [1, 2], "so3_readout": "mlp"},
        {"block_attn_res": "independent", "l_schedule": [2, 1]},
        {"n_blocks": 0, "extra_node_l": 1, "so3_readout": "mlp"},
    ],
)
def test_o3_spin_free_physical_symmetries(options: dict) -> None:
    """Every descriptor branch preserves inversion for nonzero trained weights."""
    config = copy.deepcopy(NATIVE_SPIN_CONFIG)
    config.pop("spin")
    config["descriptor"].update(parity=True, channels=8, **options)
    model = get_model(config)
    model = model.to(device="cpu", dtype=torch.float64).eval()
    generator = torch.Generator().manual_seed(517)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(
                0.05
                * torch.randn(
                    parameter.shape, generator=generator, dtype=parameter.dtype
                )
            )
    coord, atype, _, _ = (torch.from_numpy(value) for value in four_atom_spin_inputs())
    original = model(coord, atype)
    inverted = model(-coord, atype)
    assert original["force"].abs().max() > 1e-8
    torch.testing.assert_close(
        inverted["energy"], original["energy"], atol=2e-10, rtol=2e-10
    )
    torch.testing.assert_close(
        inverted["force"], -original["force"], atol=2e-9, rtol=2e-9
    )
    torch.testing.assert_close(
        inverted["virial"], original["virial"], atol=2e-9, rtol=2e-9
    )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("bridging", ["none", "zbl"])
def test_o3_energy_force_with_bridging(native: bool, bridging: str) -> None:
    """Inversion keeps axial moments fixed with and without analytical bridging."""
    config = copy.deepcopy(NATIVE_SPIN_CONFIG)
    if not native:
        config.pop("spin")
    config["bridging_method"] = bridging
    config["descriptor"].update(
        parity=True,
        channels=8,
        l_schedule=[1, 2],
        ffn_so3_grid=True,
        so3_readout="mlp",
    )
    model = get_model(config).to(device="cpu", dtype=torch.float64).eval()
    generator = torch.Generator().manual_seed(541)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(
                0.05
                * torch.randn(
                    parameter.shape, generator=generator, dtype=parameter.dtype
                )
            )
    coord, atype, spin, _ = (
        torch.from_numpy(value) for value in four_atom_spin_inputs()
    )
    args = (atype, spin) if native else (atype,)
    original = model(coord, *args)
    inverted = model(-coord, *args)
    assert original["force"].abs().max() > 1e-8
    for key, sign in (("energy", 1), ("force", -1), ("virial", 1)):
        torch.testing.assert_close(
            inverted[key], sign * original[key], atol=2e-9, rtol=2e-9
        )
    if native:
        assert original["force_mag"].abs().max() > 1e-8
        torch.testing.assert_close(
            inverted["force_mag"], original["force_mag"], atol=2e-9, rtol=2e-9
        )
        reversed_spin = model(coord, atype, -spin)
        torch.testing.assert_close(
            reversed_spin["energy"], original["energy"], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            reversed_spin["force_mag"], -original["force_mag"], atol=0.0, rtol=0.0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.usefixtures("tf32_disabled", "calibrated_so2_device")
@pytest.mark.parametrize(
    "backend,native", [("pt_expt", False), ("pt_expt", True), ("pt", True)]
)
@pytest.mark.parametrize(
    "gate,binding,training",
    [
        ("DP_CUDA_INFER", "_cuda_conv_fn", False),
        ("DP_TRITON_INFER", "_triton_value_path", False),
        ("DP_CUDA_TRAIN", "_cuda_value_train", True),
        ("DP_TRITON_TRAIN", "_flash_atten_fn", True),
    ],
)
def test_o3_accelerated_model_matches_dense(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    native: bool,
    gate: str,
    binding: str,
    training: bool,
) -> None:
    """Fused value paths preserve parity and force-loss parameter gradients."""
    config = copy.deepcopy(NATIVE_SPIN_CONFIG)
    if not native:
        config.pop("spin")
    config["enable_tf32"] = False
    # The PyTorch backend preserves inversion through native spin alone and
    # takes a constant degree schedule and mixing depth.
    schedule = (
        {"l_schedule": [1, 2], "mixing_layers": [2, 3]}
        if backend == "pt_expt"
        else {"l_schedule": [2, 2], "mixing_layers": 3}
    )
    config["descriptor"].update(
        parity=True,
        channels=48,
        focus_dim=32,
        **schedule,
        precision="float32",
        use_amp=False,
        message_node_so3=True,
        ffn_so3_grid=True,
        grid_branch=[0, 0, 1],
        so3_readout="mlp",
    )
    config["fitting_net"]["precision"] = "float32"
    for name in (
        "DP_CUDA_INFER",
        "DP_TRITON_INFER",
        "DP_CUDA_TRAIN",
        "DP_TRITON_TRAIN",
    ):
        monkeypatch.setenv(name, "0")
    factory = pt_get_model if backend == "pt" else get_model
    dense = factory(config).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(541)
    with torch.no_grad():
        for parameter in dense.parameters():
            parameter.add_(
                0.05
                * torch.randn(
                    parameter.shape,
                    generator=generator,
                    device="cuda",
                    dtype=parameter.dtype,
                )
            )
    monkeypatch.setenv(gate, "1" if training else "2")
    fused = factory(config).to("cuda")
    fused.load_state_dict(dense.state_dict())
    assert all(
        getattr(block.so2_conv, binding) is not None
        for block in fused.atomic_model.descriptor.blocks
    )
    dense.train(training)
    fused.train(training)
    coord, atype, spin, _ = (
        torch.from_numpy(value).to("cuda") for value in four_atom_spin_inputs()
    )
    args = (atype, spin.float()) if native else (atype,)
    expected = dense(coord.float(), *args)
    actual = fused(coord.float(), *args)
    inverted = fused(-coord.float(), *args)
    for key, sign in (("energy", 1), ("force", -1), ("virial", 1)):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(
            inverted[key], sign * actual[key], atol=2e-5, rtol=2e-4
        )
    if native:
        assert expected["force_mag"].abs().max() > 1e-8
        torch.testing.assert_close(
            actual["force_mag"], expected["force_mag"], atol=2e-5, rtol=2e-4
        )
        torch.testing.assert_close(
            inverted["force_mag"], actual["force_mag"], atol=2e-5, rtol=2e-4
        )
    if training:
        from ..kernels.conditioning import (
            assert_conditioned,
            deviations,
        )

        monkeypatch.setenv(gate, "0")
        gold_config = copy.deepcopy(config)
        gold_config["descriptor"]["precision"] = "float64"
        gold_config["fitting_net"]["precision"] = "float64"
        gold = factory(gold_config).to("cuda").train()
        gold.load_state_dict(dense.state_dict())
        gold_args = (atype, spin) if native else (atype,)
        gold_out = gold(coord, *gold_args)
        (gold_out["energy"].sum() + gold_out["force"].square().sum()).backward()
        for out in (expected, actual):
            (out["energy"].sum() + out["force"].square().sum()).backward()
        truth, reference, result = [], [], []
        for (name, p), q, g in zip(
            dense.named_parameters(), fused.parameters(), gold.parameters(), strict=True
        ):
            if p.grad is None:
                assert q.grad is None and g.grad is None, name
            else:
                # Normalize each parameter tensor independently before the joint
                # model-level comparison, retaining small gradient tensors while
                # sharing the observed roundoff budget of the complete chain.
                scale = g.grad.abs().max().clamp_min(1.0)
                truth.append((g.grad / scale).flatten())
                reference.append((p.grad / scale).flatten())
                result.append((q.grad / scale).flatten())
        assert_conditioned(
            deviations(
                ["force-loss parameter gradients"],
                [torch.cat(truth)],
                [torch.cat(reference)],
                [torch.cat(result)],
                factor=4.0,
                working_dtype=torch.float32,
            )
        )


@pytest.mark.parametrize(
    "scheme,expected",
    [("native", "orbit"), ("native:orbit", "orbit"), ("native:exchange", "exchange")],
)
@pytest.mark.parametrize(
    "backend,descriptor",
    [
        ("dpmodel", "dpa4"),
        ("dpmodel", "dpa4c"),
        ("pt", "dpa4"),
        ("pt_expt", "dpa4"),
        ("pt_expt", "dpa4c"),
    ],
)
def test_native_scheme_builds_and_serializes_mode(
    backend: str, descriptor: str, scheme: str, expected: str
) -> None:
    if backend == "pt":
        from deepmd.pt.model.model import (
            get_model,
        )
    elif backend == "pt_expt":
        from deepmd.pt_expt.model.get_model import (
            get_model,
        )
    else:
        from deepmd.dpmodel.model.model import (
            get_model,
        )

    config = {
        "type": "dpa4" if backend == "pt" else "standard",
        "type_map": ["Fe", "C"],
        "descriptor": {
            "type": descriptor,
            "rcut": 3.0,
            "channels": 8,
            "n_radial": 4,
            "lmax": 2,
            "precision": "float64",
            "seed": 7,
        },
        "fitting_net": {
            "type": "dpa4_ener" if descriptor == "dpa4" else "ener",
            "neuron": [4],
        },
        "spin": {"scheme": scheme, "use_spin": [True, False]},
    }
    if descriptor == "dpa4":
        config["descriptor"].update(sel=8, n_blocks=1, mmax=1, random_gamma=False)
    model = get_model(config)
    assert model.atomic_model.descriptor.spin_mode == expected
    restored = type(model).deserialize(model.serialize())
    assert restored.atomic_model.descriptor.spin_mode == expected
