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
from deepmd.pt_expt.utils import (
    env as _env,
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
        torch.testing.assert_close(
            out0["energy_redu"], out1["energy_redu"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            out0["energy_derv_r"], out1["energy_derv_r"], rtol=0, atol=0
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
