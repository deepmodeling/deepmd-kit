# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reusable grad-accumulation invariant harness for dpmodel loss tests.

This module provides ``assert_grad_accum_invariant`` for Tasks 2-5 that
verify the loss on a padded multi-frame batch equals mean(per_frame_loss).

The dpmodel losses accept numpy arrays (via the array_api_compat backend).

Scope / follow-ups (mixed_type padding fix, PR #5738)
----------------------------------------------------
- The TF backend loss is not covered here and still has the mixed_type
  dilution behavior; tracked in deepmodeling/deepmd-kit#5760.
- The pt-only losses ``dens``/``population``/``denoise`` are out of scope;
  tracked in deepmodeling/deepmd-kit#5761.
- ``ener_spin``'s ``force_mag`` MAE now uses a batch-size-independent mean
  reduction (frames/atoms/xyz), consistent with force_mag MSE and force_real
  MAE; the grad-accum invariant is asserted by the test below.

Constants
---------
NA = 3   # real atoms in the short frame
NB = 5   # real atoms in the full-width frame (NB == NP)
NP = 5   # padded width (nloc)
"""

import numpy as np

from deepmd.dpmodel.loss.dos import (
    DOSLoss,
)
from deepmd.dpmodel.loss.ener import (
    EnergyLoss,
)
from deepmd.dpmodel.loss.ener_spin import EnergySpinLoss as EnergySpinLossDPModel
from deepmd.dpmodel.loss.property import (
    PropertyLoss,
)
from deepmd.dpmodel.loss.tensor import (
    TensorLoss,
)

# ---------------------------------------------------------------------------
# Constants used by the multi-frame test harness (Tasks 2-5)
# ---------------------------------------------------------------------------

NA = 3  # real atoms in frame A
NB = 5  # real atoms in frame B (== NP, so frame B is fully real)
NP = 5  # padded width (nloc)


# ---------------------------------------------------------------------------
# Reusable harness (used by Tasks 2-5; imported from here)
# ---------------------------------------------------------------------------


def assert_grad_accum_invariant(
    loss_fn,
    make_batch_A,
    make_batch_B,
    make_padded_batch,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Assert that padded-batch loss == mean(per_frame_loss) for two frames.

    The grad-accumulation invariant: a padded batch of [frame_A (NA real atoms
    padded to NP) + frame_B (NB==NP real atoms)] must yield the same loss as
    processing each frame separately and averaging.

    Parameters
    ----------
    loss_fn : callable
        Signature ``(model_pred, label, natoms) -> float``.  The function
        receives numpy-dict inputs and returns a scalar float.
    make_batch_A : callable
        Returns ``(model_pred, label, natoms)`` for frame A alone (1 frame, NA atoms).
    make_batch_B : callable
        Returns ``(model_pred, label, natoms)`` for frame B alone (1 frame, NB atoms).
    make_padded_batch : callable
        Returns ``(model_pred, label, natoms)`` for the 2-frame padded batch
        (nf=2, nloc=NP; frame A is padded with NP-NA ghost rows).
    rtol : float
        Relative tolerance for ``np.isclose``.
    atol : float
        Absolute tolerance for ``np.isclose``.
    """
    pred_A, label_A, natoms_A = make_batch_A()
    pred_B, label_B, natoms_B = make_batch_B()
    pred_pad, label_pad, natoms_pad = make_padded_batch()

    loss_A = float(loss_fn(pred_A, label_A, natoms_A))
    loss_B = float(loss_fn(pred_B, label_B, natoms_B))
    ref = 0.5 * (loss_A + loss_B)

    loss_pad = float(loss_fn(pred_pad, label_pad, natoms_pad))

    assert np.isclose(loss_pad, ref, rtol=rtol, atol=atol), (
        f"Grad-accum invariant violated: padded_loss={loss_pad:.8f}, "
        f"ref={ref:.8f}, diff={abs(loss_pad - ref):.2e}"
    )


# ---------------------------------------------------------------------------
# Helpers for constructing test data
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
NUMB_DOS = 4
TENSOR_SIZE = 3


def _rnd(*shape):
    return RNG.standard_normal(shape).astype(np.float64)


# ---------------------------------------------------------------------------
# Task 2: DOSLoss -- atomic (ados / acdf) and global (dos / cdf)
# ---------------------------------------------------------------------------


class TestDOSLossAtomicGradAccum:
    """Per-frame masked mean (idiom 1) for atomic dos / acdf terms."""

    def _make_loss(self):
        return DOSLoss(
            starter_learning_rate=1.0,
            numb_dos=NUMB_DOS,
            start_pref_dos=0.0,
            limit_pref_dos=0.0,
            start_pref_cdf=0.0,
            limit_pref_cdf=0.0,
            start_pref_ados=1.0,
            limit_pref_ados=1.0,
            start_pref_acdf=1.0,
            limit_pref_acdf=1.0,
        )

    def _atom_dos_A(self):
        return _rnd(NA, NUMB_DOS)

    def _atom_dos_B(self):
        return _rnd(NB, NUMB_DOS)

    def _loss_fn(self, model_pred, label, natoms):
        loss, _ = self._make_loss().call(1.0, natoms, model_pred, label)
        return float(loss)

    def _make_batch_A(self):
        pred = _rnd(NA, NUMB_DOS)
        label = _rnd(NA, NUMB_DOS)
        model_pred = {
            "atom_dos": pred,
            "mask": np.ones((1, NA), dtype=np.float64),
        }
        label_dict = {"atom_dos": label, "find_atom_dos": 1.0}
        return model_pred, label_dict, NA

    def _make_batch_B(self):
        pred = _rnd(NB, NUMB_DOS)
        label = _rnd(NB, NUMB_DOS)
        model_pred = {
            "atom_dos": pred,
            "mask": np.ones((1, NB), dtype=np.float64),
        }
        label_dict = {"atom_dos": label, "find_atom_dos": 1.0}
        return model_pred, label_dict, NB

    def _make_padded_batch(self, pred_A_data, label_A_data, pred_B_data, label_B_data):
        # Frame A: NA real atoms padded to NP; ghost pred/label = 0
        pred_A_pad = np.zeros((NP, NUMB_DOS), dtype=np.float64)
        pred_A_pad[:NA] = pred_A_data
        label_A_pad = np.zeros((NP, NUMB_DOS), dtype=np.float64)
        label_A_pad[:NA] = label_A_data
        mask_A = np.array([[1.0] * NA + [0.0] * (NP - NA)], dtype=np.float64)

        # Frame B: NB == NP, all real
        mask_B = np.ones((1, NB), dtype=np.float64)

        atom_dos_pad = np.concatenate(
            [pred_A_pad, pred_B_data], axis=0
        )  # [NP+NB, ncomp]
        atom_dos_label = np.concatenate([label_A_pad, label_B_data], axis=0)
        mask_pad = np.concatenate([mask_A, mask_B], axis=0)  # [2, NP]
        model_pred = {"atom_dos": atom_dos_pad, "mask": mask_pad}
        label_dict = {"atom_dos": atom_dos_label, "find_atom_dos": 1.0}
        return model_pred, label_dict, NP

    def test_ados_grad_accum_invariant(self):
        """Atomic dos per-frame masked mean meets the grad-accum invariant."""
        pred_A = _rnd(NA, NUMB_DOS)
        label_A = _rnd(NA, NUMB_DOS)
        pred_B = _rnd(NB, NUMB_DOS)
        label_B = _rnd(NB, NUMB_DOS)

        def make_A():
            return (
                {"atom_dos": pred_A, "mask": np.ones((1, NA), dtype=np.float64)},
                {"atom_dos": label_A, "find_atom_dos": 1.0},
                NA,
            )

        def make_B():
            return (
                {"atom_dos": pred_B, "mask": np.ones((1, NB), dtype=np.float64)},
                {"atom_dos": label_B, "find_atom_dos": 1.0},
                NB,
            )

        def make_padded():
            return self._make_padded_batch(pred_A, label_A, pred_B, label_B)

        assert_grad_accum_invariant(self._loss_fn, make_A, make_B, make_padded)

    def test_acdf_grad_accum_invariant(self):
        """Atomic cdf per-frame masked mean meets the grad-accum invariant."""
        # acdf uses cumsum of atom_dos -- same data pipeline
        self.test_ados_grad_accum_invariant()

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same loss as no mask (non-mixed batch)."""
        pred = _rnd(NB, NUMB_DOS)
        label = _rnd(NB, NUMB_DOS)
        with_mask = {
            "atom_dos": pred,
            "mask": np.ones((1, NB), dtype=np.float64),
        }
        without_mask = {"atom_dos": pred}
        label_dict = {"atom_dos": label, "find_atom_dos": 1.0}
        loss_m, _ = self._make_loss().call(1.0, NB, with_mask, label_dict)
        loss_nm, _ = self._make_loss().call(1.0, NB, without_mask, label_dict)
        assert np.isclose(float(loss_m), float(loss_nm)), (
            f"all-ones mask must be no-op: {float(loss_m)} vs {float(loss_nm)}"
        )


class TestDOSLossGlobalGradAccum:
    """Plain mean (idiom 3) for global dos / cdf terms."""

    def _make_loss(self):
        return DOSLoss(
            starter_learning_rate=1.0,
            numb_dos=NUMB_DOS,
            start_pref_dos=1.0,
            limit_pref_dos=1.0,
            start_pref_cdf=1.0,
            limit_pref_cdf=1.0,
            start_pref_ados=0.0,
            limit_pref_ados=0.0,
            start_pref_acdf=0.0,
            limit_pref_acdf=0.0,
        )

    def _loss_fn(self, model_pred, label, natoms):
        loss, _ = self._make_loss().call(1.0, natoms, model_pred, label)
        return float(loss)

    def test_dos_grad_accum_invariant(self):
        """Global dos plain mean meets the grad-accum invariant."""
        pred_A = _rnd(1, NUMB_DOS)
        label_A = _rnd(1, NUMB_DOS)
        pred_B = _rnd(1, NUMB_DOS)
        label_B = _rnd(1, NUMB_DOS)

        def make_A():
            return (
                {"dos": pred_A, "mask": np.ones((1, NA), dtype=np.float64)},
                {"dos": label_A, "find_dos": 1.0},
                NA,
            )

        def make_B():
            return (
                {"dos": pred_B, "mask": np.ones((1, NB), dtype=np.float64)},
                {"dos": label_B, "find_dos": 1.0},
                NB,
            )

        def make_padded():
            pred_pad = np.concatenate([pred_A, pred_B], axis=0)  # [2, ncomp]
            label_pad = np.concatenate([label_A, label_B], axis=0)
            mask_pad = np.array(
                [[1.0] * NA + [0.0] * (NP - NA), [1.0] * NB], dtype=np.float64
            )
            return (
                {"dos": pred_pad, "mask": mask_pad},
                {"dos": label_pad, "find_dos": 1.0},
                NP,
            )

        assert_grad_accum_invariant(self._loss_fn, make_A, make_B, make_padded)

    def test_cdf_grad_accum_invariant(self):
        """Global cdf plain mean meets the grad-accum invariant."""
        # cdf uses cumsum of dos -- reuse the same logic
        self.test_dos_grad_accum_invariant()

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same loss as no mask (non-mixed batch)."""
        pred = _rnd(2, NUMB_DOS)
        label = _rnd(2, NUMB_DOS)
        with_mask = {"dos": pred, "mask": np.ones((2, NB), dtype=np.float64)}
        without_mask = {"dos": pred}
        label_dict = {"dos": label, "find_dos": 1.0}
        loss_m, _ = self._make_loss().call(1.0, NB, with_mask, label_dict)
        loss_nm, _ = self._make_loss().call(1.0, NB, without_mask, label_dict)
        assert np.isclose(float(loss_m), float(loss_nm)), (
            f"all-ones mask must be no-op: {float(loss_m)} vs {float(loss_nm)}"
        )


# ---------------------------------------------------------------------------
# Task 2: TensorLoss -- local and global tensor
# ---------------------------------------------------------------------------


class TestTensorLossLocalGradAccum:
    """Per-frame masked mean (idiom 1) for local tensor term."""

    def _make_loss(self):
        return TensorLoss(
            tensor_name="dipole",
            tensor_size=TENSOR_SIZE,
            label_name="dipole",
            pref_atomic=1.0,
            pref=0.0,
        )

    def _loss_fn(self, model_pred, label, natoms):
        loss, _ = self._make_loss().call(1.0, natoms, model_pred, label)
        return float(loss)

    def test_local_grad_accum_invariant(self):
        """Local tensor per-frame masked mean meets the grad-accum invariant."""
        pred_A = _rnd(NA, TENSOR_SIZE)
        label_A = _rnd(NA, TENSOR_SIZE)
        pred_B = _rnd(NB, TENSOR_SIZE)
        label_B = _rnd(NB, TENSOR_SIZE)

        def make_A():
            return (
                {"dipole": pred_A, "mask": np.ones((1, NA), dtype=np.float64)},
                {"atom_dipole": label_A, "find_atom_dipole": 1.0},
                NA,
            )

        def make_B():
            return (
                {"dipole": pred_B, "mask": np.ones((1, NB), dtype=np.float64)},
                {"atom_dipole": label_B, "find_atom_dipole": 1.0},
                NB,
            )

        def make_padded():
            pred_A_pad = np.zeros((NP, TENSOR_SIZE), dtype=np.float64)
            pred_A_pad[:NA] = pred_A
            label_A_pad = np.zeros((NP, TENSOR_SIZE), dtype=np.float64)
            label_A_pad[:NA] = label_A
            mask_A = np.array([[1.0] * NA + [0.0] * (NP - NA)], dtype=np.float64)
            mask_B = np.ones((1, NB), dtype=np.float64)
            dipole_pad = np.concatenate([pred_A_pad, pred_B], axis=0)
            label_pad = np.concatenate([label_A_pad, label_B], axis=0)
            mask_pad = np.concatenate([mask_A, mask_B], axis=0)
            return (
                {"dipole": dipole_pad, "mask": mask_pad},
                {"atom_dipole": label_pad, "find_atom_dipole": 1.0},
                NP,
            )

        assert_grad_accum_invariant(self._loss_fn, make_A, make_B, make_padded)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same loss as no mask (non-mixed batch)."""
        pred = _rnd(NB, TENSOR_SIZE)
        label = _rnd(NB, TENSOR_SIZE)
        with_mask = {"dipole": pred, "mask": np.ones((1, NB), dtype=np.float64)}
        without_mask = {"dipole": pred}
        label_dict = {"atom_dipole": label, "find_atom_dipole": 1.0}
        loss_m, _ = self._make_loss().call(1.0, NB, with_mask, label_dict)
        loss_nm, _ = self._make_loss().call(1.0, NB, without_mask, label_dict)
        assert np.isclose(float(loss_m), float(loss_nm)), (
            f"all-ones mask must be no-op: {float(loss_m)} vs {float(loss_nm)}"
        )


class TestTensorLossGlobalGradAccum:
    """Plain mean (idiom 3) for global tensor term."""

    def _make_loss(self):
        return TensorLoss(
            tensor_name="dipole",
            tensor_size=TENSOR_SIZE,
            label_name="dipole",
            pref_atomic=0.0,
            pref=1.0,
        )

    def _loss_fn(self, model_pred, label, natoms):
        loss, _ = self._make_loss().call(1.0, natoms, model_pred, label)
        return float(loss)

    def test_global_grad_accum_invariant(self):
        """Global tensor plain mean meets the grad-accum invariant."""
        pred_A = _rnd(1, TENSOR_SIZE)
        label_A = _rnd(1, TENSOR_SIZE)
        pred_B = _rnd(1, TENSOR_SIZE)
        label_B = _rnd(1, TENSOR_SIZE)

        def make_A():
            return (
                {"global_dipole": pred_A, "mask": np.ones((1, NA), dtype=np.float64)},
                {"dipole": label_A, "find_dipole": 1.0},
                NA,
            )

        def make_B():
            return (
                {"global_dipole": pred_B, "mask": np.ones((1, NB), dtype=np.float64)},
                {"dipole": label_B, "find_dipole": 1.0},
                NB,
            )

        def make_padded():
            pred_pad = np.concatenate([pred_A, pred_B], axis=0)
            label_pad = np.concatenate([label_A, label_B], axis=0)
            mask_pad = np.array(
                [[1.0] * NA + [0.0] * (NP - NA), [1.0] * NB], dtype=np.float64
            )
            return (
                {"global_dipole": pred_pad, "mask": mask_pad},
                {"dipole": label_pad, "find_dipole": 1.0},
                NP,
            )

        assert_grad_accum_invariant(self._loss_fn, make_A, make_B, make_padded)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same loss as no mask (non-mixed batch)."""
        pred = _rnd(2, TENSOR_SIZE)
        label = _rnd(2, TENSOR_SIZE)
        with_mask = {"global_dipole": pred, "mask": np.ones((2, NB), dtype=np.float64)}
        without_mask = {"global_dipole": pred}
        label_dict = {"dipole": label, "find_dipole": 1.0}
        loss_m, _ = self._make_loss().call(1.0, NB, with_mask, label_dict)
        loss_nm, _ = self._make_loss().call(1.0, NB, without_mask, label_dict)
        assert np.isclose(float(loss_m), float(loss_nm)), (
            f"all-ones mask must be no-op: {float(loss_m)} vs {float(loss_nm)}"
        )


# ---------------------------------------------------------------------------
# Task 3: EnergyLoss -- energy, force, virial, atom_ener, atom_pref
# ---------------------------------------------------------------------------


def _full_ener_dicts(nf, nloc, energy_pred, energy_label, mask=None, **overrides):
    """Build complete model_pred and label_dict for EnergyLoss.call.

    EnergyLoss.call fetches energy/force/virial/atom_energy unconditionally,
    so all keys must be present regardless of which term is under test.
    """
    model_pred = {
        "energy": energy_pred,  # [nf, 1]
        "force": np.zeros((nf, nloc, 3), dtype=np.float64),
        "virial": np.zeros((nf, 9), dtype=np.float64),
        "atom_energy": np.zeros((nf, nloc, 1), dtype=np.float64),
    }
    label_dict = {
        "energy": energy_label,  # [nf, 1]
        "force": np.zeros((nf, nloc, 3), dtype=np.float64),
        "virial": np.zeros((nf, 9), dtype=np.float64),
        "atom_ener": np.zeros((nf, nloc, 1), dtype=np.float64),
        "atom_pref": np.zeros((nf, nloc * 3), dtype=np.float64),
        "find_energy": 1.0,
        "find_force": 0.0,
        "find_virial": 0.0,
        "find_atom_ener": 0.0,
        "find_atom_pref": 0.0,
    }
    if mask is not None:
        model_pred["mask"] = mask
    model_pred.update({k: v for k, v in overrides.items() if k in model_pred})
    label_dict.update({k: v for k, v in overrides.items() if k in label_dict})
    return model_pred, label_dict


def _padded_force(f_A, f_B):
    """Stack force arrays for a 2-frame padded batch (NA<=NP, NB==NP)."""
    f_A_pad = np.zeros((NP, 3), dtype=np.float64)
    f_A_pad[:NA] = f_A
    return np.stack([f_A_pad, f_B], axis=0)  # [2, NP, 3]


def _padded_atom(arr_A, arr_B, ncomp):
    """Pad arr_A from [NA, ncomp] to [NP, ncomp] with zeros, stack with arr_B."""
    pad = np.zeros((NP, ncomp), dtype=np.float64)
    pad[:NA] = arr_A
    return np.stack([pad, arr_B], axis=0)  # [2, NP, ncomp]


def _padded_atom_flat(arr_A, arr_B, ncomp):
    """Pad then reshape to [2, NP*ncomp] (for atom_pref shape)."""
    return _padded_atom(arr_A, arr_B, ncomp).reshape(2, NP * ncomp)


_MASK_PAD = np.array(
    [[1.0] * NA + [0.0] * (NP - NA), [1.0] * NB], dtype=np.float64
)  # [2, NP]


class TestDPModelEnergyLossEnerGradAccum:
    """Idiom 2 (extensive) for the energy (has_e) term in EnergyLoss.

    Covers: mse (norm_exp=1 and 2), mae, huber; plus non-mixed no-op.
    """

    def _make_loss(self, loss_func="mse", intensive=False, use_huber=False):
        return EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_ae=0.0,
            limit_pref_ae=0.0,
            start_pref_pf=0.0,
            limit_pref_pf=0.0,
            loss_func=loss_func,
            intensive_ener_virial=intensive,
            use_huber=use_huber,
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        loss, _ = loss_obj.call(1.0, natoms, model_pred, label)
        return float(loss)

    def _run_invariant(self, loss_obj, e_A, e_A_hat, e_B, e_B_hat):
        def make_A():
            p, l = _full_ener_dicts(
                1, NA, e_A, e_A_hat, mask=np.ones((1, NA), dtype=np.float64)
            )
            return p, l, NA

        def make_B():
            p, l = _full_ener_dicts(
                1, NB, e_B, e_B_hat, mask=np.ones((1, NB), dtype=np.float64)
            )
            return p, l, NB

        def make_padded():
            e_pad = np.concatenate([e_A, e_B], axis=0)  # [2, 1]
            e_hat_pad = np.concatenate([e_A_hat, e_B_hat], axis=0)
            p, l = _full_ener_dicts(2, NP, e_pad, e_hat_pad, mask=_MASK_PAD)
            return p, l, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: self._loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_non_intensive_grad_accum(self):
        """Energy MSE norm_exp=1 meets the grad-accum invariant."""
        e_A, e_B = _rnd(1, 1), _rnd(1, 1)
        e_A_hat, e_B_hat = _rnd(1, 1), _rnd(1, 1)
        self._run_invariant(
            self._make_loss(loss_func="mse", intensive=False),
            e_A,
            e_A_hat,
            e_B,
            e_B_hat,
        )

    def test_mse_intensive_grad_accum(self):
        """Energy MSE norm_exp=2 meets the grad-accum invariant."""
        e_A, e_B = _rnd(1, 1), _rnd(1, 1)
        e_A_hat, e_B_hat = _rnd(1, 1), _rnd(1, 1)
        self._run_invariant(
            self._make_loss(loss_func="mse", intensive=True),
            e_A,
            e_A_hat,
            e_B,
            e_B_hat,
        )

    def test_mae_grad_accum(self):
        """Energy MAE meets the grad-accum invariant."""
        e_A, e_B = _rnd(1, 1), _rnd(1, 1)
        e_A_hat, e_B_hat = _rnd(1, 1), _rnd(1, 1)
        self._run_invariant(
            self._make_loss(loss_func="mae"),
            e_A,
            e_A_hat,
            e_B,
            e_B_hat,
        )

    def test_huber_grad_accum(self):
        """Energy Huber meets the grad-accum invariant."""
        e_A, e_B = _rnd(1, 1), _rnd(1, 1)
        e_A_hat, e_B_hat = _rnd(1, 1), _rnd(1, 1)
        self._run_invariant(
            self._make_loss(loss_func="mse", use_huber=True),
            e_A,
            e_A_hat,
            e_B,
            e_B_hat,
        )

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same energy loss as no mask."""
        e = _rnd(1, 1)
        e_hat = _rnd(1, 1)
        loss_obj = self._make_loss()
        p_mask, l_mask = _full_ener_dicts(
            1, NP, e, e_hat, mask=np.ones((1, NP), dtype=np.float64)
        )
        p_nomask, l_nomask = _full_ener_dicts(1, NP, e, e_hat)
        loss_m = self._loss_fn(loss_obj, p_mask, l_mask, NP)
        loss_nm = self._loss_fn(loss_obj, p_nomask, l_nomask, NP)
        assert np.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m} vs {loss_nm}"
        )


class TestDPModelEnergyLossForceGradAccum:
    """Idiom 1 (per-atom masked mean, ncomp=3) for the force (has_f) term.

    Covers: mse, mae, huber; plus non-mixed no-op.
    """

    def _make_loss(self, loss_func="mse", use_huber=False, f_use_norm=False):
        return EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=1.0,
            limit_pref_f=1.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_ae=0.0,
            limit_pref_ae=0.0,
            start_pref_pf=0.0,
            limit_pref_pf=0.0,
            loss_func=loss_func,
            use_huber=use_huber,
            f_use_norm=f_use_norm,
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        loss, _ = loss_obj.call(1.0, natoms, model_pred, label)
        return float(loss)

    def _run_invariant(self, loss_obj, f_A, f_A_hat, f_B, f_B_hat):
        def make_A():
            p, l = _full_ener_dicts(
                1,
                NA,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                mask=np.ones((1, NA), dtype=np.float64),
            )
            p["force"] = f_A[None]  # [1, NA, 3]
            l["force"] = f_A_hat[None]
            l["find_force"] = 1.0
            return p, l, NA

        def make_B():
            p, l = _full_ener_dicts(
                1,
                NB,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                mask=np.ones((1, NB), dtype=np.float64),
            )
            p["force"] = f_B[None]
            l["force"] = f_B_hat[None]
            l["find_force"] = 1.0
            return p, l, NB

        def make_padded():
            f_pad = _padded_force(f_A, f_B)  # [2, NP, 3]
            f_hat_pad = _padded_force(f_A_hat, f_B_hat)
            p, l = _full_ener_dicts(
                2, NP, np.zeros((2, 1)), np.zeros((2, 1)), mask=_MASK_PAD
            )
            p["force"] = f_pad
            l["force"] = f_hat_pad
            l["find_force"] = 1.0
            return p, l, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: self._loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_grad_accum(self):
        """Force MSE meets the grad-accum invariant."""
        f_A = _rnd(NA, 3)
        f_A_hat = _rnd(NA, 3)
        f_B = _rnd(NB, 3)
        f_B_hat = _rnd(NB, 3)
        self._run_invariant(self._make_loss("mse"), f_A, f_A_hat, f_B, f_B_hat)

    def test_mae_grad_accum(self):
        """Force MAE meets the grad-accum invariant."""
        f_A = _rnd(NA, 3)
        f_A_hat = _rnd(NA, 3)
        f_B = _rnd(NB, 3)
        f_B_hat = _rnd(NB, 3)
        self._run_invariant(self._make_loss("mae"), f_A, f_A_hat, f_B, f_B_hat)

    def test_huber_grad_accum(self):
        """Force Huber meets the grad-accum invariant."""
        f_A = _rnd(NA, 3)
        f_A_hat = _rnd(NA, 3)
        f_B = _rnd(NB, 3)
        f_B_hat = _rnd(NB, 3)
        self._run_invariant(
            self._make_loss("mse", use_huber=True), f_A, f_A_hat, f_B, f_B_hat
        )

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same force loss as no mask."""
        f = _rnd(NP, 3)
        f_hat = _rnd(NP, 3)
        loss_obj = self._make_loss()

        p_mask, l_mask = _full_ener_dicts(
            1, NP, np.zeros((1, 1)), np.zeros((1, 1)), mask=np.ones((1, NP))
        )
        p_mask["force"] = f[None]
        l_mask["force"] = f_hat[None]
        l_mask["find_force"] = 1.0

        p_nm, l_nm = _full_ener_dicts(1, NP, np.zeros((1, 1)), np.zeros((1, 1)))
        p_nm["force"] = f[None]
        l_nm["force"] = f_hat[None]
        l_nm["find_force"] = 1.0

        loss_m = self._loss_fn(loss_obj, p_mask, l_mask, NP)
        loss_nm = self._loss_fn(loss_obj, p_nm, l_nm, NP)
        assert np.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m} vs {loss_nm}"
        )


class TestDPModelEnergyLossVirialGradAccum:
    """Idiom 2 (extensive, k=9) for the virial (has_v) term.

    Covers: mse (norm_exp=1 and 2), mae, huber; plus non-mixed no-op.
    """

    def _make_loss(self, loss_func="mse", intensive=False, use_huber=False):
        return EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            start_pref_ae=0.0,
            limit_pref_ae=0.0,
            start_pref_pf=0.0,
            limit_pref_pf=0.0,
            loss_func=loss_func,
            intensive_ener_virial=intensive,
            use_huber=use_huber,
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        loss, _ = loss_obj.call(1.0, natoms, model_pred, label)
        return float(loss)

    def _run_invariant(self, loss_obj, v_A, v_A_hat, v_B, v_B_hat):
        def make_A():
            p, l = _full_ener_dicts(
                1, NA, np.zeros((1, 1)), np.zeros((1, 1)), mask=np.ones((1, NA))
            )
            p["virial"] = v_A[None]  # [1, 9]
            l["virial"] = v_A_hat[None]
            l["find_virial"] = 1.0
            return p, l, NA

        def make_B():
            p, l = _full_ener_dicts(
                1, NB, np.zeros((1, 1)), np.zeros((1, 1)), mask=np.ones((1, NB))
            )
            p["virial"] = v_B[None]
            l["virial"] = v_B_hat[None]
            l["find_virial"] = 1.0
            return p, l, NB

        def make_padded():
            v_pad = np.stack([v_A, v_B], axis=0)  # [2, 9]
            v_hat_pad = np.stack([v_A_hat, v_B_hat], axis=0)
            p, l = _full_ener_dicts(
                2, NP, np.zeros((2, 1)), np.zeros((2, 1)), mask=_MASK_PAD
            )
            p["virial"] = v_pad
            l["virial"] = v_hat_pad
            l["find_virial"] = 1.0
            return p, l, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: self._loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_non_intensive_grad_accum(self):
        """Virial MSE norm_exp=1 meets the grad-accum invariant."""
        v_A, v_B = _rnd(9), _rnd(9)
        v_A_hat, v_B_hat = _rnd(9), _rnd(9)
        self._run_invariant(
            self._make_loss("mse", intensive=False), v_A, v_A_hat, v_B, v_B_hat
        )

    def test_mse_intensive_grad_accum(self):
        """Virial MSE norm_exp=2 meets the grad-accum invariant."""
        v_A, v_B = _rnd(9), _rnd(9)
        v_A_hat, v_B_hat = _rnd(9), _rnd(9)
        self._run_invariant(
            self._make_loss("mse", intensive=True), v_A, v_A_hat, v_B, v_B_hat
        )

    def test_mae_grad_accum(self):
        """Virial MAE meets the grad-accum invariant."""
        v_A, v_B = _rnd(9), _rnd(9)
        v_A_hat, v_B_hat = _rnd(9), _rnd(9)
        self._run_invariant(self._make_loss("mae"), v_A, v_A_hat, v_B, v_B_hat)

    def test_huber_grad_accum(self):
        """Virial Huber meets the grad-accum invariant."""
        v_A, v_B = _rnd(9), _rnd(9)
        v_A_hat, v_B_hat = _rnd(9), _rnd(9)
        self._run_invariant(
            self._make_loss("mse", use_huber=True), v_A, v_A_hat, v_B, v_B_hat
        )

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same virial loss as no mask."""
        v = _rnd(9)
        v_hat = _rnd(9)
        loss_obj = self._make_loss()

        p_mask, l_mask = _full_ener_dicts(
            1, NP, np.zeros((1, 1)), np.zeros((1, 1)), mask=np.ones((1, NP))
        )
        p_mask["virial"] = v[None]
        l_mask["virial"] = v_hat[None]
        l_mask["find_virial"] = 1.0

        p_nm, l_nm = _full_ener_dicts(1, NP, np.zeros((1, 1)), np.zeros((1, 1)))
        p_nm["virial"] = v[None]
        l_nm["virial"] = v_hat[None]
        l_nm["find_virial"] = 1.0

        loss_m = self._loss_fn(loss_obj, p_mask, l_mask, NP)
        loss_nm = self._loss_fn(loss_obj, p_nm, l_nm, NP)
        assert np.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m} vs {loss_nm}"
        )


class TestDPModelEnergyLossAtomEnerGradAccum:
    """Idiom 1 (per-atom masked mean, ncomp=1) for the atom_ener (has_ae) term.

    Covers: mse, mae; plus non-mixed no-op.
    """

    def _make_loss(self, loss_func="mse"):
        return EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_ae=1.0,
            limit_pref_ae=1.0,
            start_pref_pf=0.0,
            limit_pref_pf=0.0,
            loss_func=loss_func,
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        loss, _ = loss_obj.call(1.0, natoms, model_pred, label)
        return float(loss)

    def _run_invariant(self, loss_obj, ae_A, ae_A_hat, ae_B, ae_B_hat):
        def make_A():
            p, l = _full_ener_dicts(
                1, NA, np.zeros((1, 1)), np.zeros((1, 1)), mask=np.ones((1, NA))
            )
            p["atom_energy"] = ae_A[None]  # [1, NA, 1]
            l["atom_ener"] = ae_A_hat[None]
            l["find_atom_ener"] = 1.0
            return p, l, NA

        def make_B():
            p, l = _full_ener_dicts(
                1, NB, np.zeros((1, 1)), np.zeros((1, 1)), mask=np.ones((1, NB))
            )
            p["atom_energy"] = ae_B[None]
            l["atom_ener"] = ae_B_hat[None]
            l["find_atom_ener"] = 1.0
            return p, l, NB

        def make_padded():
            ae_pad = _padded_atom(ae_A, ae_B, 1)  # [2, NP, 1]
            ae_hat_pad = _padded_atom(ae_A_hat, ae_B_hat, 1)
            p, l = _full_ener_dicts(
                2, NP, np.zeros((2, 1)), np.zeros((2, 1)), mask=_MASK_PAD
            )
            p["atom_energy"] = ae_pad
            l["atom_ener"] = ae_hat_pad
            l["find_atom_ener"] = 1.0
            return p, l, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: self._loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_grad_accum(self):
        """Atom energy MSE meets the grad-accum invariant."""
        ae_A = _rnd(NA, 1)
        ae_A_hat = _rnd(NA, 1)
        ae_B = _rnd(NB, 1)
        ae_B_hat = _rnd(NB, 1)
        self._run_invariant(self._make_loss("mse"), ae_A, ae_A_hat, ae_B, ae_B_hat)

    def test_mae_grad_accum(self):
        """Atom energy MAE meets the grad-accum invariant."""
        ae_A = _rnd(NA, 1)
        ae_A_hat = _rnd(NA, 1)
        ae_B = _rnd(NB, 1)
        ae_B_hat = _rnd(NB, 1)
        self._run_invariant(self._make_loss("mae"), ae_A, ae_A_hat, ae_B, ae_B_hat)

    def test_huber_grad_accum(self):
        """Atom energy Huber (use_huber=True) meets the grad-accum invariant."""
        ae_A = _rnd(NA, 1)
        ae_A_hat = _rnd(NA, 1)
        ae_B = _rnd(NB, 1)
        ae_B_hat = _rnd(NB, 1)
        loss_obj = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_ae=1.0,
            limit_pref_ae=1.0,
            start_pref_pf=0.0,
            limit_pref_pf=0.0,
            use_huber=True,
        )
        self._run_invariant(loss_obj, ae_A, ae_A_hat, ae_B, ae_B_hat)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same atom-energy loss as no mask."""
        ae = _rnd(NP, 1)
        ae_hat = _rnd(NP, 1)
        loss_obj = self._make_loss()

        p_mask, l_mask = _full_ener_dicts(
            1, NP, np.zeros((1, 1)), np.zeros((1, 1)), mask=np.ones((1, NP))
        )
        p_mask["atom_energy"] = ae[None]
        l_mask["atom_ener"] = ae_hat[None]
        l_mask["find_atom_ener"] = 1.0

        p_nm, l_nm = _full_ener_dicts(1, NP, np.zeros((1, 1)), np.zeros((1, 1)))
        p_nm["atom_energy"] = ae[None]
        l_nm["atom_ener"] = ae_hat[None]
        l_nm["find_atom_ener"] = 1.0

        loss_m = self._loss_fn(loss_obj, p_mask, l_mask, NP)
        loss_nm = self._loss_fn(loss_obj, p_nm, l_nm, NP)
        assert np.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m} vs {loss_nm}"
        )


class TestDPModelEnergyLossAtomPrefGradAccum:
    """Idiom 1 with pref weight (ncomp=3) for the atom_pref (has_pf) term.

    Covers: mse, mae; plus non-mixed no-op.
    """

    def _make_loss(self, loss_func="mse"):
        return EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=1.0,
            limit_pref_f=1.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_ae=0.0,
            limit_pref_ae=0.0,
            start_pref_pf=1.0,
            limit_pref_pf=1.0,
            loss_func=loss_func,
        )

    def _loss_fn_pf_only(self, loss_obj, model_pred, label, natoms):
        """Return the atom_pref contribution to the loss (subtract force loss)."""
        # Both pref_f and pref_pf are 1.0 here, but we want ONLY the pf term.
        # Use a loss with pref_pf=1 and pref_f=0 to isolate it.
        loss, _ = loss_obj.call(1.0, natoms, model_pred, label)
        return float(loss)

    def _make_pf_only_loss(self, loss_func="mse"):
        """Loss with pref_f=0 and pref_pf=1 to isolate atom_pref term."""
        return EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_ae=0.0,
            limit_pref_ae=0.0,
            start_pref_pf=1.0,
            limit_pref_pf=1.0,
            loss_func=loss_func,
        )

    def _run_invariant(self, loss_obj, f_A, f_A_hat, pf_A, f_B, f_B_hat, pf_B):
        """Invariant for atom_pref using force diff weighted by pref."""

        def make_A():
            p, l = _full_ener_dicts(
                1, NA, np.zeros((1, 1)), np.zeros((1, 1)), mask=np.ones((1, NA))
            )
            p["force"] = f_A[None]
            l["force"] = f_A_hat[None]
            l["atom_pref"] = pf_A.reshape(1, NA * 3)
            l["find_force"] = 1.0
            l["find_atom_pref"] = 1.0
            return p, l, NA

        def make_B():
            p, l = _full_ener_dicts(
                1, NB, np.zeros((1, 1)), np.zeros((1, 1)), mask=np.ones((1, NB))
            )
            p["force"] = f_B[None]
            l["force"] = f_B_hat[None]
            l["atom_pref"] = pf_B.reshape(1, NB * 3)
            l["find_force"] = 1.0
            l["find_atom_pref"] = 1.0
            return p, l, NB

        def make_padded():
            f_pad = _padded_force(f_A, f_B)
            f_hat_pad = _padded_force(f_A_hat, f_B_hat)
            pf_pad = _padded_atom_flat(pf_A, pf_B, 3)  # [2, NP*3]
            p, l = _full_ener_dicts(
                2, NP, np.zeros((2, 1)), np.zeros((2, 1)), mask=_MASK_PAD
            )
            p["force"] = f_pad
            l["force"] = f_hat_pad
            l["atom_pref"] = pf_pad
            l["find_force"] = 1.0
            l["find_atom_pref"] = 1.0
            return p, l, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: float(loss_obj.call(1.0, na, mp, lb)[0]),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_grad_accum(self):
        """Atom-pref MSE meets the grad-accum invariant."""
        f_A, f_A_hat = _rnd(NA, 3), _rnd(NA, 3)
        pf_A = np.abs(_rnd(NA, 3)) + 0.1  # positive pref
        f_B, f_B_hat = _rnd(NB, 3), _rnd(NB, 3)
        pf_B = np.abs(_rnd(NB, 3)) + 0.1
        self._run_invariant(
            self._make_pf_only_loss("mse"), f_A, f_A_hat, pf_A, f_B, f_B_hat, pf_B
        )

    def test_mae_grad_accum(self):
        """Atom-pref MAE meets the grad-accum invariant."""
        f_A, f_A_hat = _rnd(NA, 3), _rnd(NA, 3)
        pf_A = np.abs(_rnd(NA, 3)) + 0.1
        f_B, f_B_hat = _rnd(NB, 3), _rnd(NB, 3)
        pf_B = np.abs(_rnd(NB, 3)) + 0.1
        self._run_invariant(
            self._make_pf_only_loss("mae"), f_A, f_A_hat, pf_A, f_B, f_B_hat, pf_B
        )

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same atom-pref loss as no mask."""
        f = _rnd(NP, 3)
        f_hat = _rnd(NP, 3)
        pf = np.abs(_rnd(NP, 3)) + 0.1
        loss_obj = self._make_pf_only_loss("mse")

        p_mask, l_mask = _full_ener_dicts(
            1, NP, np.zeros((1, 1)), np.zeros((1, 1)), mask=np.ones((1, NP))
        )
        p_mask["force"] = f[None]
        l_mask["force"] = f_hat[None]
        l_mask["atom_pref"] = pf.reshape(1, NP * 3)
        l_mask["find_force"] = 1.0
        l_mask["find_atom_pref"] = 1.0

        p_nm, l_nm = _full_ener_dicts(1, NP, np.zeros((1, 1)), np.zeros((1, 1)))
        p_nm["force"] = f[None]
        l_nm["force"] = f_hat[None]
        l_nm["atom_pref"] = pf.reshape(1, NP * 3)
        l_nm["find_force"] = 1.0
        l_nm["find_atom_pref"] = 1.0

        loss_m = float(loss_obj.call(1.0, NP, p_mask, l_mask)[0])
        loss_nm = float(loss_obj.call(1.0, NP, p_nm, l_nm)[0])
        assert np.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m} vs {loss_nm}"
        )


# ---------------------------------------------------------------------------
# Task 4: PropertyLoss -- extensive (not intensive) property
# ---------------------------------------------------------------------------

PROP_TASK_DIM = 2
PROP_VAR = "test_prop"


class TestPropertyLossExtensiveGradAccum:
    """Idiom 2 (per-frame real-natoms normalization) for extensive PropertyLoss.

    The _loss_fn wrapper divides by nf so that the per-frame average matches
    the separate-frame reference (property loss uses sum, not mean, over frames).
    """

    def _make_loss(self, loss_func="mse"):
        return PropertyLoss(
            task_dim=PROP_TASK_DIM,
            var_name=PROP_VAR,
            loss_func=loss_func,
            intensive=False,
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        """Return per-frame-averaged loss (raw loss / nf)."""
        nf = model_pred[PROP_VAR].shape[0]
        loss, _ = loss_obj.call(1.0, natoms, model_pred, label)
        return float(loss) / nf

    def _run_invariant(self, loss_obj, p_A, l_A, p_B, l_B):
        def make_A():
            return (
                {PROP_VAR: p_A, "mask": np.ones((1, NA), dtype=np.float64)},
                {PROP_VAR: l_A},
                NA,
            )

        def make_B():
            return (
                {PROP_VAR: p_B, "mask": np.ones((1, NB), dtype=np.float64)},
                {PROP_VAR: l_B},
                NB,
            )

        def make_padded():
            return (
                {
                    PROP_VAR: np.concatenate([p_A, p_B], axis=0),  # [2, task_dim]
                    "mask": _MASK_PAD,
                },
                {PROP_VAR: np.concatenate([l_A, l_B], axis=0)},
                NP,
            )

        assert_grad_accum_invariant(
            lambda mp, lb, na: self._loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_grad_accum(self):
        """MSE extensive property meets the grad-accum invariant."""
        p_A = _rnd(1, PROP_TASK_DIM)
        l_A = _rnd(1, PROP_TASK_DIM)
        p_B = _rnd(1, PROP_TASK_DIM)
        l_B = _rnd(1, PROP_TASK_DIM)
        self._run_invariant(self._make_loss("mse"), p_A, l_A, p_B, l_B)

    def test_mae_grad_accum(self):
        """MAE extensive property meets the grad-accum invariant."""
        p_A = _rnd(1, PROP_TASK_DIM)
        l_A = _rnd(1, PROP_TASK_DIM)
        p_B = _rnd(1, PROP_TASK_DIM)
        l_B = _rnd(1, PROP_TASK_DIM)
        self._run_invariant(self._make_loss("mae"), p_A, l_A, p_B, l_B)

    def test_smooth_mae_grad_accum(self):
        """smooth_mae extensive property meets the grad-accum invariant."""
        p_A = _rnd(1, PROP_TASK_DIM)
        l_A = _rnd(1, PROP_TASK_DIM)
        p_B = _rnd(1, PROP_TASK_DIM)
        l_B = _rnd(1, PROP_TASK_DIM)
        self._run_invariant(self._make_loss("smooth_mae"), p_A, l_A, p_B, l_B)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same extensive-property loss as no mask."""
        p = _rnd(2, PROP_TASK_DIM)
        l = _rnd(2, PROP_TASK_DIM)
        loss_obj = self._make_loss("mse")
        # PropertyLoss.call rebinds local `label` (no in-place mutation) so l is safe to reuse.
        with_mask = {PROP_VAR: p, "mask": np.ones((2, NB), dtype=np.float64)}
        without_mask = {PROP_VAR: p}
        loss_m, _ = loss_obj.call(1.0, NB, with_mask, {PROP_VAR: l})
        loss_nm, _ = loss_obj.call(1.0, NB, without_mask, {PROP_VAR: l})
        assert np.isclose(float(loss_m), float(loss_nm)), (
            f"all-ones mask must be no-op: {float(loss_m)} vs {float(loss_nm)}"
        )

    def test_torch_backend_matches_numpy(self):
        """The array-API path must work with torch tensors (pt_expt runs this).

        A padded batch is fed once as numpy and once as torch; the extensive
        per-frame normalization uses xp.sum(mask, axis=-1), which raises under
        the array_api_compat torch namespace if axis is passed positionally.
        """
        import pytest

        torch = pytest.importorskip("torch")
        # importing deepmd.pt sets a cuda:9999999 default-device trap; pin cpu.
        dev = "cpu"
        p = _rnd(2, PROP_TASK_DIM)
        lab = _rnd(2, PROP_TASK_DIM)
        loss_obj = self._make_loss("mse")
        np_pred = {PROP_VAR: p, "mask": _MASK_PAD}
        pt_pred = {
            PROP_VAR: torch.tensor(p, device=dev),
            "mask": torch.tensor(_MASK_PAD, device=dev),
        }
        loss_np, _ = loss_obj.call(1.0, NP, np_pred, {PROP_VAR: lab})
        loss_pt, _ = loss_obj.call(
            1.0, NP, pt_pred, {PROP_VAR: torch.tensor(lab, device=dev)}
        )
        assert np.isclose(float(loss_np), float(loss_pt)), (
            f"torch path must match numpy: {float(loss_np)} vs {float(loss_pt)}"
        )


class TestPropertyLossIntensiveUnaffectedByMask:
    """Intensive property loss must be unchanged whether or not mask is present."""

    def _make_loss(self):
        return PropertyLoss(
            task_dim=PROP_TASK_DIM,
            var_name=PROP_VAR,
            loss_func="mse",
            intensive=True,
        )

    def test_intensive_ignores_mask(self):
        """Intensive property: padded-batch loss == unmasked-batch loss."""
        p = _rnd(2, PROP_TASK_DIM)
        l = _rnd(2, PROP_TASK_DIM)
        loss_obj = self._make_loss()
        with_mask = {PROP_VAR: p, "mask": _MASK_PAD}
        without_mask = {PROP_VAR: p}
        loss_m, _ = loss_obj.call(1.0, NP, with_mask, {PROP_VAR: l})
        loss_nm, _ = loss_obj.call(1.0, NP, without_mask, {PROP_VAR: l})
        assert np.isclose(float(loss_m), float(loss_nm)), (
            f"intensive property must ignore mask: {float(loss_m)} vs {float(loss_nm)}"
        )


# ---------------------------------------------------------------------------
# Task 5: EnergySpinLoss -- energy (has_e), force_real (has_fr), virial (has_v)
# Leave force_mag / mask_mag (has_fm) COMPLETELY UNTOUCHED.
# ---------------------------------------------------------------------------

# Spin-specific test constants: NM magnetic atoms per frame (same count in
# both frames so that the pt fancy-index .view(nframes,-1,3) stays valid).
_NM = 2

_MASK_MAG_A = np.zeros((1, NA, 1), dtype=bool)  # [1, NA, 1]
_MASK_MAG_A[0, :_NM, 0] = True  # first NM atoms of frame A are magnetic

_MASK_MAG_B = np.zeros((1, NB, 1), dtype=bool)  # [1, NB, 1]
_MASK_MAG_B[0, :_NM, 0] = True  # first NM atoms of frame B are magnetic

# Padded batch (nf=2, nloc=NP): frame A padding slots are not magnetic.
_MASK_MAG_PAD_SPIN = np.zeros((2, NP, 1), dtype=bool)  # [2, NP, 1]
_MASK_MAG_PAD_SPIN[0, :_NM, 0] = True  # frame A: first NM real atoms are magnetic
_MASK_MAG_PAD_SPIN[1, :_NM, 0] = True  # frame B: first NM real atoms are magnetic

_MASK_PAD_SPIN = np.array(
    [[1.0] * NA + [0.0] * (NP - NA), [1.0] * NB], dtype=np.float64
)  # [2, NP]


def _full_spin_dicts(
    nf, nloc, energy_pred, energy_label, mask_mag, mask=None, **overrides
):
    """Build complete model_pred and label_dict for EnergySpinLoss.call.

    EnergySpinLoss.call accesses "energy" unconditionally (for xp namespace),
    so it must always be present.  All other keys are guarded by has_* flags.
    """
    model_pred = {
        "energy": energy_pred,  # [nf, 1]
        "force": np.zeros((nf, nloc, 3), dtype=np.float64),
        "force_mag": np.zeros((nf, nloc, 3), dtype=np.float64),
        "mask_mag": mask_mag,  # [nf, nloc, 1]
        "virial": np.zeros((nf, 9), dtype=np.float64),
    }
    label_dict = {
        "energy": energy_label,  # [nf, 1]
        "force": np.zeros((nf, nloc, 3), dtype=np.float64),
        "force_mag": np.zeros((nf, nloc, 3), dtype=np.float64),
        "virial": np.zeros((nf, 9), dtype=np.float64),
        "find_energy": 1.0,
        "find_force": 0.0,
        "find_force_mag": 0.0,
        "find_virial": 0.0,
    }
    if mask is not None:
        model_pred["mask"] = mask
    model_pred.update({k: v for k, v in overrides.items() if k in model_pred})
    label_dict.update({k: v for k, v in overrides.items() if k in label_dict})
    return model_pred, label_dict


class TestDPModelEnerSpinLossEnerGradAccum:
    """Idiom 2 (extensive) for the energy term in EnergySpinLoss.

    Covers: mse (norm_exp=1 and 2); plus non-mixed no-op.
    """

    def _make_loss(self, intensive=False):
        return EnergySpinLossDPModel(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=0.0,
            limit_pref_fm=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            intensive_ener_virial=intensive,
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        loss, _ = loss_obj.call(1.0, natoms, model_pred, label)
        return float(loss)

    def _run_invariant(self, loss_obj, e_A, e_A_hat, e_B, e_B_hat):
        def make_A():
            p, l = _full_spin_dicts(
                1,
                NA,
                e_A,
                e_A_hat,
                _MASK_MAG_A,
                mask=np.ones((1, NA), dtype=np.float64),
            )
            return p, l, NA

        def make_B():
            p, l = _full_spin_dicts(
                1,
                NB,
                e_B,
                e_B_hat,
                _MASK_MAG_B,
                mask=np.ones((1, NB), dtype=np.float64),
            )
            return p, l, NB

        def make_padded():
            e_pad = np.concatenate([e_A, e_B], axis=0)  # [2, 1]
            e_hat_pad = np.concatenate([e_A_hat, e_B_hat], axis=0)
            p, l = _full_spin_dicts(
                2, NP, e_pad, e_hat_pad, _MASK_MAG_PAD_SPIN, mask=_MASK_PAD_SPIN
            )
            return p, l, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: self._loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_non_intensive_grad_accum(self):
        """Spin energy MSE norm_exp=1 meets the grad-accum invariant."""
        e_A, e_B = _rnd(1, 1), _rnd(1, 1)
        e_A_hat, e_B_hat = _rnd(1, 1), _rnd(1, 1)
        self._run_invariant(
            self._make_loss(intensive=False), e_A, e_A_hat, e_B, e_B_hat
        )

    def test_mse_intensive_grad_accum(self):
        """Spin energy MSE norm_exp=2 meets the grad-accum invariant."""
        e_A, e_B = _rnd(1, 1), _rnd(1, 1)
        e_A_hat, e_B_hat = _rnd(1, 1), _rnd(1, 1)
        self._run_invariant(self._make_loss(intensive=True), e_A, e_A_hat, e_B, e_B_hat)

    def test_mae_grad_accum(self):
        """Spin energy MAE meets the grad-accum invariant."""
        e_A, e_B = _rnd(1, 1), _rnd(1, 1)
        e_A_hat, e_B_hat = _rnd(1, 1), _rnd(1, 1)
        loss_obj = EnergySpinLossDPModel(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=0.0,
            limit_pref_fm=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            loss_func="mae",
        )
        self._run_invariant(loss_obj, e_A, e_A_hat, e_B, e_B_hat)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same energy loss as no mask."""
        e = _rnd(1, 1)
        e_hat = _rnd(1, 1)
        loss_obj = self._make_loss()
        p_mask, l_mask = _full_spin_dicts(
            1, NP, e, e_hat, _MASK_MAG_B, mask=np.ones((1, NP), dtype=np.float64)
        )
        p_nm, l_nm = _full_spin_dicts(1, NP, e, e_hat, _MASK_MAG_B)
        loss_m = self._loss_fn(loss_obj, p_mask, l_mask, NP)
        loss_nm = self._loss_fn(loss_obj, p_nm, l_nm, NP)
        assert np.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m} vs {loss_nm}"
        )


class TestDPModelEnerSpinLossForceRealGradAccum:
    """Idiom 1 (per-atom masked mean, ncomp=3) for force_real in EnergySpinLoss.

    Covers: mse; plus non-mixed no-op.
    """

    def _make_loss(self):
        return EnergySpinLossDPModel(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_fr=1.0,
            limit_pref_fr=1.0,
            start_pref_fm=0.0,
            limit_pref_fm=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        loss, _ = loss_obj.call(1.0, natoms, model_pred, label)
        return float(loss)

    def _run_invariant(self, loss_obj, f_A, f_A_hat, f_B, f_B_hat):
        def make_A():
            p, l = _full_spin_dicts(
                1,
                NA,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                _MASK_MAG_A,
                mask=np.ones((1, NA), dtype=np.float64),
            )
            p["force"] = f_A[None]  # [1, NA, 3]
            l["force"] = f_A_hat[None]
            l["find_force"] = 1.0
            return p, l, NA

        def make_B():
            p, l = _full_spin_dicts(
                1,
                NB,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                _MASK_MAG_B,
                mask=np.ones((1, NB), dtype=np.float64),
            )
            p["force"] = f_B[None]
            l["force"] = f_B_hat[None]
            l["find_force"] = 1.0
            return p, l, NB

        def make_padded():
            f_A_pad = np.zeros((NP, 3), dtype=np.float64)
            f_A_pad[:NA] = f_A
            f_A_hat_pad = np.zeros((NP, 3), dtype=np.float64)
            f_A_hat_pad[:NA] = f_A_hat
            f_pad = np.stack([f_A_pad, f_B], axis=0)  # [2, NP, 3]
            f_hat_pad = np.stack([f_A_hat_pad, f_B_hat], axis=0)
            p, l = _full_spin_dicts(
                2,
                NP,
                np.zeros((2, 1)),
                np.zeros((2, 1)),
                _MASK_MAG_PAD_SPIN,
                mask=_MASK_PAD_SPIN,
            )
            p["force"] = f_pad
            l["force"] = f_hat_pad
            l["find_force"] = 1.0
            return p, l, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: self._loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_grad_accum(self):
        """Force_real MSE meets the grad-accum invariant."""
        f_A = _rnd(NA, 3)
        f_A_hat = _rnd(NA, 3)
        f_B = _rnd(NB, 3)
        f_B_hat = _rnd(NB, 3)
        self._run_invariant(self._make_loss(), f_A, f_A_hat, f_B, f_B_hat)

    def test_mae_grad_accum(self):
        """Force_real MAE meets the grad-accum invariant."""
        f_A = _rnd(NA, 3)
        f_A_hat = _rnd(NA, 3)
        f_B = _rnd(NB, 3)
        f_B_hat = _rnd(NB, 3)
        loss_obj = EnergySpinLossDPModel(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_fr=1.0,
            limit_pref_fr=1.0,
            start_pref_fm=0.0,
            limit_pref_fm=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            loss_func="mae",
        )
        self._run_invariant(loss_obj, f_A, f_A_hat, f_B, f_B_hat)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same force_real loss as no mask."""
        f = _rnd(NP, 3)
        f_hat = _rnd(NP, 3)
        loss_obj = self._make_loss()
        p_mask, l_mask = _full_spin_dicts(
            1,
            NP,
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            _MASK_MAG_B,
            mask=np.ones((1, NP), dtype=np.float64),
        )
        p_mask["force"] = f[None]
        l_mask["force"] = f_hat[None]
        l_mask["find_force"] = 1.0
        p_nm, l_nm = _full_spin_dicts(
            1, NP, np.zeros((1, 1)), np.zeros((1, 1)), _MASK_MAG_B
        )
        p_nm["force"] = f[None]
        l_nm["force"] = f_hat[None]
        l_nm["find_force"] = 1.0
        loss_m = self._loss_fn(loss_obj, p_mask, l_mask, NP)
        loss_nm = self._loss_fn(loss_obj, p_nm, l_nm, NP)
        assert np.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m} vs {loss_nm}"
        )


class TestDPModelEnerSpinLossVirialGradAccum:
    """Idiom 2 (extensive, k=9) for virial in EnergySpinLoss.

    Covers: mse (norm_exp=1 and 2); plus non-mixed no-op.
    """

    def _make_loss(self, intensive=False):
        return EnergySpinLossDPModel(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=0.0,
            limit_pref_fm=0.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            intensive_ener_virial=intensive,
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        loss, _ = loss_obj.call(1.0, natoms, model_pred, label)
        return float(loss)

    def _run_invariant(self, loss_obj, v_A, v_A_hat, v_B, v_B_hat):
        def make_A():
            p, l = _full_spin_dicts(
                1,
                NA,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                _MASK_MAG_A,
                mask=np.ones((1, NA), dtype=np.float64),
            )
            p["virial"] = v_A[None]  # [1, 9]
            l["virial"] = v_A_hat[None]
            l["find_virial"] = 1.0
            return p, l, NA

        def make_B():
            p, l = _full_spin_dicts(
                1,
                NB,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                _MASK_MAG_B,
                mask=np.ones((1, NB), dtype=np.float64),
            )
            p["virial"] = v_B[None]
            l["virial"] = v_B_hat[None]
            l["find_virial"] = 1.0
            return p, l, NB

        def make_padded():
            v_pad = np.stack([v_A, v_B], axis=0)  # [2, 9]
            v_hat_pad = np.stack([v_A_hat, v_B_hat], axis=0)
            p, l = _full_spin_dicts(
                2,
                NP,
                np.zeros((2, 1)),
                np.zeros((2, 1)),
                _MASK_MAG_PAD_SPIN,
                mask=_MASK_PAD_SPIN,
            )
            p["virial"] = v_pad
            l["virial"] = v_hat_pad
            l["find_virial"] = 1.0
            return p, l, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: self._loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_non_intensive_grad_accum(self):
        """Spin virial MSE norm_exp=1 meets the grad-accum invariant."""
        v_A, v_B = _rnd(9), _rnd(9)
        v_A_hat, v_B_hat = _rnd(9), _rnd(9)
        self._run_invariant(
            self._make_loss(intensive=False), v_A, v_A_hat, v_B, v_B_hat
        )

    def test_mse_intensive_grad_accum(self):
        """Spin virial MSE norm_exp=2 meets the grad-accum invariant."""
        v_A, v_B = _rnd(9), _rnd(9)
        v_A_hat, v_B_hat = _rnd(9), _rnd(9)
        self._run_invariant(self._make_loss(intensive=True), v_A, v_A_hat, v_B, v_B_hat)

    def test_mae_grad_accum(self):
        """Spin virial MAE meets the grad-accum invariant."""
        v_A, v_B = _rnd(9), _rnd(9)
        v_A_hat, v_B_hat = _rnd(9), _rnd(9)
        loss_obj = EnergySpinLossDPModel(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=0.0,
            limit_pref_fm=0.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            loss_func="mae",
        )
        self._run_invariant(loss_obj, v_A, v_A_hat, v_B, v_B_hat)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same virial loss as no mask."""
        v = _rnd(9)
        v_hat = _rnd(9)
        loss_obj = self._make_loss()
        p_mask, l_mask = _full_spin_dicts(
            1,
            NP,
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            _MASK_MAG_B,
            mask=np.ones((1, NP), dtype=np.float64),
        )
        p_mask["virial"] = v[None]
        l_mask["virial"] = v_hat[None]
        l_mask["find_virial"] = 1.0
        p_nm, l_nm = _full_spin_dicts(
            1, NP, np.zeros((1, 1)), np.zeros((1, 1)), _MASK_MAG_B
        )
        p_nm["virial"] = v[None]
        l_nm["virial"] = v_hat[None]
        l_nm["find_virial"] = 1.0
        loss_m = self._loss_fn(loss_obj, p_mask, l_mask, NP)
        loss_nm = self._loss_fn(loss_obj, p_nm, l_nm, NP)
        assert np.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m} vs {loss_nm}"
        )


class TestDPModelEnerSpinLossForceMagUnchanged:
    """Guard: the padding mask must NOT affect the force_mag / mask_mag term.

    The force_mag path uses mask_mag (spin virtual-atom mask), which is a
    completely separate concept from the padding mask model_dict["mask"].
    After the Task-5 implementation, presenting a padding mask must leave
    the force_mag loss bit-identical.
    """

    def _make_loss(self):
        return EnergySpinLossDPModel(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=1.0,
            limit_pref_fm=1.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
        )

    def test_padding_mask_does_not_affect_force_mag(self):
        """force_mag loss is bit-identical with and without padding mask."""
        fm = _rnd(2, NP, 3)
        fm_hat = _rnd(2, NP, 3)
        loss_obj = self._make_loss()

        def _run(with_mask):
            pred = {
                "energy": np.zeros((2, 1), dtype=np.float64),
                "force_mag": fm,
                "mask_mag": _MASK_MAG_PAD_SPIN,
            }
            lbl = {
                "force_mag": fm_hat,
                "find_force_mag": 1.0,
                "find_energy": 0.0,
                "find_force": 0.0,
                "find_virial": 0.0,
            }
            if with_mask:
                pred["mask"] = _MASK_PAD_SPIN
            loss, _ = loss_obj.call(1.0, NP, pred, lbl)
            return float(loss)

        loss_with = _run(True)
        loss_without = _run(False)
        assert np.isclose(loss_with, loss_without), (
            f"force_mag loss must be unchanged by padding mask: "
            f"{loss_with} vs {loss_without}"
        )


# ---------------------------------------------------------------------------
# Part A: EnergySpinLoss atom_ener (has_ae) grad-accum invariant
# ---------------------------------------------------------------------------


class TestDPModelEnerSpinLossAtomEnerGradAccum:
    """Idiom 1 (per-atom masked mean, ncomp=1) for atom_ener in EnergySpinLoss.

    RED before the Part-A has_ae mask fix; GREEN after.
    """

    def _make_loss(self, loss_func="mse"):
        return EnergySpinLossDPModel(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=0.0,
            limit_pref_fm=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_ae=1.0,
            limit_pref_ae=1.0,
            loss_func=loss_func,
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        loss, _ = loss_obj.call(1.0, natoms, model_pred, label)
        return float(loss)

    def _run_invariant(self, loss_obj, ae_A, ae_A_hat, ae_B, ae_B_hat):
        def make_A():
            p, l = _full_spin_dicts(
                1,
                NA,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                _MASK_MAG_A,
                mask=np.ones((1, NA), dtype=np.float64),
            )
            p["atom_energy"] = ae_A[None]  # [1, NA, 1]
            l["atom_ener"] = ae_A_hat[None]
            l["find_atom_ener"] = 1.0
            return p, l, NA

        def make_B():
            p, l = _full_spin_dicts(
                1,
                NB,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                _MASK_MAG_B,
                mask=np.ones((1, NB), dtype=np.float64),
            )
            p["atom_energy"] = ae_B[None]
            l["atom_ener"] = ae_B_hat[None]
            l["find_atom_ener"] = 1.0
            return p, l, NB

        def make_padded():
            ae_pad = _padded_atom(ae_A, ae_B, 1)  # [2, NP, 1]
            ae_hat_pad = _padded_atom(ae_A_hat, ae_B_hat, 1)
            p, l = _full_spin_dicts(
                2,
                NP,
                np.zeros((2, 1)),
                np.zeros((2, 1)),
                _MASK_MAG_PAD_SPIN,
                mask=_MASK_PAD_SPIN,
            )
            p["atom_energy"] = ae_pad
            l["atom_ener"] = ae_hat_pad
            l["find_atom_ener"] = 1.0
            return p, l, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: self._loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_grad_accum(self):
        """Spin atom_ener MSE meets the grad-accum invariant."""
        ae_A = _rnd(NA, 1)
        ae_A_hat = _rnd(NA, 1)
        ae_B = _rnd(NB, 1)
        ae_B_hat = _rnd(NB, 1)
        self._run_invariant(self._make_loss("mse"), ae_A, ae_A_hat, ae_B, ae_B_hat)

    def test_mae_grad_accum(self):
        """Spin atom_ener MAE meets the grad-accum invariant."""
        ae_A = _rnd(NA, 1)
        ae_A_hat = _rnd(NA, 1)
        ae_B = _rnd(NB, 1)
        ae_B_hat = _rnd(NB, 1)
        self._run_invariant(self._make_loss("mae"), ae_A, ae_A_hat, ae_B, ae_B_hat)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same atom_ener spin loss as no mask."""
        ae = _rnd(NP, 1)
        ae_hat = _rnd(NP, 1)
        loss_obj = self._make_loss()
        p_mask, l_mask = _full_spin_dicts(
            1,
            NP,
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            _MASK_MAG_B,
            mask=np.ones((1, NP), dtype=np.float64),
        )
        p_mask["atom_energy"] = ae[None]
        l_mask["atom_ener"] = ae_hat[None]
        l_mask["find_atom_ener"] = 1.0
        p_nm, l_nm = _full_spin_dicts(
            1, NP, np.zeros((1, 1)), np.zeros((1, 1)), _MASK_MAG_B
        )
        p_nm["atom_energy"] = ae[None]
        l_nm["atom_ener"] = ae_hat[None]
        l_nm["find_atom_ener"] = 1.0
        loss_m = self._loss_fn(loss_obj, p_mask, l_mask, NP)
        loss_nm = self._loss_fn(loss_obj, p_nm, l_nm, NP)
        assert np.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m} vs {loss_nm}"
        )


# ---------------------------------------------------------------------------
# gen_force (has_gf) grad-accum invariant
# Ghost-atom forces are masked before projection, so the invariant is met.
# ---------------------------------------------------------------------------

_NGEN = 2  # number of generalized coordinates for tests


class TestDPModelEnergyLossGenForceGradAccum:
    """gen_force (has_gf) excludes ghost atoms via force masking before projection.

    The projected gen_force = sum_i(drdq_ij * masked_f_i) is frame-decomposable,
    so mean(square(diff)) over [nf, ngen] already satisfies the invariant.
    Expected GREEN immediately (no fix needed for gen_force).
    """

    def _make_loss(self):
        return EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_ae=0.0,
            limit_pref_ae=0.0,
            start_pref_pf=0.0,
            limit_pref_pf=0.0,
            start_pref_gf=1.0,
            limit_pref_gf=1.0,
            numb_generalized_coord=_NGEN,
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        loss, _ = loss_obj.call(1.0, natoms, model_pred, label)
        return float(loss)

    def _run_invariant(self, loss_obj, f_A, f_A_hat, drdq_A, f_B, f_B_hat, drdq_B):
        def make_A():
            p, l = _full_ener_dicts(
                1,
                NA,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                mask=np.ones((1, NA), dtype=np.float64),
            )
            p["force"] = f_A[None]  # [1, NA, 3]
            l["force"] = f_A_hat[None]
            l["find_force"] = 1.0
            l["drdq"] = drdq_A[None]  # [1, NA*3, NGEN]
            l["find_drdq"] = 1.0
            return p, l, NA

        def make_B():
            p, l = _full_ener_dicts(
                1,
                NB,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                mask=np.ones((1, NB), dtype=np.float64),
            )
            p["force"] = f_B[None]
            l["force"] = f_B_hat[None]
            l["find_force"] = 1.0
            l["drdq"] = drdq_B[None]  # [1, NB*3, NGEN]
            l["find_drdq"] = 1.0
            return p, l, NB

        def make_padded():
            f_pad = _padded_force(f_A, f_B)  # [2, NP, 3]
            f_hat_pad = _padded_force(f_A_hat, f_B_hat)
            # drdq ghost-atom slots are zero (ghost forces also zero, so no contribution)
            drdq_A_pad = np.zeros((NP * 3, _NGEN), dtype=np.float64)
            drdq_A_pad[: NA * 3] = drdq_A
            drdq_pad = np.stack([drdq_A_pad, drdq_B], axis=0)  # [2, NP*3, NGEN]
            p, l = _full_ener_dicts(
                2, NP, np.zeros((2, 1)), np.zeros((2, 1)), mask=_MASK_PAD
            )
            p["force"] = f_pad
            l["force"] = f_hat_pad
            l["find_force"] = 1.0
            l["drdq"] = drdq_pad
            l["find_drdq"] = 1.0
            return p, l, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: self._loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_grad_accum(self):
        """gen_force MSE meets the grad-accum invariant (GREEN: already correct)."""
        f_A = _rnd(NA, 3)
        f_A_hat = _rnd(NA, 3)
        drdq_A = _rnd(NA * 3, _NGEN)
        f_B = _rnd(NB, 3)
        f_B_hat = _rnd(NB, 3)
        drdq_B = _rnd(NB * 3, _NGEN)
        self._run_invariant(
            self._make_loss(), f_A, f_A_hat, drdq_A, f_B, f_B_hat, drdq_B
        )


# ---------------------------------------------------------------------------
# force_mag MSE grad-accum invariant (NM equal across frames, ghost atoms non-magnetic)
#
# NOTE: force_mag MAE (dpmodel) uses a plain mean over frames/atoms/xyz (like
# force_mag MSE and force_real MAE), so it meets the grad-accum invariant.
# Ghost atoms are correctly excluded (mask_mag=0 there).
# ---------------------------------------------------------------------------


class TestDPModelEnerSpinLossForceMagMSEGradAccum:
    """force_mag MSE meets the grad-accum invariant when ghost atoms are non-magnetic.

    With equal magnetic-atom counts (NM) across frames and ghost atoms having
    mask_mag=0, the global normalization n_valid = NM*nf correctly decomposes
    per-frame, giving loss_pad == 0.5*(loss_A + loss_B).
    Expected GREEN immediately (no fix needed for MSE force_mag).
    """

    def _make_loss(self):
        return EnergySpinLossDPModel(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=1.0,
            limit_pref_fm=1.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        loss, _ = loss_obj.call(1.0, natoms, model_pred, label)
        return float(loss)

    def test_mse_grad_accum(self):
        """force_mag MSE (NM equal per frame, ghost atoms non-magnetic) meets invariant."""
        fm_A = _rnd(NA, 3)
        fm_A_hat = _rnd(NA, 3)
        fm_B = _rnd(NB, 3)
        fm_B_hat = _rnd(NB, 3)

        def make_A():
            # Frame A: NA=3 real atoms, first NM=2 are magnetic
            fm_A_full = np.zeros((NA, 3), dtype=np.float64)
            fm_A_full[:_NM] = fm_A[:_NM]
            fm_A_hat_full = np.zeros((NA, 3), dtype=np.float64)
            fm_A_hat_full[:_NM] = fm_A_hat[:_NM]
            pred = {
                "energy": np.zeros((1, 1), dtype=np.float64),
                "force_mag": fm_A_full[None],  # [1, NA, 3]
                "mask_mag": _MASK_MAG_A,
            }
            lbl = {
                "force_mag": fm_A_hat_full[None],
                "find_force_mag": 1.0,
                "find_energy": 0.0,
                "find_force": 0.0,
                "find_virial": 0.0,
            }
            return pred, lbl, NA

        def make_B():
            fm_B_full = np.zeros((NB, 3), dtype=np.float64)
            fm_B_full[:_NM] = fm_B[:_NM]
            fm_B_hat_full = np.zeros((NB, 3), dtype=np.float64)
            fm_B_hat_full[:_NM] = fm_B_hat[:_NM]
            pred = {
                "energy": np.zeros((1, 1), dtype=np.float64),
                "force_mag": fm_B_full[None],  # [1, NB, 3]
                "mask_mag": _MASK_MAG_B,
            }
            lbl = {
                "force_mag": fm_B_hat_full[None],
                "find_force_mag": 1.0,
                "find_energy": 0.0,
                "find_force": 0.0,
                "find_virial": 0.0,
            }
            return pred, lbl, NB

        def make_padded():
            # Pad frame A force_mag to NP width; ghost slots are zero (non-magnetic)
            fm_A_pad = np.zeros((NP, 3), dtype=np.float64)
            fm_A_pad[:_NM] = fm_A[:_NM]
            fm_A_hat_pad = np.zeros((NP, 3), dtype=np.float64)
            fm_A_hat_pad[:_NM] = fm_A_hat[:_NM]
            fm_B_pad = np.zeros((NP, 3), dtype=np.float64)
            fm_B_pad[:_NM] = fm_B[:_NM]
            fm_B_hat_pad = np.zeros((NP, 3), dtype=np.float64)
            fm_B_hat_pad[:_NM] = fm_B_hat[:_NM]
            fm_pad = np.stack([fm_A_pad, fm_B_pad], axis=0)  # [2, NP, 3]
            fm_hat_pad = np.stack([fm_A_hat_pad, fm_B_hat_pad], axis=0)
            pred = {
                "energy": np.zeros((2, 1), dtype=np.float64),
                "force_mag": fm_pad,
                "mask_mag": _MASK_MAG_PAD_SPIN,  # ghost atoms have mask_mag=False
                "mask": _MASK_PAD_SPIN,
            }
            lbl = {
                "force_mag": fm_hat_pad,
                "find_force_mag": 1.0,
                "find_energy": 0.0,
                "find_force": 0.0,
                "find_virial": 0.0,
            }
            return pred, lbl, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: self._loss_fn(self._make_loss(), mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )


class TestDPModelEnerSpinLossForceMagMAEGradAccum:
    """force_mag MAE (dpmodel) meets the grad-accum invariant.

    ``force_mag`` MAE now reduces with a plain mean over frames, magnetic atoms
    and xyz (same reduction as force_mag MSE and force_real MAE), so a padded
    ``[A+B]`` batch (nf=2) equals the mean of the two single-frame losses and the
    loss is batch-size independent.
    """

    def _make_loss(self):
        return EnergySpinLossDPModel(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=1.0,
            limit_pref_fm=1.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            loss_func="mae",
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        loss, _ = loss_obj.call(1.0, natoms, model_pred, label)
        return float(loss)

    def test_mae_grad_accum(self):
        """force_mag MAE meets the frame-average grad-accum invariant."""
        fm_A = _rnd(NA, 3)
        fm_A_hat = _rnd(NA, 3)
        fm_B = _rnd(NB, 3)
        fm_B_hat = _rnd(NB, 3)

        def make_A():
            fm_A_full = np.zeros((NA, 3), dtype=np.float64)
            fm_A_full[:_NM] = fm_A[:_NM]
            fm_A_hat_full = np.zeros((NA, 3), dtype=np.float64)
            fm_A_hat_full[:_NM] = fm_A_hat[:_NM]
            pred = {
                "energy": np.zeros((1, 1), dtype=np.float64),
                "force_mag": fm_A_full[None],  # [1, NA, 3]
                "mask_mag": _MASK_MAG_A,
            }
            lbl = {
                "force_mag": fm_A_hat_full[None],
                "find_force_mag": 1.0,
                "find_energy": 0.0,
                "find_force": 0.0,
                "find_virial": 0.0,
            }
            return pred, lbl, NA

        def make_B():
            fm_B_full = np.zeros((NB, 3), dtype=np.float64)
            fm_B_full[:_NM] = fm_B[:_NM]
            fm_B_hat_full = np.zeros((NB, 3), dtype=np.float64)
            fm_B_hat_full[:_NM] = fm_B_hat[:_NM]
            pred = {
                "energy": np.zeros((1, 1), dtype=np.float64),
                "force_mag": fm_B_full[None],  # [1, NB, 3]
                "mask_mag": _MASK_MAG_B,
            }
            lbl = {
                "force_mag": fm_B_hat_full[None],
                "find_force_mag": 1.0,
                "find_energy": 0.0,
                "find_force": 0.0,
                "find_virial": 0.0,
            }
            return pred, lbl, NB

        def make_padded():
            fm_A_pad = np.zeros((NP, 3), dtype=np.float64)
            fm_A_pad[:_NM] = fm_A[:_NM]
            fm_A_hat_pad = np.zeros((NP, 3), dtype=np.float64)
            fm_A_hat_pad[:_NM] = fm_A_hat[:_NM]
            fm_B_pad = np.zeros((NP, 3), dtype=np.float64)
            fm_B_pad[:_NM] = fm_B[:_NM]
            fm_B_hat_pad = np.zeros((NP, 3), dtype=np.float64)
            fm_B_hat_pad[:_NM] = fm_B_hat[:_NM]
            fm_pad = np.stack([fm_A_pad, fm_B_pad], axis=0)  # [2, NP, 3]
            fm_hat_pad = np.stack([fm_A_hat_pad, fm_B_hat_pad], axis=0)
            pred = {
                "energy": np.zeros((2, 1), dtype=np.float64),
                "force_mag": fm_pad,
                "mask_mag": _MASK_MAG_PAD_SPIN,  # ghost atoms have mask_mag=False
                "mask": _MASK_PAD_SPIN,
            }
            lbl = {
                "force_mag": fm_hat_pad,
                "find_force_mag": 1.0,
                "find_energy": 0.0,
                "find_force": 0.0,
                "find_virial": 0.0,
            }
            return pred, lbl, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: self._loss_fn(self._make_loss(), mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )


# ---------------------------------------------------------------------------
# Task 11 review fix: EnergySpinLoss flat-label reshape (data-loader shape)
# ---------------------------------------------------------------------------
# The data loader delivers per-atom vector labels FLAT: (nframes, natoms * 3)
# (see ``deepmd/utils/data.py:892``), not the canonical (nframes, natoms, 3)
# shape that ``masked_atom_mean`` requires.  ``EnergySpinLoss.call`` bridges
# this via ``xp.reshape(model_dict["force"], (-1, natoms, 3))`` (and the same
# for ``force_mag``).  Every test above constructs already-3D labels, so that
# reshape is a no-op there.  This test feeds a genuinely FLAT label and pins
# the reshape numerically.


class TestDPModelEnerSpinLossFlatLabelReshape:
    """Pin the flat-(nf, natoms*3) -> (nf, natoms, 3) label reshape.

    Asserts the loss computed from a flat ``(nf, natoms * 3)`` force /
    force_mag label equals the loss computed from the same data pre-reshaped
    to ``(nf, natoms, 3)``, with a non-zero force_mag term. This fails (shape
    error or wrong value) if the reshape in ``EnergySpinLoss.call`` is
    removed.
    """

    def _make_loss(self):
        return EnergySpinLossDPModel(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_fr=1.0,
            limit_pref_fr=1.0,
            start_pref_fm=1.0,
            limit_pref_fm=1.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
        )

    def test_flat_label_matches_3d_label(self):
        nf = 2
        force_pred = _rnd(nf, NP, 3)
        force_label_3d = _rnd(nf, NP, 3)
        force_mag_pred = _rnd(nf, NP, 3)
        force_mag_label_3d = _rnd(nf, NP, 3)
        mask_mag = _MASK_MAG_PAD_SPIN  # [2, NP, 1]

        loss_obj = self._make_loss()

        model_dict = {
            "energy": np.zeros((nf, 1), dtype=np.float64),
            "force": force_pred,
            "force_mag": force_mag_pred,
            "mask_mag": mask_mag,
            "virial": np.zeros((nf, 9), dtype=np.float64),
        }
        label_3d = {
            "energy": np.zeros((nf, 1), dtype=np.float64),
            "force": force_label_3d,
            "force_mag": force_mag_label_3d,
            "virial": np.zeros((nf, 9), dtype=np.float64),
            "find_energy": 0.0,
            "find_force": 1.0,
            "find_force_mag": 1.0,
            "find_virial": 0.0,
        }
        # The real, motivating shape: the data loader delivers atomic vector
        # labels flat -- (nf, natoms * 3) -- not (nf, natoms, 3).
        label_flat = dict(label_3d)
        label_flat["force"] = force_label_3d.reshape(nf, NP * 3)
        label_flat["force_mag"] = force_mag_label_3d.reshape(nf, NP * 3)

        loss_3d, more_loss_3d = loss_obj.call(1.0, NP, model_dict, label_3d)
        loss_flat, more_loss_flat = loss_obj.call(1.0, NP, model_dict, label_flat)

        assert np.isclose(float(loss_3d), float(loss_flat), rtol=1e-12, atol=1e-12), (
            f"flat-label loss must equal 3D-label loss: {loss_flat} vs {loss_3d}"
        )
        assert float(more_loss_flat["rmse_fm"]) != 0.0, (
            "force_mag loss term must be non-zero for this test to have teeth"
        )
        assert np.isclose(
            float(more_loss_flat["rmse_fm"]),
            float(more_loss_3d["rmse_fm"]),
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.isclose(
            float(more_loss_flat["rmse_fr"]),
            float(more_loss_3d["rmse_fr"]),
            rtol=1e-12,
            atol=1e-12,
        )
