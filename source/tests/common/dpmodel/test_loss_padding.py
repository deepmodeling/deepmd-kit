# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reusable grad-accumulation invariant harness for dpmodel loss tests.

This module provides ``assert_grad_accum_invariant`` for Tasks 2-5 that
verify the loss on a padded multi-frame batch equals mean(per_frame_loss).

The dpmodel losses accept numpy arrays (via the array_api_compat backend).

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
