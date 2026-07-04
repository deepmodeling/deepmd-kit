# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for mixed_type loss padding-mask support in the pt backend.

Task 1: verify that TaskLoss._inject_atom_mask correctly recovers the per-atom
mask from atype (ghost atoms have atype < 0) so that later tasks can exclude
them from loss reductions.

Harness
-------
assert_grad_accum_invariant  -- reusable by Tasks 2-5 to check the
    grad-accumulation invariant: loss on a padded multi-frame batch must equal
    mean_over_frames(per_frame_loss).
"""

import numpy as np
import torch

from deepmd.pt.loss.dos import (
    DOSLoss,
)
from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.loss.tensor import (
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
        Signature ``(model_pred, label, natoms) -> scalar torch.Tensor``.
    make_batch_A : callable
        Returns ``(model_pred, label, natoms)`` for frame A alone (1 frame, NA atoms).
    make_batch_B : callable
        Returns ``(model_pred, label, natoms)`` for frame B alone (1 frame, NB atoms).
    make_padded_batch : callable
        Returns ``(model_pred, label, natoms)`` for the 2-frame padded batch
        (nf=2, nloc=NP; frame A is padded with NP-NA ghost rows).
    rtol : float
        Relative tolerance for ``torch.isclose``.
    atol : float
        Absolute tolerance for ``torch.isclose``.
    """
    pred_A, label_A, natoms_A = make_batch_A()
    pred_B, label_B, natoms_B = make_batch_B()
    pred_pad, label_pad, natoms_pad = make_padded_batch()

    loss_A = loss_fn(pred_A, label_A, natoms_A)
    loss_B = loss_fn(pred_B, label_B, natoms_B)
    ref = 0.5 * (loss_A + loss_B)

    loss_pad = loss_fn(pred_pad, label_pad, natoms_pad)

    assert torch.isclose(loss_pad, ref, rtol=rtol, atol=atol), (
        f"Grad-accum invariant violated: padded_loss={loss_pad.item():.8f}, "
        f"ref={ref.item():.8f}, diff={abs(loss_pad.item() - ref.item()):.2e}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
NUMB_DOS = 4
TENSOR_SIZE = 3


def _rnd_t(*shape):
    """Create a random CPU tensor (device must be explicit to avoid default-device trap)."""
    return torch.tensor(RNG.standard_normal(shape), dtype=torch.float64, device="cpu")


class _MockModel:
    """Callable that ignores inputs and returns a fixed model_pred dict.

    Used in pt loss tests to bypass the actual model forward pass while still
    exercising the loss computation code inside DOSLoss.forward / TensorLoss.forward.
    The mask is pre-populated in ``pred`` so ``_inject_atom_mask`` leaves it alone.
    """

    def __init__(self, pred: dict):
        self._pred = pred

    def __call__(self, **kwargs):
        return dict(self._pred)  # shallow copy; _inject_atom_mask may mutate it


# ---------------------------------------------------------------------------
# Task 2: DOSLoss -- atomic (ados / acdf) and global (dos / cdf)
# ---------------------------------------------------------------------------


class TestPTDOSLossAtomicGradAccum:
    """Per-frame masked mean (idiom 1) for atomic dos / acdf terms.

    _loss_fn calls the ACTUAL pt DOSLoss.forward() via a mock model so that
    RED/GREEN transitions directly reflect changes to deepmd/pt/loss/dos.py.
    """

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

    def _loss_fn(self, model_pred, label, natoms):
        loss_obj = self._make_loss()
        _, loss, _ = loss_obj.forward(
            input_dict={},  # no atype; mask already in model_pred → not re-injected
            model=_MockModel(model_pred),
            label=label,
            natoms=natoms,
            learning_rate=1.0,
        )
        return loss

    def test_ados_grad_accum_invariant(self):
        """Atomic dos per-frame masked mean meets the grad-accum invariant."""
        pred_A = _rnd_t(NA, NUMB_DOS)
        label_A = _rnd_t(NA, NUMB_DOS)
        pred_B = _rnd_t(NB, NUMB_DOS)
        label_B = _rnd_t(NB, NUMB_DOS)

        def make_A():
            return (
                {
                    "atom_dos": pred_A,
                    "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
                },
                {"atom_dos": label_A, "find_atom_dos": 1.0},
                NA,
            )

        def make_B():
            return (
                {
                    "atom_dos": pred_B,
                    "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
                },
                {"atom_dos": label_B, "find_atom_dos": 1.0},
                NB,
            )

        def make_padded():
            pred_A_pad = torch.zeros(NP, NUMB_DOS, dtype=torch.float64, device="cpu")
            pred_A_pad[:NA] = pred_A
            label_A_pad = torch.zeros(NP, NUMB_DOS, dtype=torch.float64, device="cpu")
            label_A_pad[:NA] = label_A
            mask_A = torch.tensor(
                [[1.0] * NA + [0.0] * (NP - NA)], dtype=torch.float64, device="cpu"
            )
            mask_B = torch.ones(1, NB, dtype=torch.float64, device="cpu")
            atom_dos_pad = torch.cat([pred_A_pad, pred_B], dim=0)
            atom_dos_label = torch.cat([label_A_pad, label_B], dim=0)
            mask_pad = torch.cat([mask_A, mask_B], dim=0)
            return (
                {"atom_dos": atom_dos_pad, "mask": mask_pad},
                {"atom_dos": atom_dos_label, "find_atom_dos": 1.0},
                NP,
            )

        assert_grad_accum_invariant(self._loss_fn, make_A, make_B, make_padded)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same loss as no mask (non-mixed batch)."""
        pred = _rnd_t(NB, NUMB_DOS)
        label = _rnd_t(NB, NUMB_DOS)
        with_mask = {
            "atom_dos": pred,
            "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
        }
        without_mask = {"atom_dos": pred}
        label_dict = {"atom_dos": label, "find_atom_dos": 1.0}
        loss_m = self._loss_fn(with_mask, label_dict, NB)
        loss_nm = self._loss_fn(without_mask, label_dict, NB)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


class TestPTDOSLossGlobalGradAccum:
    """Plain mean (idiom 3) for global dos / cdf terms.

    _loss_fn calls the ACTUAL pt DOSLoss.forward() via a mock model.
    """

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
        loss_obj = self._make_loss()
        _, loss, _ = loss_obj.forward(
            input_dict={},
            model=_MockModel(model_pred),
            label=label,
            natoms=natoms,
            learning_rate=1.0,
        )
        return loss

    def test_dos_grad_accum_invariant(self):
        """Global dos plain mean meets the grad-accum invariant."""
        pred_A = _rnd_t(1, NUMB_DOS)
        label_A = _rnd_t(1, NUMB_DOS)
        pred_B = _rnd_t(1, NUMB_DOS)
        label_B = _rnd_t(1, NUMB_DOS)

        def make_A():
            return (
                {
                    "dos": pred_A,
                    "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
                },
                {"dos": label_A, "find_dos": 1.0},
                NA,
            )

        def make_B():
            return (
                {
                    "dos": pred_B,
                    "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
                },
                {"dos": label_B, "find_dos": 1.0},
                NB,
            )

        def make_padded():
            pred_pad = torch.cat([pred_A, pred_B], dim=0)
            label_pad = torch.cat([label_A, label_B], dim=0)
            mask_pad = torch.tensor(
                [[1.0] * NA + [0.0] * (NP - NA), [1.0] * NB],
                dtype=torch.float64,
                device="cpu",
            )
            return (
                {"dos": pred_pad, "mask": mask_pad},
                {"dos": label_pad, "find_dos": 1.0},
                NP,
            )

        assert_grad_accum_invariant(self._loss_fn, make_A, make_B, make_padded)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same loss as no mask (non-mixed batch)."""
        pred = _rnd_t(2, NUMB_DOS)
        label = _rnd_t(2, NUMB_DOS)
        with_mask = {
            "dos": pred,
            "mask": torch.ones(2, NB, dtype=torch.float64, device="cpu"),
        }
        without_mask = {"dos": pred}
        label_dict = {"dos": label, "find_dos": 1.0}
        loss_m = self._loss_fn(with_mask, label_dict, NB)
        loss_nm = self._loss_fn(without_mask, label_dict, NB)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


# ---------------------------------------------------------------------------
# Task 2: TensorLoss -- local and global tensor
# ---------------------------------------------------------------------------


class TestPTTensorLossLocalGradAccum:
    """Per-frame masked mean (idiom 1) for local tensor term.

    _loss_fn calls the ACTUAL pt TensorLoss.forward() via a mock model.
    """

    def _make_loss(self):
        return TensorLoss(
            tensor_name="dipole",
            tensor_size=TENSOR_SIZE,
            label_name="dipole",
            pref_atomic=1.0,
            pref=0.0,
        )

    def _loss_fn(self, model_pred, label, natoms):
        loss_obj = self._make_loss()
        _, loss, _ = loss_obj.forward(
            input_dict={},
            model=_MockModel(model_pred),
            label=label,
            natoms=natoms,
            learning_rate=1.0,
        )
        return loss

    def test_local_grad_accum_invariant(self):
        """Local tensor per-frame masked mean meets the grad-accum invariant."""
        pred_A = _rnd_t(NA, TENSOR_SIZE)
        label_A = _rnd_t(NA, TENSOR_SIZE)
        pred_B = _rnd_t(NB, TENSOR_SIZE)
        label_B = _rnd_t(NB, TENSOR_SIZE)

        def make_A():
            return (
                {
                    "dipole": pred_A,
                    "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
                },
                {"atom_dipole": label_A, "find_atom_dipole": 1.0},
                NA,
            )

        def make_B():
            return (
                {
                    "dipole": pred_B,
                    "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
                },
                {"atom_dipole": label_B, "find_atom_dipole": 1.0},
                NB,
            )

        def make_padded():
            pred_A_pad = torch.zeros(NP, TENSOR_SIZE, dtype=torch.float64, device="cpu")
            pred_A_pad[:NA] = pred_A
            label_A_pad = torch.zeros(
                NP, TENSOR_SIZE, dtype=torch.float64, device="cpu"
            )
            label_A_pad[:NA] = label_A
            mask_A = torch.tensor(
                [[1.0] * NA + [0.0] * (NP - NA)], dtype=torch.float64, device="cpu"
            )
            mask_B = torch.ones(1, NB, dtype=torch.float64, device="cpu")
            dipole_pad = torch.cat([pred_A_pad, pred_B], dim=0)
            label_pad = torch.cat([label_A_pad, label_B], dim=0)
            mask_pad = torch.cat([mask_A, mask_B], dim=0)
            return (
                {"dipole": dipole_pad, "mask": mask_pad},
                {"atom_dipole": label_pad, "find_atom_dipole": 1.0},
                NP,
            )

        assert_grad_accum_invariant(self._loss_fn, make_A, make_B, make_padded)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same loss as no mask (non-mixed batch)."""
        pred = _rnd_t(NB, TENSOR_SIZE)
        label = _rnd_t(NB, TENSOR_SIZE)
        with_mask = {
            "dipole": pred,
            "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
        }
        without_mask = {"dipole": pred}
        label_dict = {"atom_dipole": label, "find_atom_dipole": 1.0}
        loss_m = self._loss_fn(with_mask, label_dict, NB)
        loss_nm = self._loss_fn(without_mask, label_dict, NB)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


class TestPTTensorLossGlobalGradAccum:
    """Plain mean (idiom 3) for global tensor term.

    _loss_fn calls the ACTUAL pt TensorLoss.forward() via a mock model.
    """

    def _make_loss(self):
        return TensorLoss(
            tensor_name="dipole",
            tensor_size=TENSOR_SIZE,
            label_name="dipole",
            pref_atomic=0.0,
            pref=1.0,
        )

    def _loss_fn(self, model_pred, label, natoms):
        loss_obj = self._make_loss()
        _, loss, _ = loss_obj.forward(
            input_dict={},
            model=_MockModel(model_pred),
            label=label,
            natoms=natoms,
            learning_rate=1.0,
        )
        return loss

    def test_global_grad_accum_invariant(self):
        """Global tensor plain mean meets the grad-accum invariant."""
        pred_A = _rnd_t(1, TENSOR_SIZE)
        label_A = _rnd_t(1, TENSOR_SIZE)
        pred_B = _rnd_t(1, TENSOR_SIZE)
        label_B = _rnd_t(1, TENSOR_SIZE)

        def make_A():
            return (
                {
                    "global_dipole": pred_A,
                    "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
                },
                {"dipole": label_A, "find_dipole": 1.0},
                NA,
            )

        def make_B():
            return (
                {
                    "global_dipole": pred_B,
                    "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
                },
                {"dipole": label_B, "find_dipole": 1.0},
                NB,
            )

        def make_padded():
            pred_pad = torch.cat([pred_A, pred_B], dim=0)
            label_pad = torch.cat([label_A, label_B], dim=0)
            mask_pad = torch.tensor(
                [[1.0] * NA + [0.0] * (NP - NA), [1.0] * NB],
                dtype=torch.float64,
                device="cpu",
            )
            return (
                {"global_dipole": pred_pad, "mask": mask_pad},
                {"dipole": label_pad, "find_dipole": 1.0},
                NP,
            )

        assert_grad_accum_invariant(self._loss_fn, make_A, make_B, make_padded)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same loss as no mask (non-mixed batch)."""
        pred = _rnd_t(2, TENSOR_SIZE)
        label = _rnd_t(2, TENSOR_SIZE)
        with_mask = {
            "global_dipole": pred,
            "mask": torch.ones(2, NB, dtype=torch.float64, device="cpu"),
        }
        without_mask = {"global_dipole": pred}
        label_dict = {"dipole": label, "find_dipole": 1.0}
        loss_m = self._loss_fn(with_mask, label_dict, NB)
        loss_nm = self._loss_fn(without_mask, label_dict, NB)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


# ---------------------------------------------------------------------------
# Task 1: unit tests for TaskLoss._inject_atom_mask
# ---------------------------------------------------------------------------


class TestInjectAtomMask:
    """Unit tests for TaskLoss._inject_atom_mask.

    NOTE: source/tests/pt/__init__.py sets torch.set_default_device("cuda:9999999")
    to catch tests that forget to specify device.  All tensor creations here
    must pass device="cpu" explicitly.
    """

    def test_injects_mask_from_atype(self) -> None:
        """When model_pred has no mask, recover it from atype (<0 = ghost)."""
        # nf=2, nloc=5; frame 0 has ghost atoms at positions 3 and 4
        atype = torch.tensor([[0, 1, 0, -1, -1], [0, 1, 0, 1, 0]], device="cpu")
        energy = torch.zeros(2, 1, dtype=torch.float64, device="cpu")
        input_dict = {"atype": atype}
        model_pred = {"energy": energy}

        result = TaskLoss._inject_atom_mask(model_pred, input_dict)

        assert "mask" in result, "mask must be injected"
        assert result["mask"].shape == (2, 5), f"wrong shape {result['mask'].shape}"
        expected = torch.tensor(
            [[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
            dtype=torch.float64,
            device="cpu",
        )
        assert torch.allclose(result["mask"], expected), (
            f"mask values wrong: {result['mask']}"
        )
        assert result["mask"].dtype == energy.dtype, (
            "mask dtype must match reference tensor dtype"
        )

    def test_mask_dtype_follows_ref(self) -> None:
        """Mask dtype follows the reference tensor (energy or coord or atype)."""
        atype = torch.tensor([[0, -1]], dtype=torch.long, device="cpu")
        model_pred = {"energy": torch.zeros(1, 1, dtype=torch.float32, device="cpu")}
        input_dict = {"atype": atype}

        result = TaskLoss._inject_atom_mask(model_pred, input_dict)

        assert result["mask"].dtype == torch.float32

    def test_fallback_to_coord_when_no_energy(self) -> None:
        """When energy absent, ref falls back to coord."""
        atype = torch.tensor([[0, -1]], dtype=torch.long, device="cpu")
        coord = torch.zeros(1, 2, 3, dtype=torch.float64, device="cpu")
        model_pred = {}
        input_dict = {"atype": atype, "coord": coord}

        result = TaskLoss._inject_atom_mask(model_pred, input_dict)

        assert "mask" in result
        assert result["mask"].dtype == torch.float64

    def test_mask_not_overwritten_if_present(self) -> None:
        """When mask is already in model_pred, leave it unchanged."""
        existing = torch.ones(2, 5, device="cpu") * 0.5
        model_pred = {
            "mask": existing,
            "energy": torch.zeros(2, 1, device="cpu"),
        }
        input_dict = {"atype": torch.zeros(2, 5, dtype=torch.long, device="cpu")}

        result = TaskLoss._inject_atom_mask(model_pred, input_dict)

        assert result["mask"] is existing, "existing mask must not be overwritten"

    def test_no_atype_no_injection(self) -> None:
        """When input_dict has no atype, mask must not be injected."""
        model_pred = {"energy": torch.zeros(2, 1, device="cpu")}
        input_dict = {}

        result = TaskLoss._inject_atom_mask(model_pred, input_dict)

        assert "mask" not in result, "mask must not appear when atype is absent"

    def test_all_real_atoms_gives_all_ones(self) -> None:
        """Non-mixed batch (all atype >= 0) yields all-ones mask."""
        atype = torch.tensor([[0, 1, 2], [1, 2, 0]], device="cpu")
        energy = torch.zeros(2, 1, dtype=torch.float32, device="cpu")
        model_pred = {"energy": energy}
        input_dict = {"atype": atype}

        result = TaskLoss._inject_atom_mask(model_pred, input_dict)

        assert torch.all(result["mask"] == 1.0), (
            "all-real atoms must give all-ones mask"
        )
