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
from deepmd.pt.loss.ener import (
    EnergyStdLoss,
)
from deepmd.pt.loss.ener_spin import EnergySpinLoss as EnergySpinLossPT
from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.loss.property import (
    PropertyLoss,
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


# ---------------------------------------------------------------------------
# Task 3: EnergyStdLoss -- energy, force, virial, atom_ener, atom_pref
# ---------------------------------------------------------------------------

# Re-use same constants from Task 2 harness (NA=3, NB=5, NP=5).

_MASK_PAD_PT = torch.tensor(
    [[1.0] * NA + [0.0] * (NP - NA), [1.0] * NB],
    dtype=torch.float64,
    device="cpu",
)  # [2, NP]


def _t(*shape, val=None):
    """Random float64 CPU tensor."""
    if val is not None:
        return torch.full(shape, val, dtype=torch.float64, device="cpu")
    return torch.tensor(RNG.standard_normal(shape), dtype=torch.float64, device="cpu")


def _padded_force_t(f_A, f_B):
    """Stack force tensors for 2-frame padded batch."""
    pad = torch.zeros(NP, 3, dtype=torch.float64, device="cpu")
    pad[:NA] = f_A
    return torch.stack([pad, f_B], dim=0)  # [2, NP, 3]


def _padded_atom_t(a_A, a_B, ncomp):
    """Pad arr_A from [NA, ncomp] to [NP, ncomp], stack with arr_B."""
    pad = torch.zeros(NP, ncomp, dtype=torch.float64, device="cpu")
    pad[:NA] = a_A
    return torch.stack([pad, a_B], dim=0)  # [2, NP, ncomp]


class _EnerLossMockModel:
    """Callable returning a fixed model_pred dict (mask pre-populated)."""

    def __init__(self, pred: dict):
        self._pred = pred

    def __call__(self, **kwargs):
        return dict(self._pred)


def _ener_loss_fn(loss_obj, model_pred, label, natoms):
    """Call EnergyStdLoss.forward via mock model; return scalar loss tensor."""
    _, loss, _ = loss_obj.forward(
        input_dict={},
        model=_EnerLossMockModel(model_pred),
        label=label,
        natoms=natoms,
        learning_rate=1.0,
    )
    return loss


class TestPTEnergyLossEnerGradAccum:
    """Idiom 2 (extensive) for the energy term in EnergyStdLoss.

    Covers: mse (norm_exp=1 and 2), mae, huber; plus non-mixed no-op.
    """

    def _make_loss(self, loss_func="mse", intensive=False, use_huber=False):
        return EnergyStdLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            loss_func=loss_func,
            intensive_ener_virial=intensive,
            use_huber=use_huber,
        )

    def _run_invariant(self, loss_obj, e_A, e_A_hat, e_B, e_B_hat):
        def make_A():
            mp = {
                "energy": e_A,
                "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
            }
            lb = {"energy": e_A_hat, "find_energy": 1.0}
            return mp, lb, NA

        def make_B():
            mp = {
                "energy": e_B,
                "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
            }
            lb = {"energy": e_B_hat, "find_energy": 1.0}
            return mp, lb, NB

        def make_padded():
            mp = {
                "energy": torch.cat([e_A, e_B], dim=0),
                "mask": _MASK_PAD_PT,
            }
            lb = {
                "energy": torch.cat([e_A_hat, e_B_hat], dim=0),
                "find_energy": 1.0,
            }
            return mp, lb, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: _ener_loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_non_intensive_grad_accum(self):
        """Energy MSE norm_exp=1 meets the grad-accum invariant."""
        e_A = _t(1, 1)
        e_A_hat = _t(1, 1)
        e_B = _t(1, 1)
        e_B_hat = _t(1, 1)
        self._run_invariant(
            self._make_loss("mse", intensive=False), e_A, e_A_hat, e_B, e_B_hat
        )

    def test_mse_intensive_grad_accum(self):
        """Energy MSE norm_exp=2 meets the grad-accum invariant."""
        e_A = _t(1, 1)
        e_A_hat = _t(1, 1)
        e_B = _t(1, 1)
        e_B_hat = _t(1, 1)
        self._run_invariant(
            self._make_loss("mse", intensive=True), e_A, e_A_hat, e_B, e_B_hat
        )

    def test_mae_grad_accum(self):
        """Energy MAE meets the grad-accum invariant."""
        e_A = _t(1, 1)
        e_A_hat = _t(1, 1)
        e_B = _t(1, 1)
        e_B_hat = _t(1, 1)
        self._run_invariant(self._make_loss("mae"), e_A, e_A_hat, e_B, e_B_hat)

    def test_huber_grad_accum(self):
        """Energy Huber meets the grad-accum invariant."""
        e_A = _t(1, 1)
        e_A_hat = _t(1, 1)
        e_B = _t(1, 1)
        e_B_hat = _t(1, 1)
        self._run_invariant(
            self._make_loss("mse", use_huber=True), e_A, e_A_hat, e_B, e_B_hat
        )

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same energy loss as no mask."""
        e = _t(1, 1)
        e_hat = _t(1, 1)
        loss_obj = self._make_loss()

        mp_mask = {
            "energy": e,
            "mask": torch.ones(1, NP, dtype=torch.float64, device="cpu"),
        }
        mp_nm = {"energy": e}
        lb = {"energy": e_hat, "find_energy": 1.0}

        loss_m = _ener_loss_fn(loss_obj, mp_mask, lb, NP)
        loss_nm = _ener_loss_fn(loss_obj, mp_nm, lb, NP)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


class TestPTEnergyLossForceGradAccum:
    """Idiom 1 (per-atom masked mean, ncomp=3) for the force term.

    Covers: mse, mae, huber; plus non-mixed no-op.
    """

    def _make_loss(self, loss_func="mse", use_huber=False, f_use_norm=False):
        return EnergyStdLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=1.0,
            limit_pref_f=1.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            loss_func=loss_func,
            use_huber=use_huber,
            f_use_norm=f_use_norm,
        )

    def _run_invariant(self, loss_obj, f_A, f_A_hat, f_B, f_B_hat):
        def make_A():
            mp = {
                "force": f_A.unsqueeze(0),  # [1, NA, 3]
                "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
            }
            lb = {"force": f_A_hat.unsqueeze(0), "find_force": 1.0}
            return mp, lb, NA

        def make_B():
            mp = {
                "force": f_B.unsqueeze(0),
                "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
            }
            lb = {"force": f_B_hat.unsqueeze(0), "find_force": 1.0}
            return mp, lb, NB

        def make_padded():
            mp = {
                "force": _padded_force_t(f_A, f_B),  # [2, NP, 3]
                "mask": _MASK_PAD_PT,
            }
            lb = {
                "force": _padded_force_t(f_A_hat, f_B_hat),
                "find_force": 1.0,
            }
            return mp, lb, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: _ener_loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_grad_accum(self):
        """Force MSE meets the grad-accum invariant."""
        f_A = _t(NA, 3)
        f_A_hat = _t(NA, 3)
        f_B = _t(NB, 3)
        f_B_hat = _t(NB, 3)
        self._run_invariant(self._make_loss("mse"), f_A, f_A_hat, f_B, f_B_hat)

    def test_mae_grad_accum(self):
        """Force MAE meets the grad-accum invariant."""
        f_A = _t(NA, 3)
        f_A_hat = _t(NA, 3)
        f_B = _t(NB, 3)
        f_B_hat = _t(NB, 3)
        self._run_invariant(self._make_loss("mae"), f_A, f_A_hat, f_B, f_B_hat)

    def test_huber_grad_accum(self):
        """Force Huber meets the grad-accum invariant."""
        f_A = _t(NA, 3)
        f_A_hat = _t(NA, 3)
        f_B = _t(NB, 3)
        f_B_hat = _t(NB, 3)
        self._run_invariant(
            self._make_loss("mse", use_huber=True), f_A, f_A_hat, f_B, f_B_hat
        )

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same force loss as no mask."""
        f = _t(NP, 3)
        f_hat = _t(NP, 3)
        loss_obj = self._make_loss()

        mp_mask = {
            "force": f.unsqueeze(0),
            "mask": torch.ones(1, NP, dtype=torch.float64, device="cpu"),
        }
        mp_nm = {"force": f.unsqueeze(0)}
        lb = {"force": f_hat.unsqueeze(0), "find_force": 1.0}

        loss_m = _ener_loss_fn(loss_obj, mp_mask, lb, NP)
        loss_nm = _ener_loss_fn(loss_obj, mp_nm, lb, NP)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


class TestPTEnergyLossVirialGradAccum:
    """Idiom 2 (extensive, k=9) for the virial term.

    Covers: mse (norm_exp=1 and 2), mae, huber; plus non-mixed no-op.
    """

    def _make_loss(self, loss_func="mse", intensive=False, use_huber=False):
        return EnergyStdLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            loss_func=loss_func,
            intensive_ener_virial=intensive,
            use_huber=use_huber,
        )

    def _run_invariant(self, loss_obj, v_A, v_A_hat, v_B, v_B_hat):
        def make_A():
            mp = {
                "virial": v_A.unsqueeze(0),  # [1, 9]
                "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
            }
            lb = {"virial": v_A_hat.unsqueeze(0), "find_virial": 1.0}
            return mp, lb, NA

        def make_B():
            mp = {
                "virial": v_B.unsqueeze(0),
                "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
            }
            lb = {"virial": v_B_hat.unsqueeze(0), "find_virial": 1.0}
            return mp, lb, NB

        def make_padded():
            mp = {
                "virial": torch.stack([v_A, v_B], dim=0),  # [2, 9]
                "mask": _MASK_PAD_PT,
            }
            lb = {
                "virial": torch.stack([v_A_hat, v_B_hat], dim=0),
                "find_virial": 1.0,
            }
            return mp, lb, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: _ener_loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_non_intensive_grad_accum(self):
        """Virial MSE norm_exp=1 meets the grad-accum invariant."""
        v_A = _t(9)
        v_A_hat = _t(9)
        v_B = _t(9)
        v_B_hat = _t(9)
        self._run_invariant(
            self._make_loss("mse", intensive=False), v_A, v_A_hat, v_B, v_B_hat
        )

    def test_mse_intensive_grad_accum(self):
        """Virial MSE norm_exp=2 meets the grad-accum invariant."""
        v_A = _t(9)
        v_A_hat = _t(9)
        v_B = _t(9)
        v_B_hat = _t(9)
        self._run_invariant(
            self._make_loss("mse", intensive=True), v_A, v_A_hat, v_B, v_B_hat
        )

    def test_mae_grad_accum(self):
        """Virial MAE meets the grad-accum invariant."""
        v_A = _t(9)
        v_A_hat = _t(9)
        v_B = _t(9)
        v_B_hat = _t(9)
        self._run_invariant(self._make_loss("mae"), v_A, v_A_hat, v_B, v_B_hat)

    def test_huber_grad_accum(self):
        """Virial Huber meets the grad-accum invariant."""
        v_A = _t(9)
        v_A_hat = _t(9)
        v_B = _t(9)
        v_B_hat = _t(9)
        self._run_invariant(
            self._make_loss("mse", use_huber=True), v_A, v_A_hat, v_B, v_B_hat
        )

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same virial loss as no mask."""
        v = _t(9)
        v_hat = _t(9)
        loss_obj = self._make_loss()

        mp_mask = {
            "virial": v.unsqueeze(0),
            "mask": torch.ones(1, NP, dtype=torch.float64, device="cpu"),
        }
        mp_nm = {"virial": v.unsqueeze(0)}
        lb = {"virial": v_hat.unsqueeze(0), "find_virial": 1.0}

        loss_m = _ener_loss_fn(loss_obj, mp_mask, lb, NP)
        loss_nm = _ener_loss_fn(loss_obj, mp_nm, lb, NP)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


class TestPTEnergyLossAtomEnerGradAccum:
    """Idiom 1 (per-atom masked mean, ncomp=1) for the atom_ener term.

    Covers: mse, mae; plus non-mixed no-op.
    """

    def _make_loss(self, loss_func="mse"):
        return EnergyStdLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_ae=1.0,
            limit_pref_ae=1.0,
            loss_func=loss_func,
        )

    def _run_invariant(self, loss_obj, ae_A, ae_A_hat, ae_B, ae_B_hat):
        def make_A():
            mp = {
                "atom_energy": ae_A.unsqueeze(0),  # [1, NA, 1]
                "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
            }
            lb = {"atom_ener": ae_A_hat.unsqueeze(0), "find_atom_ener": 1.0}
            return mp, lb, NA

        def make_B():
            mp = {
                "atom_energy": ae_B.unsqueeze(0),
                "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
            }
            lb = {"atom_ener": ae_B_hat.unsqueeze(0), "find_atom_ener": 1.0}
            return mp, lb, NB

        def make_padded():
            ae_pad = _padded_atom_t(ae_A, ae_B, 1)  # [2, NP, 1]
            ae_hat_pad = _padded_atom_t(ae_A_hat, ae_B_hat, 1)
            mp = {"atom_energy": ae_pad, "mask": _MASK_PAD_PT}
            lb = {"atom_ener": ae_hat_pad, "find_atom_ener": 1.0}
            return mp, lb, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: _ener_loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_grad_accum(self):
        """Atom-energy MSE meets the grad-accum invariant."""
        ae_A = _t(NA, 1)
        ae_A_hat = _t(NA, 1)
        ae_B = _t(NB, 1)
        ae_B_hat = _t(NB, 1)
        self._run_invariant(self._make_loss("mse"), ae_A, ae_A_hat, ae_B, ae_B_hat)

    def test_mae_grad_accum(self):
        """Atom-energy MAE meets the grad-accum invariant."""
        ae_A = _t(NA, 1)
        ae_A_hat = _t(NA, 1)
        ae_B = _t(NB, 1)
        ae_B_hat = _t(NB, 1)
        self._run_invariant(self._make_loss("mae"), ae_A, ae_A_hat, ae_B, ae_B_hat)

    def test_huber_grad_accum(self):
        """Atom-energy Huber (use_huber=True) meets the grad-accum invariant."""
        ae_A = _t(NA, 1)
        ae_A_hat = _t(NA, 1)
        ae_B = _t(NB, 1)
        ae_B_hat = _t(NB, 1)
        loss_obj = EnergyStdLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_ae=1.0,
            limit_pref_ae=1.0,
            use_huber=True,
        )
        self._run_invariant(loss_obj, ae_A, ae_A_hat, ae_B, ae_B_hat)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same atom-energy loss as no mask."""
        ae = _t(NP, 1)
        ae_hat = _t(NP, 1)
        loss_obj = self._make_loss()

        mp_mask = {
            "atom_energy": ae.unsqueeze(0),
            "mask": torch.ones(1, NP, dtype=torch.float64, device="cpu"),
        }
        mp_nm = {"atom_energy": ae.unsqueeze(0)}
        lb = {"atom_ener": ae_hat.unsqueeze(0), "find_atom_ener": 1.0}

        loss_m = _ener_loss_fn(loss_obj, mp_mask, lb, NP)
        loss_nm = _ener_loss_fn(loss_obj, mp_nm, lb, NP)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


class TestPTEnergyLossAtomPrefGradAccum:
    """Idiom 1 with pref weight (ncomp=3) for the atom_pref term.

    Covers: mse, mae; plus non-mixed no-op.
    """

    def _make_pf_only_loss(self, loss_func="mse"):
        return EnergyStdLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=1.0,
            limit_pref_f=1.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_pf=1.0,
            limit_pref_pf=1.0,
            loss_func=loss_func,
        )

    def _run_invariant(self, loss_obj, f_A, f_A_hat, pf_A, f_B, f_B_hat, pf_B):
        def make_A():
            mp = {
                "force": f_A.unsqueeze(0),
                "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
            }
            lb = {
                "force": f_A_hat.unsqueeze(0),
                "atom_pref": pf_A.unsqueeze(0),
                "find_force": 1.0,
                "find_atom_pref": 1.0,
            }
            return mp, lb, NA

        def make_B():
            mp = {
                "force": f_B.unsqueeze(0),
                "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
            }
            lb = {
                "force": f_B_hat.unsqueeze(0),
                "atom_pref": pf_B.unsqueeze(0),
                "find_force": 1.0,
                "find_atom_pref": 1.0,
            }
            return mp, lb, NB

        def make_padded():
            f_pad = _padded_force_t(f_A, f_B)
            f_hat_pad = _padded_force_t(f_A_hat, f_B_hat)
            pf_pad = _padded_atom_t(pf_A, pf_B, 3)  # [2, NP, 3]
            mp = {"force": f_pad, "mask": _MASK_PAD_PT}
            lb = {
                "force": f_hat_pad,
                "atom_pref": pf_pad,
                "find_force": 1.0,
                "find_atom_pref": 1.0,
            }
            return mp, lb, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: _ener_loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_grad_accum(self):
        """Atom-pref MSE meets the grad-accum invariant."""
        f_A, f_A_hat = _t(NA, 3), _t(NA, 3)
        pf_A = torch.abs(_t(NA, 3)) + 0.1
        f_B, f_B_hat = _t(NB, 3), _t(NB, 3)
        pf_B = torch.abs(_t(NB, 3)) + 0.1
        self._run_invariant(
            self._make_pf_only_loss("mse"), f_A, f_A_hat, pf_A, f_B, f_B_hat, pf_B
        )

    def test_mae_grad_accum(self):
        """Atom-pref MAE meets the grad-accum invariant."""
        f_A, f_A_hat = _t(NA, 3), _t(NA, 3)
        pf_A = torch.abs(_t(NA, 3)) + 0.1
        f_B, f_B_hat = _t(NB, 3), _t(NB, 3)
        pf_B = torch.abs(_t(NB, 3)) + 0.1
        self._run_invariant(
            self._make_pf_only_loss("mae"), f_A, f_A_hat, pf_A, f_B, f_B_hat, pf_B
        )

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same atom-pref loss as no mask."""
        f = _t(NP, 3)
        f_hat = _t(NP, 3)
        pf = torch.abs(_t(NP, 3)) + 0.1
        loss_obj = self._make_pf_only_loss("mse")

        mp_mask = {
            "force": f.unsqueeze(0),
            "mask": torch.ones(1, NP, dtype=torch.float64, device="cpu"),
        }
        mp_nm = {"force": f.unsqueeze(0)}
        lb = {
            "force": f_hat.unsqueeze(0),
            "atom_pref": pf.unsqueeze(0),
            "find_force": 1.0,
            "find_atom_pref": 1.0,
        }

        loss_m = _ener_loss_fn(loss_obj, mp_mask, lb, NP)
        loss_nm = _ener_loss_fn(loss_obj, mp_nm, lb, NP)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


# ---------------------------------------------------------------------------
# Task 4: PropertyLoss -- extensive (not intensive) property
# ---------------------------------------------------------------------------

PROP_TASK_DIM = 2
PROP_VAR = "test_prop"


class TestPTPropertyLossExtensiveGradAccum:
    """Idiom 2 (per-frame real-natoms normalization) for extensive pt PropertyLoss.

    _loss_fn wraps the raw loss with /nf so the per-frame average matches the
    separate-frame reference (pt PropertyLoss uses reduction='sum' over frames).
    The mask is pre-populated in model_pred so _inject_atom_mask leaves it alone.
    """

    def _make_loss(self, loss_func="mse"):
        return PropertyLoss(
            task_dim=PROP_TASK_DIM,
            var_name=PROP_VAR,
            loss_func=loss_func,
            intensive=False,
            # Provide explicit out_std/out_bias to avoid accessing model.atomic_model.
            out_std=[1.0] * PROP_TASK_DIM,
            out_bias=[0.0] * PROP_TASK_DIM,
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        """Return per-frame-averaged loss (raw loss / nf)."""
        nf = model_pred[PROP_VAR].shape[0]
        _, loss, _ = loss_obj.forward(
            input_dict={},  # no atype; mask already in model_pred
            model=_MockModel(model_pred),
            label=label,
            natoms=natoms,
            learning_rate=1.0,
        )
        return loss / nf

    def _run_invariant(self, loss_obj, p_A, l_A, p_B, l_B):
        def make_A():
            return (
                {
                    PROP_VAR: p_A,
                    "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
                },
                {PROP_VAR: l_A.clone()},
                NA,
            )

        def make_B():
            return (
                {
                    PROP_VAR: p_B,
                    "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
                },
                {PROP_VAR: l_B.clone()},
                NB,
            )

        def make_padded():
            return (
                {
                    PROP_VAR: torch.cat([p_A, p_B], dim=0),  # [2, task_dim]
                    "mask": _MASK_PAD_PT,
                },
                {PROP_VAR: torch.cat([l_A, l_B], dim=0)},
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
        p_A = _rnd_t(1, PROP_TASK_DIM)
        l_A = _rnd_t(1, PROP_TASK_DIM)
        p_B = _rnd_t(1, PROP_TASK_DIM)
        l_B = _rnd_t(1, PROP_TASK_DIM)
        self._run_invariant(self._make_loss("mse"), p_A, l_A, p_B, l_B)

    def test_mae_grad_accum(self):
        """MAE extensive property meets the grad-accum invariant."""
        p_A = _rnd_t(1, PROP_TASK_DIM)
        l_A = _rnd_t(1, PROP_TASK_DIM)
        p_B = _rnd_t(1, PROP_TASK_DIM)
        l_B = _rnd_t(1, PROP_TASK_DIM)
        self._run_invariant(self._make_loss("mae"), p_A, l_A, p_B, l_B)

    def test_smooth_mae_grad_accum(self):
        """smooth_mae extensive property meets the grad-accum invariant."""
        p_A = _rnd_t(1, PROP_TASK_DIM)
        l_A = _rnd_t(1, PROP_TASK_DIM)
        p_B = _rnd_t(1, PROP_TASK_DIM)
        l_B = _rnd_t(1, PROP_TASK_DIM)
        self._run_invariant(self._make_loss("smooth_mae"), p_A, l_A, p_B, l_B)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same extensive-property loss as no mask."""
        p = _rnd_t(2, PROP_TASK_DIM)
        l = _rnd_t(2, PROP_TASK_DIM)
        loss_obj = self._make_loss("mse")
        mp_mask = {
            PROP_VAR: p,
            "mask": torch.ones(2, NB, dtype=torch.float64, device="cpu"),
        }
        mp_nm = {PROP_VAR: p}
        # pt forward mutates label[var_name]; use separate dicts for each call.
        lb_m = {PROP_VAR: l.clone()}
        lb_nm = {PROP_VAR: l.clone()}
        loss_m = self._loss_fn(loss_obj, mp_mask, lb_m, NB)
        loss_nm = self._loss_fn(loss_obj, mp_nm, lb_nm, NB)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


class TestPTPropertyLossIntensiveUnaffectedByMask:
    """Intensive property loss must be unchanged whether or not mask is present."""

    def _make_loss(self):
        return PropertyLoss(
            task_dim=PROP_TASK_DIM,
            var_name=PROP_VAR,
            loss_func="mse",
            intensive=True,
            out_std=[1.0] * PROP_TASK_DIM,
            out_bias=[0.0] * PROP_TASK_DIM,
        )

    def _loss_fn(self, loss_obj, model_pred, label, natoms):
        _, loss, _ = loss_obj.forward(
            input_dict={},
            model=_MockModel(model_pred),
            label=label,
            natoms=natoms,
            learning_rate=1.0,
        )
        return loss

    def test_intensive_ignores_mask(self):
        """Intensive property: masked batch == unmasked batch."""
        p = _rnd_t(2, PROP_TASK_DIM)
        l = _rnd_t(2, PROP_TASK_DIM)
        loss_obj = self._make_loss()
        mp_mask = {PROP_VAR: p, "mask": _MASK_PAD_PT}
        mp_nm = {PROP_VAR: p}
        # Use separate label dicts since pt forward mutates label[var_name].
        lb_m = {PROP_VAR: l.clone()}
        lb_nm = {PROP_VAR: l.clone()}
        loss_m = self._loss_fn(loss_obj, mp_mask, lb_m, NP)
        loss_nm = self._loss_fn(loss_obj, mp_nm, lb_nm, NP)
        assert torch.isclose(loss_m, loss_nm), (
            f"intensive property must ignore mask: {loss_m.item()} vs {loss_nm.item()}"
        )


# ---------------------------------------------------------------------------
# Task 5: EnergySpinLoss (pt) -- energy (has_e), force_real (has_fr),
# virial (has_v).  Leave force_mag / mask_mag (has_fm) COMPLETELY UNTOUCHED.
# ---------------------------------------------------------------------------

# Spin-specific test constants: NM magnetic atoms per frame (same count in
# both frames so that .view(nframes,-1,3) in the pt force_mag path is valid).
_NM_PT = 2

_MASK_MAG_A_PT = torch.zeros(1, NA, 1, dtype=torch.bool, device="cpu")
_MASK_MAG_A_PT[0, :_NM_PT, 0] = True  # first NM_PT atoms of frame A magnetic

_MASK_MAG_B_PT = torch.zeros(1, NB, 1, dtype=torch.bool, device="cpu")
_MASK_MAG_B_PT[0, :_NM_PT, 0] = True  # first NM_PT atoms of frame B magnetic

_MASK_MAG_PAD_SPIN_PT = torch.zeros(2, NP, 1, dtype=torch.bool, device="cpu")
_MASK_MAG_PAD_SPIN_PT[0, :_NM_PT, 0] = True  # frame A
_MASK_MAG_PAD_SPIN_PT[1, :_NM_PT, 0] = True  # frame B

_MASK_PAD_SPIN_PT = torch.tensor(
    [[1.0] * NA + [0.0] * (NP - NA), [1.0] * NB],
    dtype=torch.float64,
    device="cpu",
)  # [2, NP]


def _spin_loss_fn(loss_obj, model_pred, label, natoms):
    """Call EnergySpinLoss.forward via mock model; return scalar loss tensor."""
    _, loss, _ = loss_obj.forward(
        input_dict={},  # mask already in model_pred; no re-injection needed
        model=_MockModel(model_pred),
        label=label,
        natoms=natoms,
        learning_rate=1.0,
    )
    return loss


class TestPTEnerSpinLossEnerGradAccum:
    """Idiom 2 (extensive) for the energy term in pt EnergySpinLoss.

    Covers: mse (norm_exp=1 and 2); plus non-mixed no-op.
    """

    def _make_loss(self, intensive=False):
        return EnergySpinLossPT(
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

    def _run_invariant(self, loss_obj, e_A, e_A_hat, e_B, e_B_hat):
        def make_A():
            mp = {
                "energy": e_A,
                "mask_mag": _MASK_MAG_A_PT,
                "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
            }
            lb = {"energy": e_A_hat, "find_energy": 1.0}
            return mp, lb, NA

        def make_B():
            mp = {
                "energy": e_B,
                "mask_mag": _MASK_MAG_B_PT,
                "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
            }
            lb = {"energy": e_B_hat, "find_energy": 1.0}
            return mp, lb, NB

        def make_padded():
            mp = {
                "energy": torch.cat([e_A, e_B], dim=0),  # [2, 1]
                "mask_mag": _MASK_MAG_PAD_SPIN_PT,
                "mask": _MASK_PAD_SPIN_PT,
            }
            lb = {
                "energy": torch.cat([e_A_hat, e_B_hat], dim=0),
                "find_energy": 1.0,
            }
            return mp, lb, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: _spin_loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_non_intensive_grad_accum(self):
        """Spin energy MSE norm_exp=1 meets the grad-accum invariant."""
        e_A = _t(1, 1)
        e_A_hat = _t(1, 1)
        e_B = _t(1, 1)
        e_B_hat = _t(1, 1)
        self._run_invariant(
            self._make_loss(intensive=False), e_A, e_A_hat, e_B, e_B_hat
        )

    def test_mse_intensive_grad_accum(self):
        """Spin energy MSE norm_exp=2 meets the grad-accum invariant."""
        e_A = _t(1, 1)
        e_A_hat = _t(1, 1)
        e_B = _t(1, 1)
        e_B_hat = _t(1, 1)
        self._run_invariant(self._make_loss(intensive=True), e_A, e_A_hat, e_B, e_B_hat)

    def test_mae_grad_accum(self):
        """Spin energy MAE meets the grad-accum invariant."""
        e_A = _t(1, 1)
        e_A_hat = _t(1, 1)
        e_B = _t(1, 1)
        e_B_hat = _t(1, 1)
        loss_obj = EnergySpinLossPT(
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
        e = _t(1, 1)
        e_hat = _t(1, 1)
        loss_obj = self._make_loss()
        mp_mask = {
            "energy": e,
            "mask_mag": _MASK_MAG_B_PT,
            "mask": torch.ones(1, NP, dtype=torch.float64, device="cpu"),
        }
        mp_nm = {"energy": e, "mask_mag": _MASK_MAG_B_PT}
        lb = {"energy": e_hat, "find_energy": 1.0}
        loss_m = _spin_loss_fn(loss_obj, mp_mask, lb, NP)
        loss_nm = _spin_loss_fn(loss_obj, mp_nm, lb, NP)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


class TestPTEnerSpinLossForceRealGradAccum:
    """Idiom 1 (per-atom masked mean, ncomp=3) for force_real in pt EnergySpinLoss.

    Covers: mse; plus non-mixed no-op.
    """

    def _make_loss(self):
        return EnergySpinLossPT(
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

    def _run_invariant(self, loss_obj, f_A, f_A_hat, f_B, f_B_hat):
        def make_A():
            mp = {
                "force": f_A.unsqueeze(0),  # [1, NA, 3]
                "mask_mag": _MASK_MAG_A_PT,
                "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
            }
            lb = {"force": f_A_hat.unsqueeze(0), "find_force": 1.0}
            return mp, lb, NA

        def make_B():
            mp = {
                "force": f_B.unsqueeze(0),
                "mask_mag": _MASK_MAG_B_PT,
                "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
            }
            lb = {"force": f_B_hat.unsqueeze(0), "find_force": 1.0}
            return mp, lb, NB

        def make_padded():
            mp = {
                "force": _padded_force_t(f_A, f_B),  # [2, NP, 3]
                "mask_mag": _MASK_MAG_PAD_SPIN_PT,
                "mask": _MASK_PAD_SPIN_PT,
            }
            lb = {
                "force": _padded_force_t(f_A_hat, f_B_hat),
                "find_force": 1.0,
            }
            return mp, lb, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: _spin_loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_grad_accum(self):
        """Force_real MSE meets the grad-accum invariant."""
        f_A = _t(NA, 3)
        f_A_hat = _t(NA, 3)
        f_B = _t(NB, 3)
        f_B_hat = _t(NB, 3)
        self._run_invariant(self._make_loss(), f_A, f_A_hat, f_B, f_B_hat)

    def test_mae_grad_accum(self):
        """Force_real MAE meets the grad-accum invariant."""
        f_A = _t(NA, 3)
        f_A_hat = _t(NA, 3)
        f_B = _t(NB, 3)
        f_B_hat = _t(NB, 3)
        loss_obj = EnergySpinLossPT(
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
        f = _t(NP, 3)
        f_hat = _t(NP, 3)
        loss_obj = self._make_loss()
        mp_mask = {
            "force": f.unsqueeze(0),
            "mask_mag": _MASK_MAG_B_PT,
            "mask": torch.ones(1, NP, dtype=torch.float64, device="cpu"),
        }
        mp_nm = {"force": f.unsqueeze(0), "mask_mag": _MASK_MAG_B_PT}
        lb = {"force": f_hat.unsqueeze(0), "find_force": 1.0}
        loss_m = _spin_loss_fn(loss_obj, mp_mask, lb, NP)
        loss_nm = _spin_loss_fn(loss_obj, mp_nm, lb, NP)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


class TestPTEnerSpinLossVirialGradAccum:
    """Idiom 2 (extensive, k=9) for virial in pt EnergySpinLoss.

    Covers: mse (norm_exp=1 and 2); plus non-mixed no-op.
    """

    def _make_loss(self, intensive=False):
        return EnergySpinLossPT(
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

    def _run_invariant(self, loss_obj, v_A, v_A_hat, v_B, v_B_hat):
        def make_A():
            mp = {
                "virial": v_A.unsqueeze(0),  # [1, 9]
                "mask_mag": _MASK_MAG_A_PT,
                "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
            }
            lb = {"virial": v_A_hat.unsqueeze(0), "find_virial": 1.0}
            return mp, lb, NA

        def make_B():
            mp = {
                "virial": v_B.unsqueeze(0),
                "mask_mag": _MASK_MAG_B_PT,
                "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
            }
            lb = {"virial": v_B_hat.unsqueeze(0), "find_virial": 1.0}
            return mp, lb, NB

        def make_padded():
            mp = {
                "virial": torch.stack([v_A, v_B], dim=0),  # [2, 9]
                "mask_mag": _MASK_MAG_PAD_SPIN_PT,
                "mask": _MASK_PAD_SPIN_PT,
            }
            lb = {
                "virial": torch.stack([v_A_hat, v_B_hat], dim=0),
                "find_virial": 1.0,
            }
            return mp, lb, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: _spin_loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_non_intensive_grad_accum(self):
        """Spin virial MSE norm_exp=1 meets the grad-accum invariant."""
        v_A = _t(9)
        v_A_hat = _t(9)
        v_B = _t(9)
        v_B_hat = _t(9)
        self._run_invariant(
            self._make_loss(intensive=False), v_A, v_A_hat, v_B, v_B_hat
        )

    def test_mse_intensive_grad_accum(self):
        """Spin virial MSE norm_exp=2 meets the grad-accum invariant."""
        v_A = _t(9)
        v_A_hat = _t(9)
        v_B = _t(9)
        v_B_hat = _t(9)
        self._run_invariant(self._make_loss(intensive=True), v_A, v_A_hat, v_B, v_B_hat)

    def test_mae_grad_accum(self):
        """Spin virial MAE meets the grad-accum invariant."""
        v_A = _t(9)
        v_A_hat = _t(9)
        v_B = _t(9)
        v_B_hat = _t(9)
        loss_obj = EnergySpinLossPT(
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
        v = _t(9)
        v_hat = _t(9)
        loss_obj = self._make_loss()
        mp_mask = {
            "virial": v.unsqueeze(0),
            "mask_mag": _MASK_MAG_B_PT,
            "mask": torch.ones(1, NP, dtype=torch.float64, device="cpu"),
        }
        mp_nm = {"virial": v.unsqueeze(0), "mask_mag": _MASK_MAG_B_PT}
        lb = {"virial": v_hat.unsqueeze(0), "find_virial": 1.0}
        loss_m = _spin_loss_fn(loss_obj, mp_mask, lb, NP)
        loss_nm = _spin_loss_fn(loss_obj, mp_nm, lb, NP)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


class TestPTEnerSpinLossForceMagUnchanged:
    """Guard: the padding mask must NOT affect the force_mag / mask_mag term.

    The force_mag path uses mask_mag (spin virtual-atom mask), which is a
    completely separate concept from the padding mask model_pred["mask"].
    After the Task-5 implementation, presenting a padding mask must leave
    the force_mag loss bit-identical.
    """

    def _make_loss(self):
        return EnergySpinLossPT(
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
        fm = _rnd_t(2, NP, 3)
        fm_hat = _rnd_t(2, NP, 3)
        loss_obj = self._make_loss()

        def _run(with_mask):
            mp = {
                "force_mag": fm,
                "mask_mag": _MASK_MAG_PAD_SPIN_PT,
            }
            lb = {
                "force_mag": fm_hat,
                "find_force_mag": 1.0,
            }
            if with_mask:
                mp["mask"] = _MASK_PAD_SPIN_PT
            return _spin_loss_fn(loss_obj, mp, lb, NP)

        loss_with = _run(True)
        loss_without = _run(False)
        assert torch.isclose(loss_with, loss_without), (
            f"force_mag loss must be unchanged by padding mask: "
            f"{loss_with.item()} vs {loss_without.item()}"
        )


# ---------------------------------------------------------------------------
# Part A: EnergySpinLoss atom_ener (has_ae) grad-accum invariant (pt)
# ---------------------------------------------------------------------------


class TestPTEnerSpinLossAtomEnerGradAccum:
    """Idiom 1 (per-atom masked mean, ncomp=1) for atom_ener in pt EnergySpinLoss.

    RED before the Part-A has_ae mask fix; GREEN after.
    """

    def _make_loss(self, loss_func="mse"):
        return EnergySpinLossPT(
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

    def _run_invariant(self, loss_obj, ae_A, ae_A_hat, ae_B, ae_B_hat):
        def make_A():
            mp = {
                "atom_energy": ae_A.unsqueeze(0),  # [1, NA, 1]
                "mask_mag": _MASK_MAG_A_PT,
                "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
            }
            lb = {"atom_ener": ae_A_hat.unsqueeze(0), "find_atom_ener": 1.0}
            return mp, lb, NA

        def make_B():
            mp = {
                "atom_energy": ae_B.unsqueeze(0),
                "mask_mag": _MASK_MAG_B_PT,
                "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
            }
            lb = {"atom_ener": ae_B_hat.unsqueeze(0), "find_atom_ener": 1.0}
            return mp, lb, NB

        def make_padded():
            ae_pad = _padded_atom_t(ae_A, ae_B, 1)  # [2, NP, 1]
            ae_hat_pad = _padded_atom_t(ae_A_hat, ae_B_hat, 1)
            mp = {
                "atom_energy": ae_pad,
                "mask_mag": _MASK_MAG_PAD_SPIN_PT,
                "mask": _MASK_PAD_SPIN_PT,
            }
            lb = {"atom_ener": ae_hat_pad, "find_atom_ener": 1.0}
            return mp, lb, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: _spin_loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_grad_accum(self):
        """Spin atom_ener MSE meets the grad-accum invariant."""
        ae_A = _t(NA, 1)
        ae_A_hat = _t(NA, 1)
        ae_B = _t(NB, 1)
        ae_B_hat = _t(NB, 1)
        self._run_invariant(self._make_loss("mse"), ae_A, ae_A_hat, ae_B, ae_B_hat)

    def test_mae_grad_accum(self):
        """Spin atom_ener MAE meets the grad-accum invariant."""
        ae_A = _t(NA, 1)
        ae_A_hat = _t(NA, 1)
        ae_B = _t(NB, 1)
        ae_B_hat = _t(NB, 1)
        self._run_invariant(self._make_loss("mae"), ae_A, ae_A_hat, ae_B, ae_B_hat)

    def test_no_op_for_non_mixed(self):
        """All-ones mask gives same atom_ener spin loss as no mask."""
        ae = _t(NP, 1)
        ae_hat = _t(NP, 1)
        loss_obj = self._make_loss()
        mp_mask = {
            "atom_energy": ae.unsqueeze(0),
            "mask_mag": _MASK_MAG_B_PT,
            "mask": torch.ones(1, NP, dtype=torch.float64, device="cpu"),
        }
        mp_nm = {
            "atom_energy": ae.unsqueeze(0),
            "mask_mag": _MASK_MAG_B_PT,
        }
        lb = {"atom_ener": ae_hat.unsqueeze(0), "find_atom_ener": 1.0}
        loss_m = _spin_loss_fn(loss_obj, mp_mask, lb, NP)
        loss_nm = _spin_loss_fn(loss_obj, mp_nm, lb, NP)
        assert torch.isclose(loss_m, loss_nm), (
            f"all-ones mask must be no-op: {loss_m.item()} vs {loss_nm.item()}"
        )


# ---------------------------------------------------------------------------
# gen_force (has_gf) grad-accum invariant (pt)
# ---------------------------------------------------------------------------

_NGEN_PT = 2  # number of generalized coordinates for tests


class TestPTEnergyLossGenForceGradAccum:
    """gen_force (has_gf) excludes ghost atoms via force masking before projection.

    Expected GREEN immediately (no fix needed for gen_force).
    """

    def _make_loss(self):
        return EnergyStdLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_gf=1.0,
            limit_pref_gf=1.0,
            numb_generalized_coord=_NGEN_PT,
        )

    def _run_invariant(self, loss_obj, f_A, f_A_hat, drdq_A, f_B, f_B_hat, drdq_B):
        def make_A():
            mp = {
                "force": f_A.unsqueeze(0),  # [1, NA, 3]
                "mask": torch.ones(1, NA, dtype=torch.float64, device="cpu"),
            }
            lb = {
                "force": f_A_hat.unsqueeze(0),
                "find_force": 1.0,
                "drdq": drdq_A.unsqueeze(0),  # [1, NA*3, NGEN]
                "find_drdq": 1.0,
            }
            return mp, lb, NA

        def make_B():
            mp = {
                "force": f_B.unsqueeze(0),
                "mask": torch.ones(1, NB, dtype=torch.float64, device="cpu"),
            }
            lb = {
                "force": f_B_hat.unsqueeze(0),
                "find_force": 1.0,
                "drdq": drdq_B.unsqueeze(0),  # [1, NB*3, NGEN]
                "find_drdq": 1.0,
            }
            return mp, lb, NB

        def make_padded():
            f_pad = _padded_force_t(f_A, f_B)  # [2, NP, 3]
            f_hat_pad = _padded_force_t(f_A_hat, f_B_hat)
            # drdq ghost-atom slots are zero (ghost forces also zero, no contribution)
            drdq_A_pad = torch.zeros(
                NP * 3, _NGEN_PT, dtype=torch.float64, device="cpu"
            )
            drdq_A_pad[: NA * 3] = drdq_A
            drdq_pad = torch.stack([drdq_A_pad, drdq_B], dim=0)  # [2, NP*3, NGEN]
            mp = {"force": f_pad, "mask": _MASK_PAD_PT}
            lb = {
                "force": f_hat_pad,
                "find_force": 1.0,
                "drdq": drdq_pad,
                "find_drdq": 1.0,
            }
            return mp, lb, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: _ener_loss_fn(loss_obj, mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )

    def test_mse_grad_accum(self):
        """gen_force MSE meets the grad-accum invariant (GREEN: already correct)."""
        f_A = _t(NA, 3)
        f_A_hat = _t(NA, 3)
        drdq_A = _t(NA * 3, _NGEN_PT)
        f_B = _t(NB, 3)
        f_B_hat = _t(NB, 3)
        drdq_B = _t(NB * 3, _NGEN_PT)
        self._run_invariant(
            self._make_loss(), f_A, f_A_hat, drdq_A, f_B, f_B_hat, drdq_B
        )


# ---------------------------------------------------------------------------
# force_mag MSE grad-accum invariant (pt)
#
# NOTE: force_mag MAE (pt) uses .sum() over frames, so MAE force_mag FAILS
# the invariant with a 2x factor when frames=2. This is a pre-existing
# frame-normalization artifact independent of ghost-atom masking. Ghost atoms
# are correctly excluded (mask_mag=False there). MAE artifact reported as
# NEEDS_CONTEXT in the audit report.
# ---------------------------------------------------------------------------


class TestPTEnerSpinLossForceMagMSEGradAccum:
    """force_mag MSE meets the grad-accum invariant when ghost atoms are non-magnetic.

    With equal magnetic-atom counts (NM) across frames and ghost atoms having
    mask_mag=False, the fancy-index selection excludes padding and the mean()
    over [nf, NM, 3] satisfies the invariant.
    Expected GREEN immediately (no fix needed for MSE force_mag).
    """

    def _make_loss(self):
        return EnergySpinLossPT(
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

    def test_mse_grad_accum(self):
        """force_mag MSE (NM equal per frame, ghost atoms non-magnetic) meets invariant."""
        # Only magnetic-atom slots (first NM_PT) have non-zero values; others zero.
        fm_A = _rnd_t(_NM_PT, 3)
        fm_A_hat = _rnd_t(_NM_PT, 3)
        fm_B = _rnd_t(_NM_PT, 3)
        fm_B_hat = _rnd_t(_NM_PT, 3)

        def make_A():
            fm_A_full = torch.zeros(NA, 3, dtype=torch.float64, device="cpu")
            fm_A_full[:_NM_PT] = fm_A
            fm_A_hat_full = torch.zeros(NA, 3, dtype=torch.float64, device="cpu")
            fm_A_hat_full[:_NM_PT] = fm_A_hat
            mp = {
                "force_mag": fm_A_full.unsqueeze(0),  # [1, NA, 3]
                "mask_mag": _MASK_MAG_A_PT,
            }
            lb = {"force_mag": fm_A_hat_full.unsqueeze(0), "find_force_mag": 1.0}
            return mp, lb, NA

        def make_B():
            fm_B_full = torch.zeros(NB, 3, dtype=torch.float64, device="cpu")
            fm_B_full[:_NM_PT] = fm_B
            fm_B_hat_full = torch.zeros(NB, 3, dtype=torch.float64, device="cpu")
            fm_B_hat_full[:_NM_PT] = fm_B_hat
            mp = {
                "force_mag": fm_B_full.unsqueeze(0),  # [1, NB, 3]
                "mask_mag": _MASK_MAG_B_PT,
            }
            lb = {"force_mag": fm_B_hat_full.unsqueeze(0), "find_force_mag": 1.0}
            return mp, lb, NB

        def make_padded():
            fm_A_pad = torch.zeros(NP, 3, dtype=torch.float64, device="cpu")
            fm_A_pad[:_NM_PT] = fm_A
            fm_A_hat_pad = torch.zeros(NP, 3, dtype=torch.float64, device="cpu")
            fm_A_hat_pad[:_NM_PT] = fm_A_hat
            fm_B_pad = torch.zeros(NP, 3, dtype=torch.float64, device="cpu")
            fm_B_pad[:_NM_PT] = fm_B
            fm_B_hat_pad = torch.zeros(NP, 3, dtype=torch.float64, device="cpu")
            fm_B_hat_pad[:_NM_PT] = fm_B_hat
            mp = {
                "force_mag": torch.stack([fm_A_pad, fm_B_pad], dim=0),  # [2, NP, 3]
                "mask_mag": _MASK_MAG_PAD_SPIN_PT,
                "mask": _MASK_PAD_SPIN_PT,
            }
            lb = {
                "force_mag": torch.stack([fm_A_hat_pad, fm_B_hat_pad], dim=0),
                "find_force_mag": 1.0,
            }
            return mp, lb, NP

        assert_grad_accum_invariant(
            lambda mp, lb, na: _spin_loss_fn(self._make_loss(), mp, lb, na),
            make_A,
            make_B,
            make_padded,
        )
