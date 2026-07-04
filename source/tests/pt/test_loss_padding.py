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

import torch

from deepmd.pt.loss.loss import (
    TaskLoss,
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
