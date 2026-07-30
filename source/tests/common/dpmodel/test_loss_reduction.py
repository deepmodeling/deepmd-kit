# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the shared masked-reduction idioms (issue #5768)."""

import numpy as np
import pytest

from deepmd.dpmodel.loss.reduction import (
    masked_atom_mean,
    masked_atom_num,
    per_frame_component_mean,
)

torch = pytest.importorskip("torch")


class TestMaskedAtomMean:
    """Idiom 1: mean of a per-atom contribution over the batch's real labels."""

    def _ref(self, elem, maskf, ncomp):
        # reference reduction, numpy: pool over every real label of the batch
        masked = elem * maskf[:, :, None]
        return masked.sum() / (maskf.sum() * ncomp)

    @pytest.mark.parametrize("ncomp", [1, 3])  # atom-energy (1) and force (3)
    def test_numpy_matches_reference(self, ncomp) -> None:
        rng = np.random.default_rng(0)
        elem = rng.random((2, 4, ncomp))
        maskf = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
        got = masked_atom_mean(elem, maskf, ncomp)
        np.testing.assert_allclose(got, self._ref(elem, maskf, ncomp), rtol=0, atol=0)

    def test_uniform_atom_count_batch_is_a_frame_mean(self) -> None:
        """Pooling and the per-frame mean agree when every frame has one size.

        This identity is what lets the pooled convention apply unconditionally:
        a batch whose frames share an atom count keeps the reduction it had
        before mixed-nloc batches existed, with no special case for it.
        """
        rng = np.random.default_rng(5)
        elem = rng.random((3, 4, 3))
        maskf = np.tile(np.array([1.0, 1.0, 1.0, 0.0]), (3, 1))
        per_frame = (elem * maskf[:, :, None]).reshape(3, -1).sum(axis=-1) / (
            maskf.sum(axis=-1) * 3
        )
        np.testing.assert_allclose(
            masked_atom_mean(elem, maskf, 3), per_frame.mean(), rtol=1e-14, atol=0
        )

    def test_frames_weigh_by_their_label_count(self) -> None:
        """A frame's weight is its share of the batch's labels, not ``1 / nf``."""
        # Frame 0 keeps one real atom, frame 1 keeps three.
        maskf = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
        elem = np.zeros((2, 4, 3))
        elem[0, :1] = 2.0
        elem[1, :3] = 6.0
        # Pooled: (1*3*2 + 3*3*6) / (4*3); a frame mean would give (2+6)/2 = 4.
        np.testing.assert_allclose(
            masked_atom_mean(elem, maskf, 3), (6.0 + 54.0) / 12.0, rtol=0, atol=0
        )

    def test_torch_autograd_and_matches_numpy(self) -> None:
        elem_np = np.random.default_rng(1).random((2, 4, 3))
        maskf_np = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
        elem = torch.tensor(elem_np, requires_grad=True)
        maskf = torch.tensor(maskf_np)
        out = masked_atom_mean(elem, maskf, 3)
        out.backward()
        assert elem.grad is not None
        np.testing.assert_allclose(
            out.item(), self._ref(elem_np, maskf_np, 3), rtol=1e-14, atol=0
        )

    def test_all_padding_batch_is_not_nan(self) -> None:
        # With the pooled reduction only a batch without a single real atom
        # drives the denominator to zero; the ratio must not become 0/0 = NaN.
        elem = np.random.default_rng(3).random((2, 4, 3))
        maskf = np.zeros((2, 4))
        got = masked_atom_mean(elem, maskf, 3)
        assert np.isfinite(got)
        np.testing.assert_allclose(got, 0.0, rtol=0, atol=0)

    def test_all_padding_batch_torch_grad_is_not_nan(self) -> None:
        # the guard must keep both the value and the autograd gradient finite
        elem_np = np.random.default_rng(4).random((2, 4, 3))
        elem = torch.tensor(elem_np, requires_grad=True)
        maskf = torch.zeros((2, 4), dtype=torch.float64)
        out = masked_atom_mean(elem, maskf, 3)
        out.backward()
        assert torch.isfinite(out).item()
        assert elem.grad is not None
        assert torch.isfinite(elem.grad).all().item()


class TestPerFrameComponentMean:
    """Idiom 2 primitive: per-frame mean over the flattened component axis."""

    @pytest.mark.parametrize("k", [1, 9])  # energy (k=1) and virial (k=9)
    def test_numpy_matches_reference(self, k) -> None:
        rng = np.random.default_rng(2)
        err = rng.random((3, k))
        got = per_frame_component_mean(err)
        np.testing.assert_allclose(
            got, err.reshape(3, -1).mean(axis=-1), rtol=0, atol=0
        )

    def test_torch_bit_identical(self) -> None:
        err_np = np.random.default_rng(3).random((3, 9))
        got = per_frame_component_mean(torch.tensor(err_np))
        ref = torch.mean(torch.tensor(err_np).reshape(3, -1), dim=-1)
        assert torch.equal(got, ref)

    def test_torch_autograd(self) -> None:
        err_np = np.random.default_rng(4).random((3, 9))
        err = torch.tensor(err_np, requires_grad=True)
        out = per_frame_component_mean(err)
        out.sum().backward()
        assert err.grad is not None


class TestMaskedAtomNum:
    """Idiom 3 companion: display-only divisor for already-reduced globals."""

    def test_none_returns_natoms(self) -> None:
        assert masked_atom_num(None, 17, np.float64) == 17

    def test_numpy_mean_real_atoms(self) -> None:
        mask = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        got = masked_atom_num(mask, 3, np.float64)
        np.testing.assert_allclose(got, np.mean(np.sum(mask, axis=-1)), rtol=0, atol=0)

    def test_torch_bit_identical_float32(self) -> None:
        mask_np = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        got = masked_atom_num(torch.tensor(mask_np), 3, torch.float32)
        ref = torch.tensor(mask_np).sum(-1).float().mean()
        assert got.item() == ref.item()
