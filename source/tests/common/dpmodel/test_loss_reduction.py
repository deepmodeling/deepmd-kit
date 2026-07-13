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
    """Idiom 1: per-atom masked mean over ncomp components, averaged over frames."""

    def _ref(self, elem, maskf, ncomp):
        # reference reduction, numpy
        nf = elem.shape[0]
        masked = elem * maskf[:, :, None]
        pfs = masked.reshape(nf, -1).sum(axis=-1)
        pfd = maskf.sum(axis=-1) * ncomp
        return (pfs / pfd).mean()

    @pytest.mark.parametrize("ncomp", [1, 3])  # atom-energy (1) and force (3)
    def test_numpy_matches_reference(self, ncomp) -> None:
        rng = np.random.default_rng(0)
        elem = rng.random((2, 4, ncomp))
        maskf = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
        got = masked_atom_mean(elem, maskf, ncomp)
        np.testing.assert_allclose(got, self._ref(elem, maskf, ncomp), rtol=0, atol=0)

    def test_torch_autograd_and_bit_identical(self) -> None:
        elem_np = np.random.default_rng(1).random((2, 4, 3))
        maskf_np = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
        elem = torch.tensor(elem_np, requires_grad=True)
        maskf = torch.tensor(maskf_np)
        out = masked_atom_mean(elem, maskf, 3)
        out.backward()
        assert elem.grad is not None
        # bit-identical to torch-native inline form
        en = torch.tensor(elem_np)
        mn = torch.tensor(maskf_np)
        pfs = (en * mn[:, :, None]).reshape(2, -1).sum(dim=-1)
        pfd = mn.sum(dim=-1) * 3
        ref = torch.mean(pfs / pfd)
        assert out.item() == ref.item()


class TestPerFrameComponentMean:
    """Idiom 2 primitive: per-frame mean over the flattened component axis."""

    @pytest.mark.parametrize("k", [1, 9])  # energy (k=1) and virial (k=9)
    def test_numpy_matches_reference(self, k) -> None:
        rng = np.random.default_rng(2)
        err = rng.random((3, k))
        got = per_frame_component_mean(err, 3)
        np.testing.assert_allclose(
            got, err.reshape(3, -1).mean(axis=-1), rtol=0, atol=0
        )

    def test_torch_bit_identical(self) -> None:
        err_np = np.random.default_rng(3).random((3, 9))
        got = per_frame_component_mean(torch.tensor(err_np), 3)
        ref = torch.mean(torch.tensor(err_np).reshape(3, -1), dim=-1)
        assert torch.equal(got, ref)


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
