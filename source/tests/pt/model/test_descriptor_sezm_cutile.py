# SPDX-License-Identifier: LGPL-3.0-or-later
"""Correctness of the cuTile SeZM inference kernels.

Each kernel is checked against the eager reference that defines it, on shapes
small enough to run quickly but large enough to exercise partial edge tiles and
multi-iteration segment walks. The tolerances follow the arithmetic: the mixing
stack uses split-compensated fp16 tensor cores and is held to 1e-5 relative,
every other kernel is plain fp32 and is held to 1e-6.
"""

from __future__ import (
    annotations,
)

import unittest

import torch

from deepmd.pt.utils import (
    env,
)
from deepmd.pt_expt.kernels.cutile.common import (
    CUTILE_AVAILABLE,
)

if CUTILE_AVAILABLE and torch.cuda.is_available():
    from deepmd.pt_expt.kernels.cutile.sezm.flash_atten import (
        _launch_backward as flash_aggregate_backward,
    )
    from deepmd.pt_expt.kernels.cutile.sezm.flash_atten import (
        _launch_forward as flash_aggregate,
    )
    from deepmd.pt_expt.kernels.cutile.sezm.flash_atten import (
        build_row_ptr,
    )
    from deepmd.pt_expt.kernels.cutile.sezm.force_assembly import (
        _launch_forward as edge_force_assembly,
    )
    from deepmd.pt_expt.kernels.cutile.sezm.indexing import (
        SO2TileLayout,
        m_major_index,
        rotation_pairs,
    )
    from deepmd.pt_expt.kernels.cutile.sezm.so2_mixing_stack import (
        _launch_backward as mixing_stack_backward,
    )
    from deepmd.pt_expt.kernels.cutile.sezm.so2_mixing_stack import (
        _launch_forward as mixing_stack,
    )
    from deepmd.pt_expt.kernels.cutile.sezm.so2_mixing_stack import (
        pack_weights,
    )
    from deepmd.pt_expt.kernels.cutile.sezm.so2_rotate_mix import (
        _launch_backward as rotate_mix_backward,
    )
    from deepmd.pt_expt.kernels.cutile.sezm.so2_rotate_mix import (
        _launch_forward as rotate_mix,
    )
    from deepmd.pt_expt.kernels.cutile.sezm.wigner_monomials import (
        _launch_backward as monomials_backward,
    )
    from deepmd.pt_expt.kernels.cutile.sezm.wigner_monomials import (
        _launch_forward as monomials,
    )
    from deepmd.pt_expt.kernels.triton.sezm.flash_atten import (
        _flash_atten_backward_reference,
        flash_atten_aggregate_reference,
    )
    from deepmd.pt_expt.kernels.triton.sezm.force_assembly import (
        _force_assembly_reference,
    )
    from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
        _mixing_stack_backward_reference,
        _mixing_stack_reference,
        _rotate_mix_backward_reference,
        _rotate_mix_reference,
    )
    from deepmd.pt_expt.kernels.triton.sezm.wigner_monomials import (
        _monomials_backward_reference,
        _monomials_reference,
    )

CUTILE_READY = CUTILE_AVAILABLE and torch.cuda.is_available()

LMAX = 2
FOCUS_DIM = 32
N_FOCUS = 1
N_LAYERS = 3
N_NODE = 96
N_EDGE = 611


def _relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got - want).abs().max() / want.abs().max()).item()


def _block_diagonal_mask(wigner: torch.Tensor, lmax: int) -> torch.Tensor:
    """Structural support of a block-diagonal Wigner-D stack."""
    mask = torch.zeros_like(wigner, dtype=torch.bool)
    for degree in range(lmax + 1):
        lo, hi = degree * degree, (degree + 1) ** 2
        mask[:, lo:hi, lo:hi] = True
    return mask


@unittest.skipUnless(CUTILE_READY, "cuda.tile and a CUDA device are required")
class TestSO2TileLayout(unittest.TestCase):
    """The reduced layout every kernel of the package addresses."""

    def test_reduced_rows_are_the_low_order_basis_indices(self) -> None:
        self.assertEqual(m_major_index(2), [0, 2, 6, 1, 5, 3, 7])

    def test_rotation_pairs_cover_each_row_own_degree_block(self) -> None:
        pairs = rotation_pairs(2)
        self.assertEqual(len(pairs), 1 + 3 + 5 + 3 + 5 + 3 + 5)
        for reduced, full in pairs:
            degree = int(m_major_index(2)[reduced] ** 0.5)
            self.assertTrue(degree * degree <= full < (degree + 1) ** 2)

    def test_padded_widths_are_powers_of_two(self) -> None:
        layout = SO2TileLayout(lmax=3, focus_dim=64, n_layers=3)
        self.assertEqual((layout.n_m0, layout.pad_m0), (4, 4))
        self.assertEqual((layout.n_m1, layout.pad_m1), (6, 8))
        self.assertEqual(layout.row, 10 * 64)


@unittest.skipUnless(CUTILE_READY, "cuda.tile and a CUDA device are required")
class TestRotateMix(unittest.TestCase):
    """Rotation into the edge frame followed by the radial degree mixing."""

    @classmethod
    def setUpClass(cls) -> None:
        generator = torch.Generator(device=env.DEVICE).manual_seed(20240611)

        def normal(*shape: int, scale: float = 1.0) -> torch.Tensor:
            return torch.randn(*shape, generator=generator, device=env.DEVICE) * scale

        cls.layout = SO2TileLayout(lmax=LMAX, focus_dim=FOCUS_DIM, n_layers=N_LAYERS)
        cls.x = normal(N_NODE, cls.layout.dim, N_FOCUS * FOCUS_DIM)
        cls.src = torch.randint(
            0, N_NODE, (N_EDGE,), generator=generator, device=env.DEVICE
        )
        cls.wigner = torch.zeros(
            N_EDGE, cls.layout.dim, cls.layout.dim, device=env.DEVICE
        )
        for degree in range(LMAX + 1):
            lo, hi = degree * degree, (degree + 1) ** 2
            cls.wigner[:, lo:hi, lo:hi] = normal(N_EDGE, hi - lo, hi - lo, scale=0.6)
        cls.mixer = normal(N_EDGE, cls.layout.kernel_size, scale=0.5)
        cls.channel = normal(N_FOCUS * FOCUS_DIM, scale=0.5)

    def _run_forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        want = _rotate_mix_reference(
            self.x, self.src, self.wigner, self.mixer, self.channel, LMAX, N_FOCUS, 1
        )
        got = rotate_mix(
            self.x,
            self.src,
            self.wigner,
            self.mixer,
            self.channel,
            self.layout,
            N_FOCUS,
        )
        return got, want

    def test_forward_matches_the_dense_reference(self) -> None:
        got, want = self._run_forward()
        self.assertLess(_relative_error(got, want), 1e-6)

    def test_backward_matches_the_closed_form_reference(self) -> None:
        _, want = self._run_forward()
        grad_out = torch.randn_like(want)
        want_edge, want_wigner, want_mixer = _rotate_mix_backward_reference(
            grad_out,
            self.x,
            self.src,
            self.wigner,
            self.mixer,
            self.channel,
            LMAX,
            N_FOCUS,
            1,
        )
        # The kernel reduces onto source nodes internally, so the per-edge
        # reference gradient is scattered before the comparison.
        want_node = torch.zeros_like(self.x).index_add(0, self.src, want_edge)
        order = torch.argsort(self.src)
        row_ptr = build_row_ptr(self.src.index_select(0, order), N_NODE)
        got_node, got_wigner, got_mixer = rotate_mix_backward(
            grad_out,
            self.x,
            order,
            row_ptr,
            self.wigner,
            self.mixer,
            self.channel,
            self.layout,
            N_FOCUS,
        )
        self.assertLess(_relative_error(got_node, want_node), 1e-6)
        self.assertLess(_relative_error(got_mixer, want_mixer), 1e-6)
        # The rotation gradient is compared on the structural block diagonal.
        # Outside it the reference differentiates coefficients the Wigner
        # construction writes as constants, so its own backward discards them
        # and the kernel does not compute them.
        mask = _block_diagonal_mask(self.wigner, LMAX)
        self.assertLess(_relative_error(got_wigner[mask], want_wigner[mask]), 1e-6)

    def test_segments_with_no_edges_produce_a_zero_gradient(self) -> None:
        """A source node absent from the edge list must still be written."""
        _, forward = self._run_forward()
        grad_out = torch.randn_like(forward)
        order = torch.argsort(self.src)
        row_ptr = build_row_ptr(self.src.index_select(0, order), N_NODE)
        got_node, _, _ = rotate_mix_backward(
            grad_out,
            self.x,
            order,
            row_ptr,
            self.wigner,
            self.mixer,
            self.channel,
            self.layout,
            N_FOCUS,
        )
        absent = torch.ones(N_NODE, dtype=torch.bool, device=env.DEVICE)
        absent[self.src] = False
        self.assertTrue(torch.all(got_node[absent] == 0.0))


@unittest.skipUnless(CUTILE_READY, "cuda.tile and a CUDA device are required")
class TestMixingStack(unittest.TestCase):
    """The complete gated SO(2) mixing stack against an fp64 ground truth."""

    @classmethod
    def setUpClass(cls) -> None:
        generator = torch.Generator(device=env.DEVICE).manual_seed(20240612)

        def normal(*shape: int, scale: float = 1.0) -> torch.Tensor:
            return torch.randn(*shape, generator=generator, device=env.DEVICE) * scale

        cls.layout = SO2TileLayout(lmax=LMAX, focus_dim=FOCUS_DIM, n_layers=N_LAYERS)
        width0 = cls.layout.n_m0 * FOCUS_DIM
        width1 = cls.layout.n_m1 * FOCUS_DIM
        cls.u0 = normal(N_FOCUS, N_EDGE, cls.layout.row)
        cls.alpha = torch.ones(N_EDGE, N_FOCUS, device=env.DEVICE)
        cls.w0 = normal(N_LAYERS, N_FOCUS, width0, width0, scale=width0**-0.5)
        cls.w1 = normal(N_LAYERS, N_FOCUS, width1, width1, scale=width1**-0.5)
        cls.gw = normal(
            N_LAYERS - 1, N_FOCUS, FOCUS_DIM, LMAX * FOCUS_DIM, scale=FOCUS_DIM**-0.5
        )
        cls.packed = pack_weights(cls.w0, cls.w1, cls.gw, cls.layout)
        # The third output is the final gated activation, which the backward
        # reference reads as its anchor.
        cls.want, cls.pre_activation, cls.u_final = _mixing_stack_reference(
            cls.u0.double(),
            cls.alpha.double(),
            cls.w0.double(),
            cls.w1.double(),
            cls.gw.double(),
            LMAX,
            FOCUS_DIM,
            False,
        )

    def test_forward_matches_the_fp64_reference(self) -> None:
        got = mixing_stack(self.u0, self.packed, self.layout)
        self.assertLess(_relative_error(got.double(), self.want), 1e-5)

    def test_backward_matches_the_fp64_reference(self) -> None:
        grad_out = torch.randn(N_EDGE, N_FOCUS, self.layout.row, device=env.DEVICE)
        # Only the input gradient is compared here. The reference also returns
        # the parameter gradients and the per-layer surfaces the training path
        # linearizes around, and it takes the upstream cotangents of those
        # surfaces, which a first-order check does not supply.
        want = _mixing_stack_backward_reference(
            grad_out.double(),
            self.want,
            self.pre_activation,
            self.u_final,
            self.alpha.double(),
            self.w0.double().transpose(-1, -2).contiguous(),
            self.w1.double().transpose(-1, -2).contiguous(),
            self.gw.double(),
            self.gw.double().transpose(-1, -2).contiguous(),
            None,
            None,
            LMAX,
            FOCUS_DIM,
            False,
        )[0]
        got = mixing_stack_backward(self.u0, grad_out, self.packed, self.layout)
        self.assertLess(_relative_error(got.double(), want), 1e-5)

    def test_split_representation_recovers_the_fp32_weight(self) -> None:
        from deepmd.pt_expt.kernels.cutile.common import (
            TAIL_SCALE,
        )

        recovered = self.packed["w0h"].float() + self.packed["w0l"].float() / TAIL_SCALE
        padded = torch.zeros_like(recovered)
        span = self.layout.n_m0 * FOCUS_DIM
        padded[:, :, :span, :span] = self.w0
        self.assertLess(_relative_error(recovered, padded), 1e-6)


@unittest.skipUnless(CUTILE_READY, "cuda.tile and a CUDA device are required")
class TestFlashAggregation(unittest.TestCase):
    """Inverse rotation, attention weighting and the destination reduction."""

    @classmethod
    def setUpClass(cls) -> None:
        generator = torch.Generator(device=env.DEVICE).manual_seed(20240613)

        def normal(*shape: int, scale: float = 1.0) -> torch.Tensor:
            return torch.randn(*shape, generator=generator, device=env.DEVICE) * scale

        cls.layout = SO2TileLayout(lmax=LMAX, focus_dim=FOCUS_DIM, n_layers=N_LAYERS)
        n_row = 3 * LMAX + 1
        cls.x_local = normal(N_EDGE, N_FOCUS, n_row, FOCUS_DIM)
        wigner = torch.zeros(N_EDGE, cls.layout.dim, cls.layout.dim, device=env.DEVICE)
        for degree in range(LMAX + 1):
            lo, hi = degree * degree, (degree + 1) ** 2
            wigner[:, lo:hi, lo:hi] = normal(N_EDGE, hi - lo, hi - lo, scale=0.6)
        cls.wigner_t = wigner.transpose(1, 2).contiguous()
        cls.alpha = torch.rand(
            N_EDGE, N_FOCUS, 1, generator=generator, device=env.DEVICE
        )
        cls.rescale = (
            torch.rand(cls.layout.dim, generator=generator, device=env.DEVICE) + 0.5
        )
        cls.dst = torch.randint(
            0, N_NODE, (N_EDGE,), generator=generator, device=env.DEVICE
        )
        cls.order = torch.argsort(cls.dst)
        cls.row_ptr = build_row_ptr(cls.dst.index_select(0, cls.order), N_NODE)

    def test_forward_matches_the_dense_reference(self) -> None:
        want = flash_atten_aggregate_reference(
            self.x_local,
            self.wigner_t,
            self.rescale,
            self.alpha,
            self.dst,
            N_NODE,
            LMAX,
            1,
        )
        got = flash_aggregate(
            self.x_local,
            self.wigner_t,
            tuple(self.rescale.tolist()),
            self.alpha,
            self.order,
            self.row_ptr,
            self.layout,
            N_FOCUS,
            1,
        )
        self.assertLess(_relative_error(got, want), 1e-6)

    def test_backward_matches_the_closed_form_reference(self) -> None:
        grad_out = torch.randn(
            N_NODE, self.layout.dim, N_FOCUS * FOCUS_DIM, device=env.DEVICE
        )
        want_local, want_wigner, want_alpha = _flash_atten_backward_reference(
            grad_out,
            self.x_local,
            self.wigner_t,
            self.rescale,
            self.alpha,
            self.dst,
            LMAX,
            1,
        )
        got_local, got_wigner, got_alpha = flash_aggregate_backward(
            grad_out,
            self.x_local,
            self.wigner_t,
            tuple(self.rescale.tolist()),
            self.alpha,
            self.dst,
            self.layout,
            N_FOCUS,
            1,
        )
        self.assertLess(_relative_error(got_local, want_local), 1e-6)
        self.assertLess(_relative_error(got_alpha, want_alpha), 1e-6)
        mask = _block_diagonal_mask(self.wigner_t, LMAX)
        self.assertLess(_relative_error(got_wigner[mask], want_wigner[mask]), 1e-6)


@unittest.skipUnless(CUTILE_READY, "cuda.tile and a CUDA device are required")
class TestWignerMonomials(unittest.TestCase):
    """Quaternion monomial basis and its analytic gradient."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.exponents = [
            e
            for a in range(5)
            for b in range(5 - a)
            for c in range(5 - a - b)
            for e in (a, b, c, 4 - a - b - c)
        ]
        cls.quaternion = torch.randn(N_EDGE, 4, device=env.DEVICE)

    def test_forward_matches_the_power_ladder_reference(self) -> None:
        want = _monomials_reference(self.quaternion, self.exponents, 4)
        got = monomials(self.quaternion, self.exponents, 4)
        self.assertLess(_relative_error(got, want), 1e-6)

    def test_backward_matches_the_leave_one_out_reference(self) -> None:
        want_forward = _monomials_reference(self.quaternion, self.exponents, 4)
        grad_out = torch.randn_like(want_forward)
        want = _monomials_backward_reference(
            grad_out, self.quaternion, self.exponents, 4
        )
        got = monomials_backward(grad_out, self.quaternion, self.exponents, 4)
        self.assertLess(_relative_error(got, want), 1e-6)


@unittest.skipUnless(CUTILE_READY, "cuda.tile and a CUDA device are required")
class TestForceAssembly(unittest.TestCase):
    """Force and per-atom virial segment reduction over both endpoints."""

    def test_matches_the_index_add_reference(self) -> None:
        generator = torch.Generator(device=env.DEVICE).manual_seed(20240614)
        grad = torch.randn(N_EDGE, 3, generator=generator, device=env.DEVICE)
        edge_vec = torch.randn(N_EDGE, 3, generator=generator, device=env.DEVICE)
        dst = torch.randint(
            0, N_NODE, (N_EDGE,), generator=generator, device=env.DEVICE
        )
        src = torch.randint(
            0, N_NODE, (N_EDGE,), generator=generator, device=env.DEVICE
        )
        dst_order, src_order = torch.argsort(dst), torch.argsort(src)
        dst_row_ptr = build_row_ptr(dst.index_select(0, dst_order), N_NODE).long()
        src_row_ptr = build_row_ptr(src.index_select(0, src_order), N_NODE).long()
        want_force, want_virial = _force_assembly_reference(
            grad, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr
        )
        got_force, got_virial = edge_force_assembly(
            grad, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr
        )
        self.assertLess(_relative_error(got_force, want_force), 1e-6)
        self.assertLess(_relative_error(got_virial, want_virial), 1e-6)


@unittest.skipUnless(CUTILE_READY, "cuda.tile and a CUDA device are required")
class TestValuePathSupport(unittest.TestCase):
    """The factory must bind the deployed layout and decline others cleanly.

    The predicate runs at construction for every convolution whenever the gate is
    enabled, so a layout it does not serve has to return ``None`` rather than
    raise: the caller's contract is to fall back to the dense reference.
    """

    @staticmethod
    def _convolution(**overrides) -> object:
        from deepmd.pt.model.descriptor.sezm_nn.so2 import (
            SO2Convolution,
        )

        options = {
            "lmax": LMAX,
            "mmax": 1,
            "channels": FOCUS_DIM,
            "n_focus": N_FOCUS,
            "mixing_layers": N_LAYERS,
            "radial_so2_mode": "degree_channel",
            "radial_so2_rank": 1,
            "n_atten_head": 1,
            "dtype": torch.float32,
            "seed": 0,
            "trainable": True,
        }
        options.update(overrides)
        return SO2Convolution(**options)

    def test_deployed_layout_is_served(self) -> None:
        from deepmd.pt_expt.kernels.cutile.sezm.so2_value_path import (
            make_cutile_value_path,
        )

        self.assertIsNotNone(make_cutile_value_path(self._convolution()))

    def test_unsupported_layouts_decline_without_raising(self) -> None:
        from deepmd.pt_expt.kernels.cutile.sezm.so2_value_path import (
            make_cutile_value_path,
        )

        for description, overrides in (
            ("order beyond one", {"mmax": 2}),
            ("focus width not a power of two", {"channels": 96}),
            ("no gated layer", {"mixing_layers": 1}),
            ("no radial degree mixer", {"radial_so2_mode": "none"}),
            ("degree-only radial mixer", {"radial_so2_mode": "degree"}),
            ("non-unit mixer rank", {"radial_so2_rank": 2}),
            ("Cartesian edge frame", {"edge_cartesian": True}),
        ):
            with self.subTest(description):
                self.assertIsNone(
                    make_cutile_value_path(self._convolution(**overrides))
                )


if __name__ == "__main__":
    unittest.main()
