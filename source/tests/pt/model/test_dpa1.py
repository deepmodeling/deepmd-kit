# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import math
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DPDescrptDPA1
from deepmd.pt.model.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt.model.descriptor.se_atten import (
    _build_degree_weights,
    _build_moment_basis,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)
from .test_mlp import (
    get_tols,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestDescrptSeAtten(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_get_numb_attn_layer(self) -> None:
        """Cover both code paths: attn_layer == 0 and attn_layer > 0."""
        dd0 = DescrptDPA1(
            self.rcut, self.rcut_smth, self.sel_mix, self.nt, attn_layer=0
        ).to(env.DEVICE)
        self.assertEqual(dd0.get_numb_attn_layer(), 0)
        dd2 = DescrptDPA1(
            self.rcut, self.rcut_smth, self.sel_mix, self.nt, attn_layer=2
        ).to(env.DEVICE)
        self.assertEqual(dd2.get_numb_attn_layer(), 2)

    def test_consistency(
        self,
    ) -> None:
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for idt, sm, to, tm, prec, ect in itertools.product(
            [False, True],  # resnet_dt
            [False, True],  # smooth_type_embedding
            [False, True],  # type_one_side
            ["concat", "strip"],  # tebd_input_mode
            [
                "float64",
            ],  # precision
            [False, True],  # use_econf_tebd
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            err_msg = f"idt={idt} prec={prec}"

            # dpa1 new impl
            dd0 = DescrptDPA1(
                self.rcut,
                self.rcut_smth,
                self.sel_mix,
                self.nt,
                attn_layer=2,
                precision=prec,
                resnet_dt=idt,
                smooth_type_embedding=sm,
                type_one_side=to,
                tebd_input_mode=tm,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)
            dd0.se_atten.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.se_atten.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            rd0, _, _, _, _ = dd0(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            )
            # serialization
            dd1 = DescrptDPA1.deserialize(dd0.serialize())
            rd1, _, _, _, _ = dd1(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd1.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )
            # dp impl. `mapping` is passed because that is the production
            # invocation (DPAtomicModel.forward_atomic always forwards it);
            # the dense `.call()` must give the same answer with or without
            # it -- mapping only enables ghost folding on graph routes, it
            # must never change the dense numerics.
            dd2 = DPDescrptDPA1.deserialize(dd0.serialize())
            rd2, _, _, _, _ = dd2.call(
                self.coord_ext,
                self.atype_ext,
                self.nlist,
                self.mapping,
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd2,
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )

    def test_jit(
        self,
    ) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for idt, prec, sm, to, tm, ect in itertools.product(
            [
                False,
            ],  # resnet_dt
            [
                "float64",
            ],  # precision
            [False, True],  # smooth_type_embedding
            [
                False,
            ],  # type_one_side
            ["concat", "strip"],  # tebd_input_mode
            [False, True],  # use_econf_tebd
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            err_msg = f"idt={idt} prec={prec}"
            # dpa1 new impl
            dd0 = DescrptDPA1(
                self.rcut,
                self.rcut_smth,
                self.sel,
                self.nt,
                precision=prec,
                resnet_dt=idt,
                smooth_type_embedding=sm,
                type_one_side=to,
                tebd_input_mode=tm,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            )
            dd0.se_atten.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.se_atten.dstd = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            # dd1 = DescrptDPA1.deserialize(dd0.serialize())
            model = torch.jit.script(dd0)
            # model = torch.jit.script(dd1)


class TestDPA1AngularMoments(unittest.TestCase):
    """Test the dense PyTorch angular moment basis of DPA1."""

    dtype = torch.float64

    @staticmethod
    def _build_descriptor(lmax: int) -> DescrptDPA1:
        return DescrptDPA1(
            rcut=3.0,
            rcut_smth=2.5,
            sel=4,
            ntypes=1,
            neuron=[4, 8, 8],
            axis_neuron=4,
            lmax=lmax,
            tebd_dim=2,
            tebd_input_mode="strip",
            set_davg_zero=True,
            attn_layer=0,
            precision="float64",
            concat_output_tebd=False,
            seed=11,
        ).to(env.DEVICE)

    @classmethod
    def _evaluate(
        cls,
        descriptor: DescrptDPA1,
        neighbors: torch.Tensor,
        *,
        requires_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coord = torch.cat(
            [
                torch.zeros(
                    (1, 3),
                    dtype=cls.dtype,
                    device=env.DEVICE,
                ),
                neighbors,
            ]
        ).reshape(1, -1)
        coord.requires_grad_(requires_grad)
        atype = torch.zeros((1, 5), dtype=torch.long, device=env.DEVICE)
        nlist = torch.tensor(
            [[[1, 2, 3, 4]]],
            dtype=torch.long,
            device=env.DEVICE,
        )
        result = descriptor(coord, atype, nlist)[0]
        return result, coord

    @classmethod
    def _square_directions(cls) -> torch.Tensor:
        return torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=cls.dtype,
            device=env.DEVICE,
        )

    @classmethod
    def _tetrahedral_directions(cls) -> torch.Tensor:
        return torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=cls.dtype,
            device=env.DEVICE,
        ) / math.sqrt(3.0)

    def test_higher_degree_addition_theorem(self) -> None:
        generator = torch.Generator(device=env.DEVICE).manual_seed(37)
        left = torch.randn(
            32,
            3,
            dtype=self.dtype,
            device=env.DEVICE,
            generator=generator,
        )
        right = torch.randn(
            32,
            3,
            dtype=self.dtype,
            device=env.DEVICE,
            generator=generator,
        )
        left = torch.nn.functional.normalize(left, dim=-1)
        right = torch.nn.functional.normalize(right, dim=-1)
        rr = torch.zeros(32, 1, 4, dtype=self.dtype, device=env.DEVICE)
        radial = torch.ones(32, 1, 1, dtype=self.dtype, device=env.DEVICE)
        left_basis = _build_moment_basis(rr, left[:, None, :], radial, 4)
        right_basis = _build_moment_basis(rr, right[:, None, :], radial, 4)
        cosine = torch.sum(left * right, dim=-1)
        expected = {
            2: 0.5 * (3.0 * cosine**2 - 1.0),
            3: 0.5 * (5.0 * cosine**3 - 3.0 * cosine),
            4: 0.125 * (35.0 * cosine**4 - 30.0 * cosine**2 + 3.0),
        }
        for degree in range(2, 5):
            current = torch.sum(
                left_basis[:, 0, degree * degree : (degree + 1) ** 2]
                * right_basis[:, 0, degree * degree : (degree + 1) ** 2],
                dim=-1,
            )
            torch.testing.assert_close(
                current, expected[degree], atol=1e-12, rtol=1e-12
            )

    def test_tetrahedral_and_octahedral_degree_signatures(self) -> None:
        tetrahedral = self._tetrahedral_directions()
        octahedral = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=self.dtype,
            device=env.DEVICE,
        )

        def moments(direction: torch.Tensor) -> torch.Tensor:
            rr = torch.zeros(
                direction.shape[0],
                1,
                4,
                dtype=self.dtype,
                device=env.DEVICE,
            )
            radial = torch.ones(
                direction.shape[0],
                1,
                1,
                dtype=self.dtype,
                device=env.DEVICE,
            )
            return _build_moment_basis(rr, direction[:, None, :], radial, 4).sum(dim=0)[
                0
            ]

        tetrahedral_moment = moments(tetrahedral)
        octahedral_moment = moments(octahedral)
        torch.testing.assert_close(
            tetrahedral_moment[4:9],
            torch.zeros_like(tetrahedral_moment[4:9]),
            atol=1e-12,
            rtol=0.0,
        )
        self.assertGreater(
            torch.linalg.vector_norm(tetrahedral_moment[9:16]).item(), 1.0
        )
        torch.testing.assert_close(
            octahedral_moment[1:16],
            torch.zeros_like(octahedral_moment[1:16]),
            atol=1e-12,
            rtol=0.0,
        )
        self.assertGreater(
            torch.linalg.vector_norm(octahedral_moment[16:25]).item(), 1.0
        )

    def test_degree_gain_zero_recovers_lmax_one(self) -> None:
        neighbors = self._square_directions()
        degree_one = self._build_descriptor(lmax=1).eval()
        degree_four = self._build_descriptor(lmax=4).eval()
        assert degree_four.se_atten.adam_degree_gain_raw is not None
        degree_four.se_atten.adam_degree_gain_raw.data.zero_()
        result_one, _ = self._evaluate(degree_one, neighbors)
        result_four, _ = self._evaluate(degree_four, neighbors)
        torch.testing.assert_close(result_four, result_one, atol=1e-12, rtol=1e-12)

        raw_gain = degree_four.se_atten.adam_degree_gain_raw
        raw_gain.data.copy_(
            torch.tensor([0.1, 0.2, 0.3], dtype=self.dtype, device=env.DEVICE)
        )
        degree_weights = _build_degree_weights(raw_gain, 4, result_four)
        self.assertTrue(bool(torch.all(degree_weights >= 0.0)))
        output, _ = self._evaluate(degree_four, neighbors)
        (gradient,) = torch.autograd.grad(output.sum(), raw_gain)
        self.assertGreater(torch.linalg.vector_norm(gradient).item(), 0.0)

    def test_lmax_two_resolves_quadrupole_collision(self) -> None:
        square = self._square_directions()
        tetrahedral = self._tetrahedral_directions()

        degree_one = self._build_descriptor(lmax=1).eval()
        square_l1, _ = self._evaluate(degree_one, square)
        tetrahedral_l1, _ = self._evaluate(degree_one, tetrahedral)
        torch.testing.assert_close(square_l1, tetrahedral_l1, atol=1e-12, rtol=1e-12)

        degree_two = self._build_descriptor(lmax=2).eval()
        square_l2, _ = self._evaluate(degree_two, square)
        tetrahedral_l2, _ = self._evaluate(degree_two, tetrahedral)
        self.assertGreater(
            torch.linalg.vector_norm(square_l2 - tetrahedral_l2).item(),
            1e-8,
        )

        restored = DescrptDPA1.deserialize(degree_two.serialize()).to(env.DEVICE).eval()
        restored_square, _ = self._evaluate(restored, square)
        torch.testing.assert_close(restored_square, square_l2)
        scripted_square, _ = self._evaluate(torch.jit.script(degree_two), square)
        torch.testing.assert_close(scripted_square, square_l2)

    def test_higher_degrees_are_rotation_and_permutation_invariant(self) -> None:
        neighbors = torch.tensor(
            [
                [1.1, 0.2, -0.1],
                [-0.4, 0.9, 0.3],
                [0.2, -0.5, 1.2],
                [-0.7, -0.3, -0.8],
            ],
            dtype=self.dtype,
            device=env.DEVICE,
        )
        rotation = torch.tensor(
            [
                [-2.0 / 3.0, 2.0 / 15.0, 11.0 / 15.0],
                [2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
                [1.0 / 3.0, 14.0 / 15.0, 2.0 / 15.0],
            ],
            dtype=self.dtype,
            device=env.DEVICE,
        )

        for lmax in (2, 3, 4):
            with self.subTest(lmax=lmax):
                descriptor = self._build_descriptor(lmax=lmax).eval()
                descriptor.se_atten.mean[..., 0] = 0.25
                descriptor.se_atten.stddev[..., 0] = 1.75
                reference, _ = self._evaluate(descriptor, neighbors)
                rotated, _ = self._evaluate(descriptor, neighbors @ rotation.T)
                permuted, _ = self._evaluate(
                    descriptor,
                    neighbors[[2, 0, 3, 1]],
                )
                torch.testing.assert_close(
                    rotated,
                    reference,
                    atol=1e-10,
                    rtol=1e-10,
                )
                torch.testing.assert_close(
                    permuted,
                    reference,
                    atol=1e-10,
                    rtol=1e-10,
                )

    def test_higher_degree_coordinate_derivatives_are_finite(self) -> None:
        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [-0.3, 0.9, -0.2]]],
            dtype=self.dtype,
            device=env.DEVICE,
            requires_grad=True,
        )
        atype = torch.zeros((1, 3), dtype=torch.long, device=env.DEVICE)
        nlist = torch.tensor(
            [[[1, 2, -1, -1]]],
            dtype=torch.long,
            device=env.DEVICE,
        )
        for lmax in (2, 3, 4):
            with self.subTest(lmax=lmax):
                descriptor = self._build_descriptor(lmax=lmax)
                assert descriptor.se_atten.adam_degree_gain_raw is not None
                descriptor.se_atten.adam_degree_gain_raw.data.copy_(
                    torch.tensor(
                        [0.7, -0.5, 0.9][: lmax - 1],
                        dtype=self.dtype,
                        device=env.DEVICE,
                    )
                )
                current_coord = coord.detach().clone().requires_grad_(True)
                result = descriptor(current_coord.reshape(1, -1), atype, nlist)[0]
                cotangent = torch.linspace(
                    -0.8,
                    1.1,
                    result.numel(),
                    dtype=self.dtype,
                    device=env.DEVICE,
                ).reshape_as(result)
                (first_derivative,) = torch.autograd.grad(
                    (result * cotangent).sum(),
                    current_coord,
                    create_graph=True,
                )
                epsilon = 1e-6
                finite_difference = torch.empty_like(current_coord)
                flat_coord = current_coord.detach().reshape(-1)
                for index in range(flat_coord.numel()):
                    positive = flat_coord.clone()
                    negative = flat_coord.clone()
                    positive[index] += epsilon
                    negative[index] -= epsilon
                    positive_value = descriptor(
                        positive.reshape(1, -1),
                        atype,
                        nlist,
                    )[0]
                    negative_value = descriptor(
                        negative.reshape(1, -1),
                        atype,
                        nlist,
                    )[0]
                    finite_difference.reshape(-1)[index] = (
                        (positive_value * cotangent).sum()
                        - (negative_value * cotangent).sum()
                    ) / (2.0 * epsilon)
                (second_derivative,) = torch.autograd.grad(
                    first_derivative.square().sum(),
                    current_coord,
                )
                torch.testing.assert_close(
                    first_derivative,
                    finite_difference,
                    atol=2e-8,
                    rtol=2e-8,
                )
                self.assertTrue(torch.isfinite(first_derivative).all())
                self.assertTrue(torch.isfinite(second_derivative).all())
                self.assertGreater(first_derivative.abs().max().item(), 0.0)
                self.assertGreater(second_derivative.abs().max().item(), 0.0)
