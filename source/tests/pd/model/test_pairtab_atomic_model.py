# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

import numpy as np
import paddle

from deepmd.dpmodel.atomic_model import PairTabAtomicModel as DPPairTabAtomicModel
from deepmd.pd.model.atomic_model import (
    PairTabAtomicModel,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.utils import (
    to_numpy_array,
)


class TestPairTab(unittest.TestCase):
    @patch("numpy.loadtxt")
    def setUp(self, mock_loadtxt) -> None:
        file_path = "dummy_path"
        mock_loadtxt.return_value = np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
            ]
        )

        self.model = PairTabAtomicModel(
            tab_file=file_path, rcut=0.02, sel=2, type_map=["H", "O"]
        )

        self.extended_coord = paddle.to_tensor(
            [
                [
                    [0.01, 0.01, 0.01],
                    [0.01, 0.02, 0.01],
                    [0.01, 0.01, 0.02],
                    [0.02, 0.01, 0.01],
                ],
                [
                    [0.01, 0.01, 0.01],
                    [0.01, 0.02, 0.01],
                    [0.01, 0.01, 0.02],
                    [0.05, 0.01, 0.01],
                ],
            ],
            place=env.DEVICE,
        )

        # nframes=2, nall=4
        self.extended_atype = paddle.to_tensor(
            [[0, 1, 0, 1], [0, 0, 1, 1]], place=env.DEVICE
        )

        # nframes=2, nloc=2, nnei=2
        self.nlist = paddle.to_tensor(
            [[[1, 2], [0, 2]], [[1, 2], [0, 3]]], place=env.DEVICE
        )

    def test_without_mask(self):
        result = self.model.forward_atomic(
            self.extended_coord, self.extended_atype, self.nlist
        )
        expected_result = paddle.to_tensor(
            [[[1.2000], [1.3614]], [[1.2000], [0.4000]]],
            dtype=paddle.float64,
            place=env.DEVICE,
        )

        np.testing.assert_allclose(
            result["energy"].numpy(), expected_result.numpy(), rtol=0.0001, atol=0.0001
        )

    @unittest.skip("Temporarily skip")
    def test_with_mask(self):
        self.nlist = paddle.to_tensor(
            [[[1, -1], [0, 2]], [[1, 2], [0, 3]]], place=env.DEVICE
        )

        result = self.model.forward_atomic(
            self.extended_coord, self.extended_atype, self.nlist
        )
        expected_result = paddle.to_tensor(
            [[[0.8000], [1.3614]], [[1.2000], [0.4000]]],
            dtype=paddle.float64,
            place=env.DEVICE,
        )

        np.testing.assert_allclose(
            result["energy"].numpy(), expected_result.numpy(), rtol=0.0001, atol=0.0001
        )

    def test_jit(self):
        model = paddle.jit.to_static(self.model)
        # atomic model no more export methods
        # self.assertEqual(model.get_rcut(), 0.02)
        # self.assertEqual(model.get_type_map(), ["H", "O"])

    def test_deserialize(self):
        model1 = PairTabAtomicModel.deserialize(self.model.serialize())
        np.testing.assert_allclose(self.model.tab_data.numpy(), model1.tab_data.numpy())
        np.testing.assert_allclose(self.model.tab_info.numpy(), model1.tab_info.numpy())

        self.nlist = paddle.to_tensor(
            [[[1, -1], [0, 2]], [[1, 2], [0, 3]]], place=env.DEVICE
        )
        result = model1.forward_atomic(
            self.extended_coord, self.extended_atype, self.nlist
        )
        expected_result = self.model.forward_atomic(
            self.extended_coord, self.extended_atype, self.nlist
        )

        np.testing.assert_allclose(
            result["energy"].numpy(),
            expected_result["energy"].numpy(),
            rtol=0.0001,
            atol=0.0001,
        )

        # model1 = paddle.jit.to_static(model1)
        # atomic model no more export methods
        # self.assertEqual(model1.get_rcut(), 0.02)
        # self.assertEqual(model1.get_type_map(), ["H", "O"])

    def test_cross_deserialize(self):
        model_dict = self.model.serialize()  # paddle model to dict
        model1 = DPPairTabAtomicModel.deserialize(model_dict)  # dict to numpy model
        np.testing.assert_allclose(self.model.tab_data, model1.tab_data)
        np.testing.assert_allclose(self.model.tab_info, model1.tab_info)

        self.nlist = np.array([[[1, -1], [0, 2]], [[1, 2], [0, 3]]])
        result = model1.forward_atomic(
            self.extended_coord.cpu().numpy(),
            self.extended_atype.cpu().numpy(),
            self.nlist,
        )
        expected_result = self.model.forward_atomic(
            self.extended_coord,
            self.extended_atype,
            paddle.to_tensor(self.nlist).to(device=env.DEVICE),
        )
        np.testing.assert_allclose(
            result["energy"], to_numpy_array(expected_result["energy"]), 0.0001, 0.0001
        )


class TestPairTabTwoAtoms(unittest.TestCase):
    @patch("numpy.loadtxt")
    def test_extrapolation_nonzero_rmax(self, mock_loadtxt) -> None:
        """Scenarios to test.

        rcut < rmax:
            rr < rcut: use table values, or interpolate.
            rr == rcut: use table values, or interpolate.
            rr > rcut: should be 0
        rcut == rmax:
            rr < rcut: use table values, or interpolate.
            rr == rcut: use table values, or interpolate.
            rr > rcut: should be 0
        rcut > rmax:
            rr < rmax: use table values, or interpolate.
            rr == rmax: use table values, or interpolate.
            rmax < rr < rcut: extrapolate
            rr >= rcut: should be 0

        """
        file_path = "dummy_path"
        mock_loadtxt.return_value = np.array(
            [
                [0.005, 1.0],
                [0.01, 0.8],
                [0.015, 0.5],
                [0.02, 0.25],
            ]
        )

        # nframes=1, nall=2
        extended_atype = paddle.to_tensor([[0, 0]]).to(device=env.DEVICE)

        # nframes=1, nloc=2, nnei=1
        nlist = paddle.to_tensor([[[1], [-1]]]).to(device=env.DEVICE)

        results = []

        for dist, rcut in zip(
            [
                0.01,
                0.015,
                0.020,
                0.015,
                0.02,
                0.021,
                0.015,
                0.02,
                0.021,
                0.025,
                0.026,
                0.025,
                0.025,
                0.0216161,
            ],
            [
                0.015,
                0.015,
                0.015,
                0.02,
                0.02,
                0.02,
                0.022,
                0.022,
                0.022,
                0.025,
                0.025,
                0.03,
                0.035,
                0.025,
            ],
        ):
            extended_coord = paddle.to_tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, dist, 0.0],
                    ],
                ],
                place=env.DEVICE,
            )

            model = PairTabAtomicModel(
                tab_file=file_path, rcut=rcut, sel=2, type_map=["H"]
            )
            results.append(
                model.forward_atomic(extended_coord, extended_atype, nlist)["energy"]
            )

        expected_result = paddle.stack(
            [
                paddle.to_tensor(
                    [
                        [
                            [0.4, 0],
                            [0.0, 0],
                            [0.0, 0],
                            [0.25, 0],
                            [0, 0],
                            [0, 0],
                            [0.25, 0],
                            [0.125, 0],
                            [0.0922, 0],
                            [0, 0],
                            [0, 0],
                            [0, 0],
                            [0.0923, 0],
                            [0.0713, 0],
                        ]
                    ],
                    dtype=paddle.float64,
                    place=env.DEVICE,
                )
            ]
        ).reshape([14, 2])
        results = paddle.stack(results).reshape([14, 2])

        np.testing.assert_allclose(results, expected_result, rtol=0.0001, atol=0.0001)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
