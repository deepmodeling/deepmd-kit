# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

import numpy as np

from deepmd.dpmodel.atomic_model.pairtab_atomic_model import (
    PairTabAtomicModel,
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

        self.extended_coord = np.array(
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
            ]
        )

        # nframes=2, nall=4
        self.extended_atype = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])

        # nframes=2, nloc=2, nnei=2
        self.nlist = np.array([[[1, 2], [0, 2]], [[1, 2], [0, 3]]])

    def test_without_mask(self) -> None:
        result = self.model.forward_atomic(
            self.extended_coord, self.extended_atype, self.nlist
        )
        expected_result = np.array([[[1.2000], [1.3614]], [[1.2000], [0.4000]]])

        np.testing.assert_allclose(result["energy"], expected_result, 0.0001, 0.0001)

    def test_with_mask(self) -> None:
        self.nlist = np.array([[[1, -1], [0, 2]], [[1, 2], [0, 3]]])

        result = self.model.forward_atomic(
            self.extended_coord, self.extended_atype, self.nlist
        )
        expected_result = np.array([[[0.8000], [1.3614]], [[1.2000], [0.4000]]])

        np.testing.assert_allclose(result["energy"], expected_result, 0.0001, 0.0001)

    def test_deserialize(self) -> None:
        model1 = PairTabAtomicModel.deserialize(self.model.serialize())
        np.testing.assert_allclose(self.model.tab_data, model1.tab_data)
        np.testing.assert_allclose(self.model.tab_info, model1.tab_info)

        self.nlist = np.array([[[1, -1], [0, 2]], [[1, 2], [0, 3]]])
        result = model1.forward_atomic(
            self.extended_coord, self.extended_atype, self.nlist
        )
        expected_result = self.model.forward_atomic(
            self.extended_coord, self.extended_atype, self.nlist
        )

        np.testing.assert_allclose(
            result["energy"], expected_result["energy"], 0.0001, 0.0001
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
        extended_atype = np.array([[0, 0]])

        # nframes=1, nloc=2, nnei=1
        nlist = np.array([[[1], [-1]]])

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
            extended_coord = np.array(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, dist, 0.0],
                    ],
                ]
            )

            model = PairTabAtomicModel(
                tab_file=file_path, rcut=rcut, sel=2, type_map=["S"]
            )
            results.append(
                model.forward_atomic(extended_coord, extended_atype, nlist)["energy"]
            )

        expected_result = np.stack(
            [
                np.array(
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
                    ]
                )
            ]
        ).reshape(14, 2)
        results = np.stack(results).reshape(14, 2)

        np.testing.assert_allclose(results, expected_result, 0.0001, 0.0001)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
