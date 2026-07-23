# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import array_api_strict as xp

from deepmd.dpmodel.utils.env_mat import (
    EnvMat,
    compute_smooth_weight,
)

from .utils import (
    ArrayAPITest,
)


class TestEnvMat(unittest.TestCase, ArrayAPITest):
    def test_compute_smooth_weight(self) -> None:
        d = xp.arange(10, dtype=xp.float64)
        w = compute_smooth_weight(
            d,
            4.0,
            6.0,
        )
        self.assert_namespace_equal(w, d)
        self.assert_device_equal(w, d)
        self.assert_dtype_equal(w, d)

    def test_virtual_center_uses_safe_normalization_indices(self) -> None:
        """Strict array indexing must never receive the negative type sentinel."""
        coord = xp.asarray([[0.0, 0.0, 0.0]], dtype=xp.float64)
        atype = xp.asarray([[-1]], dtype=xp.int64)
        nlist = xp.asarray([[[-1]]], dtype=xp.int64)
        davg = xp.asarray([[[11.0, 13.0, 17.0, 19.0]]], dtype=xp.float64)
        # A zero placeholder scale verifies that masking happens before division;
        # otherwise the virtual row can create hidden NaN or infinity values.
        dstd = xp.zeros_like(davg)

        env_mat, diff, switch = EnvMat(2.0, 0.5).call(coord, atype, nlist, davg, dstd)

        for output in (env_mat, diff, switch):
            self.assertTrue(bool(xp.all(output == xp.zeros_like(output))))
            self.assert_namespace_equal(output, coord)
            self.assert_device_equal(output, coord)
            self.assert_dtype_equal(output, coord)

    def test_mixed_centers_mask_all_virtual_outputs(self) -> None:
        """The shared JAX/pt_expt EnvMat keeps real rows and masks bad virtual rows."""
        coord = xp.asarray([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=xp.float64)
        atype = xp.asarray([[1, -1]], dtype=xp.int64)
        # Deliberately violate the normal virtual-center contract to verify that
        # em, diff, and switch all provide the documented defense in depth.
        nlist = xp.asarray([[[1], [0]]], dtype=xp.int64)
        davg = xp.asarray(
            [[[11.0, 13.0, 17.0, 19.0]], [[0.0, 0.0, 0.0, 0.0]]],
            dtype=xp.float64,
        )
        dstd = xp.asarray(
            [[[0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0]]],
            dtype=xp.float64,
        )

        env_mat, diff, switch = EnvMat(2.0, 0.5).call(coord, atype, nlist, davg, dstd)

        for output in (env_mat, diff, switch):
            real_output = output[:, :1, ...]
            virtual_output = output[:, 1:, ...]
            self.assertTrue(bool(xp.any(real_output != xp.zeros_like(real_output))))
            self.assertTrue(
                bool(xp.all(virtual_output == xp.zeros_like(virtual_output)))
            )
            self.assert_namespace_equal(output, coord)
            self.assert_device_equal(output, coord)
            self.assert_dtype_equal(output, coord)
