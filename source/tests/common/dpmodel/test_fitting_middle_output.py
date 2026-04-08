# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)

from ...seed import (
    GLOBAL_SEED,
)
from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestFittingMiddleOutput(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def _build_fitting(self, mixed_types, nfp=0, nap=0):
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        dd = ds.call(self.coord_ext, self.atype_ext, self.nlist)
        descriptor = dd[0]
        atype = self.atype_ext[:, : self.nloc]
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,  # dim_out
            mixed_types=mixed_types,
            numb_fparam=nfp,
            numb_aparam=nap,
            seed=GLOBAL_SEED,
        )
        return ft, descriptor, atype

    def test_middle_output_disabled_by_default(self) -> None:
        """Middle output should not be present when not enabled."""
        ft, descriptor, atype = self._build_fitting(mixed_types=True)
        ret = ft.call(descriptor, atype)
        self.assertIn("energy", ret)
        self.assertNotIn("middle_output", ret)

    def test_middle_output_enabled_mixed_types(self) -> None:
        """When enabled, middle_output key is present with correct shape for mixed_types=True."""
        ft, descriptor, atype = self._build_fitting(mixed_types=True)
        ft.set_return_middle_output(True)
        ret = ft.call(descriptor, atype)
        self.assertIn("middle_output", ret)
        nf, nloc, _ = descriptor.shape
        expected_shape = (nf, nloc, ft.neuron[-1])
        self.assertEqual(ret["middle_output"].shape, expected_shape)

    def test_middle_output_enabled_per_type(self) -> None:
        """When enabled, middle_output key is present with correct shape for mixed_types=False."""
        ft, descriptor, atype = self._build_fitting(mixed_types=False)
        ft.set_return_middle_output(True)
        ret = ft.call(descriptor, atype)
        self.assertIn("middle_output", ret)
        nf, nloc, _ = descriptor.shape
        expected_shape = (nf, nloc, ft.neuron[-1])
        self.assertEqual(ret["middle_output"].shape, expected_shape)

    def test_middle_output_toggle(self) -> None:
        """Verify toggling set_return_middle_output on/off works."""
        ft, descriptor, atype = self._build_fitting(mixed_types=True)

        # Enable
        ft.set_return_middle_output(True)
        ret = ft.call(descriptor, atype)
        self.assertIn("middle_output", ret)

        # Disable
        ft.set_return_middle_output(False)
        ret = ft.call(descriptor, atype)
        self.assertNotIn("middle_output", ret)

    def test_middle_output_with_fparam_aparam(self) -> None:
        """Middle output works with fparam and aparam."""
        ft, descriptor, atype = self._build_fitting(mixed_types=True, nfp=2, nap=3)
        nf, nloc, _ = descriptor.shape
        fparam = np.zeros([nf, 2], dtype=np.float64)
        aparam = np.zeros([nf, nloc, 3], dtype=np.float64)
        ft.set_return_middle_output(True)
        ret = ft.call(descriptor, atype, fparam=fparam, aparam=aparam)
        self.assertIn("middle_output", ret)
        expected_shape = (nf, nloc, ft.neuron[-1])
        self.assertEqual(ret["middle_output"].shape, expected_shape)

    def test_middle_output_deterministic(self) -> None:
        """Middle output should be deterministic."""
        ft, descriptor, atype = self._build_fitting(mixed_types=True)
        ft.set_return_middle_output(True)
        ret1 = ft.call(descriptor, atype)
        ret2 = ft.call(descriptor, atype)
        np.testing.assert_array_equal(ret1["middle_output"], ret2["middle_output"])
