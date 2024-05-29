# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Callable,
    List,
)

import numpy as np

from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)


class AtomicModelTestCase:
    """Common test case for atomic model."""

    expected_type_map: List[str]
    """Expected type map."""
    expected_rcut: float
    """Expected cut-off radius."""
    expected_dim_fparam: int
    """Expected number (dimension) of frame parameters."""
    expected_dim_aparam: int
    """Expected number (dimension) of atomic parameters."""
    expected_sel_type: List[int]
    """Expected selected atom types."""
    expected_aparam_nall: bool
    """Expected shape of atomic parameters."""
    expected_model_output_type: List[str]
    """Expected output type for the model."""
    expected_sel: List[int]
    """Expected number of neighbors."""
    forward_wrapper: Callable[[Any], Any]
    """Calss wrapper for forward method."""

    def test_get_type_map(self):
        """Test get_type_map."""
        for module in self.modules_to_test:
            self.assertEqual(module.get_type_map(), self.expected_type_map)

    def test_get_rcut(self):
        """Test get_rcut."""
        for module in self.modules_to_test:
            self.assertAlmostEqual(module.get_rcut(), self.expected_rcut)

    def test_get_dim_fparam(self):
        """Test get_dim_fparam."""
        for module in self.modules_to_test:
            self.assertEqual(module.get_dim_fparam(), self.expected_dim_fparam)

    def test_get_dim_aparam(self):
        """Test get_dim_aparam."""
        for module in self.modules_to_test:
            self.assertEqual(module.get_dim_aparam(), self.expected_dim_aparam)

    def test_get_sel_type(self):
        """Test get_sel_type."""
        for module in self.modules_to_test:
            self.assertEqual(module.get_sel_type(), self.expected_sel_type)

    def test_is_aparam_nall(self):
        """Test is_aparam_nall."""
        for module in self.modules_to_test:
            self.assertEqual(module.is_aparam_nall(), self.expected_aparam_nall)

    def test_get_nnei(self):
        """Test get_nnei."""
        expected_nnei = sum(self.expected_sel)
        for module in self.modules_to_test:
            self.assertEqual(module.get_nnei(), expected_nnei)

    def test_get_ntypes(self):
        """Test get_ntypes."""
        for module in self.modules_to_test:
            self.assertEqual(module.get_ntypes(), len(self.expected_type_map))

    def test_forward(self):
        """Test forward."""
        nf = 1
        coord = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        ).reshape([nf, -1])
        atype = np.array([0, 0, 1], dtype=int).reshape([nf, -1])
        cell = 6.0 * np.eye(3).reshape([nf, 9])
        coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.expected_rcut,
            self.expected_sel,
            mixed_types=True,
            box=cell,
        )
        ret_lower = []
        for module in self.modules_to_test:
            module = self.forward_wrapper(module)

            ret_lower.append(module(coord_ext, atype_ext, nlist))
        for kk in ret_lower[0].keys():
            subret = []
            for rr in ret_lower:
                if rr is not None:
                    subret.append(rr[kk])
            if len(subret):
                for ii, rr in enumerate(subret[1:]):
                    if subret[0] is None:
                        assert rr is None
                    else:
                        np.testing.assert_allclose(
                            subret[0], rr, err_msg=f"compare {kk} between 0 and {ii}"
                        )
