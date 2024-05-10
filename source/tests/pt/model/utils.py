# SPDX-License-Identifier: LGPL-3.0-or-later
"""Common test case."""

from typing import (
    List,
)

import numpy as np
import torch

from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)


class CommonTestCase:
    """Common test case."""

    module: torch.nn.Module
    """Module to test."""

    @property
    def script_module(self):
        return torch.jit.script(self.module)

    @property
    def deserialized_module(self):
        return self.module.deserialize(self.module.serialize())

    @property
    def modules_to_test(self):
        return [self.module, self.script_module, self.deserialized_module]

    def test_jit(self):
        self.script_module


class ModelTestCase(CommonTestCase):
    """Common test case for model."""

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

    def test_model_output_type(self):
        """Test model_output_type."""
        for module in self.modules_to_test:
            self.assertEqual(
                module.model_output_type(), self.expected_model_output_type
            )

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
        """Test forward and forward_lower."""
        nf = 1
        nloc = 3
        coord = to_torch_tensor(
            np.array(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            ).reshape([nf, -1])
        )
        atype = to_torch_tensor(np.array([0, 0, 1], dtype=int).reshape([nf, -1]))
        cell = to_torch_tensor(6.0 * np.eye(3).reshape([nf, 9]))
        coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.expected_rcut,
            self.expected_sel,
            mixed_types=True,
            box=cell,
        )
        ret = []
        ret_lower = []
        for module in self.modules_to_test:
            ret.append(module(coord, atype, cell))
            ret_lower.append(module.forward_lower(coord_ext, atype_ext, nlist))
        for r in ret[1:]:
            torch.testing.assert_close(ret[0], r)
        for r in ret_lower[1:]:
            torch.testing.assert_close(ret_lower[0], r)
        same_keys = set(ret[0].keys()) & set(ret_lower[0].keys())
        self.assertTrue(same_keys)
        for key in same_keys:
            torch.testing.assert_close(ret[0][key], ret_lower[0][key])


class AtomicModelTestCase(CommonTestCase):
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

    @property
    def modules_to_test(self):
        return [self.module, self.deserialized_module]

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
        """Test forward and forward_lower."""
        nf = 1
        nloc = 3
        coord = to_torch_tensor(
            np.array(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            ).reshape([nf, -1])
        )
        atype = to_torch_tensor(np.array([0, 0, 1], dtype=int).reshape([nf, -1]))
        cell = to_torch_tensor(6.0 * np.eye(3).reshape([nf, 9]))
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
            ret_lower.append(module.forward_common_atomic(coord_ext, atype_ext, nlist))
        for r in ret_lower[1:]:
            torch.testing.assert_close(ret_lower[0], r)
