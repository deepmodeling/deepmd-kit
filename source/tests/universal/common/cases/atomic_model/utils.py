# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Callable,
    Optional,
)

import numpy as np

from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

from .....seed import (
    GLOBAL_SEED,
)


class AtomicModelTestCase:
    """Common test case for atomic model."""

    expected_type_map: list[str]
    """Expected type map."""
    expected_rcut: float
    """Expected cut-off radius."""
    expected_dim_fparam: int
    """Expected number (dimension) of frame parameters."""
    expected_dim_aparam: int
    """Expected number (dimension) of atomic parameters."""
    expected_sel_type: list[int]
    """Expected selected atom types."""
    expected_aparam_nall: bool
    """Expected shape of atomic parameters."""
    expected_model_output_type: list[str]
    """Expected output type for the model."""
    model_output_equivariant: list[str]
    """Outputs that are equivariant to the input rotation."""
    expected_sel: list[int]
    """Expected number of neighbors."""
    expected_has_message_passing: bool
    """Expected whether having message passing."""
    forward_wrapper: Callable[[Any], Any]
    """Class wrapper for forward method."""
    aprec_dict: dict[str, Optional[float]]
    """Dictionary of absolute precision in each test."""
    rprec_dict: dict[str, Optional[float]]
    """Dictionary of relative precision in each test."""
    epsilon_dict: dict[str, Optional[float]]
    """Dictionary of epsilons in each test."""

    def test_get_type_map(self) -> None:
        """Test get_type_map."""
        for module in self.modules_to_test:
            self.assertEqual(module.get_type_map(), self.expected_type_map)

    def test_get_rcut(self) -> None:
        """Test get_rcut."""
        for module in self.modules_to_test:
            self.assertAlmostEqual(module.get_rcut(), self.expected_rcut)

    def test_get_dim_fparam(self) -> None:
        """Test get_dim_fparam."""
        for module in self.modules_to_test:
            self.assertEqual(module.get_dim_fparam(), self.expected_dim_fparam)

    def test_get_dim_aparam(self) -> None:
        """Test get_dim_aparam."""
        for module in self.modules_to_test:
            self.assertEqual(module.get_dim_aparam(), self.expected_dim_aparam)

    def test_get_sel_type(self) -> None:
        """Test get_sel_type."""
        for module in self.modules_to_test:
            self.assertEqual(module.get_sel_type(), self.expected_sel_type)

    def test_is_aparam_nall(self) -> None:
        """Test is_aparam_nall."""
        for module in self.modules_to_test:
            self.assertEqual(module.is_aparam_nall(), self.expected_aparam_nall)

    def test_get_nnei(self) -> None:
        """Test get_nnei."""
        expected_nnei = sum(self.expected_sel)
        for module in self.modules_to_test:
            self.assertEqual(module.get_nnei(), expected_nnei)

    def test_get_ntypes(self) -> None:
        """Test get_ntypes."""
        for module in self.modules_to_test:
            self.assertEqual(module.get_ntypes(), len(self.expected_type_map))

    def test_has_message_passing(self) -> None:
        """Test has_message_passing."""
        for module in self.modules_to_test:
            self.assertEqual(
                module.has_message_passing(), self.expected_has_message_passing
            )

    def test_forward(self) -> None:
        """Test forward."""
        nf = 1
        rng = np.random.default_rng(GLOBAL_SEED)
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
            mixed_types=self.module.mixed_types(),
            box=cell,
        )
        ret_lower = []
        aparam = None
        fparam = None
        if self.module.get_dim_aparam() > 0:
            aparam = rng.random([nf, 3, self.module.get_dim_aparam()])
        if self.module.get_dim_fparam() > 0:
            fparam = rng.random([nf, self.module.get_dim_fparam()])
        for module in self.modules_to_test:
            module = self.forward_wrapper(module)
            ret_lower.append(
                module(
                    coord_ext,
                    atype_ext,
                    nlist,
                    mapping=mapping,
                    fparam=fparam,
                    aparam=aparam,
                )
            )
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


#  other properties are tested in the model level
