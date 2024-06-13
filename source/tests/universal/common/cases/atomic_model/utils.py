# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
from typing import (
    Any,
    Callable,
    Dict,
    List,
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
    expected_has_message_passing: bool
    """Expected whether having message passing."""
    forward_wrapper: Callable[[Any], Any]
    """Calss wrapper for forward method."""
    aprec_dict: Dict[str, Optional[float]]
    """Dictionary of absolute precision in each test."""
    rprec_dict: Dict[str, Optional[float]]
    """Dictionary of relative precision in each test."""
    epsilon_dict: Dict[str, Optional[float]]
    """Dictionary of epsilons in each test."""

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

    def test_has_message_passing(self):
        """Test has_message_passing."""
        for module in self.modules_to_test:
            self.assertEqual(
                module.has_message_passing(), self.expected_has_message_passing
            )

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
            mixed_types=self.module.mixed_types(),
            box=cell,
        )
        ret_lower = []
        for module in self.modules_to_test:
            module = self.forward_wrapper(module)

            ret_lower.append(module(coord_ext, atype_ext, nlist, mapping=mapping))
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

    def test_permutation(self):
        """Test permutation."""
        if getattr(self, "skip_test_permutation", False):
            return
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        nf = 1
        idx_perm = [1, 0, 4, 3, 2]
        cell = rng.random([3, 3])
        cell = (cell + cell.T) + 5.0 * np.eye(3)
        coord = rng.random([natoms, 3])
        coord = np.matmul(coord, cell)
        atype = np.array([0, 0, 0, 1, 1])
        coord_perm = coord[idx_perm]
        atype_perm = atype[idx_perm]

        # reshape for input
        coord = coord.reshape([nf, -1])
        coord_perm = coord_perm.reshape([nf, -1])
        atype = atype.reshape([nf, -1])
        atype_perm = atype_perm.reshape([nf, -1])
        cell = cell.reshape([nf, 9])

        ret = []
        module = self.forward_wrapper(self.module)
        coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.expected_rcut,
            self.expected_sel,
            mixed_types=self.module.mixed_types(),
            box=cell,
        )
        ret.append(module(coord_ext, atype_ext, nlist, mapping=mapping))
        # permutation
        coord_ext_perm, atype_ext_perm, mapping_perm, nlist_perm = (
            extend_input_and_build_neighbor_list(
                coord_perm,
                atype_perm,
                self.expected_rcut,
                self.expected_sel,
                mixed_types=self.module.mixed_types(),
                box=cell,
            )
        )
        ret.append(
            module(coord_ext_perm, atype_ext_perm, nlist_perm, mapping=mapping_perm)
        )

        for kk in ret[0]:
            if kk in self.expected_model_output_type:
                atomic = self.output_def[kk].atomic
                if atomic:
                    np.testing.assert_allclose(
                        ret[0][kk][:, idx_perm],
                        ret[1][kk],
                        err_msg=f"compare {kk} before and after transform",
                    )
                else:
                    np.testing.assert_allclose(
                        ret[0][kk],
                        ret[1][kk],
                        err_msg=f"compare {kk} before and after transform",
                    )
            else:
                raise RuntimeError(f"Unknown output key: {kk}")

    def test_trans(self):
        """Test translation."""
        if getattr(self, "skip_test_trans", False):
            return
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        nf = 1
        cell = rng.random([3, 3])
        cell = (cell + cell.T) + 5.0 * np.eye(3)
        coord = rng.random([natoms, 3])
        coord = np.matmul(coord, cell)
        atype = np.array([0, 0, 0, 1, 1])
        shift = (rng.random([3]) - 0.5) * 2.0
        coord_s = np.matmul(
            np.remainder(np.matmul(coord + shift, np.linalg.inv(cell)), 1.0), cell
        )

        # reshape for input
        coord = coord.reshape([nf, -1])
        coord_s = coord_s.reshape([nf, -1])
        atype = atype.reshape([nf, -1])
        cell = cell.reshape([nf, 9])

        ret = []
        module = self.forward_wrapper(self.module)
        coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.expected_rcut,
            self.expected_sel,
            mixed_types=self.module.mixed_types(),
            box=cell,
        )
        ret.append(module(coord_ext, atype_ext, nlist, mapping=mapping))
        # translation
        coord_ext_trans, atype_ext_trans, mapping_trans, nlist_trans = (
            extend_input_and_build_neighbor_list(
                coord_s,
                atype,
                self.expected_rcut,
                self.expected_sel,
                mixed_types=self.module.mixed_types(),
                box=cell,
            )
        )
        ret.append(
            module(coord_ext_trans, atype_ext_trans, nlist_trans, mapping=mapping_trans)
        )

        for kk in ret[0]:
            if kk in self.expected_model_output_type:
                np.testing.assert_allclose(
                    ret[0][kk],
                    ret[1][kk],
                    err_msg=f"compare {kk} before and after transform",
                )
            else:
                raise RuntimeError(f"Unknown output key: {kk}")

    def test_rot(self):
        """Test rotation."""
        if getattr(self, "skip_test_rot", False):
            return
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        nf = 1

        # rotate only coord and shift to the center of cell
        cell = 10.0 * np.eye(3)
        coord = 2.0 * rng.random([natoms, 3])
        atype = np.array([0, 0, 0, 1, 1])
        shift = np.array([4.0, 4.0, 4.0])
        from scipy.stats import (
            special_ortho_group,
        )

        rmat = special_ortho_group.rvs(3)
        coord_rot = np.matmul(coord, rmat)

        # reshape for input
        coord = (coord + shift).reshape([nf, -1])
        coord_rot = (coord_rot + shift).reshape([nf, -1])
        atype = atype.reshape([nf, -1])
        cell = cell.reshape([nf, 9])

        ret = []
        module = self.forward_wrapper(self.module)
        coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.expected_rcut,
            self.expected_sel,
            mixed_types=self.module.mixed_types(),
            box=cell,
        )
        ret.append(module(coord_ext, atype_ext, nlist, mapping=mapping))
        # rotation
        coord_ext_rot, atype_ext_rot, mapping_rot, nlist_rot = (
            extend_input_and_build_neighbor_list(
                coord_rot,
                atype,
                self.expected_rcut,
                self.expected_sel,
                mixed_types=self.module.mixed_types(),
                box=cell,
            )
        )
        ret.append(module(coord_ext_rot, atype_ext_rot, nlist_rot, mapping=mapping_rot))

        for kk in ret[0]:
            if kk in self.expected_model_output_type:
                rot_invariant = self.output_def[kk].rot_invariant
                if rot_invariant:
                    np.testing.assert_allclose(
                        ret[0][kk],
                        ret[1][kk],
                        err_msg=f"compare {kk} before and after transform",
                    )
                else:
                    v_shape = self.output_def[kk].shape
                    rotated_ret_0 = (
                        np.matmul(ret[0][kk], rmat)
                        if len(v_shape) == 1
                        else np.matmul(rmat.T, np.matmul(ret[0][kk], rmat))
                    )
                    np.testing.assert_allclose(
                        rotated_ret_0,
                        ret[1][kk],
                        err_msg=f"compare {kk} before and after transform",
                    )
            else:
                raise RuntimeError(f"Unknown output key: {kk}")

        # rotate coord and cell
        cell = rng.random([3, 3])
        cell = (cell + cell.T) + 5.0 * np.eye(3)
        coord = rng.random([natoms, 3])
        coord = np.matmul(coord, cell)
        atype = np.array([0, 0, 0, 1, 1])
        coord_rot = np.matmul(coord, rmat)
        cell_rot = np.matmul(cell, rmat)

        # reshape for input
        coord = coord.reshape([nf, -1])
        coord_rot = coord_rot.reshape([nf, -1])
        atype = atype.reshape([nf, -1])
        cell = cell.reshape([nf, 9])
        cell_rot = cell_rot.reshape([nf, 9])

        ret = []
        module = self.forward_wrapper(self.module)
        coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.expected_rcut,
            self.expected_sel,
            mixed_types=self.module.mixed_types(),
            box=cell,
        )
        ret.append(module(coord_ext, atype_ext, nlist, mapping=mapping))
        # rotation
        coord_ext_rot, atype_ext_rot, mapping_rot, nlist_rot = (
            extend_input_and_build_neighbor_list(
                coord_rot,
                atype,
                self.expected_rcut,
                self.expected_sel,
                mixed_types=self.module.mixed_types(),
                box=cell_rot,
            )
        )
        ret.append(module(coord_ext_rot, atype_ext_rot, nlist_rot, mapping=mapping_rot))

        for kk in ret[0]:
            if kk in self.expected_model_output_type:
                rot_invariant = self.output_def[kk].rot_invariant
                if rot_invariant:
                    np.testing.assert_allclose(
                        ret[0][kk],
                        ret[1][kk],
                        err_msg=f"compare {kk} before and after transform",
                    )
                else:
                    v_shape = self.output_def[kk].shape
                    rotated_ret_0 = (
                        np.matmul(ret[0][kk], rmat)
                        if len(v_shape) == 1
                        else np.matmul(rmat.T, np.matmul(ret[0][kk], rmat))
                    )
                    np.testing.assert_allclose(
                        rotated_ret_0,
                        ret[1][kk],
                        err_msg=f"compare {kk} before and after transform",
                    )
            else:
                raise RuntimeError(f"Unknown output key: {kk}")

    def test_smooth(self):
        """Test smooth."""
        if getattr(self, "skip_test_smooth", False):
            return
        rng = np.random.default_rng(GLOBAL_SEED)
        epsilon = (
            1e-5
            if self.epsilon_dict.get("test_smooth", None) is None
            else self.epsilon_dict["test_smooth"]
        )
        # required prec.
        rprec = (
            1e-5
            if self.rprec_dict.get("test_smooth", None) is None
            else self.rprec_dict["test_smooth"]
        )
        aprec = (
            1e-5
            if self.aprec_dict.get("test_smooth", None) is None
            else self.aprec_dict["test_smooth"]
        )
        natoms = 10
        nf = 1
        cell = 10.0 * np.eye(3)
        atype0 = np.arange(2)
        atype1 = rng.integers(0, 2, size=natoms - 2)
        atype = np.concatenate([atype0, atype1]).reshape(natoms)
        coord0 = np.array(
            [
                0.0,
                0.0,
                0.0,
                self.expected_rcut - 0.5 * epsilon,
                0.0,
                0.0,
                0.0,
                self.expected_rcut - 0.5 * epsilon,
                0.0,
            ]
        ).reshape(-1, 3)
        coord1 = rng.random([natoms - coord0.shape[0], 3])
        coord1 = np.matmul(coord1, cell)
        coord = np.concatenate([coord0, coord1], axis=0)

        coord0 = deepcopy(coord)
        coord1 = deepcopy(coord)
        coord1[1][0] += epsilon
        coord2 = deepcopy(coord)
        coord2[2][1] += epsilon
        coord3 = deepcopy(coord)
        coord3[1][0] += epsilon
        coord3[2][1] += epsilon

        # reshape for input
        coord0 = coord0.reshape([nf, -1])
        coord1 = coord1.reshape([nf, -1])
        coord2 = coord2.reshape([nf, -1])
        coord3 = coord3.reshape([nf, -1])
        atype = atype.reshape([nf, -1])
        cell = cell.reshape([nf, 9])

        ret = []
        module = self.forward_wrapper(self.module)
        # coord0
        coord_ext0, atype_ext0, mapping0, nlist0 = extend_input_and_build_neighbor_list(
            coord0,
            atype,
            self.expected_rcut,
            self.expected_sel,
            mixed_types=self.module.mixed_types(),
            box=cell,
        )
        ret.append(module(coord_ext0, atype_ext0, nlist0, mapping=mapping0))
        # coord1
        coord_ext1, atype_ext1, mapping1, nlist1 = extend_input_and_build_neighbor_list(
            coord1,
            atype,
            self.expected_rcut,
            self.expected_sel,
            mixed_types=self.module.mixed_types(),
            box=cell,
        )
        ret.append(module(coord_ext1, atype_ext1, nlist1, mapping=mapping1))
        # coord2
        coord_ext2, atype_ext2, mapping2, nlist2 = extend_input_and_build_neighbor_list(
            coord2,
            atype,
            self.expected_rcut,
            self.expected_sel,
            mixed_types=self.module.mixed_types(),
            box=cell,
        )
        ret.append(module(coord_ext2, atype_ext2, nlist2, mapping=mapping2))
        # coord3
        coord_ext3, atype_ext3, mapping3, nlist3 = extend_input_and_build_neighbor_list(
            coord3,
            atype,
            self.expected_rcut,
            self.expected_sel,
            mixed_types=self.module.mixed_types(),
            box=cell,
        )
        ret.append(module(coord_ext3, atype_ext3, nlist3, mapping=mapping3))

        for kk in ret[0]:
            if kk in self.expected_model_output_type:
                for ii in range(len(ret) - 1):
                    np.testing.assert_allclose(
                        ret[0][kk],
                        ret[ii + 1][kk],
                        err_msg=f"compare {kk} before and after transform",
                        atol=aprec,
                        rtol=rprec,
                    )
            else:
                raise RuntimeError(f"Unknown output key: {kk}")
