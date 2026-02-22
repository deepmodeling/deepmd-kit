# SPDX-License-Identifier: LGPL-3.0-or-later
import tempfile
import unittest
from pathlib import (
    Path,
)
from typing import (
    NoReturn,
)

import h5py
import numpy as np

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.utils.path import (
    DPPath,
)

from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class FooFitting(NativeOP, BaseFitting):
    """Test fitting that returns fixed values for testing bias computation."""

    def __init__(self):
        pass

    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "foo",
                    [1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
                OutputVariableDef(
                    "bar",
                    [1, 2],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
            ]
        )

    def serialize(self) -> dict:
        return {
            "@class": "Fitting",
            "type": "foo",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls()

    def get_dim_fparam(self) -> int:
        return 0

    def get_dim_aparam(self) -> int:
        return 0

    def get_sel_type(self) -> list[int]:
        return []

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        pass

    def get_type_map(self) -> list[str]:
        return []

    def call(
        self,
        descriptor,
        atype,
        gr=None,
        g2=None,
        h2=None,
        fparam=None,
        aparam=None,
    ):
        nf, nloc, _ = descriptor.shape
        ret = {}
        ret["foo"] = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ).reshape([nf, nloc, *self.output_def()["foo"].shape])
        ret["bar"] = np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
            ]
        ).reshape([nf, nloc, *self.output_def()["bar"].shape])
        return ret


class TestAtomicModelStat(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.merged_output_stat = [
            {
                "coord": np.zeros([2, 3, 3]),
                "atype": np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32),
                "atype_ext": np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32),
                "box": np.zeros([2, 3, 3]),
                "natoms": np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32),
                # bias of foo: 5, 6
                "atom_foo": np.array([[5.0, 5.0, 5.0], [5.0, 6.0, 7.0]]).reshape(
                    2, 3, 1
                ),
                # bias of bar: [1, 5], [3, 2]
                "bar": np.array([5.0, 12.0, 7.0, 9.0]).reshape(2, 1, 2),
                "find_atom_foo": np.float32(1.0),
                "find_bar": np.float32(1.0),
            },
            {
                "coord": np.zeros([2, 3, 3]),
                "atype": np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32),
                "atype_ext": np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32),
                "box": np.zeros([2, 3, 3]),
                "natoms": np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32),
                # bias of foo: 5.5, 3 from global label.
                "foo": np.array([5.0, 7.0]).reshape(2, 1),
                # bias of bar: [1, 5], [3, 2]
                "bar": np.array([5.0, 12.0, 7.0, 9.0]).reshape(2, 1, 2),
                "find_foo": np.float32(1.0),
                "find_bar": np.float32(1.0),
            },
        ]
        self.tempdir = tempfile.TemporaryDirectory()
        h5file = str((Path(self.tempdir.name) / "testcase.h5").resolve())
        with h5py.File(h5file, "w") as f:
            pass
        self.stat_file_path = DPPath(h5file, "a")

    def test_output_stat(self) -> None:
        """Test output statistics computation for dpmodel atomic model."""
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = FooFitting()
        type_map = ["foo", "bar"]
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
        )
        args = [self.coord_ext, self.atype_ext, self.nlist]
        # nf x nloc
        at = self.atype_ext[:, :nloc]

        # 1. test run without bias
        # nf x na x odim
        ret0 = md0.forward_common_atomic(*args)
        expected_ret0 = {}
        expected_ret0["foo"] = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ).reshape([nf, nloc, *md0.fitting_output_def()["foo"].shape])
        expected_ret0["bar"] = np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
            ]
        ).reshape([nf, nloc, *md0.fitting_output_def()["bar"].shape])
        for kk in ["foo", "bar"]:
            np.testing.assert_almost_equal(ret0[kk], expected_ret0[kk])

        # 2. test bias is applied
        md0.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )
        ret1 = md0.forward_common_atomic(*args)
        expected_std = np.ones(
            (2, 2, 2), dtype=np.float64
        )  # 2 keys, 2 atypes, 2 max dims.
        expected_std[0, :, :1] = np.array([0.0, 0.816496]).reshape(
            2, 1
        )  # updating std for foo based on [5.0, 5.0, 5.0], [5.0, 6.0, 7.0]]
        np.testing.assert_almost_equal(md0.out_std, expected_std, decimal=4)
        # nt x odim
        foo_bias = np.array([5.0, 6.0]).reshape(2, 1)
        bar_bias = np.array([1.0, 5.0, 3.0, 2.0]).reshape(2, 1, 2)
        expected_ret1 = {}
        expected_ret1["foo"] = ret0["foo"] + foo_bias[at]
        expected_ret1["bar"] = ret0["bar"] + bar_bias[at]
        for kk in ["foo", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], expected_ret1[kk])

        # 3. test bias load from file
        def raise_error() -> NoReturn:
            raise RuntimeError

        md0.compute_or_load_out_stat(raise_error, stat_file_path=self.stat_file_path)
        ret2 = md0.forward_common_atomic(*args)
        for kk in ["foo", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], ret2[kk])
        np.testing.assert_almost_equal(md0.out_std, expected_std, decimal=4)

        # 4. test change bias
        md0.change_out_bias(
            self.merged_output_stat, bias_adjust_mode="change-by-statistic"
        )
        # use atype_ext from merged_output_stat for inference (matching pt backend test)
        args = [
            self.coord_ext,
            np.array(self.merged_output_stat[0]["atype_ext"], dtype=np.int64),
            self.nlist,
        ]
        ret3 = md0.forward_common_atomic(*args)
        expected_std[0, :, :1] = np.array([1.24722, 0.47140]).reshape(
            2, 1
        )  # updating std for foo based on [4.0, 3.0, 2.0], [1.0, 1.0, 1.0]]
        expected_ret3 = {}
        # new bias [2.666, 1.333]
        expected_ret3["foo"] = np.array(
            [[3.6667, 4.6667, 4.3333], [6.6667, 6.3333, 7.3333]]
        ).reshape(2, 3, 1)
        for kk in ["foo"]:
            np.testing.assert_almost_equal(ret3[kk], expected_ret3[kk], decimal=4)
        np.testing.assert_almost_equal(md0.out_std, expected_std, decimal=4)


class TestAtomicModelStatMergeGlobalAtomic(
    unittest.TestCase, TestCaseSingleFrameWithNlist
):
    """Test merging atomic and global stat when atomic label only covers some types."""

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.merged_output_stat = [
            {
                "coord": np.zeros([2, 3, 3]),
                "atype": np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int32),
                "atype_ext": np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32),
                "box": np.zeros([2, 3, 3]),
                "natoms": np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32),
                # bias of foo: 5.5, nan (only type 0 atoms)
                "atom_foo": np.array([[5.0, 5.0, 5.0], [5.0, 6.0, 7.0]]).reshape(
                    2, 3, 1
                ),
                # bias of bar: [1, 5], [3, 2]
                "bar": np.array([5.0, 12.0, 7.0, 9.0]).reshape(2, 1, 2),
                "find_atom_foo": np.float32(1.0),
                "find_bar": np.float32(1.0),
            },
            {
                "coord": np.zeros([2, 3, 3]),
                "atype": np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32),
                "atype_ext": np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32),
                "box": np.zeros([2, 3, 3]),
                "natoms": np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32),
                # bias of foo: 5.5, 3 from global label.
                "foo": np.array([5.0, 7.0]).reshape(2, 1),
                # bias of bar: [1, 5], [3, 2]
                "bar": np.array([5.0, 12.0, 7.0, 9.0]).reshape(2, 1, 2),
                "find_foo": np.float32(1.0),
                "find_bar": np.float32(1.0),
            },
        ]
        self.tempdir = tempfile.TemporaryDirectory()
        h5file = str((Path(self.tempdir.name) / "testcase.h5").resolve())
        with h5py.File(h5file, "w") as f:
            pass
        self.stat_file_path = DPPath(h5file, "a")

    def test_output_stat(self) -> None:
        """Test merging atomic (type 0 only) and global stat for type 1."""
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = FooFitting()
        type_map = ["foo", "bar"]
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
        )
        args = [self.coord_ext, self.atype_ext, self.nlist]
        # nf x nloc
        at = self.atype_ext[:, :nloc]

        # 1. test run without bias
        ret0 = md0.forward_common_atomic(*args)
        expected_ret0 = {}
        expected_ret0["foo"] = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ).reshape([nf, nloc, *md0.fitting_output_def()["foo"].shape])
        expected_ret0["bar"] = np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
            ]
        ).reshape([nf, nloc, *md0.fitting_output_def()["bar"].shape])
        for kk in ["foo", "bar"]:
            np.testing.assert_almost_equal(ret0[kk], expected_ret0[kk])

        # 2. test bias is applied
        # foo: type 0 from atomic (mean=5.5), type 1 from global (lstsq=3.0)
        md0.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )
        ret1 = md0.forward_common_atomic(*args)
        # nt x odim
        foo_bias = np.array([5.5, 3.0]).reshape(2, 1)
        bar_bias = np.array([1.0, 5.0, 3.0, 2.0]).reshape(2, 1, 2)
        expected_ret1 = {}
        expected_ret1["foo"] = ret0["foo"] + foo_bias[at]
        expected_ret1["bar"] = ret0["bar"] + bar_bias[at]
        for kk in ["foo", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], expected_ret1[kk])

        # 3. test bias load from file
        def raise_error() -> NoReturn:
            raise RuntimeError

        md0.compute_or_load_out_stat(raise_error, stat_file_path=self.stat_file_path)
        ret2 = md0.forward_common_atomic(*args)
        for kk in ["foo", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], ret2[kk])

        # 4. test change bias
        md0.change_out_bias(
            self.merged_output_stat, bias_adjust_mode="change-by-statistic"
        )
        # use atype_ext from merged_output_stat for inference
        args = [
            self.coord_ext,
            np.array(self.merged_output_stat[0]["atype_ext"], dtype=np.int64),
            self.nlist,
        ]
        ret3 = md0.forward_common_atomic(*args)
        expected_ret3 = {}
        # new bias [2, -5]
        expected_ret3["foo"] = np.array([[3, 4, -2], [6, 0, 1]]).reshape(2, 3, 1)
        for kk in ["foo"]:
            np.testing.assert_almost_equal(ret3[kk], expected_ret3[kk], decimal=4)


if __name__ == "__main__":
    unittest.main()
