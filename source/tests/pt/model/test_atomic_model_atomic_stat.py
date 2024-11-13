# SPDX-License-Identifier: LGPL-3.0-or-later
import tempfile
import unittest
from pathlib import (
    Path,
)
from typing import (
    NoReturn,
    Optional,
)

import h5py
import numpy as np
import torch

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.atomic_model import (
    BaseAtomicModel,
    DPAtomicModel,
)
from deepmd.pt.model.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt.model.task.base_fitting import (
    BaseFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.path import (
    DPPath,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class FooFitting(torch.nn.Module, BaseFitting):
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
        raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        raise NotImplementedError

    def get_type_map(self) -> list[str]:
        raise NotImplementedError

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: Optional[torch.Tensor] = None,
        g2: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ):
        nf, nloc, _ = descriptor.shape
        ret = {}
        ret["foo"] = (
            torch.Tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            )
            .view([nf, nloc, *self.output_def()["foo"].shape])
            .to(env.GLOBAL_PT_FLOAT_PRECISION)
            .to(env.DEVICE)
        )
        ret["bar"] = (
            torch.Tensor(
                [
                    [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                    [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
                ]
            )
            .view([nf, nloc, *self.output_def()["bar"].shape])
            .to(env.GLOBAL_PT_FLOAT_PRECISION)
            .to(env.DEVICE)
        )
        return ret


class TestAtomicModelStat(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.merged_output_stat = [
            {
                "coord": to_torch_tensor(np.zeros([2, 3, 3])),
                "atype": to_torch_tensor(
                    np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
                ),
                "atype_ext": to_torch_tensor(
                    np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32)
                ),
                "box": to_torch_tensor(np.zeros([2, 3, 3])),
                "natoms": to_torch_tensor(
                    np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32)
                ),
                # bias of foo: 5, 6
                "atom_foo": to_torch_tensor(
                    np.array([[5.0, 5.0, 5.0], [5.0, 6.0, 7.0]]).reshape(2, 3, 1)
                ),
                # bias of bar: [1, 5], [3, 2]
                "bar": to_torch_tensor(
                    np.array([5.0, 12.0, 7.0, 9.0]).reshape(2, 1, 2)
                ),
                "find_atom_foo": np.float32(1.0),
                "find_bar": np.float32(1.0),
            },
            {
                "coord": to_torch_tensor(np.zeros([2, 3, 3])),
                "atype": to_torch_tensor(
                    np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
                ),
                "atype_ext": to_torch_tensor(
                    np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32)
                ),
                "box": to_torch_tensor(np.zeros([2, 3, 3])),
                "natoms": to_torch_tensor(
                    np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32)
                ),
                # bias of foo: 5, 6 from atomic label.
                "foo": to_torch_tensor(np.array([5.0, 7.0]).reshape(2, 1)),
                # bias of bar: [1, 5], [3, 2]
                "bar": to_torch_tensor(
                    np.array([5.0, 12.0, 7.0, 9.0]).reshape(2, 1, 2)
                ),
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
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            sum(self.sel),
            self.nt,
        ).to(env.DEVICE)
        ft = FooFitting().to(env.DEVICE)
        type_map = ["foo", "bar"]
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
        ).to(env.DEVICE)
        args = [
            to_torch_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        # nf x nloc
        at = self.atype_ext[:, :nloc]

        def cvt_ret(x):
            return {kk: to_numpy_array(vv) for kk, vv in x.items()}

        # 1. test run without bias
        # nf x na x odim
        ret0 = md0.forward_common_atomic(*args)
        ret0 = cvt_ret(ret0)
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
        np.testing.assert_almost_equal(
            to_numpy_array(md0.out_std), expected_std, decimal=4
        )
        ret1 = cvt_ret(ret1)
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
        ret2 = cvt_ret(ret2)
        for kk in ["foo", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], ret2[kk])
        np.testing.assert_almost_equal(
            to_numpy_array(md0.out_std), expected_std, decimal=4
        )

        # 4. test change bias
        BaseAtomicModel.change_out_bias(
            md0, self.merged_output_stat, bias_adjust_mode="change-by-statistic"
        )
        args = [
            to_torch_tensor(ii)
            for ii in [
                self.coord_ext,
                to_numpy_array(self.merged_output_stat[0]["atype_ext"]),
                self.nlist,
            ]
        ]
        ret3 = md0.forward_common_atomic(*args)
        ret3 = cvt_ret(ret3)
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
        np.testing.assert_almost_equal(
            to_numpy_array(md0.out_std), expected_std, decimal=4
        )


class TestAtomicModelStatMergeGlobalAtomic(
    unittest.TestCase, TestCaseSingleFrameWithNlist
):
    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.merged_output_stat = [
            {
                "coord": to_torch_tensor(np.zeros([2, 3, 3])),
                "atype": to_torch_tensor(
                    np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int32)
                ),
                "atype_ext": to_torch_tensor(
                    np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32)
                ),
                "box": to_torch_tensor(np.zeros([2, 3, 3])),
                "natoms": to_torch_tensor(
                    np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32)
                ),
                # bias of foo: 5.5, nan
                "atom_foo": to_torch_tensor(
                    np.array([[5.0, 5.0, 5.0], [5.0, 6.0, 7.0]]).reshape(2, 3, 1)
                ),
                # bias of bar: [1, 5], [3, 2]
                "bar": to_torch_tensor(
                    np.array([5.0, 12.0, 7.0, 9.0]).reshape(2, 1, 2)
                ),
                "find_atom_foo": np.float32(1.0),
                "find_bar": np.float32(1.0),
            },
            {
                "coord": to_torch_tensor(np.zeros([2, 3, 3])),
                "atype": to_torch_tensor(
                    np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
                ),
                "atype_ext": to_torch_tensor(
                    np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32)
                ),
                "box": to_torch_tensor(np.zeros([2, 3, 3])),
                "natoms": to_torch_tensor(
                    np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32)
                ),
                # bias of foo: 5.5, 3 from atomic label.
                "foo": to_torch_tensor(np.array([5.0, 7.0]).reshape(2, 1)),
                # bias of bar: [1, 5], [3, 2]
                "bar": to_torch_tensor(
                    np.array([5.0, 12.0, 7.0, 9.0]).reshape(2, 1, 2)
                ),
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
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            sum(self.sel),
            self.nt,
        ).to(env.DEVICE)
        ft = FooFitting().to(env.DEVICE)
        type_map = ["foo", "bar"]
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
        ).to(env.DEVICE)
        args = [
            to_torch_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        # nf x nloc
        at = self.atype_ext[:, :nloc]

        def cvt_ret(x):
            return {kk: to_numpy_array(vv) for kk, vv in x.items()}

        # 1. test run without bias
        # nf x na x odim
        ret0 = md0.forward_common_atomic(*args)
        ret0 = cvt_ret(ret0)
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
        ret1 = cvt_ret(ret1)
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
        ret2 = cvt_ret(ret2)
        for kk in ["foo", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], ret2[kk])

        # 4. test change bias
        BaseAtomicModel.change_out_bias(
            md0, self.merged_output_stat, bias_adjust_mode="change-by-statistic"
        )
        args = [
            to_torch_tensor(ii)
            for ii in [
                self.coord_ext,
                to_numpy_array(self.merged_output_stat[0]["atype_ext"]),
                self.nlist,
            ]
        ]
        ret3 = md0.forward_common_atomic(*args)
        ret3 = cvt_ret(ret3)
        expected_ret3 = {}
        # new bias [2, -5]
        expected_ret3["foo"] = np.array([[3, 4, -2], [6, 0, 1]]).reshape(2, 3, 1)
        for kk in ["foo"]:
            np.testing.assert_almost_equal(ret3[kk], expected_ret3[kk], decimal=4)
