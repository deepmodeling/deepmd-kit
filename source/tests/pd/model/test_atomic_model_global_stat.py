# SPDX-License-Identifier: LGPL-3.0-or-later
import tempfile
import unittest
from pathlib import (
    Path,
)
from typing import (
    Optional,
)

import h5py
import numpy as np
import paddle

from deepmd.dpmodel.atomic_model import DPAtomicModel as DPDPAtomicModel
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pd.model.atomic_model import (
    BaseAtomicModel,
    DPAtomicModel,
)
from deepmd.pd.model.descriptor import (
    DescrptDPA1,
    DescrptSeA,
)
from deepmd.pd.model.task.base_fitting import (
    BaseFitting,
)
from deepmd.pd.model.task.ener import (
    InvarFitting,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.utils import (
    to_numpy_array,
    to_paddle_tensor,
)
from deepmd.utils.path import (
    DPPath,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PD_FLOAT_PRECISION


class FooFitting(paddle.nn.Layer, BaseFitting):
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
                    "pix",
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
        descriptor: paddle.Tensor,
        atype: paddle.Tensor,
        gr: Optional[paddle.Tensor] = None,
        g2: Optional[paddle.Tensor] = None,
        h2: Optional[paddle.Tensor] = None,
        fparam: Optional[paddle.Tensor] = None,
        aparam: Optional[paddle.Tensor] = None,
    ):
        nf, nloc, _ = descriptor.shape
        ret = {}
        ret["foo"] = (
            paddle.to_tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            )
            .reshape([nf, nloc] + self.output_def()["foo"].shape)  # noqa: RUF005
            .to(env.GLOBAL_PD_FLOAT_PRECISION)
            .to(env.DEVICE)
        )
        ret["pix"] = (
            paddle.to_tensor(
                [
                    [3.0, 2.0, 1.0],
                    [6.0, 5.0, 4.0],
                ]
            )
            .reshape([nf, nloc] + self.output_def()["pix"].shape)  # noqa: RUF005
            .to(env.GLOBAL_PD_FLOAT_PRECISION)
            .to(env.DEVICE)
        )
        ret["bar"] = (
            paddle.to_tensor(
                [
                    [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                    [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
                ]
            )
            .reshape([nf, nloc] + self.output_def()["bar"].shape)  # noqa: RUF005
            .to(env.GLOBAL_PD_FLOAT_PRECISION)
            .to(env.DEVICE)
        )
        return ret


class TestAtomicModelStat(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def tearDown(self):
        self.tempdir.cleanup()

    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)
        nf, nloc, nnei = self.nlist.shape
        self.merged_output_stat = [
            {
                "coord": to_paddle_tensor(np.zeros([2, 3, 3])),
                "atype": to_paddle_tensor(
                    np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
                ),
                "atype_ext": to_paddle_tensor(
                    np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32)
                ),
                "box": to_paddle_tensor(np.zeros([2, 3, 3])),
                "natoms": to_paddle_tensor(
                    np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32)
                ),
                # bias of foo: 1, 3
                "foo": to_paddle_tensor(np.array([5.0, 7.0]).reshape(2, 1)),
                # no bias of pix
                # bias of bar: [1, 5], [3, 2]
                "bar": to_paddle_tensor(
                    np.array([5.0, 12.0, 7.0, 9.0]).reshape(2, 1, 2)
                ),
                "find_foo": np.float32(1.0),
                "find_bar": np.float32(1.0),
            }
        ]
        self.tempdir = tempfile.TemporaryDirectory()
        h5file = str((Path(self.tempdir.name) / "testcase.h5").resolve())
        with h5py.File(h5file, "w") as f:
            pass
        self.stat_file_path = DPPath(h5file, "a")

    def test_output_stat(self):
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
            to_paddle_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
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
        ).reshape([nf, nloc] + md0.fitting_output_def()["foo"].shape)  # noqa: RUF005
        expected_ret0["pix"] = np.array(
            [
                [3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["pix"].shape)  # noqa: RUF005
        expected_ret0["bar"] = np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["bar"].shape)  # noqa: RUF005
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret0[kk], expected_ret0[kk])

        # 2. test bias is applied
        md0.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )
        ret1 = md0.forward_common_atomic(*args)
        ret1 = cvt_ret(ret1)
        expected_std = np.ones((3, 2, 2))  # 3 keys, 2 atypes, 2 max dims.
        # nt x odim
        foo_bias = np.array([1.0, 3.0]).reshape(2, 1)
        bar_bias = np.array([1.0, 5.0, 3.0, 2.0]).reshape(2, 1, 2)
        expected_ret1 = {}
        expected_ret1["foo"] = ret0["foo"] + foo_bias[at]
        expected_ret1["pix"] = ret0["pix"]
        expected_ret1["bar"] = ret0["bar"] + bar_bias[at]
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], expected_ret1[kk])
        np.testing.assert_almost_equal(to_numpy_array(md0.out_std), expected_std)

        # 3. test bias load from file
        def raise_error():
            raise RuntimeError

        md0.compute_or_load_out_stat(raise_error, stat_file_path=self.stat_file_path)
        ret2 = md0.forward_common_atomic(*args)
        ret2 = cvt_ret(ret2)
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], ret2[kk])
        np.testing.assert_almost_equal(to_numpy_array(md0.out_std), expected_std)

        # 4. test change bias
        BaseAtomicModel.change_out_bias(
            md0, self.merged_output_stat, bias_adjust_mode="change-by-statistic"
        )
        args = [
            to_paddle_tensor(ii)
            for ii in [
                self.coord_ext,
                to_numpy_array(self.merged_output_stat[0]["atype_ext"]),
                self.nlist,
            ]
        ]
        ret3 = md0.forward_common_atomic(*args)
        ret3 = cvt_ret(ret3)
        ## model output on foo: [[2, 3, 6], [5, 8, 9]] given bias [1, 3]
        ## foo sumed: [11, 22] compared with [5, 7], fit target is [-6, -15]
        ## fit bias is [1, -8]
        ## old bias + fit bias [2, -5]
        ## new model output is [[3, 4, -2], [6, 0, 1]], which sumed to [5, 7]
        expected_ret3 = {}
        expected_ret3["foo"] = np.array([[3, 4, -2], [6, 0, 1]]).reshape(2, 3, 1)
        expected_ret3["pix"] = ret0["pix"]
        for kk in ["foo", "pix"]:
            np.testing.assert_almost_equal(ret3[kk], expected_ret3[kk])
        # bar is too complicated to be manually computed.
        np.testing.assert_almost_equal(to_numpy_array(md0.out_std), expected_std)

    def test_preset_bias(self):
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            sum(self.sel),
            self.nt,
        ).to(env.DEVICE)
        ft = FooFitting().to(env.DEVICE)
        type_map = ["foo", "bar"]
        preset_out_bias = {
            # "foo": np.array(3.0, 2.0]).reshape(2, 1),
            "foo": [None, 2],
            "bar": np.array([7.0, 5.0, 13.0, 11.0]).reshape(2, 1, 2),
        }
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
            preset_out_bias=preset_out_bias,
        ).to(env.DEVICE)
        args = [
            to_paddle_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
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
        ).reshape([nf, nloc] + md0.fitting_output_def()["foo"].shape)  # noqa: RUF005
        expected_ret0["pix"] = np.array(
            [
                [3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["pix"].shape)  # noqa: RUF005
        expected_ret0["bar"] = np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["bar"].shape)  # noqa: RUF005
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret0[kk], expected_ret0[kk])

        # 2. test bias is applied
        md0.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )
        ret1 = md0.forward_common_atomic(*args)
        ret1 = cvt_ret(ret1)
        # foo sums: [5, 7],
        # given bias of type 1 being 2, the bias left for type 0 is [5-2*1, 7-2*2] = [3,3]
        # the solution of type 0 is 1.8
        foo_bias = np.array([1.8, preset_out_bias["foo"][1]]).reshape(2, 1)
        bar_bias = preset_out_bias["bar"]
        expected_ret1 = {}
        expected_ret1["foo"] = ret0["foo"] + foo_bias[at]
        expected_ret1["pix"] = ret0["pix"]
        expected_ret1["bar"] = ret0["bar"] + bar_bias[at]
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], expected_ret1[kk])

        # 3. test bias load from file
        def raise_error():
            raise RuntimeError

        md0.compute_or_load_out_stat(raise_error, stat_file_path=self.stat_file_path)
        ret2 = md0.forward_common_atomic(*args)
        ret2 = cvt_ret(ret2)
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], ret2[kk])

        # 4. test change bias
        BaseAtomicModel.change_out_bias(
            md0, self.merged_output_stat, bias_adjust_mode="change-by-statistic"
        )
        args = [
            to_paddle_tensor(ii)
            for ii in [
                self.coord_ext,
                to_numpy_array(self.merged_output_stat[0]["atype_ext"]),
                self.nlist,
            ]
        ]
        ret3 = md0.forward_common_atomic(*args)
        ret3 = cvt_ret(ret3)
        ## model output on foo: [[2.8, 3.8, 5], [5.8, 7., 8.]] given bias [1.8, 2]
        ## foo sumed: [11.6, 20.8] compared with [5, 7], fit target is [-6.6, -13.8]
        ## fit bias is [-7, 2] (2 is assigned. -7 is fit to [-8.6, -17.8])
        ## old bias[1.8,2] + fit bias[-7, 2] = [-5.2, 4]
        ## new model output is [[-4.2, -3.2, 7], [-1.2, 9, 10]]
        expected_ret3 = {}
        expected_ret3["foo"] = np.array([[-4.2, -3.2, 7.0], [-1.2, 9.0, 10.0]]).reshape(
            2, 3, 1
        )
        expected_ret3["pix"] = ret0["pix"]
        for kk in ["foo", "pix"]:
            np.testing.assert_almost_equal(ret3[kk], expected_ret3[kk])
        # bar is too complicated to be manually computed.

    def test_preset_bias_all_none(self):
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            sum(self.sel),
            self.nt,
        ).to(env.DEVICE)
        ft = FooFitting().to(env.DEVICE)
        type_map = ["foo", "bar"]
        preset_out_bias = {
            "foo": [None, None],
        }
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
            preset_out_bias=preset_out_bias,
        ).to(env.DEVICE)
        args = [
            to_paddle_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
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
        ).reshape([nf, nloc] + md0.fitting_output_def()["foo"].shape)  # noqa: RUF005
        expected_ret0["pix"] = np.array(
            [
                [3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["pix"].shape)  # noqa: RUF005
        expected_ret0["bar"] = np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
            ]
        ).reshape([nf, nloc] + md0.fitting_output_def()["bar"].shape)  # noqa: RUF005
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret0[kk], expected_ret0[kk])

        # 2. test bias is applied
        md0.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )
        ret1 = md0.forward_common_atomic(*args)
        ret1 = cvt_ret(ret1)
        # nt x odim
        foo_bias = np.array([1.0, 3.0]).reshape(2, 1)
        bar_bias = np.array([1.0, 5.0, 3.0, 2.0]).reshape(2, 1, 2)
        expected_ret1 = {}
        expected_ret1["foo"] = ret0["foo"] + foo_bias[at]
        expected_ret1["pix"] = ret0["pix"]
        expected_ret1["bar"] = ret0["bar"] + bar_bias[at]
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], expected_ret1[kk])

    def test_serialize(self):
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(env.DEVICE)
        ft = InvarFitting(
            "foo",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        ).to(env.DEVICE)
        type_map = ["A", "B"]
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
        ).to(env.DEVICE)
        args = [
            to_paddle_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        # nf x nloc
        at = self.atype_ext[:, :nloc]

        def cvt_ret(x):
            return {kk: to_numpy_array(vv) for kk, vv in x.items()}

        md0.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )
        ret0 = md0.forward_common_atomic(*args)
        ret0 = cvt_ret(ret0)
        md1 = DPAtomicModel.deserialize(md0.serialize())
        ret1 = md1.forward_common_atomic(*args)
        ret1 = cvt_ret(ret1)

        for kk in ["foo"]:
            np.testing.assert_almost_equal(ret0[kk], ret1[kk])

        md2 = DPDPAtomicModel.deserialize(md0.serialize())
        args = [self.coord_ext, self.atype_ext, self.nlist]
        ret2 = md2.forward_common_atomic(*args)
        for kk in ["foo"]:
            np.testing.assert_almost_equal(ret0[kk], ret2[kk])
