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
import torch

from deepmd.dpmodel.atomic_model import DPAtomicModel as DPDPAtomicModel
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt_expt.atomic_model import (
    DPAtomicModel,
)
from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    InvarFitting,
)
from deepmd.pt_expt.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.utils.path import (
    DPPath,
)

from ...pt.model.test_env_mat import (
    TestCaseSingleFrameWithNlist,
)
from ...seed import (
    GLOBAL_SEED,
)


class FooFitting(BaseFitting, torch.nn.Module):
    """Test fitting with multiple outputs for testing global statistics."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        BaseFitting.__init__(self)

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

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
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
            .to(dtype=torch.float64, device=env.DEVICE)
        )
        ret["pix"] = (
            torch.Tensor(
                [
                    [3.0, 2.0, 1.0],
                    [6.0, 5.0, 4.0],
                ]
            )
            .view([nf, nloc, *self.output_def()["pix"].shape])
            .to(dtype=torch.float64, device=env.DEVICE)
        )
        ret["bar"] = (
            torch.Tensor(
                [
                    [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                    [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
                ]
            )
            .view([nf, nloc, *self.output_def()["bar"].shape])
            .to(dtype=torch.float64, device=env.DEVICE)
        )
        return ret


def _to_numpy(x):
    return x.detach().cpu().numpy()


class TestAtomicModelStat(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE
        self.merged_output_stat = [
            {
                "coord": torch.tensor(np.zeros([2, 3, 3]), device=self.device),
                "atype": torch.tensor(
                    np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32),
                    device=self.device,
                ),
                "atype_ext": torch.tensor(
                    np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32),
                    device=self.device,
                ),
                "box": torch.tensor(np.zeros([2, 3, 3]), device=self.device),
                "natoms": torch.tensor(
                    np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32),
                    device=self.device,
                ),
                # bias of foo: 1, 3
                "foo": torch.tensor(
                    np.array([5.0, 7.0]).reshape(2, 1), device=self.device
                ),
                # no bias of pix
                # bias of bar: [1, 5], [3, 2]
                "bar": torch.tensor(
                    np.array([5.0, 12.0, 7.0, 9.0]).reshape(2, 1, 2),
                    device=self.device,
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

    def test_output_stat(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(self.device)
        ft = FooFitting().to(self.device)
        type_map = ["foo", "bar"]
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
        ).to(self.device)
        args = [
            torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device),
            torch.tensor(self.atype_ext, dtype=torch.int64, device=self.device),
            torch.tensor(self.nlist, dtype=torch.int64, device=self.device),
        ]
        # nf x nloc
        at = self.atype_ext[:, :nloc]

        def cvt_ret(x):
            return {kk: _to_numpy(vv) for kk, vv in x.items()}

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
        expected_ret0["pix"] = np.array(
            [
                [3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0],
            ]
        ).reshape([nf, nloc, *md0.fitting_output_def()["pix"].shape])
        expected_ret0["bar"] = np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
            ]
        ).reshape([nf, nloc, *md0.fitting_output_def()["bar"].shape])
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
        np.testing.assert_almost_equal(_to_numpy(md0.out_std), expected_std)

        # 3. test bias load from file
        def raise_error() -> NoReturn:
            raise RuntimeError

        md0.compute_or_load_out_stat(raise_error, stat_file_path=self.stat_file_path)
        ret2 = md0.forward_common_atomic(*args)
        ret2 = cvt_ret(ret2)
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], ret2[kk])
        np.testing.assert_almost_equal(_to_numpy(md0.out_std), expected_std)

        # 4. test change bias
        md0.change_out_bias(
            self.merged_output_stat, bias_adjust_mode="change-by-statistic"
        )
        # use atype_ext from merged_output_stat for inference (matching pt backend test)
        args = [
            torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device),
            self.merged_output_stat[0]["atype_ext"].to(
                dtype=torch.int64, device=self.device
            ),
            torch.tensor(self.nlist, dtype=torch.int64, device=self.device),
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
        np.testing.assert_almost_equal(_to_numpy(md0.out_std), expected_std)

    def test_preset_bias(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(self.device)
        ft = FooFitting().to(self.device)
        type_map = ["foo", "bar"]
        preset_out_bias = {
            "foo": [None, 2],
            "bar": np.array([7.0, 5.0, 13.0, 11.0]).reshape(2, 1, 2),
        }
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
            preset_out_bias=preset_out_bias,
        ).to(self.device)
        args = [
            torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device),
            torch.tensor(self.atype_ext, dtype=torch.int64, device=self.device),
            torch.tensor(self.nlist, dtype=torch.int64, device=self.device),
        ]
        # nf x nloc
        at = self.atype_ext[:, :nloc]

        def cvt_ret(x):
            return {kk: _to_numpy(vv) for kk, vv in x.items()}

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
        expected_ret0["pix"] = np.array(
            [
                [3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0],
            ]
        ).reshape([nf, nloc, *md0.fitting_output_def()["pix"].shape])
        expected_ret0["bar"] = np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
            ]
        ).reshape([nf, nloc, *md0.fitting_output_def()["bar"].shape])
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
        def raise_error() -> NoReturn:
            raise RuntimeError

        md0.compute_or_load_out_stat(raise_error, stat_file_path=self.stat_file_path)
        ret2 = md0.forward_common_atomic(*args)
        ret2 = cvt_ret(ret2)
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret1[kk], ret2[kk])

        # 4. test change bias
        md0.change_out_bias(
            self.merged_output_stat, bias_adjust_mode="change-by-statistic"
        )
        # use atype_ext from merged_output_stat for inference
        args = [
            torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device),
            self.merged_output_stat[0]["atype_ext"].to(
                dtype=torch.int64, device=self.device
            ),
            torch.tensor(self.nlist, dtype=torch.int64, device=self.device),
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

    def test_preset_bias_all_none(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(self.device)
        ft = FooFitting().to(self.device)
        type_map = ["foo", "bar"]
        preset_out_bias = {
            "foo": [None, None],
        }
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
            preset_out_bias=preset_out_bias,
        ).to(self.device)
        args = [
            torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device),
            torch.tensor(self.atype_ext, dtype=torch.int64, device=self.device),
            torch.tensor(self.nlist, dtype=torch.int64, device=self.device),
        ]
        # nf x nloc
        at = self.atype_ext[:, :nloc]

        def cvt_ret(x):
            return {kk: _to_numpy(vv) for kk, vv in x.items()}

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
        expected_ret0["pix"] = np.array(
            [
                [3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0],
            ]
        ).reshape([nf, nloc, *md0.fitting_output_def()["pix"].shape])
        expected_ret0["bar"] = np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
            ]
        ).reshape([nf, nloc, *md0.fitting_output_def()["bar"].shape])
        for kk in ["foo", "pix", "bar"]:
            np.testing.assert_almost_equal(ret0[kk], expected_ret0[kk])

        # 2. test bias is applied (all None preset = same as no preset)
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

    def test_serialize(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(self.device)
        ft = InvarFitting(
            "foo",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(self.device)
        type_map = ["A", "B"]
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
        ).to(self.device)
        args = [
            torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device),
            torch.tensor(self.atype_ext, dtype=torch.int64, device=self.device),
            torch.tensor(self.nlist, dtype=torch.int64, device=self.device),
        ]

        def cvt_ret(x):
            return {kk: _to_numpy(vv) for kk, vv in x.items()}

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
        args_np = [self.coord_ext, self.atype_ext, self.nlist]
        ret2 = md2.forward_common_atomic(*args_np)
        for kk in ["foo"]:
            np.testing.assert_almost_equal(ret0[kk], ret2[kk])


class TestChangeByStatMixedLabels(unittest.TestCase, TestCaseSingleFrameWithNlist):
    """Test change-by-statistic with mixed atomic and global labels."""

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE
        self.merged_output_stat = [
            {
                "coord": torch.tensor(np.zeros([2, 3, 3]), device=self.device),
                "atype": torch.tensor(
                    np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32),
                    device=self.device,
                ),
                "atype_ext": torch.tensor(
                    np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32),
                    device=self.device,
                ),
                "box": torch.tensor(np.zeros([2, 3, 3]), device=self.device),
                "natoms": torch.tensor(
                    np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32),
                    device=self.device,
                ),
                # foo: atomic label
                "atom_foo": torch.tensor(
                    np.array([[5.0, 5.0, 5.0], [5.0, 6.0, 7.0]]).reshape(2, 3, 1),
                    device=self.device,
                ),
                # pix: global label
                "pix": torch.tensor(
                    np.array([5.0, 12.0]).reshape(2, 1), device=self.device
                ),
                # bar: global label
                "bar": torch.tensor(
                    np.array([5.0, 12.0, 7.0, 9.0]).reshape(2, 1, 2),
                    device=self.device,
                ),
                "find_atom_foo": np.float32(1.0),
                "find_pix": np.float32(1.0),
                "find_bar": np.float32(1.0),
            },
        ]
        self.tempdir = tempfile.TemporaryDirectory()
        h5file = str((Path(self.tempdir.name) / "testcase.h5").resolve())
        with h5py.File(h5file, "w") as f:
            pass
        self.stat_file_path = DPPath(h5file, "a")

    def test_change_by_statistic(self) -> None:
        """Test change-by-statistic with atomic foo + global pix + global bar."""
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(self.device)
        ft = FooFitting().to(self.device)
        type_map = ["foo", "bar"]
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
        ).to(self.device)
        args = [
            torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device),
            torch.tensor(self.atype_ext, dtype=torch.int64, device=self.device),
            torch.tensor(self.nlist, dtype=torch.int64, device=self.device),
        ]

        def cvt_ret(x):
            return {kk: _to_numpy(vv) for kk, vv in x.items()}

        ret0 = md0.forward_common_atomic(*args)
        ret0 = cvt_ret(ret0)

        # set initial bias
        md0.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )

        # change bias
        md0.change_out_bias(
            self.merged_output_stat, bias_adjust_mode="change-by-statistic"
        )
        # use atype_ext from merged_output_stat for inference
        args = [
            torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device),
            self.merged_output_stat[0]["atype_ext"].to(
                dtype=torch.int64, device=self.device
            ),
            torch.tensor(self.nlist, dtype=torch.int64, device=self.device),
        ]
        ret3 = md0.forward_common_atomic(*args)
        ret3 = cvt_ret(ret3)
        # foo: atomic label, bias after set-by-stat: [5, 6]
        # model output with bias [5,6], atype [[0,0,1],[0,1,1]]:
        #   [[6, 7, 9], [9, 11, 12]]
        # atom_foo labels: [[5, 5, 5], [5, 6, 7]]
        # per-atom delta: [[-1, -2, -4], [-4, -5, -5]]
        # delta bias (mean per type): type0=-7/3, type1=-14/3
        # new bias = [5-7/3, 6-14/3] = [8/3, 4/3]
        # new output: [[11/3, 14/3, 13/3], [20/3, 19/3, 22/3]]
        expected_ret3 = {}
        expected_ret3["foo"] = np.array(
            [[3.6667, 4.6667, 4.3333], [6.6667, 6.3333, 7.3333]]
        ).reshape(2, 3, 1)
        # pix: global label, bias after set-by-stat: [-2/3, 19/3]
        # model pix with bias, atype [[0,0,1],[0,1,1]]:
        #   [[7/3, 4/3, 22/3], [16/3, 34/3, 31/3]], sums [11, 27]
        # labels [5, 12], delta [-6, -15]
        # lstsq: delta bias [1, -8], new bias [1/3, -5/3]
        # new output: [[10/3, 7/3, -2/3], [19/3, 10/3, 7/3]]
        expected_ret3["pix"] = np.array(
            [[3.3333, 2.3333, -0.6667], [6.3333, 3.3333, 2.3333]]
        ).reshape(2, 3, 1)
        for kk in ["foo", "pix"]:
            np.testing.assert_almost_equal(ret3[kk], expected_ret3[kk], decimal=4)
        # bar is too complicated to be manually computed.


class TestEnergyModelStat(unittest.TestCase, TestCaseSingleFrameWithNlist):
    """Test statistics computation with real energy fitting net."""

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE
        self.merged_output_stat = [
            {
                "coord": torch.tensor(np.zeros([2, 3, 3]), device=self.device),
                "atype": torch.tensor(
                    np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32),
                    device=self.device,
                ),
                "atype_ext": torch.tensor(
                    np.array([[0, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int32),
                    device=self.device,
                ),
                "box": torch.tensor(np.zeros([2, 3, 3]), device=self.device),
                "natoms": torch.tensor(
                    np.array([[3, 3, 2, 1], [3, 3, 1, 2]], dtype=np.int32),
                    device=self.device,
                ),
                # energy data
                "energy": torch.tensor(
                    np.array([10.0, 20.0]).reshape(2, 1), device=self.device
                ),
                "find_energy": np.float32(1.0),
            },
        ]
        self.tempdir = tempfile.TemporaryDirectory()
        h5file = str((Path(self.tempdir.name) / "testcase.h5").resolve())
        with h5py.File(h5file, "w") as f:
            pass
        self.stat_file_path = DPPath(h5file, "a")

    def test_energy_stat(self) -> None:
        """Test energy statistics computation with real energy fitting net."""
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(self.device)
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(self.device)
        type_map = ["foo", "bar"]
        md0 = DPAtomicModel(
            ds,
            ft,
            type_map=type_map,
        ).to(self.device)
        args = [
            torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device),
            torch.tensor(self.atype_ext, dtype=torch.int64, device=self.device),
            torch.tensor(self.nlist, dtype=torch.int64, device=self.device),
        ]

        # test run without bias
        ret0 = md0.forward_common_atomic(*args)
        self.assertIn("energy", ret0)

        # compute statistics
        md0.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )
        ret1 = md0.forward_common_atomic(*args)
        self.assertIn("energy", ret1)

        # Check that bias was computed (out_bias should be non-zero)
        self.assertFalse(torch.all(md0.out_bias == 0))

        # test bias load from file
        def raise_error() -> NoReturn:
            raise RuntimeError

        md0.compute_or_load_out_stat(raise_error, stat_file_path=self.stat_file_path)
        ret2 = md0.forward_common_atomic(*args)
        np.testing.assert_allclose(
            ret1["energy"].detach().cpu().numpy(),
            ret2["energy"].detach().cpu().numpy(),
        )

        # test change bias
        md0.change_out_bias(
            self.merged_output_stat, bias_adjust_mode="change-by-statistic"
        )
        ret3 = md0.forward_common_atomic(*args)
        self.assertIn("energy", ret3)
