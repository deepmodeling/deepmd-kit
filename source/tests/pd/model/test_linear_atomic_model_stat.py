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

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pd.model.atomic_model import (
    DPAtomicModel,
    LinearEnergyAtomicModel,
)
from deepmd.pd.model.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pd.model.task.base_fitting import (
    BaseFitting,
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


class FooFittingA(paddle.nn.Layer, BaseFitting):
    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "energy",
                    [1],
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
        ret["energy"] = (
            paddle.to_tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            )
            .reshape([nf, nloc, *self.output_def()["energy"].shape])
            .to(env.GLOBAL_PD_FLOAT_PRECISION)
            .to(env.DEVICE)
        )

        return ret


class FooFittingB(paddle.nn.Layer, BaseFitting):
    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "energy",
                    [1],
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
        ret["energy"] = (
            paddle.to_tensor(
                [
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ]
            )
            .reshape([nf, nloc, *self.output_def()["energy"].shape])
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
                "energy": to_paddle_tensor(np.array([5.0, 7.0]).reshape(2, 1)),
                "find_energy": np.float32(1.0),
            }
        ]
        self.tempdir = tempfile.TemporaryDirectory()
        h5file = str((Path(self.tempdir.name) / "testcase.h5").resolve())
        with h5py.File(h5file, "w") as f:
            pass
        self.stat_file_path = DPPath(h5file, "a")

    def test_linear_atomic_model_stat_with_bias(self):
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            sum(self.sel),
            self.nt,
        ).to(env.DEVICE)
        ft_a = FooFittingA().to(env.DEVICE)
        ft_b = FooFittingB().to(env.DEVICE)
        type_map = ["foo", "bar"]
        md0 = DPAtomicModel(
            ds,
            ft_a,
            type_map=type_map,
        ).to(env.DEVICE)
        md1 = DPAtomicModel(
            ds,
            ft_b,
            type_map=type_map,
        ).to(env.DEVICE)
        linear_model = LinearEnergyAtomicModel([md0, md1], type_map=type_map).to(
            env.DEVICE
        )

        args = [
            to_paddle_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        # nf x nloc
        at = self.atype_ext[:, :nloc]

        # 1. test run without bias
        # nf x na x odim
        ret0 = linear_model.forward_common_atomic(*args)

        ret0 = to_numpy_array(ret0["energy"])
        ret_no_bias = []
        for md in linear_model.models:
            ret_no_bias.append(
                to_numpy_array(md.forward_common_atomic(*args)["energy"])
            )
        expected_ret0 = np.array(
            [
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ).reshape(nf, nloc, *linear_model.fitting_output_def()["energy"].shape)

        np.testing.assert_almost_equal(ret0, expected_ret0)

        # 2. test bias is applied
        linear_model.compute_or_load_out_stat(
            self.merged_output_stat, stat_file_path=self.stat_file_path
        )
        # bias applied to sub atomic models.
        ener_bias = np.array([1.0, 3.0]).reshape(2, 1)
        linear_ret = []
        for idx, md in enumerate(linear_model.models):
            ret = md.forward_common_atomic(*args)
            ret = to_numpy_array(ret["energy"])
            linear_ret.append(ret_no_bias[idx] + ener_bias[at])
            np.testing.assert_almost_equal((ret_no_bias[idx] + ener_bias[at]), ret)

        # linear model not adding bias again
        ret1 = linear_model.forward_common_atomic(*args)
        ret1 = to_numpy_array(ret1["energy"])
        np.testing.assert_almost_equal(np.mean(np.stack(linear_ret), axis=0), ret1)
