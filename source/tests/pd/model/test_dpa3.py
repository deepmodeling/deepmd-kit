# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import paddle

from deepmd.dpmodel.descriptor.dpa3 import DescrptDPA3 as DPDescrptDPA3
from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)
from deepmd.pd.model.descriptor import (
    DescrptDPA3,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.env import (
    PRECISION_DICT,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)
from .test_mlp import (
    get_tols,
)

dtype = env.GLOBAL_PD_FLOAT_PRECISION


class TestDescrptDPA3(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_consistency(
        self,
    ) -> None:
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for (
            ua,
            rus,
            ruri,
            acr,
            acer,
            acus,
            nme,
            prec,
            ect,
        ) in itertools.product(
            [True, False],  # update_angle
            ["res_residual"],  # update_style
            ["norm", "const"],  # update_residual_init
            [0, 1],  # a_compress_rate
            [1, 2],  # a_compress_e_rate
            [True, False],  # a_compress_use_split
            [1, 2],  # n_multi_edge_message
            ["float64"],  # precision
            [False],  # use_econf_tebd
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            if prec == "float64":
                atol = 1e-8  # marginal GPU test cases...
            coord_ext = np.concatenate([self.coord_ext[:1], self.coord_ext[:1]], axis=0)
            atype_ext = np.concatenate([self.atype_ext[:1], self.atype_ext[:1]], axis=0)
            nlist = np.concatenate([self.nlist[:1], self.nlist[:1]], axis=0)
            mapping = np.concatenate([self.mapping[:1], self.mapping[:1]], axis=0)
            repflow = RepFlowArgs(
                n_dim=20,
                e_dim=10,
                a_dim=8,
                nlayers=3,
                e_rcut=self.rcut,
                e_rcut_smth=self.rcut_smth,
                e_sel=nnei,
                a_rcut=self.rcut - 0.1,
                a_rcut_smth=self.rcut_smth,
                a_sel=nnei - 1,
                a_compress_rate=acr,
                a_compress_e_rate=acer,
                a_compress_use_split=acus,
                n_multi_edge_message=nme,
                axis_neuron=4,
                update_angle=ua,
                update_style=rus,
                update_residual_init=ruri,
                smooth_edge_update=True,
            )

            # dpa3 new impl
            dd0 = DescrptDPA3(
                self.nt,
                repflow=repflow,
                # kwargs for descriptor
                exclude_types=[],
                precision=prec,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)

            dd0.repflows.mean = paddle.to_tensor(davg, dtype=dtype, place=env.DEVICE)
            dd0.repflows.stddev = paddle.to_tensor(dstd, dtype=dtype, place=env.DEVICE)
            rd0, _, _, _, _ = dd0(
                paddle.to_tensor(coord_ext, dtype=dtype, place=env.DEVICE),
                paddle.to_tensor(atype_ext, dtype=paddle.int64, place=env.DEVICE),
                paddle.to_tensor(nlist, dtype=paddle.int64, place=env.DEVICE),
                paddle.to_tensor(mapping, dtype=paddle.int64, place=env.DEVICE),
            )
            # serialization
            dd1 = DescrptDPA3.deserialize(dd0.serialize())
            rd1, _, _, _, _ = dd1(
                paddle.to_tensor(coord_ext, dtype=dtype, place=env.DEVICE),
                paddle.to_tensor(atype_ext, dtype=paddle.int64, place=env.DEVICE),
                paddle.to_tensor(nlist, dtype=paddle.int64, place=env.DEVICE),
                paddle.to_tensor(mapping, dtype=paddle.int64, place=env.DEVICE),
            )
            np.testing.assert_allclose(
                rd0.numpy(),
                rd1.numpy(),
                rtol=rtol,
                atol=atol,
            )
            # dp impl
            dd2 = DPDescrptDPA3.deserialize(dd0.serialize())
            rd2, _, _, _, _ = dd2.call(coord_ext, atype_ext, nlist, mapping)
            np.testing.assert_allclose(
                rd0.numpy(),
                rd2,
                rtol=rtol,
                atol=atol,
            )

    def test_jit(
        self,
    ) -> None:
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for (
            ua,
            rus,
            ruri,
            acr,
            acer,
            acus,
            nme,
            prec,
            ect,
        ) in itertools.product(
            [True],  # update_angle
            ["res_residual"],  # update_style
            ["const"],  # update_residual_init
            [0, 1],  # a_compress_rate
            [2],  # a_compress_e_rate
            [True],  # a_compress_use_split
            [1, 2],  # n_multi_edge_message
            ["float64"],  # precision
            [False],  # use_econf_tebd
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)

            repflow = RepFlowArgs(
                n_dim=20,
                e_dim=10,
                a_dim=8,
                nlayers=3,
                e_rcut=self.rcut,
                e_rcut_smth=self.rcut_smth,
                e_sel=nnei,
                a_rcut=self.rcut - 0.1,
                a_rcut_smth=self.rcut_smth,
                a_sel=nnei - 1,
                a_compress_rate=acr,
                a_compress_e_rate=acer,
                a_compress_use_split=acus,
                n_multi_edge_message=nme,
                axis_neuron=4,
                update_angle=ua,
                update_style=rus,
                update_residual_init=ruri,
                smooth_edge_update=True,
            )

            # dpa3 new impl
            dd0 = DescrptDPA3(
                self.nt,
                repflow=repflow,
                # kwargs for descriptor
                exclude_types=[],
                precision=prec,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)

            dd0.repflows.mean = paddle.to_tensor(davg, dtype=dtype, place=env.DEVICE)
            dd0.repflows.stddev = paddle.to_tensor(dstd, dtype=dtype, place=env.DEVICE)
            dd0.forward = paddle.jit.to_static(full_graph=True, backend=None)(
                dd0.forward
            )
            _ = dd0(
                paddle.to_tensor(self.coord_ext, dtype=dtype, place=env.DEVICE),
                paddle.to_tensor(self.atype_ext, dtype=paddle.int64, place=env.DEVICE),
                paddle.to_tensor(self.nlist, dtype=paddle.int64, place=env.DEVICE),
                paddle.to_tensor(self.mapping, dtype=paddle.int64, place=env.DEVICE),
            )

    @unittest.skipIf(
        not paddle.device.is_compiled_with_cinn() or not env.CINN,
        "disable by default for CINN compiler take minites to compile",
    )
    def test_cinn_compiler(
        self,
    ) -> None:
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for (
            ua,
            rus,
            ruri,
            acr,
            acer,
            acus,
            nme,
            prec,
            ect,
        ) in itertools.product(
            [True],  # update_angle
            ["res_residual"],  # update_style
            ["const"],  # update_residual_init
            [0, 1],  # a_compress_rate
            [2],  # a_compress_e_rate
            [True],  # a_compress_use_split
            [1, 2],  # n_multi_edge_message
            ["float32"],  # precision
            [False],  # use_econf_tebd
        ):
            dtype = PRECISION_DICT[prec]

            repflow = RepFlowArgs(
                n_dim=20,
                e_dim=10,
                a_dim=8,
                nlayers=3,
                e_rcut=self.rcut,
                e_rcut_smth=self.rcut_smth,
                e_sel=nnei,
                a_rcut=self.rcut - 0.1,
                a_rcut_smth=self.rcut_smth,
                a_sel=nnei - 1,
                a_compress_rate=acr,
                a_compress_e_rate=acer,
                a_compress_use_split=acus,
                n_multi_edge_message=nme,
                axis_neuron=4,
                update_angle=ua,
                update_style=rus,
                update_residual_init=ruri,
                smooth_edge_update=True,
            )

            # dpa3 new impl
            dd0 = DescrptDPA3(
                self.nt,
                repflow=repflow,
                # kwargs for descriptor
                exclude_types=[],
                precision=prec,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)

            dd0.repflows.mean = paddle.to_tensor(davg, dtype=dtype, place=env.DEVICE)
            dd0.repflows.stddev = paddle.to_tensor(dstd, dtype=dtype, place=env.DEVICE)
            dd0.forward = paddle.jit.to_static(full_graph=True, backend="CINN")(
                dd0.forward
            )
            _ = dd0(
                paddle.to_tensor(self.coord_ext, dtype=dtype, place=env.DEVICE),
                paddle.to_tensor(self.atype_ext, dtype=paddle.int64, place=env.DEVICE),
                paddle.to_tensor(self.nlist, dtype=paddle.int64, place=env.DEVICE),
                paddle.to_tensor(self.mapping, dtype=paddle.int64, place=env.DEVICE),
            )
