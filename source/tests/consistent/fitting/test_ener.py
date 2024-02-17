# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
    Tuple,
)

import numpy as np

from deepmd.dpmodel.fitting.invar_fitting import EnergyFittingNet as EnerFittingDP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_PT,
    INSTALLED_TF,
    CommonTest,
    parameterized,
)
from .common import (
    FittingTest,
)

if INSTALLED_PT:
    import torch

    from deepmd.pt.model.task.ener import EnergyFittingNet as EnerFittingPT
    from deepmd.pt.utils.env import DEVICE as PT_DEVICE
else:
    EnerFittingPT = object
if INSTALLED_TF:
    from deepmd.tf.fit.ener import EnerFitting as EnerFittingTF
else:
    EnerFittingTF = object
from deepmd.utils.argcheck import (
    fitting_ener,
)


@parameterized(
    (True, False),  # resnet_dt
    ("float64", "float32"),  # precision
    (True, False),  # distinguish_types
    (0, 1),  # numb_fparam
)
class TestEner(CommonTest, FittingTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            resnet_dt,
            precision,
            distinguish_types,
            numb_fparam,
        ) = self.param
        return {
            "neuron": [5, 5, 5],
            "resnet_dt": resnet_dt,
            "precision": precision,
            "numb_fparam": numb_fparam,
            "seed": 20240217,
        }

    @property
    def skip_tf(self) -> bool:
        (
            resnet_dt,
            precision,
            distinguish_types,
            numb_fparam,
        ) = self.param
        # TODO: distinguish_types
        return not distinguish_types or CommonTest.skip_pt

    @property
    def skip_pt(self) -> bool:
        (
            resnet_dt,
            precision,
            distinguish_types,
            numb_fparam,
        ) = self.param
        # TODO: float32 has bug
        return precision == "float32" or CommonTest.skip_pt

    @property
    def skip_dp(self) -> bool:
        (
            resnet_dt,
            precision,
            distinguish_types,
            numb_fparam,
        ) = self.param
        # TODO: float32 has bug
        return precision == "float32" or CommonTest.skip_dp

    tf_class = EnerFittingTF
    dp_class = EnerFittingDP
    pt_class = EnerFittingPT
    args = fitting_ener()

    def setUp(self):
        CommonTest.setUp(self)

        self.ntypes = 2
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)
        self.inputs = np.ones((1, 6, 20), dtype=GLOBAL_NP_FLOAT_PRECISION)
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        # inconsistent if not sorted
        self.atype.sort()
        self.fparam = -np.ones((1,), dtype=GLOBAL_NP_FLOAT_PRECISION)

    @property
    def addtional_data(self) -> dict:
        (
            resnet_dt,
            precision,
            distinguish_types,
            numb_fparam,
        ) = self.param
        return {
            "ntypes": self.ntypes,
            "dim_descrpt": self.inputs.shape[-1],
            "distinguish_types": distinguish_types,
        }

    def build_tf(self, obj: Any, suffix: str) -> Tuple[list, dict]:
        (
            resnet_dt,
            precision,
            distinguish_types,
            numb_fparam,
        ) = self.param
        return self.build_tf_fitting(
            obj,
            self.inputs.ravel(),
            self.natoms,
            self.atype,
            self.fparam if numb_fparam else None,
            suffix,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            distinguish_types,
            numb_fparam,
        ) = self.param
        return (
            pt_obj(
                torch.from_numpy(self.inputs).to(device=PT_DEVICE),
                torch.from_numpy(self.atype).to(device=PT_DEVICE),
                fparam=torch.from_numpy(self.fparam).to(device=PT_DEVICE)
                if numb_fparam
                else None,
            )["energy"]
            .detach()
            .cpu()
            .numpy()
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            distinguish_types,
            numb_fparam,
        ) = self.param
        return dp_obj(
            self.inputs,
            self.atype,
            fparam=self.fparam if numb_fparam else None,
        )["energy"]

    def extract_ret(self, ret: Any, backend) -> Tuple[np.ndarray, ...]:
        if backend == self.RefBackend.TF:
            # shape is not same
            ret = ret[0].reshape(-1, self.natoms[0], 1)
        return (ret,)
