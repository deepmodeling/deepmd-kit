# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.fitting.property_fitting import (
    PropertyFittingNet as PropertyFittingDP,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils.argcheck import (
    fitting_property,
)

from ..common import (
    INSTALLED_PT,
    CommonTest,
    parameterized,
)
from .common import (
    FittingTest,
)

if INSTALLED_PT:
    import torch

    from deepmd.pt.model.task.property import PropertyFittingNet as PropertyFittingPT
    from deepmd.pt.utils.env import DEVICE as PT_DEVICE
else:
    PropertyFittingPT = object
PropertyFittingTF = object


@parameterized(
    (True, False),  # resnet_dt
    ("float64", "float32"),  # precision
    (True, False),  # mixed_types
    (0, 1),  # numb_fparam
    (1, 3),  # task_dim
    (True, False),  # intensive
)
class TestProperty(CommonTest, FittingTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            task_dim,
            intensive,
        ) = self.param
        return {
            "neuron": [5, 5, 5],
            "resnet_dt": resnet_dt,
            "precision": precision,
            "numb_fparam": numb_fparam,
            "seed": 20240217,
            "task_dim": task_dim,
            "intensive": intensive,
        }

    @property
    def skip_pt(self) -> bool:
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            task_dim,
            intensive,
        ) = self.param
        return CommonTest.skip_pt

    @property
    def skip_tf(self) -> bool:
        return True

    tf_class = PropertyFittingTF
    dp_class = PropertyFittingDP
    pt_class = PropertyFittingPT
    args = fitting_property()

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
            mixed_types,
            numb_fparam,
            task_dim,
            intensive,
        ) = self.param
        return {
            "ntypes": self.ntypes,
            "dim_descrpt": self.inputs.shape[-1],
            "mixed_types": mixed_types,
        }

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            task_dim,
            intensive,
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
            mixed_types,
            numb_fparam,
            task_dim,
            intensive,
        ) = self.param
        return (
            pt_obj(
                torch.from_numpy(self.inputs).to(device=PT_DEVICE),
                torch.from_numpy(self.atype.reshape(1, -1)).to(device=PT_DEVICE),
                fparam=torch.from_numpy(self.fparam).to(device=PT_DEVICE)
                if numb_fparam
                else None,
            )["property"]
            .detach()
            .cpu()
            .numpy()
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            task_dim,
            intensive,
        ) = self.param
        return dp_obj(
            self.inputs,
            self.atype.reshape(1, -1),
            fparam=self.fparam if numb_fparam else None,
        )["property"]

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        if backend == self.RefBackend.TF:
            # shape is not same
            ret = ret[0].reshape(-1, self.natoms[0], 1)
        return (ret,)

    @property
    def rtol(self) -> float:
        """Relative tolerance for comparing the return value."""
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            task_dim,
            intensive,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")

    @property
    def atol(self) -> float:
        """Absolute tolerance for comparing the return value."""
        (
            resnet_dt,
            precision,
            mixed_types,
            numb_fparam,
            task_dim,
            intensive,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
