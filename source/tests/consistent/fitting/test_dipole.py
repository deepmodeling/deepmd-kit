# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
    Tuple,
)

import numpy as np

from deepmd.dpmodel.fitting.dipole_fitting import DipoleFitting as DipoleFittingDP
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
    DipoleFittingTest,
)

if INSTALLED_PT:
    import torch

    from deepmd.pt.model.task.dipole import DipoleFittingNet as DipoleFittingPT
    from deepmd.pt.utils.env import DEVICE as PT_DEVICE
else:
    DipoleFittingPT = object
if INSTALLED_TF:
    from deepmd.tf.fit.dipole import DipoleFittingSeA as DipoleFittingTF
else:
    DipoleFittingTF = object
from deepmd.utils.argcheck import (
    fitting_dipole,
)


@parameterized(
    (True, False),  # resnet_dt
    ("float64", "float32"),  # precision
    (True, False),  # mixed_types
)
class TestDipole(CommonTest, DipoleFittingTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            resnet_dt,
            precision,
            mixed_types,
        ) = self.param
        return {
            "neuron": [5, 5, 5],
            "resnet_dt": resnet_dt,
            "precision": precision,
            "seed": 20240217,
        }

    @property
    def skip_pt(self) -> bool:
        (
            resnet_dt,
            precision,
            mixed_types,
        ) = self.param
        return CommonTest.skip_pt

    tf_class = DipoleFittingTF
    dp_class = DipoleFittingDP
    pt_class = DipoleFittingPT
    args = fitting_dipole()

    def setUp(self):
        CommonTest.setUp(self)

        self.ntypes = 2
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)
        self.inputs = np.ones((1, 6, 20), dtype=GLOBAL_NP_FLOAT_PRECISION)
        self.gr = np.ones((1, 6, 30, 3), dtype=GLOBAL_NP_FLOAT_PRECISION)
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        # inconsistent if not sorted
        self.atype.sort()

    @property
    def addtional_data(self) -> dict:
        (
            resnet_dt,
            precision,
            mixed_types,
        ) = self.param
        return {
            "ntypes": self.ntypes,
            "dim_descrpt": self.inputs.shape[-1],
            "mixed_types": mixed_types,
            "var_name": "dipole",
            "embedding_width": 30,
        }

    def build_tf(self, obj: Any, suffix: str) -> Tuple[list, dict]:
        (
            resnet_dt,
            precision,
            mixed_types,
        ) = self.param
        return self.build_tf_fitting(
            obj,
            self.inputs.ravel(),
            self.gr,
            self.natoms,
            self.atype,
            None,
            suffix,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
        ) = self.param
        return (
            pt_obj(
                torch.from_numpy(self.inputs).to(device=PT_DEVICE),
                torch.from_numpy(self.atype.reshape(1, -1)).to(device=PT_DEVICE),
                torch.from_numpy(self.gr).to(device=PT_DEVICE),
                None,
            )["dipole"]
            .detach()
            .cpu()
            .numpy()
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
        ) = self.param
        return dp_obj(
            self.inputs,
            self.atype.reshape(1, -1),
            self.gr,
            None,
        )["dipole"]

    def extract_ret(self, ret: Any, backend) -> Tuple[np.ndarray, ...]:
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
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
