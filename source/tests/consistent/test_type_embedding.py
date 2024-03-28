# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
    Tuple,
)

import numpy as np

from deepmd.dpmodel.utils.type_embed import TypeEmbedNet as TypeEmbedNetDP
from deepmd.utils.argcheck import (
    type_embedding_args,
)

from .common import (
    INSTALLED_PT,
    INSTALLED_TF,
    CommonTest,
    parameterized,
)

if INSTALLED_PT:
    import torch

    from deepmd.pt.model.network.network import TypeEmbedNetConsistent as TypeEmbedNetPT
    from deepmd.pt.utils.env import DEVICE as PT_DEVICE
else:
    TypeEmbedNetPT = object
if INSTALLED_TF:
    from deepmd.tf.utils.type_embed import TypeEmbedNet as TypeEmbedNetTF
else:
    TypeEmbedNetTF = object


@parameterized(
    (True, False),  # resnet_dt
    ("float32", "float64"),  # precision
    (True, False),  # padding
)
class TestTypeEmbedding(CommonTest, unittest.TestCase):
    """Useful utilities for descriptor tests."""

    @property
    def data(self) -> dict:
        (
            resnet_dt,
            precision,
            padding,
        ) = self.param
        return {
            "neuron": [2, 4, 4],
            "resnet_dt": resnet_dt,
            "precision": precision,
            "seed": 20240327,
        }

    tf_class = TypeEmbedNetTF
    dp_class = TypeEmbedNetDP
    pt_class = TypeEmbedNetPT
    args = type_embedding_args()

    @property
    def addtional_data(self) -> dict:
        (
            resnet_dt,
            precision,
            padding,
        ) = self.param
        # implict argument not input by users
        return {
            "ntypes": self.ntypes,
            "padding": padding,
        }

    def setUp(self):
        CommonTest.setUp(self)

        self.ntypes = 2

    def build_tf(self, obj: Any, suffix: str) -> Tuple[list, dict]:
        return [
            obj.build(
                obj.ntypes,
                suffix=suffix,
            ),
        ], {}

    def eval_dp(self, dp_obj: Any) -> Any:
        return (dp_obj(),)

    def eval_pt(self, pt_obj: Any) -> Any:
        return [
            x.detach().cpu().numpy() if torch.is_tensor(x) else x
            for x in (pt_obj(device=PT_DEVICE),)
        ]

    def extract_ret(self, ret: Any, backend) -> Tuple[np.ndarray, ...]:
        return (ret[0],)

    @property
    def rtol(self) -> float:
        """Relative tolerance for comparing the return value."""
        (
            resnet_dt,
            precision,
            padding,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        elif precision == "bfloat16":
            return 1e-1
        else:
            raise ValueError(f"Unknown precision: {precision}")

    @property
    def atol(self) -> float:
        """Absolute tolerance for comparing the return value."""
        (
            resnet_dt,
            precision,
            padding,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        elif precision == "bfloat16":
            return 1e-1
        else:
            raise ValueError(f"Unknown precision: {precision}")
