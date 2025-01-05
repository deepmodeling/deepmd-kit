# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.type_embed import TypeEmbedNet as TypeEmbedNetDP
from deepmd.utils.argcheck import (
    type_embedding_args,
)

from .common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PD,
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
if INSTALLED_JAX:
    from deepmd.jax.env import (
        jnp,
    )
    from deepmd.jax.utils.type_embed import TypeEmbedNet as TypeEmbedNetJAX
else:
    TypeEmbedNetJAX = object
if INSTALLED_ARRAY_API_STRICT:
    from ..array_api_strict.utils.type_embed import TypeEmbedNet as TypeEmbedNetStrict
else:
    TypeEmbedNetStrict = None
if INSTALLED_PD:
    import paddle

    from deepmd.pd.model.network.network import TypeEmbedNetConsistent as TypeEmbedNetPD
    from deepmd.pd.utils.env import DEVICE as PD_DEVICE
else:
    TypeEmbedNetPD = object


@parameterized(
    (True, False),  # resnet_dt
    ("float32", "float64"),  # precision
    (True, False),  # padding
    (True, False),  # use_econf_tebd
    (True, False),  # use_tebd_bias
)
class TestTypeEmbedding(CommonTest, unittest.TestCase):
    """Useful utilities for descriptor tests."""

    @property
    def data(self) -> dict:
        (
            resnet_dt,
            precision,
            padding,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        return {
            "neuron": [2, 4, 4],
            "resnet_dt": resnet_dt,
            "precision": precision,
            "use_econf_tebd": use_econf_tebd,
            "use_tebd_bias": use_tebd_bias,
            "seed": 20240327,
        }

    tf_class = TypeEmbedNetTF
    dp_class = TypeEmbedNetDP
    pt_class = TypeEmbedNetPT
    jax_class = TypeEmbedNetJAX
    pd_class = TypeEmbedNetPD
    array_api_strict_class = TypeEmbedNetStrict
    args = type_embedding_args()
    skip_jax = not INSTALLED_JAX
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT

    @property
    def additional_data(self) -> dict:
        (
            resnet_dt,
            precision,
            padding,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        # implicit argument not input by users
        return {
            "ntypes": self.ntypes,
            "padding": padding,
            "type_map": ["O", "H"] if use_econf_tebd else None,
        }

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 2

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
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

    def eval_jax(self, jax_obj: Any) -> Any:
        out = jax_obj()
        # ensure output is not numpy array
        for x in (out,):
            if isinstance(x, np.ndarray):
                raise ValueError("Output is numpy array")
        return [np.array(x) if isinstance(x, jnp.ndarray) else x for x in (out,)]

    def eval_pd(self, pd_obj: Any) -> Any:
        return [
            x.detach().cpu().numpy() if paddle.is_tensor(x) else x
            for x in (pd_obj(device=PD_DEVICE),)
        ]

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        out = array_api_strict_obj()
        return [
            to_numpy_array(x) if hasattr(x, "__array_namespace__") else x
            for x in (out,)
        ]

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        return (ret[0],)

    @property
    def rtol(self) -> float:
        """Relative tolerance for comparing the return value."""
        (
            resnet_dt,
            precision,
            padding,
            use_econf_tebd,
            use_tebd_bias,
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
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        elif precision == "bfloat16":
            return 1e-1
        else:
            raise ValueError(f"Unknown precision: {precision}")
