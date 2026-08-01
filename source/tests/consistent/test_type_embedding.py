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
    INSTALLED_TF2,
    CommonTest,
    parameterized_cases,
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
if INSTALLED_TF2:
    from deepmd.tf2.utils.type_embed import TypeEmbedNet as TypeEmbedNetTF2
else:
    TypeEmbedNetTF2 = None


TYPE_EMBEDDING_CASE_FIELDS = (
    "resnet_dt",
    "precision",
    "padding",
    "use_econf_tebd",
    "use_tebd_bias",
)

TYPE_EMBEDDING_BASELINE_CASE = {
    "resnet_dt": True,
    "precision": "float64",
    "padding": True,
    "use_econf_tebd": True,
    "use_tebd_bias": True,
}


def type_embedding_case(**overrides: Any) -> tuple:
    unknown = set(overrides) - set(TYPE_EMBEDDING_CASE_FIELDS)
    if unknown:
        raise ValueError(f"Unknown type-embedding case fields: {sorted(unknown)}")
    case = TYPE_EMBEDDING_BASELINE_CASE | overrides
    return tuple(case[field] for field in TYPE_EMBEDDING_CASE_FIELDS)


TYPE_EMBEDDING_CURATED_CASES = (
    type_embedding_case(),
    type_embedding_case(resnet_dt=False),
    type_embedding_case(precision="float32"),
    type_embedding_case(padding=False),
    type_embedding_case(use_econf_tebd=False),
    type_embedding_case(use_tebd_bias=False),
    type_embedding_case(
        resnet_dt=False,
        precision="float32",
        padding=False,
        use_econf_tebd=False,
        use_tebd_bias=False,
    ),
)


@parameterized_cases(*TYPE_EMBEDDING_CURATED_CASES)
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
    tf2_class = TypeEmbedNetTF2
    dp_class = TypeEmbedNetDP
    pt_class = TypeEmbedNetPT
    jax_class = TypeEmbedNetJAX
    pd_class = TypeEmbedNetPD
    array_api_strict_class = TypeEmbedNetStrict
    args = type_embedding_args()
    skip_jax = not INSTALLED_JAX
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT
    skip_tf2 = not INSTALLED_TF2

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

    def eval_tf2(self, tf2_obj: Any) -> Any:
        out = tf2_obj()
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


@unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
class TestDpmodelElectronicConfigurationBackend(unittest.TestCase):
    """Exercise dpmodel with backend weights but portable NumPy constants."""

    def test_econf_input_follows_torch_weights(self) -> None:
        type_embedding = TypeEmbedNetDP(
            ntypes=2,
            neuron=[4, 4],
            precision="float64",
            use_econf_tebd=True,
            type_map=["O", "H"],
            seed=20260717,
        )
        self.assertIsInstance(type_embedding.econf_tebd, np.ndarray)
        numpy_reference = TypeEmbedNetDP.deserialize(type_embedding.serialize())
        expected = numpy_reference()

        # Model converters can replace the trainable arrays without touching
        # portable constants. This reproduces that mixed-backend boundary
        # directly, without relying on a wrapper that eagerly converts both.
        for layer in type_embedding.embedding_net.layers:
            for name in ("w", "b", "idt"):
                value = getattr(layer, name)
                if isinstance(value, np.ndarray):
                    setattr(layer, name, torch.as_tensor(value))

        sample_weight = type_embedding.embedding_net[0]["w"]
        result = type_embedding()

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, sample_weight.dtype)
        self.assertEqual(result.device, sample_weight.device)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected)

        # change_type_map intentionally regenerates the portable NumPy table;
        # call-time conversion must continue to follow the existing weights.
        new_type_map = ["C", "H", "O"]
        numpy_reference.change_type_map(new_type_map)
        type_embedding.change_type_map(new_type_map)
        remapped_result = type_embedding()

        self.assertIsInstance(type_embedding.econf_tebd, np.ndarray)
        self.assertIsInstance(remapped_result, torch.Tensor)
        self.assertEqual(remapped_result.dtype, sample_weight.dtype)
        self.assertEqual(remapped_result.device, sample_weight.device)
        np.testing.assert_allclose(
            remapped_result.detach().cpu().numpy(), numpy_reference()
        )
