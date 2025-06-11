# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeA as DescrptSeADP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PD,
    INSTALLED_PT,
    INSTALLED_TF,
    CommonTest,
    parameterized,
)
from .common import (
    DescriptorTest,
)

if INSTALLED_PT:
    import torch

    from deepmd.pt.model.descriptor.se_a import DescrptSeA as DescrptSeAPT
    from deepmd.pt.utils import (
        env,
    )
else:
    DescrptSeAPT = None
if INSTALLED_TF:
    from deepmd.tf.descriptor.se_a import DescrptSeA as DescrptSeATF
else:
    DescrptSeATF = None
if INSTALLED_PD:
    import paddle

    from deepmd.pd.model.descriptor.se_a import DescrptSeA as DescrptSeAPD
    from deepmd.pd.utils.env import DEVICE as PD_DEVICE
else:
    DescrptSeAPD = None
from deepmd.utils.argcheck import (
    descrpt_se_a_args,
)

if INSTALLED_JAX:
    from deepmd.jax.descriptor.se_e2_a import DescrptSeA as DescrptSeAJAX
    from deepmd.jax.env import (
        jnp,
    )
else:
    DescrptSeAJAX = None
if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict

    from ...array_api_strict.descriptor.se_e2_a import (
        DescrptSeA as DescrptSeAArrayAPIStrict,
    )
else:
    DescrptSeAArrayAPIStrict = None


@parameterized(
    (True, False),  # resnet_dt
    (True, False),  # type_one_side
    ([], [[0, 1]]),  # excluded_types
    ("float32", "float64"),  # precision
    (0.0, 1e-8, 1e-2),  # env_protection
)
class TestSeA(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return {
            "sel": [9, 10],
            "rcut_smth": 5.80,
            "rcut": 6.00,
            "neuron": [6, 12, 24],
            "axis_neuron": 3,
            "resnet_dt": resnet_dt,
            "type_one_side": type_one_side,
            "exclude_types": excluded_types,
            "env_protection": env_protection,
            "precision": precision,
            "seed": 1145141919810,
        }

    @property
    def skip_pt(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return CommonTest.skip_pt

    @property
    def skip_dp(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return CommonTest.skip_dp

    @property
    def skip_tf(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return env_protection != 0.0

    @property
    def skip_jax(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return not type_one_side or not INSTALLED_JAX

    @property
    def skip_pd(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return not INSTALLED_PD

    @property
    def skip_array_api_strict(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return not type_one_side or not INSTALLED_ARRAY_API_STRICT

    tf_class = DescrptSeATF
    dp_class = DescrptSeADP
    pt_class = DescrptSeAPT
    jax_class = DescrptSeAJAX
    pd_class = DescrptSeAPD
    array_api_strict_class = DescrptSeAArrayAPIStrict
    args = descrpt_se_a_args()

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 2
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)
        # TF se_e2_a type_one_side=False requires atype sorted
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        if not type_one_side:
            idx = np.argsort(self.atype)
            self.atype = self.atype[idx]
            self.coords = self.coords.reshape(-1, 3)[idx].ravel()

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        return self.build_tf_descriptor(
            obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            suffix,
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        return self.eval_dp_descriptor(
            dp_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return self.eval_pt_descriptor(
            pt_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_jax(self, jax_obj: Any) -> Any:
        return self.eval_jax_descriptor(
            jax_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_pd(self, pd_obj: Any) -> Any:
        return self.eval_pd_descriptor(
            pd_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        return self.eval_array_api_strict_descriptor(
            array_api_strict_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        return (ret[0], ret[1])

    @property
    def rtol(self) -> float:
        """Relative tolerance for comparing the return value."""
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
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
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")


@parameterized(
    (True,),  # resnet_dt
    (True,),  # type_one_side
    ([],),  # excluded_types
    ("float64",),  # precision
    (0.0, 1e-8, 1e-2),  # env_protection
)
class TestSeAStat(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return {
            "sel": [9, 10],
            "rcut_smth": 5.80,
            "rcut": 6.00,
            "neuron": [6, 12, 24],
            "axis_neuron": 3,
            "resnet_dt": resnet_dt,
            "type_one_side": type_one_side,
            "exclude_types": excluded_types,
            "env_protection": env_protection,
            "precision": precision,
            "seed": 1145141919810,
        }

    @property
    def skip_pt(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return CommonTest.skip_pt

    @property
    def skip_dp(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return CommonTest.skip_dp

    @property
    def skip_tf(self) -> bool:
        return True

    @property
    def skip_jax(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return not type_one_side or not INSTALLED_JAX

    @property
    def skip_pd(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return not INSTALLED_PD

    @property
    def skip_array_api_strict(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return not type_one_side or not INSTALLED_ARRAY_API_STRICT

    tf_class = DescrptSeATF
    dp_class = DescrptSeADP
    pt_class = DescrptSeAPT
    jax_class = DescrptSeAJAX
    pd_class = DescrptSeAPD
    array_api_strict_class = DescrptSeAArrayAPIStrict
    args = descrpt_se_a_args()

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 2
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)
        # TF se_e2_a type_one_side=False requires atype sorted
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        if not type_one_side:
            idx = np.argsort(self.atype)
            self.atype = self.atype[idx]
            self.coords = self.coords.reshape(-1, 3)[idx].ravel()

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        return self.build_tf_descriptor(
            obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            suffix,
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        dp_obj.compute_input_stats(
            [
                {
                    "r0": None,
                    "coord": self.coords.reshape(-1, self.natoms[0], 3),
                    "atype": self.atype.reshape(1, -1),
                    "box": self.box.reshape(1, 3, 3),
                    "natoms": self.natoms[0],
                }
            ]
        )
        return self.eval_dp_descriptor(
            dp_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        pt_obj.compute_input_stats(
            [
                {
                    "r0": None,
                    "coord": torch.from_numpy(self.coords)
                    .reshape(-1, self.natoms[0], 3)
                    .to(env.DEVICE),
                    "atype": torch.from_numpy(self.atype.reshape(1, -1)).to(env.DEVICE),
                    "box": torch.from_numpy(self.box.reshape(1, 3, 3)).to(env.DEVICE),
                    "natoms": self.natoms[0],
                }
            ]
        )
        return self.eval_pt_descriptor(
            pt_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_jax(self, jax_obj: Any) -> Any:
        jax_obj.compute_input_stats(
            [
                {
                    "r0": None,
                    "coord": jnp.array(self.coords).reshape(-1, self.natoms[0], 3),
                    "atype": jnp.array(self.atype.reshape(1, -1)),
                    "box": jnp.array(self.box.reshape(1, 3, 3)),
                    "natoms": self.natoms[0],
                }
            ]
        )
        return self.eval_jax_descriptor(
            jax_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_pd(self, pd_obj: Any) -> Any:
        pd_obj.compute_input_stats(
            [
                {
                    "r0": None,
                    "coord": paddle.to_tensor(
                        self.coords.reshape(-1, self.natoms[0], 3)
                    ).to(PD_DEVICE),
                    "atype": paddle.to_tensor(self.atype.reshape(1, -1)).to(PD_DEVICE),
                    "box": paddle.to_tensor(self.box.reshape(1, 3, 3)).to(PD_DEVICE),
                    "natoms": self.natoms[0],
                }
            ]
        )
        return self.eval_pd_descriptor(
            pd_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        array_api_strict_obj.compute_input_stats(
            [
                {
                    "r0": None,
                    "coord": array_api_strict.asarray(
                        self.coords.reshape(-1, self.natoms[0], 3)
                    ),
                    "atype": array_api_strict.asarray(self.atype.reshape(1, -1)),
                    "box": array_api_strict.asarray(self.box.reshape(1, 3, 3)),
                    "natoms": self.natoms[0],
                }
            ]
        )
        return self.eval_array_api_strict_descriptor(
            array_api_strict_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        return (ret[0], ret[1])

    @property
    def rtol(self) -> float:
        """Relative tolerance for comparing the return value."""
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
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
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
