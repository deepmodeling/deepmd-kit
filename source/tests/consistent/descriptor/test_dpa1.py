# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
    Tuple,
)

import numpy as np
from dargs import (
    Argument,
)

from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DescrptDPA1DP
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
    DescriptorTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.descriptor.dpa1 import DescrptDPA1 as DescrptDPA1PT
else:
    DescrptDPA1PT = None
if INSTALLED_TF:
    from deepmd.tf.descriptor.se_atten import DescrptDPA1Compat as DescrptDPA1TF
else:
    DescrptDPA1TF = None
from deepmd.utils.argcheck import (
    descrpt_se_atten_args,
)


@parameterized(
    (True, False),  # resnet_dt
    ([], [[0, 1]]),  # excluded_types
    ("float32", "float64"),  # precision
    (0.0, 1e-8, 1e-2),  # env_protection
    (True, False),  # smooth_type_embedding
    (True, False),  # type_one_side
    (True, False),  # set_davg_zero
    (0, 2),  # attn_layer
)
class TestDPA1(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            resnet_dt,
            excluded_types,
            precision,
            env_protection,
            smooth_type_embedding,
            type_one_side,
            set_davg_zero,
            attn_layer,
        ) = self.param
        return {
            "sel": [10],
            "rcut_smth": 5.80,
            "rcut": 6.00,
            "neuron": [6, 12, 24],
            "ntypes": self.ntypes,
            "axis_neuron": 3,
            "tebd_dim": 4,
            # "tebd_input_mode": tebd_input_mode,
            "attn": 20,
            "attn_layer": attn_layer,
            "attn_dotr": True,
            "attn_mask": False,
            "scaling_factor": 1.0,
            "normalize": True,
            "temperature": 1.0,
            "concat_output_tebd": True,
            "resnet_dt": resnet_dt,
            "type_one_side": type_one_side,
            "exclude_types": excluded_types,
            "env_protection": env_protection,
            "precision": precision,
            "set_davg_zero": set_davg_zero,
            "smooth_type_embedding": smooth_type_embedding,
            "seed": 1145141919810,
        }

    @property
    def skip_pt(self) -> bool:
        (
            resnet_dt,
            excluded_types,
            precision,
            env_protection,
            smooth_type_embedding,
            type_one_side,
            set_davg_zero,
            attn_layer,
        ) = self.param
        return CommonTest.skip_pt

    @property
    def skip_dp(self) -> bool:
        (
            resnet_dt,
            excluded_types,
            precision,
            env_protection,
            smooth_type_embedding,
            type_one_side,
            set_davg_zero,
            attn_layer,
        ) = self.param
        return CommonTest.skip_pt

    @property
    def skip_tf(self) -> bool:
        (
            resnet_dt,
            excluded_types,
            precision,
            env_protection,
            smooth_type_embedding,
            type_one_side,
            set_davg_zero,
            attn_layer,
        ) = self.param
        # TODO (excluded_types != [] and attn_layer > 0) need fix
        return (
            env_protection != 0.0
            or smooth_type_embedding
            or (excluded_types != [] and attn_layer > 0)
        )

    tf_class = DescrptDPA1TF
    dp_class = DescrptDPA1DP
    pt_class = DescrptDPA1PT
    args = descrpt_se_atten_args().append(Argument("ntypes", int, optional=False))

    def setUp(self):
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
        (
            resnet_dt,
            excluded_types,
            precision,
            env_protection,
            smooth_type_embedding,
            type_one_side,
            set_davg_zero,
            attn_layer,
        ) = self.param

    def build_tf(self, obj: Any, suffix: str) -> Tuple[list, dict]:
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
            mixed_types=True,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return self.eval_pt_descriptor(
            pt_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            mixed_types=True,
        )

    def extract_ret(self, ret: Any, backend) -> Tuple[np.ndarray, ...]:
        return (ret[0],)

    @property
    def rtol(self) -> float:
        """Relative tolerance for comparing the return value."""
        (
            resnet_dt,
            excluded_types,
            precision,
            env_protection,
            smooth_type_embedding,
            type_one_side,
            set_davg_zero,
            attn_layer,
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
            excluded_types,
            precision,
            env_protection,
            smooth_type_embedding,
            type_one_side,
            set_davg_zero,
            attn_layer,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
