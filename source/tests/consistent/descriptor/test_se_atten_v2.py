# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
    Optional,
    Tuple,
)

import numpy as np
from dargs import (
    Argument,
)

from deepmd.dpmodel.descriptor.se_atten_v2 import DescrptSeAttenV2 as DescrptSeAttenV2DP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_PT,
    CommonTest,
    parameterized,
)
from .common import (
    DescriptorTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.descriptor.se_atten_v2 import (
        DescrptSeAttenV2 as DescrptSeAttenV2PT,
    )
else:
    DescrptSeAttenV2PT = None
DescrptSeAttenV2TF = None
from deepmd.utils.argcheck import (
    descrpt_se_atten_args,
)


@parameterized(
    (4,),  # tebd_dim
    (True,),  # resnet_dt
    (True, False),  # type_one_side
    (20,),  # attn
    (0, 2),  # attn_layer
    (True, False),  # attn_dotr
    ([], [[0, 1]]),  # excluded_types
    (0.0,),  # env_protection
    (True, False),  # set_davg_zero
    (1.0,),  # scaling_factor
    (True, False),  # normalize
    (None, 1.0),  # temperature
    (1e-5,),  # ln_eps
    (True,),  # concat_output_tebd
    ("float64",),  # precision
    (True, False),  # use_econf_tebd
    (False,),  # use_tebd_bias
)
class TestSeAttenV2(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            tebd_dim,
            resnet_dt,
            type_one_side,
            attn,
            attn_layer,
            attn_dotr,
            excluded_types,
            env_protection,
            set_davg_zero,
            scaling_factor,
            normalize,
            temperature,
            ln_eps,
            concat_output_tebd,
            precision,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        return {
            "sel": [10],
            "rcut_smth": 5.80,
            "rcut": 6.00,
            "neuron": [6, 12, 24],
            "ntypes": self.ntypes,
            "axis_neuron": 3,
            "tebd_dim": tebd_dim,
            "attn": attn,
            "attn_layer": attn_layer,
            "attn_dotr": attn_dotr,
            "attn_mask": False,
            "scaling_factor": scaling_factor,
            "normalize": normalize,
            "temperature": temperature,
            "ln_eps": ln_eps,
            "concat_output_tebd": concat_output_tebd,
            "resnet_dt": resnet_dt,
            "type_one_side": type_one_side,
            "exclude_types": excluded_types,
            "env_protection": env_protection,
            "precision": precision,
            "set_davg_zero": set_davg_zero,
            "use_econf_tebd": use_econf_tebd,
            "use_tebd_bias": use_tebd_bias,
            "type_map": ["O", "H"] if use_econf_tebd else None,
            "seed": 1145141919810,
        }

    def is_meaningless_zero_attention_layer_tests(
        self,
        attn_layer: int,
        attn_dotr: bool,
        normalize: bool,
        temperature: Optional[float],
    ) -> bool:
        return attn_layer == 0 and (attn_dotr or normalize or temperature is not None)

    @property
    def skip_pt(self) -> bool:
        (
            tebd_dim,
            resnet_dt,
            type_one_side,
            attn,
            attn_layer,
            attn_dotr,
            excluded_types,
            env_protection,
            set_davg_zero,
            scaling_factor,
            normalize,
            temperature,
            ln_eps,
            concat_output_tebd,
            precision,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        return CommonTest.skip_pt or self.is_meaningless_zero_attention_layer_tests(
            attn_layer,
            attn_dotr,
            normalize,
            temperature,
        )

    @property
    def skip_dp(self) -> bool:
        (
            tebd_dim,
            resnet_dt,
            type_one_side,
            attn,
            attn_layer,
            attn_dotr,
            excluded_types,
            env_protection,
            set_davg_zero,
            scaling_factor,
            normalize,
            temperature,
            ln_eps,
            concat_output_tebd,
            precision,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        return CommonTest.skip_dp or self.is_meaningless_zero_attention_layer_tests(
            attn_layer,
            attn_dotr,
            normalize,
            temperature,
        )

    @property
    def skip_tf(self) -> bool:
        return True

    tf_class = DescrptSeAttenV2TF
    dp_class = DescrptSeAttenV2DP
    pt_class = DescrptSeAttenV2PT
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
            tebd_dim,
            resnet_dt,
            type_one_side,
            attn,
            attn_layer,
            attn_dotr,
            excluded_types,
            env_protection,
            set_davg_zero,
            scaling_factor,
            normalize,
            temperature,
            ln_eps,
            concat_output_tebd,
            precision,
            use_econf_tebd,
            use_tebd_bias,
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
            tebd_dim,
            resnet_dt,
            type_one_side,
            attn,
            attn_layer,
            attn_dotr,
            excluded_types,
            env_protection,
            set_davg_zero,
            scaling_factor,
            normalize,
            temperature,
            ln_eps,
            concat_output_tebd,
            precision,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
