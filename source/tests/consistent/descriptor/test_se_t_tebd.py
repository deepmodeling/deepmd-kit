# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np
from dargs import (
    Argument,
)

from deepmd.dpmodel.descriptor.se_t_tebd import DescrptSeTTebd as DescrptSeTTebdDP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PD,
    INSTALLED_PT,
    CommonTest,
    parameterized,
)
from .common import (
    DescriptorTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.descriptor.se_t_tebd import DescrptSeTTebd as DescrptSeTTebdPT
else:
    DescrptSeTTebdPT = None
DescrptSeTTebdTF = None
if INSTALLED_JAX:
    from deepmd.jax.descriptor.se_t_tebd import DescrptSeTTebd as DescrptSeTTebdJAX
else:
    DescrptSeTTebdJAX = None
if INSTALLED_PD:
    from deepmd.pd.model.descriptor.se_t_tebd import DescrptSeTTebd as DescrptSeTTebdPD
else:
    DescrptSeTTebdPD = None
if INSTALLED_ARRAY_API_STRICT:
    from ...array_api_strict.descriptor.se_t_tebd import (
        DescrptSeTTebd as DescrptSeTTebdStrict,
    )
else:
    DescrptSeTTebdStrict = None
from deepmd.utils.argcheck import (
    descrpt_se_e3_tebd_args,
)


@parameterized(
    (4,),  # tebd_dim
    ("strip",),  # tebd_input_mode
    (True,),  # resnet_dt
    ([], [[0, 1]]),  # excluded_types
    (0.0,),  # env_protection
    (True, False),  # set_davg_zero
    (True, False),  # smooth
    (True,),  # concat_output_tebd
    ("float64",),  # precision
    (True, False),  # use_econf_tebd
    (False, True),  # use_tebd_bias
)
class TestSeTTebd(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            tebd_dim,
            tebd_input_mode,
            resnet_dt,
            excluded_types,
            env_protection,
            set_davg_zero,
            smooth,
            concat_output_tebd,
            precision,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        return {
            "sel": [10],
            "rcut_smth": 3.50,
            "rcut": 4.00,
            "neuron": [2, 4, 8],
            "ntypes": self.ntypes,
            "tebd_dim": tebd_dim,
            "tebd_input_mode": tebd_input_mode,
            "concat_output_tebd": concat_output_tebd,
            "resnet_dt": resnet_dt,
            "exclude_types": excluded_types,
            "env_protection": env_protection,
            "precision": precision,
            "set_davg_zero": set_davg_zero,
            "smooth": smooth,
            "use_econf_tebd": use_econf_tebd,
            "use_tebd_bias": use_tebd_bias,
            "type_map": ["O", "H"] if use_econf_tebd else None,
            "seed": 1145141919810,
        }

    @property
    def skip_pt(self) -> bool:
        (
            tebd_dim,
            tebd_input_mode,
            resnet_dt,
            excluded_types,
            env_protection,
            set_davg_zero,
            smooth,
            concat_output_tebd,
            precision,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        return CommonTest.skip_pt

    @property
    def skip_dp(self) -> bool:
        (
            tebd_dim,
            tebd_input_mode,
            resnet_dt,
            excluded_types,
            env_protection,
            set_davg_zero,
            smooth,
            concat_output_tebd,
            precision,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        return CommonTest.skip_dp

    @property
    def skip_tf(self) -> bool:
        (
            tebd_dim,
            tebd_input_mode,
            resnet_dt,
            excluded_types,
            env_protection,
            set_davg_zero,
            smooth,
            concat_output_tebd,
            precision,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        return True

    skip_pd = not INSTALLED_PD
    skip_jax = not INSTALLED_JAX
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT

    tf_class = DescrptSeTTebdTF
    dp_class = DescrptSeTTebdDP
    pt_class = DescrptSeTTebdPT
    pd_class = DescrptSeTTebdPD
    jax_class = DescrptSeTTebdJAX
    array_api_strict_class = DescrptSeTTebdStrict
    args = descrpt_se_e3_tebd_args().append(Argument("ntypes", int, optional=False))

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
        (
            tebd_dim,
            tebd_input_mode,
            resnet_dt,
            excluded_types,
            env_protection,
            set_davg_zero,
            smooth,
            concat_output_tebd,
            precision,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param

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

    def eval_jax(self, jax_obj: Any) -> Any:
        return self.eval_jax_descriptor(
            jax_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            mixed_types=True,
        )

    def eval_pd(self, pd_obj: Any) -> Any:
        return self.eval_pd_descriptor(
            pd_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            mixed_types=True,
        )

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        return self.eval_array_api_strict_descriptor(
            array_api_strict_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            mixed_types=True,
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        return (ret[0],)

    @property
    def rtol(self) -> float:
        """Relative tolerance for comparing the return value."""
        (
            tebd_dim,
            tebd_input_mode,
            resnet_dt,
            excluded_types,
            env_protection,
            set_davg_zero,
            smooth,
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
            tebd_input_mode,
            resnet_dt,
            excluded_types,
            env_protection,
            set_davg_zero,
            smooth,
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
