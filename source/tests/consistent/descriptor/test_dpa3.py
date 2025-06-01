# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np
from dargs import (
    Argument,
)

from deepmd.dpmodel.descriptor.dpa3 import DescrptDPA3 as DescrptDPA3DP
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
    from deepmd.pt.model.descriptor.dpa3 import DescrptDPA3 as DescrptDPA3PT
else:
    DescrptDPA3PT = None

if INSTALLED_JAX:
    from deepmd.jax.descriptor.dpa3 import DescrptDPA3 as DescrptDPA3JAX
else:
    DescrptDPA3JAX = None

if INSTALLED_PD:
    from deepmd.pd.model.descriptor.dpa3 import DescrptDPA3 as DescrptDPA3PD
else:
    DescrptDPA3PD = None

if INSTALLED_ARRAY_API_STRICT:
    from ...array_api_strict.descriptor.dpa3 import DescrptDPA3 as DescrptDPA3Strict
else:
    DescrptDPA3Strict = None

# not implemented
DescrptDPA3TF = None

from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)
from deepmd.utils.argcheck import (
    descrpt_dpa3_args,
)


@parameterized(
    ("const",),  # update_residual_init
    ([], [[0, 1]]),  # exclude_types
    (True,),  # update_angle
    (0, 1),  # a_compress_rate
    (1, 2),  # a_compress_e_rate
    (True,),  # a_compress_use_split
    (True, False),  # optim_update
    (True, False),  # edge_init_use_dist
    (True, False),  # use_exp_switch
    (True, False),  # use_dynamic_sel
    (True, False),  # use_loc_mapping
    (0.3, 0.0),  # fix_stat_std
    (1,),  # n_multi_edge_message
    ("float64",),  # precision
)
class TestDPA3(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            update_residual_init,
            exclude_types,
            update_angle,
            a_compress_rate,
            a_compress_e_rate,
            a_compress_use_split,
            optim_update,
            edge_init_use_dist,
            use_exp_switch,
            use_dynamic_sel,
            use_loc_mapping,
            fix_stat_std,
            n_multi_edge_message,
            precision,
        ) = self.param
        return {
            "ntypes": self.ntypes,
            # kwargs for repinit
            "repflow": RepFlowArgs(
                **{
                    "n_dim": 20,
                    "e_dim": 10,
                    "a_dim": 8,
                    "nlayers": 3,
                    "e_rcut": 6.0,
                    "e_rcut_smth": 5.0,
                    "e_sel": 10,
                    "a_rcut": 4.0,
                    "a_rcut_smth": 3.5,
                    "a_sel": 8,
                    "a_compress_rate": a_compress_rate,
                    "a_compress_e_rate": a_compress_e_rate,
                    "a_compress_use_split": a_compress_use_split,
                    "optim_update": optim_update,
                    "edge_init_use_dist": edge_init_use_dist,
                    "use_exp_switch": use_exp_switch,
                    "use_dynamic_sel": use_dynamic_sel,
                    "smooth_edge_update": True,
                    "fix_stat_std": fix_stat_std,
                    "n_multi_edge_message": n_multi_edge_message,
                    "axis_neuron": 4,
                    "update_angle": update_angle,
                    "update_style": "res_residual",
                    "update_residual": 0.1,
                    "update_residual_init": update_residual_init,
                }
            ),
            # kwargs for descriptor
            "activation_function": "silu",
            "precision": precision,
            "exclude_types": exclude_types,
            "env_protection": 0.0,
            "use_loc_mapping": use_loc_mapping,
            "trainable": True,
        }

    @property
    def skip_pt(self) -> bool:
        (
            update_residual_init,
            exclude_types,
            update_angle,
            a_compress_rate,
            a_compress_e_rate,
            a_compress_use_split,
            optim_update,
            edge_init_use_dist,
            use_exp_switch,
            use_dynamic_sel,
            use_loc_mapping,
            fix_stat_std,
            n_multi_edge_message,
            precision,
        ) = self.param
        return CommonTest.skip_pt

    @property
    def skip_pd(self) -> bool:
        (
            update_residual_init,
            exclude_types,
            update_angle,
            a_compress_rate,
            a_compress_e_rate,
            a_compress_use_split,
            optim_update,
            edge_init_use_dist,
            use_exp_switch,
            use_dynamic_sel,
            use_loc_mapping,
            fix_stat_std,
            n_multi_edge_message,
            precision,
        ) = self.param
        return (
            not INSTALLED_PD
            or precision == "bfloat16"
            or edge_init_use_dist
            or use_exp_switch
            or use_dynamic_sel
            or use_loc_mapping
        )  # not supported yet

    @property
    def skip_dp(self) -> bool:
        (
            update_residual_init,
            exclude_types,
            update_angle,
            a_compress_rate,
            a_compress_e_rate,
            a_compress_use_split,
            optim_update,
            edge_init_use_dist,
            use_exp_switch,
            use_dynamic_sel,
            use_loc_mapping,
            fix_stat_std,
            n_multi_edge_message,
            precision,
        ) = self.param
        return CommonTest.skip_dp

    @property
    def skip_tf(self) -> bool:
        (
            update_residual_init,
            exclude_types,
            update_angle,
            a_compress_rate,
            a_compress_e_rate,
            a_compress_use_split,
            optim_update,
            edge_init_use_dist,
            use_exp_switch,
            use_dynamic_sel,
            use_loc_mapping,
            fix_stat_std,
            n_multi_edge_message,
            precision,
        ) = self.param
        return True

    skip_jax = not INSTALLED_JAX
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT

    tf_class = DescrptDPA3TF
    dp_class = DescrptDPA3DP
    pt_class = DescrptDPA3PT
    pd_class = DescrptDPA3PD
    jax_class = DescrptDPA3JAX
    array_api_strict_class = DescrptDPA3Strict
    args = descrpt_dpa3_args().append(Argument("ntypes", int, optional=False))

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
            update_residual_init,
            exclude_types,
            update_angle,
            a_compress_rate,
            a_compress_e_rate,
            a_compress_use_split,
            optim_update,
            edge_init_use_dist,
            use_exp_switch,
            use_dynamic_sel,
            use_loc_mapping,
            fix_stat_std,
            n_multi_edge_message,
            precision,
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

    def eval_pd(self, pd_obj: Any) -> Any:
        return self.eval_pd_descriptor(
            pd_obj,
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
            update_residual_init,
            exclude_types,
            update_angle,
            a_compress_rate,
            a_compress_e_rate,
            a_compress_use_split,
            optim_update,
            edge_init_use_dist,
            use_exp_switch,
            use_dynamic_sel,
            use_loc_mapping,
            fix_stat_std,
            n_multi_edge_message,
            precision,
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
            update_residual_init,
            exclude_types,
            update_angle,
            a_compress_rate,
            a_compress_e_rate,
            a_compress_use_split,
            optim_update,
            edge_init_use_dist,
            use_exp_switch,
            use_dynamic_sel,
            use_loc_mapping,
            fix_stat_std,
            n_multi_edge_message,
            precision,
        ) = self.param
        if precision == "float64":
            return 1e-6  # need to fix in the future, see issue https://github.com/deepmodeling/deepmd-kit/issues/3786
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
