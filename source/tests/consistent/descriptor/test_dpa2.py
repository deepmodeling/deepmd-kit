# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np
from dargs import (
    Argument,
)

from deepmd.dpmodel.descriptor.dpa2 import DescrptDPA2 as DescrptDPA2DP
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
    from deepmd.pt.model.descriptor.dpa2 import DescrptDPA2 as DescrptDPA2PT
else:
    DescrptDPA2PT = None

if INSTALLED_JAX:
    from deepmd.jax.descriptor.dpa2 import DescrptDPA2 as DescrptDPA2JAX
else:
    DescrptDPA2JAX = None

if INSTALLED_PD:
    from deepmd.pd.model.descriptor.dpa2 import DescrptDPA2 as DescrptDPA2PD
else:
    DescrptDPA2PD = None

if INSTALLED_ARRAY_API_STRICT:
    from ...array_api_strict.descriptor.dpa2 import DescrptDPA2 as DescrptDPA2Strict
else:
    DescrptDPA2Strict = None

# not implemented
DescrptDPA2TF = None

from deepmd.dpmodel.descriptor.dpa2 import (
    RepformerArgs,
    RepinitArgs,
)
from deepmd.utils.argcheck import (
    descrpt_dpa2_args,
)


@parameterized(
    ("concat", "strip"),  # repinit_tebd_input_mode
    (True,),  # repinit_set_davg_zero
    (False,),  # repinit_type_one_side
    (True, False),  # repinit_use_three_body
    (True, False),  # repformer_direct_dist
    (True,),  # repformer_update_g1_has_conv
    (True,),  # repformer_update_g1_has_drrd
    (True,),  # repformer_update_g1_has_grrg
    (True,),  # repformer_update_g1_has_attn
    (True,),  # repformer_update_g2_has_g1g1
    (True,),  # repformer_update_g2_has_attn
    (False,),  # repformer_update_h2
    (True,),  # repformer_attn2_has_gate
    ("res_avg", "res_residual"),  # repformer_update_style
    ("norm", "const"),  # repformer_update_residual_init
    (True,),  # repformer_set_davg_zero
    (True,),  # repformer_trainable_ln
    (1e-5,),  # repformer_ln_eps
    (True,),  # repformer_use_sqrt_nnei
    (True,),  # repformer_g1_out_conv
    (True,),  # repformer_g1_out_mlp
    (True, False),  # smooth
    ([], [[0, 1]]),  # exclude_types
    ("float64",),  # precision
    (True, False),  # add_tebd_to_repinit_out
    (True, False),  # use_econf_tebd
    (False,),  # use_tebd_bias
)
class TestDPA2(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
            repinit_use_three_body,
            repformer_update_g1_has_conv,
            repformer_direct_dist,
            repformer_update_g1_has_drrd,
            repformer_update_g1_has_grrg,
            repformer_update_g1_has_attn,
            repformer_update_g2_has_g1g1,
            repformer_update_g2_has_attn,
            repformer_update_h2,
            repformer_attn2_has_gate,
            repformer_update_style,
            repformer_update_residual_init,
            repformer_set_davg_zero,
            repformer_trainable_ln,
            repformer_ln_eps,
            repformer_use_sqrt_nnei,
            repformer_g1_out_conv,
            repformer_g1_out_mlp,
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        return {
            "ntypes": self.ntypes,
            # kwargs for repinit
            "repinit": RepinitArgs(
                **{
                    "rcut": 6.00,
                    "rcut_smth": 5.80,
                    "nsel": 10,
                    "neuron": [6, 12, 24],
                    "axis_neuron": 3,
                    "tebd_dim": 4,
                    "tebd_input_mode": repinit_tebd_input_mode,
                    "set_davg_zero": repinit_set_davg_zero,
                    "activation_function": "tanh",
                    "type_one_side": repinit_type_one_side,
                    "use_three_body": repinit_use_three_body,
                    "three_body_sel": 8,
                    "three_body_rcut": 4.0,
                    "three_body_rcut_smth": 3.5,
                }
            ),
            # kwargs for repformer
            "repformer": RepformerArgs(
                **{
                    "rcut": 4.00,
                    "rcut_smth": 3.50,
                    "nsel": 8,
                    "nlayers": 3,
                    "g1_dim": 20,
                    "g2_dim": 10,
                    "axis_neuron": 3,
                    "direct_dist": repformer_direct_dist,
                    "update_g1_has_conv": repformer_update_g1_has_conv,
                    "update_g1_has_drrd": repformer_update_g1_has_drrd,
                    "update_g1_has_grrg": repformer_update_g1_has_grrg,
                    "update_g1_has_attn": repformer_update_g1_has_attn,
                    "update_g2_has_g1g1": repformer_update_g2_has_g1g1,
                    "update_g2_has_attn": repformer_update_g2_has_attn,
                    "update_h2": repformer_update_h2,
                    "attn1_hidden": 12,
                    "attn1_nhead": 2,
                    "attn2_hidden": 10,
                    "attn2_nhead": 2,
                    "attn2_has_gate": repformer_attn2_has_gate,
                    "activation_function": "tanh",
                    "update_style": repformer_update_style,
                    "update_residual": 0.001,
                    "update_residual_init": repformer_update_residual_init,
                    "set_davg_zero": True,
                    "trainable_ln": repformer_trainable_ln,
                    "ln_eps": repformer_ln_eps,
                    "use_sqrt_nnei": repformer_use_sqrt_nnei,
                    "g1_out_conv": repformer_g1_out_conv,
                    "g1_out_mlp": repformer_g1_out_mlp,
                }
            ),
            # kwargs for descriptor
            "concat_output_tebd": True,
            "precision": precision,
            "smooth": smooth,
            "exclude_types": exclude_types,
            "env_protection": 0.0,
            "trainable": True,
            "use_econf_tebd": use_econf_tebd,
            "use_tebd_bias": use_tebd_bias,
            "type_map": ["O", "H"] if use_econf_tebd else None,
            "add_tebd_to_repinit_out": add_tebd_to_repinit_out,
        }

    @property
    def skip_pt(self) -> bool:
        (
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
            repinit_use_three_body,
            repformer_update_g1_has_conv,
            repformer_direct_dist,
            repformer_update_g1_has_drrd,
            repformer_update_g1_has_grrg,
            repformer_update_g1_has_attn,
            repformer_update_g2_has_g1g1,
            repformer_update_g2_has_attn,
            repformer_update_h2,
            repformer_attn2_has_gate,
            repformer_update_style,
            repformer_update_residual_init,
            repformer_set_davg_zero,
            repformer_trainable_ln,
            repformer_ln_eps,
            repformer_use_sqrt_nnei,
            repformer_g1_out_conv,
            repformer_g1_out_mlp,
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        return CommonTest.skip_pt

    @property
    def skip_pd(self) -> bool:
        (
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
            repinit_use_three_body,
            repformer_update_g1_has_conv,
            repformer_direct_dist,
            repformer_update_g1_has_drrd,
            repformer_update_g1_has_grrg,
            repformer_update_g1_has_attn,
            repformer_update_g2_has_g1g1,
            repformer_update_g2_has_attn,
            repformer_update_h2,
            repformer_attn2_has_gate,
            repformer_update_style,
            repformer_update_residual_init,
            repformer_set_davg_zero,
            repformer_trainable_ln,
            repformer_ln_eps,
            repformer_use_sqrt_nnei,
            repformer_g1_out_conv,
            repformer_g1_out_mlp,
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        return not INSTALLED_PD or precision == "bfloat16"

    @property
    def skip_dp(self) -> bool:
        (
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
            repinit_use_three_body,
            repformer_update_g1_has_conv,
            repformer_direct_dist,
            repformer_update_g1_has_drrd,
            repformer_update_g1_has_grrg,
            repformer_update_g1_has_attn,
            repformer_update_g2_has_g1g1,
            repformer_update_g2_has_attn,
            repformer_update_h2,
            repformer_attn2_has_gate,
            repformer_update_style,
            repformer_update_residual_init,
            repformer_set_davg_zero,
            repformer_trainable_ln,
            repformer_ln_eps,
            repformer_use_sqrt_nnei,
            repformer_g1_out_conv,
            repformer_g1_out_mlp,
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        return CommonTest.skip_dp

    @property
    def skip_tf(self) -> bool:
        (
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
            repinit_use_three_body,
            repformer_update_g1_has_conv,
            repformer_direct_dist,
            repformer_update_g1_has_drrd,
            repformer_update_g1_has_grrg,
            repformer_update_g1_has_attn,
            repformer_update_g2_has_g1g1,
            repformer_update_g2_has_attn,
            repformer_update_h2,
            repformer_attn2_has_gate,
            repformer_update_style,
            repformer_update_residual_init,
            repformer_set_davg_zero,
            repformer_trainable_ln,
            repformer_ln_eps,
            repformer_use_sqrt_nnei,
            repformer_g1_out_conv,
            repformer_g1_out_mlp,
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        return True

    skip_jax = not INSTALLED_JAX
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT

    tf_class = DescrptDPA2TF
    dp_class = DescrptDPA2DP
    pt_class = DescrptDPA2PT
    pd_class = DescrptDPA2PD
    jax_class = DescrptDPA2JAX
    array_api_strict_class = DescrptDPA2Strict
    args = descrpt_dpa2_args().append(Argument("ntypes", int, optional=False))

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
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
            repinit_use_three_body,
            repformer_update_g1_has_conv,
            repformer_direct_dist,
            repformer_update_g1_has_drrd,
            repformer_update_g1_has_grrg,
            repformer_update_g1_has_attn,
            repformer_update_g2_has_g1g1,
            repformer_update_g2_has_attn,
            repformer_update_h2,
            repformer_attn2_has_gate,
            repformer_update_style,
            repformer_update_residual_init,
            repformer_set_davg_zero,
            repformer_trainable_ln,
            repformer_ln_eps,
            repformer_use_sqrt_nnei,
            repformer_g1_out_conv,
            repformer_g1_out_mlp,
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
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
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
            repinit_use_three_body,
            repformer_update_g1_has_conv,
            repformer_direct_dist,
            repformer_update_g1_has_drrd,
            repformer_update_g1_has_grrg,
            repformer_update_g1_has_attn,
            repformer_update_g2_has_g1g1,
            repformer_update_g2_has_attn,
            repformer_update_h2,
            repformer_attn2_has_gate,
            repformer_update_style,
            repformer_update_residual_init,
            repformer_set_davg_zero,
            repformer_trainable_ln,
            repformer_ln_eps,
            repformer_use_sqrt_nnei,
            repformer_g1_out_conv,
            repformer_g1_out_mlp,
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
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
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
            repinit_use_three_body,
            repformer_update_g1_has_conv,
            repformer_direct_dist,
            repformer_update_g1_has_drrd,
            repformer_update_g1_has_grrg,
            repformer_update_g1_has_attn,
            repformer_update_g2_has_g1g1,
            repformer_update_g2_has_attn,
            repformer_update_h2,
            repformer_attn2_has_gate,
            repformer_update_style,
            repformer_update_residual_init,
            repformer_set_davg_zero,
            repformer_trainable_ln,
            repformer_ln_eps,
            repformer_use_sqrt_nnei,
            repformer_g1_out_conv,
            repformer_g1_out_mlp,
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
            use_econf_tebd,
            use_tebd_bias,
        ) = self.param
        if precision == "float64":
            return 1e-6  # need to fix in the future, see issue https://github.com/deepmodeling/deepmd-kit/issues/3786
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
