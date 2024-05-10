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

from deepmd.dpmodel.descriptor.dpa2 import DescrptDPA2 as DescrptDPA2DP
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
    from deepmd.pt.model.descriptor.dpa2 import DescrptDPA2 as DescrptDPA2PT
else:
    DescrptDPA2PT = None

# not implemented
DescrptDPA2TF = None

from deepmd.utils.argcheck import (
    descrpt_se_atten_args,
)


@parameterized(
    ("concat", "strip"),  # repinit_tebd_input_mode
    (True,),  # repinit_set_davg_zero
    (False,),  # repinit_type_one_side
    (True, False),  # repformer_direct_dist
    (True,),  # repformer_update_g1_has_conv
    (True,),  # repformer_update_g1_has_drrd
    (True,),  # repformer_update_g1_has_grrg
    (True,),  # repformer_update_g1_has_attn
    (True,),  # repformer_update_g2_has_g1g1
    (True,),  # repformer_update_g2_has_attn
    (False,),  # repformer_update_h2
    (True, False),  # repformer_attn2_has_gate
    ("res_avg", "res_residual"),  # repformer_update_style
    ("norm", "const"),  # repformer_update_residual_init
    (True,),  # repformer_set_davg_zero
    (True,),  # repformer_trainable_ln
    (1e-5,),  # repformer_ln_eps
    (True, False),  # smooth
    ([], [[0, 1]]),  # exclude_types
    ("float64",),  # precision
    (True, False),  # add_tebd_to_repinit_out
)
class TestDPA2(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
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
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
        ) = self.param
        return {
            "ntypes": self.ntypes,
            "repinit_rcut": 6.00,
            "repinit_rcut_smth": 5.80,
            "repinit_nsel": 10,
            "repformer_rcut": 4.00,
            "repformer_rcut_smth": 3.50,
            "repformer_nsel": 8,
            # kwargs for repinit
            "repinit_neuron": [6, 12, 24],
            "repinit_axis_neuron": 3,
            "repinit_tebd_dim": 4,
            "repinit_tebd_input_mode": repinit_tebd_input_mode,
            "repinit_set_davg_zero": repinit_set_davg_zero,
            "repinit_activation_function": "tanh",
            "repinit_type_one_side": repinit_type_one_side,
            # kwargs for repformer
            "repformer_nlayers": 3,
            "repformer_g1_dim": 20,
            "repformer_g2_dim": 10,
            "repformer_axis_neuron": 3,
            "repformer_direct_dist": repformer_direct_dist,
            "repformer_update_g1_has_conv": repformer_update_g1_has_conv,
            "repformer_update_g1_has_drrd": repformer_update_g1_has_drrd,
            "repformer_update_g1_has_grrg": repformer_update_g1_has_grrg,
            "repformer_update_g1_has_attn": repformer_update_g1_has_attn,
            "repformer_update_g2_has_g1g1": repformer_update_g2_has_g1g1,
            "repformer_update_g2_has_attn": repformer_update_g2_has_attn,
            "repformer_update_h2": repformer_update_h2,
            "repformer_attn1_hidden": 12,
            "repformer_attn1_nhead": 2,
            "repformer_attn2_hidden": 10,
            "repformer_attn2_nhead": 2,
            "repformer_attn2_has_gate": repformer_attn2_has_gate,
            "repformer_activation_function": "tanh",
            "repformer_update_style": repformer_update_style,
            "repformer_update_residual": 0.001,
            "repformer_update_residual_init": repformer_update_residual_init,
            "repformer_set_davg_zero": True,
            "repformer_trainable_ln": repformer_trainable_ln,
            "repformer_ln_eps": repformer_ln_eps,
            # kwargs for descriptor
            "concat_output_tebd": True,
            "precision": precision,
            "smooth": smooth,
            "exclude_types": exclude_types,
            "env_protection": 0.0,
            "trainable": True,
            "add_tebd_to_repinit_out": add_tebd_to_repinit_out,
        }

    @property
    def skip_pt(self) -> bool:
        (
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
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
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
        ) = self.param
        return CommonTest.skip_pt

    @property
    def skip_dp(self) -> bool:
        (
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
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
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
        ) = self.param
        return CommonTest.skip_pt

    @property
    def skip_tf(self) -> bool:
        (
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
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
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
        ) = self.param
        return True

    tf_class = DescrptDPA2TF
    dp_class = DescrptDPA2DP
    pt_class = DescrptDPA2PT
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
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
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
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
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
            repinit_tebd_input_mode,
            repinit_set_davg_zero,
            repinit_type_one_side,
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
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
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
            smooth,
            exclude_types,
            precision,
            add_tebd_to_repinit_out,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
