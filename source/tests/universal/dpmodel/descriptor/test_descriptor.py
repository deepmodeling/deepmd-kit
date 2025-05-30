# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from collections import (
    OrderedDict,
)

from deepmd.dpmodel.descriptor import (
    DescrptDPA1,
    DescrptDPA2,
    DescrptDPA3,
    DescrptHybrid,
    DescrptSeA,
    DescrptSeR,
    DescrptSeT,
    DescrptSeTTebd,
)
from deepmd.dpmodel.descriptor.dpa2 import (
    RepformerArgs,
    RepinitArgs,
)
from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)

from ....consistent.common import (
    parameterize_func,
    parameterized,
)
from ....seed import (
    GLOBAL_SEED,
)
from ....utils import (
    CI,
    TEST_DEVICE,
)
from ...common.cases.descriptor.descriptor import (
    DescriptorTest,
)
from ..backend import (
    DPTestCase,
)


def DescriptorParamSeA(
    ntypes,
    rcut,
    rcut_smth,
    sel,
    type_map,
    env_protection=0.0,
    exclude_types=[],
    resnet_dt=False,
    type_one_side=True,
    precision="float64",
):
    input_dict = {
        "ntypes": ntypes,
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "sel": sel,
        "type_map": type_map,
        "seed": GLOBAL_SEED,
        "env_protection": env_protection,
        "resnet_dt": resnet_dt,
        "type_one_side": type_one_side,
        "exclude_types": exclude_types,
        "precision": precision,
    }
    return input_dict


DescriptorParamSeAList = parameterize_func(
    DescriptorParamSeA,
    OrderedDict(
        {
            "resnet_dt": (False, True),
            "type_one_side": (True, False),
            "exclude_types": ([], [[0, 1]]),
            "env_protection": (0.0, 1e-8, 1e-2),
            "precision": ("float64",),
        }
    ),
)
# to get name for the default function
DescriptorParamSeA = DescriptorParamSeAList[0]


def DescriptorParamSeR(
    ntypes,
    rcut,
    rcut_smth,
    sel,
    type_map,
    env_protection=0.0,
    exclude_types=[],
    resnet_dt=False,
    type_one_side=True,
    precision="float64",
):
    input_dict = {
        "ntypes": ntypes,
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "sel": sel,
        "type_map": type_map,
        "seed": GLOBAL_SEED,
        "env_protection": env_protection,
        "resnet_dt": resnet_dt,
        "type_one_side": type_one_side,
        "exclude_types": exclude_types,
        "precision": precision,
    }
    return input_dict


DescriptorParamSeRList = parameterize_func(
    DescriptorParamSeR,
    OrderedDict(
        {
            "resnet_dt": (False, True),
            "type_one_side": (True,),  # type_one_side == False not implemented
            "exclude_types": ([], [[0, 1]]),
            "env_protection": (0.0, 1e-8),
            "precision": ("float64",),
        }
    ),
)
# to get name for the default function
DescriptorParamSeR = DescriptorParamSeRList[0]


def DescriptorParamSeT(
    ntypes,
    rcut,
    rcut_smth,
    sel,
    type_map,
    env_protection=0.0,
    exclude_types=[],
    resnet_dt=False,
    precision="float64",
):
    input_dict = {
        "ntypes": ntypes,
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "sel": sel,
        "type_map": type_map,
        "seed": GLOBAL_SEED,
        "env_protection": env_protection,
        "resnet_dt": resnet_dt,
        "exclude_types": exclude_types,
        "precision": precision,
    }
    return input_dict


DescriptorParamSeTList = parameterize_func(
    DescriptorParamSeT,
    OrderedDict(
        {
            "resnet_dt": (False, True),
            "exclude_types": ([], [[0, 1]]),
            "env_protection": (0.0, 1e-8),
            "precision": ("float64",),
        }
    ),
)
# to get name for the default function
DescriptorParamSeT = DescriptorParamSeTList[0]


def DescriptorParamSeTTebd(
    ntypes,
    rcut,
    rcut_smth,
    sel,
    type_map,
    env_protection=0.0,
    exclude_types=[],
    tebd_dim=4,
    tebd_input_mode="concat",
    concat_output_tebd=True,
    resnet_dt=True,
    set_davg_zero=True,
    smooth=True,
    use_econf_tebd=False,
    use_tebd_bias=False,
    precision="float64",
):
    input_dict = {
        "ntypes": ntypes,
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "sel": sel,  # use a small sel for efficiency
        "type_map": type_map,
        "seed": GLOBAL_SEED,
        "tebd_dim": tebd_dim,
        "tebd_input_mode": tebd_input_mode,
        "concat_output_tebd": concat_output_tebd,
        "resnet_dt": resnet_dt,
        "exclude_types": exclude_types,
        "env_protection": env_protection,
        "set_davg_zero": set_davg_zero,
        "smooth": smooth,
        "use_econf_tebd": use_econf_tebd,
        "use_tebd_bias": use_tebd_bias,
        "precision": precision,
    }
    return input_dict


DescriptorParamSeTTebdList = parameterize_func(
    DescriptorParamSeTTebd,
    OrderedDict(
        {
            "tebd_dim": (4,),
            "tebd_input_mode": ("concat", "strip"),
            "resnet_dt": (True,),
            "exclude_types": ([], [[0, 1]]),
            "env_protection": (0.0,),
            "set_davg_zero": (False,),
            "smooth": (True, False),
            "concat_output_tebd": (True,),
            "use_econf_tebd": (False, True),
            "use_tebd_bias": (False,),
            "precision": ("float64",),
        }
    ),
)
# to get name for the default function
DescriptorParamSeTTebd = DescriptorParamSeTTebdList[0]


def DescriptorParamDPA1(
    ntypes,
    rcut,
    rcut_smth,
    sel,
    type_map,
    env_protection=0.0,
    exclude_types=[],
    tebd_dim=4,
    tebd_input_mode="concat",
    attn=20,
    attn_layer=2,
    attn_dotr=True,
    scaling_factor=1.0,
    normalize=True,
    temperature=None,
    ln_eps=1e-5,
    concat_output_tebd=True,
    resnet_dt=True,
    type_one_side=True,
    set_davg_zero=True,
    smooth_type_embedding=True,
    use_econf_tebd=False,
    use_tebd_bias=False,
    precision="float64",
):
    input_dict = {
        "ntypes": ntypes,
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "sel": sel,
        "type_map": type_map,
        "seed": GLOBAL_SEED,
        "tebd_dim": tebd_dim,
        "tebd_input_mode": tebd_input_mode,
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
        "exclude_types": exclude_types,
        "env_protection": env_protection,
        "set_davg_zero": set_davg_zero,
        "smooth_type_embedding": smooth_type_embedding,
        "use_econf_tebd": use_econf_tebd,
        "use_tebd_bias": use_tebd_bias,
        "precision": precision,
    }
    return input_dict


DescriptorParamDPA1List = parameterize_func(
    DescriptorParamDPA1,
    OrderedDict(
        {
            "tebd_dim": (4,),
            "tebd_input_mode": ("concat", "strip"),
            "resnet_dt": (True,),
            "type_one_side": (False,),
            "attn": (20,),
            "attn_layer": (0, 2),
            "attn_dotr": (True,),
            "exclude_types": ([], [[0, 1]]),
            "env_protection": (0.0,),
            "set_davg_zero": (False,),
            "scaling_factor": (1.0,),
            "normalize": (True,),
            "temperature": (None, 1.0),
            "ln_eps": (1e-5,),
            "smooth_type_embedding": (True, False),
            "concat_output_tebd": (True,),
            "use_econf_tebd": (False, True),
            "use_tebd_bias": (False,),
            "precision": ("float64",),
        }
    ),
)
# to get name for the default function
DescriptorParamDPA1 = DescriptorParamDPA1List[0]


def DescriptorParamDPA2(
    ntypes,
    rcut,
    rcut_smth,
    sel,
    type_map,
    repinit_tebd_input_mode="concat",
    repinit_set_davg_zero=False,
    repinit_type_one_side=False,
    repinit_use_three_body=False,
    repformer_direct_dist=False,
    repformer_update_g1_has_conv=True,
    repformer_update_g1_has_drrd=True,
    repformer_update_g1_has_grrg=True,
    repformer_update_g1_has_attn=True,
    repformer_update_g2_has_g1g1=True,
    repformer_update_g2_has_attn=True,
    repformer_update_h2=False,
    repformer_attn2_has_gate=True,
    repformer_update_style="res_avg",
    repformer_update_residual_init="norm",
    repformer_set_davg_zero=False,
    repformer_trainable_ln=True,
    repformer_ln_eps=1e-5,
    repformer_use_sqrt_nnei=False,
    repformer_g1_out_conv=False,
    repformer_g1_out_mlp=False,
    smooth=True,
    add_tebd_to_repinit_out=True,
    use_econf_tebd=False,
    use_tebd_bias=False,
    env_protection=0.0,
    exclude_types=[],
    precision="float64",
):
    input_dict = {
        "ntypes": ntypes,
        # kwargs for repinit
        "repinit": RepinitArgs(
            **{
                "rcut": rcut,
                "rcut_smth": rcut_smth,
                "nsel": sum(sel),
                "neuron": [6, 12, 24],
                "axis_neuron": 3,
                "tebd_dim": 4,
                "tebd_input_mode": repinit_tebd_input_mode,
                "set_davg_zero": repinit_set_davg_zero,
                "activation_function": "tanh",
                "type_one_side": repinit_type_one_side,
                "use_three_body": repinit_use_three_body,
                "three_body_sel": min(sum(sel) // 2, 10),
                "three_body_rcut": rcut / 2,
                "three_body_rcut_smth": rcut_smth / 2,
            }
        ),
        # kwargs for repformer
        "repformer": RepformerArgs(
            **{
                "rcut": rcut / 2,
                "rcut_smth": rcut_smth / 2,
                "nsel": sum(sel) // 2,
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
                "set_davg_zero": repformer_set_davg_zero,
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
        "env_protection": env_protection,
        "trainable": True,
        "use_econf_tebd": use_econf_tebd,
        "use_tebd_bias": use_tebd_bias,
        "type_map": type_map,
        "seed": GLOBAL_SEED,
        "add_tebd_to_repinit_out": add_tebd_to_repinit_out,
    }
    return input_dict


DescriptorParamDPA2List = parameterize_func(
    DescriptorParamDPA2,
    OrderedDict(
        {
            "repinit_tebd_input_mode": ("concat", "strip"),
            "repinit_set_davg_zero": (True,),
            "repinit_type_one_side": (False,),
            "repinit_use_three_body": (True, False),
            "repformer_direct_dist": (False,),
            "repformer_update_g1_has_conv": (True,),
            "repformer_update_g1_has_drrd": (True,),
            "repformer_update_g1_has_grrg": (True,),
            "repformer_update_g1_has_attn": (True,),
            "repformer_update_g2_has_g1g1": (True,),
            "repformer_update_g2_has_attn": (True,),
            "repformer_update_h2": (False,),
            "repformer_attn2_has_gate": (True,),
            "repformer_update_style": ("res_avg", "res_residual"),
            "repformer_update_residual_init": ("norm",),
            "repformer_set_davg_zero": (True,),
            "repformer_trainable_ln": (True,),
            "repformer_ln_eps": (1e-5,),
            "repformer_use_sqrt_nnei": (True,),
            "repformer_g1_out_conv": (True,),
            "repformer_g1_out_mlp": (True,),
            "smooth": (True, False),
            "exclude_types": ([], [[0, 1]]),
            "precision": ("float64",),
            "add_tebd_to_repinit_out": (True, False),
            "use_econf_tebd": (False,),
            "use_tebd_bias": (False,),
        }
    ),
)
# to get name for the default function
DescriptorParamDPA2 = DescriptorParamDPA2List[0]


def DescriptorParamDPA3(
    ntypes,
    rcut,
    rcut_smth,
    sel,
    type_map,
    env_protection=0.0,
    exclude_types=[],
    update_style="res_residual",
    update_residual=0.1,
    update_residual_init="const",
    update_angle=True,
    n_multi_edge_message=1,
    a_compress_rate=0,
    a_compress_e_rate=1,
    a_compress_use_split=False,
    optim_update=True,
    smooth_edge_update=False,
    edge_init_use_dist=False,
    use_exp_switch=False,
    fix_stat_std=0.3,
    use_dynamic_sel=False,
    precision="float64",
    use_loc_mapping=True,
):
    input_dict = {
        # kwargs for repformer
        "repflow": RepFlowArgs(
            **{
                "n_dim": 20,
                "e_dim": 10,
                "a_dim": 8,
                "nlayers": 2,
                "e_rcut": rcut,
                "e_rcut_smth": rcut_smth
                if not use_exp_switch
                else (rcut - 1.0),  # suitable for ut
                "e_sel": sum(sel),
                "a_rcut": rcut / 2,
                "a_rcut_smth": rcut_smth / 2
                if not use_exp_switch
                else (rcut - 1.0) / 2,  # suitable for ut
                "a_sel": sum(sel) // 4,
                "a_compress_rate": a_compress_rate,
                "a_compress_e_rate": a_compress_e_rate,
                "a_compress_use_split": a_compress_use_split,
                "optim_update": optim_update,
                "use_exp_switch": use_exp_switch,
                "smooth_edge_update": smooth_edge_update,
                "edge_init_use_dist": edge_init_use_dist,
                "fix_stat_std": fix_stat_std,
                "n_multi_edge_message": n_multi_edge_message,
                "axis_neuron": 2,
                "use_dynamic_sel": use_dynamic_sel,
                "sel_reduce_factor": 1.0,
                "update_angle": update_angle,
                "update_style": update_style,
                "update_residual": update_residual,
                "update_residual_init": update_residual_init,
            }
        ),
        "ntypes": ntypes,
        "concat_output_tebd": False,
        "precision": precision,
        "activation_function": "silu",
        "exclude_types": exclude_types,
        "env_protection": env_protection,
        "trainable": True,
        "use_econf_tebd": False,
        "use_tebd_bias": False,
        "use_loc_mapping": use_loc_mapping,
        "type_map": type_map,
        "seed": GLOBAL_SEED,
    }
    return input_dict


DescriptorParamDPA3List = parameterize_func(
    DescriptorParamDPA3,
    OrderedDict(
        {
            "update_residual_init": ("const",),
            "exclude_types": ([], [[0, 1]]),
            "update_angle": (True, False),
            "a_compress_rate": (1,),
            "a_compress_e_rate": (2,),
            "a_compress_use_split": (True,),
            "optim_update": (True, False),
            "smooth_edge_update": (True,),
            "edge_init_use_dist": (True, False),
            "use_exp_switch": (True, False),
            "fix_stat_std": (0.3,),
            "n_multi_edge_message": (1,),
            "use_dynamic_sel": (True, False),
            "env_protection": (0.0, 1e-8),
            "precision": ("float64",),
            "use_loc_mapping": (True, False),
        }
    ),
)
# to get name for the default function
DescriptorParamDPA3 = DescriptorParamDPA3List[0]


def DescriptorParamHybrid(ntypes, rcut, rcut_smth, sel, type_map, **kwargs):
    ddsub0 = {
        "type": "se_e2_a",
        **DescriptorParamSeA(ntypes, rcut, rcut_smth, sel, type_map, **kwargs),
    }
    ddsub1 = {
        "type": "dpa1",
        **DescriptorParamDPA1(ntypes, rcut, rcut_smth, sum(sel), type_map, **kwargs),
    }
    input_dict = {
        "list": [ddsub0, ddsub1],
    }
    return input_dict


def DescriptorParamHybridMixed(ntypes, rcut, rcut_smth, sel, type_map, **kwargs):
    ddsub0 = {
        "type": "dpa1",
        **DescriptorParamDPA1(ntypes, rcut, rcut_smth, sum(sel), type_map, **kwargs),
    }
    ddsub1 = {
        "type": "dpa1",
        **DescriptorParamDPA1(ntypes, rcut, rcut_smth, sum(sel), type_map, **kwargs),
    }
    input_dict = {
        "list": [ddsub0, ddsub1],
    }
    return input_dict


def DescriptorParamHybridMixedTTebd(ntypes, rcut, rcut_smth, sel, type_map, **kwargs):
    ddsub0 = {
        "type": "dpa1",
        **DescriptorParamDPA1(ntypes, rcut, rcut_smth, sum(sel), type_map, **kwargs),
    }
    ddsub1 = {
        "type": "se_e3_tebd",
        **DescriptorParamSeTTebd(
            ntypes, rcut / 2, rcut_smth / 2, min(sum(sel) // 2, 10), type_map, **kwargs
        ),
    }  # use a small sel for efficiency
    input_dict = {
        "list": [ddsub0, ddsub1],
    }
    return input_dict


@parameterized(
    (
        (DescriptorParamSeA, DescrptSeA),
        (DescriptorParamSeR, DescrptSeR),
        (DescriptorParamSeT, DescrptSeT),
        (DescriptorParamSeTTebd, DescrptSeTTebd),
        (DescriptorParamDPA1, DescrptDPA1),
        (DescriptorParamDPA2, DescrptDPA2),
        (DescriptorParamDPA3, DescrptDPA3),
        (DescriptorParamHybrid, DescrptHybrid),
        (DescriptorParamHybridMixed, DescrptHybrid),
        (DescriptorParamHybridMixedTTebd, DescrptHybrid),
    )  # class_param & class
)
@unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
class TestDescriptorDP(unittest.TestCase, DescriptorTest, DPTestCase):
    def setUp(self) -> None:
        DescriptorTest.setUp(self)
        (DescriptorParam, Descrpt) = self.param[0]
        self.module_class = Descrpt
        self.input_dict = DescriptorParam(
            self.nt, self.rcut, self.rcut_smth, self.sel, ["O", "H"]
        )
        self.module = Descrpt(**self.input_dict)
