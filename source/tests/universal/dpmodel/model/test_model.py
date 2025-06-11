# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

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
from deepmd.dpmodel.fitting import (
    EnergyFittingNet,
)
from deepmd.dpmodel.model import (
    EnergyModel,
    SpinModel,
)
from deepmd.utils.spin import (
    Spin,
)

from ....consistent.common import (
    parameterized,
)
from ....utils import (
    CI,
    TEST_DEVICE,
)
from ...common.cases.model.model import (
    EnerModelTest,
    SpinEnerModelTest,
)
from ..backend import (
    DPTestCase,
)
from ..descriptor.test_descriptor import (
    DescriptorParamDPA1,
    DescriptorParamDPA1List,
    DescriptorParamDPA2,
    DescriptorParamDPA2List,
    DescriptorParamDPA3,
    DescriptorParamDPA3List,
    DescriptorParamHybrid,
    DescriptorParamHybridMixed,
    DescriptorParamHybridMixedTTebd,
    DescriptorParamSeA,
    DescriptorParamSeAList,
    DescriptorParamSeR,
    DescriptorParamSeRList,
    DescriptorParamSeT,
    DescriptorParamSeTList,
    DescriptorParamSeTTebd,
    DescriptorParamSeTTebdList,
)
from ..fitting.test_fitting import (
    FittingParamEnergy,
    FittingParamEnergyList,
)


def skip_model_tests(test_obj):
    if not test_obj.input_dict_ds.get(
        "smooth_type_embedding", True
    ) or not test_obj.input_dict_ds.get("smooth", True):
        test_obj.skip_test_smooth = True
        test_obj.skip_test_autodiff = True
    if hasattr(test_obj, "spin_dict") and test_obj.input_dict_ds.get(
        "use_econf_tebd", False
    ):
        return True, "Spin model do not support electronic configuration type embedding"
    if (
        "attn_layer" in test_obj.input_dict_ds
        and test_obj.input_dict_ds["attn_layer"] == 0
        and (
            not test_obj.input_dict_ds["attn_dotr"]
            or not test_obj.input_dict_ds["normalize"]
            or test_obj.input_dict_ds["temperature"] is not None
        )
    ):
        return True, "Meaningless for zero attention test in DPA1."
    return False, None


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
            *[(param_func, DescrptSeR) for param_func in DescriptorParamSeRList],
            *[(param_func, DescrptSeT) for param_func in DescriptorParamSeTList],
            *[
                (param_func, DescrptSeTTebd)
                for param_func in DescriptorParamSeTTebdList
            ],
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            *[(param_func, DescrptDPA3) for param_func in DescriptorParamDPA3List],
            (DescriptorParamHybrid, DescrptHybrid),
            (DescriptorParamHybridMixed, DescrptHybrid),
            (DescriptorParamHybridMixedTTebd, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamEnergy, EnergyFittingNet),),  # fitting_class_param & class
    ),
    fit_parameterized=(
        (
            (DescriptorParamSeA, DescrptSeA),
            (DescriptorParamSeR, DescrptSeR),
            (DescriptorParamSeT, DescrptSeT),
            (DescriptorParamSeTTebd, DescrptSeTTebd),
            (DescriptorParamDPA1, DescrptDPA1),
            (DescriptorParamDPA2, DescrptDPA2),
            (DescriptorParamDPA3, DescrptDPA3),
        ),  # descrpt_class_param & class
        (
            *[(param_func, EnergyFittingNet) for param_func in FittingParamEnergyList],
        ),  # fitting_class_param & class
    ),
)
@unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
class TestEnergyModelDP(unittest.TestCase, EnerModelTest, DPTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        EnerModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
        # set special precision
        if Descrpt in [DescrptDPA2]:
            cls.epsilon_dict["test_smooth"] = 1e-8
            cls.rprec_dict["test_smooth"] = 5e-5
            cls.aprec_dict["test_smooth"] = 5e-5
        if Descrpt in [DescrptDPA1]:
            cls.epsilon_dict["test_smooth"] = 1e-6
        if Descrpt in [DescrptSeT, DescrptSeTTebd]:
            # computational expensive
            cls.expected_sel = [i // 2 for i in cls.expected_sel]
        cls.input_dict_ds = DescriptorParam(
            len(cls.expected_type_map),
            cls.expected_rcut,
            cls.expected_rcut / 2,
            cls.expected_sel,
            cls.expected_type_map,
        )

        # set skip tests
        skiptest, skip_reason = skip_model_tests(cls)
        if skiptest:
            raise cls.skipTest(cls, skip_reason)

        ds = Descrpt(**cls.input_dict_ds)

        cls.input_dict_ft = FittingParam(
            ntypes=len(cls.expected_type_map),
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            type_map=cls.expected_type_map,
        )
        ft = Fitting(
            **cls.input_dict_ft,
        )
        cls.module = EnergyModel(
            ds,
            ft,
            type_map=cls.expected_type_map,
        )
        cls.output_def = cls.module.model_output_def().get_data()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()
        cls.skip_test_autodiff = True


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
            *[(param_func, DescrptSeR) for param_func in DescriptorParamSeRList],
            *[(param_func, DescrptSeT) for param_func in DescriptorParamSeTList],
            *[
                (param_func, DescrptSeTTebd)
                for param_func in DescriptorParamSeTTebdList
            ],
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            *[(param_func, DescrptDPA3) for param_func in DescriptorParamDPA3List],
            # (DescriptorParamHybrid, DescrptHybrid),
            # unsupported for SpinModel to hybrid both mixed_types and no-mixed_types descriptor
            (DescriptorParamHybridMixed, DescrptHybrid),
            (DescriptorParamHybridMixedTTebd, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamEnergy, EnergyFittingNet),),  # fitting_class_param & class
    ),
    fit_parameterized=(
        (
            (DescriptorParamSeA, DescrptSeA),
            (DescriptorParamSeR, DescrptSeR),
            (DescriptorParamSeT, DescrptSeT),
            (DescriptorParamSeTTebd, DescrptSeTTebd),
            (DescriptorParamDPA1, DescrptDPA1),
            (DescriptorParamDPA2, DescrptDPA2),
            (DescriptorParamDPA3, DescrptDPA3),
        ),  # descrpt_class_param & class
        (
            *[(param_func, EnergyFittingNet) for param_func in FittingParamEnergyList],
        ),  # fitting_class_param & class
    ),
)
@unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
class TestSpinEnergyModelDP(unittest.TestCase, SpinEnerModelTest, DPTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        SpinEnerModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
        cls.epsilon_dict["test_smooth"] = 1e-6
        cls.aprec_dict["test_smooth"] = 5e-5
        cls.rprec_dict["test_smooth"] = 5e-5
        # set special precision
        if Descrpt in [DescrptDPA2, DescrptHybrid]:
            cls.epsilon_dict["test_smooth"] = 1e-8
        if Descrpt in [DescrptSeT, DescrptSeTTebd]:
            # computational expensive
            cls.expected_sel = [i // 2 for i in cls.expected_sel]

        spin = Spin(
            use_spin=cls.spin_dict["use_spin"],
            virtual_scale=cls.spin_dict["virtual_scale"],
        )
        spin_type_map = cls.expected_type_map + [
            item + "_spin" for item in cls.expected_type_map
        ]
        if Descrpt in [DescrptSeA, DescrptSeR, DescrptSeT]:
            spin_sel = cls.expected_sel + cls.expected_sel
        else:
            cls.expected_sel = [i * 2 for i in cls.expected_sel]
            spin_sel = cls.expected_sel
        pair_exclude_types = spin.get_pair_exclude_types()
        atom_exclude_types = spin.get_atom_exclude_types()
        cls.input_dict_ds = DescriptorParam(
            len(spin_type_map),
            cls.expected_rcut,
            cls.expected_rcut / 2,
            spin_sel,
            spin_type_map,
            env_protection=1e-6,
            exclude_types=pair_exclude_types,
        )

        # set skip tests
        skiptest, skip_reason = skip_model_tests(cls)
        if skiptest:
            raise cls.skipTest(cls, skip_reason)

        ds = Descrpt(**cls.input_dict_ds)
        cls.input_dict_ft = FittingParam(
            ntypes=len(spin_type_map),
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            type_map=spin_type_map,
        )
        ft = Fitting(
            **cls.input_dict_ft,
        )
        backbone_model = EnergyModel(
            ds,
            ft,
            type_map=spin_type_map,
            atom_exclude_types=atom_exclude_types,
            pair_exclude_types=pair_exclude_types,
        )
        cls.module = SpinModel(backbone_model=backbone_model, spin=spin)
        cls.output_def = cls.module.model_output_def().get_data()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()
        cls.skip_test_autodiff = True
