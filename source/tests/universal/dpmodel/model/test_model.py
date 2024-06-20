# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.dpmodel.descriptor import (
    DescrptDPA1,
    DescrptDPA2,
    DescrptHybrid,
    DescrptSeA,
    DescrptSeR,
    DescrptSeT,
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
from ...common.cases.model.model import (
    EnerModelTest,
    SpinEnerModelTest,
)
from ..backend import (
    DPTestCase,
)
from ..descriptor.test_descriptor import (
    DescriptorParamDPA1List,
    DescriptorParamDPA2List,
    DescriptorParamHybrid,
    DescriptorParamHybridMixed,
    DescriptorParamSeAList,
    DescriptorParamSeRList,
    DescriptorParamSeTList,
)
from ..fitting.test_fitting import (
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
            test_obj.input_dict_ds["attn_dotr"]
            or test_obj.input_dict_ds["normalize"]
            or test_obj.input_dict_ds["temperature"] is not None
        )
    ):
        return True, "Meaningless for zero attention test in DPA1."
    return False, None


@parameterized(
    (
        *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
        *[(param_func, DescrptSeR) for param_func in DescriptorParamSeRList],
        *[(param_func, DescrptSeT) for param_func in DescriptorParamSeTList],
        *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
        *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
        (DescriptorParamHybrid, DescrptHybrid),
        (DescriptorParamHybridMixed, DescrptHybrid),
    ),  # descrpt_class_param & class
    (
        *[(param_func, EnergyFittingNet) for param_func in FittingParamEnergyList],
    ),  # fitting_class_param & class
)
class TestEnergyModelDP(unittest.TestCase, EnerModelTest, DPTestCase):
    def setUp(self):
        EnerModelTest.setUp(self)
        (DescriptorParam, Descrpt) = self.param[0]
        (FittingParam, Fitting) = self.param[1]
        # set special precision
        if Descrpt in [DescrptDPA2]:
            self.epsilon_dict["test_smooth"] = 1e-8
        self.input_dict_ds = DescriptorParam(
            len(self.expected_type_map),
            self.expected_rcut,
            self.expected_rcut / 2,
            self.expected_sel,
            self.expected_type_map,
        )

        # set skip tests
        skiptest, skip_reason = skip_model_tests(self)
        if skiptest:
            raise self.skipTest(skip_reason)

        ds = Descrpt(**self.input_dict_ds)

        self.input_dict_ft = FittingParam(
            ntypes=len(self.expected_type_map),
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            type_map=self.expected_type_map,
        )
        ft = Fitting(
            **self.input_dict_ft,
        )
        self.module = EnergyModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.model_output_def().get_data()
        self.expected_has_message_passing = ds.has_message_passing()
        self.expected_sel_type = ft.get_sel_type()
        self.expected_dim_fparam = ft.get_dim_fparam()
        self.expected_dim_aparam = ft.get_dim_aparam()
        self.skip_test_autodiff = True


@parameterized(
    (
        *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
        *[(param_func, DescrptSeR) for param_func in DescriptorParamSeRList],
        *[(param_func, DescrptSeT) for param_func in DescriptorParamSeTList],
        *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
        *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
        # (DescriptorParamHybrid, DescrptHybrid),
        # unsupported for SpinModel to hybrid both mixed_types and no-mixed_types descriptor
        (DescriptorParamHybridMixed, DescrptHybrid),
    ),  # descrpt_class_param & class
    (
        *[(param_func, EnergyFittingNet) for param_func in FittingParamEnergyList],
    ),  # fitting_class_param & class
)
class TestSpinEnergyModelDP(unittest.TestCase, SpinEnerModelTest, DPTestCase):
    def setUp(self):
        SpinEnerModelTest.setUp(self)
        (DescriptorParam, Descrpt) = self.param[0]
        (FittingParam, Fitting) = self.param[1]
        self.epsilon_dict["test_smooth"] = 1e-6
        # set special precision
        if Descrpt in [DescrptDPA2]:
            self.epsilon_dict["test_smooth"] = 1e-8

        spin = Spin(
            use_spin=self.spin_dict["use_spin"],
            virtual_scale=self.spin_dict["virtual_scale"],
        )
        spin_type_map = self.expected_type_map + [
            item + "_spin" for item in self.expected_type_map
        ]
        if Descrpt in [DescrptSeA, DescrptSeR, DescrptSeT]:
            spin_sel = self.expected_sel + self.expected_sel
        else:
            spin_sel = self.expected_sel
        pair_exclude_types = spin.get_pair_exclude_types()
        atom_exclude_types = spin.get_atom_exclude_types()
        self.input_dict_ds = DescriptorParam(
            len(spin_type_map),
            self.expected_rcut,
            self.expected_rcut / 2,
            spin_sel,
            spin_type_map,
            env_protection=1e-6,
            exclude_types=pair_exclude_types,
        )

        # set skip tests
        skiptest, skip_reason = skip_model_tests(self)
        if skiptest:
            raise self.skipTest(skip_reason)

        ds = Descrpt(**self.input_dict_ds)
        self.input_dict_ft = FittingParam(
            ntypes=len(spin_type_map),
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            type_map=spin_type_map,
        )
        ft = Fitting(
            **self.input_dict_ft,
        )
        backbone_model = EnergyModel(
            ds,
            ft,
            type_map=spin_type_map,
            atom_exclude_types=atom_exclude_types,
            pair_exclude_types=pair_exclude_types,
        )
        self.module = SpinModel(backbone_model=backbone_model, spin=spin)
        self.output_def = self.module.model_output_def().get_data()
        self.expected_has_message_passing = ds.has_message_passing()
        self.expected_sel_type = ft.get_sel_type()
        self.expected_dim_fparam = ft.get_dim_fparam()
        self.expected_dim_aparam = ft.get_dim_aparam()
        self.skip_test_autodiff = True
