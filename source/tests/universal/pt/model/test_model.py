# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
    PairTabAtomicModel,
)
from deepmd.pt.model.descriptor import (
    DescrptDPA1,
    DescrptDPA2,
    DescrptHybrid,
    DescrptSeA,
    DescrptSeR,
    DescrptSeT,
)
from deepmd.pt.model.model import (
    DipoleModel,
    DOSModel,
    DPZBLModel,
    EnergyModel,
    PolarModel,
    SpinEnergyModel,
)
from deepmd.pt.model.task import (
    DipoleFittingNet,
    DOSFittingNet,
    EnergyFittingNet,
    PolarFittingNet,
)
from deepmd.utils.spin import (
    Spin,
)

from ....consistent.common import (
    parameterized,
)
from ...common.cases.model.model import (
    DipoleModelTest,
    DosModelTest,
    EnerModelTest,
    PolarModelTest,
    SpinEnerModelTest,
    ZBLModelTest,
)
from ...dpmodel.descriptor.test_descriptor import (
    DescriptorParamDPA1List,
    DescriptorParamDPA2List,
    DescriptorParamHybrid,
    DescriptorParamHybridMixed,
    DescriptorParamSeAList,
    DescriptorParamSeRList,
    DescriptorParamSeTList,
)
from ...dpmodel.fitting.test_fitting import (
    FittingParamDipoleList,
    FittingParamDosList,
    FittingParamEnergyList,
    FittingParamPolarList,
)
from ...dpmodel.model.test_model import (
    skip_model_tests,
)
from ..backend import (
    PTTestCase,
)


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
class TestEnergyModelPT(unittest.TestCase, EnerModelTest, PTTestCase):
    # @property
    # def modules_to_test(self):
    #     # for Model, we can test script module API
    #     modules = [
    #         *PTTestCase.modules_to_test.fget(self),
    #         self.script_module,
    #     ]
    #     return modules

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
        self.output_def = self.module.translated_output_def()
        self.expected_has_message_passing = ds.has_message_passing()
        self.expected_sel_type = ft.get_sel_type()
        self.expected_dim_fparam = ft.get_dim_fparam()
        self.expected_dim_aparam = ft.get_dim_aparam()


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
        *[(param_func, DOSFittingNet) for param_func in FittingParamDosList],
    ),  # fitting_class_param & class
)
class TestDosModelPT(unittest.TestCase, DosModelTest, PTTestCase):
    # @property
    # def modules_to_test(self):
    #     # for Model, we can test script module API
    #     modules = [
    #         *PTTestCase.modules_to_test.fget(self),
    #         self.script_module,
    #     ]
    #     return modules

    def setUp(self):
        DosModelTest.setUp(self)
        (DescriptorParam, Descrpt) = self.param[0]
        (FittingParam, Fitting) = self.param[1]
        # set special precision
        self.aprec_dict["test_smooth"] = 1e-4
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
        self.module = DOSModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.translated_output_def()
        self.expected_has_message_passing = ds.has_message_passing()
        self.expected_sel_type = ft.get_sel_type()
        self.expected_dim_fparam = ft.get_dim_fparam()
        self.expected_dim_aparam = ft.get_dim_aparam()


@parameterized(
    (
        *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
        *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
        *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
        (DescriptorParamHybrid, DescrptHybrid),
        (DescriptorParamHybridMixed, DescrptHybrid),
    ),  # descrpt_class_param & class
    (
        *[(param_func, DipoleFittingNet) for param_func in FittingParamDipoleList],
    ),  # fitting_class_param & class
)
class TestDipoleModelPT(unittest.TestCase, DipoleModelTest, PTTestCase):
    # @property
    # def modules_to_test(self):
    #     # for Model, we can test script module API
    #     modules = [
    #         *PTTestCase.modules_to_test.fget(self),
    #         self.script_module,
    #     ]
    #     return modules

    def setUp(self):
        DipoleModelTest.setUp(self)
        (DescriptorParam, Descrpt) = self.param[0]
        (FittingParam, Fitting) = self.param[1]
        # set special precision
        if Descrpt in [DescrptDPA2]:
            self.epsilon_dict["test_smooth"] = 1e-8
        self.aprec_dict["test_forward"] = 1e-10  # for dipole force when near zero
        self.aprec_dict["test_rot"] = 1e-10  # for dipole force when near zero
        self.aprec_dict["test_trans"] = 1e-10  # for dipole force when near zero
        self.aprec_dict["test_permutation"] = 1e-10  # for dipole force when near zero
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
            embedding_width=ds.get_dim_emb(),
        )
        ft = Fitting(
            **self.input_dict_ft,
        )
        self.module = DipoleModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.translated_output_def()
        self.expected_has_message_passing = ds.has_message_passing()
        self.expected_sel_type = ft.get_sel_type()
        self.expected_dim_fparam = ft.get_dim_fparam()
        self.expected_dim_aparam = ft.get_dim_aparam()


@parameterized(
    (
        *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
        *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
        *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
        (DescriptorParamHybrid, DescrptHybrid),
        (DescriptorParamHybridMixed, DescrptHybrid),
    ),  # descrpt_class_param & class
    (
        *[(param_func, PolarFittingNet) for param_func in FittingParamPolarList],
    ),  # fitting_class_param & class
)
class TestPolarModelPT(unittest.TestCase, PolarModelTest, PTTestCase):
    # @property
    # def modules_to_test(self):
    #     # for Model, we can test script module API
    #     modules = [
    #         *PTTestCase.modules_to_test.fget(self),
    #         self.script_module,
    #     ]
    #     return modules

    def setUp(self):
        PolarModelTest.setUp(self)
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
            embedding_width=ds.get_dim_emb(),
        )
        ft = Fitting(
            **self.input_dict_ft,
        )
        self.module = PolarModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.translated_output_def()
        self.expected_has_message_passing = ds.has_message_passing()
        self.expected_sel_type = ft.get_sel_type()
        self.expected_dim_fparam = ft.get_dim_fparam()
        self.expected_dim_aparam = ft.get_dim_aparam()


@parameterized(
    (
        *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
        *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
        (DescriptorParamHybridMixed, DescrptHybrid),
    ),  # descrpt_class_param & class
    (
        *[(param_func, EnergyFittingNet) for param_func in FittingParamEnergyList],
    ),  # fitting_class_param & class
)
class TestZBLModelPT(unittest.TestCase, ZBLModelTest, PTTestCase):
    def setUp(self):
        ZBLModelTest.setUp(self)
        (DescriptorParam, Descrpt) = self.param[0]
        (FittingParam, Fitting) = self.param[1]
        # set special precision
        # zbl weights not so smooth
        self.aprec_dict["test_smooth"] = 5e-2
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
        dp_model = DPAtomicModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        pt_model = PairTabAtomicModel(
            self.tab_file["use_srtab"],
            self.expected_rcut,
            self.expected_sel,
            type_map=self.expected_type_map,
        )
        self.module = DPZBLModel(
            dp_model,
            pt_model,
            sw_rmin=self.tab_file["sw_rmin"],
            sw_rmax=self.tab_file["sw_rmax"],
            smin_alpha=self.tab_file["smin_alpha"],
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.translated_output_def()
        self.expected_has_message_passing = ds.has_message_passing()
        self.expected_dim_fparam = ft.get_dim_fparam()
        self.expected_dim_aparam = ft.get_dim_aparam()


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
class TestSpinEnergyModelDP(unittest.TestCase, SpinEnerModelTest, PTTestCase):
    def setUp(self):
        SpinEnerModelTest.setUp(self)
        (DescriptorParam, Descrpt) = self.param[0]
        (FittingParam, Fitting) = self.param[1]
        self.epsilon_dict["test_smooth"] = 1e-6
        # set special precision
        if Descrpt in [DescrptDPA2, DescrptHybrid]:
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
        self.module = SpinEnergyModel(backbone_model=backbone_model, spin=spin)
        self.output_def = self.module.translated_output_def()
        self.expected_has_message_passing = ds.has_message_passing()
        self.expected_sel_type = ft.get_sel_type()
        self.expected_dim_fparam = ft.get_dim_fparam()
        self.expected_dim_aparam = ft.get_dim_aparam()
