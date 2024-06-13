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
)
from deepmd.pt.model.task import (
    DipoleFittingNet,
    DOSFittingNet,
    EnergyFittingNet,
    PolarFittingNet,
)

from ....consistent.common import (
    parameterized,
)
from ...common.cases.model.model import (
    DipoleModelTest,
    DosModelTest,
    EnerModelTest,
    PolarModelTest,
    ZBLModelTest,
)
from ..backend import (
    PTTestCase,
)
from ..descriptor.test_descriptor import (
    DescriptorParamDPA1,
    DescriptorParamDPA2,
    DescriptorParamHybrid,
    DescriptorParamHybridMixed,
    DescriptorParamSeA,
    DescriptorParamSeR,
    DescriptorParamSeT,
)


@parameterized(
    (
        (DescriptorParamSeA, DescrptSeA),
        (DescriptorParamSeR, DescrptSeR),
        (DescriptorParamSeT, DescrptSeT),
        (DescriptorParamDPA1, DescrptDPA1),
        (DescriptorParamDPA2, DescrptDPA2),
        (DescriptorParamHybrid, DescrptHybrid),
        (DescriptorParamHybridMixed, DescrptHybrid),
    )  # class_param & class
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
        ds = Descrpt(**self.input_dict_ds)
        ft = EnergyFittingNet(
            **self.input_dict_ft,
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        )
        self.module = EnergyModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.translated_output_def()
        self.expected_has_message_passing = ds.has_message_passing()


@parameterized(
    (
        (DescriptorParamSeA, DescrptSeA),
        (DescriptorParamSeR, DescrptSeR),
        (DescriptorParamSeT, DescrptSeT),
        (DescriptorParamDPA1, DescrptDPA1),
        (DescriptorParamDPA2, DescrptDPA2),
        (DescriptorParamHybrid, DescrptHybrid),
        (DescriptorParamHybridMixed, DescrptHybrid),
    )  # class_param & class
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
        ds = Descrpt(**self.input_dict_ds)
        ft = DOSFittingNet(
            **self.input_dict_ft,
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        )
        self.module = DOSModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.translated_output_def()
        self.expected_has_message_passing = ds.has_message_passing()


@parameterized(
    (
        (DescriptorParamSeA, DescrptSeA),
        (DescriptorParamDPA1, DescrptDPA1),
        (DescriptorParamDPA2, DescrptDPA2),
        (DescriptorParamHybrid, DescrptHybrid),
        (DescriptorParamHybridMixed, DescrptHybrid),
    )  # class_param & class
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
        ds = Descrpt(**self.input_dict_ds)
        ft = DipoleFittingNet(
            **self.input_dict_ft,
            embedding_width=ds.get_dim_emb(),
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        )
        self.module = DipoleModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.translated_output_def()
        self.expected_has_message_passing = ds.has_message_passing()


@parameterized(
    (
        (DescriptorParamSeA, DescrptSeA),
        (DescriptorParamDPA1, DescrptDPA1),
        (DescriptorParamDPA2, DescrptDPA2),
        (DescriptorParamHybrid, DescrptHybrid),
        (DescriptorParamHybridMixed, DescrptHybrid),
    )  # class_param & class
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
        ds = Descrpt(**self.input_dict_ds)
        ft = PolarFittingNet(
            **self.input_dict_ft,
            embedding_width=ds.get_dim_emb(),
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        )
        self.module = PolarModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.translated_output_def()
        self.expected_has_message_passing = ds.has_message_passing()


@parameterized(
    (
        (DescriptorParamDPA1, DescrptDPA1),
        (DescriptorParamDPA2, DescrptDPA2),
        (DescriptorParamHybridMixed, DescrptHybrid),
    )  # class_param & class
)
class TestZBLModelPT(unittest.TestCase, ZBLModelTest, PTTestCase):
    def setUp(self):
        ZBLModelTest.setUp(self)
        (DescriptorParam, Descrpt) = self.param[0]
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
        ds = Descrpt(**self.input_dict_ds)
        ft = EnergyFittingNet(
            **self.input_dict_ft,
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
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
