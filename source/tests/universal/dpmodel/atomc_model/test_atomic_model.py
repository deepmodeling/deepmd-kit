# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
    DPZBLLinearEnergyAtomicModel,
    PairTabAtomicModel,
)
from deepmd.dpmodel.descriptor import (
    DescrptDPA1,
    DescrptDPA2,
    DescrptHybrid,
    DescrptSeA,
    DescrptSeR,
    DescrptSeT,
)
from deepmd.dpmodel.fitting import (
    DipoleFitting,
    DOSFittingNet,
    EnergyFittingNet,
    PolarFitting,
)

from ....consistent.common import (
    parameterized,
)
from ...common.cases.atomic_model.atomic_model import (
    DipoleAtomicModelTest,
    DosAtomicModelTest,
    EnerAtomicModelTest,
    PolarAtomicModelTest,
    ZBLAtomicModelTest,
)
from ..backend import (
    DPTestCase,
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
class TestEnergyAtomicModelDP(unittest.TestCase, EnerAtomicModelTest, DPTestCase):
    def setUp(self):
        EnerAtomicModelTest.setUp(self)
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
        self.module = DPAtomicModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.atomic_output_def().get_data()
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
class TestDosAtomicModelDP(unittest.TestCase, DosAtomicModelTest, DPTestCase):
    def setUp(self):
        DosAtomicModelTest.setUp(self)
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
        self.module = DPAtomicModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.atomic_output_def().get_data()
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
class TestDipoleAtomicModelDP(unittest.TestCase, DipoleAtomicModelTest, DPTestCase):
    def setUp(self):
        DipoleAtomicModelTest.setUp(self)
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
        ft = DipoleFitting(
            **self.input_dict_ft,
            embedding_width=ds.get_dim_emb(),
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        )
        self.module = DPAtomicModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.atomic_output_def().get_data()
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
class TestPolarAtomicModelDP(unittest.TestCase, PolarAtomicModelTest, DPTestCase):
    def setUp(self):
        PolarAtomicModelTest.setUp(self)
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
        ft = PolarFitting(
            **self.input_dict_ft,
            embedding_width=ds.get_dim_emb(),
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        )
        self.module = DPAtomicModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.atomic_output_def().get_data()
        self.expected_has_message_passing = ds.has_message_passing()


@parameterized(
    (
        (DescriptorParamDPA1, DescrptDPA1),
        (DescriptorParamDPA2, DescrptDPA2),
        (DescriptorParamHybridMixed, DescrptHybrid),
    )  # class_param & class
)
class TestZBLAtomicModelDP(unittest.TestCase, ZBLAtomicModelTest, DPTestCase):
    def setUp(self):
        ZBLAtomicModelTest.setUp(self)
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
        self.module = DPZBLLinearEnergyAtomicModel(
            dp_model,
            pt_model,
            sw_rmin=self.tab_file["sw_rmin"],
            sw_rmax=self.tab_file["sw_rmax"],
            smin_alpha=self.tab_file["smin_alpha"],
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.atomic_output_def().get_data()
        self.expected_has_message_passing = ds.has_message_passing()
