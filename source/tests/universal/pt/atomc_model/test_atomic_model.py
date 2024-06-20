# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
    DPZBLLinearEnergyAtomicModel,
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
from deepmd.pt.model.task import (
    DipoleFittingNet,
    DOSFittingNet,
    EnergyFittingNet,
    PolarFittingNet,
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
class TestEnergyAtomicModelPT(unittest.TestCase, EnerAtomicModelTest, PTTestCase):
    def setUp(self):
        EnerAtomicModelTest.setUp(self)
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
        self.module = DPAtomicModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.atomic_output_def().get_data()
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
class TestDosAtomicModelPT(unittest.TestCase, DosAtomicModelTest, PTTestCase):
    def setUp(self):
        DosAtomicModelTest.setUp(self)
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
        self.module = DPAtomicModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.atomic_output_def().get_data()
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
class TestDipoleAtomicModelPT(unittest.TestCase, DipoleAtomicModelTest, PTTestCase):
    def setUp(self):
        DipoleAtomicModelTest.setUp(self)
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
        self.module = DPAtomicModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.atomic_output_def().get_data()
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
class TestPolarAtomicModelPT(unittest.TestCase, PolarAtomicModelTest, PTTestCase):
    def setUp(self):
        PolarAtomicModelTest.setUp(self)
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
        self.module = DPAtomicModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.atomic_output_def().get_data()
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
class TestZBLAtomicModelPT(unittest.TestCase, ZBLAtomicModelTest, PTTestCase):
    def setUp(self):
        ZBLAtomicModelTest.setUp(self)
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
        self.expected_dim_fparam = ft.get_dim_fparam()
        self.expected_dim_aparam = ft.get_dim_aparam()
