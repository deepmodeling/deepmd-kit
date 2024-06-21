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
    DescriptorParamDPA1,
    DescriptorParamDPA1List,
    DescriptorParamDPA2,
    DescriptorParamDPA2List,
    DescriptorParamHybrid,
    DescriptorParamHybridMixed,
    DescriptorParamSeA,
    DescriptorParamSeAList,
    DescriptorParamSeR,
    DescriptorParamSeRList,
    DescriptorParamSeT,
    DescriptorParamSeTList,
)
from ...dpmodel.fitting.test_fitting import (
    FittingParamDipole,
    FittingParamDipoleList,
    FittingParamDos,
    FittingParamDosList,
    FittingParamEnergy,
    FittingParamEnergyList,
    FittingParamPolar,
    FittingParamPolarList,
)
from ...dpmodel.model.test_model import (
    skip_model_tests,
)
from ..backend import (
    PTTestCase,
)


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
            *[(param_func, DescrptSeR) for param_func in DescriptorParamSeRList],
            *[(param_func, DescrptSeT) for param_func in DescriptorParamSeTList],
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            (DescriptorParamHybrid, DescrptHybrid),
            (DescriptorParamHybridMixed, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamEnergy, EnergyFittingNet),),  # fitting_class_param & class
    ),
    fit_parameterized=(
        (
            (DescriptorParamSeA, DescrptSeA),
            (DescriptorParamSeR, DescrptSeR),
            (DescriptorParamSeT, DescrptSeT),
            (DescriptorParamDPA1, DescrptDPA1),
            (DescriptorParamDPA2, DescrptDPA2),
        ),  # descrpt_class_param & class
        (
            *[(param_func, EnergyFittingNet) for param_func in FittingParamEnergyList],
        ),  # fitting_class_param & class
    ),
)
class TestEnergyAtomicModelPT(unittest.TestCase, EnerAtomicModelTest, PTTestCase):
    @classmethod
    def setUpClass(cls):
        EnerAtomicModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
        # set special precision
        if Descrpt in [DescrptDPA2]:
            cls.epsilon_dict["test_smooth"] = 1e-8
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
        cls.module = DPAtomicModel(
            ds,
            ft,
            type_map=cls.expected_type_map,
        )
        cls.output_def = cls.module.atomic_output_def().get_data()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
            *[(param_func, DescrptSeR) for param_func in DescriptorParamSeRList],
            *[(param_func, DescrptSeT) for param_func in DescriptorParamSeTList],
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            (DescriptorParamHybrid, DescrptHybrid),
            (DescriptorParamHybridMixed, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamDos, DOSFittingNet),),  # fitting_class_param & class
    ),
    fit_parameterized=(
        (
            (DescriptorParamSeA, DescrptSeA),
            (DescriptorParamSeR, DescrptSeR),
            (DescriptorParamSeT, DescrptSeT),
            (DescriptorParamDPA1, DescrptDPA1),
            (DescriptorParamDPA2, DescrptDPA2),
        ),  # descrpt_class_param & class
        (
            *[(param_func, DOSFittingNet) for param_func in FittingParamDosList],
        ),  # fitting_class_param & class
    ),
)
class TestDosAtomicModelPT(unittest.TestCase, DosAtomicModelTest, PTTestCase):
    @classmethod
    def setUpClass(cls):
        DosAtomicModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
        # set special precision
        cls.aprec_dict["test_smooth"] = 1e-4
        if Descrpt in [DescrptDPA2]:
            cls.epsilon_dict["test_smooth"] = 1e-8
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
        cls.module = DPAtomicModel(
            ds,
            ft,
            type_map=cls.expected_type_map,
        )
        cls.output_def = cls.module.atomic_output_def().get_data()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            (DescriptorParamHybrid, DescrptHybrid),
            (DescriptorParamHybridMixed, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamDipole, DipoleFittingNet),),  # fitting_class_param & class
    ),
    fit_parameterized=(
        (
            (DescriptorParamSeA, DescrptSeA),
            (DescriptorParamDPA1, DescrptDPA1),
            (DescriptorParamDPA2, DescrptDPA2),
        ),  # descrpt_class_param & class
        (
            *[(param_func, DipoleFittingNet) for param_func in FittingParamDipoleList],
        ),  # fitting_class_param & class
    ),
)
class TestDipoleAtomicModelPT(unittest.TestCase, DipoleAtomicModelTest, PTTestCase):
    @classmethod
    def setUpClass(cls):
        DipoleAtomicModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
        # set special precision
        if Descrpt in [DescrptDPA2]:
            cls.epsilon_dict["test_smooth"] = 1e-8
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
            embedding_width=ds.get_dim_emb(),
        )
        ft = Fitting(
            **cls.input_dict_ft,
        )
        cls.module = DPAtomicModel(
            ds,
            ft,
            type_map=cls.expected_type_map,
        )
        cls.output_def = cls.module.atomic_output_def().get_data()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            (DescriptorParamHybrid, DescrptHybrid),
            (DescriptorParamHybridMixed, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamPolar, PolarFittingNet),),  # fitting_class_param & class
    ),
    fit_parameterized=(
        (
            (DescriptorParamSeA, DescrptSeA),
            (DescriptorParamDPA1, DescrptDPA1),
            (DescriptorParamDPA2, DescrptDPA2),
        ),  # descrpt_class_param & class
        (
            *[(param_func, PolarFittingNet) for param_func in FittingParamPolarList],
        ),  # fitting_class_param & class
    ),
)
class TestPolarAtomicModelPT(unittest.TestCase, PolarAtomicModelTest, PTTestCase):
    @classmethod
    def setUpClass(cls):
        PolarAtomicModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
        # set special precision
        if Descrpt in [DescrptDPA2]:
            cls.epsilon_dict["test_smooth"] = 1e-8
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
            embedding_width=ds.get_dim_emb(),
        )
        ft = Fitting(
            **cls.input_dict_ft,
        )
        cls.module = DPAtomicModel(
            ds,
            ft,
            type_map=cls.expected_type_map,
        )
        cls.output_def = cls.module.atomic_output_def().get_data()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            (DescriptorParamHybridMixed, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamEnergy, EnergyFittingNet),),  # fitting_class_param & class
    ),
    fit_parameterized=(
        (
            (DescriptorParamDPA1, DescrptDPA1),
            (DescriptorParamDPA2, DescrptDPA2),
        ),  # descrpt_class_param & class
        (
            *[(param_func, EnergyFittingNet) for param_func in FittingParamEnergyList],
        ),  # fitting_class_param & class
    ),
)
class TestZBLAtomicModelPT(unittest.TestCase, ZBLAtomicModelTest, PTTestCase):
    @classmethod
    def setUpClass(cls):
        ZBLAtomicModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
        # set special precision
        # zbl weights not so smooth
        cls.aprec_dict["test_smooth"] = 5e-2
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
        dp_model = DPAtomicModel(
            ds,
            ft,
            type_map=cls.expected_type_map,
        )
        pt_model = PairTabAtomicModel(
            cls.tab_file["use_srtab"],
            cls.expected_rcut,
            cls.expected_sel,
            type_map=cls.expected_type_map,
        )
        cls.module = DPZBLLinearEnergyAtomicModel(
            dp_model,
            pt_model,
            sw_rmin=cls.tab_file["sw_rmin"],
            sw_rmax=cls.tab_file["sw_rmax"],
            smin_alpha=cls.tab_file["smin_alpha"],
            type_map=cls.expected_type_map,
        )
        cls.output_def = cls.module.atomic_output_def().get_data()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()
