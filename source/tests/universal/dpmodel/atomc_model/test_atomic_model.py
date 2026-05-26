# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

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
    PropertyFittingNet,
)

from ....consistent.common import (
    parameterized,
)
from ....utils import (
    CI,
    TEST_DEVICE,
)
from ...common.cases.atomic_model.atomic_model import (
    DipoleAtomicModelTest,
    DosAtomicModelTest,
    EnerAtomicModelTest,
    PolarAtomicModelTest,
    PropertyAtomicModelTest,
    ZBLAtomicModelTest,
)
from ...dpmodel.descriptor.test_descriptor import (
    DescriptorParamDPA1,
    DescriptorParamDPA1List,
    DescriptorParamDPA2,
    DescriptorParamDPA2List,
    DescriptorParamHybrid,
    DescriptorParamHybridMixed,
    DescriptorParamHybridMixedTTebd,
    DescriptorParamSeA,
    DescriptorParamSeAList,
    DescriptorParamSeR,
    DescriptorParamSeRList,
    DescriptorParamSeT,
    DescriptorParamSeTList,
)
from ...dpmodel.model.test_model import (
    skip_model_tests,
)
from ..backend import (
    DPTestCase,
)
from ..fitting.test_fitting import (
    FittingParamDipole,
    FittingParamDipoleList,
    FittingParamDos,
    FittingParamDosList,
    FittingParamEnergy,
    FittingParamEnergyList,
    FittingParamPolar,
    FittingParamPolarList,
    FittingParamProperty,
    FittingParamPropertyList,
)


def make_sel_type_from_atom_exclude_types(type_map, atom_exclude_types):
    """Get sel_type from complement of atom_exclude_types."""
    full_type_list = np.arange(len(type_map), dtype=int)
    sel_type = np.setdiff1d(full_type_list, atom_exclude_types, assume_unique=True)
    return sel_type.tolist()


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
            (DescriptorParamHybridMixedTTebd, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamEnergy, EnergyFittingNet),),  # fitting_class_param & class
        ([], [0]),  # atom_exclude_types
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
        ([], [0]),  # atom_exclude_types
    ),
)
@unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
class TestEnergyAtomicModelDP(unittest.TestCase, EnerAtomicModelTest, DPTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        EnerAtomicModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
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
            ds, ft, type_map=cls.expected_type_map, atom_exclude_types=cls.param[2]
        )
        cls.output_def = cls.module.atomic_output_def().get_data()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()

    def test_sel_type_from_atom_exclude_types(self):
        self.assertEqual(
            make_sel_type_from_atom_exclude_types(
                self.expected_type_map, self.param[2]
            ),
            self.expected_sel_type,
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
            (DescriptorParamHybridMixedTTebd, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamDos, DOSFittingNet),),  # fitting_class_param & class
        ([], [0]),  # atom_exclude_types
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
        ([], [0]),  # atom_exclude_types
    ),
)
@unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
class TestDosAtomicModelDP(unittest.TestCase, DosAtomicModelTest, DPTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        DosAtomicModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
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
            ds, ft, type_map=cls.expected_type_map, atom_exclude_types=cls.param[2]
        )
        cls.output_def = cls.module.atomic_output_def().get_data()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()

    def test_sel_type_from_atom_exclude_types(self):
        self.assertEqual(
            make_sel_type_from_atom_exclude_types(
                self.expected_type_map, self.param[2]
            ),
            self.expected_sel_type,
        )


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            (DescriptorParamHybrid, DescrptHybrid),
            (DescriptorParamHybridMixed, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamDipole, DipoleFitting),),  # fitting_class_param & class
        ([], [0]),  # atom_exclude_types
    ),
    fit_parameterized=(
        (
            (DescriptorParamSeA, DescrptSeA),
            (DescriptorParamDPA1, DescrptDPA1),
            (DescriptorParamDPA2, DescrptDPA2),
        ),  # descrpt_class_param & class
        (
            *[(param_func, DipoleFitting) for param_func in FittingParamDipoleList],
        ),  # fitting_class_param & class
        ([], [0]),  # atom_exclude_types
    ),
)
@unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
class TestDipoleAtomicModelDP(unittest.TestCase, DipoleAtomicModelTest, DPTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        DipoleAtomicModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
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
            ds, ft, type_map=cls.expected_type_map, atom_exclude_types=cls.param[2]
        )
        cls.output_def = cls.module.atomic_output_def().get_data()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()

    def test_sel_type_from_atom_exclude_types(self):
        self.assertEqual(
            make_sel_type_from_atom_exclude_types(
                self.expected_type_map, self.param[2]
            ),
            self.expected_sel_type,
        )


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            (DescriptorParamHybrid, DescrptHybrid),
            (DescriptorParamHybridMixed, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamPolar, PolarFitting),),  # fitting_class_param & class
        ([], [0]),  # atom_exclude_types
    ),
    fit_parameterized=(
        (
            (DescriptorParamSeA, DescrptSeA),
            (DescriptorParamDPA1, DescrptDPA1),
            (DescriptorParamDPA2, DescrptDPA2),
        ),  # descrpt_class_param & class
        (
            *[(param_func, PolarFitting) for param_func in FittingParamPolarList],
        ),  # fitting_class_param & class
        ([], [0]),  # atom_exclude_types
    ),
)
@unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
class TestPolarAtomicModelDP(unittest.TestCase, PolarAtomicModelTest, DPTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        PolarAtomicModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
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
            ds, ft, type_map=cls.expected_type_map, atom_exclude_types=cls.param[2]
        )
        cls.output_def = cls.module.atomic_output_def().get_data()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()

    def test_sel_type_from_atom_exclude_types(self):
        self.assertEqual(
            make_sel_type_from_atom_exclude_types(
                self.expected_type_map, self.param[2]
            ),
            self.expected_sel_type,
        )


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            (DescriptorParamHybridMixed, DescrptHybrid),
            (DescriptorParamHybridMixedTTebd, DescrptHybrid),
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
@unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
class TestZBLAtomicModelDP(unittest.TestCase, ZBLAtomicModelTest, DPTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ZBLAtomicModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
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


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            (DescriptorParamHybridMixed, DescrptHybrid),
            (DescriptorParamHybridMixedTTebd, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamProperty, PropertyFittingNet),),  # fitting_class_param & class
        ([], [0]),  # atom_exclude_types
    ),
    fit_parameterized=(
        (
            (DescriptorParamSeA, DescrptSeA),
            (DescriptorParamDPA1, DescrptDPA1),
            (DescriptorParamDPA2, DescrptDPA2),
        ),  # descrpt_class_param & class
        (
            *[
                (param_func, PropertyFittingNet)
                for param_func in FittingParamPropertyList
            ],
        ),  # fitting_class_param & class
        ([], [0]),  # atom_exclude_types
    ),
)
@unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
class TestPropertyAtomicModelDP(unittest.TestCase, PropertyAtomicModelTest, DPTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        PropertyAtomicModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
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
            ds, ft, type_map=cls.expected_type_map, atom_exclude_types=cls.param[2]
        )
        cls.output_def = cls.module.atomic_output_def().get_data()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()

    def test_sel_type_from_atom_exclude_types(self):
        self.assertEqual(
            make_sel_type_from_atom_exclude_types(
                self.expected_type_map, self.param[2]
            ),
            self.expected_sel_type,
        )
