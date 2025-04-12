# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
    PairTabAtomicModel,
)
from deepmd.pt.model.descriptor import (
    DescrptDPA1,
    DescrptDPA2,
    DescrptDPA3,
    DescrptHybrid,
    DescrptSeA,
    DescrptSeR,
    DescrptSeT,
    DescrptSeTTebd,
)
from deepmd.pt.model.model import (
    DipoleModel,
    DOSModel,
    DPZBLModel,
    EnergyModel,
    LinearEnergyModel,
    PolarModel,
    PropertyModel,
    SpinEnergyModel,
)
from deepmd.pt.model.task import (
    DipoleFittingNet,
    DOSFittingNet,
    EnergyFittingNet,
    PolarFittingNet,
    PropertyFittingNet,
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
    LinearEnerModelTest,
    PolarModelTest,
    PropertyModelTest,
    SpinEnerModelTest,
    ZBLModelTest,
)
from ...dpmodel.descriptor.test_descriptor import (
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
from ...dpmodel.fitting.test_fitting import (
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
from ...dpmodel.model.test_model import (
    skip_model_tests,
)
from ..backend import (
    PTTestCase,
)

defalut_des_param = [
    DescriptorParamSeA,
    DescriptorParamSeR,
    DescriptorParamSeT,
    DescriptorParamSeTTebd,
    DescriptorParamDPA1,
    DescriptorParamDPA2,
    DescriptorParamDPA3,
    DescriptorParamHybrid,
    DescriptorParamHybridMixed,
]
defalut_fit_param = [
    FittingParamEnergy,
    FittingParamDos,
    FittingParamDipole,
    FittingParamPolar,
    FittingParamProperty,
]


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
class TestEnergyModelPT(unittest.TestCase, EnerModelTest, PTTestCase):
    @property
    def modules_to_test(self):
        skip_test_jit = getattr(self, "skip_test_jit", False)
        modules = PTTestCase.modules_to_test.fget(self)
        if not skip_test_jit:
            # for Model, we can test script module API
            modules += [
                self._script_module
                if hasattr(self, "_script_module")
                else self.script_module
            ]
        return modules

    @classmethod
    def setUpClass(cls) -> None:
        EnerModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
        # set special precision
        if Descrpt in [DescrptDPA2]:
            cls.epsilon_dict["test_smooth"] = 1e-8
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
        # only test jit API once for different models
        if (
            DescriptorParam not in defalut_des_param
            or FittingParam not in defalut_fit_param
        ):
            cls.skip_test_jit = True
        else:
            with torch.jit.optimized_execution(False):
                cls._script_module = torch.jit.script(cls.module)
        cls.output_def = cls.module.translated_output_def()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()

    @classmethod
    def tearDownClass(cls) -> None:
        PTTestCase.tearDownClass()


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
        ((FittingParamDos, DOSFittingNet),),  # fitting_class_param & class
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
            *[(param_func, DOSFittingNet) for param_func in FittingParamDosList],
        ),  # fitting_class_param & class
    ),
)
class TestDosModelPT(unittest.TestCase, DosModelTest, PTTestCase):
    @property
    def modules_to_test(self):
        skip_test_jit = getattr(self, "skip_test_jit", False)
        modules = PTTestCase.modules_to_test.fget(self)
        if not skip_test_jit:
            # for Model, we can test script module API
            modules += [
                self._script_module
                if hasattr(self, "_script_module")
                else self.script_module
            ]
        return modules

    @classmethod
    def setUpClass(cls) -> None:
        DosModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
        # set special precision
        cls.aprec_dict["test_smooth"] = 1e-4
        if Descrpt in [DescrptDPA2]:
            cls.epsilon_dict["test_smooth"] = 1e-8
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
        cls.module = DOSModel(
            ds,
            ft,
            type_map=cls.expected_type_map,
        )
        # only test jit API once for different models
        if (
            DescriptorParam not in defalut_des_param
            or FittingParam not in defalut_fit_param
        ):
            cls.skip_test_jit = True
        else:
            with torch.jit.optimized_execution(False):
                cls._script_module = torch.jit.script(cls.module)
        cls.output_def = cls.module.translated_output_def()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()

    @classmethod
    def tearDownClass(cls) -> None:
        PTTestCase.tearDownClass()


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            *[(param_func, DescrptDPA3) for param_func in DescriptorParamDPA3List],
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
            (DescriptorParamDPA3, DescrptDPA3),
        ),  # descrpt_class_param & class
        (
            *[(param_func, DipoleFittingNet) for param_func in FittingParamDipoleList],
        ),  # fitting_class_param & class
    ),
)
class TestDipoleModelPT(unittest.TestCase, DipoleModelTest, PTTestCase):
    @property
    def modules_to_test(self):
        skip_test_jit = getattr(self, "skip_test_jit", False)
        modules = PTTestCase.modules_to_test.fget(self)
        if not skip_test_jit:
            # for Model, we can test script module API
            modules += [
                self._script_module
                if hasattr(self, "_script_module")
                else self.script_module
            ]
        return modules

    @classmethod
    def setUpClass(cls) -> None:
        DipoleModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
        # set special precision
        if Descrpt in [DescrptDPA2]:
            cls.epsilon_dict["test_smooth"] = 1e-8
        cls.aprec_dict["test_forward"] = 1e-10  # for dipole force when near zero
        cls.aprec_dict["test_rot"] = 1e-10  # for dipole force when near zero
        cls.aprec_dict["test_trans"] = 1e-10  # for dipole force when near zero
        cls.aprec_dict["test_permutation"] = 1e-10  # for dipole force when near zero
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
        cls.module = DipoleModel(
            ds,
            ft,
            type_map=cls.expected_type_map,
        )
        # only test jit API once for different models
        if (
            DescriptorParam not in defalut_des_param
            or FittingParam not in defalut_fit_param
        ):
            cls.skip_test_jit = True
        else:
            with torch.jit.optimized_execution(False):
                cls._script_module = torch.jit.script(cls.module)
        cls.output_def = cls.module.translated_output_def()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()

    @classmethod
    def tearDownClass(cls) -> None:
        PTTestCase.tearDownClass()


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            *[(param_func, DescrptDPA3) for param_func in DescriptorParamDPA3List],
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
            (DescriptorParamDPA3, DescrptDPA3),
        ),  # descrpt_class_param & class
        (
            *[(param_func, PolarFittingNet) for param_func in FittingParamPolarList],
        ),  # fitting_class_param & class
    ),
)
class TestPolarModelPT(unittest.TestCase, PolarModelTest, PTTestCase):
    @property
    def modules_to_test(self):
        skip_test_jit = getattr(self, "skip_test_jit", False)
        modules = PTTestCase.modules_to_test.fget(self)
        if not skip_test_jit:
            # for Model, we can test script module API
            modules += [
                self._script_module
                if hasattr(self, "_script_module")
                else self.script_module
            ]
        return modules

    @classmethod
    def setUpClass(cls) -> None:
        PolarModelTest.setUpClass()
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
        cls.module = PolarModel(
            ds,
            ft,
            type_map=cls.expected_type_map,
        )
        # only test jit API once for different models
        if (
            DescriptorParam not in defalut_des_param
            or FittingParam not in defalut_fit_param
        ):
            cls.skip_test_jit = True
        else:
            with torch.jit.optimized_execution(False):
                cls._script_module = torch.jit.script(cls.module)
        cls.output_def = cls.module.translated_output_def()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()

    @classmethod
    def tearDownClass(cls) -> None:
        PTTestCase.tearDownClass()


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
class TestZBLModelPT(unittest.TestCase, ZBLModelTest, PTTestCase):
    @property
    def modules_to_test(self):
        skip_test_jit = getattr(self, "skip_test_jit", False)
        modules = PTTestCase.modules_to_test.fget(self)
        if not skip_test_jit:
            # for Model, we can test script module API
            modules += [
                self._script_module
                if hasattr(self, "_script_module")
                else self.script_module
            ]
        return modules

    @classmethod
    def setUpClass(cls) -> None:
        ZBLModelTest.setUpClass()
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
        cls.module = DPZBLModel(
            dp_model,
            pt_model,
            sw_rmin=cls.tab_file["sw_rmin"],
            sw_rmax=cls.tab_file["sw_rmax"],
            smin_alpha=cls.tab_file["smin_alpha"],
            type_map=cls.expected_type_map,
        )
        # only test jit API once for different models
        if (
            DescriptorParam not in defalut_des_param
            or FittingParam not in defalut_fit_param
        ):
            cls.skip_test_jit = True
        else:
            with torch.jit.optimized_execution(False):
                cls._script_module = torch.jit.script(cls.module)
        cls.output_def = cls.module.translated_output_def()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()

    @classmethod
    def tearDownClass(cls) -> None:
        PTTestCase.tearDownClass()


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
        ),  # descrpt_class_param & class
        (
            *[(param_func, EnergyFittingNet) for param_func in FittingParamEnergyList],
        ),  # fitting_class_param & class
    ),
)
class TestSpinEnergyModelDP(unittest.TestCase, SpinEnerModelTest, PTTestCase):
    @property
    def modules_to_test(self):
        skip_test_jit = getattr(self, "skip_test_jit", False)
        modules = PTTestCase.modules_to_test.fget(self)
        if not skip_test_jit:
            # for Model, we can test script module API
            modules += [
                self._script_module
                if hasattr(self, "_script_module")
                else self.script_module
            ]
        return modules

    @classmethod
    def setUpClass(cls) -> None:
        SpinEnerModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
        cls.epsilon_dict["test_smooth"] = 1e-6
        cls.aprec_dict["test_smooth"] = 5e-5
        cls.aprec_dict["test_rot"] = 1e-10  # for test stability
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
        cls.module = SpinEnergyModel(backbone_model=backbone_model, spin=spin)
        # only test jit API once for different models
        if (
            DescriptorParam not in defalut_des_param
            or FittingParam not in defalut_fit_param
        ):
            cls.skip_test_jit = True
        else:
            with torch.jit.optimized_execution(False):
                cls._script_module = torch.jit.script(cls.module)
        cls.output_def = cls.module.translated_output_def()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()

    @classmethod
    def tearDownClass(cls) -> None:
        PTTestCase.tearDownClass()


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptSeA) for param_func in DescriptorParamSeAList],
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            *[(param_func, DescrptDPA3) for param_func in DescriptorParamDPA3List],
            (DescriptorParamHybrid, DescrptHybrid),
            (DescriptorParamHybridMixed, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamProperty, PropertyFittingNet),),  # fitting_class_param & class
    ),
    fit_parameterized=(
        (
            (DescriptorParamSeA, DescrptSeA),
            (DescriptorParamDPA1, DescrptDPA1),
            (DescriptorParamDPA2, DescrptDPA2),
            (DescriptorParamDPA3, DescrptDPA3),
        ),  # descrpt_class_param & class
        (
            *[
                (param_func, PropertyFittingNet)
                for param_func in FittingParamPropertyList
            ],
        ),  # fitting_class_param & class
    ),
)
class TestPropertyModelPT(unittest.TestCase, PropertyModelTest, PTTestCase):
    @property
    def modules_to_test(self):
        skip_test_jit = getattr(self, "skip_test_jit", False)
        modules = PTTestCase.modules_to_test.fget(self)
        if not skip_test_jit:
            # for Model, we can test script module API
            modules += [
                self._script_module
                if hasattr(self, "_script_module")
                else self.script_module
            ]
        return modules

    @classmethod
    def setUpClass(cls) -> None:
        PropertyModelTest.setUpClass()
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
        cls.module = PropertyModel(
            ds,
            ft,
            type_map=cls.expected_type_map,
        )
        # only test jit API once for different models
        if (
            DescriptorParam not in defalut_des_param
            or FittingParam not in defalut_fit_param
        ):
            cls.skip_test_jit = True
        else:
            with torch.jit.optimized_execution(False):
                cls._script_module = torch.jit.script(cls.module)
        cls.output_def = cls.module.translated_output_def()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()

    @classmethod
    def tearDownClass(cls) -> None:
        PTTestCase.tearDownClass()


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptDPA1) for param_func in DescriptorParamDPA1List],
            *[(param_func, DescrptDPA2) for param_func in DescriptorParamDPA2List],
            *[(param_func, DescrptDPA3) for param_func in DescriptorParamDPA3List],
            (DescriptorParamHybridMixed, DescrptHybrid),
            (DescriptorParamHybridMixedTTebd, DescrptHybrid),
        ),  # descrpt_class_param & class
        ((FittingParamEnergy, EnergyFittingNet),),  # fitting_class_param & class
    ),
    fit_parameterized=(
        (
            (DescriptorParamDPA1, DescrptDPA1),
            (DescriptorParamDPA2, DescrptDPA2),
            (DescriptorParamDPA3, DescrptDPA3),
        ),  # descrpt_class_param & class
        (
            *[(param_func, EnergyFittingNet) for param_func in FittingParamEnergyList],
        ),  # fitting_class_param & class
    ),
)
class TestLinearEnergyModelPT(unittest.TestCase, LinearEnerModelTest, PTTestCase):
    @property
    def modules_to_test(self):
        skip_test_jit = getattr(self, "skip_test_jit", False)
        modules = PTTestCase.modules_to_test.fget(self)
        if not skip_test_jit:
            # for Model, we can test script module API
            modules += [
                self._script_module
                if hasattr(self, "_script_module")
                else self.script_module
            ]
        return modules

    @classmethod
    def setUpClass(cls) -> None:
        LinearEnerModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
        # set special precision
        cls.aprec_dict["test_smooth"] = 1e-5
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

        ds1, ds2 = Descrpt(**cls.input_dict_ds), Descrpt(**cls.input_dict_ds)
        cls.input_dict_ft = FittingParam(
            ntypes=len(cls.expected_type_map),
            dim_descrpt=ds1.get_dim_out(),
            mixed_types=ds1.mixed_types(),
            type_map=cls.expected_type_map,
        )
        ft1 = Fitting(
            **cls.input_dict_ft,
        )
        ft2 = Fitting(
            **cls.input_dict_ft,
        )
        dp_model1 = DPAtomicModel(
            ds1,
            ft1,
            type_map=cls.expected_type_map,
        )
        dp_model2 = DPAtomicModel(
            ds2,
            ft2,
            type_map=cls.expected_type_map,
        )
        cls.module = LinearEnergyModel(
            [dp_model1, dp_model2],
            type_map=cls.expected_type_map,
        )
        # only test jit API once for different models
        if (
            DescriptorParam not in defalut_des_param
            or FittingParam not in defalut_fit_param
        ):
            cls.skip_test_jit = True
        else:
            with torch.jit.optimized_execution(False):
                cls._script_module = torch.jit.script(cls.module)
        cls.output_def = cls.module.translated_output_def()
        cls.expected_has_message_passing = ds1.has_message_passing()
        cls.expected_dim_fparam = ft1.get_dim_fparam()
        cls.expected_dim_aparam = ft1.get_dim_aparam()
        cls.expected_sel_type = ft1.get_sel_type()

    @classmethod
    def tearDownClass(cls) -> None:
        PTTestCase.tearDownClass()
