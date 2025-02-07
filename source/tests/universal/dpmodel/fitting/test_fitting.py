# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from collections import (
    OrderedDict,
)

from deepmd.dpmodel.fitting import (
    DipoleFitting,
    DOSFittingNet,
    EnergyFittingNet,
    PolarFitting,
    PropertyFittingNet,
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
from ...common.cases.fitting.fitting import (
    FittingTest,
)
from ..backend import (
    DPTestCase,
)


def FittingParamEnergy(
    ntypes,
    dim_descrpt,
    mixed_types,
    type_map,
    exclude_types=[],
    precision="float64",
    embedding_width=None,
    numb_param=0,  # test numb_fparam, numb_aparam and dim_case_embd together
):
    input_dict = {
        "ntypes": ntypes,
        "dim_descrpt": dim_descrpt,
        "mixed_types": mixed_types,
        "type_map": type_map,
        "exclude_types": exclude_types,
        "seed": GLOBAL_SEED,
        "precision": precision,
        "numb_fparam": numb_param,
        "numb_aparam": numb_param,
        "dim_case_embd": numb_param,
    }
    return input_dict


FittingParamEnergyList = parameterize_func(
    FittingParamEnergy,
    OrderedDict(
        {
            "exclude_types": ([], [0]),
            "precision": ("float64",),
            "numb_param": (0, 2),
        }
    ),
)
# to get name for the default function
FittingParamEnergy = FittingParamEnergyList[0]


def FittingParamDos(
    ntypes,
    dim_descrpt,
    mixed_types,
    type_map,
    exclude_types=[],
    precision="float64",
    embedding_width=None,
    numb_param=0,  # test numb_fparam, numb_aparam and dim_case_embd together
):
    input_dict = {
        "ntypes": ntypes,
        "dim_descrpt": dim_descrpt,
        "mixed_types": mixed_types,
        "type_map": type_map,
        "exclude_types": exclude_types,
        "seed": GLOBAL_SEED,
        "precision": precision,
        "numb_fparam": numb_param,
        "numb_aparam": numb_param,
        "dim_case_embd": numb_param,
    }
    return input_dict


FittingParamDosList = parameterize_func(
    FittingParamDos,
    OrderedDict(
        {
            "exclude_types": ([], [0]),
            "precision": ("float64",),
            "numb_param": (0, 2),
        }
    ),
)
# to get name for the default function
FittingParamDos = FittingParamDosList[0]


def FittingParamDipole(
    ntypes,
    dim_descrpt,
    mixed_types,
    type_map,
    exclude_types=[],
    precision="float64",
    embedding_width=None,
    numb_param=0,  # test numb_fparam, numb_aparam and dim_case_embd together
):
    assert embedding_width is not None, (
        "embedding_width for dipole fitting is required."
    )
    input_dict = {
        "ntypes": ntypes,
        "dim_descrpt": dim_descrpt,
        "mixed_types": mixed_types,
        "embedding_width": embedding_width,
        "type_map": type_map,
        "exclude_types": exclude_types,
        "seed": GLOBAL_SEED,
        "precision": precision,
        "numb_fparam": numb_param,
        "numb_aparam": numb_param,
        "dim_case_embd": numb_param,
    }
    return input_dict


FittingParamDipoleList = parameterize_func(
    FittingParamDipole,
    OrderedDict(
        {
            "exclude_types": ([], [0]),
            "precision": ("float64",),
            "numb_param": (0, 2),
        }
    ),
)
# to get name for the default function
FittingParamDipole = FittingParamDipoleList[0]


def FittingParamPolar(
    ntypes,
    dim_descrpt,
    mixed_types,
    type_map,
    exclude_types=[],
    precision="float64",
    embedding_width=None,
    numb_param=0,  # test numb_fparam, numb_aparam and dim_case_embd together
):
    assert embedding_width is not None, "embedding_width for polar fitting is required."
    input_dict = {
        "ntypes": ntypes,
        "dim_descrpt": dim_descrpt,
        "mixed_types": mixed_types,
        "embedding_width": embedding_width,
        "type_map": type_map,
        "exclude_types": exclude_types,
        "seed": GLOBAL_SEED,
        "precision": precision,
        "numb_fparam": numb_param,
        "numb_aparam": numb_param,
        "dim_case_embd": numb_param,
    }
    return input_dict


FittingParamPolarList = parameterize_func(
    FittingParamPolar,
    OrderedDict(
        {
            "exclude_types": ([], [0]),
            "precision": ("float64",),
            "numb_param": (0, 2),
        }
    ),
)
# to get name for the default function
FittingParamPolar = FittingParamPolarList[0]


def FittingParamProperty(
    ntypes,
    dim_descrpt,
    mixed_types,
    type_map,
    exclude_types=[],
    precision="float64",
    embedding_width=None,
    numb_param=0,  # test numb_fparam, numb_aparam and dim_case_embd together
):
    input_dict = {
        "ntypes": ntypes,
        "dim_descrpt": dim_descrpt,
        "mixed_types": mixed_types,
        "type_map": type_map,
        "task_dim": 3,
        "property_name": "band_prop",
        "exclude_types": exclude_types,
        "seed": GLOBAL_SEED,
        "precision": precision,
        "numb_fparam": numb_param,
        "numb_aparam": numb_param,
        "dim_case_embd": numb_param,
    }
    return input_dict


FittingParamPropertyList = parameterize_func(
    FittingParamProperty,
    OrderedDict(
        {
            "exclude_types": ([], [0]),
            "precision": ("float64",),
            "numb_param": (0, 2),
        }
    ),
)
# to get name for the default function
FittingParamProperty = FittingParamPropertyList[0]


@parameterized(
    (
        (FittingParamEnergy, EnergyFittingNet),
        (FittingParamDos, DOSFittingNet),
        (FittingParamDipole, DipoleFitting),
        (FittingParamPolar, PolarFitting),
        (FittingParamProperty, PropertyFittingNet),
    ),  # class_param & class
    (True, False),  # mixed_types
)
@unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
class TestFittingDP(unittest.TestCase, FittingTest, DPTestCase):
    def setUp(self) -> None:
        ((FittingParam, Fitting), self.mixed_types) = self.param
        FittingTest.setUp(self)
        self.module_class = Fitting
        self.input_dict = FittingParam(
            self.nt,
            self.dim_descrpt,
            self.mixed_types,
            ["O", "H"],
            embedding_width=self.dim_embed,
        )
        self.module = Fitting(**self.input_dict)
