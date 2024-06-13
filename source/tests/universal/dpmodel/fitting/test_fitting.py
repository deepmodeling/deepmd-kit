# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.dpmodel.fitting import (
    DipoleFitting,
    DOSFittingNet,
    EnergyFittingNet,
    PolarFitting,
)

from ....consistent.common import (
    parameterized,
)
from ...common.cases.fitting.fitting import (
    FittingTest,
)
from ..backend import (
    DPTestCase,
)


def FittingParamEnergy(ntypes, dim_descrpt, mixed_types, embedding_width, type_map):
    input_dict = {
        "ntypes": ntypes,
        "dim_descrpt": dim_descrpt,
        "mixed_types": mixed_types,
        "type_map": type_map,
    }
    return input_dict


def FittingParamDos(ntypes, dim_descrpt, mixed_types, embedding_width, type_map):
    input_dict = {
        "ntypes": ntypes,
        "dim_descrpt": dim_descrpt,
        "mixed_types": mixed_types,
        "type_map": type_map,
    }
    return input_dict


def FittingParamDipole(ntypes, dim_descrpt, mixed_types, embedding_width, type_map):
    input_dict = {
        "ntypes": ntypes,
        "dim_descrpt": dim_descrpt,
        "mixed_types": mixed_types,
        "embedding_width": embedding_width,
        "type_map": type_map,
    }
    return input_dict


def FittingParamPolar(ntypes, dim_descrpt, mixed_types, embedding_width, type_map):
    input_dict = {
        "ntypes": ntypes,
        "dim_descrpt": dim_descrpt,
        "mixed_types": mixed_types,
        "embedding_width": embedding_width,
        "type_map": type_map,
    }
    return input_dict


@parameterized(
    (
        (FittingParamEnergy, EnergyFittingNet),
        (FittingParamDos, DOSFittingNet),
        (FittingParamDipole, DipoleFitting),
        (FittingParamPolar, PolarFitting),
    ),  # class_param & class
    (True, False),  # mixed_types
)
class TestFittingDP(unittest.TestCase, FittingTest, DPTestCase):
    def setUp(self):
        ((FittingParam, Fitting), self.mixed_types) = self.param
        FittingTest.setUp(self)
        self.module_class = Fitting
        self.input_dict = FittingParam(
            self.nt, self.dim_descrpt, self.mixed_types, self.dim_embed, ["O", "H"]
        )
        self.module = Fitting(**self.input_dict)
