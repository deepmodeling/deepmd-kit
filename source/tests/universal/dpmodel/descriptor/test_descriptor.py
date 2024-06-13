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

from ....consistent.common import (
    parameterized,
)
from ....seed import (
    GLOBAL_SEED,
)
from ...common.cases.descriptor.descriptor import (
    DescriptorTest,
)
from ..backend import (
    DPTestCase,
)


def DescriptorParamSeA(ntypes, rcut, rcut_smth, sel, type_map):
    input_dict = {
        "ntypes": ntypes,
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "sel": sel,
        "type_map": type_map,
        "seed": GLOBAL_SEED,
    }
    return input_dict


def DescriptorParamSeR(ntypes, rcut, rcut_smth, sel, type_map):
    input_dict = {
        "ntypes": ntypes,
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "sel": sel,
        "type_map": type_map,
        "seed": GLOBAL_SEED,
    }
    return input_dict


def DescriptorParamSeT(ntypes, rcut, rcut_smth, sel, type_map):
    input_dict = {
        "ntypes": ntypes,
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "sel": sel,
        "type_map": type_map,
        "seed": GLOBAL_SEED,
    }
    return input_dict


def DescriptorParamDPA1(ntypes, rcut, rcut_smth, sel, type_map):
    input_dict = {
        "ntypes": ntypes,
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "sel": sel,
        "type_map": type_map,
        "seed": GLOBAL_SEED,
    }
    return input_dict


def DescriptorParamDPA2(ntypes, rcut, rcut_smth, sel, type_map):
    input_dict = {
        "ntypes": ntypes,
        "repinit": {
            "rcut": rcut,
            "rcut_smth": rcut_smth,
            "nsel": sum(sel),
        },
        "repformer": {
            "rcut": rcut / 2,
            "rcut_smth": rcut_smth / 2,
            "nsel": sum(sel) // 2,
        },
        "type_map": type_map,
        "seed": GLOBAL_SEED,
    }
    return input_dict


def DescriptorParamHybrid(ntypes, rcut, rcut_smth, sel, type_map):
    ddsub0 = {
        "type": "se_e2_a",
        "ntypes": ntypes,
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "sel": sel,
        "type_map": type_map,
        "seed": GLOBAL_SEED,
    }
    ddsub1 = {
        "type": "dpa1",
        "ntypes": ntypes,
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "sel": sum(sel),
        "type_map": type_map,
        "seed": GLOBAL_SEED,
    }
    input_dict = {
        "list": [ddsub0, ddsub1],
    }
    return input_dict


def DescriptorParamHybridMixed(ntypes, rcut, rcut_smth, sel, type_map):
    ddsub0 = {
        "type": "dpa1",
        "ntypes": ntypes,
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "sel": sum(sel),
        "type_map": type_map,
        "seed": GLOBAL_SEED,
    }
    ddsub1 = {
        "type": "dpa1",
        "ntypes": ntypes,
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "sel": sum(sel),
        "type_map": type_map,
        "seed": GLOBAL_SEED,
    }
    input_dict = {
        "list": [ddsub0, ddsub1],
    }
    return input_dict


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
class TestDescriptorDP(unittest.TestCase, DescriptorTest, DPTestCase):
    def setUp(self):
        DescriptorTest.setUp(self)
        (DescriptorParam, Descrpt) = self.param[0]
        self.module_class = Descrpt
        self.input_dict = DescriptorParam(
            self.nt, self.rcut, self.rcut_smth, self.sel, ["O", "H"]
        )
        self.module = Descrpt(**self.input_dict)
