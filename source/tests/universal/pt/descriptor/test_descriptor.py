# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.pt.model.descriptor import (
    DescrptDPA1,
    DescrptDPA2,
    DescrptHybrid,
    DescrptSeA,
    DescrptSeR,
    DescrptSeT,
)

from ...common.cases.descriptor.descriptor import (
    DescriptorTest,
)
from ..backend import (
    PTTestCase,
)


class TestDescriptorSeAPT(unittest.TestCase, DescriptorTest, PTTestCase):
    def setUp(self):
        DescriptorTest.setUp(self)
        self.module = DescrptSeA(**self.input_dict)


class TestDescriptorSeRPT(unittest.TestCase, DescriptorTest, PTTestCase):
    def setUp(self):
        DescriptorTest.setUp(self)
        self.module = DescrptSeR(**self.input_dict)


class TestDescriptorSeTPT(unittest.TestCase, DescriptorTest, PTTestCase):
    def setUp(self):
        DescriptorTest.setUp(self)
        self.module = DescrptSeT(**self.input_dict)


class TestDescriptorDPA1PT(unittest.TestCase, DescriptorTest, PTTestCase):
    def setUp(self):
        DescriptorTest.setUp(self)
        self.module = DescrptDPA1(**self.input_dict)


class TestDescriptorDPA2PT(unittest.TestCase, DescriptorTest, PTTestCase):
    def setUp(self):
        DescriptorTest.setUp(self)
        self.input_dict = {
            "ntypes": self.nt,
            "repinit": {
                "rcut": self.rcut,
                "rcut_smth": self.rcut_smth,
                "nsel": self.sel_mix,
            },
            "repformer": {
                "rcut": self.rcut / 2,
                "rcut_smth": self.rcut_smth,
                "nsel": self.sel_mix[0] // 2,
            },
        }
        self.module = DescrptDPA2(**self.input_dict)


class TestDescriptorHybridPT(unittest.TestCase, DescriptorTest, PTTestCase):
    def setUp(self):
        DescriptorTest.setUp(self)
        ddsub0 = {
            "type": "se_e2_a",
            "ntypes": self.nt,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
        }
        ddsub1 = {
            "type": "dpa1",
            "ntypes": self.nt,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel_mix,
        }
        self.input_dict = {
            "list": [ddsub0, ddsub1],
        }
        self.module = DescrptHybrid(**self.input_dict)
