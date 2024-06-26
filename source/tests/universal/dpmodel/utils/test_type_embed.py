# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.dpmodel.utils.type_embed import (
    TypeEmbedNet,
)

from ...common.cases.utils.type_embed import (
    TypeEmbdTest,
)
from ...pt.utils.utils import (
    TEST_DEVICE,
)
from ..backend import (
    DPTestCase,
)


@unittest.skipIf(TEST_DEVICE != "cpu", "Only test on CPU.")
class TestTypeEmbd(unittest.TestCase, TypeEmbdTest, DPTestCase):
    def setUp(self):
        TypeEmbdTest.setUp(self)
        self.module_class = TypeEmbedNet
        self.module = TypeEmbedNet(**self.input_dict)
