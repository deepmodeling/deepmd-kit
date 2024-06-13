# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.dpmodel.utils.type_embed import (
    TypeEmbedNet,
)

from ...common.cases.utils.type_embed import (
    TypeEmbdTest,
)
from ..backend import (
    DPTestCase,
)


class TestTypeEmbd(unittest.TestCase, TypeEmbdTest, DPTestCase):
    def setUp(self):
        TypeEmbdTest.setUp(self)
        self.module_class = TypeEmbedNet
        self.module = TypeEmbedNet(**self.input_dict)
