# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.dpmodel.utils.type_embed import (
    TypeEmbedNet,
)

from ....utils import (
    CI,
    TEST_DEVICE,
)
from ...common.cases.utils.type_embed import (
    TypeEmbdTest,
)
from ..backend import (
    DPTestCase,
)


@unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
class TestTypeEmbd(unittest.TestCase, TypeEmbdTest, DPTestCase):
    def setUp(self) -> None:
        TypeEmbdTest.setUp(self)
        self.module_class = TypeEmbedNet
        self.module = TypeEmbedNet(**self.input_dict)
