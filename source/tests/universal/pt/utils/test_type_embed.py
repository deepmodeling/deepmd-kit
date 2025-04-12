# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.pt.model.network.network import (
    TypeEmbedNetConsistent,
)
from deepmd.pt.utils import (
    env,
)

from ...common.cases.utils.type_embed import (
    TypeEmbdTest,
)
from ..backend import (
    PTTestCase,
)


class TestTypeEmbd(unittest.TestCase, TypeEmbdTest, PTTestCase):
    def setUp(self) -> None:
        TypeEmbdTest.setUp(self)
        self.module_class = TypeEmbedNetConsistent
        self.module = TypeEmbedNetConsistent(**self.input_dict)
        self.module_input = {"device": env.DEVICE}

    @classmethod
    def tearDownClass(cls) -> None:
        PTTestCase.tearDownClass()
