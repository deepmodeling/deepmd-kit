import unittest

from deepmd.common import (
    add_data_requirement,
    data_requirement,
)


class TestDataRequirement(unittest.TestCase):
    def test_add(self):
        add_data_requirement("test", 3)
        self.assertEqual(data_requirement["test"]["ndof"], 3)
        self.assertEqual(data_requirement["test"]["atomic"], False)
        self.assertEqual(data_requirement["test"]["must"], False)
        self.assertEqual(data_requirement["test"]["high_prec"], False)
        self.assertEqual(data_requirement["test"]["repeat"], 1)
        self.assertEqual(data_requirement["test"]["default"], 0.0)
