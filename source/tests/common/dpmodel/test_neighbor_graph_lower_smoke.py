# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest


class TestPrASymbolsExist(unittest.TestCase):
    def test_edge_env_mat_importable(self) -> None:
        from deepmd.dpmodel.utils.neighbor_graph import edge_env_mat

        self.assertTrue(callable(edge_env_mat))
