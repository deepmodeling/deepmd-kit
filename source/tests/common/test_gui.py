# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from dpgui import (
    generate_dpgui_templates,
)


class TestDPGUI(unittest.TestCase):
    def test_dpgui_entrypoints(self) -> None:
        self.assertTrue(len(generate_dpgui_templates()) > 0)
