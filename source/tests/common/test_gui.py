# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import unittest

from dpgui import (
    generate_dpgui_templates,
)


class TestDPGUI(unittest.TestCase):
    def test_dpgui_entrypoints(self):
        self.assertTrue(len(generate_dpgui_templates()) > 0)
