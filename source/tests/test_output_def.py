# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd_utils.model_format import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)


class TestDef(unittest.TestCase):
    def test_model_output_def(self):
        defs = [
            OutputVariableDef("energy", [1], True, True),
            OutputVariableDef("dos", [10], True, False),
            OutputVariableDef("foo", [3], False, False),
        ]
        # fitting definition
        fd = FittingOutputDef(defs)
        expected_keys = ["energy", "dos", "foo"]
        self.assertEqual(
            set(expected_keys),
            set(fd.keys()),
        )
        # shape
        self.assertEqual(fd["energy"].shape, [1])
        self.assertEqual(fd["dos"].shape, [10])
        self.assertEqual(fd["foo"].shape, [3])
        # atomic
        self.assertEqual(fd["energy"].atomic, True)
        self.assertEqual(fd["dos"].atomic, True)
        self.assertEqual(fd["foo"].atomic, True)
        # reduce
        self.assertEqual(fd["energy"].reduciable, True)
        self.assertEqual(fd["dos"].reduciable, True)
        self.assertEqual(fd["foo"].reduciable, False)
        # derivative
        self.assertEqual(fd["energy"].differentiable, True)
        self.assertEqual(fd["dos"].differentiable, False)
        self.assertEqual(fd["foo"].differentiable, False)
        # model definition
        md = ModelOutputDef(fd)
        expected_keys = [
            "energy",
            "dos",
            "foo",
            "energy_redu",
            "energy_derv_r",
            "energy_derv_c",
            "dos_redu",
        ]
        self.assertEqual(
            set(expected_keys),
            set(md.keys()),
        )
        for kk in expected_keys:
            self.assertEqual(md[kk].name, kk)
        # shape
        self.assertEqual(md["energy"].shape, [1])
        self.assertEqual(md["dos"].shape, [10])
        self.assertEqual(md["foo"].shape, [3])
        self.assertEqual(md["energy_redu"].shape, [1])
        self.assertEqual(md["energy_derv_r"].shape, [1, 3])
        self.assertEqual(md["energy_derv_c"].shape, [1, 3, 3])
        # atomic
        self.assertEqual(md["energy"].atomic, True)
        self.assertEqual(md["dos"].atomic, True)
        self.assertEqual(md["foo"].atomic, True)
        self.assertEqual(md["energy_redu"].atomic, False)
        self.assertEqual(md["energy_derv_r"].atomic, True)
        self.assertEqual(md["energy_derv_c"].atomic, False)

    def test_raise_no_redu_deriv(self):
        with self.assertRaises(ValueError) as context:
            (OutputVariableDef("energy", [1], False, True),)
