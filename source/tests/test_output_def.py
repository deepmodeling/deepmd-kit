# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    List,
)

import numpy as np

from deepmd_utils.model_format import (
    FittingOutputDef,
    ModelOutputDef,
    NativeOP,
    OutputVariableDef,
    fitting_check_output,
    model_check_output,
)
from deepmd_utils.model_format.output_def import (
    check_var,
)


class VariableDef:
    def __init__(
        self,
        name: str,
        shape: List[int],
        atomic: bool = True,
    ):
        self.name = name
        self.shape = list(shape)
        self.atomic = atomic


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
        # reduce
        self.assertEqual(md["energy"].reduciable, True)
        self.assertEqual(md["dos"].reduciable, True)
        self.assertEqual(md["foo"].reduciable, False)
        # derivative
        self.assertEqual(md["energy"].differentiable, True)
        self.assertEqual(md["dos"].differentiable, False)
        self.assertEqual(md["foo"].differentiable, False)
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
        self.assertEqual(md["energy_derv_c"].atomic, True)

    def test_raise_no_redu_deriv(self):
        with self.assertRaises(ValueError) as context:
            (OutputVariableDef("energy", [1], False, True),)

    def test_model_decorator(self):
        nf = 2
        nloc = 3
        nall = 4

        @model_check_output
        class Foo(NativeOP):
            def output_def(self):
                defs = [
                    OutputVariableDef("energy", [1], True, True),
                ]
                return ModelOutputDef(FittingOutputDef(defs))

            def call(self):
                return {
                    "energy": np.zeros([nf, nloc, 1]),
                    "energy_redu": np.zeros([nf, 1]),
                    "energy_derv_r": np.zeros([nf, nall, 1, 3]),
                    "energy_derv_c": np.zeros([nf, nall, 1, 3, 3]),
                }

        ff = Foo()
        ff()

    def test_model_decorator_keyerror(self):
        nf = 2
        nloc = 3
        nall = 4

        @model_check_output
        class Foo(NativeOP):
            def __init__(self):
                super().__init__()

            def output_def(self):
                defs = [
                    OutputVariableDef("energy", [1], True, True),
                ]
                return ModelOutputDef(FittingOutputDef(defs))

            def call(self):
                return {
                    "energy": np.zeros([nf, nloc, 1]),
                    "energy_redu": np.zeros([nf, 1]),
                    "energy_derv_c": np.zeros([nf, nall, 1, 3, 3]),
                }

        ff = Foo()
        with self.assertRaises(KeyError) as context:
            ff()
            self.assertIn("energy_derv_r", context.exception)

    def test_model_decorator_shapeerror(self):
        nf = 2
        nloc = 3
        nall = 4

        @model_check_output
        class Foo(NativeOP):
            def __init__(
                self,
                shape_rd=[nf, 1],
                shape_dr=[nf, nall, 1, 3],
            ):
                self.shape_rd, self.shape_dr = shape_rd, shape_dr

            def output_def(self):
                defs = [
                    OutputVariableDef("energy", [1], True, True),
                ]
                return ModelOutputDef(FittingOutputDef(defs))

            def call(self):
                return {
                    "energy": np.zeros([nf, nloc, 1]),
                    "energy_redu": np.zeros(self.shape_rd),
                    "energy_derv_r": np.zeros(self.shape_dr),
                    "energy_derv_c": np.zeros([nf, nall, 1, 3, 3]),
                }

        ff = Foo()
        ff()
        # shape of reduced energy
        with self.assertRaises(ValueError) as context:
            ff = Foo(shape_rd=[nf, nloc, 1])
            ff()
            self.assertIn("not matching", context.exception)
        with self.assertRaises(ValueError) as context:
            ff = Foo(shape_rd=[nf, 2])
            ff()
            self.assertIn("not matching", context.exception)
        # shape of dr
        with self.assertRaises(ValueError) as context:
            ff = Foo(shape_dr=[nf, nloc, 1])
            ff()
            self.assertIn("not matching", context.exception)
        with self.assertRaises(ValueError) as context:
            ff = Foo(shape_dr=[nf, nloc, 1, 3, 3])
            ff()
            self.assertIn("not matching", context.exception)
        with self.assertRaises(ValueError) as context:
            ff = Foo(shape_dr=[nf, nloc, 1, 4])
            ff()
            self.assertIn("not matching", context.exception)

    def test_fitting_decorator(self):
        nf = 2
        nloc = 3
        nall = 4

        @fitting_check_output
        class Foo(NativeOP):
            def output_def(self):
                defs = [
                    OutputVariableDef("energy", [1], True, True),
                ]
                return FittingOutputDef(defs)

            def call(self):
                return {
                    "energy": np.zeros([nf, nloc, 1]),
                }

        ff = Foo()
        ff()

    def test_fitting_decorator_shapeerror(self):
        nf = 2
        nloc = 3

        @fitting_check_output
        class Foo(NativeOP):
            def __init__(
                self,
                shape=[nf, nloc, 1],
            ):
                self.shape = shape

            def output_def(self):
                defs = [
                    OutputVariableDef("energy", [1], True, True),
                ]
                return FittingOutputDef(defs)

            def call(self):
                return {
                    "energy": np.zeros(self.shape),
                }

        ff = Foo()
        ff()
        # shape of reduced energy
        with self.assertRaises(ValueError) as context:
            ff = Foo(shape=[nf, 1])
            ff()
            self.assertIn("not matching", context.exception)
        with self.assertRaises(ValueError) as context:
            ff = Foo(shape=[nf, nloc, 2])
            ff()
            self.assertIn("not matching", context.exception)

    def test_check_var(self):
        var_def = VariableDef("foo", [2, 3], atomic=True)
        with self.assertRaises(ValueError) as context:
            check_var(np.zeros([2, 3, 4, 5, 6]), var_def)
            self.assertIn("length not matching", context.exception)
        with self.assertRaises(ValueError) as context:
            check_var(np.zeros([2, 3, 4, 5]), var_def)
            self.assertIn("shape not matching", context.exception)
        check_var(np.zeros([2, 3, 2, 3]), var_def)

        var_def = VariableDef("foo", [2, 3], atomic=False)
        with self.assertRaises(ValueError) as context:
            check_var(np.zeros([2, 3, 4, 5]), var_def)
            self.assertIn("length not matching", context.exception)
        with self.assertRaises(ValueError) as context:
            check_var(np.zeros([2, 3, 4]), var_def)
            self.assertIn("shape not matching", context.exception)
        check_var(np.zeros([2, 2, 3]), var_def)

        var_def = VariableDef("foo", [2, -1], atomic=True)
        with self.assertRaises(ValueError) as context:
            check_var(np.zeros([2, 3, 4, 5, 6]), var_def)
            self.assertIn("length not matching", context.exception)
        with self.assertRaises(ValueError) as context:
            check_var(np.zeros([2, 3, 4, 5]), var_def)
            self.assertIn("shape not matching", context.exception)
        check_var(np.zeros([2, 3, 2, 8]), var_def)

        var_def = VariableDef("foo", [2, -1], atomic=False)
        with self.assertRaises(ValueError) as context:
            check_var(np.zeros([2, 3, 4, 5]), var_def)
            self.assertIn("length not matching", context.exception)
        with self.assertRaises(ValueError) as context:
            check_var(np.zeros([2, 3, 4]), var_def)
            self.assertIn("shape not matching", context.exception)
        check_var(np.zeros([2, 2, 8]), var_def)
