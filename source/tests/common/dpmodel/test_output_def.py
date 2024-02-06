# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    List,
)

import numpy as np

from deepmd.dpmodel import (
    FittingOutputDef,
    ModelOutputDef,
    NativeOP,
    OutputVariableDef,
    fitting_check_output,
    model_check_output,
)
from deepmd.dpmodel.output_def import (
    OutputVariableCategory,
    OutputVariableOperation,
    apply_operation,
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
            "energy_derv_c_redu",
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
        self.assertEqual(md["energy_derv_c_redu"].shape, [1, 3, 3])
        # atomic
        self.assertEqual(md["energy"].atomic, True)
        self.assertEqual(md["dos"].atomic, True)
        self.assertEqual(md["foo"].atomic, True)
        self.assertEqual(md["energy_redu"].atomic, False)
        self.assertEqual(md["energy_derv_r"].atomic, True)
        self.assertEqual(md["energy_derv_c"].atomic, True)
        self.assertEqual(md["energy_derv_c_redu"].atomic, False)
        # category
        self.assertEqual(md["energy"].category, OutputVariableCategory.OUT)
        self.assertEqual(md["dos"].category, OutputVariableCategory.OUT)
        self.assertEqual(md["foo"].category, OutputVariableCategory.OUT)
        self.assertEqual(md["energy_redu"].category, OutputVariableCategory.REDU)
        self.assertEqual(md["energy_derv_r"].category, OutputVariableCategory.DERV_R)
        self.assertEqual(md["energy_derv_c"].category, OutputVariableCategory.DERV_C)
        self.assertEqual(
            md["energy_derv_c_redu"].category, OutputVariableCategory.DERV_C_REDU
        )
        # flag
        self.assertEqual(md["energy"].category & OutputVariableOperation.REDU, 0)
        self.assertEqual(md["energy"].category & OutputVariableOperation.DERV_R, 0)
        self.assertEqual(md["energy"].category & OutputVariableOperation.DERV_C, 0)
        self.assertEqual(md["dos"].category & OutputVariableOperation.REDU, 0)
        self.assertEqual(md["dos"].category & OutputVariableOperation.DERV_R, 0)
        self.assertEqual(md["dos"].category & OutputVariableOperation.DERV_C, 0)
        self.assertEqual(md["foo"].category & OutputVariableOperation.REDU, 0)
        self.assertEqual(md["foo"].category & OutputVariableOperation.DERV_R, 0)
        self.assertEqual(md["foo"].category & OutputVariableOperation.DERV_C, 0)
        self.assertEqual(
            md["energy_redu"].category & OutputVariableOperation.REDU,
            OutputVariableOperation.REDU,
        )
        self.assertEqual(md["energy_redu"].category & OutputVariableOperation.DERV_R, 0)
        self.assertEqual(md["energy_redu"].category & OutputVariableOperation.DERV_C, 0)
        self.assertEqual(md["energy_derv_r"].category & OutputVariableOperation.REDU, 0)
        self.assertEqual(
            md["energy_derv_r"].category & OutputVariableOperation.DERV_R,
            OutputVariableOperation.DERV_R,
        )
        self.assertEqual(
            md["energy_derv_r"].category & OutputVariableOperation.DERV_C, 0
        )
        self.assertEqual(md["energy_derv_c"].category & OutputVariableOperation.REDU, 0)
        self.assertEqual(
            md["energy_derv_c"].category & OutputVariableOperation.DERV_R, 0
        )
        self.assertEqual(
            md["energy_derv_c"].category & OutputVariableOperation.DERV_C,
            OutputVariableOperation.DERV_C,
        )
        self.assertEqual(
            md["energy_derv_c_redu"].category & OutputVariableOperation.REDU,
            OutputVariableOperation.REDU,
        )
        self.assertEqual(
            md["energy_derv_c_redu"].category & OutputVariableOperation.DERV_R, 0
        )
        self.assertEqual(
            md["energy_derv_c_redu"].category & OutputVariableOperation.DERV_C,
            OutputVariableOperation.DERV_C,
        )

        # apply_operation
        self.assertEqual(
            apply_operation(md["energy"], OutputVariableOperation.REDU),
            md["energy_redu"].category,
        )
        self.assertEqual(
            apply_operation(md["energy"], OutputVariableOperation.DERV_R),
            md["energy_derv_r"].category,
        )
        self.assertEqual(
            apply_operation(md["energy"], OutputVariableOperation.DERV_C),
            md["energy_derv_c"].category,
        )
        self.assertEqual(
            apply_operation(md["energy_derv_c"], OutputVariableOperation.REDU),
            md["energy_derv_c_redu"].category,
        )
        # raise ValueError
        with self.assertRaises(ValueError):
            apply_operation(md["energy_redu"], OutputVariableOperation.REDU)
        with self.assertRaises(ValueError):
            apply_operation(md["energy_derv_c"], OutputVariableOperation.DERV_C)
        with self.assertRaises(ValueError):
            apply_operation(md["energy_derv_c_redu"], OutputVariableOperation.REDU)
        # hession
        hession_cat = apply_operation(
            md["energy_derv_r"], OutputVariableOperation.SEC_DERV_R
        )
        self.assertEqual(
            hession_cat & OutputVariableOperation.DERV_R, OutputVariableOperation.DERV_R
        )
        self.assertEqual(
            hession_cat & OutputVariableOperation.SEC_DERV_R,
            OutputVariableOperation.SEC_DERV_R,
        )
        self.assertEqual(hession_cat, OutputVariableCategory.DERV_R_DERV_R)
        hession_vardef = OutputVariableDef(
            "energy_derv_r_derv_r", [1], False, False, category=hession_cat
        )
        with self.assertRaises(ValueError):
            apply_operation(hession_vardef, OutputVariableOperation.DERV_R)

    def test_raise_no_redu_deriv(self):
        with self.assertRaises(ValueError) as context:
            (OutputVariableDef("energy", [1], False, True),)

    def test_raise_redu_not_atomic(self):
        with self.assertRaises(ValueError) as context:
            (OutputVariableDef("energy", [1], True, False, atomic=False),)

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
