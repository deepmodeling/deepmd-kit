# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

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
        shape: list[int],
        atomic: bool = True,
    ) -> None:
        self.name = name
        self.shape = list(shape)
        self.atomic = atomic


class TestDef(unittest.TestCase):
    def test_model_output_def(self) -> None:
        defs = [
            OutputVariableDef(
                "energy",
                [1],
                reducible=True,
                r_differentiable=True,
                c_differentiable=True,
                atomic=True,
                r_hessian=False,
            ),
            OutputVariableDef(
                "energy2",
                [1],
                reducible=True,
                r_differentiable=True,
                c_differentiable=True,
                atomic=True,
                r_hessian=True,
            ),
            OutputVariableDef(
                "energy3",
                [1],
                reducible=True,
                r_differentiable=True,
                c_differentiable=True,
                atomic=True,
                magnetic=True,
            ),
            OutputVariableDef(
                "dos",
                [10],
                reducible=True,
                r_differentiable=False,
                c_differentiable=False,
                atomic=True,
            ),
            OutputVariableDef(
                "foo",
                [3],
                reducible=False,
                r_differentiable=False,
                c_differentiable=False,
                atomic=True,
            ),
            OutputVariableDef(
                "gap",
                [13],
                reducible=True,
                r_differentiable=False,
                c_differentiable=False,
                atomic=True,
                intensive=True,
            ),
        ]
        # fitting definition
        fd = FittingOutputDef(defs)
        expected_keys = ["energy", "energy2", "energy3", "dos", "foo", "gap"]
        self.assertEqual(
            set(expected_keys),
            set(fd.keys()),
        )
        # shape
        self.assertEqual(fd["energy"].shape, [1])
        self.assertEqual(fd["energy2"].shape, [1])
        self.assertEqual(fd["energy3"].shape, [1])
        self.assertEqual(fd["dos"].shape, [10])
        self.assertEqual(fd["foo"].shape, [3])
        self.assertEqual(fd["gap"].shape, [13])
        # atomic
        self.assertEqual(fd["energy"].atomic, True)
        self.assertEqual(fd["energy2"].atomic, True)
        self.assertEqual(fd["energy3"].atomic, True)
        self.assertEqual(fd["dos"].atomic, True)
        self.assertEqual(fd["foo"].atomic, True)
        self.assertEqual(fd["gap"].atomic, True)
        # reduce
        self.assertEqual(fd["energy"].reducible, True)
        self.assertEqual(fd["energy2"].reducible, True)
        self.assertEqual(fd["energy3"].reducible, True)
        self.assertEqual(fd["dos"].reducible, True)
        self.assertEqual(fd["foo"].reducible, False)
        self.assertEqual(fd["gap"].reducible, True)
        # derivative
        self.assertEqual(fd["energy"].r_differentiable, True)
        self.assertEqual(fd["energy"].c_differentiable, True)
        self.assertEqual(fd["energy"].r_hessian, False)
        self.assertEqual(fd["energy2"].r_differentiable, True)
        self.assertEqual(fd["energy2"].c_differentiable, True)
        self.assertEqual(fd["energy2"].r_hessian, True)
        self.assertEqual(fd["energy3"].r_differentiable, True)
        self.assertEqual(fd["energy3"].c_differentiable, True)
        self.assertEqual(fd["energy3"].r_hessian, False)
        self.assertEqual(fd["dos"].r_differentiable, False)
        self.assertEqual(fd["foo"].r_differentiable, False)
        self.assertEqual(fd["gap"].r_differentiable, False)
        self.assertEqual(fd["dos"].c_differentiable, False)
        self.assertEqual(fd["foo"].c_differentiable, False)
        self.assertEqual(fd["gap"].c_differentiable, False)
        # magnetic
        self.assertEqual(fd["energy"].magnetic, False)
        self.assertEqual(fd["energy2"].magnetic, False)
        self.assertEqual(fd["energy3"].magnetic, True)
        self.assertEqual(fd["dos"].magnetic, False)
        self.assertEqual(fd["foo"].magnetic, False)
        self.assertEqual(fd["gap"].magnetic, False)
        # model definition
        md = ModelOutputDef(fd)
        expected_keys = [
            "energy",
            "energy2",
            "energy3",
            "dos",
            "foo",
            "energy_redu",
            "energy_derv_r",
            "energy_derv_c",
            "energy_derv_c_redu",
            "energy2_redu",
            "energy2_derv_r",
            "energy2_derv_r_derv_r",
            "energy2_derv_c",
            "energy2_derv_c_redu",
            "energy3_redu",
            "energy3_derv_r",
            "energy3_derv_c",
            "energy3_derv_c_redu",
            "energy3_derv_r_mag",
            "energy3_derv_c_mag",
            "dos_redu",
            "mask",
            "mask_mag",
            "gap",
            "gap_redu",
        ]
        self.assertEqual(
            set(expected_keys),
            set(md.keys()),
        )
        for kk in expected_keys:
            self.assertEqual(md[kk].name, kk)
        # reduce
        self.assertEqual(md["energy"].reducible, True)
        self.assertEqual(md["energy2"].reducible, True)
        self.assertEqual(md["energy3"].reducible, True)
        self.assertEqual(md["dos"].reducible, True)
        self.assertEqual(md["foo"].reducible, False)
        self.assertEqual(md["gap"].reducible, True)
        # derivative
        self.assertEqual(md["energy"].r_differentiable, True)
        self.assertEqual(md["energy"].c_differentiable, True)
        self.assertEqual(md["energy"].r_hessian, False)
        self.assertEqual(md["energy2"].r_differentiable, True)
        self.assertEqual(md["energy2"].c_differentiable, True)
        self.assertEqual(md["energy2"].r_hessian, True)
        self.assertEqual(md["energy3"].r_differentiable, True)
        self.assertEqual(md["energy3"].c_differentiable, True)
        self.assertEqual(md["energy3"].r_hessian, False)
        self.assertEqual(md["dos"].r_differentiable, False)
        self.assertEqual(md["foo"].r_differentiable, False)
        self.assertEqual(md["gap"].r_differentiable, False)
        self.assertEqual(md["dos"].c_differentiable, False)
        self.assertEqual(md["foo"].c_differentiable, False)
        self.assertEqual(md["gap"].c_differentiable, False)
        # shape
        self.assertEqual(md["mask"].shape, [1])
        self.assertEqual(md["mask_mag"].shape, [1])
        self.assertEqual(md["energy"].shape, [1])
        self.assertEqual(md["energy2"].shape, [1])
        self.assertEqual(md["energy3"].shape, [1])
        self.assertEqual(md["dos"].shape, [10])
        self.assertEqual(md["foo"].shape, [3])
        self.assertEqual(md["energy_redu"].shape, [1])
        self.assertEqual(md["energy_derv_r"].shape, [1, 3])
        self.assertEqual(md["energy_derv_c"].shape, [1, 9])
        self.assertEqual(md["energy_derv_c_redu"].shape, [1, 9])
        self.assertEqual(md["energy2_redu"].shape, [1])
        self.assertEqual(md["energy2_derv_r"].shape, [1, 3])
        self.assertEqual(md["energy2_derv_c"].shape, [1, 9])
        self.assertEqual(md["energy2_derv_c_redu"].shape, [1, 9])
        self.assertEqual(md["energy2_derv_r_derv_r"].shape, [1, 3, 3])
        self.assertEqual(md["energy3_derv_r"].shape, [1, 3])
        self.assertEqual(md["energy3_derv_c"].shape, [1, 9])
        self.assertEqual(md["energy3_derv_c_redu"].shape, [1, 9])
        self.assertEqual(md["energy3_derv_r_mag"].shape, [1, 3])
        self.assertEqual(md["energy3_derv_c_mag"].shape, [1, 9])
        self.assertEqual(md["gap"].shape, [13])
        self.assertEqual(md["gap_redu"].shape, [13])
        # atomic
        self.assertEqual(md["energy"].atomic, True)
        self.assertEqual(md["energy2"].atomic, True)
        self.assertEqual(md["dos"].atomic, True)
        self.assertEqual(md["foo"].atomic, True)
        self.assertEqual(md["energy_redu"].atomic, False)
        self.assertEqual(md["energy_derv_r"].atomic, True)
        self.assertEqual(md["energy_derv_c"].atomic, True)
        self.assertEqual(md["energy_derv_c_redu"].atomic, False)
        self.assertEqual(md["energy2_redu"].atomic, False)
        self.assertEqual(md["energy2_derv_r"].atomic, True)
        self.assertEqual(md["energy2_derv_c"].atomic, True)
        self.assertEqual(md["energy2_derv_c_redu"].atomic, False)
        self.assertEqual(md["energy2_derv_r_derv_r"].atomic, True)
        self.assertEqual(md["energy3_redu"].atomic, False)
        self.assertEqual(md["energy3_derv_r"].atomic, True)
        self.assertEqual(md["energy3_derv_c"].atomic, True)
        self.assertEqual(md["energy3_derv_r_mag"].atomic, True)
        self.assertEqual(md["energy3_derv_c_mag"].atomic, True)
        self.assertEqual(md["energy3_derv_c_redu"].atomic, False)
        self.assertEqual(md["gap"].atomic, True)
        self.assertEqual(md["gap_redu"].atomic, False)
        # category
        self.assertEqual(md["mask"].category, OutputVariableCategory.OUT)
        self.assertEqual(md["mask_mag"].category, OutputVariableCategory.OUT)
        self.assertEqual(md["energy"].category, OutputVariableCategory.OUT)
        self.assertEqual(md["energy2"].category, OutputVariableCategory.OUT)
        self.assertEqual(md["energy3"].category, OutputVariableCategory.OUT)
        self.assertEqual(md["dos"].category, OutputVariableCategory.OUT)
        self.assertEqual(md["foo"].category, OutputVariableCategory.OUT)
        self.assertEqual(md["energy_redu"].category, OutputVariableCategory.REDU)
        self.assertEqual(md["energy_derv_r"].category, OutputVariableCategory.DERV_R)
        self.assertEqual(md["energy_derv_c"].category, OutputVariableCategory.DERV_C)
        self.assertEqual(
            md["energy_derv_c_redu"].category, OutputVariableCategory.DERV_C_REDU
        )
        self.assertEqual(md["energy2_redu"].category, OutputVariableCategory.REDU)
        self.assertEqual(md["energy2_derv_r"].category, OutputVariableCategory.DERV_R)
        self.assertEqual(md["energy2_derv_c"].category, OutputVariableCategory.DERV_C)
        self.assertEqual(
            md["energy2_derv_c_redu"].category, OutputVariableCategory.DERV_C_REDU
        )
        self.assertEqual(
            md["energy2_derv_r_derv_r"].category, OutputVariableCategory.DERV_R_DERV_R
        )
        self.assertEqual(md["energy3_redu"].category, OutputVariableCategory.REDU)
        self.assertEqual(md["energy3_derv_r"].category, OutputVariableCategory.DERV_R)
        self.assertEqual(md["energy3_derv_c"].category, OutputVariableCategory.DERV_C)
        self.assertEqual(
            md["energy3_derv_c_redu"].category, OutputVariableCategory.DERV_C_REDU
        )
        self.assertEqual(
            md["energy3_derv_r_mag"].category, OutputVariableCategory.DERV_R
        )
        self.assertEqual(
            md["energy3_derv_c_mag"].category, OutputVariableCategory.DERV_C
        )
        self.assertEqual(md["gap"].category, OutputVariableCategory.OUT)
        self.assertEqual(md["gap_redu"].category, OutputVariableCategory.REDU)
        # flag
        OVO = OutputVariableOperation
        self.assertEqual(md["energy"].category & OVO.REDU, 0)
        self.assertEqual(md["energy"].category & OVO.DERV_R, 0)
        self.assertEqual(md["energy"].category & OVO.DERV_C, 0)
        self.assertEqual(md["energy2"].category & OVO.REDU, 0)
        self.assertEqual(md["energy2"].category & OVO.DERV_R, 0)
        self.assertEqual(md["energy2"].category & OVO.DERV_C, 0)
        self.assertEqual(md["energy3"].category & OVO.REDU, 0)
        self.assertEqual(md["energy3"].category & OVO.DERV_R, 0)
        self.assertEqual(md["energy3"].category & OVO.DERV_C, 0)
        self.assertEqual(md["dos"].category & OVO.REDU, 0)
        self.assertEqual(md["dos"].category & OVO.DERV_R, 0)
        self.assertEqual(md["dos"].category & OVO.DERV_C, 0)
        self.assertEqual(md["foo"].category & OVO.REDU, 0)
        self.assertEqual(md["foo"].category & OVO.DERV_R, 0)
        self.assertEqual(md["foo"].category & OVO.DERV_C, 0)
        self.assertEqual(md["gap"].category & OVO.REDU, 0)
        self.assertEqual(md["gap"].category & OVO.DERV_R, 0)
        self.assertEqual(md["gap"].category & OVO.DERV_C, 0)
        # flag: energy
        self.assertEqual(
            md["energy_redu"].category & OVO.REDU,
            OVO.REDU,
        )
        self.assertEqual(md["energy_redu"].category & OVO.DERV_R, 0)
        self.assertEqual(md["energy_redu"].category & OVO.DERV_C, 0)
        self.assertEqual(md["energy_derv_r"].category & OVO.REDU, 0)
        self.assertEqual(
            md["energy_derv_r"].category & OVO.DERV_R,
            OVO.DERV_R,
        )
        self.assertEqual(md["energy_derv_r"].category & OVO.DERV_C, 0)
        self.assertEqual(md["energy_derv_c"].category & OVO.REDU, 0)
        self.assertEqual(md["energy_derv_c"].category & OVO.DERV_R, 0)
        self.assertEqual(
            md["energy_derv_c"].category & OVO.DERV_C,
            OVO.DERV_C,
        )
        self.assertEqual(
            md["energy_derv_c_redu"].category & OVO.REDU,
            OVO.REDU,
        )
        self.assertEqual(md["energy_derv_c_redu"].category & OVO.DERV_R, 0)
        self.assertEqual(
            md["energy_derv_c_redu"].category & OVO.DERV_C,
            OVO.DERV_C,
        )
        # flag: energy2
        kk = "energy2_redu"
        self.assertEqual(md[kk].category & OVO.REDU, OVO.REDU)
        self.assertEqual(md[kk].category & OVO.DERV_R, 0)
        self.assertEqual(md[kk].category & OVO.DERV_C, 0)
        self.assertEqual(md[kk].category & OVO._SEC_DERV_R, 0)
        kk = "energy2_derv_r"
        self.assertEqual(md[kk].category & OVO.REDU, 0)
        self.assertEqual(md[kk].category & OVO.DERV_R, OVO.DERV_R)
        self.assertEqual(md[kk].category & OVO.DERV_C, 0)
        self.assertEqual(md[kk].category & OVO._SEC_DERV_R, 0)
        kk = "energy2_derv_c"
        self.assertEqual(md[kk].category & OVO.REDU, 0)
        self.assertEqual(md[kk].category & OVO.DERV_R, 0)
        self.assertEqual(md[kk].category & OVO.DERV_C, OVO.DERV_C)
        self.assertEqual(md[kk].category & OVO._SEC_DERV_R, 0)
        kk = "energy2_derv_c_redu"
        self.assertEqual(md[kk].category & OVO.REDU, OVO.REDU)
        self.assertEqual(md[kk].category & OVO.DERV_R, 0)
        self.assertEqual(md[kk].category & OVO.DERV_C, OVO.DERV_C)
        self.assertEqual(md[kk].category & OVO._SEC_DERV_R, 0)
        kk = "energy2_derv_r_derv_r"
        self.assertEqual(md[kk].category & OVO.REDU, 0)
        self.assertEqual(md[kk].category & OVO.DERV_R, OVO.DERV_R)
        self.assertEqual(md[kk].category & OVO.DERV_C, 0)
        self.assertEqual(md[kk].category & OVO._SEC_DERV_R, OVO._SEC_DERV_R)
        # flag: energy3
        self.assertEqual(
            md["energy3_redu"].category & OVO.REDU,
            OVO.REDU,
        )
        self.assertEqual(md["energy3_redu"].category & OVO.DERV_R, 0)
        self.assertEqual(md["energy3_redu"].category & OVO.DERV_C, 0)
        self.assertEqual(md["energy3_derv_r"].category & OVO.REDU, 0)
        self.assertEqual(
            md["energy3_derv_r"].category & OVO.DERV_R,
            OVO.DERV_R,
        )
        self.assertEqual(md["energy3_derv_r"].category & OVO.DERV_C, 0)
        self.assertEqual(md["energy3_derv_c"].category & OVO.REDU, 0)
        self.assertEqual(md["energy3_derv_c"].category & OVO.DERV_R, 0)
        self.assertEqual(
            md["energy3_derv_c"].category & OVO.DERV_C,
            OVO.DERV_C,
        )
        self.assertEqual(
            md["energy3_derv_c_redu"].category & OVO.REDU,
            OVO.REDU,
        )
        self.assertEqual(md["energy3_derv_c_redu"].category & OVO.DERV_R, 0)
        self.assertEqual(
            md["energy3_derv_c_redu"].category & OVO.DERV_C,
            OVO.DERV_C,
        )
        self.assertEqual(md["energy3_derv_r_mag"].category & OVO.REDU, 0)
        self.assertEqual(
            md["energy3_derv_r_mag"].category & OVO.DERV_R,
            OVO.DERV_R,
        )
        self.assertEqual(md["energy3_derv_r_mag"].category & OVO.DERV_C, 0)
        self.assertEqual(md["energy3_derv_c_mag"].category & OVO.REDU, 0)
        self.assertEqual(md["energy3_derv_c_mag"].category & OVO.DERV_R, 0)
        self.assertEqual(
            md["energy3_derv_c_mag"].category & OVO.DERV_C,
            OVO.DERV_C,
        )
        # apply_operation: energy
        self.assertEqual(
            apply_operation(md["energy"], OVO.REDU),
            md["energy_redu"].category,
        )
        self.assertEqual(
            apply_operation(md["energy"], OVO.DERV_R),
            md["energy_derv_r"].category,
        )
        self.assertEqual(
            apply_operation(md["energy"], OVO.DERV_C),
            md["energy_derv_c"].category,
        )
        self.assertEqual(
            apply_operation(md["energy_derv_c"], OVO.REDU),
            md["energy_derv_c_redu"].category,
        )
        # apply_operation: energy2
        self.assertEqual(
            apply_operation(md["energy2"], OVO.REDU),
            md["energy2_redu"].category,
        )
        self.assertEqual(
            apply_operation(md["energy2"], OVO.DERV_R),
            md["energy2_derv_r"].category,
        )
        self.assertEqual(
            apply_operation(md["energy2"], OVO.DERV_C),
            md["energy2_derv_c"].category,
        )
        self.assertEqual(
            apply_operation(md["energy2_derv_c"], OVO.REDU),
            md["energy2_derv_c_redu"].category,
        )
        self.assertEqual(
            apply_operation(md["energy2_derv_r"], OVO.DERV_R),
            md["energy2_derv_r_derv_r"].category,
        )
        # apply_operation: energy3
        self.assertEqual(
            apply_operation(md["energy3"], OVO.REDU),
            md["energy3_redu"].category,
        )
        self.assertEqual(
            apply_operation(md["energy3"], OVO.DERV_R),
            md["energy3_derv_r"].category,
        )
        self.assertEqual(
            apply_operation(md["energy3"], OVO.DERV_C),
            md["energy3_derv_c"].category,
        )
        self.assertEqual(
            apply_operation(md["energy3_derv_c"], OVO.REDU),
            md["energy3_derv_c_redu"].category,
        )
        self.assertEqual(
            apply_operation(md["energy3"], OVO.DERV_R),
            md["energy3_derv_r_mag"].category,
        )
        self.assertEqual(
            apply_operation(md["energy3"], OVO.DERV_C),
            md["energy3_derv_c_mag"].category,
        )
        # raise ValueError
        with self.assertRaises(ValueError):
            apply_operation(md["energy_redu"], OVO.REDU)
        with self.assertRaises(ValueError):
            apply_operation(md["energy_derv_c"], OVO.DERV_C)
        with self.assertRaises(ValueError):
            apply_operation(md["energy_derv_c_redu"], OVO.REDU)
        # raise ValueError
        with self.assertRaises(ValueError):
            apply_operation(md["energy2_redu"], OVO.REDU)
        with self.assertRaises(ValueError):
            apply_operation(md["energy2_derv_c"], OVO.DERV_C)
        with self.assertRaises(ValueError):
            apply_operation(md["energy2_derv_c_redu"], OVO.REDU)
        with self.assertRaises(ValueError):
            apply_operation(md["energy2_derv_r_derv_r"], OVO.DERV_R)
        # raise ValueError
        with self.assertRaises(ValueError):
            apply_operation(md["energy3_redu"], OVO.REDU)
        with self.assertRaises(ValueError):
            apply_operation(md["energy3_derv_c"], OVO.DERV_C)
        with self.assertRaises(ValueError):
            apply_operation(md["energy3_derv_c_redu"], OVO.REDU)
        with self.assertRaises(ValueError):
            apply_operation(md["energy3_derv_c_mag"], OVO.DERV_C)
        # hession
        hession_cat = apply_operation(md["energy_derv_r"], OVO.DERV_R)
        self.assertEqual(hession_cat & OVO.DERV_R, OVO.DERV_R)
        self.assertEqual(
            hession_cat & OVO._SEC_DERV_R,
            OVO._SEC_DERV_R,
        )
        self.assertEqual(hession_cat, OutputVariableCategory.DERV_R_DERV_R)
        hession_vardef = OutputVariableDef(
            "energy_derv_r_derv_r", [1], False, False, category=hession_cat
        )
        with self.assertRaises(ValueError):
            apply_operation(hession_vardef, OVO.DERV_R)

    def test_no_raise_no_redu_deriv(self) -> None:
        OutputVariableDef(
            "energy",
            [1],
            reducible=False,
            r_differentiable=True,
            c_differentiable=False,
        )

    def test_raise_requires_r_deriv(self) -> None:
        with self.assertRaises(ValueError) as context:
            OutputVariableDef(
                "energy",
                [1],
                reducible=True,
                r_differentiable=False,
                c_differentiable=True,
            )

    def test_raise_redu_not_atomic(self) -> None:
        with self.assertRaises(ValueError) as context:
            (OutputVariableDef("energy", [1], reducible=True, atomic=False),)

    def test_hessian_not_reducible(self) -> None:
        with self.assertRaises(ValueError) as context:
            (
                OutputVariableDef(
                    "energy",
                    [1],
                    reducible=False,
                    atomic=False,
                    r_differentiable=True,
                    r_hessian=True,
                ),
            )

    def test_hessian_not_r_differentiable(self) -> None:
        with self.assertRaises(ValueError) as context:
            (
                OutputVariableDef(
                    "energy",
                    [1],
                    reducible=True,
                    atomic=False,
                    r_differentiable=False,
                    r_hessian=True,
                ),
            )

    def test_energy_magnetic(self) -> None:
        with self.assertRaises(ValueError) as context:
            (
                OutputVariableDef(
                    "energy",
                    [1],
                    reducible=False,
                    atomic=False,
                    r_differentiable=True,
                    r_hessian=True,
                    magnetic=True,
                ),
            )

    def test_inten_requires_redu(self) -> None:
        with self.assertRaises(ValueError) as context:
            (
                OutputVariableDef(
                    "foo",
                    [20],
                    reducible=False,
                    atomic=True,
                    r_differentiable=False,
                    r_hessian=False,
                    magnetic=False,
                    intensive=True,
                ),
            )

    def test_model_decorator(self) -> None:
        nf = 2
        nloc = 3
        nall = 4

        @model_check_output
        class Foo(NativeOP):
            def output_def(self):
                defs = [
                    OutputVariableDef(
                        "energy",
                        [1],
                        reducible=True,
                        r_differentiable=True,
                        c_differentiable=True,
                    ),
                ]
                return ModelOutputDef(FittingOutputDef(defs))

            def call(self):
                return {
                    "energy": np.zeros([nf, nloc, 1]),
                    "energy_redu": np.zeros([nf, 1]),
                    "energy_derv_r": np.zeros([nf, nall, 1, 3]),
                    "energy_derv_c": np.zeros([nf, nall, 1, 9]),
                }

        ff = Foo()
        ff()

    def test_model_decorator_keyerror(self) -> None:
        nf = 2
        nloc = 3
        nall = 4

        @model_check_output
        class Foo(NativeOP):
            def __init__(self) -> None:
                super().__init__()

            def output_def(self):
                defs = [
                    OutputVariableDef(
                        "energy",
                        [1],
                        reducible=True,
                        r_differentiable=True,
                        c_differentiable=True,
                    ),
                ]
                return ModelOutputDef(FittingOutputDef(defs))

            def call(self):
                return {
                    "energy": np.zeros([nf, nloc, 1]),
                    "energy_redu": np.zeros([nf, 1]),
                    "energy_derv_c": np.zeros([nf, nall, 1, 9]),
                }

        ff = Foo()
        with self.assertRaises(KeyError) as context:
            ff()
            self.assertIn("energy_derv_r", context.exception)

    def test_model_decorator_shapeerror(self) -> None:
        nf = 2
        nloc = 3
        nall = 4

        @model_check_output
        class Foo(NativeOP):
            def __init__(
                self,
                shape_rd=[nf, 1],
                shape_dr=[nf, nall, 1, 3],
            ) -> None:
                self.shape_rd, self.shape_dr = shape_rd, shape_dr

            def output_def(self):
                defs = [
                    OutputVariableDef(
                        "energy",
                        [1],
                        reducible=True,
                        r_differentiable=True,
                        c_differentiable=True,
                    ),
                ]
                return ModelOutputDef(FittingOutputDef(defs))

            def call(self):
                return {
                    "energy": np.zeros([nf, nloc, 1]),
                    "energy_redu": np.zeros(self.shape_rd),
                    "energy_derv_r": np.zeros(self.shape_dr),
                    "energy_derv_c": np.zeros([nf, nall, 1, 9]),
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

    def test_fitting_decorator(self) -> None:
        nf = 2
        nloc = 3
        nall = 4

        @fitting_check_output
        class Foo(NativeOP):
            def output_def(self):
                defs = [
                    OutputVariableDef(
                        "energy",
                        [1],
                        reducible=True,
                        r_differentiable=True,
                        c_differentiable=True,
                    ),
                ]
                return FittingOutputDef(defs)

            def call(self):
                return {
                    "energy": np.zeros([nf, nloc, 1]),
                }

        ff = Foo()
        ff()

    def test_fitting_decorator_shapeerror(self) -> None:
        nf = 2
        nloc = 3

        @fitting_check_output
        class Foo(NativeOP):
            def __init__(
                self,
                shape=[nf, nloc, 1],
            ) -> None:
                self.shape = shape

            def output_def(self):
                defs = [
                    OutputVariableDef(
                        "energy",
                        [1],
                        reducible=True,
                        r_differentiable=True,
                        c_differentiable=True,
                    ),
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

    def test_check_var(self) -> None:
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

    def test_squeeze(self) -> None:
        out_var = OutputVariableDef("foo", [])
        out_var.squeeze(0)
        self.assertEqual(out_var.shape, [])
        out_var = OutputVariableDef("foo", [1])
        out_var.squeeze(0)
        self.assertEqual(out_var.shape, [])
        out_var = OutputVariableDef("foo", [1, 1])
        out_var.squeeze(0)
        self.assertEqual(out_var.shape, [1])
        out_var = OutputVariableDef("foo", [1, 3])
        out_var.squeeze(0)
        self.assertEqual(out_var.shape, [3])
        out_var = OutputVariableDef("foo", [3, 3])
        out_var.squeeze(0)
        self.assertEqual(out_var.shape, [3, 3])
