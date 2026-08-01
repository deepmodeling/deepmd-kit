# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test that Paddle freeze static signatures include fparam/aparam when used."""

import unittest

from deepmd.pd.utils.serialization import (
    _fparam_aparam_input_specs,
)


class _StubModel:
    def __init__(self, dim_fparam: int, dim_aparam: int) -> None:
        self._dim_fparam = dim_fparam
        self._dim_aparam = dim_aparam

    def get_dim_fparam(self) -> int:
        return self._dim_fparam

    def get_dim_aparam(self) -> int:
        return self._dim_aparam


class TestFparamAparamInputSpecs(unittest.TestCase):
    def test_absent_when_unused(self) -> None:
        fparam_spec, aparam_spec = _fparam_aparam_input_specs(_StubModel(0, 0))
        self.assertIsNone(fparam_spec)
        self.assertIsNone(aparam_spec)

    def test_present_when_used(self) -> None:
        fparam_spec, aparam_spec = _fparam_aparam_input_specs(_StubModel(2, 3))
        self.assertIsNotNone(fparam_spec)
        self.assertIsNotNone(aparam_spec)
        self.assertEqual(fparam_spec.name, "fparam")
        self.assertEqual(aparam_spec.name, "aparam")
        self.assertEqual(list(fparam_spec.shape), [-1, 2])
        self.assertEqual(list(aparam_spec.shape), [-1, -1, 3])

    def test_only_fparam(self) -> None:
        fparam_spec, aparam_spec = _fparam_aparam_input_specs(_StubModel(2, 0))
        self.assertIsNotNone(fparam_spec)
        self.assertIsNone(aparam_spec)
