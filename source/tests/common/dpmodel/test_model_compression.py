# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.dpmodel.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.dpmodel.descriptor.se_r import (
    DescrptSeR,
)
from deepmd.dpmodel.entrypoints.compress import (
    enable_compression,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.model import (
    get_model,
)
from deepmd.dpmodel.utils.serialization import (
    load_dp_model,
    save_dp_model,
)


class TestDPModelCompression(unittest.TestCase):
    def setUp(self) -> None:
        self.coord = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.2, 0.1, 0.0],
                    [0.1, 1.4, 0.0],
                    [1.5, 1.5, 0.1],
                ]
            ],
            dtype=np.float64,
        )
        self.atype = np.array([[0, 0, 1, 1]], dtype=np.int32)
        self.nlist = np.array([[[1, 2], [0, 2], [0, 3], [0, 2]]], dtype=np.int64)

    def _make_descriptor(
        self,
        type_one_side: bool = True,
        exclude_types: list[list[int]] | None = None,
    ) -> DescrptSeA:
        return DescrptSeA(
            rcut=4.0,
            rcut_smth=3.5,
            sel=[1, 1],
            neuron=[4, 8],
            axis_neuron=2,
            resnet_dt=False,
            type_one_side=type_one_side,
            exclude_types=[] if exclude_types is None else exclude_types,
            precision="float64",
            seed=1234,
        )

    @staticmethod
    def _poly5(coeff: np.ndarray, xx: np.ndarray | float) -> np.ndarray:
        return (
            coeff[..., 0]
            + (
                coeff[..., 1]
                + (
                    coeff[..., 2]
                    + (coeff[..., 3] + (coeff[..., 4] + coeff[..., 5] * xx) * xx) * xx
                )
                * xx
            )
            * xx
        )

    @staticmethod
    def _poly5_grad(coeff: np.ndarray, xx: np.ndarray | float) -> np.ndarray:
        return (
            coeff[..., 1]
            + (
                2 * coeff[..., 2]
                + (
                    3 * coeff[..., 3]
                    + (4 * coeff[..., 4] + 5 * coeff[..., 5] * xx) * xx
                )
                * xx
            )
            * xx
        )

    def _make_c1_tabulation_case(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        table_info = np.array([1.0, 3.0, 5.0, 1.0, 1.0, -1.0], dtype=np.float64)
        coeff = np.array(
            [
                [
                    [1.0, 0.5, 0.25, -0.10, 0.03, 0.02],
                    [2.0, -0.4, 0.20, 0.05, -0.02, 0.01],
                ],
                [
                    [3.0, -0.2, 0.40, 0.15, -0.03, 0.02],
                    [4.0, 0.3, -0.10, 0.07, 0.04, -0.01],
                ],
                [
                    [5.0, 0.7, -0.20, 0.12, 0.01, -0.04],
                    [6.0, -0.6, 0.30, -0.05, 0.02, 0.03],
                ],
                [
                    [7.0, 1.1, -0.35, 0.25, 0.08, -0.02],
                    [8.0, -0.9, 0.45, -0.15, 0.06, 0.01],
                ],
            ],
            dtype=np.float64,
        )
        xx = np.array([[0.5, 1.0, 1.5, 3.25, 5.0, 5.5]], dtype=np.float64)
        expected = self._expected_c1_tabulation(coeff, table_info, xx[0])
        table = np.reshape(coeff, (coeff.shape[0], -1))
        return table, coeff, table_info, xx, expected

    def _expected_c1_tabulation(
        self,
        coeff: np.ndarray,
        table_info: np.ndarray,
        xx: np.ndarray,
    ) -> np.ndarray:
        lower, upper, table_max, stride0, stride1 = table_info[:5]
        first_stride = int(np.floor((upper - lower) / stride0))
        values = []
        for value in xx:
            delta = 0.0
            if value < lower:
                table_idx = 0
                dx = 0.0
                delta = value - lower
            elif value < upper:
                table_idx = int(np.floor((value - lower) / stride0))
                dx = value - (table_idx * stride0 + lower)
            elif value < table_max:
                table_idx = first_stride + int(np.floor((value - upper) / stride1))
                dx = value - ((table_idx - first_stride) * stride1 + upper)
            else:
                table_idx = coeff.shape[0] - 1
                dx = table_max - ((table_idx - first_stride) * stride1 + upper)
                delta = value - table_max
            values.append(
                self._poly5(coeff[table_idx], dx)
                + self._poly5_grad(coeff[table_idx], dx) * delta
            )
        return np.asarray(values, dtype=np.float64)

    def test_tabulate_fusion_se_r_c1_extrapolates_outside_table(self) -> None:
        table, coeff, table_info, xx, expected = self._make_c1_tabulation_case()
        descriptor = DescrptSeR(
            rcut=4.0,
            rcut_smth=3.5,
            sel=[1, 1],
            neuron=[4, 2],
            resnet_dt=False,
            precision="float64",
            seed=1234,
        )

        actual = descriptor._tabulate_fusion_se_r(
            table,
            table_info,
            xx[:, :, None],
            expected.shape[-1],
        )

        np.testing.assert_allclose(actual[0], expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            actual[0, 1] - actual[0, 0],
            0.5 * self._poly5_grad(coeff[0], 0.0),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual[0, 5] - actual[0, 4],
            0.5 * self._poly5_grad(coeff[-1], 1.0),
            rtol=0.0,
            atol=1e-12,
        )

    def test_tabulate_fusion_se_a_c1_extrapolates_outside_table(self) -> None:
        table, _, table_info, xx, expected = self._make_c1_tabulation_case()
        descriptor = self._make_descriptor()
        em = np.zeros((1, xx.shape[1], 4), dtype=np.float64)
        em[:, :, 0] = 1.0
        expected_out = np.zeros((1, 4, expected.shape[-1]), dtype=np.float64)
        expected_out[:, 0, :] = np.sum(expected, axis=0)

        actual = descriptor._tabulate_fusion_se_a(
            table,
            table_info,
            xx[:, :, None],
            em,
            expected.shape[-1],
        )

        np.testing.assert_allclose(actual, expected_out, rtol=0.0, atol=1e-12)

    def test_tabulate_fusion_se_atten_c1_extrapolates_outside_table(self) -> None:
        table, _, table_info, xx, expected = self._make_c1_tabulation_case()
        descriptor = DescrptDPA1(
            rcut=4.0,
            rcut_smth=3.5,
            sel=2,
            ntypes=2,
            neuron=[4, 2],
            axis_neuron=2,
            tebd_dim=4,
            tebd_input_mode="strip",
            resnet_dt=False,
            attn_layer=0,
            precision="float64",
            seed=1234,
        )
        em = np.zeros((1, xx.shape[1], 4), dtype=np.float64)
        em[:, :, 0] = 1.0
        two_embed = np.full_like(expected[None, :, :], 0.5)
        expected_out = np.zeros((1, 4, expected.shape[-1]), dtype=np.float64)
        expected_out[:, 0, :] = np.sum(expected * (1.0 + two_embed[0]), axis=0)

        actual = descriptor.se_atten._tabulate_fusion_se_atten(
            table,
            table_info,
            xx[:, :, None],
            em,
            two_embed,
            expected.shape[-1],
        )

        np.testing.assert_allclose(actual, expected_out, rtol=0.0, atol=1e-12)

    def test_se_e2_a_enable_compression(self) -> None:
        for type_one_side, exclude_types in (
            (True, []),
            (False, []),
            (True, [[0, 1]]),
            (False, [[0, 1]]),
        ):
            with self.subTest(type_one_side=type_one_side, exclude_types=exclude_types):
                descriptor = self._make_descriptor(type_one_side, exclude_types)
                expected = descriptor.call(self.coord, self.atype, self.nlist)

                compressed = DescrptSeA.deserialize(
                    copy.deepcopy(descriptor.serialize())
                )
                compressed.enable_compression(1.0, 5, 0.001, 0.01, -1)
                actual = compressed.call(self.coord, self.atype, self.nlist)

                self.assertTrue(compressed.compress)
                serialized = compressed.serialize()
                self.assertEqual(serialized["@version"], 3)
                self.assertIn("compress", serialized)
                reloaded = DescrptSeA.deserialize(copy.deepcopy(serialized))
                reloaded_actual = reloaded.call(self.coord, self.atype, self.nlist)

                for expected_item, actual_item, reloaded_item in zip(
                    expected, actual, reloaded_actual, strict=True
                ):
                    if expected_item is None:
                        self.assertIsNone(actual_item)
                        self.assertIsNone(reloaded_item)
                    else:
                        np.testing.assert_allclose(
                            actual_item, expected_item, rtol=0.0, atol=1e-10
                        )
                        np.testing.assert_allclose(
                            reloaded_item, expected_item, rtol=0.0, atol=1e-10
                        )

    def test_se_e2_r_enable_compression(self) -> None:
        descriptor = DescrptSeR(
            rcut=4.0,
            rcut_smth=3.5,
            sel=[1, 1],
            neuron=[4, 8],
            resnet_dt=False,
            precision="float64",
            seed=1234,
        )
        expected = descriptor.call(self.coord, self.atype, self.nlist)

        compressed = DescrptSeR.deserialize(copy.deepcopy(descriptor.serialize()))
        compressed.enable_compression(1.0, 5, 0.001, 0.01, -1)
        actual = compressed.call(self.coord, self.atype, self.nlist)

        self.assertTrue(compressed.compress)
        serialized = compressed.serialize()
        self.assertEqual(serialized["@version"], 3)
        self.assertIn("compress", serialized)
        reloaded = DescrptSeR.deserialize(copy.deepcopy(serialized))
        reloaded_actual = reloaded.call(self.coord, self.atype, self.nlist)

        for expected_item, actual_item, reloaded_item in zip(
            expected, actual, reloaded_actual, strict=True
        ):
            if expected_item is None:
                self.assertIsNone(actual_item)
                self.assertIsNone(reloaded_item)
            else:
                np.testing.assert_allclose(
                    actual_item, expected_item, rtol=0.0, atol=1e-10
                )
                np.testing.assert_allclose(
                    reloaded_item, expected_item, rtol=0.0, atol=1e-10
                )

    def test_se_atten_enable_compression(self) -> None:
        descriptor = DescrptDPA1(
            rcut=4.0,
            rcut_smth=3.5,
            sel=2,
            ntypes=2,
            neuron=[4, 8],
            axis_neuron=2,
            tebd_dim=4,
            tebd_input_mode="strip",
            resnet_dt=False,
            attn_layer=0,
            precision="float64",
            seed=1234,
        )
        expected = descriptor.call(self.coord, self.atype, self.nlist)

        compressed = DescrptDPA1.deserialize(copy.deepcopy(descriptor.serialize()))
        compressed.enable_compression(1.0, 5, 0.001, 0.01, -1)
        actual = compressed.call(self.coord, self.atype, self.nlist)

        self.assertTrue(compressed.compress)
        self.assertTrue(compressed.geo_compress)
        serialized = compressed.serialize()
        self.assertEqual(serialized["@version"], 3)
        self.assertIn("compress", serialized)
        reloaded = DescrptDPA1.deserialize(copy.deepcopy(serialized))
        reloaded_actual = reloaded.call(self.coord, self.atype, self.nlist)

        for expected_item, actual_item, reloaded_item in zip(
            expected, actual, reloaded_actual, strict=True
        ):
            if expected_item is None:
                self.assertIsNone(actual_item)
                self.assertIsNone(reloaded_item)
            else:
                np.testing.assert_allclose(
                    actual_item, expected_item, rtol=0.0, atol=1e-10
                )
                np.testing.assert_allclose(
                    reloaded_item, expected_item, rtol=0.0, atol=1e-10
                )

    def test_dpmodel_compress_entrypoint(self) -> None:
        model_data = {
            "type": "standard",
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "rcut": 4.0,
                "rcut_smth": 3.5,
                "sel": [1, 1],
                "neuron": [4, 8],
                "axis_neuron": 2,
                "resnet_dt": False,
                "type_one_side": True,
                "precision": "float64",
                "seed": 1234,
            },
            "fitting_net": {
                "type": "ener",
                "neuron": [8],
                "resnet_dt": False,
                "precision": "float64",
                "seed": 5678,
            },
        }
        model = get_model(model_data)
        model.min_nbor_dist = 1.0
        expected_output = model.call(self.coord, self.atype)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "model.dp"
            output_file = Path(tmpdir) / "model-compressed.dp"
            save_dp_model(
                str(input_file),
                {
                    "model": model.serialize(),
                    "model_def_script": model_data,
                    "min_nbor_dist": 1.0,
                },
            )

            enable_compression(
                str(input_file),
                str(output_file),
                stride=0.01,
                extrapolate=5,
                check_frequency=-1,
            )

            compressed = load_dp_model(str(output_file))
            descriptor = compressed["model"]["descriptor"]
            self.assertEqual(descriptor["@version"], 3)
            self.assertIn("compress", descriptor)
            self.assertEqual(compressed["min_nbor_dist"], 1.0)

            reloaded_model = BaseModel.deserialize(copy.deepcopy(compressed["model"]))
            actual_output = reloaded_model.call(self.coord, self.atype)
            self.assertEqual(actual_output.keys(), expected_output.keys())
            for key, expected_value in expected_output.items():
                np.testing.assert_allclose(
                    actual_output[key], expected_value, rtol=0.0, atol=1e-10
                )


if __name__ == "__main__":
    unittest.main()
