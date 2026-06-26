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
                            actual_item, expected_item, atol=1e-10
                        )
                        np.testing.assert_allclose(
                            reloaded_item, expected_item, atol=1e-10
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
                np.testing.assert_allclose(actual_item, expected_item, atol=1e-10)
                np.testing.assert_allclose(reloaded_item, expected_item, atol=1e-10)

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
                np.testing.assert_allclose(actual_item, expected_item, atol=1e-10)
                np.testing.assert_allclose(reloaded_item, expected_item, atol=1e-10)

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


if __name__ == "__main__":
    unittest.main()
