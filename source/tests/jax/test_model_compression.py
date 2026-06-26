# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import tempfile
import unittest
from importlib.util import (
    find_spec,
)
from pathlib import (
    Path,
)

import numpy as np

INSTALLED_JAX = find_spec("jax") is not None and find_spec("orbax") is not None

if INSTALLED_JAX:
    from deepmd.dpmodel.utils.serialization import (
        load_dp_model,
        save_dp_model,
    )
    from deepmd.jax.descriptor.dpa1 import (
        DescrptDPA1,
    )
    from deepmd.jax.descriptor.se_e2_a import (
        DescrptSeA,
    )
    from deepmd.jax.descriptor.se_e2_r import (
        DescrptSeR,
    )
    from deepmd.jax.env import (
        jax,
        jnp,
    )
    from deepmd.jax.model.model import (
        get_model,
    )
    from deepmd.jax.utils.serialization import (
        serialize_from_file,
    )
    from deepmd.main import main as dp_main


class TestJAXModelCompression(unittest.TestCase):
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

    def _make_model_data(self) -> dict:
        return {
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

    @unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
    def test_se_e2_a_enable_compression(self) -> None:
        coord = jnp.array(self.coord)
        atype = jnp.array(self.atype)
        nlist = jnp.array(self.nlist)
        descriptor = DescrptSeA(
            rcut=4.0,
            rcut_smth=3.5,
            sel=[1, 1],
            neuron=[4, 8],
            axis_neuron=2,
            resnet_dt=False,
            type_one_side=True,
            precision="float64",
            seed=1234,
        )
        expected = descriptor.call(coord, atype, nlist)

        compressed = DescrptSeA.deserialize(copy.deepcopy(descriptor.serialize()))
        compressed.enable_compression(1.0, 5, 0.001, 0.01, -1)
        actual = compressed.call(coord, atype, nlist)

        self.assertTrue(compressed.compress)
        serialized = compressed.serialize()
        self.assertEqual(serialized["@version"], 3)
        self.assertIn("compress", serialized)
        reloaded = DescrptSeA.deserialize(copy.deepcopy(serialized))
        reloaded_actual = reloaded.call(coord, atype, nlist)

        for expected_item, actual_item, reloaded_item in zip(
            expected, actual, reloaded_actual, strict=True
        ):
            if expected_item is None:
                self.assertIsNone(actual_item)
                self.assertIsNone(reloaded_item)
            else:
                np.testing.assert_allclose(
                    np.asarray(actual_item), np.asarray(expected_item), atol=1e-10
                )
                np.testing.assert_allclose(
                    np.asarray(reloaded_item), np.asarray(expected_item), atol=1e-10
                )

    @unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
    def test_se_e2_r_enable_compression(self) -> None:
        coord = jnp.array(self.coord)
        atype = jnp.array(self.atype)
        nlist = jnp.array(self.nlist)
        descriptor = DescrptSeR(
            rcut=4.0,
            rcut_smth=3.5,
            sel=[1, 1],
            neuron=[4, 8],
            resnet_dt=False,
            precision="float64",
            seed=1234,
        )
        expected = descriptor.call(coord, atype, nlist)

        compressed = DescrptSeR.deserialize(copy.deepcopy(descriptor.serialize()))
        compressed.enable_compression(1.0, 5, 0.001, 0.01, -1)
        actual = compressed.call(coord, atype, nlist)

        self.assertTrue(compressed.compress)
        serialized = compressed.serialize()
        self.assertEqual(serialized["@version"], 3)
        self.assertIn("compress", serialized)
        reloaded = DescrptSeR.deserialize(copy.deepcopy(serialized))
        reloaded_actual = reloaded.call(coord, atype, nlist)

        for expected_item, actual_item, reloaded_item in zip(
            expected, actual, reloaded_actual, strict=True
        ):
            if expected_item is None:
                self.assertIsNone(actual_item)
                self.assertIsNone(reloaded_item)
            else:
                np.testing.assert_allclose(
                    np.asarray(actual_item), np.asarray(expected_item), atol=1e-10
                )
                np.testing.assert_allclose(
                    np.asarray(reloaded_item), np.asarray(expected_item), atol=1e-10
                )

    @unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
    def test_se_atten_enable_compression(self) -> None:
        coord = jnp.array(self.coord)
        atype = jnp.array(self.atype)
        nlist = jnp.array(self.nlist)
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
        expected = descriptor.call(coord, atype, nlist)

        compressed = DescrptDPA1.deserialize(copy.deepcopy(descriptor.serialize()))
        compressed.enable_compression(1.0, 5, 0.001, 0.01, -1)
        actual = compressed.call(coord, atype, nlist)

        self.assertTrue(compressed.compress)
        self.assertTrue(compressed.geo_compress)
        serialized = compressed.serialize()
        self.assertEqual(serialized["@version"], 3)
        self.assertIn("compress", serialized)
        reloaded = DescrptDPA1.deserialize(copy.deepcopy(serialized))
        reloaded_actual = reloaded.call(coord, atype, nlist)

        for expected_item, actual_item, reloaded_item in zip(
            expected, actual, reloaded_actual, strict=True
        ):
            if expected_item is None:
                self.assertIsNone(actual_item)
                self.assertIsNone(reloaded_item)
            else:
                np.testing.assert_allclose(
                    np.asarray(actual_item), np.asarray(expected_item), atol=1e-10
                )
                np.testing.assert_allclose(
                    np.asarray(reloaded_item), np.asarray(expected_item), atol=1e-10
                )

    @unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
    def test_jax_compress_entrypoint(self) -> None:
        model_data = self._make_model_data()
        model = get_model(copy.deepcopy(model_data))

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "model.hlo"
            output_file = Path(tmpdir) / "model-compressed.jax"
            save_dp_model(
                str(input_file),
                {
                    "backend": "JAX",
                    "jax_version": jax.__version__,
                    "model": model.serialize(),
                    "model_def_script": model_data,
                    "@variables": {
                        "stablehlo": np.void(b"stablehlo"),
                    },
                    "constants": {
                        "min_nbor_dist": 1.0,
                    },
                },
            )

            dp_main(
                [
                    "--jax",
                    "compress",
                    "-i",
                    str(input_file),
                    "-o",
                    str(output_file),
                    "-s",
                    "0.01",
                ]
            )

            compressed = serialize_from_file(str(output_file))
            descriptor = compressed["model"]["descriptor"]
            self.assertEqual(descriptor["@version"], 3)
            self.assertIn("compress", descriptor)
            self.assertEqual(compressed["min_nbor_dist"], 1.0)

    @unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
    def test_jax_compress_entrypoint_can_write_hlo(self) -> None:
        model_data = self._make_model_data()
        model = get_model(copy.deepcopy(model_data))

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "model.hlo"
            output_file = Path(tmpdir) / "model-compressed.hlo"
            save_dp_model(
                str(input_file),
                {
                    "backend": "JAX",
                    "jax_version": jax.__version__,
                    "model": model.serialize(),
                    "model_def_script": model_data,
                    "@variables": {
                        "stablehlo": np.void(b"stablehlo"),
                    },
                    "constants": {
                        "min_nbor_dist": 1.0,
                    },
                },
            )

            dp_main(
                [
                    "--jax",
                    "compress",
                    "-i",
                    str(input_file),
                    "-o",
                    str(output_file),
                    "-s",
                    "0.01",
                ]
            )

            compressed = load_dp_model(str(output_file))
            descriptor = compressed["model"]["descriptor"]
            self.assertEqual(descriptor["@version"], 3)
            self.assertIn("compress", descriptor)
            self.assertEqual(compressed["constants"]["min_nbor_dist"], 1.0)
            self.assertIn("stablehlo", compressed["@variables"])
            self.assertIn("stablehlo_no_ghost", compressed["@variables"])


if __name__ == "__main__":
    unittest.main()
