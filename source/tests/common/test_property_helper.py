# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for high-level property helpers."""

from __future__ import (
    annotations,
)

import unittest
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
)

from deepmd.property import (
    PropertyTrainer,
    resolve_model_name,
)


class TestPropertyHelpers(unittest.TestCase):
    def test_build_input(self) -> None:
        trainer = PropertyTrainer.from_systems(
            type_map=["H", "C", "N", "O"],
            train_systems=["train0", "train1"],
            valid_systems=["valid0"],
            property_name="band_prop",
            task="regression",
            data_type="molecule",
            task_dim=3,
            batch_size=2,
            validation_batch_size=4,
            numb_steps=1000,
            learning_rate=1e-3,
            stop_lr=1e-6,
            decay_steps=200,
            seed=42,
            metrics=["mae"],
            disp_file="curve.out",
            disp_freq=10,
            save_freq=50,
        )
        config = trainer.build_input()
        self.assertEqual(config["model"]["type_map"], ["H", "C", "N", "O"])
        self.assertEqual(config["model"]["fitting_net"]["property_name"], "band_prop")
        self.assertEqual(config["model"]["fitting_net"]["task_dim"], 3)
        self.assertEqual(config["training"]["training_data"]["systems"], ["train0", "train1"])
        self.assertEqual(config["training"]["validation_data"]["systems"], ["valid0"])
        self.assertEqual(config["training"]["validation_data"]["batch_size"], 4)

    def test_write_input(self) -> None:
        trainer = PropertyTrainer.from_systems(
            type_map=["H", "O"],
            train_systems=["train0"],
            save_path="ignored.json",
        )
        out = Path("test_property_helper_input.json")
        try:
            path = trainer.write_input(out)
            self.assertEqual(path, out)
            self.assertTrue(out.exists())
            text = out.read_text()
            self.assertIn('"property_name": "property"', text)
        finally:
            out.unlink(missing_ok=True)

    def test_resolve_model_name_alias(self) -> None:
        with patch(
            "deepmd.property.resolve_model_path",
            return_value=Path("/tmp/model.pt"),
        ) as mocked_resolve:
            resolved = resolve_model_name("DPA-3.2-5M", cache_dir="/tmp/cache")
        self.assertEqual(resolved, "/tmp/model.pt")
        mocked_resolve.assert_called_once()

    def test_resolve_model_name_passthrough(self) -> None:
        spec = "/abs/path/to/model.pt"
        self.assertEqual(resolve_model_name(spec), spec)


if __name__ == "__main__":
    unittest.main()
