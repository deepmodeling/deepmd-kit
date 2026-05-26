# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.utils.compat import (
    convert_optimizer_v31_to_v32,
)


class TestConvertOptimizerV31ToV32(unittest.TestCase):
    """Test convert_optimizer_v31_to_v32 function."""

    def test_legacy_opt_type_conversion(self) -> None:
        """Test conversion of opt_type to type."""
        jdata = {
            "training": {
                "opt_type": "Adam",
                "adam_beta1": 0.95,
                "adam_beta2": 0.99,
            }
        }
        result = convert_optimizer_v31_to_v32(jdata, warning=False)
        self.assertIn("optimizer", result)
        self.assertEqual(result["optimizer"]["type"], "Adam")
        self.assertEqual(result["optimizer"]["adam_beta1"], 0.95)
        self.assertEqual(result["optimizer"]["adam_beta2"], 0.99)
        self.assertNotIn("opt_type", result["training"])

    def test_kf_optimizer_parameters(self) -> None:
        """Test extraction of KF optimizer parameters."""
        jdata = {
            "training": {
                "opt_type": "KF",
                "kf_blocksize": 5120,
                "kf_start_pref_e": 1.0,
                "kf_limit_pref_e": 2.0,
                "kf_start_pref_f": 1.0,
                "kf_limit_pref_f": 1.0,
            }
        }
        result = convert_optimizer_v31_to_v32(jdata, warning=False)
        self.assertEqual(result["optimizer"]["type"], "KF")
        self.assertEqual(result["optimizer"]["kf_blocksize"], 5120)
        self.assertEqual(result["optimizer"]["kf_start_pref_e"], 1.0)
        self.assertEqual(result["optimizer"]["kf_limit_pref_e"], 2.0)
        self.assertNotIn("kf_blocksize", result["training"])

    def test_default_values_adam(self) -> None:
        """Test default values are filled for Adam optimizer."""
        jdata = {"training": {}}
        result = convert_optimizer_v31_to_v32(jdata, warning=False)
        self.assertEqual(result["optimizer"]["type"], "Adam")
        self.assertEqual(result["optimizer"]["adam_beta1"], 0.9)
        self.assertEqual(result["optimizer"]["adam_beta2"], 0.999)
        self.assertEqual(result["optimizer"]["weight_decay"], 0.0)

    def test_default_values_adamw(self) -> None:
        """Test default values are filled for AdamW optimizer."""
        jdata = {"training": {"opt_type": "AdamW"}}
        result = convert_optimizer_v31_to_v32(jdata, warning=False)
        self.assertEqual(result["optimizer"]["type"], "AdamW")
        self.assertEqual(result["optimizer"]["adam_beta1"], 0.9)
        self.assertEqual(result["optimizer"]["adam_beta2"], 0.999)
        self.assertEqual(result["optimizer"]["weight_decay"], 0.0)

    def test_existing_optimizer_section(self) -> None:
        """Test merging with existing optimizer section."""
        jdata = {
            "training": {
                "opt_type": "Adam",
                "adam_beta1": 0.95,
            },
            "optimizer": {
                "weight_decay": 0.01,
            },
        }
        result = convert_optimizer_v31_to_v32(jdata, warning=False)
        # Legacy parameters should take precedence
        self.assertEqual(result["optimizer"]["type"], "Adam")
        self.assertEqual(result["optimizer"]["adam_beta1"], 0.95)
        self.assertEqual(result["optimizer"]["weight_decay"], 0.01)


if __name__ == "__main__":
    unittest.main()
