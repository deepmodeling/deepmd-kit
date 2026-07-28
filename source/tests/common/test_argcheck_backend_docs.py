# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

from deepmd.utils import (
    argcheck,
)


class TestBackendDocumentation(unittest.TestCase):
    def tearDown(self) -> None:
        # Tests that temporarily change visibility must not leak cached labels.
        argcheck.supported_backends.cache_clear()

    def test_all_backends_and_cache(self) -> None:
        backends = ("tf", "pt", "jax", "pd", "pt_expt", "tf2")
        argcheck.supported_backends.cache_clear()
        first = argcheck.supported_backends(*backends)
        second = argcheck.supported_backends(*backends)

        self.assertEqual(
            first,
            "(Supported Backend: TensorFlow, PyTorch, JAX, PaddlePaddle, "
            "PyTorch Exportable, TensorFlow 2) ",
        )
        self.assertIs(first, second)
        self.assertEqual(argcheck.supported_backends.cache_info().hits, 1)

    def test_hidden_backend_is_omitted(self) -> None:
        hidden_jax = argcheck.BackendDocumentation("JAX", visible=False)
        with patch.dict(argcheck.BACKEND_DOCUMENTATION, {"jax": hidden_jax}):
            argcheck.supported_backends.cache_clear()
            self.assertEqual(
                argcheck.supported_backends("pt", "jax", "tf2"),
                "(Supported Backend: PyTorch, TensorFlow 2) ",
            )

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown"):
            argcheck.supported_backends("unknown")

    def test_representative_support_matrices(self) -> None:
        self.assertTrue(
            argcheck.descrpt_args_plugin.get_argument("dpa4").doc.startswith(
                argcheck.supported_backends("pt", "jax", "pt_expt")
            )
        )
        self.assertEqual(
            argcheck.fitting_args_plugin.get_argument("property").doc,
            argcheck.supported_backends("pt", "pt_expt", "tf2"),
        )
        self.assertEqual(
            argcheck.opt_args_plugin.get_argument("AdamW").doc,
            argcheck.supported_backends("pt", "pd", "tf2"),
        )
        adam_weight_decay = argcheck.opt_args_plugin.get_argument("Adam")[
            "weight_decay"
        ]
        self.assertTrue(
            adam_weight_decay.doc.startswith(argcheck.supported_backends("pt", "pd"))
        )
        self.assertEqual(
            argcheck.loss_args_plugin.get_argument("dos").doc,
            argcheck.supported_backends("tf", "pt", "pt_expt", "tf2"),
        )
        energy_loss = argcheck.loss_args_plugin.get_argument("ener")
        self.assertTrue(
            energy_loss["start_pref_h"].doc.startswith(
                argcheck.supported_backends("pt", "pd")
            )
        )
        self.assertTrue(
            energy_loss["f_use_norm"].doc.startswith(
                argcheck.supported_backends("pt", "jax", "pt_expt", "tf2")
            )
        )
        preset_out_bias = argcheck.model_args()["preset_out_bias"]
        self.assertTrue(
            preset_out_bias.doc.startswith(argcheck.supported_backends("pt", "pd"))
        )
        rglob_patterns = argcheck.training_data_args()["rglob_patterns"]
        self.assertTrue(
            rglob_patterns.doc.startswith(
                argcheck.supported_backends("tf", "pt", "jax", "pd", "pt_expt", "tf2")
            )
        )
        enable_compile = argcheck.training_args()["enable_compile"]
        self.assertTrue(
            enable_compile.doc.startswith(argcheck.supported_backends("pt_expt", "tf2"))
        )


if __name__ == "__main__":
    unittest.main()
