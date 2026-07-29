# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

from deepmd.utils import (
    argcheck,
)


class TestBackendDocumentation(unittest.TestCase):
    def test_registry_order_and_duplicate_keys(self) -> None:
        self.assertEqual(
            argcheck.supported_backends(
                "tf2", "jax", "pt", "tf", "pd", "jax", "pt_expt"
            ),
            "(Supported Backend: TensorFlow, PyTorch, JAX, PaddlePaddle, "
            "PyTorch Exportable, TensorFlow 2) ",
        )

    def test_hidden_backend_is_omitted(self) -> None:
        hidden_jax = argcheck.BackendDocumentation("JAX", visible=False)
        with patch.dict(argcheck.BACKEND_DOCUMENTATION, {"jax": hidden_jax}):
            self.assertEqual(
                argcheck.supported_backends("pt", "jax", "tf2"),
                "(Supported Backend: PyTorch, TensorFlow 2) ",
            )

    def test_all_hidden_backends_return_empty_label(self) -> None:
        hidden_backends = {
            key: argcheck.BackendDocumentation(backend.display_name, visible=False)
            for key, backend in argcheck.BACKEND_DOCUMENTATION.items()
        }
        with patch.dict(argcheck.BACKEND_DOCUMENTATION, hidden_backends, clear=True):
            self.assertEqual(argcheck.supported_backends(*hidden_backends), "")

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown"):
            argcheck.supported_backends("unknown")

    def test_representative_declared_support_labels(self) -> None:
        """Guard selected declarations with labels independent of the formatter."""
        self.assertTrue(
            argcheck.descrpt_args_plugin.get_argument("dpa4").doc.startswith(
                "(Supported Backend: PyTorch, JAX, PyTorch Exportable) "
            )
        )
        self.assertEqual(
            argcheck.fitting_args_plugin.get_argument("property").doc,
            "(Supported Backend: PyTorch, PyTorch Exportable, TensorFlow 2) ",
        )
        self.assertEqual(
            argcheck.opt_args_plugin.get_argument("AdamW").doc,
            "(Supported Backend: PyTorch, PaddlePaddle, PyTorch Exportable, "
            "TensorFlow 2) ",
        )
        adam_weight_decay = argcheck.opt_args_plugin.get_argument("Adam")[
            "weight_decay"
        ]
        self.assertTrue(
            adam_weight_decay.doc.startswith(
                "(Supported Backend: PyTorch, PaddlePaddle) "
            )
        )
        self.assertEqual(
            argcheck.loss_args_plugin.get_argument("dos").doc,
            "(Supported Backend: TensorFlow, PyTorch, PyTorch Exportable, "
            "TensorFlow 2) ",
        )
        energy_loss = argcheck.loss_args_plugin.get_argument("ener")
        self.assertTrue(
            energy_loss["start_pref_h"].doc.startswith(
                "(Supported Backend: PyTorch, PaddlePaddle) "
            )
        )
        self.assertTrue(
            energy_loss["f_use_norm"].doc.startswith(
                "(Supported Backend: PyTorch, JAX, PyTorch Exportable, TensorFlow 2) "
            )
        )
        preset_out_bias = argcheck.model_args()["preset_out_bias"]
        self.assertTrue(
            preset_out_bias.doc.startswith(
                "(Supported Backend: PyTorch, PaddlePaddle) "
            )
        )
        rglob_patterns = argcheck.training_data_args()["rglob_patterns"]
        self.assertTrue(
            rglob_patterns.doc.startswith(
                "(Supported Backend: TensorFlow, PyTorch, JAX, PaddlePaddle, "
                "PyTorch Exportable, TensorFlow 2) "
            )
        )
        enable_compile = argcheck.training_args()["enable_compile"]
        self.assertTrue(
            enable_compile.doc.startswith(
                "(Supported Backend: PyTorch Exportable, TensorFlow 2) "
            )
        )

    def test_corrected_support_labels(self) -> None:
        all_backends = (
            "(Supported Backend: TensorFlow, PyTorch, JAX, PaddlePaddle, "
            "PyTorch Exportable, TensorFlow 2) "
        )
        self.assertEqual(
            argcheck.lr_args_plugin.get_argument("cosine").doc, all_backends
        )
        self.assertEqual(argcheck.lr_args_plugin.get_argument("wsd").doc, all_backends)
        self.assertTrue(
            argcheck.fitting_args_plugin.get_argument("dipole")[
                "sel_type"
            ].doc.startswith("(Supported Backend: TensorFlow) ")
        )
        self.assertTrue(
            argcheck.fitting_args_plugin.get_argument("polar")[
                "sel_type"
            ].doc.startswith("(Supported Backend: TensorFlow) ")
        )
        self.assertEqual(
            argcheck.opt_args_plugin.get_argument("Adam").doc,
            "(Supported Backend: TensorFlow, PyTorch, PaddlePaddle, "
            "PyTorch Exportable, TensorFlow 2) ",
        )
        self.assertTrue(
            argcheck.opt_args_plugin.get_argument("AdamW")[
                "weight_decay"
            ].doc.startswith(
                "(Supported Backend: PyTorch, PaddlePaddle, PyTorch Exportable, "
                "TensorFlow 2) "
            )
        )


if __name__ == "__main__":
    unittest.main()
