# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.infer.deep_eval import (
    DeepEval,
)

from ..consistent.common import (
    parameterized,
)
from .case import (
    get_cases,
)


@parameterized(
    (
        "se_e2_a",
        "fparam_aparam",
    ),  # key
    (".pb", ".pth"),  # model extension
)
class TestGetModelMethod(unittest.TestCase):
    """Test the new get_model method functionality."""

    @classmethod
    def setUpClass(cls) -> None:
        key, extension = cls.param
        cls.case = get_cases()[key]
        cls.model_name = cls.case.get_model(extension)
        cls.dp = DeepEval(cls.model_name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dp = None

    def test_get_model_method_exists(self):
        """Test that get_model method exists."""
        self.assertTrue(
            hasattr(self.dp, "get_model"), "DeepEval should have get_model method"
        )

    def test_get_model_returns_valid_object(self):
        """Test that get_model returns a valid model object."""
        model = self.dp.get_model()
        self.assertIsNotNone(model, "get_model should return a non-None object")

    def test_get_model_backend_specific(self):
        """Test that get_model returns the expected type for each backend."""
        key, extension = self.param
        model = self.dp.get_model()

        if extension == ".pth":
            # For PyTorch .pth models (TorchScript), should return torch.jit.ScriptModule
            import torch

            self.assertIsInstance(
                model,
                torch.jit.ScriptModule,
                "PyTorch .pth model should return TorchScript ScriptModule instance",
            )
            # TorchScript modules are also nn.Module instances
            self.assertIsInstance(
                model,
                torch.nn.Module,
                "PyTorch .pth model should be a torch.nn.Module instance",
            )
            # Check if it has common model methods
            self.assertTrue(
                hasattr(model, "get_type_map"),
                "PyTorch model should have get_type_map method",
            )
            self.assertTrue(
                hasattr(model, "get_rcut"),
                "PyTorch model should have get_rcut method",
            )
        elif extension == ".pb":
            # For TensorFlow models, should return graph
            try:
                # Should be a TensorFlow graph or have graph-like properties
                self.assertTrue(
                    hasattr(model, "get_operations")
                    or str(type(model)).find("Graph") >= 0,
                    "TensorFlow model should be a graph or graph-like object",
                )
            except ImportError:
                # If TensorFlow not available, skip this assertion
                pass

    def test_get_model_consistency(self):
        """Test that get_model always returns the same object."""
        model1 = self.dp.get_model()
        model2 = self.dp.get_model()
        # Should return the same object (not necessarily equal, but same reference)
        self.assertIs(
            model1, model2, "get_model should return consistent object reference"
        )


if __name__ == "__main__":
    unittest.main()
