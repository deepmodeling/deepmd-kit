# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    Mock,
    patch,
)

from deepmd.infer.deep_eval import (
    DeepEvalBackend,
)


class _DefaultSerializeBackend(DeepEvalBackend):
    def __init__(self, model: object) -> None:
        self._model = model

    def eval(self, *args: object, **kwargs: object) -> dict:
        return {}

    def get_rcut(self) -> float:
        return 0.0

    def get_ntypes(self) -> int:
        return 0

    def get_type_map(self) -> list[str]:
        return []

    def get_dim_fparam(self) -> int:
        return 0

    def get_dim_aparam(self) -> int:
        return 0

    @property
    def model_type(self) -> type:
        return object

    def get_sel_type(self) -> list[int]:
        return []

    def get_ntypes_spin(self) -> int:
        return 0

    def get_model(self) -> object:
        return self._model


class TestDeepEvalBackendSerialize(unittest.TestCase):
    def test_default_serialize_delegates_to_model_when_available(self) -> None:
        model = Mock()
        model.serialize.return_value = {"@class": "Model"}
        backend = _DefaultSerializeBackend(model)

        self.assertEqual(backend.serialize(), {"@class": "Model"})
        model.serialize.assert_called_once_with()

    def test_default_serialize_has_clear_error_without_model_method(self) -> None:
        backend = _DefaultSerializeBackend(object())

        with self.assertRaisesRegex(
            NotImplementedError, "does not implement serialize"
        ):
            backend.serialize()


def _load_deep_eval_backend(module_name: str, backend_name: str):
    try:
        module = __import__(module_name, fromlist=["DeepEval"])
    except ImportError as exc:
        raise unittest.SkipTest(
            f"{backend_name} backend is not importable: {exc}"
        ) from exc
    return module.DeepEval


class TestPaddleDeepEvalSerialize(unittest.TestCase):
    def test_jit_model_falls_back_to_file_serializer(self) -> None:
        PaddleDeepEvalBackend = _load_deep_eval_backend(
            "deepmd.pd.infer.deep_eval", "Paddle"
        )
        backend = PaddleDeepEvalBackend.__new__(PaddleDeepEvalBackend)
        backend.model_path = "frozen_model.json"
        backend.dp = object()

        with patch("deepmd.pd.utils.serialization.serialize_from_file") as serialize:
            serialize.return_value = {"model": {"@class": "RecoveredModel"}}

            self.assertEqual(backend.serialize(), {"@class": "RecoveredModel"})

        serialize.assert_called_once_with("frozen_model.json")


class TestPyTorchDeepEvalSerialize(unittest.TestCase):
    def test_jit_model_falls_back_to_file_serializer(self) -> None:
        PyTorchDeepEvalBackend = _load_deep_eval_backend(
            "deepmd.pt.infer.deep_eval", "PyTorch"
        )
        backend = PyTorchDeepEvalBackend.__new__(PyTorchDeepEvalBackend)
        backend.model_path = "frozen_model.pth"
        backend.dp = Mock()
        backend.dp.model = {"Default": object()}

        with patch("deepmd.pt.utils.serialization.serialize_from_file") as serialize:
            serialize.return_value = {"model": {"@class": "RecoveredModel"}}

            self.assertEqual(backend.serialize(), {"@class": "RecoveredModel"})

        serialize.assert_called_once_with("frozen_model.pth")


class TestPyTorchExportableDeepEvalSerialize(unittest.TestCase):
    def test_raw_model_payload_fallback_is_preserved(self) -> None:
        PyTorchExportableDeepEvalBackend = _load_deep_eval_backend(
            "deepmd.pt_expt.infer.deep_eval", "PyTorch exportable"
        )
        backend = PyTorchExportableDeepEvalBackend.__new__(
            PyTorchExportableDeepEvalBackend
        )
        backend.model_path = "frozen_model.pt"

        with patch(
            "deepmd.pt_expt.utils.serialization.serialize_from_file"
        ) as serialize:
            serialize.return_value = {"@class": "RawExportedModel"}

            self.assertEqual(backend.serialize(), {"@class": "RawExportedModel"})

        serialize.assert_called_once_with("frozen_model.pt")


class TestJAXDeepEvalSerialize(unittest.TestCase):
    def test_savedmodel_reconstructs_tree_from_model_def_script(self) -> None:
        JAXDeepEvalBackend = _load_deep_eval_backend(
            "deepmd.jax.infer.deep_eval", "JAX"
        )
        backend = JAXDeepEvalBackend.__new__(JAXDeepEvalBackend)
        backend.model_path = "frozen_model.savedmodel"
        backend.get_model_def_script = Mock(return_value={"type_map": ["O", "H"]})

        model = Mock()
        model.serialize.return_value = {"@class": "SavedModelTree"}
        with patch("deepmd.jax.model.model.get_model", return_value=model) as get_model:
            self.assertEqual(backend.serialize(), {"@class": "SavedModelTree"})

        get_model.assert_called_once_with({"type_map": ["O", "H"]})
        model.serialize.assert_called_once_with()
