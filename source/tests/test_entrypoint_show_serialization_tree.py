# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

from deepmd.entrypoints.show import (
    show,
)


class TestShowSerializationTree(unittest.TestCase):
    def test_serialization_tree_uses_deep_eval_model_payload(self) -> None:
        with (
            patch("deepmd.entrypoints.show.DeepEval") as mock_deep_eval,
            patch("deepmd.entrypoints.show.Node.deserialize") as mock_deserialize,
            patch("deepmd.entrypoints.show.log.info") as mock_log_info,
        ):
            model = mock_deep_eval.return_value
            model.get_model_def_script.return_value = {"type_map": ["H", "O"]}
            model.get_model_size.return_value = {}
            model.serialize.return_value = {"@class": "MockModel"}
            mock_deserialize.return_value = "ROOT"

            show(INPUT="mock.pte", ATTRIBUTES=["serialization-tree"])

            model.serialize.assert_called_once_with()
            mock_deserialize.assert_called_once_with({"@class": "MockModel"})
            mock_log_info.assert_any_call("Model serialization tree:\nROOT")
