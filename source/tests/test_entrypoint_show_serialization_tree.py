# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    Mock,
    call,
    patch,
)

from deepmd.entrypoints.show import (
    show,
)


class TestShowSerializationTree(unittest.TestCase):
    def test_serialization_tree_uses_deep_eval_model_payload(self) -> None:
        with (
            patch("deepmd.entrypoints.show.DeepEval") as mock_deep_eval,
            patch(
                "deepmd.dpmodel.utils.serialization.Node.deserialize"
            ) as mock_deserialize,
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
            mock_log_info.assert_any_call("Model serialization tree:\n%s", "ROOT")

    def test_serialization_tree_iterates_multitask_branches(self) -> None:
        with (
            patch("deepmd.entrypoints.show.DeepEval") as mock_deep_eval,
            patch(
                "deepmd.dpmodel.utils.serialization.Node.deserialize"
            ) as mock_deserialize,
            patch("deepmd.entrypoints.show.log.info") as mock_log_info,
        ):
            initial_model = mock_deep_eval.return_value
            branch_a_model = mock_deep_eval.return_value
            branch_b_model = Mock()
            mock_deep_eval.side_effect = [initial_model, branch_a_model, branch_b_model]

            initial_model.get_model_def_script.return_value = {
                "model_dict": {"branch_a": {}, "branch_b": {}}
            }
            initial_model.get_model_size.return_value = {}
            branch_a_model.serialize.return_value = {"@class": "BranchA"}
            branch_b_model.serialize.return_value = {"@class": "BranchB"}
            mock_deserialize.side_effect = ["ROOT_A", "ROOT_B"]

            show(INPUT="mock-multitask.pte", ATTRIBUTES=["serialization-tree"])

            mock_deep_eval.assert_has_calls(
                [
                    call("mock-multitask.pte", head=0),
                    call("mock-multitask.pte", head="branch_a"),
                    call("mock-multitask.pte", head="branch_b"),
                ]
            )
            mock_deserialize.assert_has_calls(
                [call({"@class": "BranchA"}), call({"@class": "BranchB"})]
            )
            mock_log_info.assert_any_call(
                "Model serialization tree of branch %s:\n%s", "branch_a", "ROOT_A"
            )
            mock_log_info.assert_any_call(
                "Model serialization tree of branch %s:\n%s", "branch_b", "ROOT_B"
            )
