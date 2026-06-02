# SPDX-License-Identifier: LGPL-3.0-or-later
"""Smoke tests for ``dp dpa`` CLI integration.

Test that the ``dpa`` subcommand group is registered in the main parser,
all verbs are reachable, and ``--help`` does not trigger eager loading of
torch or any DPA implementation.
"""

from __future__ import annotations

import sys


class TestDpaParserRegistration:
    """Verify ``dpa`` appears in the top-level command list."""

    def test_dpa_in_subparser_choices(self):
        from deepmd.main import main_parser

        parser = main_parser()
        # argparse stores subcommand choices in the subparser action
        sub_action = next(
            a for a in parser._actions if a.dest == "command"
        )
        assert "dpa" in sub_action.choices, (
            f"dpa not found in top-level commands: {sorted(sub_action.choices)}"
        )

    def test_dpa_verbs_registered(self):
        from deepmd.main import main_parser

        parser = main_parser()
        sub_action = next(a for a in parser._actions if a.dest == "command")
        dpa_parser = sub_action.choices["dpa"]
        dpa_sub_action = next(
            a for a in dpa_parser._actions if a.dest == "dpa_command"
        )
        verbs = sorted(dpa_sub_action.choices)
        for expected in (
            "extract-descriptors", "fit", "cv", "predict", "evaluate", "data",
        ):
            assert expected in verbs, f"{expected!r} missing from {verbs}"
        assert "mft" not in verbs, "mft should be folded into fit --strategy mft"

    def test_data_subcommands_registered(self):
        from deepmd.main import main_parser

        parser = main_parser()
        sub_action = next(a for a in parser._actions if a.dest == "command")
        dpa_parser = sub_action.choices["dpa"]
        dpa_sub_action = next(a for a in dpa_parser._actions if a.dest == "dpa_command")
        data_parser = dpa_sub_action.choices["data"]
        data_sub_action = next(
            a for a in data_parser._actions if a.dest == "dpa_data_command"
        )
        data_verbs = sorted(data_sub_action.choices)
        for expected in ("convert", "validate", "attach-labels"):
            assert expected in data_verbs, f"{expected!r} missing from {data_verbs}"


class TestDpaHelpNoTorch:
    """``dp dpa --help`` must NOT trigger a torch import."""

    def test_help_does_not_load_torch(self):
        from unittest.mock import MagicMock

        from deepmd.main import main_parser

        # Other tests may inject a mock torch into sys.modules; that's fine
        # as long as OUR parser path doesn't cause a *new* import.
        torch_already = "torch" in sys.modules
        if torch_already:
            existing = sys.modules["torch"]
            if not isinstance(existing, MagicMock):
                import pytest
                pytest.skip("torch already loaded by another test")

        parser = main_parser()
        sub_action = next(a for a in parser._actions if a.dest == "command")
        dpa_parser = sub_action.choices["dpa"]

        # Format the help text — this is the code path that argparse runs
        # when --help is requested.
        dpa_parser.format_help()

        if not torch_already:
            assert "torch" not in sys.modules, (
                "torch was loaded during dp dpa --help path!"
            )


class TestDpaDispatch:
    """Verify the dispatch table covers all registered verbs."""

    def test_dispatch_keys_match_parser_verbs(self):
        from deepmd.main import main_parser

        from deepmd.dpa_tools.cli import _DISPATCH, _DATA_DISPATCH

        parser = main_parser()
        sub_action = next(a for a in parser._actions if a.dest == "command")
        dpa_parser = sub_action.choices["dpa"]
        dpa_sub_action = next(a for a in dpa_parser._actions if a.dest == "dpa_command")

        parser_verbs = set(dpa_sub_action.choices)
        dispatch_verbs = set(_DISPATCH) | {"data"}

        extra_in_parser = parser_verbs - dispatch_verbs
        extra_in_dispatch = dispatch_verbs - parser_verbs
        assert not extra_in_parser, (
            f"Verbs in parser but not in dispatch: {extra_in_parser}"
        )
        assert not extra_in_dispatch, (
            f"Verbs in dispatch but not in parser: {extra_in_dispatch}"
        )

    def test_data_dispatch_keys_match_parser_verbs(self):
        from deepmd.main import main_parser

        from deepmd.dpa_tools.cli import _DATA_DISPATCH

        parser = main_parser()
        sub_action = next(a for a in parser._actions if a.dest == "command")
        dpa_parser = sub_action.choices["dpa"]
        dpa_sub_action = next(a for a in dpa_parser._actions if a.dest == "dpa_command")
        data_parser = dpa_sub_action.choices["data"]
        data_sub_action = next(a for a in data_parser._actions if a.dest == "dpa_data_command")

        parser_verbs = set(data_sub_action.choices)
        dispatch_verbs = set(_DATA_DISPATCH)

        extra_in_parser = parser_verbs - dispatch_verbs
        extra_in_dispatch = dispatch_verbs - parser_verbs
        assert not extra_in_parser, (
            f"Data verbs in parser but not in dispatch: {extra_in_parser}"
        )
        assert not extra_in_dispatch, (
            f"Data verbs in dispatch but not in parser: {extra_in_dispatch}"
        )


class TestInitAllExports:
    """Verify __all__ covers the key public names."""

    def test_all_exports(self):
        from deepmd import dpa_tools

        for name in [
            "DPAFineTuner", "DPAPredictor", "MFTFineTuner", "DPATrainer",
            "cross_validate", "train_test_split", "extract_descriptors",
            "convert", "batch_convert", "attach_labels", "check_data",
            "load_dataset", "ConditionManager", "DPAConditionError",
        ]:
            assert hasattr(dpa_tools, name), f"{name!r} not found on dpa_tools"
