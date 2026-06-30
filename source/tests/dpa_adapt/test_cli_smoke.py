# SPDX-License-Identifier: LGPL-3.0-or-later
"""Smoke tests for the standalone ``dpa-adapt`` / ``dpaad`` CLI.

Test that all verbs are reachable, ``--help`` does not trigger eager loading
of torch or any DPA implementation, and dispatch tables cover all verbs.
"""

from __future__ import (
    annotations,
)

import sys


class TestDpaAdaptParserRegistration:
    """Verify all dpa-adapt verbs are registered in the standalone parser."""

    def test_dpa_verbs_registered(self):
        from dpa_adapt.cli import (
            get_parser,
        )

        parser = get_parser()
        sub_action = next(a for a in parser._actions if a.dest == "command")
        verbs = sorted(sub_action.choices)
        for expected in (
            "extract-descriptors",
            "fit",
            "cv",
            "predict",
            "evaluate",
            "data",
        ):
            assert expected in verbs, f"{expected!r} missing from {verbs}"
        assert "mft" not in verbs, "mft should be folded into fit --strategy mft"

    def test_data_subcommands_registered(self):
        from dpa_adapt.cli import (
            get_parser,
        )

        parser = get_parser()
        sub_action = next(a for a in parser._actions if a.dest == "command")
        data_parser = sub_action.choices["data"]
        data_sub_action = next(
            a for a in data_parser._actions if a.dest == "data_command"
        )
        data_verbs = sorted(data_sub_action.choices)
        for expected in ("convert", "validate", "attach-labels"):
            assert expected in data_verbs, f"{expected!r} missing from {data_verbs}"


class TestDpaAdaptHelpNoTorch:
    """``dpa-adapt --help`` must NOT trigger a torch import."""

    def test_help_does_not_load_torch(self):
        from unittest.mock import (
            MagicMock,
        )

        from dpa_adapt.cli import (
            get_parser,
        )

        # Other tests may inject a mock torch into sys.modules; that's fine
        # as long as OUR parser path doesn't cause a *new* import.
        torch_already = "torch" in sys.modules
        if torch_already:
            existing = sys.modules["torch"]
            if not isinstance(existing, MagicMock):
                import pytest

                pytest.skip("torch already loaded by another test")

        parser = get_parser()

        # Format the help text — this is the code path that argparse runs
        # when --help is requested.
        parser.format_help()

        if not torch_already:
            assert "torch" not in sys.modules, (
                "torch was loaded during dpa-adapt --help path!"
            )

    def test_main_without_subcommand_prints_help(self, capsys):
        from dpa_adapt.cli import (
            main,
        )

        main([])
        captured = capsys.readouterr()

        assert "usage:" in captured.out
        assert "subcommands" in captured.out
        assert captured.err == ""


class TestDpaDispatch:
    """Verify the dispatch table covers all registered verbs."""

    def test_dispatch_keys_match_parser_verbs(self):
        from dpa_adapt.cli import (
            _DISPATCH,
            get_parser,
        )

        parser = get_parser()
        sub_action = next(a for a in parser._actions if a.dest == "command")

        parser_verbs = set(sub_action.choices)
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
        from dpa_adapt.cli import (
            _DATA_DISPATCH,
            get_parser,
        )

        parser = get_parser()
        sub_action = next(a for a in parser._actions if a.dest == "command")
        data_parser = sub_action.choices["data"]
        data_sub_action = next(
            a for a in data_parser._actions if a.dest == "data_command"
        )

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


class TestDpaFitArgumentNormalization:
    """Verify fit list arguments normalize argparse ``nargs`` values."""

    def test_maybe_split_list_accepts_string_sequences(self):
        from dpa_adapt.cli import (
            _maybe_split_list,
        )

        assert _maybe_split_list(["train_a", "train_b,train_c"]) == [
            "train_a",
            "train_b",
            "train_c",
        ]
        assert _maybe_split_list("H,C, O") == ["H", "C", "O"]
        assert _maybe_split_list(None) is None

    def test_batch_size_parser_preserves_deepmd_specs(self):
        from dpa_adapt.cli import (
            _parse_batch_size,
        )

        assert _parse_batch_size("128") == 128
        assert _parse_batch_size("auto:512") == "auto:512"

    def test_fit_accepts_downstream_auto_batch_size(self):
        from dpa_adapt.cli import (
            get_parser,
        )

        args = get_parser().parse_args(
            [
                "fit",
                "--train-data",
                "train",
                "--strategy",
                "mft",
                "--downstream-batch-size",
                "auto:512",
            ]
        )

        assert args.downstream_batch_size == "auto:512"

    def test_fit_batch_size_numbers_parse_to_int(self):
        from dpa_adapt.cli import (
            get_parser,
        )

        args = get_parser().parse_args(
            [
                "fit",
                "--train-data",
                "train",
                "--batch-size",
                "64",
                "--aux-batch-size",
                "128",
                "--downstream-batch-size",
                "256",
            ]
        )

        assert args.batch_size == 64
        assert args.aux_batch_size == 128
        assert args.downstream_batch_size == 256


class TestInitAllExports:
    """Verify __all__ covers the key public names."""

    def test_all_exports(self):
        import dpa_adapt

        for name in [
            "DPAFineTuner",
            "DPAPredictor",
            "MFTFineTuner",
            "DPATrainer",
            "cross_validate",
            "train_test_split",
            "extract_descriptors",
            "convert",
            "attach_labels",
            "check_data",
            "load_dataset",
            "ConditionManager",
            "DPAConditionError",
        ]:
            assert hasattr(dpa_adapt, name), f"{name!r} not found on dpa_adapt"
