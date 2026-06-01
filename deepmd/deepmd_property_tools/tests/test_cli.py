# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the deprecated deepmd-property-tools CLI redirect."""

from __future__ import annotations

from deepmd_property_tools import cli


def test_main_redirects_to_dp_dpa(capsys) -> None:
    exit_code = cli.main([])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "dp dpa" in captured.err


def test_main_with_args_redirects(capsys) -> None:
    exit_code = cli.main(["train", "--dataset", "d.csv"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "dp dpa" in captured.err
