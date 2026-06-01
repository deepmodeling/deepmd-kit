"""Tests for batch_convert() and convert()'s validation wiring.

Uses hand-written VASP POSCAR files as inputs — a single-file, structure-only
format dpdata reads reliably, which is enough to exercise globbing, tree
mirroring, the manifest, and skip-on-failure.
"""
import importlib
import json
import logging
from pathlib import Path

import pytest

from deepmd.dpa_tools.data.convert import batch_convert, convert, _glob_base
from deepmd.dpa_tools.data.validate import Issue

# The dpa_tools.data package re-exports the convert() function, which shadows
# the submodule name — grab the real module object for monkeypatching.
convert_mod = importlib.import_module("deepmd.dpa_tools.data.convert")


_POSCAR = """\
Cu O test
1.0
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
Cu O
1 1
Cartesian
0.0 0.0 0.0
1.0 1.0 1.0
"""


def _write_poscar(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_POSCAR)


# ---------------------------------------------------------------------------
# _glob_base
# ---------------------------------------------------------------------------

def test_glob_base_recursive_wildcard():
    assert _glob_base("calcs/**/OUTCAR") == Path("calcs")


def test_glob_base_single_wildcard():
    assert _glob_base("data/raw/*.xyz") == Path("data/raw")


def test_glob_base_no_wildcard_uses_parent(tmp_path):
    f = tmp_path / "only" / "POSCAR"
    _write_poscar(f)
    assert _glob_base(str(f)) == f.parent


# ---------------------------------------------------------------------------
# batch_convert
# ---------------------------------------------------------------------------

def test_batch_convert_mirrors_input_tree(tmp_path):
    _write_poscar(tmp_path / "in" / "a" / "POSCAR")
    _write_poscar(tmp_path / "in" / "b" / "c" / "POSCAR")
    out = tmp_path / "out"

    results = batch_convert(
        glob_pattern=str(tmp_path / "in" / "**" / "POSCAR"),
        output_dir=str(out),
        fmt="vasp/poscar",
        type_map=["Cu", "O"],
    )

    assert len(results) == 2
    # input tree mirrored, file stem used as the leaf system directory
    assert (out / "a" / "POSCAR" / "type.raw").exists()
    assert (out / "b" / "c" / "POSCAR" / "type.raw").exists()
    assert (out / "a" / "POSCAR" / "set.000" / "coord.npy").exists()
    # returned paths point at the created system dirs
    assert all(Path(r).is_dir() for r in results)


def test_batch_convert_writes_manifest(tmp_path):
    _write_poscar(tmp_path / "in" / "a" / "POSCAR")
    out = tmp_path / "out"
    batch_convert(
        glob_pattern=str(tmp_path / "in" / "**" / "POSCAR"),
        output_dir=str(out), fmt="vasp/poscar", type_map=["Cu", "O"],
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["fmt"] == "vasp/poscar"
    assert manifest["type_map"] == ["Cu", "O"]
    assert len(manifest["converted"]) == 1
    assert manifest["skipped"] == []
    assert manifest["converted"][0]["input"].endswith("POSCAR")


def test_batch_convert_skips_bad_file(tmp_path, caplog):
    _write_poscar(tmp_path / "in" / "good" / "POSCAR")
    bad = tmp_path / "in" / "bad" / "POSCAR"
    bad.parent.mkdir(parents=True)
    bad.write_text("garbage not a poscar\n")
    out = tmp_path / "out"

    with caplog.at_level(logging.WARNING, logger="dpa_tools"):
        results = batch_convert(
            glob_pattern=str(tmp_path / "in" / "**" / "POSCAR"),
            output_dir=str(out), fmt="vasp/poscar", type_map=["Cu", "O"],
        )

    # good file converted, bad file skipped and recorded
    assert len(results) == 1
    assert "good" in results[0]
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["converted"]) == 1
    assert len(manifest["skipped"]) == 1
    assert "bad" in manifest["skipped"][0]["input"]
    assert manifest["skipped"][0]["error"]
    assert "skipping" in caplog.text
    # the empty output subdir left by the failed convert is cleaned up
    assert not (out / "bad" / "POSCAR").exists()


def test_batch_convert_strict_fails_fast_on_bad_file(tmp_path):
    bad = tmp_path / "in" / "bad" / "POSCAR"
    bad.parent.mkdir(parents=True)
    bad.write_text("garbage not a poscar\n")
    out = tmp_path / "out"
    with pytest.raises(Exception):
        batch_convert(
            glob_pattern=str(tmp_path / "in" / "**" / "POSCAR"),
            output_dir=str(out), fmt="vasp/poscar",
            type_map=["Cu", "O"], strict=True,
        )


# ---------------------------------------------------------------------------
# convert() validation wiring
# ---------------------------------------------------------------------------

def test_convert_validate_true_runs_check(tmp_path, monkeypatch):
    _write_poscar(tmp_path / "POSCAR")
    seen = {}

    def _fake_check(data, strict=False):
        seen["is_system"] = hasattr(data, "data")  # dpdata.System
        seen["strict"] = strict
        return []

    monkeypatch.setattr(convert_mod, "check_data", _fake_check)
    out = convert(str(tmp_path / "POSCAR"), str(tmp_path / "out"),
                  fmt="vasp/poscar", type_map=["Cu", "O"], validate=True)
    assert seen["is_system"] is True  # check_data received a dpdata object
    assert seen["strict"] is False
    assert Path(out).exists()


def test_convert_validate_false_skips_check(tmp_path, monkeypatch):
    _write_poscar(tmp_path / "POSCAR")

    def _boom(*a, **k):
        raise AssertionError("check_data must not run when validate=False")

    monkeypatch.setattr(convert_mod, "check_data", _boom)
    out = convert(str(tmp_path / "POSCAR"), str(tmp_path / "out"),
                  fmt="vasp/poscar", type_map=["Cu", "O"], validate=False)
    assert Path(out).exists()


def test_convert_validation_issues_are_logged(tmp_path, monkeypatch, caplog):
    _write_poscar(tmp_path / "POSCAR")
    fake = Issue("error", "sys", "", "energies", "boom description")
    monkeypatch.setattr(convert_mod, "check_data",
                        lambda data, strict=False: [fake])
    with caplog.at_level(logging.WARNING, logger="dpa_tools"):
        convert(str(tmp_path / "POSCAR"), str(tmp_path / "out"),
                fmt="vasp/poscar", type_map=["Cu", "O"], validate=True)
    assert "boom description" in caplog.text


def test_convert_strict_passed_through(tmp_path, monkeypatch):
    _write_poscar(tmp_path / "POSCAR")
    seen = {}

    def _fake_check(path, strict=False):
        seen["strict"] = strict
        return []

    monkeypatch.setattr(convert_mod, "check_data", _fake_check)
    convert(str(tmp_path / "POSCAR"), str(tmp_path / "out"),
            fmt="vasp/poscar", type_map=["Cu", "O"],
            validate=True, strict=True)
    assert seen["strict"] is True
