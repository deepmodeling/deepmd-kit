# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for convert() routing and validation wiring.

Uses hand-written VASP POSCAR files as inputs — a single-file, structure-only
format dpdata reads reliably, which is enough to exercise globbing, tree
mirroring, the manifest, and skip-on-failure.
"""

import importlib
import json
import logging
from pathlib import (
    Path,
)

import pytest

from dpa_adapt.data.convert import (
    _glob_base,
    convert,
)
from dpa_adapt.data.validate import (
    Issue,
)

# The dpa_adapt.data package re-exports the convert() function, which shadows
# the submodule name — grab the real module object for monkeypatching.
convert_mod = importlib.import_module("dpa_adapt.data.convert")


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
# convert() glob batch routing
# ---------------------------------------------------------------------------


def test_convert_glob_mirrors_input_tree(tmp_path):
    _write_poscar(tmp_path / "in" / "a" / "POSCAR")
    _write_poscar(tmp_path / "in" / "b" / "c" / "POSCAR")
    out = tmp_path / "out"

    result = convert(
        str(tmp_path / "in" / "**" / "POSCAR"),
        str(out),
        fmt="vasp/poscar",
        type_map=["Cu", "O"],
    )
    results = result["output_dirs"]

    assert result["method"] == "batch_dpdata"
    assert len(results) == 2
    # input tree mirrored, file stem used as the leaf system directory
    assert (out / "a" / "POSCAR" / "type.raw").exists()
    assert (out / "b" / "c" / "POSCAR" / "type.raw").exists()
    assert (out / "a" / "POSCAR" / "set.000" / "coord.npy").exists()
    # returned paths point at the created system dirs
    assert all(Path(r).is_dir() for r in results)


def test_convert_glob_writes_manifest(tmp_path):
    _write_poscar(tmp_path / "in" / "a" / "POSCAR")
    out = tmp_path / "out"
    result = convert(
        str(tmp_path / "in" / "**" / "POSCAR"),
        str(out),
        fmt="vasp/poscar",
        type_map=["Cu", "O"],
    )
    assert result["manifest"] == str(out.resolve() / "manifest.json")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["fmt"] == "vasp/poscar"
    assert manifest["type_map"] == ["Cu", "O"]
    assert len(manifest["converted"]) == 1
    assert manifest["skipped"] == []
    assert manifest["converted"][0]["input"].endswith("POSCAR")


def test_convert_glob_skips_bad_file(tmp_path, caplog):
    _write_poscar(tmp_path / "in" / "good" / "POSCAR")
    bad = tmp_path / "in" / "bad" / "POSCAR"
    bad.parent.mkdir(parents=True)
    bad.write_text("garbage not a poscar\n")
    out = tmp_path / "out"

    with caplog.at_level(logging.WARNING, logger="dpa_adapt"):
        result = convert(
            str(tmp_path / "in" / "**" / "POSCAR"),
            str(out),
            fmt="vasp/poscar",
            type_map=["Cu", "O"],
        )
    results = result["output_dirs"]

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


def test_convert_glob_strict_fails_fast_on_bad_file(tmp_path):
    bad = tmp_path / "in" / "bad" / "POSCAR"
    bad.parent.mkdir(parents=True)
    bad.write_text("garbage not a poscar\n")
    out = tmp_path / "out"
    with pytest.raises(Exception):
        convert(
            str(tmp_path / "in" / "**" / "POSCAR"),
            str(out),
            fmt="vasp/poscar",
            type_map=["Cu", "O"],
            strict=True,
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
    result = convert(
        str(tmp_path / "POSCAR"),
        str(tmp_path / "out"),
        fmt="vasp/poscar",
        type_map=["Cu", "O"],
        validate=True,
    )
    out = result["output_dir"]
    assert seen["is_system"] is True  # check_data received a dpdata object
    assert seen["strict"] is False
    assert Path(out).exists()


def test_convert_validate_false_skips_check(tmp_path, monkeypatch):
    _write_poscar(tmp_path / "POSCAR")

    def _boom(*a, **k):
        raise AssertionError("check_data must not run when validate=False")

    monkeypatch.setattr(convert_mod, "check_data", _boom)
    result = convert(
        str(tmp_path / "POSCAR"),
        str(tmp_path / "out"),
        fmt="vasp/poscar",
        type_map=["Cu", "O"],
        validate=False,
    )
    out = result["output_dir"]
    assert Path(out).exists()


def test_convert_validation_issues_are_logged(tmp_path, monkeypatch, caplog):
    _write_poscar(tmp_path / "POSCAR")
    fake = Issue("error", "sys", "", "energies", "boom description")
    monkeypatch.setattr(convert_mod, "check_data", lambda data, strict=False: [fake])
    with caplog.at_level(logging.WARNING, logger="dpa_adapt"):
        convert(
            str(tmp_path / "POSCAR"),
            str(tmp_path / "out"),
            fmt="vasp/poscar",
            type_map=["Cu", "O"],
            validate=True,
        )
    assert "boom description" in caplog.text


def test_convert_strict_passed_through(tmp_path, monkeypatch):
    _write_poscar(tmp_path / "POSCAR")
    seen = {}

    def _fake_check(path, strict=False):
        seen["strict"] = strict
        return []

    monkeypatch.setattr(convert_mod, "check_data", _fake_check)
    convert(
        str(tmp_path / "POSCAR"),
        str(tmp_path / "out"),
        fmt="vasp/poscar",
        type_map=["Cu", "O"],
        validate=True,
        strict=True,
    )
    assert seen["strict"] is True


# ---------------------------------------------------------------------------
# convert() glob support
# ---------------------------------------------------------------------------


def test_convert_glob_single_match(tmp_path):
    """Pass a glob pattern that matches exactly one file → batch output."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_poscar(raw_dir / "input.sdf")

    out = tmp_path / "out"
    result = convert(
        str(raw_dir / "*.sdf"),
        str(out),
        fmt="vasp/poscar",
        type_map=["Cu", "O"],
        validate=False,
    )
    assert result["method"] == "batch_dpdata"
    assert len(result["output_dirs"]) == 1
    system_dir = out / "input"
    assert system_dir.is_dir()
    assert (system_dir / "type.raw").exists()
    assert (system_dir / "set.000" / "coord.npy").exists()
    assert (out / "manifest.json").exists()


def test_convert_glob_multi_match(tmp_path):
    """Pass a glob pattern matching 3 files → mirrored batch output."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in ("a.sdf", "b.sdf", "c.sdf"):
        _write_poscar(raw_dir / name)

    out = tmp_path / "out"
    result = convert(
        str(raw_dir / "*.sdf"),
        str(out),
        fmt="vasp/poscar",
        type_map=["Cu", "O"],
        validate=False,
    )
    assert result["method"] == "batch_dpdata"
    assert len(result["output_dirs"]) == 3
    for sub in ("a", "b", "c"):
        sub_dir = out / sub
        assert sub_dir.is_dir(), f"missing {sub}"
        assert (sub_dir / "type.raw").exists()
        assert (sub_dir / "set.000" / "coord.npy").exists()
    subdirs = [p.name for p in out.iterdir() if p.is_dir()]
    assert sorted(subdirs) == ["a", "b", "c"]


def test_convert_glob_no_match(tmp_path):
    """Pass a glob pattern with no matches → FileNotFoundError."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No files matched pattern"):
        convert(
            str(raw_dir / "*.sdf"),
            str(tmp_path / "out"),
            fmt="vasp/poscar",
            type_map=["Cu", "O"],
            validate=False,
        )


def test_convert_literal_path_unchanged(tmp_path):
    """Pass a literal path with no wildcards → works as before."""
    _write_poscar(tmp_path / "POSCAR")
    out = tmp_path / "out"
    result = convert(
        str(tmp_path / "POSCAR"),
        str(out),
        fmt="vasp/poscar",
        type_map=["Cu", "O"],
        validate=False,
    )
    assert result["method"] == "dpdata"
    assert Path(result["output_dir"]).is_dir()
    assert (Path(result["output_dir"]) / "type.raw").exists()
