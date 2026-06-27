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


# ---------------------------------------------------------------------------
# convert — formula pipeline (fmt="formula")
# ---------------------------------------------------------------------------


class TestAutoConvertFormula:
    """convert routes fmt="formula" to formula_to_npy."""

    def test_formula_fmt_routes_to_formula_pipeline(self, tmp_path, monkeypatch):
        """fmt="formula" with poscar → delegates to formula_to_npy."""
        csv = tmp_path / "comps.csv"
        csv.write_text("Ni0.5Fe0.5O2,1.23\n")
        poscar = tmp_path / "POSCAR"
        poscar.write_text(
            "Si\n1.0\n5.43 0 0\n0 5.43 0\n0 0 5.43\nSi\n1\nCartesian\n0 0 0\n"
        )
        out = tmp_path / "npy"
        fake_sys_dir = str(out / "sys_0000")

        # The convert() function does "from .formula import formula_to_npy"
        # at call time, so we mock the formula module's attribute directly.
        def _fake_formula_to_npy(**kwargs):
            Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
            return [fake_sys_dir]

        monkeypatch.setattr(
            "dpa_adapt.data.formula.formula_to_npy",
            _fake_formula_to_npy,
        )

        result = convert(
            str(csv),
            str(out),
            fmt="formula",
            poscar=str(poscar),
            formula_col=0,
            property_col=1,
            property_name="bandgap",
            seed=123,
        )

        assert result["method"] == "formula"
        assert result["output_dir"] == str(out.resolve())
        assert result["output_systems"] == [fake_sys_dir]

    def test_formula_fmt_base_element_passed_through(self, tmp_path, monkeypatch):
        """fmt="formula" with explicit base_element passes it through."""
        csv = tmp_path / "comps.csv"
        csv.write_text("Ni0.8Fe0.2O2,0.5\n")
        poscar = tmp_path / "POSCAR"
        poscar.write_text(
            "NiO\n1.0\n4.17 0 0\n0 4.17 0\n0 0 4.17\nNi O\n1 1\nCartesian\n0 0 0\n0.5 0.5 0.5\n"
        )
        out = tmp_path / "npy"

        captured = {}

        def _fake_formula_to_npy(**kwargs):
            captured.update(kwargs)
            Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
            return [str(out / "sys_0000")]

        monkeypatch.setattr(
            "dpa_adapt.data.formula.formula_to_npy",
            _fake_formula_to_npy,
        )

        convert(
            str(csv),
            str(out),
            fmt="formula",
            poscar=str(poscar),
            base_element="Ni",
            sets=5,
            seed=99,
        )

        assert captured["base_element"] == "Ni"
        assert captured["sets"] == 5
        assert captured["seed"] == 99
        assert captured["csv_path"] == str(csv)
        assert captured["poscar"] == str(poscar)

    def test_formula_fmt_base_element_none_by_default(self, tmp_path, monkeypatch):
        """Convert defaults base_element=None → formula_to_npy infers it."""
        csv = tmp_path / "comps.csv"
        csv.write_text("Ni0.5Fe0.5O2,1.0\n")
        poscar = tmp_path / "POSCAR"
        poscar.write_text(
            "NiO\n1.0\n4.17 0 0\n0 4.17 0\n0 0 4.17\nNi O\n1 1\nCartesian\n0 0 0\n0.5 0.5 0.5\n"
        )
        out = tmp_path / "npy"

        captured = {}

        def _fake_formula_to_npy(**kwargs):
            captured.update(kwargs)
            Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
            return [str(out / "sys_0000")]

        monkeypatch.setattr(
            "dpa_adapt.data.formula.formula_to_npy",
            _fake_formula_to_npy,
        )

        # Call WITHOUT base_element — should pass None through.
        convert(str(csv), str(out), fmt="formula", poscar=str(poscar))

        assert captured["base_element"] is None

    def test_formula_fmt_verbose_prints_system_count(
        self, tmp_path, monkeypatch, caplog
    ):
        """fmt="formula" with verbose=True logs system count."""
        csv = tmp_path / "comps.csv"
        csv.write_text("Ni0.5Fe0.5O2,1.0\nGd0.5Fe0.5O2,2.0\n")
        poscar = tmp_path / "POSCAR"
        poscar.write_text(
            "NiO\n1.0\n4.17 0 0\n0 4.17 0\n0 0 4.17\nNi O\n1 1\nCartesian\n0 0 0\n0.5 0.5 0.5\n"
        )
        out = tmp_path / "npy"

        def _fake_formula_to_npy(**kwargs):
            Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
            return ["/tmp/fake/sys_0000", "/tmp/fake/sys_0001"]

        monkeypatch.setattr(
            "dpa_adapt.data.formula.formula_to_npy",
            _fake_formula_to_npy,
        )

        with caplog.at_level(logging.INFO, logger="dpa_adapt"):
            convert(str(csv), str(out), fmt="formula", poscar=str(poscar), verbose=True)

        assert "2 systems" in caplog.text


# ---------------------------------------------------------------------------
# parse_formula and infer_base_element (formula pipeline helpers)
# ---------------------------------------------------------------------------


class TestParseFormula:
    """Unit tests for formula string parsing."""

    def test_parse_simple_binary(self):
        from dpa_adapt.data.formula import (
            parse_formula,
        )

        result = parse_formula("Ni0.65Gd0.35O2H1")
        assert pytest.approx(result.get("Ni", 0)) == 0.65
        assert pytest.approx(result.get("Gd", 0)) == 0.35
        assert result["O"] == 2.0
        assert result["H"] == 1.0

    def test_parse_base_element_inferred_as_remainder(self):
        from dpa_adapt.data.formula import (
            parse_formula,
        )

        # Co0.10Yb0.05 totals 0.15; remainder assigned to base_element=Ni
        result = parse_formula("Co0.10Yb0.05O2H1", base_element="Ni")
        assert pytest.approx(result.get("Ni", 0)) == pytest.approx(0.85)
        assert pytest.approx(result.get("Co", 0)) == pytest.approx(0.10)
        assert pytest.approx(result.get("Yb", 0)) == pytest.approx(0.05)

    def test_parse_base_element_not_assigned_when_total_is_one(self):
        from dpa_adapt.data.formula import (
            parse_formula,
        )

        result = parse_formula("Ni0.65Gd0.35O2", base_element="Fe")
        assert "Fe" not in result
        assert (
            pytest.approx(sum(v for k, v in result.items() if k not in ("O", "H")))
            == 1.0
        )

    def test_parse_empty_formula_raises(self):
        from dpa_adapt.data.formula import (
            parse_formula,
        )

        with pytest.raises(ValueError, match="Could not parse"):
            parse_formula("")

    def test_parse_single_element_implicit_one(self):
        from dpa_adapt.data.formula import (
            parse_formula,
        )

        # "C" with no number → treated as fraction 1.0
        result = parse_formula("O2H1")
        assert result["O"] == 2.0
        assert result["H"] == 1.0

    def test_parse_substitution_sublattice_normalised_to_one(self):
        from dpa_adapt.data.formula import (
            parse_formula,
        )

        # Raw: Ni0.13, Gd0.03, Fe0.02, Co0.01, Yb0.01 — sum=0.20
        # After normalisation: each divided by 0.20
        result = parse_formula("Ni0.13Gd0.03Fe0.02Co0.01Yb0.01O2H1")
        total_sub = sum(v for k, v in result.items() if k not in ("O", "H"))
        assert pytest.approx(total_sub) == 1.0


class TestInferBaseElement:
    """Unit tests for base_element auto-inference from template atoms."""

    def test_returns_most_frequent_non_oh_element(self):
        from dpa_adapt.data.formula import (
            infer_base_element,
        )

        symbols = ["Ni", "Ni", "Ni", "O", "O", "H"]
        assert infer_base_element(symbols) == "Ni"

    def test_skips_oh_when_other_element_present(self):
        from dpa_adapt.data.formula import (
            infer_base_element,
        )

        symbols = ["O", "O", "H", "H", "Fe", "Fe", "Fe"]
        assert infer_base_element(symbols) == "Fe"

    def test_returns_none_when_only_oh(self):
        from dpa_adapt.data.formula import (
            infer_base_element,
        )

        symbols = ["O", "H", "O", "H"]
        assert infer_base_element(symbols) is None

    def test_returns_none_for_empty_list(self):
        from dpa_adapt.data.formula import (
            infer_base_element,
        )

        assert infer_base_element([]) is None

    def test_tie_gives_first_encountered(self):
        from dpa_adapt.data.formula import (
            infer_base_element,
        )

        # Ni and Fe each appear twice, Ni encountered first.
        symbols = ["Ni", "Ni", "Fe", "Fe", "O", "O"]
        assert infer_base_element(symbols) == "Ni"
