# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for ``convert`` and the CSV-sniffing helpers."""

from __future__ import (
    annotations,
)

from pathlib import (
    Path,
)

import pytest

try:
    import rdkit  # noqa: F401

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

from dpa_adapt.data.convert import (
    _is_smiles_input,
    _sniff_csv,
    _sniff_xlsx,
    convert,
)

# ---------------------------------------------------------------------------
# CSV sniffing
# ---------------------------------------------------------------------------


class TestSniffCsv:
    def test_detects_smiles_column(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("id,SMILES,Property\n0,CCO,1.23\n1,c1ccccc1,4.56\n")
        assert _is_smiles_input(str(f)) is True

    def test_detects_smi_column(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("idx,smi,target\n0,CCO,1.0\n")
        assert _is_smiles_input(str(f)) is True

    def test_rejects_non_smiles_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("formula,energy\nH2O,-1.0\n")
        assert _is_smiles_input(str(f)) is False

    def test_non_csv_extension(self, tmp_path):
        f = tmp_path / "POSCAR"
        f.write_text("Si\n1.0\n0 0 0\n")
        assert _is_smiles_input(str(f)) is False

    def test_malformed_csv(self, tmp_path):
        f = tmp_path / "bad.csv"
        f.write_bytes(b"\x00\x01\x02")
        assert _sniff_csv(str(f)) is None

    def test_empty_csv(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        assert _sniff_csv(str(f)) is None


class TestSniffXlsx:
    @pytest.fixture(autouse=True)
    def _require_openpyxl(self):
        pytest.importorskip("openpyxl")

    @pytest.mark.parametrize("filename", ["data.xlsx", "data.xls"])
    def test_detects_smiles_column(self, tmp_path, filename):
        pd = pytest.importorskip("pandas")
        f = tmp_path / filename
        pd.DataFrame({"SMILES": ["CCO", "c1ccccc1"], "Prop": [1.0, 2.0]}).to_excel(
            f,
            index=False,
            engine="openpyxl",
        )
        assert _is_smiles_input(str(f)) is True

    def test_rejects_non_smiles_xlsx(self, tmp_path):
        pd = pytest.importorskip("pandas")
        f = tmp_path / "data.xlsx"
        pd.DataFrame({"formula": ["H2O"], "energy": [1.0]}).to_excel(
            f,
            index=False,
            engine="openpyxl",
        )
        assert _is_smiles_input(str(f)) is False

    def test_pandas_not_installed(self, tmp_path, monkeypatch):
        f = tmp_path / "data.xlsx"
        f.write_text("dummy")  # not a real xlsx, but we won't reach pandas
        monkeypatch.setitem(__import__("sys").modules, "pandas", None)
        assert _sniff_xlsx(str(f)) is None


# ---------------------------------------------------------------------------
# convert routing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
class TestAutoConvertSmiles:
    """convert routes CSV-with-SMILES to the SMILES pipeline."""

    def test_routes_csv_smiles_to_smiles_method(self, tmp_path):
        f = tmp_path / "mol.csv"
        f.write_text("SMILES,Property\nCCO,1.5\nCN,2.0\n")
        out = tmp_path / "npy"

        result = convert(str(f), str(out))

        assert result["method"] == "smiles"
        assert result["samples_used"] == 2
        assert "C" in result["type_map"]
        assert len(result["train_systems"]) > 0
        assert len(result["valid_systems"]) > 0

    def test_explicit_fmt_smiles_overrides_sniff(self, tmp_path):
        f = tmp_path / "mol.csv"
        f.write_text("SMILES,val\nC,1.0\nCC,2.0\n")
        out = tmp_path / "npy2"

        result = convert(str(f), str(out), fmt="smiles", property_col="val")

        assert result["method"] == "smiles"
        assert result["samples_used"] == 2
        assert "failed_rows" in result
        assert "skipped_zero" in result
        assert "skipped_overlap" in result

    def test_explicit_fmt_smiles_is_case_insensitive(self, tmp_path):
        f = tmp_path / "mol.csv"
        f.write_text("SMILES,val\nC,1.0\nCC,2.0\n")
        out = tmp_path / "npy3"

        result = convert(str(f), str(out), fmt="SMILES", property_col="val")

        assert result["method"] == "smiles"
        assert result["samples_used"] == 2


class TestAutoConvertStructure:
    """convert routes structure files through dpdata."""

    def test_routes_poscar_to_dpdata(self, tmp_path):
        f = tmp_path / "POSCAR"
        f.write_text("Si\n1.0\n5.43 0 0\n0 5.43 0\n0 0 5.43\nSi\n1\nCartesian\n0 0 0\n")
        out = tmp_path / "npy"

        result = convert(str(f), str(out))

        assert result["method"] == "dpdata"
        out_dir = result["output_dir"]
        assert (Path(out_dir) / "type.raw").exists()
        assert (Path(out_dir) / "set.000" / "coord.npy").exists()

    def test_explicit_fmt_passed_through(self, tmp_path):
        f = tmp_path / "POSCAR"
        f.write_text("Si\n1.0\n5.43 0 0\n0 5.43 0\n0 0 5.43\nSi\n1\nCartesian\n0 0 0\n")
        out = tmp_path / "npy2"

        result = convert(str(f), str(out), fmt="vasp/poscar")

        assert result["method"] == "dpdata"


class TestAutoConvertNoSmiles:
    """CSV without recognised SMILES column falls through to dpdata."""

    def test_falls_through_to_dpdata(self, tmp_path):
        f = tmp_path / "props.csv"
        f.write_text("formula,energy\nH2O,-1.0\n")
        out = tmp_path / "npy"

        # dpdata may or may not handle this, but it must NOT go to SMILES
        with pytest.raises(Exception):  # dpdata won't recognise it either
            convert(str(f), str(out))


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit not installed")
class TestSmoke:
    """Minimal round-trip: SMILES → npy → load_data."""

    def test_smiles_round_trip(self, tmp_path):
        from dpa_adapt.data.loader import (
            load_data,
        )

        f = tmp_path / "round.csv"
        f.write_text("SMILES,Property\nCCO,1.5\nCN,2.0\n")
        out = tmp_path / "npy"

        result = convert(
            str(f),
            str(out),
            property_name="homo",
            property_col="Property",
        )
        assert result["method"] == "smiles"

        # Verify one of the output systems is loadable and carries the label.
        systems = load_data(result["train_systems"])
        assert len(systems) > 0
        assert "homo" in systems[0].data
