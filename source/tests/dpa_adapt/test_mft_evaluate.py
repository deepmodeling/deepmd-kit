# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for dpa_adapt.mft.MFTFineTuner.evaluate output parsing and pipeline."""

from __future__ import (
    annotations,
)

import os
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
)

import pytest

from dpa_adapt.mft import (
    MFTFineTuner,
)

DUMMY_TYPE_MAP = ["H", "C", "N", "O"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_systems(tmp_path, prefix: str, n: int) -> str:
    """Create n empty system dirs and return a glob pattern matching them."""
    root = tmp_path / prefix
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / f"sys_{i:03d}").mkdir()
    return str(root / "sys_*")


def _make_finetuner(tmp_path, max_steps=100):
    """
    Build an MFTFineTuner without going through __init__'s ckpt-reading path.
    We bypass __init__ because fitting_net auto-read calls torch.load on the
    pretrained ckpt, which we don't have in unit tests.
    """
    ft = MFTFineTuner.__new__(MFTFineTuner)
    ft.pretrained = str(tmp_path / "dummy.pt")
    ft.aux_branch = "SPICE2"
    ft.aux_prob = 0.5
    ft.aux_type_map = DUMMY_TYPE_MAP
    ft.downstream_type_map = DUMMY_TYPE_MAP
    ft.fitting_net_params = {}
    # Paper property-mode evaluation: downstream head is named "property".
    ft.downstream_task_type = "property"
    ft.property_name = "homo"
    ft.task_dim = 1
    ft.intensive = True
    ft.learning_rate = 1e-3
    ft.stop_lr = 1e-5
    ft.max_steps = max_steps
    ft.batch_size = "auto:32"
    ft.seed = 42
    ft.output_dir = str(tmp_path / "out")
    ft.save_freq = 10
    ft.disp_freq = 10
    ft.train_data = None
    ft.aux_data = None
    ft.valid_data = None
    os.makedirs(ft.output_dir, exist_ok=True)
    return ft


# ---------------------------------------------------------------------------
# Parser: real DeepMD-kit 3.1.3 output shape
# ---------------------------------------------------------------------------


def test_parse_real_dp_output_shape():
    """The real `dp --pt test` output prints both 'Energy MAE' (per-molecule)
    and 'Energy MAE/Natoms' (per-atom). The parser must pick only the
    per-molecule one.
    """
    stdout = (
        "[2026-05-19 INFO] # number of test data : 1000\n"
        "[2026-05-19 INFO] Energy MAE         : 4.314543e-02 eV\n"
        "[2026-05-19 INFO] Energy MAE/Natoms  : 3.318879e-03 eV\n"
        "[2026-05-19 INFO] Energy RMSE        : 6.000000e-02 eV\n"
        "[2026-05-19 INFO] Energy RMSE/Natoms : 4.500000e-03 eV\n"
    )
    out = MFTFineTuner._parse_test_output(stdout)
    assert out["mae"] == pytest.approx(4.314543e-02)
    assert out["rmse"] == pytest.approx(6.000000e-02)


def test_parse_excludes_natoms_variant_explicitly():
    """If only the /Natoms variant appears, the parser should NOT match it.
    This guards against a regex that accidentally allows /Natoms through.
    """
    stdout = (
        "[INFO] Energy MAE/Natoms  : 1.234567e-03 eV\n"
        "[INFO] Energy RMSE/Natoms : 2.345678e-03 eV\n"
    )
    with pytest.raises(RuntimeError, match="Could not parse"):
        MFTFineTuner._parse_test_output(stdout)


# ---------------------------------------------------------------------------
# Parser: weighted-average behavior (must take LAST match)
# ---------------------------------------------------------------------------


def test_parse_takes_weighted_average_last_match():
    """Dp --pt test prints per-system blocks followed by a
    'weighted average of errors' block. Parser must return the weighted
    average (the LAST occurrence), not the first per-system value.
    """
    stdout = (
        "[INFO] # ---------------system 0--------------\n"
        "[INFO] Energy MAE         : 1.00e-01 eV\n"
        "[INFO] Energy RMSE        : 2.00e-01 eV\n"
        "[INFO] # ---------------system 1--------------\n"
        "[INFO] Energy MAE         : 5.00e-01 eV\n"
        "[INFO] Energy RMSE        : 6.00e-01 eV\n"
        "[INFO] # ----------weighted average of errors-----------\n"
        "[INFO] Energy MAE         : 3.50e-01 eV\n"
        "[INFO] Energy RMSE        : 4.50e-01 eV\n"
    )
    out = MFTFineTuner._parse_test_output(stdout)
    # Must be the weighted-average (final) values.
    assert out["mae"] == pytest.approx(3.50e-01)
    assert out["rmse"] == pytest.approx(4.50e-01)


# ---------------------------------------------------------------------------
# Parser: n_systems extraction
# ---------------------------------------------------------------------------


def test_parse_extracts_n_systems():
    stdout = (
        "[INFO] # number of systems : 7\n"
        "[INFO] Energy MAE  : 1.00e-02 eV\n"
        "[INFO] Energy RMSE : 2.00e-02 eV\n"
    )
    out = MFTFineTuner._parse_test_output(stdout)
    assert out["n_systems"] == 7


def test_parse_n_systems_falls_back_to_resolved_count():
    """If the 'number of systems' line is missing, fall back to the count of
    resolved system paths so the caller still gets a usable number.
    """
    stdout = "[INFO] Energy MAE  : 1.00e-02 eV\n[INFO] Energy RMSE : 2.00e-02 eV\n"
    out = MFTFineTuner._parse_test_output(stdout, n_resolved=42)
    assert out["n_systems"] == 42


# ---------------------------------------------------------------------------
# Parser: failure mode (was previously silent NaN — must now raise)
# ---------------------------------------------------------------------------


def test_parse_failure_raises_runtimeerror():
    """When dp test produced no Energy MAE/RMSE lines (the Bug-1 all-zero
    failure mode), raise RuntimeError instead of silently returning NaN.
    """
    stdout = "no MAE or RMSE lines here, just garbage"
    with pytest.raises(RuntimeError) as exc_info:
        MFTFineTuner._parse_test_output(stdout)
    msg = str(exc_info.value)
    assert "Could not parse" in msg
    # Tail should be included for diagnostics.
    assert "garbage" in msg


def test_parse_failure_includes_tail_of_output():
    """Long unparseable input: tail of last 100 lines must appear in the
    error message so the user can diagnose without grepping logs.
    """
    lines = [f"line_{i}" for i in range(200)]
    stdout = "\n".join(lines)
    with pytest.raises(RuntimeError) as exc_info:
        MFTFineTuner._parse_test_output(stdout)
    msg = str(exc_info.value)
    # Last line should appear; very early lines should be trimmed.
    assert "line_199" in msg
    assert "line_0\n" not in msg


# ---------------------------------------------------------------------------
# Parser: scientific notation handling
# ---------------------------------------------------------------------------


def test_parse_scientific_notation():
    stdout = (
        "[INFO] Energy MAE         : 4.314543e-02 eV\n"
        "[INFO] Energy RMSE        : 1.23E+01 eV\n"
    )
    out = MFTFineTuner._parse_test_output(stdout)
    assert out["mae"] == pytest.approx(4.314543e-02)
    assert out["rmse"] == pytest.approx(1.23e01)


# ---------------------------------------------------------------------------
# Parser: property-mode output (PROPERTY MAE / PROPERTY RMSE)
# ---------------------------------------------------------------------------


def test_parse_property_output_weighted_average():
    """Property-task dp test prints per-system blocks then a
    'weighted average of errors' block. Parser must return the LAST match.
    """
    stdout = (
        "[INFO] # ---------------system 0--------------\n"
        "[INFO] PROPERTY MAE            : 2.395307e-03 units\n"
        "[INFO] PROPERTY RMSE           : 2.395307e-03 units\n"
        "[INFO] # ---------------system 1--------------\n"
        "[INFO] PROPERTY MAE            : 1.500000e-03 units\n"
        "[INFO] PROPERTY RMSE           : 1.500000e-03 units\n"
        "[INFO] # ----------weighted average of errors----------- \n"
        "[INFO] # number of systems : 291\n"
        "[INFO] PROPERTY MAE            : 1.972088e-03 units\n"
        "[INFO] PROPERTY RMSE           : 2.837059e-03 units\n"
    )
    out = MFTFineTuner._parse_test_output(stdout)
    assert out["mae"] == pytest.approx(1.972088e-03)
    assert out["rmse"] == pytest.approx(2.837059e-03)
    assert out["n_systems"] == 291
    assert "PROPERTY" in out["_parser_pattern_used"]


def test_parse_property_scientific_notation():
    stdout = (
        "[INFO] PROPERTY MAE  : 1.23e-04 units\n[INFO] PROPERTY RMSE : 5.67E+02 units\n"
    )
    out = MFTFineTuner._parse_test_output(stdout)
    assert out["mae"] == pytest.approx(1.23e-04)
    assert out["rmse"] == pytest.approx(5.67e02)


def test_parse_property_n_systems_extraction():
    stdout = (
        "[INFO] # number of systems : 42\n"
        "[INFO] PROPERTY MAE  : 0.01 units\n"
        "[INFO] PROPERTY RMSE : 0.02 units\n"
    )
    out = MFTFineTuner._parse_test_output(stdout)
    assert out["n_systems"] == 42


def test_parse_property_n_systems_fallback():
    stdout = "[INFO] PROPERTY MAE  : 0.01 units\n[INFO] PROPERTY RMSE : 0.02 units\n"
    out = MFTFineTuner._parse_test_output(stdout, n_resolved=99)
    assert out["n_systems"] == 99


# ---------------------------------------------------------------------------
# evaluate(): end-to-end pipeline with mocked subprocess
# ---------------------------------------------------------------------------


def test_evaluate_freezes_then_tests(tmp_path):
    """evaluate() must (a) call dp freeze first to produce frozen .pth,
    (b) then call dp test with -m pointing to that .pth, (c) parse output.
    """
    ft = _make_finetuner(tmp_path, max_steps=100)
    # Pretend training produced a ckpt
    (Path(ft.output_dir) / "model.ckpt-100.pt").write_bytes(b"")
    test_glob = _make_systems(tmp_path, "test_sys", 5)

    canned_test_output = (
        "[INFO] # number of systems : 5\n"
        "[INFO] # number of test data : 50\n"
        "[INFO] Energy MAE         : 1.234567e-02 eV\n"
        "[INFO] Energy MAE/Natoms  : 9.876543e-04 eV\n"
        "[INFO] Energy RMSE        : 2.345678e-02 eV\n"
        "[INFO] Energy RMSE/Natoms : 1.234567e-03 eV\n"
    )

    calls = []

    class _Result:
        def __init__(self, stdout="", stderr="", rc=0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = rc

    def _fake_run(cmd, *args, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        # First call is freeze (shell command); simulate by creating frozen.pth
        if isinstance(cmd, str) and "freeze" in cmd:
            cwd = kwargs.get("cwd", ".")
            Path(cwd, "frozen_property.pth").write_bytes(b"")
            return _Result(stdout="frozen ok", stderr="", rc=0)
        # Second call is dp test
        return _Result(stdout="", stderr=canned_test_output, rc=0)

    with patch("subprocess.run", side_effect=_fake_run):
        out = ft.evaluate(test_glob)

    # 1. freeze was called first as a shell command with cwd=output_dir
    assert len(calls) == 2
    assert isinstance(calls[0]["cmd"], str)
    assert "dp --pt freeze" in calls[0]["cmd"]
    assert "--head property" in calls[0]["cmd"]
    assert calls[0]["kwargs"].get("cwd") == ft.output_dir

    # 2. dp test was called with frozen .pth via -m, list-form cmd
    test_cmd = calls[1]["cmd"]
    assert isinstance(test_cmd, list)
    m_idx = test_cmd.index("-m")
    assert test_cmd[m_idx + 1].endswith("frozen_property.pth")
    assert "-f" in test_cmd
    assert "-s" not in test_cmd

    # 3. Parsed values are per-molecule MAE/RMSE, not /Natoms.
    assert out["mae"] == pytest.approx(1.234567e-02)
    assert out["rmse"] == pytest.approx(2.345678e-02)
    assert out["n_systems"] == 5


def test_evaluate_skips_freeze_if_pth_exists(tmp_path):
    """If frozen_property.pth already exists, do NOT call dp freeze again."""
    ft = _make_finetuner(tmp_path, max_steps=100)
    (Path(ft.output_dir) / "model.ckpt-100.pt").write_bytes(b"")
    (Path(ft.output_dir) / "frozen_property.pth").write_bytes(b"")
    test_glob = _make_systems(tmp_path, "test_skip", 3)

    canned = (
        "[INFO] # number of systems : 3\n"
        "[INFO] Energy MAE  : 5.0e-03 eV\n"
        "[INFO] Energy RMSE : 6.0e-03 eV\n"
    )

    calls = []

    class _Result:
        stdout = ""
        stderr = canned
        returncode = 0

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return _Result()

    with patch("subprocess.run", side_effect=_fake_run):
        out = ft.evaluate(test_glob)

    assert len(calls) == 1, f"Expected only dp test, got {len(calls)} calls"
    assert isinstance(calls[0], list)
    assert calls[0][:3] == ["dp", "--pt", "test"]
    assert out["mae"] == pytest.approx(5.0e-03)


def test_evaluate_freeze_failure_raises(tmp_path):
    """If dp freeze fails, evaluate() must raise RuntimeError with diagnostics
    rather than proceeding into a doomed dp test.
    """
    ft = _make_finetuner(tmp_path, max_steps=100)
    (Path(ft.output_dir) / "model.ckpt-100.pt").write_bytes(b"")
    test_glob = _make_systems(tmp_path, "test_fz_fail", 2)

    class _Result:
        stdout = "freeze stdout"
        stderr = "freeze failed: missing branch"
        returncode = 1

    with patch("subprocess.run", return_value=_Result()):
        with pytest.raises(RuntimeError, match="freeze"):
            ft.evaluate(test_glob)


def test_evaluate_accepts_single_path(tmp_path):
    """A single non-glob string path should be written verbatim into the
    datafile (single line) and passed via -f.
    """
    ft = _make_finetuner(tmp_path, max_steps=100)
    (Path(ft.output_dir) / "model.ckpt-100.pt").write_bytes(b"")
    (Path(ft.output_dir) / "frozen_property.pth").write_bytes(b"")

    single = tmp_path / "single_sys"
    single.mkdir()
    test_data = str(single)

    canned = (
        "[INFO] # number of systems : 1\n"
        "[INFO] Energy MAE  : 7.0e-03 eV\n"
        "[INFO] Energy RMSE : 8.0e-03 eV\n"
    )

    captured = {}

    class _Result:
        stdout = ""
        stderr = canned
        returncode = 0

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _Result()

    with patch("subprocess.run", side_effect=_fake_run):
        out = ft.evaluate(test_data)

    cmd = captured["cmd"]
    f_idx = cmd.index("-f")
    datafile = cmd[f_idx + 1]
    lines = [l for l in open(datafile).read().split("\n") if l.strip()]
    assert lines == [test_data]
    assert out["mae"] == pytest.approx(7.0e-03)
    assert out["n_systems"] == 1


def test_evaluate_accepts_list(tmp_path):
    """A list of paths should be written one-per-line into the datafile."""
    ft = _make_finetuner(tmp_path, max_steps=100)
    (Path(ft.output_dir) / "model.ckpt-100.pt").write_bytes(b"")
    (Path(ft.output_dir) / "frozen_property.pth").write_bytes(b"")

    paths = []
    for i in range(4):
        d = tmp_path / f"list_sys_{i}"
        d.mkdir()
        paths.append(str(d))

    canned = (
        "[INFO] # number of systems : 4\n"
        "[INFO] Energy MAE  : 9.0e-03 eV\n"
        "[INFO] Energy RMSE : 1.0e-02 eV\n"
    )

    captured = {}

    class _Result:
        stdout = ""
        stderr = canned
        returncode = 0

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _Result()

    with patch("subprocess.run", side_effect=_fake_run):
        out = ft.evaluate(paths)

    cmd = captured["cmd"]
    datafile = cmd[cmd.index("-f") + 1]
    lines = [l for l in open(datafile).read().split("\n") if l.strip()]
    assert lines == paths
    assert out["n_systems"] == 4


def test_evaluate_missing_ckpt_raises(tmp_path):
    """If no model.ckpt-{max_steps}.pt exists and frozen.pth also missing,
    _freeze_ckpt must raise rather than silently call freeze and explode.
    """
    ft = _make_finetuner(tmp_path, max_steps=100)
    test_glob = _make_systems(tmp_path, "test_no_ckpt", 2)

    with pytest.raises(RuntimeError, match="not found"):
        ft.evaluate(test_glob)
