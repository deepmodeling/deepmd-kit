# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for dpa_adapt.trainer.DPATrainer."""

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

from dpa_adapt.trainer import (
    DPATrainer,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

DUMMY_TYPE_MAP = ["H", "C", "N", "O"]


def _make_systems(tmp_path, prefix: str, n: int) -> str:
    """Create n empty system dirs and return a glob pattern matching them."""
    root = tmp_path / prefix
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / f"sys_{i:03d}").mkdir()
    return str(root / "sys_*")


def _fake_descriptor_sd() -> dict:
    """Minimal checkpoint state_dict with the descriptor path the trainer reads."""
    descriptor = {
        "type": "dpa3",
        "repflow": {
            "n_dim": 128,
            "e_dim": 64,
            "a_dim": 32,
            "nlayers": 16,
            "e_rcut": 6.0,
            "e_rcut_smth": 5.3,
            "e_sel": 1200,
            "a_rcut": 4.0,
            "a_rcut_smth": 3.5,
            "a_sel": 300,
            "axis_neuron": 4,
            "skip_stat": True,
            "a_compress_rate": 1,
            "a_compress_e_rate": 2,
            "a_compress_use_split": True,
            "update_angle": True,
            "smooth_edge_update": True,
            "use_dynamic_sel": True,
            "sel_reduce_factor": 10.0,
            "update_style": "res_residual",
            "update_residual": 0.1,
            "update_residual_init": "const",
            "n_multi_edge_message": 1,
            "optim_update": True,
            "use_exp_switch": True,
        },
        "activation_function": "custom_silu:3.0",
        "precision": "float32",
        "use_tebd_bias": False,
        "concat_output_tebd": False,
        "exclude_types": [],
        "env_protection": 0.0,
        "trainable": True,
        "use_econf_tebd": False,
    }
    return {
        "model": {
            "_extra_state": {
                "model_params": {
                    "shared_dict": {"dpa3_descriptor": descriptor},
                }
            }
        }
    }


@pytest.fixture
def systems(tmp_path):
    """Build train + valid system globs in a tmp directory."""
    train_glob = _make_systems(tmp_path, "train", 60)
    valid_glob = _make_systems(tmp_path, "valid", 60)
    return train_glob, valid_glob


@pytest.fixture
def dummy_ckpt(tmp_path):
    """Create an empty file to act as a 'pretrained' checkpoint path."""
    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"")
    return str(ckpt)


def _patch_torch_load():
    """Patch torch.load to return our fake descriptor state_dict."""
    return patch("torch.load", lambda *a, **kw: _fake_descriptor_sd())


# ---------------------------------------------------------------------------
# 1. init validation
# ---------------------------------------------------------------------------


def test_init_validation(tmp_path, systems):
    train_glob, valid_glob = systems

    # train_systems is None
    with pytest.raises(ValueError, match="train_systems"):
        DPATrainer(
            valid_systems=valid_glob,
            type_map=DUMMY_TYPE_MAP,
        )

    # type_map is None
    with pytest.raises(ValueError, match="type_map"):
        DPATrainer(
            train_systems=train_glob,
            valid_systems=valid_glob,
        )

    # freeze_backbone=True without pretrained
    with pytest.raises(ValueError, match="LP requires"):
        DPATrainer(
            train_systems=train_glob,
            valid_systems=valid_glob,
            type_map=DUMMY_TYPE_MAP,
            freeze_backbone=True,
        )

    # pretrained path does not exist
    with pytest.raises(ValueError, match="not found"):
        DPATrainer(
            pretrained=str(tmp_path / "does_not_exist.pt"),
            train_systems=train_glob,
            valid_systems=valid_glob,
            type_map=DUMMY_TYPE_MAP,
        )


# ---------------------------------------------------------------------------
# 2. FT config
# ---------------------------------------------------------------------------


def test_config_ft(systems, dummy_ckpt, tmp_path):
    train_glob, valid_glob = systems
    t = DPATrainer(
        pretrained=dummy_ckpt,
        freeze_backbone=False,
        train_systems=train_glob,
        valid_systems=valid_glob,
        type_map=DUMMY_TYPE_MAP,
        output_dir=str(tmp_path / "out"),
    )
    with _patch_torch_load():
        config = t._build_config()
    cmd = t._build_cmd("input.json")

    assert "--finetune" in cmd
    # pretrained must immediately follow --finetune
    assert cmd[cmd.index("--finetune") + 1] == dummy_ckpt
    # Paper alignment: single-task fine-tune passes NO --model-branch.
    assert "--model-branch" not in cmd
    assert "--skip-neighbor-stat" in cmd

    assert config["model"]["descriptor"]["trainable"] is True


# ---------------------------------------------------------------------------
# 4. LP config
# ---------------------------------------------------------------------------


def test_config_lp(systems, dummy_ckpt, tmp_path):
    train_glob, valid_glob = systems
    t = DPATrainer(
        pretrained=dummy_ckpt,
        freeze_backbone=True,
        train_systems=train_glob,
        valid_systems=valid_glob,
        type_map=DUMMY_TYPE_MAP,
        output_dir=str(tmp_path / "out"),
    )
    with _patch_torch_load():
        config = t._build_config()
    cmd = t._build_cmd("input.json")

    assert "--finetune" in cmd
    assert cmd[cmd.index("--finetune") + 1] == dummy_ckpt
    # Paper alignment: single-task fine-tune passes NO --model-branch.
    assert "--model-branch" not in cmd
    assert "--skip-neighbor-stat" in cmd
    assert config["model"]["descriptor"]["trainable"] is False


# ---------------------------------------------------------------------------
# 5. Glob expansion
# ---------------------------------------------------------------------------


def test_glob_expansion(tmp_path):
    train_glob = _make_systems(tmp_path, "train", 70)
    valid_glob = _make_systems(tmp_path, "valid", 70)

    t = DPATrainer(
        train_systems=train_glob,
        valid_systems=valid_glob,
        type_map=DUMMY_TYPE_MAP,
        output_dir=str(tmp_path / "out"),
    )
    config = t._build_config()
    assert len(config["training"]["training_data"]["systems"]) == 70
    assert len(config["training"]["validation_data"]["systems"]) == 70

    # Empty glob raises
    empty_glob = str(tmp_path / "nope" / "*")
    t_empty = DPATrainer(
        train_systems=empty_glob,
        valid_systems=valid_glob,
        type_map=DUMMY_TYPE_MAP,
        output_dir=str(tmp_path / "out2"),
    )
    with pytest.raises(ValueError, match="resolved to 0 systems"):
        t_empty._build_config()


# ---------------------------------------------------------------------------
# 6. evaluate() output parsing
# ---------------------------------------------------------------------------


def test_evaluate_parse(systems, tmp_path):
    train_glob, valid_glob = systems
    t = DPATrainer(
        train_systems=train_glob,
        valid_systems=valid_glob,
        type_map=DUMMY_TYPE_MAP,
        output_dir=str(tmp_path / "out"),
    )

    # Place a fake checkpoint so _final_ckpt_path() finds it.
    os.makedirs(t.output_dir, exist_ok=True)
    fake_ckpt = os.path.join(t.output_dir, "model.ckpt-100.pt")
    open(fake_ckpt, "w").close()

    # Need an existing system path for the test glob to resolve.
    test_glob = _make_systems(tmp_path, "test", 5)

    canned_stdout = (
        "DEEPMD INFO    # number of test data : 42\n"
        "DEEPMD INFO    PROPERTY MAE            : 0.006789 units\n"
        "DEEPMD INFO    PROPERTY RMSE           : 0.012345 units\n"
    )

    class _Result:
        stdout = canned_stdout
        stderr = ""
        returncode = 0

    with patch("subprocess.run", return_value=_Result()):
        out = t.evaluate(test_glob)

    assert out["rmse"] == pytest.approx(0.012345)
    assert out["mae"] == pytest.approx(0.006789)
    assert out["n_frames"] == 42
    # evaluate() concatenates stdout + "\n" + stderr; canned_stdout must be in it.
    assert canned_stdout in out["_raw_stdout"]
    assert (
        "rmse" in out["_parser_pattern_used"].lower()
        or "mae" in out["_parser_pattern_used"].lower()
    )


# ---------------------------------------------------------------------------
# 7. Parser: property-explicit pattern
# ---------------------------------------------------------------------------


def test_evaluate_parse_property_explicit():
    stdout = (
        "DEEPMD INFO    PROPERTY RMSE           : 0.0123 units\n"
        "DEEPMD INFO    PROPERTY MAE            : 0.0080 units\n"
    )
    out = DPATrainer._parse_test_output(stdout)
    assert out["rmse"] == pytest.approx(0.0123)
    assert out["mae"] == pytest.approx(0.0080)
    assert "PROPERTY" in out["_parser_pattern_used"]
    assert out["_raw_stdout"] == stdout


# ---------------------------------------------------------------------------
# 8. Parser: property format (no generic fallback — removed during refactor)
# ---------------------------------------------------------------------------


def test_evaluate_parse_property_format_explicit():
    r"""Parser auto-detects PROPERTY output and matches the well-anchored regex.
    Generic \brmse\b / \bmae\b fallback patterns were removed.
    """
    stdout = (
        "DEEPMD INFO    PROPERTY MAE            : 0.0234 units\n"
        "DEEPMD INFO    PROPERTY RMSE           : 0.0150 units\n"
    )
    out = DPATrainer._parse_test_output(stdout)
    assert out["mae"] == pytest.approx(0.0234)
    assert out["rmse"] == pytest.approx(0.0150)
    assert "PROPERTY" in out["_parser_pattern_used"]


# ---------------------------------------------------------------------------
# 9. Parser: unparseable input raises RuntimeError
# ---------------------------------------------------------------------------


def test_evaluate_parse_unparseable():
    stdout = "no numbers here"
    with pytest.raises(RuntimeError) as exc_info:
        DPATrainer._parse_test_output(stdout)
    assert "no numbers here" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 10. Idempotency: skip when a longer checkpoint exists
# ---------------------------------------------------------------------------


def test_idempotency_skip_when_longer_ckpt_exists(systems, tmp_path):
    train_glob, valid_glob = systems
    out_dir = tmp_path / "out_skip"
    out_dir.mkdir()
    # Place a model.ckpt-100.pt; ask for max_steps=50 → should skip.
    longer_ckpt = out_dir / "model.ckpt-100.pt"
    longer_ckpt.write_bytes(b"")

    t = DPATrainer(
        train_systems=train_glob,
        valid_systems=valid_glob,
        type_map=DUMMY_TYPE_MAP,
        max_steps=50,
        output_dir=str(out_dir),
    )
    with patch("subprocess.run") as run_mock:
        result = t.fit()
        run_mock.assert_not_called()
    assert result == str(longer_ckpt)


# ---------------------------------------------------------------------------
# 11. Idempotency: retrain when only a shorter checkpoint exists
# ---------------------------------------------------------------------------


def test_idempotency_retrain_when_shorter_ckpt_exists(systems, tmp_path):
    train_glob, valid_glob = systems
    out_dir = tmp_path / "out_retrain"
    out_dir.mkdir()
    # Place a model.ckpt-50.pt; ask for max_steps=100 → should retrain.
    (out_dir / "model.ckpt-50.pt").write_bytes(b"")

    t = DPATrainer(
        train_systems=train_glob,
        valid_systems=valid_glob,
        type_map=DUMMY_TYPE_MAP,
        max_steps=100,
        output_dir=str(out_dir),
    )

    # Mock subprocess.run so we never call real `dp`. After "training",
    # create the model.ckpt-100.pt the production code will look for.
    final_ckpt = out_dir / "model.ckpt-100.pt"

    def _fake_run(cmd, *args, **kwargs):
        final_ckpt.write_bytes(b"")

        class R:
            returncode = 0

        return R()

    with patch("subprocess.run", side_effect=_fake_run) as run_mock:
        result = t.fit()
        run_mock.assert_called_once()
    assert result == str(final_ckpt)


# ---------------------------------------------------------------------------
# 12. Seed propagation
# ---------------------------------------------------------------------------


def test_seed_propagation(systems, tmp_path):
    train_glob, valid_glob = systems
    t = DPATrainer(
        pretrained=None,
        train_systems=train_glob,
        valid_systems=valid_glob,
        type_map=DUMMY_TYPE_MAP,
        seed=12345,
        output_dir=str(tmp_path / "out_seed"),
    )
    cfg = t._build_config()
    assert cfg["model"]["descriptor"]["seed"] == 12345
    assert cfg["model"]["fitting_net"]["seed"] == 12345
    assert cfg["training"]["seed"] == 12345
    # Top-level "seed" was removed: deepmd 3.1.3 dargs is strict-mode and
    # rejects unknown root keys. Seeds live on descriptor, fitting_net, and
    # training instead.
    assert "seed" not in cfg


# ---------------------------------------------------------------------------
# 13. Parser: takes weighted-average (last) match
# ---------------------------------------------------------------------------


def test_evaluate_parse_takes_weighted_average():
    """When dp prints per-system + weighted-average blocks, return the
    weighted average (last match).
    """
    stdout = (
        "PROPERTY MAE  : 0.10 units\n"
        "PROPERTY RMSE : 0.20 units\n"
        "# ----------weighted average of errors-----------\n"
        "PROPERTY MAE  : 0.05 units\n"
        "PROPERTY RMSE : 0.08 units\n"
    )
    out = DPATrainer._parse_test_output(stdout)
    # Must be the weighted-average (second/last) values, not the per-system
    # (first) values.
    assert out["mae"] == pytest.approx(0.05)
    assert out["rmse"] == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# 14. evaluate() combines stdout + stderr
# ---------------------------------------------------------------------------


def test_evaluate_combines_stderr(systems, tmp_path):
    train_glob, valid_glob = systems
    t = DPATrainer(
        train_systems=train_glob,
        valid_systems=valid_glob,
        type_map=DUMMY_TYPE_MAP,
        output_dir=str(tmp_path / "out_stderr"),
    )
    os.makedirs(t.output_dir, exist_ok=True)
    (Path(t.output_dir) / "model.ckpt-100.pt").write_bytes(b"")
    test_glob = _make_systems(tmp_path, "test_stderr", 5)

    canned_stderr = (
        "DEEPMD INFO    # number of test data : 100\n"
        "DEEPMD INFO    PROPERTY MAE  : 0.0123 units\n"
        "DEEPMD INFO    PROPERTY RMSE : 0.0456 units\n"
    )

    class _Result:
        stdout = ""
        stderr = canned_stderr
        returncode = 0

    with patch("subprocess.run", return_value=_Result()):
        out = t.evaluate(test_glob)
    assert out["mae"] == pytest.approx(0.0123)
    assert out["rmse"] == pytest.approx(0.0456)


# ---------------------------------------------------------------------------
# 15. evaluate() writes datafile and passes -f, not -s
# ---------------------------------------------------------------------------


def test_evaluate_writes_datafile_and_uses_f_flag(systems, tmp_path):
    """evaluate() must write a datafile with one system per line and
    pass it to dp test via -f (single value), not multiplex -s flags.
    """
    train_glob, valid_glob = systems
    out_dir = tmp_path / "out_datafile"
    t = DPATrainer(
        train_systems=train_glob,
        valid_systems=valid_glob,
        type_map=DUMMY_TYPE_MAP,
        output_dir=str(out_dir),
    )
    os.makedirs(t.output_dir, exist_ok=True)
    (Path(t.output_dir) / "model.ckpt-100.pt").write_bytes(b"")
    test_glob = _make_systems(tmp_path, "test_df", 5)

    captured_cmd = []
    canned_stderr = (
        "DEEPMD INFO    # number of test data : 50\n"
        "DEEPMD INFO    # number of systems : 5\n"
        "DEEPMD INFO    PROPERTY MAE : 0.01 units\n"
        "DEEPMD INFO    PROPERTY RMSE : 0.02 units\n"
    )

    class _Result:
        stdout = ""
        stderr = canned_stderr
        returncode = 0

    def _capture(cmd, *args, **kwargs):
        captured_cmd.extend(cmd)
        return _Result()

    with patch("subprocess.run", side_effect=_capture):
        out = t.evaluate(test_glob)

    # No -s anywhere; exactly one -f flag.
    assert "-s" not in captured_cmd, f"-s should not appear: {captured_cmd}"
    assert captured_cmd.count("-f") == 1

    # -f points to a real datafile with 5 lines.
    f_idx = captured_cmd.index("-f")
    datafile = captured_cmd[f_idx + 1]
    assert os.path.isfile(datafile), f"datafile not written: {datafile}"
    lines = [l for l in open(datafile).read().split("\n") if l.strip()]
    assert len(lines) == 5, f"Expected 5 systems in datafile, got {len(lines)}"

    assert out["mae"] == pytest.approx(0.01)
    assert out["rmse"] == pytest.approx(0.02)
    assert out["n_systems"] == 5
