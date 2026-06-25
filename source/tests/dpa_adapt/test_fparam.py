# SPDX-License-Identifier: LGPL-3.0-or-later
# Tests for fparam (frame-level condition input) support.
# Heavy deps (torch, dpdata, dp subprocess) are mocked throughout.

from __future__ import (
    annotations,
)

from unittest.mock import (
    patch,
)

import numpy as np
import pytest

from dpa_adapt.data.errors import (
    DPADataError,
)
from dpa_adapt.trainer import (
    DPATrainer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DUMMY_TYPE_MAP = ["H", "C", "N", "O"]


def _make_systems(tmp_path, prefix: str, n: int) -> str:
    """Create n empty system dirs and return a glob pattern matching them."""
    root = tmp_path / prefix
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / f"sys_{i:03d}").mkdir()
    return str(root / "sys_*")


def _make_dummy_trainer(fparam_dim=0, **kwargs):
    """Construct a DPATrainer with minimal valid args."""
    defaults = {
        "pretrained": None,
        "train_systems": "dummy_train",
        "valid_systems": "dummy_valid",
        "type_map": DUMMY_TYPE_MAP,
        "fparam_dim": fparam_dim,
    }
    defaults.update(kwargs)
    return DPATrainer(**defaults)


# ---------------------------------------------------------------------------
# Tests: trainer fparam_dim validation in __init__
# ---------------------------------------------------------------------------


def test_trainer_fparam_dim_negative_raises():
    """DPATrainer(fparam_dim=-1) raises ValueError."""
    with pytest.raises(ValueError, match="fparam_dim must be a non-negative"):
        _make_dummy_trainer(fparam_dim=-1)


def test_trainer_fparam_dim_non_int_raises():
    """DPATrainer(fparam_dim='3') raises ValueError."""
    with pytest.raises(ValueError, match="fparam_dim must be a non-negative"):
        _make_dummy_trainer(fparam_dim="3")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: trainer._build_fitting_net fparam injection
# ---------------------------------------------------------------------------


def test_trainer_fparam_dim_injected_in_fitting_net():
    """DPATrainer(fparam_dim=3)._build_fitting_net() includes numb_fparam=3."""
    t = _make_dummy_trainer(fparam_dim=3)
    fn = t._build_fitting_net()
    assert fn["numb_fparam"] == 3


def test_trainer_fparam_dim_zero_not_injected():
    """DPATrainer(fparam_dim=0)._build_fitting_net() does NOT contain 'fparam_dim'."""
    t = _make_dummy_trainer(fparam_dim=0)
    fn = t._build_fitting_net()
    assert "fparam_dim" not in fn


# ---------------------------------------------------------------------------
# Tests: trainer._validate_fparam
# ---------------------------------------------------------------------------


def test_validate_fparam_missing_file_raises(tmp_path):
    """_validate_fparam raises DPADataError when fparam.npy is missing."""
    sys_dir = tmp_path / "system"
    set_dir = sys_dir / "set.000"
    set_dir.mkdir(parents=True)

    with pytest.raises(DPADataError, match="is missing"):
        DPATrainer._validate_fparam([str(sys_dir)], fparam_dim=2)


def test_validate_fparam_wrong_shape_raises(tmp_path):
    """_validate_fparam raises DPADataError when shape[1] != fparam_dim."""
    sys_dir = tmp_path / "system"
    set_dir = sys_dir / "set.000"
    set_dir.mkdir(parents=True)
    # shape (5, 3), expected dim 2
    np.save(str(set_dir / "fparam.npy"), np.zeros((5, 3)))

    with pytest.raises(DPADataError, match="has shape"):
        DPATrainer._validate_fparam([str(sys_dir)], fparam_dim=2)


def test_validate_fparam_correct_passes(tmp_path):
    """_validate_fparam does NOT raise when shape matches."""
    sys_dir = tmp_path / "system"
    set_dir = sys_dir / "set.000"
    set_dir.mkdir(parents=True)
    np.save(str(set_dir / "fparam.npy"), np.zeros((5, 2)))

    # Should not raise
    DPATrainer._validate_fparam([str(sys_dir)], fparam_dim=2)


def test_validate_fparam_multiple_systems(tmp_path):
    """_validate_fparam checks all set.* dirs across multiple systems."""
    for i in range(2):
        sys_dir = tmp_path / f"sys_{i}"
        for s in ("set.000", "set.001"):
            (sys_dir / s).mkdir(parents=True)
            np.save(str(sys_dir / s / "fparam.npy"), np.zeros((10, 3)))

    DPATrainer._validate_fparam(
        [str(tmp_path / "sys_0"), str(tmp_path / "sys_1")],
        fparam_dim=3,
    )


# ---------------------------------------------------------------------------
# Tests: DPAFineTuner forwards fparam_dim to DPATrainer
# ---------------------------------------------------------------------------


def test_finetuner_fparam_forwarded_to_trainer():
    """DPAFineTuner(fparam_dim=4, strategy='finetune') passes fparam_dim=4 to DPATrainer."""
    with patch("dpa_adapt.trainer.DPATrainer") as mock_trainer_cls:
        from dpa_adapt.finetuner import (
            DPAFineTuner,
        )

        ft = DPAFineTuner(
            pretrained="dummy.pt",
            strategy="finetune",
            fparam_dim=4,
        )

        # Call _fit_training directly (skip type_map resolution, skip actual fit)
        ft._fit_training("dummy_train", "dummy_valid", DUMMY_TYPE_MAP)

        mock_trainer_cls.assert_called_once()
        _, kwargs = mock_trainer_cls.call_args
        assert kwargs["fparam_dim"] == 4


def test_finetuner_fparam_zero_not_forwarded():
    """DPAFineTuner(fparam_dim=0) passes fparam_dim=0 (default, disabled)."""
    with patch("dpa_adapt.trainer.DPATrainer") as mock_trainer_cls:
        from dpa_adapt.finetuner import (
            DPAFineTuner,
        )

        ft = DPAFineTuner(
            pretrained="dummy.pt",
            strategy="finetune",
        )

        ft._fit_training("dummy_train", "dummy_valid", DUMMY_TYPE_MAP)

        mock_trainer_cls.assert_called_once()
        _, kwargs = mock_trainer_cls.call_args
        assert kwargs["fparam_dim"] == 0


# ---------------------------------------------------------------------------
# Tests: CLI --fparam-dim parsing
# ---------------------------------------------------------------------------


def test_cli_fparam_dim_parsed():
    """--fparam-dim 3 is parsed to args.fparam_dim == 3."""
    from dpa_adapt.cli import (
        get_parser,
    )

    parser = get_parser()
    args = parser.parse_args(
        [
            "fit",
            "--train-data",
            "x",
            "--fparam-dim",
            "3",
        ]
    )
    assert args.fparam_dim == 3


def test_cli_fparam_dim_default_zero():
    """Without --fparam-dim, args.fparam_dim defaults to 0."""
    from dpa_adapt.cli import (
        get_parser,
    )

    parser = get_parser()
    args = parser.parse_args(
        [
            "fit",
            "--train-data",
            "x",
        ]
    )
    assert args.fparam_dim == 0


# ---------------------------------------------------------------------------
# Tests: MFTFineTuner.fit() calls _validate_fparam
# ---------------------------------------------------------------------------


def test_mft_fparam_validate_called_on_fit():
    """MFTFineTuner.fit() calls _validate_fparam when fparam_dim > 0."""
    with (
        patch("dpa_adapt.trainer.DPATrainer._validate_fparam") as mock_validate,
        patch("dpa_adapt.config.manager.MFTConfigManager") as mock_cm_class,
        patch("dpa_adapt.mft.subprocess.Popen") as mock_popen,
    ):
        from dpa_adapt.mft import (
            MFTFineTuner,
        )

        mock_process = mock_popen.return_value
        mock_process.stdout = []
        mock_process.returncode = 0

        mft = MFTFineTuner(
            pretrained="dummy.pt",
            property_name="homo",
            fparam_dim=3,
            type_map=["H"],
        )
        mft.fit(train_data="dummy_train", aux_data="dummy_aux")

        mock_validate.assert_called_once()
        args, _kwargs = mock_validate.call_args
        assert args[0] == "dummy_train"
        assert args[1] == 3


def test_mft_fparam_validate_skipped_when_zero():
    """MFTFineTuner.fit() does NOT call _validate_fparam when fparam_dim=0."""
    with (
        patch("dpa_adapt.trainer.DPATrainer._validate_fparam") as mock_validate,
        patch("dpa_adapt.config.manager.MFTConfigManager") as mock_cm_class,
        patch("dpa_adapt.mft.subprocess.Popen") as mock_popen,
    ):
        from dpa_adapt.mft import (
            MFTFineTuner,
        )

        mock_process = mock_popen.return_value
        mock_process.stdout = []
        mock_process.returncode = 0

        mft = MFTFineTuner(
            pretrained="dummy.pt",
            property_name="homo",
            fparam_dim=0,
            type_map=["H"],
        )
        mft.fit(train_data="dummy_train", aux_data="dummy_aux")

        mock_validate.assert_not_called()
