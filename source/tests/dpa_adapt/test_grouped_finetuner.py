# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import (
    annotations,
)

import json
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
)

import numpy as np

from dpa_adapt.finetuner import (
    DPAFineTuner,
)
from source.tests.dpa_adapt.test_finetuner_strategies import (
    _FULL_TYPE_MAP,
    _fake_ckpt_sd,
    _make_system_dirs,
    _mock_dp_train,
)


def test_grouped_training_strategy_uses_group_property_config(monkeypatch, tmp_path):
    import torch

    monkeypatch.setattr(torch, "load", lambda *a, **kw: _fake_ckpt_sd())
    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"")
    out_dir = tmp_path / "out"
    systems = _make_system_dirs(tmp_path, formulas=("GroupedTrain",), n=2)
    valid_systems = _make_system_dirs(tmp_path, formulas=("GroupedValid",), n=1)
    for sid, sysdir in enumerate(systems + valid_systems):
        set_dir = Path(sysdir) / "set.000"
        np.save(set_dir / "group_id.npy", np.array([sid, sid], dtype=np.int64))
        np.save(set_dir / "weight.npy", np.array([0.5, 0.5], dtype=float))
        np.save(set_dir / "pool_mask.npy", np.ones((2, 2), dtype=float))

    model = DPAFineTuner(
        pretrained=str(ckpt),
        strategy="finetune",
        property_name="overpotential",
        max_steps=20,
        output_dir=str(out_dir),
    )
    with patch("subprocess.run", side_effect=_mock_dp_train(str(out_dir))):
        result = model.fit(train_data=systems, valid_data=valid_systems)

    assert result is not None
    cfg = json.loads((out_dir / "input.json").read_text())
    assert cfg["model"]["type_map"] == _FULL_TYPE_MAP
    assert cfg["model"]["fitting_net"]["type"] == "group_property"
    assert cfg["model"]["fitting_net"]["property_name"] == "overpotential"
    # Grouped head defaults to GELU (tanh saturates on the un-normalized
    # embedding+fparam input and collapses predictions to a constant).
    assert cfg["model"]["fitting_net"]["activation_function"] == "gelu"
    assert cfg["loss"]["type"] == "group_property"


def test_grouped_target_alias_and_auto_fparam_dim(monkeypatch, tmp_path):
    """target= aliases property_name; fit(train=/valid=) work; fparam_dim auto."""
    import torch

    monkeypatch.setattr(torch, "load", lambda *a, **kw: _fake_ckpt_sd())
    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"")
    out_dir = tmp_path / "out"
    systems = _make_system_dirs(tmp_path, formulas=("GroupedTrain",), n=2)
    valid_systems = _make_system_dirs(tmp_path, formulas=("GroupedValid",), n=1)
    for sid, sysdir in enumerate(systems + valid_systems):
        set_dir = Path(sysdir) / "set.000"
        np.save(set_dir / "group_id.npy", np.array([sid, sid], dtype=np.int64))
        np.save(set_dir / "weight.npy", np.array([0.5, 0.5], dtype=float))
        np.save(set_dir / "pool_mask.npy", np.ones((2, 2), dtype=float))
        # per-group side features -> fparam_dim should be auto-detected as 3
        np.save(set_dir / "fparam.npy", np.ones((2, 3), dtype=float))

    model = DPAFineTuner(
        pretrained=str(ckpt),
        strategy="finetune",
        target="overpotential",  # alias for property_name
        max_steps=20,
        output_dir=str(out_dir),
    )
    assert model.property_name == "overpotential"

    with patch("subprocess.run", side_effect=_mock_dp_train(str(out_dir))):
        model.fit(train=systems, valid=valid_systems)  # train=/valid= aliases

    cfg = json.loads((out_dir / "input.json").read_text())
    assert cfg["model"]["fitting_net"]["type"] == "group_property"
    assert cfg["model"]["fitting_net"]["property_name"] == "overpotential"
    # fparam_dim auto-inferred from set.*/fparam.npy -> numb_fparam wired into head
    assert cfg["model"]["fitting_net"]["numb_fparam"] == 3
