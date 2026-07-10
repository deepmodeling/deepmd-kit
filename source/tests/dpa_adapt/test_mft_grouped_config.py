# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MFT downstream_task_type='group_property'.

Mirrors test_mft_property_task.py's coverage of the paper-faithful
downstream_task_type='property' branch, but for the grouped/assembly
variant: a fresh group_property fitting_net + group_property loss for the
DOWNSTREAM head (sharing a descriptor with the aux ener branch, exactly like
property mode), while the aux branch keeps its ener fitting_net pulled from
the ckpt.

GroupPropertyFittingNet is a small standalone MLP, not built on
GeneralFitting like the property head is, so several property-schema
fields (resnet_dt, intensive, distinguish_types, numb_aparam) don't exist
on it -- dargs strict-mode rejects them outright, so the config builder
must not carry them over from the property path (see
_build_group_property_fitting_net, independent of
_build_property_fitting_net).
"""

from __future__ import (
    annotations,
)

import os
from pathlib import (
    Path,
)
from typing import (
    ClassVar,
)

import numpy as np
import pytest

from dpa_adapt.config.manager import (
    MFTConfigManager,
)
from dpa_adapt.mft import (
    MFTFineTuner,
)

DUMMY_TYPE_MAP = ["H", "C", "N", "O"]


class _FakeGroupPropertyTuner:
    """Tuner-shaped object configured for downstream_task_type='group_property'.

    Bypasses MFTFineTuner.__init__ so tests don't need a real ckpt.
    """

    pretrained = "/share/DPA-3.1-3M.pt"
    aux_branch = "SPICE2"
    aux_prob = 0.5
    type_map: ClassVar[list[str]] = DUMMY_TYPE_MAP
    # aux fitting_net pulled from ckpt — an ener config (the actual SPICE2
    # head). dim_case_embd=31 mirrors what a real DPA-3.1-3M checkpoint's aux
    # branch fitting_net carries (it was itself trained as one of 31 branches).
    fitting_net_params: ClassVar[dict[str, object]] = {
        "type": "ener",
        "neuron": [240, 240, 240],
        "dim_case_embd": 31,
    }
    downstream_task_type = "group_property"
    property_name = "overpotential"
    task_dim = 1
    group_reduce = "sum"
    learning_rate = 1e-3
    stop_lr = 1e-5
    max_steps = 1000
    batch_size = "auto:32"
    seed = 42
    output_dir = "/tmp/mft_group_property_test"
    save_freq = 500
    disp_freq = 100
    train_data = "/data/oer_downstream"
    aux_data = "/data/spice2"
    valid_data = None


def _build() -> dict:
    return MFTConfigManager(_FakeGroupPropertyTuner()).build()


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------


def test_group_property_task_uses_task_type_as_branch_key():
    config = _build()
    assert "group_property" in config["model"]["model_dict"]
    assert "SPICE2" in config["model"]["model_dict"]


def test_group_property_task_config_has_group_property_fitting_net():
    config = _build()
    fn = config["model"]["model_dict"]["group_property"]["fitting_net"]
    assert fn["type"] == "group_property"
    assert fn["property_name"] == "overpotential"
    assert fn["task_dim"] == 1


def test_group_property_task_group_reduce_propagated():
    config = _build()
    fn = config["model"]["model_dict"]["group_property"]["fitting_net"]
    assert fn["group_reduce"] == "sum"


def test_group_property_task_group_reduce_defaults_to_mean():
    # A tuner-shaped object that genuinely has no group_reduce attribute
    # (rather than mutating the shared _FakeGroupPropertyTuner class), so
    # _build_group_property_fitting_net's getattr(t, "group_reduce", "mean")
    # exercises its real default.
    attrs = {
        k: v
        for k, v in vars(_FakeGroupPropertyTuner).items()
        if not k.startswith("__") and k != "group_reduce"
    }
    tuner = type("_TunerWithoutGroupReduce", (), attrs)()
    config = MFTConfigManager(tuner).build()
    fn = config["model"]["model_dict"]["group_property"]["fitting_net"]
    assert fn["group_reduce"] == "mean"


def test_group_property_task_config_has_group_property_loss():
    config = _build()
    loss = config["loss_dict"]["group_property"]
    assert loss["type"] == "group_property"
    assert loss["loss_func"] == "mse"


def test_group_property_task_dim_case_embd_matches_aux_branch_requirement():
    """The descriptor is shared with the aux branch; deepmd-kit's multi-task
    trainer requires every model_dict branch to declare the same
    dim_case_embd (deepmd.pt.train.training.get_case_embd_config), so the
    group_property head must declare it too, exactly like the property head
    does -- read from the aux branch's own fitting_net_params, not hardcoded
    (a checkpoint other than DPA-3.1-3M has a different branch count; see
    test_dim_case_embd_is_read_from_aux_fitting_net_not_hardcoded below).
    """
    config = _build()
    fn = config["model"]["model_dict"]["group_property"]["fitting_net"]
    assert fn["dim_case_embd"] == 31


def test_dim_case_embd_is_read_from_aux_fitting_net_not_hardcoded():
    """Regression pin: a checkpoint with a different branch count (e.g. the
    23-branch OMol25/Organic_Reactions/ODAC23 checkpoint used for the
    cloud-point OER grouped run) must produce a matching, not hardcoded-31,
    dim_case_embd on the group_property downstream head. Hardcoding 31 here
    silently mismatches every checkpoint that isn't DPA-3.1-3M.
    """
    t = _FakeGroupPropertyTuner()
    t.fitting_net_params = {
        "type": "ener",
        "neuron": [240, 240, 240],
        "dim_case_embd": 23,
    }
    config = MFTConfigManager(t).build()
    fn = config["model"]["model_dict"]["group_property"]["fitting_net"]
    assert fn["dim_case_embd"] == 23


def test_dim_case_embd_omitted_when_aux_fitting_net_has_none():
    """A single-task-pretrained (non-multitask) checkpoint's aux fitting_net
    has no dim_case_embd at all; the downstream head must not invent one.
    """
    t = _FakeGroupPropertyTuner()
    t.fitting_net_params = {"type": "ener", "neuron": [240, 240, 240]}
    config = MFTConfigManager(t).build()
    fn = config["model"]["model_dict"]["group_property"]["fitting_net"]
    assert "dim_case_embd" not in fn


def test_group_property_task_no_unsupported_fitting_fields():
    """GroupPropertyFittingNet has no wiring for these property-schema
    fields; dargs strict mode rejects them outright rather than ignoring
    them (deepmd.utils.argcheck.fitting_group_property), so the config
    builder must never emit them.
    """
    config = _build()
    fn = config["model"]["model_dict"]["group_property"]["fitting_net"]
    unsupported = {"resnet_dt", "intensive", "distinguish_types", "numb_aparam"}
    assert not (unsupported & fn.keys())


def test_group_property_task_finetune_head_random_for_downstream():
    config = _build()
    downstream = config["model"]["model_dict"]["group_property"]
    assert downstream["finetune_head"] == "RANDOM"


def test_group_property_task_aux_branch_keeps_ener_fitting_net():
    config = _build()
    aux = config["model"]["model_dict"]["SPICE2"]
    assert aux["fitting_net"]["type"] == "ener"
    assert aux["finetune_head"] == "SPICE2"


def test_group_property_task_aux_branch_keeps_ener_loss():
    config = _build()
    assert config["loss_dict"]["SPICE2"]["type"] == "ener"


def test_group_property_task_model_prob_splits_by_aux_prob():
    config = _build()
    prob = config["training"]["model_prob"]
    assert prob["SPICE2"] == pytest.approx(0.5)
    assert prob["group_property"] == pytest.approx(0.5)


def test_group_property_task_gradient_clipping_matches_paper_alignment():
    config = _build()
    assert config["training"]["gradient_max_norm"] == 5.0


def test_group_property_task_descriptor_matches_property_alignment():
    """group_property is a "random downstream head" mode just like property,
    so it must follow the same paper-alignment descriptor tweaks (silut
    activation, fix_stat_std) rather than the legacy ener-mode descriptor.
    """
    config = _build()
    desc = config["model"]["shared_dict"]["dpa3_descriptor"]
    assert desc["activation_function"] == "silut:3.0"
    assert desc["repflow"]["fix_stat_std"] == 0.3


def test_group_property_task_multidim_task_dim():
    t = _FakeGroupPropertyTuner()
    t.task_dim = 3
    config = MFTConfigManager(t).build()
    fn = config["model"]["model_dict"]["group_property"]["fitting_net"]
    assert fn["task_dim"] == 3


def test_group_property_task_fparam_dim_propagated():
    t = _FakeGroupPropertyTuner()
    t.fparam_dim = 2
    config = MFTConfigManager(t).build()
    fn = config["model"]["model_dict"]["group_property"]["fitting_net"]
    assert fn["numb_fparam"] == 2


# ---------------------------------------------------------------------------
# MFTFineTuner.__init__ validation
# ---------------------------------------------------------------------------


def test_group_property_task_requires_property_name(monkeypatch):
    import torch

    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **kw: {
            "model": {
                "_extra_state": {
                    "model_params": {
                        "model_dict": {"SPICE2": {"fitting_net": {"type": "ener"}}}
                    }
                }
            }
        },
    )
    with pytest.raises(ValueError, match="property_name"):
        MFTFineTuner(
            pretrained="/does/not/exist.pt",
            aux_branch="SPICE2",
            downstream_task_type="group_property",
        )


def test_group_property_task_invalid_group_reduce_raises():
    with pytest.raises(ValueError, match="group_reduce"):
        MFTFineTuner(
            pretrained="/does/not/exist.pt",
            aux_branch="SPICE2",
            downstream_task_type="group_property",
            property_name="overpotential",
            group_reduce="max",
        )


def test_group_property_task_group_reduce_stored(monkeypatch):
    import torch

    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **kw: {
            "model": {
                "_extra_state": {
                    "model_params": {
                        "model_dict": {"SPICE2": {"fitting_net": {"type": "ener"}}}
                    }
                }
            }
        },
    )
    t = MFTFineTuner(
        pretrained="/does/not/exist.pt",
        aux_branch="SPICE2",
        downstream_task_type="group_property",
        property_name="overpotential",
        group_reduce="sum",
    )
    assert t.group_reduce == "sum"


def test_downstream_head_is_group_property():
    t = MFTFineTuner.__new__(MFTFineTuner)
    t.downstream_task_type = "group_property"
    assert t._downstream_head == "group_property"


# ---------------------------------------------------------------------------
# evaluate()/predict(): multi-frame-group safety guard
#
# `dp --pt test` never threads group_id/weight/pool_mask through, so a frozen
# group_property head silently scores every test frame as its own one-frame
# group. Harmless when every real group is one frame; silently wrong for
# genuine multi-frame assemblies. evaluate()/predict() must refuse the
# latter rather than return numbers that look plausible but aren't.
# ---------------------------------------------------------------------------


def _make_group_property_finetuner(tmp_path):
    ft = MFTFineTuner.__new__(MFTFineTuner)
    ft.pretrained = str(tmp_path / "dummy.pt")
    ft.aux_branch = "SPICE2"
    ft.aux_prob = 0.5
    ft.type_map = DUMMY_TYPE_MAP
    ft.fitting_net_params = {}
    ft.downstream_task_type = "group_property"
    ft.property_name = "overpotential"
    ft.task_dim = 1
    ft.group_reduce = "mean"
    ft.learning_rate = 1e-3
    ft.stop_lr = 1e-5
    ft.max_steps = 100
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


def _write_group_id(set_dir: Path, group_ids: list[int]) -> None:
    set_dir.mkdir(parents=True, exist_ok=True)
    np.save(set_dir / "group_id.npy", np.asarray(group_ids, dtype=np.int64))


def test_check_no_multi_frame_groups_passes_when_every_group_is_one_frame(tmp_path):
    sys_dir = tmp_path / "sys_ok"
    _write_group_id(sys_dir / "set.000", [0, 1, 2])
    ft = _make_group_property_finetuner(tmp_path)
    ft._check_no_multi_frame_groups([str(sys_dir)])  # must not raise


def test_check_no_multi_frame_groups_passes_when_group_id_absent(tmp_path):
    sys_dir = tmp_path / "sys_no_marker"
    (sys_dir / "set.000").mkdir(parents=True)
    ft = _make_group_property_finetuner(tmp_path)
    ft._check_no_multi_frame_groups([str(sys_dir)])  # must not raise


def test_check_no_multi_frame_groups_raises_for_genuine_multi_frame_group(tmp_path):
    sys_dir = tmp_path / "sys_oer"
    # group 0 spans two frames (e.g. O* + OH* of one OER assembly)
    _write_group_id(sys_dir / "set.000", [0, 0, 1])
    ft = _make_group_property_finetuner(tmp_path)
    with pytest.raises(RuntimeError, match="spanning more than one frame"):
        ft._check_no_multi_frame_groups([str(sys_dir)])


def test_evaluate_rejects_multi_frame_groups_before_touching_subprocess(tmp_path):
    """The guard must fire before any dp --pt freeze/test subprocess call,
    so evaluate() fails fast without a spurious freeze/test invocation.
    """
    from unittest.mock import patch

    ft = _make_group_property_finetuner(tmp_path)
    (Path(ft.output_dir) / "model.ckpt-100.pt").write_bytes(b"")
    sys_dir = tmp_path / "sys_oer"
    _write_group_id(sys_dir / "set.000", [0, 0, 1])

    with (
        patch("subprocess.run", side_effect=AssertionError("must not be called")),
        pytest.raises(RuntimeError, match="spanning more than one frame"),
    ):
        ft.evaluate(str(sys_dir))


def test_predict_allows_group_property_head():
    t = MFTFineTuner.__new__(MFTFineTuner)
    t.downstream_task_type = "ener"
    with pytest.raises(RuntimeError, match="'property' or 'group_property'"):
        t.predict("some/data")
    t.downstream_task_type = "group_property"
    # Advances past the head-type guard; next it calls _freeze_ckpt(), which
    # fails for an unrelated reason (no output_dir/checkpoint set up here) --
    # proving the group_property RuntimeError above is gone.
    with pytest.raises(Exception) as exc_info:
        t.predict("some/data")
    assert "only supported for downstream_task_type" not in str(exc_info.value)
