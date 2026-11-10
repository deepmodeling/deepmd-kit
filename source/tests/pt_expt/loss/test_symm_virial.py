# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gradients and validation use the same symmetric virial convention."""

from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.loss.ener import (
    EnergyLoss,
)
from deepmd.dpmodel.loss.ener_spin import (
    EnergySpinLoss,
)
from deepmd.pt_expt.train.validation import (
    FullValidationResult,
    FullValidator,
    build_full_validators,
)


@pytest.mark.parametrize(
    "spin, mode",
    [(False, "mse"), (False, "mae"), (False, "huber"), (True, "mse"), (True, "mae")],
)
def test_virial_projection_gradients(spin: bool, mode: str) -> None:
    parameters = {
        "starter_learning_rate": 1.0,
        "start_pref_e": 0.0,
        "limit_pref_e": 0.0,
        "start_pref_v": 1.0,
        "limit_pref_v": 1.0,
        "symm_virial": True,
        "loss_func": "mse" if mode == "huber" else mode,
    }
    if not spin:
        parameters.update(start_pref_f=0.0, limit_pref_f=0.0, use_huber=mode == "huber")
    loss = (EnergySpinLoss if spin else EnergyLoss)(**parameters)
    virial = (
        torch.arange(9, dtype=torch.float64, device="cpu")
        .reshape(1, 9)
        .requires_grad_()
    )
    labels = {"virial": torch.zeros_like(virial), "find_virial": 1.0}

    def objective(value: torch.Tensor) -> torch.Tensor:
        prediction = {
            "energy": torch.zeros((1, 1), dtype=value.dtype, device="cpu"),
            "virial": value,
        }
        return loss(1.0, 2, prediction, labels)[0]

    assert torch.autograd.gradcheck(objective, (virial,))
    (gradient,) = torch.autograd.grad(objective(virial), virial, create_graph=True)
    torch.testing.assert_close(gradient.reshape(3, 3), gradient.reshape(3, 3).T)
    antisymmetric = torch.tensor(
        [[0.0, 2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=virial.dtype,
        device="cpu",
    )
    torch.testing.assert_close(objective(virial + antisymmetric), objective(virial))
    assert torch.isfinite(torch.autograd.functional.hessian(objective, virial)).all()


def test_full_validation_inherits_comparison_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = SimpleNamespace(
        has_spin=lambda: False,
        get_dim_fparam=lambda: 0,
        get_dim_aparam=lambda: 0,
        has_chg_spin_ebd=lambda: False,
    )
    labels = {
        "type": np.zeros((1, 2), dtype=np.int32),
        "coord": np.zeros((1, 6)),
        "box": np.eye(3).reshape(1, 9),
        "virial": np.zeros((1, 9)),
        "find_virial": 1.0,
    }
    data = SimpleNamespace(pbc=True, get_test=lambda: labels)
    ema = SimpleNamespace(validation_state={}, apply_shadow=lambda _: None)
    state = {}
    params = {
        "full_validation": True,
        "ema_full_validation": True,
        "full_val_start": 0.0,
        "save_best": False,
        "full_val_file": str(tmp_path / "val.log"),
        "validation_metric": "V:RMSE",
    }
    live, averaged = build_full_validators(
        validating_params=params,
        validation_data=data,
        model=model,
        state_store=state,
        num_steps=2,
        rank=0,
        restart_training=False,
        checkpoint_dir=tmp_path,
        ensure_supported=lambda: None,
        model_ema=ema,
        symm_virial=True,
    )
    for validator in (live, averaged):
        monkeypatch.setattr(
            validator,
            "_predict_outputs",
            lambda **_: {
                "virial": np.array([[0.0, 3.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            },
        )
        metrics = validator._evaluate_system(data)
        assert metrics["rmse_v_per_atom"][0] == 0.0
        assert validator.state_store["full_validation_symm_virial"] is True
        validator._write_log_file(
            FullValidationResult(
                display_step=1,
                metrics={k: v[0] for k, v in metrics.items()},
                selected_metric_key="rmse_v_per_atom",
                selected_metric_value=0.0,
                saved_best_path=None,
            )
        )
        content = validator.full_val_file.read_text()
        assert "symm_v=" not in content
        assert "SYMM_V_RMSE" in content
        assert "SYMM_V_MAE" in content
    with pytest.raises(ValueError, match="different symm_virial"):
        FullValidator(
            validating_params=params,
            validation_data=data,
            model=model,
            state_store=state,
            num_steps=2,
            rank=0,
            restart_training=True,
            checkpoint_dir=tmp_path,
            symm_virial=False,
        )


def test_multitask_without_full_validation_has_no_default_loss(tmp_path: Path) -> None:
    from deepmd.pt_expt.train.training import (
        Trainer,
    )

    trainer = SimpleNamespace(
        model={"a": object(), "b": object()},
        loss={"a": EnergyLoss(1.0, symm_virial=True), "b": EnergyLoss(1.0)},
        enable_compile=False,
        multi_task=True,
        _unwrapped=SimpleNamespace(train_infos={}),
        num_steps=2,
        rank=0,
        restart_training=False,
        save_ckpt=str(tmp_path / "model"),
        model_ema=None,
        sharding=None,
        matmul_precision=None,
    )
    assert Trainer._create_full_validators(
        trainer, validating_params={}, validation_data=None
    ) == (None, None)


@pytest.mark.parametrize("family", ["e", "v", "s"])
def test_full_validation_symmetric_metric_labels(family: str) -> None:
    from deepmd.dpmodel.train.validation import format_metric_for_log as shared_format
    from deepmd.pt_expt.train.validation import format_metric_for_log as torch_format
    from deepmd.pt_expt.train.validation import (
        parse_validation_metric,
    )
    from deepmd.utils.eval_metrics import (
        ENERGY_FULL_VALIDATION_PROFILE,
    )

    profile = ENERGY_FULL_VALIDATION_PROFILE
    selector, metric_key = parse_validation_metric(f"{family.upper()}:RMSE", profile)
    assert selector == f"{family}:rmse"
    assert metric_key == profile.metric_key_map[selector]
    for enabled in (False, True):
        expected = (
            "SYMM_" if enabled and family in {"v", "s"} else ""
        ) + family.upper()
        columns = profile.columns(f"{family}:rmse", symm_virial=enabled)
        assert any(label == f"{expected}_RMSE" for label, _ in columns)
        # The shared formatter reads the one profile of the portable trainer.
        for formatter, arguments in ((shared_format, ()), (torch_format, (profile,))):
            label, value, unit = formatter(
                f"{family}:rmse", 1.0, *arguments, symm_virial=enabled
            )
            assert label == f"{expected}:RMSE"
            assert value == 1000.0
            assert unit
