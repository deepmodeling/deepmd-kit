# SPDX-License-Identifier: LGPL-3.0-or-later
from unittest.mock import (
    Mock,
    patch,
)

import pytest

from deepmd.dpmodel.train import (
    TrainEntrypointOptions,
)
from deepmd.pt_expt.entrypoints.main import (
    PTExptTrainEntrypoint,
    _ensure_pt_expt_model_suffix,
    get_trainer,
    train,
)


@pytest.mark.parametrize("failure", ["validation", "stat_file", "summary", "trainer"])
def test_get_trainer_closes_data_on_setup_failure(failure: str) -> None:
    """Release acquired datasets even when the factory has not returned yet."""
    config = {
        "model": {"type_map": ["H"]},
        "training": {"training_data": {}, "validation_data": {}},
    }
    training_data, validation_data = Mock(), Mock()
    build_results = [training_data, validation_data]
    expected_error = RuntimeError
    if failure == "validation":
        build_results[1] = RuntimeError("validation failed")
    elif failure == "stat_file":
        # Exercise the real stat_file_spec property after both datasets open.
        config["training"]["stat_file_mode"] = "invalid"
        expected_error = ValueError
    with (
        patch(
            "deepmd.pt_expt.entrypoints.main._build_data_system",
            side_effect=build_results,
        ),
        patch(
            "deepmd.pt_expt.entrypoints.main.print_data_summaries",
            side_effect=RuntimeError("summary failed")
            if failure == "summary"
            else None,
        ),
        patch(
            "deepmd.pt_expt.entrypoints.main.training.Trainer",
            side_effect=RuntimeError("trainer failed")
            if failure == "trainer"
            else None,
        ),
    ):
        with pytest.raises(expected_error, match=failure):
            get_trainer(config)
    training_data.close.assert_called_once_with()
    if failure == "validation":
        validation_data.close.assert_not_called()
    else:
        validation_data.close.assert_called_once_with()


def test_get_trainer_closes_completed_and_partial_tasks() -> None:
    """Factory-local cleanup complements make_task_maps cleanup of prior tasks."""
    config = {
        "model": {
            "model_dict": {"first": {"type_map": ["H"]}, "second": {"type_map": ["H"]}}
        },
        "training": {
            "data_dict": {
                "first": {"training_data": {}, "validation_data": {}},
                "second": {"training_data": {}, "validation_data": {}},
            }
        },
    }
    datasets = [Mock(), Mock(), Mock()]
    with patch(
        "deepmd.pt_expt.entrypoints.main._build_data_system",
        side_effect=[*datasets, RuntimeError("second validation failed")],
    ):
        with pytest.raises(RuntimeError, match="second validation failed"):
            get_trainer(config)
    for dataset in datasets:
        dataset.close.assert_called_once_with()


def test_get_trainer_transfers_data_ownership_on_success() -> None:
    """Successful setup keeps both datasets open for the training loop."""
    config = {
        "model": {"type_map": ["H"]},
        "training": {"training_data": {}, "validation_data": {}},
    }
    training_data, validation_data = Mock(), Mock()
    with (
        patch(
            "deepmd.pt_expt.entrypoints.main._build_data_system",
            side_effect=[training_data, validation_data],
        ),
        patch("deepmd.pt_expt.entrypoints.main.print_data_summaries"),
        patch("deepmd.pt_expt.entrypoints.main.training.Trainer") as trainer,
    ):
        assert get_trainer(config) is trainer.return_value
    assert trainer.call_args.args[1] is training_data
    assert trainer.call_args.kwargs["validation_data"] is validation_data
    training_data.close.assert_not_called()
    validation_data.close.assert_not_called()


@pytest.mark.parametrize(
    ("model_path", "expected"),
    [
        (None, None),
        ("model", "model.pt"),
        ("model.pt", "model.pt"),
        ("model.pte", "model.pte"),
        ("model.pt2", "model.pt2"),
    ],
)
def test_pt_expt_model_suffix_accepts_checkpoint_and_export_suffixes(
    model_path: str | None,
    expected: str | None,
) -> None:
    assert _ensure_pt_expt_model_suffix(model_path) == expected


def test_pt_expt_train_entrypoint_normalizes_checkpoint_prefixes() -> None:
    options = TrainEntrypointOptions(
        input_file="input.json",
        init_model="init",
        restart="restart.pte",
        finetune="pretrain.pt2",
    )

    prepared = PTExptTrainEntrypoint().prepare_options(options)

    assert prepared.init_model == "init.pt"
    assert prepared.restart == "restart.pte"
    assert prepared.finetune == "pretrain.pt2"


def test_pt_expt_train_wrapper_uses_common_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[TrainEntrypointOptions] = []

    def fake_run(
        self: PTExptTrainEntrypoint,
        options: TrainEntrypointOptions,
    ) -> None:
        captured.append(options)

    monkeypatch.setattr(PTExptTrainEntrypoint, "run", fake_run)

    train(
        input_file="input.json",
        init_model="init",
        restart="restart",
        finetune="pretrain.pte",
        model_branch="head",
        use_pretrain_script=True,
        skip_neighbor_stat=True,
        output="normalized.json",
    )

    assert captured == [
        TrainEntrypointOptions(
            input_file="input.json",
            output="normalized.json",
            init_model="init",
            restart="restart",
            finetune="pretrain.pte",
            model_branch="head",
            use_pretrain_script=True,
            skip_neighbor_stat=True,
        )
    ]


def test_pt_expt_entrypoint_keeps_caller_owned_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed as dist

    calls: list[str] = []
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dist,
        "init_process_group",
        lambda *args, **kwargs: calls.append("init"),
    )
    monkeypatch.setattr(dist, "destroy_process_group", lambda: calls.append("destroy"))

    entrypoint = PTExptTrainEntrypoint()
    entrypoint.setup_run(TrainEntrypointOptions(input_file="input.json"), {})
    entrypoint.teardown_run(TrainEntrypointOptions(input_file="input.json"), {})

    assert calls == []


def test_pt_expt_entrypoint_destroys_only_owned_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed as dist

    state = {"initialized": False}
    calls: list[str] = []
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: state["initialized"])

    def init_process_group(*args, **kwargs) -> None:
        calls.append("init")
        state["initialized"] = True

    def destroy_process_group() -> None:
        calls.append("destroy")
        state["initialized"] = False

    monkeypatch.setattr(dist, "init_process_group", init_process_group)
    monkeypatch.setattr(dist, "destroy_process_group", destroy_process_group)

    entrypoint = PTExptTrainEntrypoint()
    entrypoint.setup_run(TrainEntrypointOptions(input_file="input.json"), {})
    entrypoint.teardown_run(TrainEntrypointOptions(input_file="input.json"), {})
    entrypoint.teardown_run(TrainEntrypointOptions(input_file="input.json"), {})

    assert calls == ["init", "destroy"]


def test_pt_expt_entrypoint_rejects_random_model_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deepmd.pt_expt.utils.multi_task.preprocess_shared_params",
        lambda model_params: (model_params, None),
    )
    entrypoint = PTExptTrainEntrypoint()

    with pytest.raises(ValueError, match="RANDOM"):
        entrypoint.preprocess_config(
            {"model": {"model_dict": {"RANDOM": {}}}},
            TrainEntrypointOptions(input_file="input.json"),
        )
