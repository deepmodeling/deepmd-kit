# SPDX-License-Identifier: LGPL-3.0-or-later
import pytest

from deepmd.dpmodel.train import (
    TrainEntrypointOptions,
)
from deepmd.pt_expt.entrypoints.main import (
    PTExptTrainEntrypoint,
    _ensure_pt_expt_model_suffix,
    train,
)


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


@pytest.mark.parametrize("restore", [None, "restart", "init_model", "finetune"])
def test_graph_neighbor_statistics_skip_is_per_task(monkeypatch, restore):
    """Graph tasks must not even open the full neighbor-stat dataset."""
    from deepmd.pt_expt.model import (
        BaseModel,
    )
    from deepmd.utils.argcheck import (
        DEFAULT_DPA4_SEL,
    )

    calls = []
    monkeypatch.setattr(
        "deepmd.pt_expt.entrypoints.main.get_neighbor_stat_data",
        lambda params, type_map: calls.append(params["systems"]),
    )
    monkeypatch.setattr(BaseModel, "update_sel", lambda *args: (args[-1], 0.5))
    models = {
        "graph": {"descriptor": {"type": "dpa4", "sel": 120}},
        "graph_auto": {"descriptor": {"type": "dpa4", "sel": "auto"}},
        "dense": {"descriptor": {"type": "se_e2_a", "sel": [20, 20]}},
        "auto": {"descriptor": {"type": "dpa1", "sel": "auto"}},
    }
    config = {
        "model": {"model_dict": models},
        "training": {
            "data_dict": {key: {"training_data": {"systems": key}} for key in models}
        },
    }
    options = TrainEntrypointOptions(input_file="unused.json")
    if restore:
        setattr(options, restore, "checkpoint.pt")
    _, distances = PTExptTrainEntrypoint().update_neighbor_stat(
        config, options, multi_task=True
    )
    assert calls == (
        ["graph", "graph_auto", "dense", "auto"] if restore else ["dense", "auto"]
    )
    assert distances == {
        "graph": 0.5 if restore else None,
        "graph_auto": 0.5 if restore else None,
        "dense": 0.5,
        "auto": 0.5,
    }
    tasks = config["model"]["model_dict"]
    assert tasks["graph"]["descriptor"]["sel"] == 120
    assert tasks["graph_auto"]["descriptor"]["sel"] == (
        "auto" if restore else DEFAULT_DPA4_SEL
    )


def test_graph_single_task_skips_neighbor_statistics(monkeypatch):
    """A graph-only task never opens data for neighbor statistics."""

    def unexpected(*args):
        raise AssertionError("graph-only model opened neighbor-stat data")

    monkeypatch.setattr(
        "deepmd.pt_expt.entrypoints.main.get_neighbor_stat_data", unexpected
    )
    config = {
        "model": {"descriptor": {"type": "dpa4c"}},
        "training": {"training_data": {}},
    }
    actual, distance = PTExptTrainEntrypoint().update_neighbor_stat(
        config, TrainEntrypointOptions(input_file="unused.json"), multi_task=False
    )
    assert actual is config
    assert distance is None


@pytest.mark.parametrize(
    "model",
    [
        {"type": "dpa4", "descriptor": {"sel": "auto:1.5"}, "fitting_net": {}},
        {"preset": "dpa4c-nano-v20260911"},
    ],
)
def test_graph_default_skip_after_normalization(monkeypatch, tmp_path, model):
    """Preset expansion and auto normalization precede the trainer and output dump."""
    import json

    from deepmd.utils.argcheck import (
        DEFAULT_DPA4_SEL,
    )

    model["type_map"] = ["H"]
    config = {
        "model": model,
        "learning_rate": {"type": "exp", "start_lr": 0.001, "stop_lr": 0.0001},
        "loss": {"type": "ener"},
        "training": {
            "training_data": {"systems": "must-not-open"},
            "numb_steps": 1,
        },
    }
    input_file = tmp_path / "input.json"
    output = tmp_path / "out.json"
    input_file.write_text(json.dumps(config))
    entrypoint = PTExptTrainEntrypoint()
    observed = []
    monkeypatch.setattr(
        entrypoint, "run_training", lambda c, o, n: observed.append((c, n))
    )
    monkeypatch.setattr(entrypoint, "print_summary", lambda: None)
    monkeypatch.setattr(entrypoint, "setup_run", lambda *args: None)
    monkeypatch.setattr(entrypoint, "teardown_run", lambda *args: None)

    def unexpected(*args):
        raise AssertionError("default graph training must not scan neighbor statistics")

    monkeypatch.setattr(
        "deepmd.pt_expt.entrypoints.main.get_neighbor_stat_data", unexpected
    )
    entrypoint.run(
        TrainEntrypointOptions(input_file=str(input_file), output=str(output))
    )
    normalized, distance = observed[0]
    assert distance is None
    descriptor = normalized["model"]["descriptor"]
    if descriptor["type"] == "dpa4":
        assert descriptor["sel"] == DEFAULT_DPA4_SEL
    assert json.loads(output.read_text()) == normalized
