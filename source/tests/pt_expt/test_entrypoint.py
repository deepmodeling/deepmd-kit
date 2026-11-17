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


@pytest.mark.parametrize("model_type", [None, "standard", "ener"])
def test_dpa4c_statistics_use_selected_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog,
    model_type,
) -> None:
    import copy
    import logging

    from deepmd.dpmodel.utils.neighbor_stat import NeighborStat as NumpyStat
    from deepmd.pt_expt.utils.env import (
        DEVICE,
    )
    from deepmd.pt_expt.utils.neighbor_stat import NeighborStat as TorchStat

    from ..common.test_neighbor_stat_batching import (
        make_data,
        reference_stat,
    )

    data = make_data(tmp_path / "system")
    expected = reference_stat(data, 1e-6, True)[0]
    monkeypatch.setenv("DP_NEIGHBOR_STAT_BATCH_SIZE", "3")
    monkeypatch.setenv("DP_INFER_BATCH_SIZE", "1000000")
    config = {
        "model": {
            "descriptor": {"type": "dpa4c", "rcut": 6.0},
            "type_map": ["A", "B"],
            "fitting": {"type": "ener"},
        },
        "training": {
            "training_data": {"systems": [str(tmp_path / "system")], "batch_size": 1}
        },
    }
    if model_type is not None:
        config["model"]["type"] = model_type
    original_config = copy.deepcopy(config)

    def forbidden(*args, **kwargs):
        raise AssertionError("DPA4C entered the NumPy statistics path")

    monkeypatch.setattr(NumpyStat, "iterator", forbidden)
    original_op = TorchStat.__init__
    devices, sizes = [], []

    def instrument(self, *args, **kwargs):
        original_op(self, *args, **kwargs)
        op = self.op

        def record(coord, atype, cell, nbuff):
            devices.append(coord.device)
            sizes.append(coord.shape[0])
            assert nbuff is not None and nbuff.device == coord.device
            return op(coord, atype, cell, nbuff)

        self.op = record

    monkeypatch.setattr(TorchStat, "__init__", instrument)
    with caplog.at_level(logging.INFO):
        updated, minimum = PTExptTrainEntrypoint().update_neighbor_stat(
            config,
            TrainEntrypointOptions(input_file="input.json"),
            multi_task=False,
        )
    assert updated == original_config
    assert minimum == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert devices and all(device == DEVICE for device in devices)
    assert sum(sizes) == 17 and max(sizes) <= 3
    assert f"DPA4C neighbor statistics device: {DEVICE}" in caplog.text


@pytest.mark.parametrize(
    "model",
    [
        {"descriptor": {"type": "se_e2_a"}},
        {"type": "linear_ener", "descriptor": {"type": "dpa4c"}},
        {
            "type": "standard",
            "fitting": {"type": "dipole"},
            "descriptor": {"type": "dpa4c"},
        },
    ],
)
def test_other_models_keep_common_statistics(monkeypatch, model) -> None:
    from deepmd.pt_expt.descriptor.dpa4c import (
        DescrptDPA4C,
    )
    from deepmd.pt_expt.model import (
        BaseModel,
    )

    calls = []
    monkeypatch.setattr(
        "deepmd.pt_expt.entrypoints.main.get_neighbor_stat_data",
        lambda *args: None,
    )

    def fallback(cls, data, type_map, params):
        calls.append(params)
        return {**params, "visited": True}, 0.25

    def forbidden(*args):
        raise AssertionError("Unexpected DPA4C route")

    monkeypatch.setattr(BaseModel, "update_sel", classmethod(fallback))
    monkeypatch.setattr(DescrptDPA4C, "update_sel", classmethod(forbidden))
    config = {"model": model, "training": {"training_data": {}}}
    result, minimum = PTExptTrainEntrypoint().update_neighbor_stat(
        config,
        TrainEntrypointOptions(input_file="input.json"),
        multi_task=False,
    )
    assert calls == [model]
    assert result["model"]["visited"] and minimum == 0.25


def test_multitask_neighbor_statistics_keep_each_result(monkeypatch) -> None:
    from deepmd.pt_expt.descriptor.dpa4c import (
        DescrptDPA4C,
    )
    from deepmd.pt_expt.model import (
        BaseModel,
    )

    monkeypatch.setattr(
        "deepmd.pt_expt.entrypoints.main.get_neighbor_stat_data",
        lambda *args: None,
    )
    monkeypatch.setattr(
        DescrptDPA4C,
        "update_sel",
        classmethod(lambda cls, data, type_map, params: (params.copy(), 0.125)),
    )
    monkeypatch.setattr(
        BaseModel,
        "update_sel",
        classmethod(
            lambda cls, data, type_map, params: ({**params, "visited": True}, 0.75)
        ),
    )
    config = {
        "model": {
            "model_dict": {
                "dpa4c": {"type": "standard", "descriptor": {"type": "dpa4c"}},
                "other": {"descriptor": {"type": "se_e2_a"}},
            }
        },
        "training": {
            "data_dict": {key: {"training_data": {}} for key in ("dpa4c", "other")}
        },
    }
    result, distances = PTExptTrainEntrypoint().update_neighbor_stat(
        config,
        TrainEntrypointOptions(input_file="input.json"),
        multi_task=True,
    )
    assert distances == {"dpa4c": 0.125, "other": 0.75}
    assert result["model"]["model_dict"]["other"]["visited"]
