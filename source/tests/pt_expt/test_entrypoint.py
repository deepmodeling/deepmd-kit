# SPDX-License-Identifier: LGPL-3.0-or-later
import pytest

from deepmd.dpmodel.train import (
    TrainEntrypointOptions,
)
from deepmd.pt_expt.entrypoints.main import (
    PTExptTrainEntrypoint,
    _ensure_pt_expt_model_suffix,
    _ensure_stat_file_path,
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


def test_pt_expt_checkpoint_cleanup_keeps_newest_steps(tmp_path) -> None:
    from deepmd.pt_expt.train.training import (
        Trainer,
    )

    trainer = Trainer.__new__(Trainer)
    trainer.save_ckpt = str(tmp_path / "model.ckpt")
    trainer.max_ckpt_keep = 2
    for step in (1, 2, 3):
        (tmp_path / f"model.ckpt-{step}.pt").write_text("")
    (tmp_path / "model.ckpt.pt").symlink_to("model.ckpt-3.pt")

    trainer._cleanup_old_checkpoints()

    assert not (tmp_path / "model.ckpt-1.pt").exists()
    assert (tmp_path / "model.ckpt-2.pt").exists()
    assert (tmp_path / "model.ckpt-3.pt").exists()
    assert (tmp_path / "model.ckpt.pt").exists()


def test_pt_expt_latest_checkpoint_link_uses_relative_target(tmp_path) -> None:
    from deepmd.pt_expt.train.training import (
        _replace_latest_checkpoint_link,
    )

    ckpt_path = tmp_path / "ckpts" / "model-1.pt"
    ckpt_path.parent.mkdir()
    ckpt_path.write_text("")
    latest = tmp_path / "ckpts" / "model.pt"

    _replace_latest_checkpoint_link(latest, ckpt_path)

    assert latest.is_symlink()
    assert latest.resolve() == ckpt_path
    assert latest.readlink().as_posix() == "model-1.pt"


def test_pt_expt_save_checkpoint_creates_parent_and_latest_link(tmp_path) -> None:
    from deepmd.pt_expt.train.training import (
        Trainer,
    )

    class DummyWrapper:
        train_infos: dict[str, int]
        model: dict[str, object]

        def __init__(self) -> None:
            self.train_infos = {}
            self.model = {}

        def state_dict(self) -> dict[str, object]:
            return {}

    class DummyOptimizer:
        def state_dict(self) -> dict[str, object]:
            return {}

    trainer = Trainer.__new__(Trainer)
    trainer.wrapper = DummyWrapper()
    trainer.optimizer = DummyOptimizer()
    trainer.save_ckpt = str(tmp_path / "ckpts" / "model")
    trainer.max_ckpt_keep = 2

    trainer.save_checkpoint(1)

    ckpt_path = tmp_path / "ckpts" / "model-1.pt"
    latest = tmp_path / "ckpts" / "model.pt"
    assert ckpt_path.exists()
    assert latest.is_symlink()
    assert latest.resolve() == ckpt_path
    assert latest.readlink().as_posix() == "model-1.pt"


def test_pt_expt_stat_file_path_creates_hdf5_parent(tmp_path) -> None:
    stat_file = tmp_path / "stats" / "model_stat.hdf5"

    stat_path = _ensure_stat_file_path(str(stat_file))

    assert stat_file.exists()
    assert stat_path is not None
