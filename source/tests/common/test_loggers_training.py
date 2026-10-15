# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the training log messages."""

import datetime
import logging
import os
import pickle
import sys
from collections.abc import (
    Iterator,
)
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)

import pytest

from deepmd.loggers import (
    WorkerLogConfig,
    is_node_main_process,
    set_log_handles,
)
from deepmd.loggers.training import (
    format_training_message,
    log_parameter_counts,
)

_LOGGER = "deepmd.loggers.training"


@pytest.fixture
def isolated_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep CLI logging configuration local to each test."""
    logger = logging.getLogger("deepmd")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logger.level)
    monkeypatch.setattr(logger, "propagate", logger.propagate)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    for name in ("KMP_WARNINGS", "TF_CPP_MIN_LOG_LEVEL"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    yield
    for handler in logger.handlers:
        handler.close()


def _configure_rank(monkeypatch: pytest.MonkeyPatch, rank: int) -> None:
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(rank % 2))
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")


@pytest.mark.parametrize(
    ("rank", "local_rank", "node_visible"),
    [(0, 0, True), (1, 1, False), (2, 0, True), (2, None, False)],
)
@pytest.mark.usefixtures("isolated_logging")
def test_console_scopes_before_distributed_initialization(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rank: int,
    local_rank: int | None,
    node_visible: bool,
) -> None:
    _configure_rank(monkeypatch, rank)
    if local_rank is None:
        monkeypatch.delenv("LOCAL_RANK")
    set_log_handles(logging.DEBUG)
    logger = logging.getLogger("deepmd.dpmodel.utils.lmdb_data")
    logger.info("Dataset summary")
    logger.info("Compile finished", extra={"rank_scope": "all"})
    logger.info("Global metric", extra={"rank_scope": "global"})
    logger.debug("Cache detail")
    logger.warning("Decoder failure")
    logger.warning("Configuration warning", extra={"rank_scope": "node"})
    logger.error(
        "Device failure",
        exc_info=RuntimeError("Worker failed"),
        extra={"rank_scope": "global"},
    )

    output = capsys.readouterr().err
    assert ("Dataset summary" in output) == node_visible
    assert ("Global metric" in output) == (rank == 0)
    assert ("Configuration warning" in output) == node_visible
    for message in (
        "Compile finished",
        "Cache detail",
        "Decoder failure",
        "Device failure",
    ):
        assert f"[rank={rank}] {message}" in output
    for message in ("Dataset summary", "Global metric", "Configuration warning"):
        for line in output.splitlines():
            if line.endswith(message):
                assert "[rank=" not in line
    assert "RuntimeError: Worker failed" in output
    assert "local=" not in output
    assert "MainProcess" not in output


@pytest.mark.usefixtures("isolated_logging")
def test_single_process_logging_keeps_its_format_and_filename(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "train.log"
    set_log_handles(logging.INFO, log_path)
    logging.getLogger("deepmd.train").info("Single process")

    output = capsys.readouterr().err
    assert "DEEPMD INFO    Single process" in output
    assert "rank=" not in output
    assert "Single process" in log_path.read_text()
    assert list(tmp_path.iterdir()) == [log_path]
    assert is_node_main_process(0)
    assert not is_node_main_process(1)


@pytest.mark.usefixtures("isolated_logging")
def test_rank_files_retain_records_filtered_from_the_console(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "train.log"
    logger = logging.getLogger("deepmd.train")
    for rank in (2, 3):
        _configure_rank(monkeypatch, rank)
        set_log_handles(logging.INFO, log_path)
        logger.info("Rank %d dataset summary", rank)
        logger.debug("Disabled detail")
    set_log_handles(logging.INFO, log_path)
    logger.warning("After reinitialization")

    output = capsys.readouterr().err
    assert "Rank 2 dataset summary" in output
    assert "Rank 3 dataset summary" not in output
    assert output.count("After reinitialization") == 1
    for rank in (2, 3):
        rank_path = tmp_path / f"{log_path.stem}.rank{rank}{log_path.suffix}"
        content = rank_path.read_text()
        assert f"Rank {rank} dataset summary" in content
        assert content.count("dataset summary") == 1
        assert "Disabled detail" not in content
        assert "[rank=" not in content
        assert "MainProcess" not in content
    assert "After reinitialization" in rank_path.read_text()
    assert not log_path.exists()


@pytest.mark.parametrize("external_logging", [False, True])
@pytest.mark.usefixtures("isolated_logging")
def test_lmdb_scan_worker_inherits_the_rank_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    external_logging: bool,
) -> None:
    import lmdb
    import msgpack

    from deepmd.dpmodel.utils.lmdb_data import (
        LmdbDataReader,
        collect_lmdb_sampling_groups,
    )
    from deepmd.utils.data import (
        DataRequirementItem,
    )

    from .dpmodel.test_lmdb_data import (
        _create_lmdb,
    )

    _configure_rank(monkeypatch, 2)
    set_log_handles(logging.INFO, tmp_path / "train.log")
    if external_logging:
        # Logging integrations can keep terminal streams in their formatters.
        logger = logging.getLogger("deepmd")
        formatter = logging.Formatter("%(message)s")
        formatter.stream = sys.__stderr__
        logger.handlers[0].setFormatter(formatter)
        external_handler = logging.StreamHandler()
        external_handler.setFormatter(formatter)
        logger.addHandler(external_handler)
    # Worker logging retains its configured owner when the environment differs.
    _configure_rank(monkeypatch, 1)
    path = _create_lmdb(str(tmp_path / "frames.lmdb"), nframes=2)
    with lmdb.open(path) as environment:
        with environment.begin(write=True) as transaction:
            key = b"000000000001"
            frame = msgpack.unpackb(transaction.get(key), raw=False)
            frame.pop("forces")
            transaction.put(key, msgpack.packb(frame, use_bin_type=True))

    reader = LmdbDataReader(path, ["O", "H"], batch_size=1)
    shared_reader = LmdbDataReader(path, ["O", "H"], batch_size=1)
    try:
        reader.add_data_requirement(
            [DataRequirementItem("force", 3, atomic=True, must=False)]
        )
        groups = collect_lmdb_sampling_groups(reader)
    finally:
        reader.close()
        shared_reader.close()

    assert sorted(indices.tolist() for _, indices in groups) == [[0], [1]]
    content = (tmp_path / "train.rank2.log").read_text()
    assert "label-availability scan started" in content
    assert "label-availability scan completed" in content
    assert "SpawnProcess" not in content


@pytest.mark.usefixtures("isolated_logging")
def test_node_summary_does_not_take_checkpoint_ownership(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from deepmd.dpmodel.train import (
        build_checkpoint_stores,
        resolve_step_schedule,
    )

    _configure_rank(monkeypatch, 2)
    set_log_handles(logging.INFO)
    schedule = resolve_step_schedule(
        {"numb_epoch": 2},
        multi_task=False,
        model_keys=["Default"],
        training_data={},
        epoch_length=lambda _: 10,
        rank=2,
    )
    task_schedule = resolve_step_schedule(
        {"numb_steps": 20},
        multi_task=True,
        model_keys=["a", "b"],
        training_data={"a": [0], "b": [0, 1]},
        epoch_length=lambda _: 10,
        rank=2,
    )
    checkpoint_dir = tmp_path / "checkpoints"
    store, _ = build_checkpoint_stores(
        {
            "save_dir": str(checkpoint_dir),
            "ckpt_keep_ratio": 0.5,
            "save_freq": 5,
        },
        num_steps=schedule.num_steps,
        ema_prefix=tmp_path / "model.ema",
        rank=2,
    )

    output = capsys.readouterr().err
    assert "Computed num_steps=20" in output
    assert "defaulting to the number of systems per task" in output
    assert task_schedule.model_prob.tolist() == [1 / 3, 2 / 3]
    assert "Resolved checkpoint retention to 2" in output
    assert store.max_keep == 2
    assert not checkpoint_dir.exists()


@pytest.mark.parametrize("mpi_log", ["master", "collect", "workers"])
@pytest.mark.usefixtures("isolated_logging")
def test_explicit_mpi_logging_keeps_its_console_policy(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mpi_log: str,
) -> None:
    _configure_rank(monkeypatch, 3)
    mpi = SimpleNamespace(COMM_WORLD=SimpleNamespace(Get_rank=lambda: 1))
    monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=mpi))
    set_log_handles(logging.INFO, mpi_log=mpi_log)
    config = pickle.loads(pickle.dumps(WorkerLogConfig.capture()))
    monkeypatch.delitem(sys.modules, "mpi4py")
    config.configure()
    logging.getLogger("deepmd.train").info("MPI message")

    output = capsys.readouterr().err
    assert ("MPI message" in output) == (mpi_log != "master")
    assert "rank=3" not in output


def test_progress_message_reports_wall_time_alone() -> None:
    assert (
        format_training_message(batch=100, wall_time=18.41)
        == "Batch     100: total wall time = 18.41 s"
    )


def test_progress_message_appends_the_estimated_finish() -> None:
    message = format_training_message(
        batch=100,
        wall_time=18.41,
        eta=100,
        current_time=datetime.datetime(
            2026, 6, 7, 5, 21, 29, tzinfo=datetime.timezone.utc
        ),
    )

    assert message.startswith("Batch     100: total wall time = 18.41 s, eta = 0:01:40")


def test_single_task_parameter_count_is_reported_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger=_LOGGER):
        log_parameter_counts({"Default": (1_500_000, 2_000_000)}, multi_task=False)

    assert caplog.records[-1].message == "Model Params:  2.000 M   (Trainable: 1.500 M)"


def test_multi_task_parameter_counts_are_flagged_as_approximate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger=_LOGGER):
        log_parameter_counts(
            {"a": (1_000_000, 1_000_000), "b": (500_000, 500_000)},
            multi_task=True,
        )

    messages = [record.message for record in caplog.records]
    assert "may include duplicates" in messages[0]
    assert messages[1].startswith("Model Params [a]: 1.000 M")
    assert messages[2].startswith("Model Params [b]: 0.500 M")
