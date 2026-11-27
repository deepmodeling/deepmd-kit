# SPDX-License-Identifier: LGPL-3.0-or-later
"""PT must not silently accept a PT-expt-only task sampling policy."""

import json
from copy import (
    deepcopy,
)
from unittest.mock import (
    patch,
)

import pytest

from deepmd.pt.entrypoints import (
    main,
)
from deepmd.utils.argcheck import (
    normalize,
)


def _multitask_config():
    keys = ("first", "second")
    model = {
        "type_map": ["H"],
        "descriptor": {"type": "se_e2_a", "sel": [1]},
        "fitting_net": {},
    }
    return {
        "model": {"model_dict": {key: deepcopy(model) for key in keys}},
        "training": {
            "numb_steps": 1,
            "data_dict": {
                key: {"training_data": {"systems": "unused-dataset"}} for key in keys
            },
        },
    }


def _train(path):
    main.train(
        input_file=str(path),
        init_model=None,
        restart=None,
        finetune=None,
        init_frz_model=None,
        model_branch="",
        skip_neighbor_stat=True,
    )


@pytest.mark.parametrize("sampling", ["synchronized", "independent"])
def test_pt_train_rejects_explicit_task_sampling(tmp_path, monkeypatch, sampling):
    config = _multitask_config()
    config["training"]["task_sampling"] = sampling
    # This is a valid shared-schema configuration; the backend must reject it.
    config = normalize(config, multi_task=True)
    assert config["training"]["task_sampling"] == sampling
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(main.env, "CUSTOM_OP_USE_JIT", False)
    with (
        patch.object(main, "SummaryPrinter"),
        patch.object(main, "preprocess_shared_params") as preprocess,
        patch.object(main, "get_trainer") as get_trainer,
        pytest.raises(ValueError, match=r"task_sampling.*not supported.*pt backend"),
    ):
        _train(path)
    preprocess.assert_not_called()
    get_trainer.assert_not_called()


@pytest.mark.parametrize("sampling", ["synchronized", "independent"])
def test_pt_get_trainer_rejects_explicit_task_sampling(sampling):
    config = _multitask_config()
    config["training"]["task_sampling"] = sampling
    config = normalize(config, multi_task=True)
    with (
        patch.object(main, "create_dataset") as create_dataset,
        patch.object(main.training, "Trainer") as trainer,
        pytest.raises(ValueError, match=r"task_sampling.*not supported.*pt backend"),
    ):
        main.get_trainer(config)
    create_dataset.assert_not_called()
    trainer.assert_not_called()


def test_pt_train_without_task_sampling_keeps_existing_path(tmp_path, monkeypatch):
    config = normalize(_multitask_config(), multi_task=True)
    assert "task_sampling" not in config["training"]
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(main.env, "CUSTOM_OP_USE_JIT", False)

    class ReachedPreprocessing(Exception):
        pass

    with (
        patch.object(main, "SummaryPrinter"),
        patch.object(
            main, "preprocess_shared_params", side_effect=ReachedPreprocessing
        ) as preprocess,
        pytest.raises(ReachedPreprocessing),
    ):
        _train(path)
    preprocess.assert_called_once()
