# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
)

import numpy as np
import pytest
import torch

from deepmd.pt_expt.entrypoints.main import (
    change_bias,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)
from deepmd.utils.data import (
    min_pair_dist_requirement,
)

from .test_change_bias import (
    EXAMPLE_DIR,
    _make_config,
    _make_subset_dataset,
    run_dp,
)


@pytest.fixture
def checkpoint(tmp_path):
    """Two branches with different type maps and shared descriptor storage."""
    params = _make_config(str(tmp_path))["model"]
    other_params = deepcopy(params)
    other_params["type_map"] = ["H", "O"]
    models = {"water": get_model(params), "other": get_model(other_params)}
    for model in models.values():
        bias = model.get_out_bias()
        model.set_out_bias(torch.full_like(bias, 7.0))
    wrapper = ModelWrapper(
        models,
        model_params={"model_dict": {"water": params, "other": other_params}},
    )
    state = wrapper.state_dict()
    # Real multi-task checkpoints can alias shared descriptor tensors.
    descriptor_key = next(
        k
        for k in state
        if k.startswith("model.water.") and "descr" in k and torch.is_tensor(state[k])
    )
    other_key = descriptor_key.replace("model.water.", "model.other.")
    state[other_key] = state[descriptor_key]
    checkpoint = {
        "model": state,
        "optimizer": {
            "step": torch.tensor(12, device=DEVICE),
            "shared": state[descriptor_key],
        },
        "lr_scheduler": {"last_epoch": 12},
    }
    path = tmp_path / "multi.pt"
    torch.save(checkpoint, path)
    return path, checkpoint, descriptor_key, other_key


def load(path):
    return torch.load(path, map_location=DEVICE, weights_only=True)


def assert_preserved(before, after, descriptor_key, other_key):
    """Only the selected branch's output statistics may change."""
    assert after["model"]["_extra_state"] == before["model"]["_extra_state"]
    assert after["lr_scheduler"] == before["lr_scheduler"]
    assert torch.equal(after["optimizer"]["step"], before["optimizer"]["step"])
    for key, value in before["model"].items():
        if key == "_extra_state" or key.startswith("model.water."):
            continue
        torch.testing.assert_close(after["model"][key], value, rtol=0, atol=0)
    shared = after["model"][descriptor_key]
    assert shared.data_ptr() == after["model"][other_key].data_ptr()
    assert shared.data_ptr() == after["optimizer"]["shared"].data_ptr()
    torch.testing.assert_close(shared, before["model"][descriptor_key], rtol=0, atol=0)


@pytest.mark.parametrize("wrapped", [True, False])
def test_explicit_bias(checkpoint, tmp_path, wrapped):
    path, before, descriptor_key, other_key = checkpoint
    if not wrapped:
        torch.save(before["model"], path)
    output = tmp_path / "updated.pt"
    run_dp(
        f"dp --pt-expt change-bias {path} --model-branch water -b 1.0 2.0 -o {output}"
    )
    after = load(output)
    if not wrapped:
        after = {
            **before,
            "model": after,
            "optimizer": {**before["optimizer"], "shared": after[descriptor_key]},
        }
    assert_preserved(before, after, descriptor_key, other_key)
    params = before["model"]["_extra_state"]["model_params"]
    models = {k: get_model(v) for k, v in params["model_dict"].items()}
    wrapper = ModelWrapper(models)
    wrapper.load_state_dict(after["model"])
    np.testing.assert_array_equal(
        models["water"].get_out_bias().detach().cpu().reshape(-1), [1.0, 2.0]
    )
    np.testing.assert_array_equal(
        models["other"].get_out_bias().detach().cpu().reshape(-1), [7.0, 7.0]
    )


@pytest.mark.parametrize("branch", [None, "missing"])
def test_invalid_branch(checkpoint, tmp_path, branch):
    path, *_ = checkpoint
    output = tmp_path / "invalid.pt"
    with pytest.raises(ValueError, match=r"Available branches.*water.*other"):
        change_bias(
            str(path), model_branch=branch, bias_value=[1.0, 2.0], output=str(output)
        )
    assert not output.exists()


@pytest.mark.parametrize("mode", ["set", "change"])
def test_statistical_bias(checkpoint, tmp_path, mode):
    """A selected branch agrees with the same model in a single-task checkpoint."""
    path, before, descriptor_key, other_key = checkpoint
    system = tmp_path / "data"
    _make_subset_dataset(str(Path(EXAMPLE_DIR) / "data" / "data_0"), str(system), 2)
    params = before["model"]["_extra_state"]["model_params"]["model_dict"]["water"]
    single = ModelWrapper(get_model(params), model_params=params).state_dict()
    for key, value in before["model"].items():
        if key.startswith("model.water."):
            single[key.replace("model.water.", "model.Default.")] = value.clone()
    single_path = tmp_path / "single.pt"
    torch.save(single, single_path)
    expected = tmp_path / "expected.pt"
    output = tmp_path / "updated.pt"
    change_bias(str(single_path), mode=mode, system=str(system), output=str(expected))
    with patch(
        "deepmd.pt_expt.entrypoints.main.min_pair_dist_requirement",
        wraps=min_pair_dist_requirement,
    ) as frame_filter:
        change_bias(
            str(path),
            mode=mode,
            system=str(system),
            model_branch="water",
            output=str(output),
        )
    frame_filter.assert_called_once_with(params, 0.0)
    after = load(output)
    expected_state = load(expected)
    assert_preserved(before, after, descriptor_key, other_key)
    changed = False
    for key, value in after["model"].items():
        if key.startswith("model.water."):
            torch.testing.assert_close(
                value,
                expected_state[key.replace("model.water.", "model.Default.")],
                rtol=0,
                atol=1e-10,
            )
            changed |= not torch.equal(value, before["model"][key])
    assert changed


@pytest.mark.parametrize("numb_batch", [0, 1, 5])
@pytest.mark.parametrize("mode", ["set", "change"])
def test_unequal_system_lengths(checkpoint, tmp_path, numb_batch, mode):
    """All-frame calibration visits each system fully without wrapping short ones."""
    path, before, descriptor_key, other_key = checkpoint
    systems = []
    for index, length in enumerate((3, 7)):
        system = tmp_path / f"system_{index}"
        _make_subset_dataset(
            str(Path(EXAMPLE_DIR) / "data" / "data_0"), str(system), length
        )
        # Distinct frame labels let the test detect repeated draws as well as
        # the former shortest-system truncation.
        np.save(
            system / "set.000" / "energy.npy",
            (np.arange(length, dtype=float) + index * 100).reshape(-1, 1),
        )
        systems.append(system)
    datafile = tmp_path / "systems.txt"
    datafile.write_text("\n".join(str(p) for p in systems) + "\n")
    output = tmp_path / "updated.pt"
    captured = []

    def capture(model, sampled, **kwargs):
        captured.extend(sampled)
        return model

    with patch(
        "deepmd.pt_expt.train.training.model_change_out_bias", side_effect=capture
    ):
        change_bias(
            str(path),
            mode=mode,
            datafile=str(datafile),
            numb_batch=numb_batch,
            model_branch="water",
            output=str(output),
        )
    assert len(captured) == 2
    for index, (sample, length) in enumerate(zip(captured, (3, 7), strict=True)):
        labels = np.asarray(sample["energy"]).reshape(-1)
        expected_count = length if numb_batch == 0 else min(numb_batch, length)
        assert len(labels) == expected_count
        assert len(np.unique(labels)) == expected_count
        assert set(labels) <= set(np.arange(length) + index * 100)
        if numb_batch == 0:
            np.testing.assert_array_equal(
                np.sort(labels), np.arange(length) + index * 100
            )
    assert_preserved(before, load(output), descriptor_key, other_key)
