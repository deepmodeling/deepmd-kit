# SPDX-License-Identifier: LGPL-3.0-or-later
import json
from pathlib import (
    Path,
)
from typing import (
    Any,
)
from unittest.mock import (
    Mock,
    patch,
)

import h5py
import numpy as np
import pytest
import torch
from dargs.dargs import (
    ArgumentValueError,
)

from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.training import (
    Trainer,
)
from deepmd.pt.utils.env import (
    DEVICE,
)
from deepmd.pt.utils.stat import (
    compute_output_stats,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.stat_file import (
    StatFileSpec,
    open_stat_file,
)


def _load_dpa4_example() -> dict[str, Any]:
    """Load the DPA4 example used by configuration validation tests."""
    example_path = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "water"
        / "dpa4"
        / "input.json"
    )
    with example_path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_pt_training_example() -> dict[str, Any]:
    """Load a minimal PT training configuration with absolute data paths."""
    test_root = Path(__file__).resolve().parent
    with (test_root / "water" / "se_atten.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    systems = [str(test_root / "water" / "data" / "single")]
    config["training"]["training_data"]["systems"] = systems
    config["training"]["validation_data"]["systems"] = systems
    config["training"]["numb_steps"] = 0
    return config


def _energy_model_params() -> dict[str, Any]:
    """Build a minimal PT energy-model configuration."""
    return {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [4, 4],
            "rcut": 3.0,
            "rcut_smth": 2.5,
            "neuron": [4, 8],
            "axis_neuron": 4,
            "precision": "float64",
        },
        "fitting_net": {
            "type": "ener",
            "neuron": [8],
            "precision": "float64",
        },
    }


def _energy_stat_sample() -> list[dict[str, Any]]:
    """Build statistics data for the minimal PT energy model."""
    return [
        {
            "coord": torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                ],
                dtype=torch.float64,
                device=DEVICE,
            ),
            "atype": torch.tensor(
                [[0, 0], [1, 1]],
                dtype=torch.int64,
                device=DEVICE,
            ),
            "box": None,
            "natoms": torch.tensor(
                [[2, 2, 2, 0], [2, 2, 0, 2]],
                dtype=torch.int64,
                device=DEVICE,
            ),
            "energy": torch.tensor(
                [[2.0], [4.0]],
                dtype=torch.float64,
                device=DEVICE,
            ),
            "find_energy": np.float32(1.0),
        }
    ]


def test_default_stat_file_mode_remains_writable(tmp_path: Path) -> None:
    stat_file = tmp_path / "stat.hdf5"
    with open_stat_file(StatFileSpec(str(stat_file))) as stat_path:
        assert stat_path is not None
        assert stat_path.mode == "a"
        (stat_path / "value").save_numpy(np.array([1.0]))
        assert (stat_path / "value").load_numpy().tolist() == [1.0]


def test_read_stat_file_mode_reads_existing_cache(tmp_path: Path) -> None:
    stat_file = tmp_path / "stat.hdf5"
    with h5py.File(stat_file, "w") as file:
        file.create_dataset("value", data=[1.0])

    with open_stat_file(StatFileSpec(str(stat_file), "read")) as stat_path:
        assert stat_path is not None
        assert stat_path.mode == "r"
        assert (stat_path / "value").load_numpy().tolist() == [1.0]


def test_read_stat_file_mode_requires_existing_cache(tmp_path: Path) -> None:
    stat_file = tmp_path / "missing.hdf5"
    with pytest.raises(FileNotFoundError, match="does not exist in read mode"):
        with open_stat_file(StatFileSpec(str(stat_file), "read")):
            pass


def test_read_stat_file_mode_requires_cache_path() -> None:
    with pytest.raises(ValueError, match="requires `stat_file`"):
        StatFileSpec(None, "read")


def test_frozen_initialization_does_not_open_read_cache(tmp_path: Path) -> None:
    config = _load_pt_training_example()
    missing_cache = tmp_path / "missing.hdf5"
    config["training"]["stat_file"] = str(missing_cache)
    config["training"]["stat_file_mode"] = "read"
    frozen_model = Mock()
    frozen_model.state_dict.return_value = {}

    with patch(
        "deepmd.pt.train.training.torch.jit.load",
        return_value=frozen_model,
    ) as load_frozen:
        trainer = get_trainer(config, init_frz_model="model.pth")

    load_frozen.assert_called_once()
    assert trainer.model is not None
    assert not missing_cache.exists()


def test_read_stat_file_mode_rejects_incomplete_statistics_cache(
    tmp_path: Path,
) -> None:
    stat_file = tmp_path / "stat.hdf5"
    with h5py.File(stat_file, "w") as file:
        file.create_dataset("bias_atom_energy", data=np.zeros((1, 1)))

    with open_stat_file(StatFileSpec(str(stat_file), "read")) as stat_path:
        assert stat_path is not None
        with pytest.raises(FileNotFoundError, match="std_atom_energy"):
            compute_output_stats(
                lambda: pytest.fail("read-only statistics must not sample data"),
                ntypes=1,
                keys=["energy"],
                stat_file_path=stat_path,
            )


def test_read_stat_file_mode_loads_complete_cache_from_two_readers(
    tmp_path: Path,
) -> None:
    stat_file = tmp_path / "stat.hdf5"
    with h5py.File(stat_file, "w") as file:
        file.create_dataset("bias_atom_energy", data=np.zeros((1, 1)))
        file.create_dataset("std_atom_energy", data=np.ones((1, 1)))

    spec = StatFileSpec(str(stat_file), "read")
    with open_stat_file(spec) as reader_one, open_stat_file(spec) as reader_two:
        assert reader_one is not None
        assert reader_two is not None
        for reader in (reader_one, reader_two):
            bias, std = compute_output_stats(
                lambda: pytest.fail("complete read-only cache must not sample data"),
                ntypes=1,
                keys=["energy"],
                stat_file_path=reader,
            )
            assert bias["energy"].shape == (1, 1)
            assert std["energy"].shape == (1, 1)


def test_distributed_statistics_failure_reaches_peer_rank() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.is_distributed = True
    trainer.rank = 1
    action = Mock()

    def report_chief_failure(holder: list[bool], **_: Any) -> None:
        holder[0] = True

    with (
        patch(
            "deepmd.pt.train.training.dist.broadcast_object_list",
            side_effect=report_chief_failure,
        ),
        pytest.raises(RuntimeError, match="Rank 0 failed during statistics"),
    ):
        trainer._run_stat_on_chief(action, operation="statistics initialization")

    action.assert_not_called()


def test_update_mode_recomputes_partial_multi_output_cache(tmp_path: Path) -> None:
    stat_file = tmp_path / "stat.hdf5"
    sampled = _energy_stat_sample()
    sampled[0]["property"] = torch.tensor(
        [[1.0], [3.0]],
        dtype=torch.float64,
        device=DEVICE,
    )
    sampled[0]["find_property"] = np.float32(1.0)
    sample_count = 0

    def sample() -> list[dict[str, Any]]:
        nonlocal sample_count
        sample_count += 1
        return sampled

    with open_stat_file(StatFileSpec(str(stat_file))) as stat_path:
        assert stat_path is not None
        (stat_path / "bias_atom_energy").save_numpy(np.full((2, 1), 100.0))
        (stat_path / "bias_atom_property").save_numpy(np.full((2, 1), 100.0))
        (stat_path / "std_atom_energy").save_numpy(np.full((2, 1), 100.0))
        bias, std = compute_output_stats(
            sample,
            ntypes=2,
            keys=["energy", "property"],
            stat_file_path=stat_path,
        )

        assert sample_count == 1
        assert set(bias) == {"energy", "property"}
        assert set(std) == {"energy", "property"}

    with h5py.File(stat_file, "r") as file:
        assert set(file) == {
            "bias_atom_energy",
            "bias_atom_property",
            "std_atom_energy",
            "std_atom_property",
        }
        assert not np.all(file["bias_atom_energy"][:] == 100.0)


def test_update_mode_replaces_orphaned_output_pair(tmp_path: Path) -> None:
    stat_file = tmp_path / "stat.hdf5"
    with h5py.File(stat_file, "w") as file:
        file.create_dataset("bias_atom_property", data=np.zeros((2, 1)))

    sampler = Mock(return_value=_energy_stat_sample())
    with open_stat_file(StatFileSpec(str(stat_file))) as stat_path:
        assert stat_path is not None
        bias, std = compute_output_stats(
            sampler,
            ntypes=2,
            keys=["energy", "property"],
            stat_file_path=stat_path,
        )

    sampler.assert_called_once_with()
    assert set(bias) == {"energy"}
    assert set(std) == {"energy"}
    with h5py.File(stat_file, "r") as file:
        assert set(file) == {"bias_atom_energy", "std_atom_energy"}
    original = stat_file.read_bytes()

    sampler.reset_mock(side_effect=True)
    sampler.side_effect = AssertionError(
        "A complete update cache must not sample data."
    )
    with open_stat_file(StatFileSpec(str(stat_file))) as stat_path:
        assert stat_path is not None
        bias, std = compute_output_stats(
            sampler,
            ntypes=2,
            keys=["energy", "property"],
            stat_file_path=stat_path,
        )

    sampler.assert_not_called()
    assert set(bias) == {"energy"}
    assert set(std) == {"energy"}
    assert stat_file.read_bytes() == original


def test_energy_model_reloads_update_cache_in_read_mode(tmp_path: Path) -> None:
    sampled = _energy_stat_sample()
    stat_file = tmp_path / "stat.hdf5"

    with open_stat_file(StatFileSpec(str(stat_file))) as update_path:
        assert update_path is not None
        update_model = get_model(_energy_model_params()).to(DEVICE)
        update_model.compute_or_load_stat(lambda: sampled, update_path)
        stat_root = update_path / "O H"
        assert (stat_root / "bias_atom_energy").is_file()
        assert not (stat_root / "bias_atom_mask").is_file()
    original = stat_file.read_bytes()

    with open_stat_file(StatFileSpec(str(stat_file), "read")) as read_path:
        assert read_path is not None
        read_model = get_model(_energy_model_params()).to(DEVICE)
        read_model.compute_or_load_stat(
            lambda: pytest.fail("complete read-only cache must not sample data"),
            read_path,
        )
    assert stat_file.read_bytes() == original


def test_energy_model_reuses_update_cache_without_modifying_hdf5(
    tmp_path: Path,
) -> None:
    stat_file = tmp_path / "stat.hdf5"
    with open_stat_file(StatFileSpec(str(stat_file))) as stat_path:
        assert stat_path is not None
        model = get_model(_energy_model_params()).to(DEVICE)
        model.compute_or_load_stat(_energy_stat_sample, stat_path)
    original = stat_file.read_bytes()

    with open_stat_file(StatFileSpec(str(stat_file))) as stat_path:
        assert stat_path is not None
        model = get_model(_energy_model_params()).to(DEVICE)
        model.compute_or_load_stat(
            lambda: pytest.fail("complete update cache must not sample data"),
            stat_path,
        )

    assert stat_file.read_bytes() == original


def test_read_mode_rejects_missing_descriptor_stats_before_sampling(
    tmp_path: Path,
) -> None:
    stat_file = tmp_path / "stat.hdf5"
    with h5py.File(stat_file, "w") as file:
        file.create_group("O H")

    with open_stat_file(StatFileSpec(str(stat_file), "read")) as read_path:
        assert read_path is not None
        model = get_model(_energy_model_params()).to(DEVICE)
        with pytest.raises(FileNotFoundError, match="environment statistics"):
            model.compute_or_load_stat(
                lambda: pytest.fail("read-only cache miss must not sample data"),
                read_path,
            )

    with h5py.File(stat_file, "r") as file:
        assert list(file.keys()) == ["O H"]
        assert len(file["O H"]) == 0


def test_read_mode_rejects_partial_descriptor_stats_before_sampling(
    tmp_path: Path,
) -> None:
    stat_file = tmp_path / "stat.hdf5"
    with open_stat_file(StatFileSpec(str(stat_file))) as update_path:
        assert update_path is not None
        model = get_model(_energy_model_params()).to(DEVICE)
        model.compute_or_load_stat(_energy_stat_sample, update_path)

    with h5py.File(stat_file, "r+") as file:
        type_map_group = file["O H"]
        descriptor_groups = [
            value for value in type_map_group.values() if isinstance(value, h5py.Group)
        ]
        assert len(descriptor_groups) == 1
        del descriptor_groups[0]["r_0"]

    with open_stat_file(StatFileSpec(str(stat_file), "read")) as read_path:
        assert read_path is not None
        model = get_model(_energy_model_params()).to(DEVICE)
        with pytest.raises(FileNotFoundError, match="'r_0'"):
            model.compute_or_load_stat(
                lambda: pytest.fail("partial read-only cache must not sample data"),
                read_path,
            )


def test_stat_file_mode_configuration_validation() -> None:
    config = _load_dpa4_example()
    config["training"].pop("stat_file_mode")
    normalized = normalize(config)
    assert normalized["training"]["stat_file_mode"] == "update"

    config["training"]["stat_file_mode"] = "read"
    normalized = normalize(config)
    assert normalized["training"]["stat_file_mode"] == "read"

    config["training"]["stat_file_mode"] = "invalid"
    with pytest.raises(ArgumentValueError, match="stat_file_mode"):
        normalize(config)

    config["training"]["stat_file_mode"] = "read"
    config["training"].pop("stat_file")
    with pytest.raises(ValueError, match="stat_file_mode"):
        normalize(config)
