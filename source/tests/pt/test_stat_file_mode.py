# SPDX-License-Identifier: LGPL-3.0-or-later
import json
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import h5py
import numpy as np
import pytest
import torch
from dargs.dargs import (
    ArgumentValueError,
)

from deepmd.pt.entrypoints.main import (
    _prepare_stat_file_path,
)
from deepmd.pt.model.model import (
    get_model,
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
from deepmd.utils.path import (
    DPH5Path,
    DPPath,
)


def _close_stat_path(stat_path: DPPath) -> None:
    """Close a test HDF5 path and reset its process-local caches."""
    assert isinstance(stat_path, DPH5Path)
    stat_path.root.close()
    DPH5Path._load_h5py.cache_clear()
    DPH5Path._file_keys.cache_clear()


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
    stat_path = _prepare_stat_file_path(str(stat_file))
    assert isinstance(stat_path, DPH5Path)
    try:
        assert stat_path.mode == "a"
        (stat_path / "value").save_numpy(np.array([1.0]))
        assert (stat_path / "value").load_numpy().tolist() == [1.0]
    finally:
        _close_stat_path(stat_path)


def test_read_stat_file_mode_reads_existing_cache(tmp_path: Path) -> None:
    stat_file = tmp_path / "stat.hdf5"
    with h5py.File(stat_file, "w") as file:
        file.create_dataset("value", data=[1.0])

    stat_path = _prepare_stat_file_path(str(stat_file), "read")
    assert isinstance(stat_path, DPH5Path)
    try:
        assert stat_path.mode == "r"
        assert stat_path.root.mode == "r"
        assert (stat_path / "value").load_numpy().tolist() == [1.0]
    finally:
        _close_stat_path(stat_path)


def test_read_stat_file_mode_requires_existing_cache(tmp_path: Path) -> None:
    stat_file = tmp_path / "missing.hdf5"
    with pytest.raises(FileNotFoundError, match="does not exist in read mode"):
        _prepare_stat_file_path(str(stat_file), "read")


def test_read_stat_file_mode_requires_cache_path() -> None:
    with pytest.raises(ValueError, match="requires `stat_file`"):
        _prepare_stat_file_path(None, "read")


def test_read_stat_file_mode_rejects_incomplete_statistics_cache(
    tmp_path: Path,
) -> None:
    stat_file = tmp_path / "stat.hdf5"
    with h5py.File(stat_file, "w") as file:
        file.create_dataset("bias_atom_energy", data=np.zeros((1, 1)))

    stat_path = _prepare_stat_file_path(str(stat_file), "read")
    try:
        with pytest.raises(FileNotFoundError, match="std_atom_energy"):
            compute_output_stats(
                lambda: pytest.fail("read-only statistics must not sample data"),
                ntypes=1,
                keys=["energy"],
                stat_file_path=stat_path,
            )
    finally:
        _close_stat_path(stat_path)


def test_read_stat_file_mode_loads_complete_cache_from_two_readers(
    tmp_path: Path,
) -> None:
    stat_file = tmp_path / "stat.hdf5"
    with h5py.File(stat_file, "w") as file:
        file.create_dataset("bias_atom_energy", data=np.zeros((1, 1)))
        file.create_dataset("std_atom_energy", data=np.ones((1, 1)))

    reader_one = _prepare_stat_file_path(str(stat_file), "read")
    DPH5Path._load_h5py.cache_clear()
    reader_two = _prepare_stat_file_path(str(stat_file), "read")
    assert isinstance(reader_one, DPH5Path)
    assert isinstance(reader_two, DPH5Path)
    try:
        for reader in (reader_one, reader_two):
            bias, std = compute_output_stats(
                lambda: pytest.fail("complete read-only cache must not sample data"),
                ntypes=1,
                keys=["energy"],
                stat_file_path=reader,
            )
            assert bias["energy"].shape == (1, 1)
            assert std["energy"].shape == (1, 1)
    finally:
        reader_one.root.close()
        reader_two.root.close()
        DPH5Path._load_h5py.cache_clear()
        DPH5Path._file_keys.cache_clear()


def test_energy_model_reloads_update_cache_in_read_mode(tmp_path: Path) -> None:
    sampled = _energy_stat_sample()
    stat_file = tmp_path / "stat.hdf5"

    update_path = _prepare_stat_file_path(str(stat_file), "update")
    assert isinstance(update_path, DPH5Path)
    try:
        update_model = get_model(_energy_model_params()).to(DEVICE)
        update_model.compute_or_load_stat(lambda: sampled, update_path)
        stat_root = update_path / "O H"
        assert (stat_root / "bias_atom_energy").is_file()
        assert not (stat_root / "bias_atom_mask").is_file()
    finally:
        _close_stat_path(update_path)

    read_path = _prepare_stat_file_path(str(stat_file), "read")
    assert isinstance(read_path, DPH5Path)
    try:
        read_model = get_model(_energy_model_params()).to(DEVICE)
        read_model.compute_or_load_stat(
            lambda: pytest.fail("complete read-only cache must not sample data"),
            read_path,
        )
    finally:
        _close_stat_path(read_path)


def test_read_mode_rejects_missing_descriptor_stats_before_sampling(
    tmp_path: Path,
) -> None:
    stat_file = tmp_path / "stat.hdf5"
    with h5py.File(stat_file, "w") as file:
        file.create_group("O H")

    read_path = _prepare_stat_file_path(str(stat_file), "read")
    assert isinstance(read_path, DPH5Path)
    try:
        model = get_model(_energy_model_params()).to(DEVICE)
        with pytest.raises(FileNotFoundError, match="environment statistics"):
            model.compute_or_load_stat(
                lambda: pytest.fail("read-only cache miss must not sample data"),
                read_path,
            )
    finally:
        _close_stat_path(read_path)

    with h5py.File(stat_file, "r") as file:
        assert list(file.keys()) == ["O H"]
        assert len(file["O H"]) == 0


def test_read_mode_rejects_partial_descriptor_stats_before_sampling(
    tmp_path: Path,
) -> None:
    stat_file = tmp_path / "stat.hdf5"
    update_path = _prepare_stat_file_path(str(stat_file), "update")
    assert isinstance(update_path, DPH5Path)
    try:
        model = get_model(_energy_model_params()).to(DEVICE)
        model.compute_or_load_stat(_energy_stat_sample, update_path)
    finally:
        _close_stat_path(update_path)

    with h5py.File(stat_file, "r+") as file:
        type_map_group = file["O H"]
        descriptor_groups = [
            value for value in type_map_group.values() if isinstance(value, h5py.Group)
        ]
        assert len(descriptor_groups) == 1
        del descriptor_groups[0]["r_0"]

    read_path = _prepare_stat_file_path(str(stat_file), "read")
    assert isinstance(read_path, DPH5Path)
    try:
        model = get_model(_energy_model_params()).to(DEVICE)
        with pytest.raises(FileNotFoundError, match="'r_0'"):
            model.compute_or_load_stat(
                lambda: pytest.fail("partial read-only cache must not sample data"),
                read_path,
            )
    finally:
        _close_stat_path(read_path)


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
