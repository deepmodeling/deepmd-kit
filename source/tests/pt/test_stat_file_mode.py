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
from dargs.dargs import (
    ArgumentValueError,
)

from deepmd.pt.entrypoints.main import (
    _prepare_stat_file_path,
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


def test_read_stat_file_mode_is_read_only(tmp_path: Path) -> None:
    stat_file = tmp_path / "stat.hdf5"
    with h5py.File(stat_file, "w") as file:
        file.create_dataset("value", data=[1.0])

    stat_path = _prepare_stat_file_path(str(stat_file), "read")
    assert isinstance(stat_path, DPH5Path)
    try:
        assert stat_path.mode == "r"
        assert stat_path.root.mode == "r"
        assert (stat_path / "value").load_numpy().tolist() == [1.0]
        with pytest.raises(ValueError, match="read-only"):
            (stat_path / "other").save_numpy(np.array([2.0]))
    finally:
        _close_stat_path(stat_path)


def test_read_stat_file_mode_requires_existing_cache(tmp_path: Path) -> None:
    stat_file = tmp_path / "missing.hdf5"
    with pytest.raises(FileNotFoundError, match="does not exist in read mode"):
        _prepare_stat_file_path(str(stat_file), "read")


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
