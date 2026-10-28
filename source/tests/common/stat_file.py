# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared fixtures for statistics-cache round-trip tests."""

from collections.abc import (
    Callable,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import h5py
import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.utils.stat_file import (
    StatFileSpec,
    open_stat_file,
)


def energy_model_params() -> dict[str, Any]:
    """Return a minimal energy-model configuration.

    Returns
    -------
    dict[str, Any]
        Model parameters shared by backend round-trip tests.
    """
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
            "numb_fparam": 1,
            "numb_aparam": 1,
            "precision": "float64",
        },
    }


def energy_stat_sample() -> list[dict[str, Any]]:
    """Return NumPy statistics input for the minimal energy model.

    Returns
    -------
    list[dict[str, Any]]
        One sampled system containing energy labels for two atom types.
    """
    return [
        {
            "coord": np.array(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                ],
                dtype=np.float64,
            ),
            "atype": np.array([[0, 0], [1, 1]], dtype=np.int64),
            "box": None,
            "natoms": np.array(
                [[2, 2, 2, 0], [2, 2, 0, 2]],
                dtype=np.int64,
            ),
            "energy": np.array([[2.0], [4.0]], dtype=np.float64),
            "find_energy": np.float32(1.0),
            "fparam": np.array([[1.0], [3.0]], dtype=np.float64),
            "find_fparam": np.float32(1.0),
            "aparam": np.array(
                [
                    [[1.0], [2.0]],
                    [[4.0], [7.0]],
                ],
                dtype=np.float64,
            ),
            "find_aparam": np.float32(1.0),
        }
    ]


def _model_stat_values(model: Any) -> dict[str, np.ndarray]:
    """Collect numerical statistics applied to a backend model."""
    descriptor_mean, descriptor_stddev = (
        model.get_descriptor().get_stat_mean_and_stddev()
    )
    fitting = model.get_fitting_net()
    values = {
        "descriptor_mean": descriptor_mean,
        "descriptor_stddev": descriptor_stddev,
        "fparam_avg": fitting.fparam_avg,
        "fparam_inv_std": fitting.fparam_inv_std,
        "aparam_avg": fitting.aparam_avg,
        "aparam_inv_std": fitting.aparam_inv_std,
        "out_bias": model.atomic_model.out_bias,
        "out_std": model.atomic_model.out_std,
    }
    result = {}
    for name, value in values.items():
        array = to_numpy_array(value)
        if array is None:
            raise AssertionError(f"Model statistic {name!r} was not initialized.")
        result[name] = np.array(array, copy=True)
    return result


def assert_energy_stat_cache_round_trip(
    model_factory: Callable[[], Any],
    stat_file: Path,
    *,
    sample_factory: Callable[[], list[dict[str, Any]]] = energy_stat_sample,
) -> None:
    """Verify that an energy model reads the cache it writes without sampling.

    Parameters
    ----------
    model_factory
        Factory returning a new backend model for each cache pass.
    stat_file
        HDF5 statistics-cache path.
    sample_factory
        Factory returning statistics input in the backend's array format.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the generated cache has an invalid key set or read mode samples data.
    """
    update_model = model_factory()
    with open_stat_file(StatFileSpec(str(stat_file))) as stat_path:
        assert stat_path is not None
        update_model.compute_or_load_stat(sample_factory, stat_path)
    expected_values = _model_stat_values(update_model)

    with h5py.File(stat_file, "r") as file:
        assert "tasks" not in file
        type_map_cache = file["O H"]
        assert "bias_atom_energy" in type_map_cache
        assert "std_atom_energy" in type_map_cache
        assert "bias_atom_mask" not in type_map_cache
        assert "std_atom_mask" not in type_map_cache
    original = stat_file.read_bytes()

    def unexpected_sample() -> list[dict[str, Any]]:
        raise AssertionError("A complete read-only cache must not sample data.")

    for mode in ("update", "read"):
        read_model = model_factory()
        with open_stat_file(StatFileSpec(str(stat_file), mode)) as stat_path:
            assert stat_path is not None
            read_model.compute_or_load_stat(unexpected_sample, stat_path)
        actual_values = _model_stat_values(read_model)
        assert actual_values.keys() == expected_values.keys()
        for name, expected in expected_values.items():
            np.testing.assert_allclose(actual_values[name], expected)
        assert stat_file.read_bytes() == original
