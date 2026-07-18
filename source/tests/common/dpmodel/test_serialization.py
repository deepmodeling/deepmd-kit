# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import h5py
import numpy as np
import pytest

from deepmd.dpmodel.utils.serialization import (
    load_dp_model,
    save_dp_model,
)


@pytest.mark.parametrize(
    "suffix",
    [
        pytest.param(".yaml", id="yaml"),
        pytest.param(".dp", id="native-hdf5"),
    ],
)
def test_save_dp_model_preserves_nested_input(tmp_path: Path, suffix: str) -> None:
    """Saving must not modify caller-owned containers or variables."""
    weights = np.arange(6, dtype=np.float64).reshape(2, 3)
    model_dict = {
        "model": {
            "layers": [
                {
                    "@variables": {
                        "weights": weights,
                        "bias": np.array([0.25, -0.5], dtype=np.float64),
                    }
                }
            ],
            "metadata": {"labels": ["energy", "force"]},
        }
    }
    expected = deepcopy(model_dict)

    save_dp_model(str(tmp_path / f"model{suffix}"), model_dict)

    np.testing.assert_equal(model_dict, expected)
    assert model_dict["model"]["layers"][0]["@variables"]["weights"] is weights


def test_save_dp_model_accepts_hdf5_dataset_without_mutation(tmp_path: Path) -> None:
    """Native saving must preserve non-copyable HDF5 variable objects."""
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    with h5py.File(tmp_path / "variables.h5", "w") as variable_file:
        weights = variable_file.create_dataset("weights", data=values)
        variables = {"weights": weights}
        layer = {"@variables": variables}
        layers = [layer]
        model_dict = {"model": {"layers": layers}}

        output = tmp_path / "model.dp"
        save_dp_model(str(output), model_dict)

        assert model_dict["model"]["layers"] is layers
        assert layers[0] is layer
        assert layer["@variables"] is variables
        assert variables["weights"] is weights
        np.testing.assert_equal(weights[()], values)

    loaded_model = load_dp_model(str(output))
    np.testing.assert_equal(
        loaded_model["model"]["layers"][0]["@variables"]["weights"], values
    )
