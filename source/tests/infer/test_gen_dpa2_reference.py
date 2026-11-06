# SPDX-License-Identifier: LGPL-3.0-or-later
"""Keep the independent DPA2 oracle outputs while avoiding unused export work."""

import importlib
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)

import numpy as np
import pytest


@pytest.fixture
def generator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return importlib.import_module(".gen_dpa2", __package__)


def test_dense_oracle_exports_only_the_required_capabilities(generator, monkeypatch):
    pytest.importorskip("torch")
    from deepmd.pt_expt.utils import (
        serialization,
    )

    calls = []

    def export(path, data, **kwargs):
        calls.append((path, data, kwargs))
        data["model"]["weights"].append(2)

    monkeypatch.setattr(serialization, "deserialize_to_file", export)
    data = {"model": {"weights": [1]}}
    generator._export_nlist_reference("reference.pt2", data)
    assert len(calls) == 1
    assert calls[0][0] == "reference.pt2"
    assert calls[0][2] == {
        "do_atomic_virial": False,
        "lower_kind": "nlist",
        "with_comm": False,
    }
    assert data == {"model": {"weights": [1]}}


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("flat", [False, True])
def test_dense_oracle_retains_atomic_energy_and_total_virial(generator, periodic, flat):
    output = {
        "energy_redu": np.arange(2),
        "energy_derv_r": np.arange(2 * 6 * 3),
        "energy_derv_c_redu": np.arange(2 * 9),
        "energy": np.arange(2 * 6),
    }
    coords = np.zeros((2, 6, 3))
    box = np.tile(np.eye(3), (2, 1, 1)) if periodic else None
    types = [0, 1, 1, 0, 1, 1]

    def evaluate(actual_coords, actual_box, actual_types, atomic):
        np.testing.assert_array_equal(actual_coords, coords)
        if box is None:
            assert actual_box is None
        else:
            np.testing.assert_array_equal(actual_box, box.reshape(2, 9))
        assert actual_types is types
        assert atomic is False
        return output

    dp = SimpleNamespace(deep_eval=SimpleNamespace(eval=evaluate))
    energy, force, virial, atomic_energy = generator._eval_nlist_reference(
        dp,
        coords.ravel() if flat else coords,
        box.ravel() if flat and box is not None else box,
        types,
    )
    for value, key, shape in (
        (energy, "energy_redu", (2, 1)),
        (force, "energy_derv_r", (2, 6, 3)),
        (virial, "energy_derv_c_redu", (2, 9)),
        (atomic_energy, "energy", (2, 6, 1)),
    ):
        assert value.shape == shape
        np.testing.assert_array_equal(value.ravel(), output[key])
