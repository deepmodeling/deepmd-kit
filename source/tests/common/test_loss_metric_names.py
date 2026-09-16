# SPDX-License-Identifier: LGPL-3.0-or-later
"""Declared display columns match real loss outputs without sampling a model."""

import importlib
from typing import (
    Any,
)

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _data() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(812)
    shapes = {
        "energy": (2, 1),
        "force": (2, 3, 3),
        "force_mag": (2, 3, 3),
        "virial": (2, 9),
        "atom_energy": (2, 3, 1),
        "hessian": (2, 9, 9),
        "dos": (2, 4),
        "atom_dos": (2, 3, 4),
        "dipole": (2, 3, 3),
        "global_dipole": (2, 3),
        "property": (2, 2),
        "population": (2, 3, 2),
    }
    prediction = {key: rng.random(shape) for key, shape in shapes.items()}
    labels = {key: rng.random(shape) for key, shape in shapes.items()}
    labels["atom_ener"] = labels.pop("atom_energy")
    labels["atom_dipole"] = labels.pop("dipole")
    labels["dipole"] = labels.pop("global_dipole")
    labels["atom_population"] = labels.pop("population")
    labels["atom_pref"] = np.ones((2, 3, 3))
    labels["drdq"] = np.ones((2, 9, 1))
    prediction["mask_mag"] = np.ones((2, 3, 1), dtype=bool)
    return prediction, labels


def _check_metrics(backend: str, kind: str, options: dict, found: float) -> None:
    class_names = {
        "ener": "EnergyStdLoss" if backend == "pt" else "EnergyLoss",
        "ener_spin": "EnergySpinLoss",
        "dos": "DOSLoss",
        "tensor": "TensorLoss",
        "property": "PropertyLoss",
        "population": "PopulationLoss",
    }
    loss_type = getattr(
        importlib.import_module(f"deepmd.{backend}.loss.{kind}"), class_names[kind]
    )
    loss = loss_type(starter_learning_rate=1.0, **options)
    names = loss.training_metric_names
    prediction, labels = _data()
    labels.update({f"find_{name}": found for name in list(labels)})
    if backend == "pt":
        from deepmd.pt.utils.env import (
            DEVICE,
        )

        prediction = {
            key: torch.as_tensor(value, device=DEVICE)
            for key, value in prediction.items()
        }
        labels = {
            key: torch.as_tensor(value, device=DEVICE) for key, value in labels.items()
        }

        class Model(torch.nn.Module):
            def forward(self) -> dict[str, Any]:
                return prediction

        _, _, metrics = loss({}, Model(), labels, natoms=3, learning_rate=1.0)
    else:
        _, metrics = loss(1.0, 3, prediction, labels)
    assert set(names) == {key for key in metrics if "l2_" not in key}
    assert loss.training_metric_names == names


@pytest.mark.parametrize("backend", ["pt", "dpmodel"])
@pytest.mark.parametrize(
    "kind,loss_func,all_terms,found",
    [
        ("ener", "mse", True, 1.0),
        ("ener", "mae", True, 1.0),
        ("ener", "mse", False, 0.0),
        ("ener_spin", "mse", True, 1.0),
        ("ener_spin", "mae", True, 1.0),
    ],
)
def test_energy_metric_names(
    backend: str, found: float, loss_func: str, kind: str, all_terms: bool
) -> None:
    terms = ("e", "f", "v", "ae", "pf", "gf", "h")
    if kind == "ener_spin":
        terms = ("e", "fr", "fm", "v", "ae")
    options: dict[str, Any] = {"loss_func": loss_func}
    for term in terms:
        for endpoint in ("start", "limit"):
            options[f"{endpoint}_pref_{term}"] = float(all_terms or term == "e")
    if kind == "ener":
        options["numb_generalized_coord"] = 1
    _check_metrics(backend, kind, options, found)


@pytest.mark.parametrize("backend", ["pt", "dpmodel"])
@pytest.mark.parametrize(
    "kind,local,global_",
    [
        ("dos", True, True),
        ("dos", False, True),
        ("tensor", True, True),
        ("tensor", True, False),
    ],
)
def test_tensor_metric_names(
    backend: str, local: bool, global_: bool, kind: str
) -> None:
    if kind == "tensor":
        options = {
            "tensor_name": "dipole",
            "label_name": "dipole",
            "tensor_size": 3,
            "pref_atomic": float(local),
            "pref": float(global_),
        }
    else:
        options = {"numb_dos": 4}
        for term in ("dos", "cdf", "ados", "acdf"):
            for endpoint in ("start", "limit"):
                options[f"{endpoint}_pref_{term}"] = float(
                    local if term.startswith("a") else global_
                )
    _check_metrics(backend, kind, options, 1.0)


@pytest.mark.parametrize("backend", ["pt", "dpmodel"])
@pytest.mark.parametrize(
    "metric", [["mae"], ["smooth_mae", "mae", "mse", "rmse", "mape"]]
)
def test_property_metric_names(backend: str, metric: list[str]) -> None:
    _check_metrics(
        backend,
        "property",
        {
            "task_dim": 2,
            "var_name": "property",
            "metric": metric,
            "out_bias": [0.0, 0.0],
            "out_std": [1.0, 1.0],
        },
        1.0,
    )


def test_population_metric_names() -> None:
    _check_metrics("pt", "population", {"metric": ["rmse"]}, 1.0)
