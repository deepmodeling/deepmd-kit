# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 spin construction contracts for the PyTorch backend."""

import copy

import pytest

from deepmd.pt.model.model import (
    get_model,
    get_standard_model,
)
from deepmd.pt.model.model.dp_linear_model import (
    LinearEnergyModel,
)
from deepmd.pt.model.model.sezm_native_spin_model import (
    SeZMNativeSpinModel,
)
from deepmd.pt.model.model.spin_model import (
    SpinEnergyModel,
)
from deepmd.utils.spin import (
    Spin,
)


def _config() -> dict:
    return {
        "type_map": ["Ni", "O"],
        "descriptor": {
            "type": "dpa4",
            "rcut": 4.0,
            "sel": 8,
            "channels": 4,
            "n_radial": 3,
            "lmax": 1,
            "mmax": 1,
            "n_blocks": 1,
            "precision": "float64",
            "seed": 7,
            "random_gamma": False,
        },
        "fitting_net": {"type": "ener", "neuron": [4]},
    }


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("descriptor_type", ["dpa4", "DPA4", "sezm", "SeZM"])
@pytest.mark.parametrize("model_type", [None, "standard"])
@pytest.mark.parametrize("scheme", [None, "invalid"])
def test_standard_spin_requires_compatible_backbone(
    descriptor_type: str, model_type: str | None, scheme: str | None, hybrid: bool
) -> None:
    config = _config()
    config["descriptor"]["type"] = descriptor_type
    if hybrid:
        config["descriptor"] = {"type": "hybrid", "list": [config["descriptor"]]}
    if model_type is not None:
        config["type"] = model_type
    config["spin"] = {"use_spin": [True, False], "virtual_scale": 0.3}
    if scheme is not None:
        config["spin"]["scheme"] = scheme
    with pytest.raises(ValueError, match=r"require.*native"):
        get_model(config)


def test_native_spin_construction() -> None:
    config = _config()
    config["spin"] = {"use_spin": [True, False], "scheme": "native"}
    config["type"] = "dpa4"
    config["fitting_net"]["type"] = "dpa4_ener"
    original = copy.deepcopy(config)
    model = get_model(config)
    assert isinstance(model, SeZMNativeSpinModel)
    assert model.get_type_map() == ["Ni", "O"]
    assert config == original


@pytest.mark.parametrize(
    "serialized, depth", [(False, 0), (False, 1), (False, 2), (True, 0)]
)
def test_spin_wrapper_requires_compatible_backbone(
    serialized: bool, depth: int
) -> None:
    backbone = get_standard_model(_config())
    for _ in range(depth):
        backbone = LinearEnergyModel(
            models=[backbone.atomic_model], type_map=backbone.get_type_map()
        )
    spin = Spin(use_spin=[True, False], virtual_scale=0.3)
    with pytest.raises(ValueError, match="require a native spin model"):
        if serialized:
            SpinEnergyModel.deserialize(
                {
                    "type": "spin_ener",
                    "backbone_model": backbone.serialize(),
                    "spin": spin.serialize(),
                }
            )
        else:
            SpinEnergyModel(backbone_model=backbone, spin=spin)


@pytest.mark.parametrize("serialized", [False, True])
def test_spin_wrapper_checks_hybrid_descriptors(serialized: bool) -> None:
    config = _config()
    config["descriptor"] = {"type": "hybrid", "list": [config["descriptor"]]}
    backbone = get_standard_model(config)
    spin = Spin(use_spin=[True, False], virtual_scale=0.3)
    with pytest.raises(ValueError, match="require a native spin model"):
        if serialized:
            SpinEnergyModel.deserialize(
                {
                    "type": "spin_ener",
                    "backbone_model": backbone.serialize(),
                    "spin": spin.serialize(),
                }
            )
        else:
            SpinEnergyModel(backbone_model=backbone, spin=spin)
