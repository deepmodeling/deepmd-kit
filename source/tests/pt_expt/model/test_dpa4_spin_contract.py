# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 spin construction contracts for the exportable backend."""

import copy

import pytest

from deepmd.pt_expt.model.get_model import (
    get_model,
    get_standard_model,
)
from deepmd.pt_expt.model.native_spin_model import (
    NativeSpinEnergyModel,
)
from deepmd.pt_expt.model.spin_ener_model import (
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
        "fitting_net": {"type": "dpa4_ener", "neuron": [4]},
    }


@pytest.mark.parametrize("descriptor_type", ["dpa4", "DPA4", "sezm", "SeZM"])
@pytest.mark.parametrize("model_type", [None, "standard"])
@pytest.mark.parametrize("scheme", [None, "invalid"])
def test_standard_spin_requires_native_scheme(
    descriptor_type: str, model_type: str | None, scheme: str | None
) -> None:
    config = _config()
    config["descriptor"]["type"] = descriptor_type
    if model_type is not None:
        config["type"] = model_type
    config["spin"] = {"use_spin": [True, False]}
    if scheme is not None:
        config["spin"]["scheme"] = scheme
    with pytest.raises(ValueError, match=r"require.*native"):
        get_model(config)


def test_standard_native_spin_construction() -> None:
    config = _config()
    config["spin"] = {"use_spin": [True, False], "scheme": "native"}
    original = copy.deepcopy(config)
    model = get_model(config)
    assert isinstance(model, NativeSpinEnergyModel)
    assert model.get_type_map() == ["Ni", "O"]
    assert config == original


@pytest.mark.parametrize("serialized", [False, True])
def test_spin_wrapper_requires_compatible_backbone(serialized: bool) -> None:
    backbone = get_standard_model(_config())
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
