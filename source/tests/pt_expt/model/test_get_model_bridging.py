# SPDX-License-Identifier: LGPL-3.0-or-later
"""Analytical bridging has exactly ONE owner per backend.

Bridging builds a COMPOSITION (``LinearEnergyModel`` over
``[learned, InnerPotential]``), so it is not expressible on a non-composite
model type: ``type: "standard"`` would have to return a model of a
different kind than the one requested. pt_expt therefore owns bridging on
the DPA4/SeZM route only and REJECTS it in the standard builder -- loudly,
because silently dropping the term yields a physically different model.

Two builders accepting the flag is exactly how the routes drifted:
``get_sezm_model`` promotes ``descriptor.exclude_types`` to model-level
``pair_exclude_types`` and the standard route never did, which changes a
0.9 A Ni-O dimer by ~80 eV (issue #5947). Issue #5948 replaces the flag
with an explicit ``linear_ener`` composition, after which this restriction
becomes moot.
"""

import copy

import pytest

from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
    get_standard_model,
)


def _dpa4_standard_config() -> dict:
    """The ZBL config of test_zbl_bridging.py minus the 'dpa4' model type."""
    return {
        "type_map": ["Ni", "O"],
        "descriptor": {
            "type": "dpa4",
            "rcut": 4.0,
            "sel": 8,
            "channels": 16,
            "n_radial": 8,
            "lmax": 2,
            "mmax": 1,
            "n_blocks": 2,
            "precision": "float64",
            "seed": 7,
            "random_gamma": False,
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [8, 8],
            "precision": "float64",
            "seed": 7,
        },
    }


def _bridged(data: dict) -> dict:
    data["bridging_method"] = "ZBL"
    data["bridging_r_inner"] = 0.8
    data["bridging_r_outer"] = 1.2
    return data


def test_standard_builder_rejects_bridging() -> None:
    """The standard builder must not hand back a composition."""
    with pytest.raises(ValueError, match="bridging_method"):
        get_standard_model(_bridged(_dpa4_standard_config()))


def test_get_model_rejects_bridging_without_dpa4_model_type() -> None:
    """Same contract through the dispatcher: an omitted model type defaults
    to the standard route, so it must reject rather than compose.
    """
    with pytest.raises(ValueError, match="bridging_method"):
        get_model(_bridged(_dpa4_standard_config()))


def test_standard_builder_without_bridging_is_unaffected() -> None:
    """The rejection keys on the flag, not on the DPA4 components: a plain
    DPA4 standard model still builds and carries no bridging switch.
    """
    model = get_standard_model(_dpa4_standard_config())
    assert not isinstance(model.atomic_model, LinearEnergyAtomicModel)
    assert model.atomic_model.descriptor.bridging_switch is None


@pytest.mark.parametrize(
    "model_type",
    [
        "dpa4",  # canonical spelling
        "sezm",  # pt-compatible alias
    ],
)
def test_dpa4_model_type_owns_the_composition(model_type: str) -> None:
    """The one supported spelling composes [learned, InnerPotential] and
    injects the radii into the learned child's descriptor.
    """
    data = _bridged(_dpa4_standard_config())
    data["type"] = model_type
    model = get_model(copy.deepcopy(data))
    assert isinstance(model.atomic_model, LinearEnergyAtomicModel)
    assert len(model.atomic_model.models) == 2
    # a composition without the inner-clamp radii would be half-applied
    assert model.atomic_model.models[0].descriptor.bridging_switch is not None


def test_pt_checkpoint_eval_works_for_composition(tmp_path) -> None:
    """``DeepEval`` on a ``.pt`` checkpoint of a bridging composition.

    ``LinearEnergyAtomicModel`` has no single ``.descriptor``;
    ``_load_pt``'s metadata block previously crashed with
    ``AttributeError`` on ``model.get_descriptor()`` (issue #5906 Task 4
    audit, gap 1). ``ntypes`` now comes from the model API.
    """
    import numpy as np
    import torch

    from deepmd.infer import (
        DeepPot,
    )
    from deepmd.pt_expt.train.wrapper import (
        ModelWrapper,
    )

    config = _bridged(_dpa4_standard_config())
    config["type"] = "dpa4"
    model = get_model(copy.deepcopy(config)).to(torch.float64).eval()
    ckpt = str(tmp_path / "dpa4_zbl.pt")
    wrapper = ModelWrapper(model, model_params=copy.deepcopy(config))
    torch.save({"model": wrapper.state_dict()}, ckpt)

    dp = DeepPot(ckpt)
    assert dp.get_ntypes() == 2
    rng = np.random.default_rng(5)
    coord = rng.uniform(1.5, 5.5, size=(1, 6, 3))
    coord[0, 1] = coord[0, 0] + np.array([0.9, 0.0, 0.0])
    atype = np.array([[0, 0, 1, 0, 1, 1]], dtype=np.int64)
    box = 8.0 * np.eye(3, dtype=np.float64).reshape(1, 9)
    e, f, v = dp.eval(coord.reshape(1, -1), box, atype[0].tolist())
    assert np.all(np.isfinite(e))
    assert np.all(np.isfinite(f))


def test_compile_attention_probe_tolerates_composition() -> None:
    """``enable_compile``'s DPA1-attention warning probe must degrade
    gracefully for compositions instead of crashing on
    ``model.get_descriptor()`` (issue #5906 Task 4 audit, gap 1 twin).
    """
    from deepmd.pt_expt.train.training import (
        _warn_compiled_attention,
    )

    data = _bridged(_dpa4_standard_config())
    data["type"] = "dpa4"
    model = get_model(data)
    assert isinstance(model.atomic_model, LinearEnergyAtomicModel)
    # must not raise
    _warn_compiled_attention(model, "Default")
