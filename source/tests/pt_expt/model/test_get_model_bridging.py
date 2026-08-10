# SPDX-License-Identifier: LGPL-3.0-or-later
"""The ``bridging_method`` sugar has exactly ONE owner.

Bridging builds a COMPOSITION (``LinearEnergyModel`` over
``[learned, InnerPotential]``). Since issue #5948 the canonical spelling is
``type: "linear_ener"`` with an ``inner_potential`` sub-model, and the
``bridging_method`` flag is sugar expanded by the shared
``deepmd.utils.bridging.expand_bridging_method`` normalizer at the
``get_model`` entry. The non-composite builders (``get_standard_model``,
``get_sezm_model``) REJECT the flag -- loudly, because silently dropping
the term yields a physically different model.
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


def test_get_model_expands_bridging_without_dpa4_model_type() -> None:
    """Through the dispatcher the flag is sugar: an omitted model type
    defaults to 'standard', and the normalizer expands the flag into the
    canonical composition instead of rejecting it.
    """
    model = get_model(_bridged(_dpa4_standard_config()))
    assert isinstance(model.atomic_model, LinearEnergyAtomicModel)
    assert len(model.atomic_model.models) == 2
    assert model.atomic_model.models[0].descriptor.bridging_switch is not None


def test_sezm_builder_rejects_bridging() -> None:
    """The DPA4/SeZM builder must not hand back a composition either: the
    flag's one owner is the shared normalizer at the get_model entry.
    """
    from deepmd.pt_expt.model.get_model import (
        get_sezm_model,
    )

    data = _bridged(_dpa4_standard_config())
    data["type"] = "dpa4"
    with pytest.raises(ValueError, match="bridging_method"):
        get_sezm_model(data)


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


def _canonical_config() -> dict:
    """The bridged config spelled canonically (issue #5948)."""
    base = _dpa4_standard_config()
    return {
        "type": "linear_ener",
        "weights": "sum",
        "type_map": base["type_map"],
        "models": [
            {
                "type": "dpa4",
                "descriptor": base["descriptor"],
                "fitting_net": base["fitting_net"],
            },
            {
                "type": "inner_potential",
                "mode": "ZBL",
                "r_inner": 0.8,
                "r_outer": 1.2,
            },
        ],
    }


def test_canonical_composition_builds() -> None:
    """The canonical spelling composes [learned, InnerPotential] with the
    clamp radii derived onto the learned child's descriptor.
    """
    model = get_model(_canonical_config())
    assert isinstance(model.atomic_model, LinearEnergyAtomicModel)
    assert len(model.atomic_model.models) == 2
    learned = model.atomic_model.models[0]
    assert learned.descriptor.bridging_switch is not None
    assert float(learned.descriptor.inner_clamp.r_inner) == 0.8


def test_canonical_matches_sugar_serialize() -> None:
    """Both spellings serialize to the same wire dict."""
    import numpy as np

    data = _bridged(_dpa4_standard_config())
    data["type"] = "dpa4"
    d_sugar = get_model(data).serialize()
    d_canon = get_model(_canonical_config()).serialize()

    def _strip_arrays(obj):
        if isinstance(obj, dict):
            return {k: _strip_arrays(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_strip_arrays(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return ("ndarray", obj.shape)
        return obj

    assert _strip_arrays(d_canon) == _strip_arrays(d_sugar)


def test_canonical_native_spin_composition() -> None:
    """A top-level native-spin section wraps the canonical composition."""
    from deepmd.pt_expt.model.native_spin_model import (
        NativeSpinEnergyModel,
    )

    cfg = _canonical_config()
    cfg["spin"] = {"scheme": "native", "use_spin": [True, False]}
    model = get_model(cfg)
    assert isinstance(model, NativeSpinEnergyModel)
    assert isinstance(model.atomic_model, LinearEnergyAtomicModel)
