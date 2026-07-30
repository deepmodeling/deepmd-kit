# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt ``get_standard_model`` must honor ``bridging_method`` like its
dpmodel twin (``deepmd/dpmodel/model/model.py``) -- issue #5906 Task 4
variant-alignment audit, gap 2: a ``type: "standard"`` config with bridging
silently dropped the InterPotential composition in pt_expt.
"""

import copy

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


def test_standard_model_type_builds_bridging_composition() -> None:
    """pt_expt twin of dpmodel model.py's get_standard_model: a config with
    bridging_method must compose [learned, InterPotential], not silently
    drop the bridging term.
    """
    data = _dpa4_standard_config()
    data["bridging_method"] = "ZBL"
    data["bridging_r_inner"] = 0.8
    data["bridging_r_outer"] = 1.2
    model = get_standard_model(copy.deepcopy(data))
    assert isinstance(model.atomic_model, LinearEnergyAtomicModel)
    assert len(model.atomic_model.models) == 2
    # The descriptor radii injection must ride the same seam (a composition
    # without the inner-clamp radii would be a half-applied bridging config):
    desc = model.atomic_model.models[0].descriptor
    assert desc.bridging_switch is not None
    # And the get_model router (type omitted -> "standard") reaches the same
    # composition:
    routed = get_model(copy.deepcopy(data))
    assert isinstance(routed.atomic_model, LinearEnergyAtomicModel)


def test_standard_model_type_maps_dpa4_fitting() -> None:
    """Dpmodel maps dpa4_ener/sezm_ener fitting under type:'standard' via the
    model registry; pt_expt must not raise where dpmodel builds.
    """
    model = get_standard_model(_dpa4_standard_config())
    assert model is not None
    assert model.atomic_model.descriptor.bridging_switch is None


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

    config = _dpa4_standard_config()
    config["type"] = "dpa4"
    config["bridging_method"] = "ZBL"
    config["bridging_r_inner"] = 0.8
    config["bridging_r_outer"] = 1.2
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

    data = _dpa4_standard_config()
    data["bridging_method"] = "ZBL"
    model = get_standard_model(data)
    # must not raise
    _warn_compiled_attention(model, "Default")
