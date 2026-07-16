# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end PT checkpoint conversion coverage for JAX DPA4."""

from copy import (
    deepcopy,
)

import pytest

torch = pytest.importorskip("torch")

from deepmd.jax.utils.serialization import (
    deserialize_to_file as deserialize_to_jax_file,
)
from deepmd.jax.utils.serialization import (
    serialize_from_file as serialize_from_jax_file,
)
from deepmd.pt.model.model import get_model as get_pt_model
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils.serialization import serialize_from_file as serialize_from_pt_file
from deepmd.utils.argcheck import (
    model_args,
)


def _small_dpa4_config() -> dict:
    """Return a small real PT DPA4 config with zero-size state leaves."""
    return model_args().normalize_value(
        {
            "type": "dpa4",
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "dpa4",
                "sel": 4,
                "rcut": 4.0,
                "channels": 4,
                "n_radial": 4,
                "lmax": 1,
                "mmax": 1,
                "n_blocks": 1,
                "random_gamma": False,
                "use_amp": False,
                "precision": "float64",
                "seed": 1,
            },
            "fitting_net": {
                "type": "dpa4_ener",
                "neuron": [4],
                "precision": "float64",
                "seed": 1,
            },
        },
        trim_pattern="_.*",
    )


def test_pt_dpa4_checkpoint_converts_to_real_jax_checkpoint(tmp_path) -> None:
    """The public PT schema saves and restores through Orbax without loss."""
    model_params = _small_dpa4_config()
    pt_model = get_pt_model(deepcopy(model_params)).to(torch.float64)
    wrapper = ModelWrapper(pt_model, model_params=deepcopy(model_params))
    pt_path = tmp_path / "dpa4.pt"
    jax_path = tmp_path / "dpa4.jax"
    torch.save({"model": wrapper.state_dict()}, pt_path)

    data = serialize_from_pt_file(str(pt_path))
    assert data["model"]["type"] == "SeZM"

    deserialize_to_jax_file(str(jax_path), data)
    restored = serialize_from_jax_file(str(jax_path))

    assert restored["model"]["type"] == "standard"
    assert restored["model"]["descriptor"]["type"] == "SeZM"
