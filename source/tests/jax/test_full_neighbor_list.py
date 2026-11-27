# SPDX-License-Identifier: LGPL-3.0-or-later
"""Full-neighbor capacity selection must preserve excluded-list holes."""

from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
)
from deepmd.infer import (
    DeepPot,
)
from deepmd.jax.env import (
    jnp,
)
from deepmd.jax.utils.neighbor_list import (
    FullNeighborList,
)
from deepmd.jax.utils.serialization import (
    deserialize_to_file,
)

from .test_dpa4_model import (
    _reference,
    model_config,
)


def _geometry() -> tuple[np.ndarray, np.ndarray]:
    coord = np.zeros((1, 5, 3))
    coord[0, :, 0] = [0.0, 0.3, 0.4, 0.5, 0.9]
    return coord, np.array([[0, 1, 1, 1, 0]], dtype=np.int32)


def test_capacity_compacts_exclusion_holes() -> None:
    coord, atype = [jnp.asarray(value) for value in _geometry()]
    exclusion = PairExcludeMask(2, [(0, 1)])
    expected = build_neighbor_list(
        coord, atype, 5, 2.0, [5], distinguish_types=False, pair_excl=exclusion
    )
    actual = FullNeighborList().build(
        coord, atype, None, 2.0, [1], pair_excl=exclusion
    )[2]
    assert actual.shape[-1] == 2
    for actual_row, expected_row in zip(
        np.asarray(actual[0]), np.asarray(expected[0]), strict=True
    ):
        np.testing.assert_array_equal(
            actual_row[actual_row >= 0], expected_row[expected_row >= 0]
        )


def test_hlo_preserves_farther_unexcluded_neighbors(tmp_path: Path) -> None:
    config = model_config()
    config["pair_exclude_types"] = [[0, 1]]
    reference = _reference(config)
    path = tmp_path / "excluded.hlo"
    deserialize_to_file(
        str(path), {"model": reference.serialize(), "model_def_script": config}
    )
    coord, atype = _geometry()
    expected = reference(torch.tensor(coord), torch.tensor(atype))
    actual = DeepPot(str(path), auto_batch_size=False).eval(
        coord, None, atype, mixed_type=True
    )
    for value, key in zip(actual, ("energy", "force", "virial"), strict=True):
        np.testing.assert_allclose(
            value, expected[key].detach().numpy(), rtol=2e-9, atol=2e-10
        )
