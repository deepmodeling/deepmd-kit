# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from dataclasses import (
    dataclass,
)
from importlib.util import (
    find_spec,
)

import numpy as np

from deepmd.jax.env import (
    jax,
    jnp,
)
from deepmd.jax.jax_md import (
    as_jax_md,
    energy_fn,
    force_fn,
    load_model,
)


class HarmonicModel:
    def get_type_map(self):
        return ["O", "H"]

    def get_rcut(self):
        return 6.0

    def get_dim_fparam(self):
        return 0

    def get_dim_aparam(self):
        return 0

    def __call__(
        self,
        coord,
        atype,
        box=None,
        fparam=None,
        aparam=None,
        charge_spin=None,
    ):
        del atype, box, fparam, aparam, charge_spin
        return {"energy": jnp.sum(coord**2, axis=(1, 2))[:, None]}


class EdgeModel(HarmonicModel):
    def call_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping,
        fparam=None,
        aparam=None,
        charge_spin=None,
    ):
        del extended_atype, mapping, fparam, aparam, charge_spin
        valid = nlist >= 0
        safe_nlist = jnp.where(valid, nlist, 0)
        neighbor_coord = jax.vmap(lambda coord, idx: coord[idx])(
            extended_coord, safe_nlist
        )
        nloc = nlist.shape[1]
        center_coord = extended_coord[:, :nloc, None, :]
        edge_vec = jnp.where(valid[..., None], neighbor_coord - center_coord, 0.0)
        return {"energy": 0.5 * jnp.sum(edge_vec**2, axis=(1, 2, 3))[:, None]}


@dataclass
class DenseNeighbor:
    idx: jax.Array


def test_energy_and_force_fn():
    potential = energy_fn(HarmonicModel(), ["O", "H"])
    coord = jnp.asarray(
        [
            [1.0, 2.0, 3.0],
            [0.5, 0.0, -1.0],
        ]
    )

    np.testing.assert_allclose(potential(coord), 15.25)
    np.testing.assert_allclose(force_fn(potential)(coord), -2.0 * coord)
    np.testing.assert_allclose(jax.jit(potential)(coord), 15.25)


def test_hlo_model_raises_not_implemented():
    with np.testing.assert_raises(NotImplementedError):
        load_model("model.hlo")


def test_dense_neighbor_uses_jax_md_displacement_convention():
    potential = energy_fn(
        EdgeModel(),
        [0, 1],
        displacement_fn=lambda ra, rb: (ra - rb) - 10.0 * jnp.round((ra - rb) / 10.0),
    )
    coord = jnp.asarray(
        [
            [0.1, 0.0, 0.0],
            [9.9, 0.0, 0.0],
        ]
    )
    neighbor = DenseNeighbor(jnp.asarray([[1], [0]], dtype=jnp.int32))

    np.testing.assert_allclose(potential(coord, neighbor=neighbor), 0.04, atol=1e-12)


@unittest.skipIf(find_spec("jax_md") is None, "jax-md is not installed")
def test_actual_jax_md_neighbor_list():
    from jax_md import (
        space,
    )

    displacement, _ = space.periodic(10.0)
    neighbor_fn, potential = as_jax_md(
        EdgeModel(),
        displacement,
        10.0,
        [0, 1],
        dr_threshold=0.1,
    )
    coord = jnp.asarray(
        [
            [0.1, 0.0, 0.0],
            [9.9, 0.0, 0.0],
        ]
    )
    neighbor = neighbor_fn.allocate(coord)

    np.testing.assert_array_equal(np.asarray(neighbor.idx), [[1], [0]])
    np.testing.assert_allclose(potential(coord, neighbor=neighbor), 0.04, atol=1e-12)
    np.testing.assert_allclose(
        force_fn(potential)(coord, neighbor=neighbor),
        [[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]],
        atol=1e-12,
    )
