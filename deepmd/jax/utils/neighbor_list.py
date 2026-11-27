# SPDX-License-Identifier: LGPL-3.0-or-later
"""Complete neighbor lists for JAX models whose selection sets normalization."""

from typing import (
    TYPE_CHECKING,
)

from deepmd.dpmodel.utils.neighbor_list import (
    NeighborList,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.jax.env import (
    jax,
    jnp,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )


class FullNeighborList(NeighborList):
    """Retain every cutoff neighbor in a bounded dense-interface capacity.

    Eager construction chooses the smallest power-of-two row capacity that
    contains all neighbors. A traced, nonperiodic construction uses the total
    atom count as a static upper bound. Model evaluation receives fixed shapes;
    the descriptor's ``sel`` remains a normalization constant.
    """

    def build(
        self,
        coord: jnp.ndarray,
        atype: jnp.ndarray,
        box: jnp.ndarray | None,
        rcut: float,
        sel: list[int],
        return_mode: str = "extended",
        pair_excl: "PairExcludeMask | None" = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Build the extended quartet without imposing a neighbor count cutoff."""
        del sel
        if return_mode != "extended":
            raise NotImplementedError(
                "FullNeighborList supports only extended outputs."
            )
        nframes, nloc = atype.shape
        coord = coord.reshape(nframes, nloc, 3)
        if box is not None:
            coord = normalize_coord(coord, box.reshape(nframes, 3, 3))
        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            coord, atype, box, rcut
        )
        nall = extended_atype.shape[1]
        nlist = build_neighbor_list(
            extended_coord,
            extended_atype,
            nloc,
            rcut,
            [nall],
            distinguish_types=False,
            pair_excl=pair_excl,
        )
        if not isinstance(coord, jax.core.Tracer):
            if pair_excl is not None and pair_excl.exclude_types:
                # Exclusions leave holes in distance-sorted rows. A capacity
                # derived from counts is valid only after compacting the rows.
                order = jnp.argsort(nlist < 0, axis=-1, stable=True)
                nlist = jnp.take_along_axis(nlist, order, axis=-1)
            required = max(1, int(jnp.max(jnp.sum(nlist >= 0, axis=-1))))
            capacity = 1 << (required - 1).bit_length()
            nlist = nlist[:, :, :capacity]
            nlist = jnp.pad(
                nlist,
                ((0, 0), (0, 0), (0, max(0, capacity - nlist.shape[2]))),
                constant_values=-1,
            )
        return extended_coord.reshape(nframes, nall, 3), extended_atype, nlist, mapping
