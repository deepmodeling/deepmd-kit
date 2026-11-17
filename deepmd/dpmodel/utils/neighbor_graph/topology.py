# SPDX-License-Identifier: LGPL-3.0-or-later
"""Topologies an atomic model declares beside its neighbor graph.

An interaction that the short-range neighbor graph cannot describe reads a
topology of its own, declared by name (see
:meth:`~deepmd.dpmodel.atomic_model.base_atomic_model.BaseAtomicModel.required_topologies`):

- :data:`~.pair_graph.ALL_PAIRS`, the pairs of the non-periodic frames;
- :data:`~.reciprocal_graph.K_GRID` and :data:`~.reciprocal_graph.PME_MESH`,
  the reciprocal space of the periodic frames.

Every topology is a dataclass of arrays that declares which of its fields
carry geometry (``GEOMETRY``), which an autograd backend makes leaves of the
energy (``LEAVES``), and the dynamic axes of every field (``AXES``), so that
casting, re-leafing and the inputs of a compiled or exported graph lower are
derived from the declaration alone.
"""

from __future__ import (
    annotations,
)

import dataclasses
from typing import (
    TYPE_CHECKING,
    Any,
)

import array_api_compat

from .pair_graph import (
    ALL_PAIRS,
    PairGraph,
    build_pair_graph,
    check_complete_frames,
    empty_pair_graph,
)
from .reciprocal_graph import (
    K_GRID,
    PME_MESH,
    KGrid,
    MeshGrid,
    build_fixed_k_grid,
    build_k_grid,
    build_mesh_grid,
    check_periodic_cells,
    empty_k_grid,
    empty_mesh_grid,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Sequence,
    )

    from deepmd.dpmodel.array_api import (
        Array,
    )
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )

    Topology = PairGraph | KGrid | MeshGrid

TOPOLOGY_TYPES: dict[str, type] = {
    ALL_PAIRS: PairGraph,
    K_GRID: KGrid,
    PME_MESH: MeshGrid,
}
"""The dataclass of every topology name."""


def _build_reciprocal(
    name: str,
    box: Array,
    coord: Array,
    n_node: Array,
    params: dict[str, Any],
) -> KGrid | MeshGrid:
    """Build a reciprocal-space topology of periodic frames from its parameters."""
    required = {K_GRID: "k_max", PME_MESH: "spacing"}[name]
    if required not in params:
        raise ValueError(
            f"the {name!r} topology requires its `{required}` build parameter"
        )
    check_periodic_cells(box)
    if name == PME_MESH:
        return build_mesh_grid(box, params["spacing"], coord)
    if "bounds" in params:
        return build_fixed_k_grid(box, params["bounds"], params["k_max"], coord, n_node)
    return build_k_grid(box, params["k_max"], coord, n_node)


def build_topologies(
    names: Iterable[str],
    coord: Array,
    n_node: Array,
    box: Array | None = None,
    node_mask: Array | None = None,
    *,
    atype: Array,
    pair_excl: PairExcludeMask | None = None,
    topology_params: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Topology]:
    """Build the topologies an atomic model declares, beside its neighbor graph.

    A batch is periodic when it carries cells. A declaration of the pairs
    alone serves non-periodic batches only. Beside a reciprocal-space
    topology, the pairs of a periodic batch come back guard-only and the
    reciprocal-space topology of a non-periodic batch masked, so that one
    model serves both.

    Parameters
    ----------
    names
        The declared topology names.
    coord
        Node coordinates with shape ``(N, 3)``, frame-major over ``n_node``.
    n_node
        Per-frame node counts with shape ``(nf,)``.
    box
        Simulation cells with shape ``(nf, 9)`` or ``(nf, 3, 3)``, or
        ``None`` for non-periodic systems.
    node_mask
        Validity of the nodes with shape ``(N,)``, or ``None`` when every
        node is valid.
    atype
        Node types with shape ``(N,)``; invalid nodes carry any type.
    pair_excl
        The model-level pair exclusion, or ``None``.
    topology_params
        Build parameters by name: ``{"k_max": ...}`` for the k grid, with
        ``"bounds"`` for a fixed integer grid (see
        :func:`~.reciprocal_graph.build_fixed_k_grid`), and
        ``{"spacing": ...}`` for the mesh.

    Returns
    -------
    dict[str, PairGraph | KGrid | MeshGrid]
        Each declared topology under its name; empty when none is declared.

    Raises
    ------
    NotImplementedError
        If cells are given for a declaration of the pairs alone, or a pair
        exclusion for a periodic batch, whose reciprocal-space sum is global.
    ValueError
        If a name is unknown, a build parameter is missing, or a cell spans
        no lattice.
    """
    names = tuple(names)
    unknown = [name for name in names if name not in TOPOLOGY_TYPES]
    if unknown:
        raise ValueError(
            f"unknown topology {unknown[0]!r}; expected one of {tuple(TOPOLOGY_TYPES)}"
        )
    periodic = any(name in (K_GRID, PME_MESH) for name in names)
    if box is None:
        nf = n_node.shape[0]
        empty = {K_GRID: empty_k_grid, PME_MESH: empty_mesh_grid}
        return {
            name: (
                build_pair_graph(coord, n_node, atype, node_mask, pair_excl)
                if name == ALL_PAIRS
                else empty[name](nf, coord)
            )
            for name in names
        }
    if not periodic:
        if names:
            check_complete_frames(box)
        return {}
    if pair_excl is not None:
        raise NotImplementedError(
            "the reciprocal-space sum of a periodic system is global; "
            "`pair_exclude_types` cannot exclude pairs from it"
        )
    params = topology_params or {}
    return {
        name: (
            empty_pair_graph(coord)
            if name == ALL_PAIRS
            else _build_reciprocal(name, box, coord, n_node, params.get(name, {}))
        )
        for name in names
    }


def topology_flat_fields(topology: Topology) -> tuple[Array, ...]:
    """The fields of a topology in their declaration order."""
    return tuple(
        getattr(topology, field.name) for field in dataclasses.fields(topology)
    )


def flatten_topologies(topologies: dict[str, Topology]) -> tuple[Array, ...]:
    """Lay topologies out as positional inputs of a graph lower.

    A compiled or exported graph lower receives the fields of every topology,
    in the order of the dict and of the declaration of each dataclass, after
    its other inputs; :func:`unflatten_topologies` restores them.

    Parameters
    ----------
    topologies
        The topologies by name.

    Returns
    -------
    tuple
        The fields of every topology, concatenated.
    """
    return tuple(
        value
        for topology in topologies.values()
        for value in topology_flat_fields(topology)
    )


def unflatten_topologies(
    names: Sequence[str], values: Sequence[Array]
) -> dict[str, Topology]:
    """Restore topologies laid out by :func:`flatten_topologies`.

    Parameters
    ----------
    names
        The topology names, in the order they were laid out.
    values
        The fields of every topology, concatenated.

    Returns
    -------
    dict[str, PairGraph | KGrid | MeshGrid]
        The topologies by name.

    Raises
    ------
    ValueError
        If a name is unknown, or the number of values does not match the
        layout of the names.
    """
    topologies = {}
    offset = 0
    for name in names:
        if name not in TOPOLOGY_TYPES:
            raise ValueError(f"unknown topology {name!r}")
        cls = TOPOLOGY_TYPES[name]
        arity = len(dataclasses.fields(cls))
        if offset + arity > len(values):
            break
        topologies[name] = cls(*values[offset : offset + arity])
        offset += arity
    if len(topologies) != len(names) or offset != len(values):
        raise ValueError(
            f"{len(values)} flattened values given, which do not lay out the "
            f"topologies {tuple(names)}"
        )
    return topologies


def cast_topology_geometry(topology: Topology, dtype: Any) -> Topology:
    """A copy of a topology with its geometry fields cast to ``dtype``."""
    xp = array_api_compat.array_namespace(getattr(topology, topology.GEOMETRY[0]))
    return dataclasses.replace(
        topology,
        **{
            name: xp.astype(getattr(topology, name), dtype)
            for name in topology.GEOMETRY
        },
    )
