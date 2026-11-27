# SPDX-License-Identifier: LGPL-3.0-or-later
"""Static sparse batches with compact nodes and bounded capacity overhead."""

from collections.abc import (
    Sequence,
)
from itertools import (
    pairwise,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    pad_and_guard_edges,
)
from deepmd.dpmodel.utils.neighbor_graph.vesin_builder import (
    build_neighbor_graph_vesin,
)
from deepmd.utils.data import (
    DataRequirementItem,
)


def bucket_capacity(size: int, minimum: int = 1, *, subdivisions: int = 8) -> int:
    """Round a size to a configurable power-of-two interval grid.

    Parameters
    ----------
    size : int
        Required axis length.
    minimum : int
        Lower bound on the returned capacity.
    subdivisions : int
        Positive power of two controlling the number of buckets in each
        power-of-two interval. Eight bounds padding overhead by 12.5 percent;
        one rounds to a power of two and reduces compiled shape variants.

    Returns
    -------
    int
        Capacity at least as large as both ``size`` and ``minimum``.

    Raises
    ------
    ValueError
        If ``subdivisions`` is not a positive power-of-two integer.
    """
    if (
        not isinstance(subdivisions, int)
        or subdivisions < 1
        or subdivisions & (subdivisions - 1)
    ):
        raise ValueError("subdivisions must be a positive power-of-two integer")
    size = max(size, minimum)
    quantum = 1 << max(size.bit_length() - subdivisions.bit_length(), 0)
    return ((size + quantum - 1) // quantum) * quantum


def _pad_rows(value: np.ndarray, capacity: int, fill: int = 0) -> np.ndarray:
    padding = ((0, capacity - value.shape[0]),) + ((0, 0),) * (value.ndim - 1)
    return np.pad(value, padding, constant_values=fill)


#: Batch entries with one row per atom, besides the atomic data requirements.
ATOMIC_BATCH_KEYS = frozenset(
    {
        "coord",
        "type",
        "force",
        "force_mag",
        "spin",
        "aparam",
        "atom_ener",
        "atom_pref",
    }
)


def atomic_batch_keys(data_requirements: Sequence[DataRequirementItem]) -> set[str]:
    """Return the batch entries that have one row per atom."""
    return ATOMIC_BATCH_KEYS | {item.key for item in data_requirements if item.atomic}


def prepare_graph_batch(
    batch: dict[str, Any],
    *,
    rcut: float,
    ntypes: int,
    data_requirements: Sequence[DataRequirementItem] = (),
    pair_exclude_types: Sequence[tuple[int, int]] = (),
    bucket_subdivisions: int = 8,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build a full sparse graph and matching labels before device transfer.

    Physical nodes occupy a compact frame-major prefix. Node, edge and frame
    axes are padded independently to static capacities; neither the
    largest molecule nor the largest coordination number sets the padding
    of every other molecule or atom. ``type=-1``, ``edge_mask`` and
    ``frame_mask`` identify the three padding domains.

    Parameters
    ----------
    batch : dict
        NumPy data batch, rectangular or ragged with ``n_node``.
    rcut : float
        Neighbor cutoff, in Angstrom.
    ntypes : int
        Number of real atom types.
    data_requirements : sequence of DataRequirementItem
        Label and auxiliary-input contracts used to identify atomic fields.
    pair_exclude_types : sequence of tuple[int, int]
        Model-level atom-type pairs excluded from the graph.
    bucket_subdivisions : int
        Number of node/edge capacity buckets per power-of-two interval.

    Returns
    -------
    labels : dict
        Labels on the padded node/frame axes, including validity masks.
    inputs : dict
        Arguments for the model's graph lower interface.
    """
    return prepare_graph_batches(
        (batch,),
        rcut=rcut,
        ntypes=ntypes,
        data_requirements=data_requirements,
        pair_exclude_types=pair_exclude_types,
        bucket_subdivisions=bucket_subdivisions,
    )


def prepare_graph_batches(
    batches: Sequence[dict[str, Any]],
    *,
    rcut: float,
    ntypes: int,
    data_requirements: Sequence[DataRequirementItem] = (),
    pair_exclude_types: Sequence[tuple[int, int]] = (),
    bucket_subdivisions: int = 8,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Combine original device batches into one independently padded graph.

    Parameters
    ----------
    batches : sequence of dict
        Rectangular or ragged NumPy batches. Each batch retains its own
        periodicity and label-availability flags.
    rcut : float
        Neighbor cutoff, in Angstrom.
    ntypes : int
        Number of real atom types.
    data_requirements : sequence of DataRequirementItem
        Atomic/frame field contracts and defaults for missing optional fields.
    pair_exclude_types : sequence of tuple[int, int]
        Model-level atom-type pairs excluded from each graph.
    bucket_subdivisions : int
        Positive power-of-two number of node/edge buckets per interval.

    Returns
    -------
    labels : dict
        Padded labels. Multiple batches additionally carry ``data_group``
        frame identifiers and per-batch ``group_find_*`` availability arrays.
    inputs : dict
        One graph with compact physical nodes and edges preceding padding.

    Notes
    -----
    Neighbor searches run independently before graph concatenation, so a
    periodic batch can share an update with a nonperiodic batch. Padding is
    applied only after concatenation; no padding node occurs between frames.
    """
    if not batches:
        raise ValueError("Graph preparation requires at least one input batch.")
    atomic_keys = atomic_batch_keys(data_requirements)
    frame_keys = {item.key for item in data_requirements if not item.atomic}
    frame_keys.update({"energy", "virial", "fparam", "charge_spin"})
    pair_excl = (
        PairExcludeMask(ntypes, list(pair_exclude_types))
        if pair_exclude_types
        else None
    )
    compact = [
        _compact_graph_batch(batch, rcut, pair_excl, atomic_keys, frame_keys)
        for batch in batches
    ]
    labels, inputs = compact[0]
    if len(compact) > 1:
        labels, inputs = _merge_graph_batches(
            compact,
            atomic_keys,
            {item.key: item.default for item in data_requirements},
        )
    return _pad_graph(
        labels, inputs, atomic_keys, bucket_subdivisions=bucket_subdivisions
    )


def _compact_graph_batch(
    batch: dict[str, Any],
    rcut: float,
    pair_excl: PairExcludeMask | None,
    atomic_keys: set[str],
    frame_keys: set[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    atype = np.asarray(batch["atype"] if "atype" in batch else batch["type"])
    if atype.ndim == 2:
        n_node = np.full(atype.shape[0], atype.shape[1], dtype=np.int64)
    else:
        n_node = np.asarray(batch["n_node"], dtype=np.int64)
    nframes = n_node.size
    original_nodes = atype.size
    physical = atype.reshape(-1) >= 0
    offsets = np.concatenate(([0], np.cumsum(n_node)))
    n_node = np.array(
        [np.count_nonzero(physical[a:b]) for a, b in pairwise(offsets)],
        dtype=np.int64,
    )
    atype = atype.reshape(-1)[physical]
    coord = np.asarray(batch["coord"]).reshape(original_nodes, 3)[physical]
    box = batch.get("box") if batch.get("find_box", True) else None
    graph = build_neighbor_graph_vesin(
        coord,
        atype,
        box,
        rcut,
        GraphLayout(min_edges=0),
        n_node=n_node,
        pair_excl=pair_excl,
    )
    inputs = {
        "atype": atype,
        "n_node": n_node,
        "edge_index": graph.edge_index,
        "edge_vec": graph.edge_vec,
    }
    labels: dict[str, Any] = {
        "type": inputs["atype"],
        "n_node": inputs["n_node"],
        "frame_mask": np.ones(nframes, dtype=np.bool_),
    }
    for key, value in batch.items():
        if value is None or key in {"atype", "type", "n_node"}:
            continue
        if key.startswith("find_"):
            labels[key] = bool(value)
        elif key in atomic_keys:
            labels[key] = np.asarray(value).reshape(original_nodes, -1)[physical]
        elif key in frame_keys:
            labels[key] = np.asarray(value).reshape(nframes, -1)
    return labels, inputs


def _merge_graph_batches(
    batches: Sequence[tuple[dict[str, Any], dict[str, np.ndarray]]],
    atomic_keys: set[str],
    defaults: dict[str, float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    node_counts = [inputs["atype"].size for _, inputs in batches]
    frame_counts = [inputs["n_node"].size for _, inputs in batches]
    offsets = np.concatenate(([0], np.cumsum(node_counts)))
    inputs = {
        "edge_vec": np.concatenate([graph["edge_vec"] for _, graph in batches]),
    }
    inputs["edge_index"] = np.concatenate(
        [
            graph["edge_index"] + offset
            for (_, graph), offset in zip(batches, offsets[:-1], strict=True)
        ],
        axis=1,
    )
    labels = {}
    keys = sorted({key for current, _ in batches for key in current})
    for key in keys:
        if key.startswith("find_"):
            available = np.array(
                [current.get(key, False) for current, _ in batches], dtype=np.bool_
            )
            labels[key] = bool(np.any(available))
            labels[f"group_{key}"] = available
            continue
        template = next(current[key] for current, _ in batches if key in current)
        counts = node_counts if key in atomic_keys else frame_counts
        values = [
            current[key]
            if key in current
            else np.full(
                (count, *template.shape[1:]),
                defaults.get(key, 0.0),
                dtype=template.dtype,
            )
            for (current, _), count in zip(batches, counts, strict=True)
        ]
        labels[key] = np.concatenate(values, axis=0)
    labels["data_group"] = np.repeat(
        np.arange(len(batches), dtype=np.int32), frame_counts
    )
    inputs["atype"] = labels["type"]
    inputs["n_node"] = labels["n_node"]
    return labels, inputs


def _pad_graph(
    labels: dict[str, Any],
    inputs: dict[str, np.ndarray],
    atomic_keys: set[str],
    *,
    bucket_subdivisions: int = 8,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ncapacity = bucket_capacity(inputs["atype"].size, subdivisions=bucket_subdivisions)
    # Each physical frame contains at least one node. Deriving frame capacity
    # from node capacity removes an independent compilation dimension while
    # padding only the small count, label and conditioning arrays. The explicit
    # frame count also admits input frames made entirely of phantom atoms.
    frame_size = max(ncapacity, inputs["n_node"].size)
    fcapacity = 1 << (frame_size - 1).bit_length()
    ecapacity = bucket_capacity(
        inputs["edge_vec"].shape[0],
        minimum=2,
        subdivisions=bucket_subdivisions,
    )
    inputs["edge_index"], inputs["edge_vec"], inputs["edge_mask"] = pad_and_guard_edges(
        inputs["edge_index"],
        inputs["edge_vec"],
        ecapacity,
    )
    for key, value in labels.items():
        if key.startswith(("find_", "group_find_")):
            continue
        capacity = ncapacity if key in atomic_keys else fcapacity
        labels[key] = _pad_rows(
            value, capacity, -1 if key in ("type", "data_group") else 0
        )
    inputs["atype"] = labels["type"]
    inputs["n_node"] = labels["n_node"]
    for key in ("spin", "charge_spin", "fparam", "aparam"):
        if key in labels:
            inputs[key] = labels[key]
    return labels, inputs
