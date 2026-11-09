# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Conditions that select and mix the weights of a mixture of experts (MoE).

An expert map (:mod:`deepmd.dpmodel.utils.expert_linear`) computes a row with
the weight

.. math::

    W + V_\mathrm{task} + U_{z} + \sum_k c_k A_k ,

where :math:`W` is shared, :math:`V_\mathrm{task}` is the increment of the
task of the model, :math:`U_z` the increment of the element :math:`z` of the
atom of the row, and the :math:`A_k` are experts mixed by the coefficients
:math:`c` of a router. The task and the element are discrete conditions that
select an increment; the router reads the conditions of the frame, the task
and the element. None of the conditions depends on the coordinates, so the
weights are constant along a trajectory and the energy keeps the smoothness of
the model.

Merging the task increment and the experts at the coefficients of one
scenario into :math:`W` gives a model without a router, with or without the
element increments. Merge-aware training (MAT) trains the model for the merge
without them: a training frame is evaluated with a probability as that merged
model evaluates it, with no weight specific to an element.

The weight of a row is therefore one of a few: all rows of one element in the
frames of one charge/spin state share it. Rows that share their weight form a
group, and a map contracts the groups with their weights in one batched
product (:class:`RowLayout`), so neither the number of elements nor the number
of experts enters the cost of a contraction.

This module holds what decides the weights of a row: the configuration
(:func:`normalize_moe`), the conditions of a forward (:class:`ExpertCondition`),
the router (:class:`ConditionRouter`), the weights of the rows of one forward
(:class:`ExpertRoute`, :class:`UnitRoute`), the scenario of a merge
(:class:`ExpertScenario`) and what an MoE descriptor or fitting owns and does
next to its maps (:class:`ExpertComponent`). It also numbers the
(frame, type) pairs of a forward (:func:`frame_type_counts`,
:func:`reference_keys`, :func:`node_reference_index`), by which an atomic
model references every atom to an isolated atom of its type with the weights
of its frame.
"""

from __future__ import (
    annotations,
)

import copy
import dataclasses
import logging
import math
from typing import (
    TYPE_CHECKING,
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    Array,
    xp_add_at,
    xp_asarray_nodetach,
    xp_hint_dynamic_size,
    xp_matmul,
    xp_sigmoid,
    xp_softmax,
)
from deepmd.dpmodel.common import (
    get_xp_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils.expert_linear import (
    ChunkWeights,
    chunk_weights,
    init_task_increments,
    load_task_variables,
    pair_task_increments,
    task_variables,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )

log = logging.getLogger(__name__)

# Rows of one chunk of a row layout (see :class:`RowLayout`). A larger chunk
# holds fewer copies of the weights and pads more rows per group. Rows grouped
# by element slot alone form few large groups; rows grouped by the state of
# their frame as well form many small ones.
SLOT_ROW_CHUNK = 64
STATE_ROW_CHUNK = 32
# A layout of at most this many element slots, slot 0 included, takes a shape
# that follows from the number of rows alone.
STATIC_GROUPS = 16
# Value of ``ExpertScenario.elements`` that keeps every element increment.
ALL_ELEMENTS = "all"


def normalize_moe(
    moe: dict[str, Any] | None, *, charge_spin: bool = True
) -> dict[str, Any] | None:
    """
    Normalize the mixture-of-experts configuration of a descriptor or a fitting.

    Parameters
    ----------
    moe : dict or None
        Configuration of the MoE:

        - ``n_experts`` (8): number of experts mixed by the router; ``0``
          builds no router.
        - ``element`` (true): the elements with an increment of their own.
          ``true`` gives every atom type one, ``false`` none, and a list of
          element symbols gives the listed types one. With element increments
          the router also reads the element of the atom.
        - ``mat_prob`` (0.37): probability of merge-aware training (MAT),
          with which a training frame is evaluated as the model merged
          without elements evaluates it, with no weight specific to an
          element. MAT trains the weights shared by the elements as a model
          of their own, which that merge yields.
        - ``router``: shape of the router, with ``dim`` (64) the width of the
          condition embedding and ``layers`` (1) the number of silu
          activations before the output heads: the first acts on the
          condition embedding, every further one on a linear map of that
          width.

    charge_spin : bool
        Whether the router reads the charge and spin multiplicity of the
        frame. A router without them reads the task and the element, so it
        requires element increments.

    Returns
    -------
    dict or None
        The configuration with every key present; None without ``moe``.

    Raises
    ------
    ValueError
        If a key is unknown or a value is out of range, or a router has no
        condition to read.
    """
    if moe is None:
        return None
    unknown = sorted(set(moe) - {"n_experts", "element", "mat_prob", "router"})
    if unknown:
        raise ValueError(
            "`moe` takes `n_experts`, `element`, `mat_prob` and `router`; "
            f"got {unknown}"
        )
    n_experts = int(moe.get("n_experts", 8))
    if n_experts < 0:
        raise ValueError(f"`n_experts` is non-negative; got {n_experts}")
    element = moe.get("element", True)
    if isinstance(element, (list, tuple)):
        element = [str(name) for name in element]
        if len(set(element)) != len(element):
            raise ValueError(f"`element` lists every element once; got {element}")
        element = element or False
    elif not isinstance(element, bool):
        raise ValueError(
            f"`element` is true, false or a list of element symbols; got {element!r}"
        )
    mat_prob = float(moe.get("mat_prob", 0.37))
    if not 0.0 <= mat_prob < 1.0:
        raise ValueError(f"`mat_prob` is in [0, 1); got {mat_prob}")
    if n_experts and not (charge_spin or element):
        raise ValueError(
            "the router of this component reads the task and the element of the "
            "atom, so `n_experts` needs `element`; set `n_experts` to 0 for the "
            "task increment alone"
        )
    router = dict(moe.get("router") or {})
    unknown = sorted(set(router) - {"dim", "layers"})
    if unknown:
        raise ValueError(f"`router` takes `dim` and `layers`; got {unknown}")
    dim = int(router.get("dim", 64))
    layers = int(router.get("layers", 1))
    if dim < 1 or layers < 1:
        raise ValueError(
            f"router `dim` and `layers` are positive; got {dim} and {layers}"
        )
    return {
        "n_experts": n_experts,
        "element": element,
        "mat_prob": mat_prob,
        "router": {"dim": dim, "layers": layers},
    }


def reject_moe(options: dict[str, Any], component: str) -> None:
    """
    Refuse the ``moe`` option of a component in a backend without it.

    Parameters
    ----------
    options : dict
        Options of the component.
    component : str
        Name of the component in the message.

    Raises
    ------
    NotImplementedError
        If the options configure MoE.
    """
    if options.get("moe") is not None:
        raise NotImplementedError(
            f"MoE of the {component} (`moe`) is implemented by the PyTorch "
            "Exportable backend; train with `dp --pt-expt`, or merge the model "
            "into a plain model with `dp --pt-expt moe-merge` first"
        )


def element_names(
    element: bool | list[str], type_map: list[str] | None
) -> list[str] | None:
    """
    Elements with an increment of their own, in the order of their slots.

    Parameters
    ----------
    element : bool or list[str]
        The normalized ``element`` option (see :func:`normalize_moe`).
    type_map : list[str] or None
        Element symbol of every atom type.

    Returns
    -------
    list[str] or None
        The elements of the slots ``1, 2, ...``; None when ``element`` is
        ``True`` and the type map is unknown, where the slots are the types.
    """
    if element is False:
        return []
    if element is True:
        return None if type_map is None else list(type_map)
    return list(element)


def element_slots(
    element: bool | list[str], ntypes: int, type_map: list[str] | None
) -> list[int] | None:
    """
    Element slot of every atom type.

    Slot 0 holds no increment; the elements with an increment take the slots
    ``1, 2, ...`` in the order of ``element``, or of the types for ``True``.
    A type whose element is not listed takes slot 0, and a listed element
    that is not in the type map keeps its slot, which no atom takes.

    Parameters
    ----------
    element : bool or list[str]
        The normalized ``element`` option (see :func:`normalize_moe`).
    ntypes : int
        Number of atom types.
    type_map : list[str] or None
        Element symbol of every atom type; required by a list of elements.

    Returns
    -------
    list[int] or None
        The slot of every atom type; None without element increments.

    Raises
    ------
    ValueError
        If a list of elements is given without the type map.
    """
    if element is False:
        return None
    if element is True:
        return list(range(1, ntypes + 1))
    if type_map is None:
        raise ValueError("a list of elements needs the `type_map` of the model")
    return [element.index(name) + 1 if name in element else 0 for name in type_map]


def n_element_increments(element: bool | list[str], ntypes: int) -> int:
    """Number of element increments that the ``element`` option builds."""
    if element is False:
        return 0
    return ntypes if element is True else len(element)


@dataclasses.dataclass(frozen=True)
class ExpertCondition:
    """Conditions of the frames of one forward.

    Attributes
    ----------
    node_frame : Array
        Frame of every node with shape (N,). A reference node of an isolated
        atom takes the frame it is the reference of.
    charge_spin : Array or None
        Charge and spin multiplicity of the frames with shape (nf, 2), the
        default state of the model where none is given; None for a model
        that takes no charge/spin condition or has no default.
    mat_draw : Array or None
        One uniform sample in [0, 1) per frame with shape (nf,), drawn by a
        training forward for the merge-aware training (MAT). A component
        with the MAT probability ``p`` evaluates the frames whose sample is
        below ``p`` without its element-specific weights, so components of
        equal ``p`` train the same frames merge-aware. None uses the
        element-specific weights in every frame.
    """

    node_frame: Array
    charge_spin: Array | None = None
    mat_draw: Array | None = None

    def element_keep(self, mat_prob: float) -> Array | None:
        """Whether every node uses element-specific weights, with shape (N,).

        None when every node does.
        """
        if self.mat_draw is None or not mat_prob:
            return None
        xp = array_api_compat.array_namespace(self.mat_draw)
        keep = xp.astype(self.mat_draw >= mat_prob, xp.int64)
        return xp.take(keep, self.node_frame, axis=0)


def node_slots(
    node_type: Array, element_slot: list[int] | None, keep: Array | None = None
) -> Array | None:
    """
    Element slot of every node.

    Parameters
    ----------
    node_type : Array
        Node types with shape (N,).
    element_slot : list[int] or None
        Element slot of every atom type (see :func:`element_slots`).
    keep : Array, optional
        Whether every node uses element-specific weights, with shape (N,).

    Returns
    -------
    Array or None
        The slots with shape (N,); None without element increments.
    """
    if element_slot is None:
        return None
    xp = array_api_compat.array_namespace(node_type)
    slot = xp.take(
        xp.asarray(
            element_slot,
            dtype=node_type.dtype,
            device=array_api_compat.device(node_type),
        ),
        xp.maximum(node_type, xp.zeros_like(node_type)),
        axis=0,
    )
    return slot if keep is None else slot * xp.astype(keep, slot.dtype)


def frame_states(charge_spin: Array) -> tuple[Array, Array]:
    """
    Index the distinct charge/spin states of the frames.

    Parameters
    ----------
    charge_spin : Array
        Charge and spin multiplicity of the frames with shape (nf, 2).

    Returns
    -------
    state : Array
        Index of the state of every frame with shape (nf,); frames of equal
        charge and spin multiplicity share an index, and the indices of the
        ``S`` distinct states are ``0, ..., S - 1``.
    state_frame : Array
        A frame of every state with shape (nf,); meaningful below ``S``.
    """
    xp = array_api_compat.array_namespace(charge_spin)
    device = array_api_compat.device(charge_spin)
    nf = charge_spin.shape[0]
    # === Step 1. Sort the frames by charge, then by spin multiplicity ===
    order = xp.argsort(charge_spin[:, 1], stable=True)
    order = xp.take(
        order, xp.argsort(xp.take(charge_spin[:, 0], order), stable=True), axis=0
    )
    ordered = xp.take(charge_spin, order, axis=0)
    # === Step 2. Number the runs of equal states ===
    # A frame opens a run when it is the first one or differs from the frame
    # before it. Every frame is compared with a gathered predecessor, the
    # first one with itself, so the comparison holds for any number of frames.
    rank = xp.arange(nf, dtype=xp.int64, device=device)
    before = xp.take(ordered, xp.maximum(rank - 1, xp.zeros_like(rank)), axis=0)
    first = xp.astype(
        xp.logical_or(rank == 0, xp.any(ordered != before, axis=-1)), xp.int64
    )
    run = xp.cumulative_sum(first) - 1
    zeros = xp.zeros((nf,), dtype=xp.int64, device=device)
    state = xp_add_at(zeros, order, run)
    state_frame = xp_add_at(xp.zeros_like(zeros), run, order * first)
    return state, state_frame


@dataclasses.dataclass(frozen=True)
class ExpertScenario:
    """Scenario that an MoE model is merged for.

    Attributes
    ----------
    charge_spin : list[float] or None
        Charge and spin multiplicity of the frames of the scenario; None takes
        the default state of the model.
    elements : tuple[str, ...] or str
        Elements that keep an increment of their own in the merged model, or
        :data:`ALL_ELEMENTS` for every element with one. The other elements
        use the weights shared by the elements, so the merged model computes
        what the model computes on systems of the kept elements; without
        any, the merged model has no MoE.
    """

    charge_spin: list[float] | None = None
    elements: tuple[str, ...] | str = ()


def merged_elements(
    scenario: ExpertScenario, moe: dict[str, Any], type_map: list[str] | None
) -> tuple[list[str], list[int]]:
    """
    Elements of a component that keep their increment in a merge.

    Parameters
    ----------
    scenario : ExpertScenario
        The scenario of the merge.
    moe : dict
        The normalized expert configuration of the component.
    type_map : list[str] or None
        Element symbol of every atom type of the component.

    Returns
    -------
    kept : list[str]
        The kept elements in the order of their slots; empty for a component
        without element increments.
    slots : list[int]
        The slots of the kept elements.

    Raises
    ------
    ValueError
        If the elements of the component are not known by name, the scenario
        names an element that has no increment in a component with element
        increments, or it drops every increment of a component whose shared
        weights were never trained without them.

    Notes
    -----
    A component trained without merge-aware training (``mat_prob`` 0) never
    ran its shared weights without an element increment. A scenario that
    keeps a part of
    its elements is merged with a warning: the merged component serves
    systems of the kept elements, and the atoms of the other elements take
    shared weights that were not trained for them.
    """
    if moe["element"] is False:
        return [], []
    owned = element_names(moe["element"], type_map)
    if owned is None:
        raise ValueError("merging element weights needs the `type_map` of the model")
    if scenario.elements == ALL_ELEMENTS:
        kept = owned
    else:
        missing = sorted(set(scenario.elements) - set(owned))
        if missing:
            raise ValueError(
                f"{missing} have no weights of their own to keep; the elements "
                f"with weights of their own are {owned}"
            )
        if not scenario.elements and not moe["mat_prob"]:
            raise ValueError(
                "the model was trained without merge-aware training (`mat_prob` "
                "0), so the weights shared by the elements never ran without "
                "the weights of the elements; keep the elements of the scenario "
                "in the merged model"
            )
        kept = [name for name in owned if name in scenario.elements]
        if len(kept) < len(owned) and not moe["mat_prob"]:
            log.warning(
                "The model was trained without merge-aware training (`mat_prob` "
                "0), so the weights shared by the elements never ran without "
                "the weights of an element: the merged model serves systems of "
                "%s only.",
                kept,
            )
    return kept, [owned.index(name) + 1 for name in kept]


def merged_moe(moe: dict[str, Any], kept: list[str]) -> dict[str, Any] | None:
    """
    Expert configuration of a merged component.

    Parameters
    ----------
    moe : dict
        The normalized expert configuration of the component.
    kept : list[str]
        The elements that keep their increment (see :func:`merged_elements`).

    Returns
    -------
    dict or None
        The configuration with the kept elements and no router; None when no
        element is kept, which is the plain component.
    """
    return {**moe, "n_experts": 0, "element": list(kept)} if kept else None


@dataclasses.dataclass(frozen=True)
class RowLayout:
    """Rows grouped into chunks of equal size that share their weights.

    The rows of a group fill consecutive chunks of ``chunk`` rows, so a map
    contracts every chunk with the weights of its group in one batched
    product, whose cost does not grow with the number of groups. A chunk
    that its group does not fill repeats a row in its free rows; their
    outputs are dropped on the way back.

    The number of chunks is chosen in one of two ways. A static layout holds
    ``(N + (chunk - 1) * n_groups) // chunk`` chunks, which bounds the chunks
    any assignment of ``N`` rows needs; its shape follows from the number of
    rows alone, which lets an exported graph plan its memory, at the price of
    at most one idle chunk per group, so it serves few groups. Any other
    layout holds exactly the chunks in use, whose number follows the groups
    of the rows.

    Attributes
    ----------
    position : Array
        Row of the layout that every row fills, with shape (N,).
    source : Array
        Row that fills every row of the layout, with shape
        (n_chunks * chunk,).
    chunk_group : Array
        Group of every chunk with shape (n_chunks,).
    chunk : int
        Number of rows of a chunk.
    """

    position: Array
    source: Array
    chunk_group: Array
    chunk: int

    @classmethod
    def build(
        cls, group: Array, n_groups: int, *, chunk: int, static: bool
    ) -> RowLayout:
        """
        Lay out rows by their group.

        Parameters
        ----------
        group : Array
            Group of every row with shape (N,), in ``[0, n_groups)``; a
            layout holds at least one row.
        n_groups : int
            Number of groups.
        chunk : int
            Number of rows of a chunk.
        static : bool
            Whether the number of chunks follows from the number of rows
            alone.

        Returns
        -------
        RowLayout
            The layout of the rows.
        """
        xp = array_api_compat.array_namespace(group)
        device = array_api_compat.device(group)
        n_rows = group.shape[0]
        rows = xp.arange(n_rows, dtype=group.dtype, device=device)

        # === Step 1. Rows sorted by group ===
        # ``order`` lists the rows group by group; the rows of group ``g``
        # are the sorted rows ``[group_first[g], group_first[g] + counts[g])``.
        order = xp.argsort(group, stable=True)
        sorted_group = xp.take(group, order, axis=0)
        bounds = xp.searchsorted(
            sorted_group, xp.arange(n_groups + 1, dtype=group.dtype, device=device)
        )
        group_first = bounds[:-1]
        counts = bounds[1:] - group_first

        # === Step 2. Chunks of every group ===
        group_chunks = (counts + chunk - 1) // chunk
        chunk_end = xp.cumulative_sum(group_chunks)
        chunk_first = chunk_end - group_chunks
        if static:
            chunks = xp.arange(
                (n_rows + (chunk - 1) * n_groups) // chunk,
                dtype=group.dtype,
                device=device,
            )
        else:
            # Every chunk in use holds a row, so the rows bound their number.
            chunks = xp.nonzero(rows < chunk_end[-1])[0]
            xp_hint_dynamic_size(chunks)
        # A chunk belongs to the first group that ends behind it, which skips
        # the groups without rows; the idle chunks behind the last group take
        # its weights.
        chunk_group = xp.clip(
            xp.searchsorted(chunk_end, chunks, side="right"), max=n_groups - 1
        )

        # === Step 3. Rows of the layout ===
        # A row fills the layout row of its rank among the rows of its group.
        sorted_position = (
            xp.take(chunk_first, sorted_group, axis=0) * chunk
            + rows
            - xp.take(group_first, sorted_group, axis=0)
        )
        position = xp.take(sorted_position, xp.argsort(order), axis=0)
        # A layout row is filled by the row of its rank in the group of its
        # chunk; the free rows repeat the first sorted row.
        rank = (chunks - xp.take(chunk_first, chunk_group, axis=0))[
            :, None
        ] * chunk + xp.arange(chunk, dtype=group.dtype, device=device)[None, :]
        sorted_row = xp.where(
            rank < xp.take(counts, chunk_group, axis=0)[:, None],
            xp.take(group_first, chunk_group, axis=0)[:, None] + rank,
            xp.zeros_like(rank),
        )
        source = xp.take(order, xp.reshape(sorted_row, (-1,)), axis=0)
        return cls(
            position=position, source=source, chunk_group=chunk_group, chunk=chunk
        )

    def to_chunks(self, x: Array) -> Array:
        """Rows (N, ...) into the rows (n_chunks * chunk, ...) of the layout."""
        xp = array_api_compat.array_namespace(x)
        return xp.take(x, self.source, axis=0)

    def from_chunks(self, x: Array) -> Array:
        """Rows (n_chunks * chunk, ...) of the layout back into rows (N, ...)."""
        xp = array_api_compat.array_namespace(x)
        return xp.take(x, self.position, axis=0)


def layout_rows(
    slot: Array | None,
    n_slots: int,
    state: Array | None = None,
    n_states: int = 1,
) -> tuple[RowLayout | None, Array | None, Array | None]:
    """
    Lay out rows by the conditions that select their weights.

    Rows of one element slot share their weights, and under a router that
    reads the state of the frames, rows of one element slot in frames of one
    state do.

    Parameters
    ----------
    slot : Array or None
        Element slot of every row with shape (N,), in ``[0, n_slots)``; None
        when the weights do not depend on the element.
    n_slots : int
        Number of element slots.
    state : Array or None
        State of the frame of every row with shape (N,), in
        ``[0, n_states)``; None when the weights do not depend on the frame.
    n_states : int
        Number of states.

    Returns
    -------
    layout : RowLayout or None
        The layout of the rows; None when all rows share one set of weights.
    chunk_slot : Array or None
        Element slot of every chunk with shape (n_chunks,); None without
        ``slot``.
    chunk_state : Array or None
        State of every chunk with shape (n_chunks,); None without ``state``.
    """
    if slot is None and state is None:
        return None, None, None
    if slot is None:
        n_slots = 1
        group = state
    elif state is None:
        group = slot
    else:
        group = state * n_slots + slot
    # The number of states follows the frames, so only a layout by element
    # slot has a number of groups that is known to be small.
    layout = RowLayout.build(
        group,
        n_states * n_slots,
        chunk=SLOT_ROW_CHUNK if state is None else STATE_ROW_CHUNK,
        static=state is None and n_slots <= STATIC_GROUPS,
    )
    return (
        layout,
        None if slot is None else layout.chunk_group % n_slots,
        None if state is None else layout.chunk_group // n_slots,
    )


@dataclasses.dataclass(frozen=True)
class UnitRoute:
    """MoE weights of the rows of one forward for one unit.

    A unit that owns expert maps (a feed-forward network, a fitting network)
    enters the route once (:meth:`enter`), so that its rows become the rows
    of the layout on which every map of the unit works, hands every map its
    weights (:meth:`map`, :meth:`maps`), and leaves the route with its output
    (:meth:`leave`).

    Attributes
    ----------
    layout : RowLayout or None
        Layout of the rows; None when all rows share one set of weights.
    weights : Sequence[ChunkWeights]
        Weights of every chunk for the expert maps the route is handed to,
        in their order.
    """

    layout: RowLayout | None
    weights: Sequence[ChunkWeights]

    def maps(self, start: int, stop: int | None = None) -> UnitRoute:
        """Route of the maps ``start`` to ``stop`` among those of this route."""
        return dataclasses.replace(self, weights=self.weights[start:stop])

    def map(self, index: int) -> ChunkWeights:
        """Weights of one map among those of this route."""
        return self.weights[index]

    def enter(self, x: Array) -> Array:
        """Rows of a unit as the maps of the unit take them."""
        return x if self.layout is None else self.layout.to_chunks(x)

    def leave(self, x: Array) -> Array:
        """Output of the maps of a unit as the rows the unit was given."""
        return x if self.layout is None else self.layout.from_chunks(x)


@dataclasses.dataclass(frozen=True)
class ExpertRoute:
    """MoE weights of the rows of one forward for a component.

    The rows are laid out in chunks that share their weights (see
    :class:`RowLayout`), and the route holds the element slot and the
    coefficients of the routed experts of every chunk. Every unit of the
    component takes its own route from it (:meth:`unit`).

    Attributes
    ----------
    task : Sequence
        Task increments ``(weight, bias)`` of the expert maps of the
        component, in their order.
    layout : RowLayout or None
        Layout of the rows; None when all rows share one set of weights,
        which are those of a single chunk.
    chunk_slot : Array or None
        Element slot of every chunk with shape (n_chunks,); None without
        element increments.
    chunk_coeff : Array or None
        Coefficients of the routed experts of every chunk and unit with shape
        (n_chunks, n_units, K); None without a router.
    """

    task: Sequence[Any]
    layout: RowLayout | None = None
    chunk_slot: Array | None = None
    chunk_coeff: Array | None = None

    def unit(self, index: int, maps: Sequence[Any], start: int) -> UnitRoute:
        """
        Route of one unit of the component.

        Parameters
        ----------
        index : int
            Position of the unit among the units of the component, which
            selects its coefficients.
        maps : Sequence
            The expert maps of the unit in their order.
        start : int
            Position of the first map of the unit among the expert maps of
            the component, which selects the task increments.

        Returns
        -------
        UnitRoute
            The route with the weights of every chunk for the maps of the
            unit.
        """
        return UnitRoute(
            self.layout,
            chunk_weights(
                maps,
                self.task[start : start + len(maps)],
                None if self.layout is None else self.layout.chunk,
                self.chunk_slot,
                None if self.chunk_coeff is None else self.chunk_coeff[:, index, :],
            ),
        )


class ConditionRouter(NativeOP):
    r"""
    Router of the expert coefficients.

    The router embeds the conditions of a row additively,

    .. math::

        u = e_\mathrm{task} + e_\mathrm{cs} W_\mathrm{cs} + E_z ,

    with the task embedding of the model, the charge/spin embedding of the
    frame and the embedding of the element slot :math:`z` of the atom (zero
    for slot 0), applies silu and ``layers - 1`` further silu layers of width
    ``dim``, and gives every unit the softmax of its own linear head over the
    experts.

    Parameters
    ----------
    n_units
        Number of units, each with coefficients of its own.
    n_experts
        Number of experts ``K`` of a unit.
    dim
        Width of the condition embedding.
    layers
        Number of silu activations before the heads: the first acts on the
        condition embedding, every further one on a linear map of width
        ``dim``.
    chg_spin_dim
        Width of the charge/spin embedding of the frames; ``0`` reads none.
    n_elements
        Number of element slots with an embedding; ``0`` reads no element.
    precision
        Parameter precision.
    seed
        Random seed for initialization.
    trainable
        Whether parameters are trainable.
    """

    # Arrays that a backend trains.
    PARAMETER_ATTRIBUTES = (
        "adamw_chg_spin",
        "adamw_element",
        "adamw_hidden",
        "hidden_bias",
        "adamw_head",
        "head_bias",
    )

    def __init__(
        self,
        *,
        n_units: int,
        n_experts: int,
        dim: int = 64,
        layers: int = 1,
        chg_spin_dim: int = 0,
        n_elements: int = 0,
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        self.n_units = int(n_units)
        self.n_experts = int(n_experts)
        self.dim = int(dim)
        self.layers = int(layers)
        self.chg_spin_dim = int(chg_spin_dim)
        self.n_elements = int(n_elements)
        self.precision = precision
        self.trainable = bool(trainable)
        prec = PRECISION_DICT[self.precision.lower()]
        rng = np.random.default_rng(seed)
        # Unit-variance embeddings make ``u`` of order one.
        self.adamw_chg_spin = (
            rng.normal(
                0.0, 1.0 / math.sqrt(self.chg_spin_dim), (self.chg_spin_dim, self.dim)
            ).astype(prec)
            if self.chg_spin_dim
            else None
        )
        self.adamw_element = (
            rng.normal(0.0, 1.0, (self.n_elements, self.dim)).astype(prec)
            if self.n_elements
            else None
        )
        n_hidden = self.layers - 1
        self.adamw_hidden = (
            rng.normal(
                0.0, 1.0 / math.sqrt(self.dim), (n_hidden, self.dim, self.dim)
            ).astype(prec)
            if n_hidden
            else None
        )
        self.hidden_bias = (
            np.zeros((n_hidden, self.dim), dtype=prec) if n_hidden else None
        )
        n_out = self.n_units * self.n_experts
        self.adamw_head = rng.normal(
            0.0, 1.0 / math.sqrt(self.dim), (self.dim, n_out)
        ).astype(prec)
        self.head_bias = np.zeros((n_out,), dtype=prec)

    def call(
        self,
        task_embedding: Array,
        frame: Array | None = None,
        slot: Array | None = None,
        chg_spin_embedding: Array | None = None,
    ) -> Array:
        """
        Expert coefficients of every group of rows.

        Parameters
        ----------
        task_embedding
            Task embedding of the model with shape (dim,).
        frame
            A frame of every group with shape (G,), whose charge/spin state
            the group has; required when the router reads the charge/spin of
            the frames.
        slot
            Element slot of every group with shape (G,); required when the
            router reads the element.
        chg_spin_embedding
            Charge/spin embedding of the frames with shape (nf, chg_spin_dim);
            required when the router reads it.

        Returns
        -------
        Array
            Coefficients with shape (G, n_units, n_experts), with one group
            when the router reads neither.
        """
        xp = array_api_compat.array_namespace(task_embedding)
        device = array_api_compat.device(task_embedding)
        dtype = get_xp_precision(xp, self.precision)

        def parameter(value: Any) -> Array:
            return xp_asarray_nodetach(xp, value[...], device=device)

        # === Step 1. Condition embedding ===
        u = xp.astype(task_embedding, dtype)[None, :]
        if self.adamw_chg_spin is not None:
            state = xp_matmul(
                xp.astype(chg_spin_embedding, dtype), parameter(self.adamw_chg_spin)
            )
            u = u + xp.take(state, frame, axis=0)
        if self.adamw_element is not None:
            # Slot 0 carries no element, so its embedding is zero.
            element = xp.take(
                parameter(self.adamw_element),
                xp.maximum(slot - 1, xp.zeros_like(slot)),
                axis=0,
            )
            u = u + element * xp.astype(slot > 0, dtype)[:, None]
        # === Step 2. Hidden layers and heads ===
        hidden = u * xp_sigmoid(u)
        if self.adamw_hidden is not None:
            weights = parameter(self.adamw_hidden)
            biases = parameter(self.hidden_bias)
            for i in range(weights.shape[0]):
                z = xp_matmul(hidden, weights[i, ...]) + biases[i, ...]
                hidden = z * xp_sigmoid(z)
        logits = xp_matmul(hidden, parameter(self.adamw_head)) + parameter(
            self.head_bias
        )
        return xp_softmax(
            xp.reshape(logits, (logits.shape[0], self.n_units, self.n_experts))
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the router to a dict."""
        variables = {}
        for name in self.PARAMETER_ATTRIBUTES:
            if getattr(self, name) is not None:
                variables[name] = to_numpy_array(getattr(self, name))
        return {
            "@class": "ConditionRouter",
            "@version": 1,
            "config": {
                "n_units": self.n_units,
                "n_experts": self.n_experts,
                "dim": self.dim,
                "layers": self.layers,
                "chg_spin_dim": self.chg_spin_dim,
                "n_elements": self.n_elements,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": variables,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ConditionRouter:
        """Deserialize a router from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "ConditionRouter":
            raise ValueError(f"Invalid class for ConditionRouter: {data_cls}")
        check_version_compatibility(int(data.pop("@version")), 1, 1)
        obj = cls(**data.pop("config"))
        prec = PRECISION_DICT[obj.precision.lower()]
        for name, value in data.pop("@variables").items():
            setattr(obj, name, np.asarray(value, dtype=prec))
        return obj


class ExpertComponent:
    """
    Mixture of experts (MoE) of a descriptor or a fitting.

    A component owns units of expert maps (see :meth:`_expert_units`), and
    next to them the increment of its task for every map (``task_weight``
    and ``task_bias``), the router of the expert coefficients
    (``expert_router``) and the task embedding that the router reads
    (``adam_task_embedding``). The increments and the embedding are the task
    of the component: replicas that share the maps and the router keep their
    own. They start at zero, as the MoE weights of the maps, so a fresh
    component computes the plain component.

    A component with its ``ntypes`` and ``type_map`` set takes its ``moe``
    option with :meth:`_init_moe`, names its units in :meth:`_expert_units`,
    and builds the task and the router with :meth:`_init_task_and_router`. A
    component whose router reads the charge/spin state of the frames defines
    :meth:`_router_reads_state`, :meth:`_state_embedding` and
    :meth:`_scenario_charge_spin`.
    """

    moe: dict[str, Any] | None
    ntypes: int
    type_map: list[str] | None
    element_slot: list[int] | None
    n_elements: int

    # === Configuration ===

    def _init_moe(self, moe: dict[str, Any] | None, *, charge_spin: bool) -> None:
        """
        Take the ``moe`` option and assign every atom type its element slot.

        Parameters
        ----------
        moe
            The ``moe`` option of the component (see :func:`normalize_moe`).
        charge_spin
            Whether the router reads the charge/spin state of the frames.
        """
        self.moe = normalize_moe(moe, charge_spin=charge_spin)
        # Element slot of every atom type; slot 0 takes no element increment.
        self.element_slot = (
            None
            if self.moe is None
            else element_slots(self.moe["element"], self.ntypes, self.type_map)
        )
        self.n_elements = (
            0
            if self.moe is None
            else n_element_increments(self.moe["element"], self.ntypes)
        )

    def _expert_counts(self) -> list[int] | None:
        """Number of experts and of element increments of every map; None without MoE."""
        return None if self.moe is None else [self.moe["n_experts"], self.n_elements]

    # === Units and maps ===

    def _expert_units(self) -> list[Any]:
        """The units of expert maps, each with router coefficients of its own."""
        raise NotImplementedError

    def _unit_maps(self) -> list[tuple[Any, int, int]]:
        """Every unit with the range of its maps among those of the component."""
        ranges = []
        start = 0
        for unit in self._expert_units():
            stop = start + len(unit.expert_maps())
            ranges.append((unit, start, stop))
            start = stop
        return ranges

    def _expert_maps(self) -> list[Any]:
        """The expert maps of all units, in the order of the task increments."""
        return [layer for unit in self._expert_units() for layer in unit.expert_maps()]

    # === Task and router ===

    def _reset_task_and_router(self) -> None:
        """Give every expert map a zero task increment and remove the router."""
        self.task_weight, self.task_bias = init_task_increments(self._expert_maps())
        self.expert_router: ConditionRouter | None = None
        self.adam_task_embedding = None

    def _init_task_and_router(
        self,
        *,
        chg_spin_dim: int,
        precision: str,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        """
        Build the task increments and, with experts, the router.

        Parameters
        ----------
        chg_spin_dim
            Width of the charge/spin embedding that the router reads; ``0``
            reads none.
        precision
            Precision of the router and of the task embedding.
        seed
            Random seed of the router.
        trainable
            Whether the router is trainable.
        """
        self._reset_task_and_router()
        if self.moe is None or not self.moe["n_experts"]:
            return
        dim = self.moe["router"]["dim"]
        self.expert_router = ConditionRouter(
            n_units=len(self._expert_units()),
            n_experts=self.moe["n_experts"],
            dim=dim,
            layers=self.moe["router"]["layers"],
            chg_spin_dim=chg_spin_dim,
            n_elements=self.n_elements,
            precision=precision,
            seed=seed,
            trainable=trainable,
        )
        self.adam_task_embedding = np.zeros(
            (dim,), dtype=PRECISION_DICT[precision.lower()]
        )

    def has_experts(self) -> bool:
        """Whether the component has MoE (``moe``)."""
        return self.moe is not None

    def has_expert_router(self) -> bool:
        """Whether a router mixes the experts of the component."""
        return self.expert_router is not None

    def expert_options(self) -> dict[str, Any]:
        """The ``moe`` option as the input of this component states it."""
        return {"moe": self.moe}

    def branch_local_parameters(self) -> list[Any]:
        """The task increments and the task embedding, which every branch owns."""
        embedding = (
            [] if self.adam_task_embedding is None else [self.adam_task_embedding]
        )
        return [*self.task_weight, *self.task_bias, *embedding]

    def element_banks(self) -> tuple[list[Any], list[int]]:
        """
        Parameters with one row per element slot, and the slot of every type.

        Row ``z - 1`` of a bank belongs to slot ``z``. A forward reads the
        row of a slot only for the atoms of its types, so the gradient of
        the rows of the types absent from a batch is zero.

        Returns
        -------
        banks : list
            The element increments of every expert map and the element
            embedding of the router; empty without element increments.
        slots : list[int]
            The element slot of every atom type; slot 0 has no row.
        """
        if self.element_slot is None:
            return [], []
        banks = [
            bank
            for layer in self._expert_maps()
            for bank in (layer.element_weight, layer.element_bias)
            if bank is not None
        ]
        router = self.expert_router
        if router is not None and router.adamw_element is not None:
            banks.append(router.adamw_element)
        return banks, list(self.element_slot)

    # === Charge/spin state of the frames ===

    def _router_reads_state(self) -> bool:
        """Whether the router reads the charge/spin state of the frames."""
        return False

    def _state_embedding(self, charge_spin: Array) -> Array:
        """
        Embedding of charge/spin states that the router reads.

        Parameters
        ----------
        charge_spin
            Charge and spin multiplicity with shape (nf, 2).

        Returns
        -------
        Array
            The embedding with shape (nf, chg_spin_dim).
        """
        raise NotImplementedError

    def _scenario_charge_spin(self, scenario: ExpertScenario) -> list[float] | None:
        """
        Charge/spin state at which the router is merged.

        Returns
        -------
        list[float] or None
            The state of the scenario; None for a router that reads none.
        """
        return None

    # === Forward ===

    def _expert_route(
        self,
        node_type: Array,
        condition: ExpertCondition | None,
        n_rows: int | None = None,
    ) -> ExpertRoute | None:
        """
        MoE weights of every row.

        A row takes the increment of the task of the component, the
        increment of the element slot of its type and the experts at the
        coefficients the router gives its element and, for a router that
        reads it, the charge/spin state of its frame. Rows of one element
        slot in frames of one state therefore share their weights. A frame
        that merge-aware training (MAT) selects uses nothing specific to the
        element: its rows take slot 0, in the maps and in the router.

        Parameters
        ----------
        node_type
            Atom type of every row with shape (N,).
        condition
            Conditions of the frames of the rows; required by a router that
            reads the charge/spin state. Without it every row keeps the slot
            of its type.
        n_rows
            Number of leading rows that the route covers; all by default.

        Returns
        -------
        ExpertRoute or None
            The route of the rows for all units; None without MoE.

        Raises
        ------
        ValueError
            If the router reads the state of the frames and no conditions
            are given.
        """
        if self.moe is None:
            return None
        reads_state = self._router_reads_state()
        if reads_state and condition is None:
            raise ValueError(
                "the router of this component reads the conditions of the "
                "frames, which the atomic model supplies on the neighbor-graph "
                "path only; evaluate the model on a neighbor graph, or merge the "
                "model for one scenario first"
            )
        xp = array_api_compat.array_namespace(node_type)
        device = array_api_compat.device(node_type)
        if n_rows is None:
            n_rows = node_type.shape[0]

        # === Step 1. Rows that share their weights ===
        keep = (
            None if condition is None else condition.element_keep(self.moe["mat_prob"])
        )
        slot = node_slots(
            node_type[:n_rows],
            self.element_slot,
            None if keep is None else keep[:n_rows],
        )
        state = state_frame = None
        n_states = 1
        if reads_state:
            frame_state, state_frame = frame_states(condition.charge_spin)
            state = xp.take(frame_state, condition.node_frame[:n_rows], axis=0)
            n_states = condition.charge_spin.shape[0]
        layout, chunk_slot, chunk_state = layout_rows(
            slot, 1 + self.n_elements, state, n_states
        )

        # === Step 2. Coefficients of the routed experts ===
        coeff = None
        if self.expert_router is not None:
            coeff = self.expert_router(
                xp_asarray_nodetach(xp, self.adam_task_embedding[...], device=device),
                xp.take(state_frame, chunk_state, axis=0) if reads_state else None,
                chunk_slot,
                self._state_embedding(condition.charge_spin) if reads_state else None,
            )

        # === Step 3. Increments of the task ===
        task = pair_task_increments(
            self._expert_maps(),
            [xp_asarray_nodetach(xp, w[...], device=device) for w in self.task_weight],
            [xp_asarray_nodetach(xp, b[...], device=device) for b in self.task_bias],
        )
        return ExpertRoute(task, layout, chunk_slot, coeff)

    def _unit_routes(
        self, route: ExpertRoute, first: int = 0, stop: int | None = None
    ) -> list[UnitRoute]:
        """
        Routes of the units ``first`` to ``stop`` of the component.

        Parameters
        ----------
        route
            Route of the rows that the units evaluate.
        first, stop
            Range of the units in the order of :meth:`_expert_units`; all
            units by default.

        Returns
        -------
        list[UnitRoute]
            The route of every unit of the range, with the weights of its
            maps.
        """
        return [
            route.unit(index, unit.expert_maps(), start)
            for index, (unit, start, _) in enumerate(self._unit_maps())
            if index >= first and (stop is None or index < stop)
        ]

    # === Merge ===

    def merge_experts(self, scenario: ExpertScenario) -> Any:
        """
        Component that this component computes in one scenario.

        The increments of the task and the experts at the coefficients of
        the scenario are added to the shared weights, and the router and the
        task embedding are dropped. Without kept elements the result is the
        plain component, which every atom evaluates with the weights shared
        by the elements. An element kept by the scenario retains an
        increment, with which its atoms use the weights they have in this
        component, so the result computes what this component computes on
        systems of the kept elements; it then has the expert configuration
        ``{"n_experts": 0, "element": [...]}``.

        Parameters
        ----------
        scenario
            The charge/spin state of the frames, for a router that reads it,
            and the elements that keep an increment.

        Returns
        -------
        Any
            The merged component; this component itself without MoE.

        Raises
        ------
        ValueError
            If the charge/spin state of the scenario cannot be merged (see
            :meth:`_scenario_charge_spin`), or its elements cannot be kept or
            dropped (see :func:`merged_elements`).
        """
        if self.moe is None:
            return self
        kept, keep = merged_elements(scenario, self.moe, self.type_map)

        # === Step 1. Coefficients of every element slot in the scenario ===
        charge_spin = self._scenario_charge_spin(scenario)
        coeff = None
        if self.expert_router is not None:
            n_slots = 1 + self.n_elements
            coeff = to_numpy_array(
                self.expert_router(
                    to_numpy_array(self.adam_task_embedding),
                    np.zeros((n_slots,), dtype=np.int64),
                    np.arange(n_slots, dtype=np.int64),
                    None
                    if charge_spin is None
                    else self._state_embedding(np.asarray([charge_spin])),
                )
            )  # (n_slots, n_units, n_experts)

        # === Step 2. Merge the maps of every unit ===
        merged = copy.deepcopy(self)
        task = pair_task_increments(
            self._expert_maps(), self.task_weight, self.task_bias
        )
        for index, (unit, start, stop) in enumerate(merged._unit_maps()):
            unit.merge_experts(
                task[start:stop],
                None if coeff is None else coeff[:, index, :],
                keep,
            )

        # === Step 3. Rebuild the component of the merged configuration ===
        merged.moe = merged_moe(self.moe, kept)
        merged._reset_task_and_router()
        merged._finish_merge(charge_spin)
        return type(self).deserialize(merged.serialize())

    def _finish_merge(self, charge_spin: list[float] | None) -> None:
        """
        Complete the state of a merged component before it is rebuilt.

        Parameters
        ----------
        charge_spin
            The charge/spin state at which the router was merged, or None.
        """

    # === Serialization ===

    def _task_and_router_variables(self) -> dict[str, Any]:
        """
        The task increments, the task embedding and the router as variables.

        Returns
        -------
        dict
            The increments under the keys of
            :func:`~deepmd.dpmodel.utils.expert_linear.task_variables`, the
            embedding under ``adam_task_embedding`` and the variables of the
            router under the prefix ``expert_router.``; empty without MoE.
        """
        variables = task_variables(self.task_weight, self.task_bias)
        if self.expert_router is not None:
            for key, value in self.expert_router.serialize()["@variables"].items():
                variables[f"expert_router.{key}"] = value
            variables["adam_task_embedding"] = to_numpy_array(self.adam_task_embedding)
        return variables

    def _load_task_and_router(self, variables: dict[str, Any]) -> dict[str, Any]:
        """
        Load the task increments, the task embedding and the router.

        Parameters
        ----------
        variables
            Variables of a record of the component, which hold those of
            :meth:`_task_and_router_variables`.

        Returns
        -------
        dict
            The variables that are not part of the task and the router.
        """
        own = set(self._task_and_router_variables())
        self.task_weight, self.task_bias = load_task_variables(
            self._expert_maps(), variables
        )
        if self.expert_router is not None:
            prefix = "expert_router."
            record = self.expert_router.serialize()
            record["@variables"] = {
                key[len(prefix) :]: value
                for key, value in variables.items()
                if key.startswith(prefix)
            }
            self.expert_router = type(self.expert_router).deserialize(record)
            self.adam_task_embedding = np.asarray(
                variables["adam_task_embedding"],
                dtype=PRECISION_DICT[self.expert_router.precision.lower()],
            )
        return {key: value for key, value in variables.items() if key not in own}


def frame_type_counts(
    atype: Array,
    node_mask: Array,
    node_frame: Array,
    nframes: int,
    ntypes: int,
) -> Array:
    """Number of counted nodes of every frame and type.

    Parameters
    ----------
    atype : Array
        Node types with shape (N,), clamped to valid types.
    node_mask : Array
        Mask of the counted nodes with shape (N,).
    node_frame : Array
        Frame of every node with shape (N,).
    nframes : int
        Number of frames.
    ntypes : int
        Number of atom types.

    Returns
    -------
    Array
        The counts with shape (nframes, ntypes).
    """
    xp = array_api_compat.array_namespace(atype, node_mask)
    counts = xp_add_at(
        xp.zeros(
            (nframes * ntypes,), dtype=xp.int64, device=array_api_compat.device(atype)
        ),
        node_frame * ntypes + atype,
        xp.astype(node_mask, xp.int64),
    )
    return xp.reshape(counts, (nframes, ntypes))


def reference_keys(counts: Array) -> tuple[Array, Array]:
    """Enumerate the (frame, type) pairs present in the counts.

    Parameters
    ----------
    counts : Array
        Node counts of every frame and type with shape (nframes, ntypes).

    Returns
    -------
    keys : Array
        Flat keys ``frame * ntypes + type`` of the present pairs, ascending,
        with a data-dependent length R.
    rank : Array
        Position in ``keys`` of every flat key with shape (nframes * ntypes,);
        meaningful at present keys only.
    """
    xp = array_api_compat.array_namespace(counts)
    present = xp.reshape(counts > 0, (-1,))
    keys = xp.nonzero(present)[0]
    xp_hint_dynamic_size(keys)
    rank = xp.cumulative_sum(xp.astype(present, xp.int64)) - 1
    return keys, rank


def node_reference_index(
    atype: Array,
    node_frame: Array,
    rank: Array,
    ntypes: int,
) -> Array:
    """Index of the reference pair of every node.

    Parameters
    ----------
    atype : Array
        Node types with shape (N,), clamped to valid types.
    node_frame : Array
        Frame of every node with shape (N,).
    rank : Array
        Rank of every flat key, from :func:`reference_keys`.
    ntypes : int
        Number of atom types.

    Returns
    -------
    Array
        Index into the reference pairs with shape (N,). Nodes that are not
        counted read a valid index whose value carries no meaning.
    """
    xp = array_api_compat.array_namespace(atype, rank)
    index = xp.take(rank, node_frame * ntypes + atype, axis=0)
    return xp.maximum(index, xp.zeros_like(index))
