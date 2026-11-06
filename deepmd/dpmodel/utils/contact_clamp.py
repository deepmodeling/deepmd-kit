# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Contact clamp of a bridged model, defined once for every backend.

A bridged model adds an analytical pair repulsion to a learned energy. The
learned part is fitted only at the separations its training frames contain,
and below the shortest of them it extrapolates without any constraint. The
clamp records, for every pair of types, the data edge :math:`e_{ab}`, the
:math:`K`-th smallest of the shortest separations that the training frames
hold for the pair, and maps the separation the learned part reads to

.. math::

   \tilde r=\begin{cases}r, & r\ge e,\\[2pt]
   \dfrac{1+a}{2}\,e+(1-a)\,e\,S(t), & r<e,\end{cases}\qquad
   t=\operatorname{clip}\!\left(\frac{r-a\,e}{(1-a)\,e},\,0,\,1\right),

with :math:`S(t)=7t^5-14t^6+10t^7-\tfrac52t^8`, the antiderivative of the
septic smootherstep :math:`h(t)=35t^4-84t^5+70t^6-20t^7`. Hence
:math:`d\tilde r/dr=h(t)`: one at the edge, zero at and below :math:`a\,e`, and
never above one in between, so :math:`\tilde r` is a non-decreasing C⁴ function
of :math:`r` that is frozen at :math:`(1+a)\,e/2` for every closer pair. Below
the edge the learned part therefore only ever reads separations between
:math:`(1+a)\,e/2` and :math:`e`, and the analytical term alone carries the
repulsion of a closer pair. The directions of the bonds are read unchanged.

The edges follow from a record of contacts kept over every training batch,
updated before the batch is evaluated: for every pair, the :math:`K` smallest
distinct values of the shortest separation the pair has in a frame. The edge
is the :math:`K`-th of them, capped at the covalent bond length of the two
elements, which is also the edge of a pair that fewer than :math:`K` frames
hold below it. With :math:`K = 1` the edge is the shortest contact of any
frame and the clamp is the identity on every training contact. With a larger
:math:`K` a single unusual frame cannot move the edge into a range that no
other frame covers, where the learned part would be fitted to that one frame
alone; the contacts of the fewer frames below an edge are then read through
the clamp. Contacts in the same bin of width :data:`CONTACT_RESOLUTION` count
as one, so a frame seen again in a later epoch counts once when its
separations are recomputed with a rounding difference, except where that
difference carries a contact across a bin boundary.
"""

from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
    xp_asarray_nodetach,
)
from deepmd.dpmodel.common import (
    NativeOP,
    to_numpy_array,
)
from deepmd.dpmodel.utils.neighbor_graph.segment import (
    segment_max,
)
from deepmd.utils.bridging import (
    DEFAULT_SUPPORT,
)
from deepmd.utils.element_radii import (
    covalent_radii_from_type_map,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

#: Edge of every pair that involves the padding type, in Å. Such edges are
#: masked wherever they occur; a positive value keeps their profile finite.
PADDING_EDGE = 1.0

#: Width in Å of the bins in which recorded contacts of a pair count as one.
#: It lies far above the rounding of a recomputed separation and far below
#: the spacing of the contacts of different frames.
CONTACT_RESOLUTION = 1e-6


def _sorted_by_row(xp: Any, rows: Array, values: Array) -> tuple[Array, Array]:
    """Order entries by row and, within a row, by ascending value."""
    order = xp.argsort(values, stable=True)
    rows, values = xp.take(rows, order), xp.take(values, order)
    order = xp.argsort(rows, stable=True)
    return xp.take(rows, order), xp.take(values, order)


def _smallest_distinct(
    xp: Any, rows: Array, values: Array, n_rows: int, count: int
) -> Array:
    """The ``count`` smallest distinct values of every row, padded with +inf.

    A value that falls in the same bin of width :data:`CONTACT_RESOLUTION` as a
    smaller value of its row repeats it, so the result depends on the set of
    values of a row alone: not on their order, on how they are split into
    batches, or on a contact that is observed twice.

    Parameters
    ----------
    xp : Any
        Array namespace of the inputs.
    rows : Array
        Row of every entry with shape (M,) in int64; every row in
        ``[0, n_rows)`` holds at least ``count`` entries.
    values : Array
        Value of every entry with shape (M,) in float64.
    n_rows : int
        Number of rows.
    count : int
        Number of values kept per row.

    Returns
    -------
    Array
        The kept values with shape ``(n_rows, count)``, ascending in a row.
    """
    device = array_api_compat.device(values)
    rows, values = _sorted_by_row(xp, rows, values)
    # The entries of a row ascend, so those of one bin are adjacent and the
    # smallest of them is the one kept; an unfilled +inf slot after another one
    # stays +inf.
    bins = xp.floor(values / CONTACT_RESOLUTION)
    repeated = (rows[1:] == rows[:-1]) & (bins[1:] == bins[:-1])
    repeated = xp.concat([xp.zeros((1,), dtype=xp.bool, device=device), repeated])
    values = xp.where(repeated, xp.full_like(values, xp.inf), values)
    rows, values = _sorted_by_row(xp, rows, values)
    first = xp.searchsorted(rows, xp.arange(n_rows, dtype=rows.dtype, device=device))
    slots = first[:, None] + xp.arange(count, dtype=first.dtype, device=device)
    return xp.reshape(xp.take(values, xp.reshape(slots, (-1,))), (n_rows, count))


class ContactClamp(NativeOP):
    r"""Freeze the pair separations a learned model reads below its data edges.

    Parameters
    ----------
    type_map : list[str]
        Element symbols of the types; the index is the ``atype`` value.
    inner : float
        Fraction :math:`a` of the data edge at and below which the read
        separation is frozen, with ``0 < inner < 1``.
    support : int
        Rank :math:`K`, counted from the smallest, of the distinct per-frame
        shortest separation of a pair that becomes its data edge, at least one.

    Raises
    ------
    ValueError
        If ``inner`` lies outside ``(0, 1)``, ``support`` is below one or a
        type is not an element.
    """

    # The covalent start is rebuilt from the type map, never stored.
    CONFIG_DERIVED_ARRAYS = ("start",)

    def __init__(
        self, type_map: list[str], inner: float = 0.75, support: int = DEFAULT_SUPPORT
    ) -> None:
        super().__init__()
        inner = float(inner)
        if not 0.0 < inner < 1.0:
            raise ValueError(f"`inner` must lie strictly between 0 and 1, got {inner}")
        support = int(support)
        if support < 1:
            raise ValueError(f"`support` must be at least 1, got {support}")
        self.inner = inner
        self.support = support
        self.type_map = list(type_map)
        self.ntypes = len(self.type_map)
        start = self.initial_edges(self.type_map)
        self.start = start
        self.edges = start.copy()
        self.contacts = np.full((*start.shape, support), np.inf, dtype=np.float64)

    def check_cutoff(self, rcut: float) -> None:
        """Require every initial contact edge to lie inside the cutoff in Å."""
        largest = 2.0 * float(np.max(covalent_radii_from_type_map(self.type_map)))
        if largest >= rcut:
            raise ValueError(
                f"The largest contact edge is {largest:.3f} Å, at or beyond "
                f"the smallest block cutoff {rcut} Å; enlarge `rcut`."
            )

    @staticmethod
    def initial_edges(type_map: list[str]) -> np.ndarray:
        """Tabulate the edge every pair of types starts from.

        Parameters
        ----------
        type_map : list[str]
            Element symbols of the types.

        Returns
        -------
        np.ndarray
            Edges with shape ``(T + 1, T + 1)`` in Å, in float64: the covalent
            bond length of every pair of real types, :data:`PADDING_EDGE` on the
            row and column of the padding type ``T``.
        """
        radius = covalent_radii_from_type_map(type_map)
        ntypes = radius.shape[0]
        edges = np.full((ntypes + 1, ntypes + 1), PADDING_EDGE, dtype=np.float64)
        edges[:ntypes, :ntypes] = radius[:, None] + radius[None, :]
        return edges

    def pair_index(self, center_type: Array, neighbor_type: Array) -> Array:
        """Flat index ``center * (T + 1) + neighbor`` of every edge's type pair.

        Types outside ``[0, T)``, the padding conventions of the different
        neighbor graphs, address the padding row.

        Parameters
        ----------
        center_type : Array
            Type of the center atom of every edge, with shape (E,).
        neighbor_type : Array
            Type of the neighbor atom of every edge, with shape (E,).

        Returns
        -------
        Array
            Pair indices with shape (E,) in int64.
        """
        xp = array_api_compat.array_namespace(center_type)
        stride = self.ntypes + 1

        def row(types: Array) -> Array:
            types = xp.astype(types, xp.int64)
            valid = (types >= 0) & (types < self.ntypes)
            return xp.where(valid, types, xp.full_like(types, self.ntypes))

        return row(center_type) * stride + row(neighbor_type)

    def call(self, distance: Array, center_type: Array, neighbor_type: Array) -> Array:
        """Map pair separations to the separations the learned part reads.

        Parameters
        ----------
        distance : Array
            Pair separations with shape (E,) or (E, 1) in Å.
        center_type : Array
            Type of the center atom of every edge, with shape (E,).
        neighbor_type : Array
            Type of the neighbor atom of every edge, with shape (E,).

        Returns
        -------
        Array
            Read separations with the shape and dtype of ``distance``.
        """
        xp = array_api_compat.array_namespace(distance)
        table = xp_asarray_nodetach(
            xp,
            self.edges,
            dtype=distance.dtype,
            device=array_api_compat.device(distance),
        )
        edge = xp.take(
            xp.reshape(table, (-1,)), self.pair_index(center_type, neighbor_type)
        )
        edge = xp.reshape(edge, distance.shape)
        width = (1.0 - self.inner) * edge
        t = xp.clip((distance - self.inner * edge) / width, 0.0, 1.0)
        t2 = t * t
        # S(t) = 7 t^5 - 14 t^6 + 10 t^7 - 2.5 t^8 in Horner form, S(1) = 1/2.
        rise = t2 * t2 * t * (7.0 + t * (-14.0 + t * (10.0 - 2.5 * t)))
        frozen = 0.5 * (1.0 + self.inner) * edge + width * rise
        # Both branches agree in value and in four derivatives at the edge, so
        # the selection keeps the read separation C⁴ there.
        return xp.where(distance >= edge, distance, frozen)

    def edges_of(self, contacts: Array) -> Array:
        """Data edges of a contact record.

        Parameters
        ----------
        contacts : Array
            Record with shape ``(T + 1, T + 1, K)`` in Å, in float64.

        Returns
        -------
        Array
            Edges with shape ``(T + 1, T + 1)`` in Å: the ``K``-th recorded
            contact of every pair, capped at its covalent start and floored at
            :data:`CONTACT_RESOLUTION`, so that the ramp of a coincident contact
            keeps a positive width and the profile a finite derivative.
        """
        xp = array_api_compat.array_namespace(contacts)
        start = xp_asarray_nodetach(
            xp, self.start, dtype=xp.float64, device=array_api_compat.device(contacts)
        )
        edge = xp.minimum(start, contacts[..., self.support - 1])
        return xp.clip(edge, CONTACT_RESOLUTION)

    def merged(self, records: Array) -> Array:
        """Merge contact records, such as those of the ranks of a training.

        Parameters
        ----------
        records : Array
            Records with shape ``(R, T + 1, T + 1, K)`` in Å, in float64.

        Returns
        -------
        Array
            The record of all of them with shape ``(T + 1, T + 1, K)``: the
            ``K`` smallest distinct contacts of every pair.
        """
        xp = array_api_compat.array_namespace(records)
        stride = self.ntypes + 1
        values = xp.reshape(records, (-1,))
        position = xp.arange(
            values.shape[0], dtype=xp.int64, device=array_api_compat.device(values)
        )
        rows = (position // self.support) % (stride * stride)
        merged = _smallest_distinct(xp, rows, values, stride * stride, self.support)
        return xp.reshape(merged, (stride, stride, self.support))

    def observed(
        self,
        distance: Array,
        center_type: Array,
        neighbor_type: Array,
        mask: Array,
        frame: Array,
    ) -> Array:
        """Return the contact record extended by the contacts of a batch.

        Parameters
        ----------
        distance : Array
            Pair separations of the batch with shape (E,) or (E, 1) in Å.
        center_type : Array
            Type of the center atom of every edge, with shape (E,).
        neighbor_type : Array
            Type of the neighbor atom of every edge, with shape (E,).
        mask : Array
            Whether each edge is a real pair, with shape (E,).
        frame : Array
            Training frame of every edge with shape (E,); frames are told apart
            within the batch only.

        Returns
        -------
        Array
            Record with shape ``(T + 1, T + 1, K)`` in Å, in float64 and in the
            namespace of ``distance``, symmetric in the two types.
        """
        xp = array_api_compat.array_namespace(distance)
        device = array_api_compat.device(distance)
        stride = self.ntypes + 1
        n_pair = stride * stride
        pair = self.pair_index(center_type, neighbor_type)
        start = xp.reshape(
            xp_asarray_nodetach(xp, self.start, dtype=xp.float64, device=device),
            (-1,),
        )
        separation = xp.astype(xp.reshape(distance, (-1,)), xp.float64)
        # Pairs of the padding type keep their fixed edge, and a contact at or
        # beyond the covalent start never sets an edge.
        keep = (
            xp.astype(xp.reshape(mask, (-1,)), xp.bool)
            & (pair // stride < self.ntypes)
            & (pair % stride < self.ntypes)
            & (separation < xp.take(start, pair))
        )
        index = xp.nonzero(keep)[0]
        pair = xp.take(pair, index)
        separation = xp.take(separation, index)
        frame = xp.astype(xp.take(xp.reshape(frame, (-1,)), index), xp.int64)
        # Both orders of every pair, so the record is symmetric for a neighbor
        # list of either kind.
        pair = xp.concat([pair, (pair % stride) * stride + pair // stride])
        separation = xp.concat([separation, separation])
        frame = xp.concat([frame, frame])
        # The shortest separation of every pair in every frame.
        frame_pair, inverse = xp.unique_inverse(frame * n_pair + pair)
        shortest = -segment_max(-separation, inverse, frame_pair.shape[0])
        record = xp.reshape(
            xp_asarray_nodetach(xp, self.contacts, dtype=xp.float64, device=device),
            (-1,),
        )
        position = xp.arange(record.shape[0], dtype=xp.int64, device=device)
        rows = xp.concat([position // self.support, frame_pair % n_pair])
        values = xp.concat([record, shortest])
        contacts = _smallest_distinct(xp, rows, values, n_pair, self.support)
        return xp.reshape(contacts, (stride, stride, self.support))

    def observe(
        self,
        distance: Array,
        center_type: Array,
        neighbor_type: Array,
        mask: Array,
        frame: Array,
    ) -> None:
        """Record the contacts of a training batch and update the edges.

        Parameters
        ----------
        distance : Array
            Pair separations of the batch with shape (E,) or (E, 1) in Å.
        center_type : Array
            Type of the center atom of every edge, with shape (E,).
        neighbor_type : Array
            Type of the neighbor atom of every edge, with shape (E,).
        mask : Array
            Whether each edge is a real pair, with shape (E,).
        frame : Array
            Training frame of every edge with shape (E,).
        """
        contacts = self.observed(distance, center_type, neighbor_type, mask, frame)
        self.contacts = to_numpy_array(contacts)
        self.edges = to_numpy_array(self.edges_of(contacts))

    def observe_edges(
        self,
        edge_vec: Array,
        edge_index: Array,
        atype: Array,
        edge_mask: Array,
        node_frame: Array,
    ) -> None:
        """Record the contacts of a training graph and update the edges.

        Parameters
        ----------
        edge_vec : Array
            Edge vectors with shape (E, 3) in Å.
        edge_index : Array
            ``[source, destination]`` node indices with shape (2, E); the
            destination is the center of the edge.
        atype : Array
            Types of the nodes with shape (N,).
        edge_mask : Array
            Whether each edge is a real pair, with shape (E,).
        node_frame : Array
            Training frame of every node with shape (N,).
        """
        xp = array_api_compat.array_namespace(edge_vec)
        distance = xp.sqrt(xp.sum(edge_vec * edge_vec, axis=-1))
        self.observe(
            distance,
            xp.take(atype, edge_index[1, :], axis=0),
            xp.take(atype, edge_index[0, :], axis=0),
            edge_mask,
            xp.take(node_frame, edge_index[1, :], axis=0),
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the clamp, including its contact record and edges."""
        return {
            "@class": "ContactClamp",
            "@version": 2,
            "type_map": self.type_map,
            "inner": self.inner,
            "support": self.support,
            "@variables": {
                "edges": to_numpy_array(self.edges).astype(np.float64),
                "contacts": to_numpy_array(self.contacts).astype(np.float64),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "ContactClamp":
        """Rebuild a clamp from :meth:`serialize` output.

        Parameters
        ----------
        data : dict
            The serialized clamp.

        Returns
        -------
        ContactClamp
            The clamp with its contact record and edges.
        """
        data = dict(data)
        check_version_compatibility(data.pop("@version", 1), 2, 2)
        data.pop("@class", None)
        variables = data.pop("@variables")
        clamp = cls(**data)
        clamp.contacts = np.asarray(variables["contacts"], dtype=np.float64)
        clamp.edges = np.asarray(variables["edges"], dtype=np.float64)
        return clamp
