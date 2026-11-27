# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Linear maps of a mixture of experts (MoE).

An expert map is a linear map whose weight depends on the row it acts on,

.. math::

    W + V_\mathrm{task} + U_{z} + \sum_k c_k A_k ,

with the shared weight :math:`W` of the plain map, the increment
:math:`V_\mathrm{task}` of the task of the model, the increment :math:`U_z`
of the element slot :math:`z` of the row and experts :math:`A_k` mixed by the
coefficients :math:`c` of a router (see
:mod:`deepmd.dpmodel.utils.expert_route`). A bias carries the same terms.
Every term has the layout of the weight of the plain map, so the map keeps
the symmetry of the plain map for any conditions, and all terms start at
zero, so a fresh expert map computes the plain map.

The task increment belongs to the model rather than to the map: maps shared
by the branches of a multi-task model receive the increment of the branch
with the route of a forward. :class:`ExpertWeights` holds the mechanics, which
do not depend on the map; a map defines its expert variant next to its plain
class and registers it in :data:`EXPERT_VARIANTS`. :class:`ExpertContainer`
is the interface of a module that owns expert maps.

Rows that share their conditions share their weights, so a forward lays the
rows out in chunks of one set of weights each (see
:class:`~deepmd.dpmodel.utils.expert_route.RowLayout`), forms the weights of
every chunk once for all maps of a unit (:func:`chunk_weights`) and contracts
the chunks of a map with their weights in one batched product.
"""

from __future__ import (
    annotations,
)

import dataclasses
import math
from typing import (
    TYPE_CHECKING,
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
    xp_matmul,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )

#: Attributes that only a map with element increments or experts holds.
MAP_EXPERT_ATTRS: tuple[str, ...] = (
    "element_weight",
    "element_bias",
    "expert_weight",
    "expert_bias",
)

#: Leaf names of every parameter that only MoE holds: the attributes of the
#: maps above and the task of a component. A plain model therefore keeps every
#: key of the same model with MoE, and the parameters MoE adds are exactly the
#: keys whose leaf carries one of these names.
EXPERT_PARAMETER_NAMES: tuple[str, ...] = (
    "task_weight",
    "task_bias",
    *MAP_EXPERT_ATTRS,
    "adam_task_embedding",
    "expert_router",
)


def is_expert_parameter(name: str) -> bool:
    """Whether a parameter name carries a term that only MoE holds.

    The name of a term is the leaf of its parameter, except for the router,
    whose own parameters carry names of their own under the segment
    ``expert_router``.

    Parameters
    ----------
    name
        Parameter name, relative to the model or to a component of it.

    Returns
    -------
    bool
        Whether a segment of the name is one of
        :data:`EXPERT_PARAMETER_NAMES`.
    """
    return any(segment in EXPERT_PARAMETER_NAMES for segment in name.lower().split("."))


@dataclasses.dataclass(frozen=True)
class ChunkWeights:
    """Weight and bias of one expert map for every chunk of rows.

    Attributes
    ----------
    weight : Array
        Weight of every chunk with shape (n_chunks, *weight shape).
    bias : Array or None
        Bias of every chunk with shape (n_chunks, *bias shape); None for a map
        without bias.
    chunk : int or None
        Number of rows of a chunk; None when all rows form the single chunk.
    """

    weight: Any
    bias: Any
    chunk: int | None


def chunk_weights(
    maps: Sequence[ExpertWeights],
    task: Sequence[tuple[Any, Any]],
    chunk: int | None,
    chunk_slot: Any = None,
    chunk_coeff: Any = None,
) -> list[ChunkWeights]:
    """
    Weights of the expert maps of one unit for every chunk of rows.

    The shared weights, the task increments and the element increments of all
    maps are joined into one table with a row per element slot, so the terms
    of a slot are summed once for the unit; every map then takes the rows of
    its chunks and adds its experts at the coefficients of the chunks.

    Parameters
    ----------
    maps
        The expert maps of the unit in their order.
    task
        Task increment ``(weight, bias)`` of every map; the bias is None for
        a map without one.
    chunk
        Number of rows of a chunk; None when all rows form the single chunk.
    chunk_slot
        Element slot of every chunk with shape (n_chunks,); None without
        element increments.
    chunk_coeff
        Coefficients of the experts of every chunk with shape (n_chunks, K);
        None without a router.

    Returns
    -------
    list[ChunkWeights]
        The weights of every map.
    """
    xp = array_api_compat.array_namespace(task[0][0])
    device = array_api_compat.device(task[0][0])

    def parameter(value: Any) -> Any:
        return xp_asarray_nodetach(xp, value[...], device=device)

    def joined(values: list[Any], leading: int) -> Any:
        """Arrays flattened behind their leading axes and joined along the last."""
        return xp.concat(
            [xp.reshape(v, (*v.shape[:leading], -1)) for v in values], axis=-1
        )

    # === Step 1. Terms of every weight and bias, in the order of the maps ===
    terms = []
    for layer, (task_weight, task_bias) in zip(maps, task, strict=True):
        terms.append(
            (
                parameter(layer._plain_weight()),
                task_weight,
                layer.element_weight,
                layer.expert_weight,
            )
        )
        if layer.has_bias():
            terms.append(
                (
                    parameter(layer._plain_bias()),
                    task_bias,
                    layer.element_bias,
                    layer.expert_bias,
                )
            )

    # === Step 2. Shared, task and element terms of every element slot ===
    table = (joined([t[0] for t in terms], 0) + joined([t[1] for t in terms], 0))[
        None, :
    ]
    by_slot = chunk_slot is not None and terms[0][2] is not None
    if by_slot:
        # Slot 0 adds no increment; the other slots take their own.
        own = joined([parameter(t[2]) for t in terms], 1)
        table = table + xp.concat([xp.zeros_like(own[:1, :]), own], axis=0)

    # === Step 3. Terms of every chunk ===
    values = []
    offset = 0
    for shared, _, _, experts in terms:
        size = math.prod(shared.shape)
        value = table[:, offset : offset + size]
        offset += size
        if by_slot:
            value = xp.take(value, chunk_slot, axis=0)
        if experts is not None:
            bank = xp.reshape(parameter(experts), (experts.shape[0], size))
            value = value + xp_matmul(xp.astype(chunk_coeff, bank.dtype), bank)
        if value.dtype != shared.dtype:
            value = xp.astype(value, shared.dtype)
        values.append(xp.reshape(value, (value.shape[0], *shared.shape)))

    remaining = iter(values)
    return [
        ChunkWeights(
            next(remaining), next(remaining) if layer.has_bias() else None, chunk
        )
        for layer in maps
    ]


class ExpertWeights:
    """
    MoE weights of a linear map; mixed into the class of the plain map.

    The map holds the element increments ``element_weight`` with shape
    (n_elements, *weight shape) and the experts ``expert_weight`` with shape
    (n_experts, *weight shape), and their biases when the plain map has one.
    The route of a forward supplies the task increment and, for every chunk
    of rows that share their weights, the element slot and the coefficients
    of the experts. Slot 0 takes no element increment; slot ``z > 0`` takes
    ``element_weight[z - 1]``.

    The weight ``W + V + U_z + sum_k c_k A_k`` of every chunk is formed
    once (:func:`chunk_weights`), and the chunks are contracted with their
    weights in one batched product, so the cost of the contraction depends
    neither on the number of elements nor on the number of experts.

    A subclass names its plain class ``PLAIN``. A map whose weight does not
    act on the last axis alone overrides :meth:`_chunk_affine`, the affine
    map of chunks of rows with one weight and one bias each.

    Parameters
    ----------
    n_experts
        Number of experts ``K`` mixed by the router; ``0`` holds none.
    n_elements
        Number of element increments; ``0`` holds none.
    **kwargs
        Arguments of the plain map.
    """

    PLAIN: type
    precision: str

    def __init__(
        self, *, n_experts: int = 0, n_elements: int = 0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._init_experts(n_experts, n_elements)

    def _init_experts(self, n_experts: int, n_elements: int) -> None:
        """
        Allocate the element increments and the experts of the map, all zero.

        Parameters
        ----------
        n_experts
            Number of experts ``K`` mixed by the router; ``0`` holds none.
        n_elements
            Number of element increments; ``0`` holds none.
        """
        self.n_experts = int(n_experts)
        self.n_elements = int(n_elements)
        if self.n_experts < 0 or self.n_elements < 0:
            raise ValueError(
                "`n_experts` and `n_elements` are non-negative; got "
                f"{self.n_experts} and {self.n_elements}"
            )
        prec = PRECISION_DICT[self.precision.lower()]
        weight_shape, bias_shape = self.task_shapes()

        def zeros(count: int, shape: tuple[int, ...] | None) -> np.ndarray | None:
            if not count or shape is None:
                return None
            return np.zeros((count, *shape), dtype=prec)

        self.element_weight = zeros(self.n_elements, weight_shape)
        self.element_bias = zeros(self.n_elements, bias_shape)
        self.expert_weight = zeros(self.n_experts, weight_shape)
        self.expert_bias = zeros(self.n_experts, bias_shape)

    # === Plain map ===

    def _plain_weight(self) -> Any:
        """Weight ``W`` of the plain map."""
        return self.weight

    def _plain_bias(self) -> Any:
        """Bias of the plain map, or None."""
        return self.bias

    def has_bias(self) -> bool:
        """Whether the map adds a bias."""
        return self._plain_bias() is not None

    def task_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        """Shapes of the weight and of the bias, which every increment takes."""
        bias = self._plain_bias()
        return (
            tuple(self._plain_weight().shape),
            None if bias is None else tuple(bias.shape),
        )

    # === Forward ===

    def call(self, x: Any, weights: ChunkWeights) -> Any:
        """
        Apply the map with the weight of every chunk of rows.

        Parameters
        ----------
        x
            Input of the plain map, whose rows are those of the layout that
            the unit owning the map has entered.
        weights
            Weight and bias of this map for every chunk.

        Returns
        -------
        Array
            Output of the plain map with the weight of every row.
        """
        return self._affine(x, weights)

    def _affine(self, x: Any, weights: ChunkWeights) -> Any:
        """
        Apply the map with the weight of every chunk of rows.

        Parameters
        ----------
        x
            Input of the plain map, whose rows are those of the layout that
            the unit owning the map has entered.
        weights
            Weight and bias of this map for every chunk.

        Returns
        -------
        Array
            Output of the map with the weight of every row.
        """
        xp = array_api_compat.array_namespace(x)
        n_chunks = weights.weight.shape[0]
        if weights.chunk is None:
            chunks = x[None, ...]
        else:
            chunks = xp.reshape(x, (n_chunks, weights.chunk, *x.shape[1:]))
        out = self._chunk_affine(chunks, weights.weight, weights.bias)
        if weights.chunk is None:
            return out[0, ...]
        return xp.reshape(out, (n_chunks * weights.chunk, *out.shape[2:]))

    def _chunk_affine(self, x: Any, weight: Any, bias: Any) -> Any:
        """
        Apply one weight and one bias to every chunk of rows.

        Parameters
        ----------
        x
            Chunks of rows with shape (n_chunks, m, ..., C).
        weight
            Weight of every chunk with shape (n_chunks, C, Cout).
        bias
            Bias of every chunk with shape (n_chunks, Cout), or None.

        Returns
        -------
        Array
            Chunks of rows with shape (n_chunks, m, ..., Cout).
        """
        xp = array_api_compat.array_namespace(x)
        out = xp_matmul(xp.reshape(x, (x.shape[0], -1, x.shape[-1])), weight)
        if bias is not None:
            out = out + bias[:, None, :]
        return xp.reshape(out, (*x.shape[:-1], weight.shape[-1]))

    # === Merge ===

    def merge(
        self,
        task: tuple[np.ndarray, np.ndarray | None],
        coeff: np.ndarray | None,
        keep: Sequence[int],
    ) -> NativeOP:
        """
        Map that this map computes for one task and fixed coefficients.

        The task increment and the experts are added to the shared weight,
        ``W' = W + V + sum_k c_k(0) A_k`` with the coefficients ``c(0)`` of
        slot 0, and a kept slot ``z`` takes the increment
        ``U_z + sum_k (c_k(z) - c_k(0)) A_k``, so that the merged map equals
        this map on the rows of slot 0 and of the kept slots. Rows of the
        other slots use the shared weight.

        Parameters
        ----------
        task
            Task increment of the weight and of the bias of this map.
        coeff
            Coefficients of the experts of every element slot with shape
            (1 + n_elements, K); None without experts.
        keep
            Element slots (``1..n_elements``) that keep an increment, in the
            order of the slots of the merged map.

        Returns
        -------
        NativeOP
            The expert map with the kept element increments and no experts,
            or the plain map when no slot is kept.
        """
        data = super().serialize()
        variables = data["@variables"]
        names = self._variable_names()
        slots = np.asarray(list(keep), dtype=np.int64)
        table = None if coeff is None else np.asarray(coeff, dtype=np.float64)

        def merged(
            plain: np.ndarray,
            increment: Any,
            element: Any,
            experts: Any,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Shared term and kept element increments of a weight or a bias."""
            plain = np.asarray(plain)
            shared = plain.astype(np.float64) + to_numpy_array(increment)
            kept = np.zeros((slots.size, *plain.shape), dtype=np.float64)
            if element is not None:
                kept = kept + to_numpy_array(element)[slots - 1]
            if experts is not None:
                experts = to_numpy_array(experts).astype(np.float64)
                shared = shared + np.tensordot(table[0], experts, axes=1)
                kept = kept + np.tensordot(table[slots] - table[0], experts, axes=1)
            return shared.astype(plain.dtype), kept.astype(plain.dtype)

        task_weight, task_bias = task
        variables[names["weight"]], element_weight = merged(
            variables[names["weight"]],
            task_weight,
            self.element_weight,
            self.expert_weight,
        )
        element_bias = None
        if self.has_bias():
            variables[names["bias"]], element_bias = merged(
                variables[names["bias"]],
                task_bias,
                self.element_bias,
                self.expert_bias,
            )
        if not slots.size:
            return self.PLAIN.deserialize(data)
        self._expert_record(data, n_experts=0, n_elements=int(slots.size))
        variables["element_weight"] = element_weight
        if element_bias is not None:
            variables["element_bias"] = element_bias
        return type(self).deserialize(data)

    # === Serialization ===

    def _variable_names(self) -> dict[str, str]:
        """Names of the weight and the bias among the variables of the plain map."""
        return {"weight": "weight", "bias": "bias"}

    @staticmethod
    def _record_config(data: dict[str, Any]) -> dict[str, Any]:
        """The part of a record of the map that holds its configuration."""
        return data["config"]

    def _expert_record(
        self, data: dict[str, Any], n_experts: int, n_elements: int
    ) -> dict[str, Any]:
        """Add the expert configuration to the record of the plain map."""
        data["@class"] = type(self).__name__
        config = self._record_config(data)
        config["n_experts"] = n_experts
        config["n_elements"] = n_elements
        return data

    def _expert_variables(self) -> dict[str, np.ndarray]:
        """The expert arrays that the map holds."""
        return {
            name: to_numpy_array(getattr(self, name))
            for name in (
                "element_weight",
                "element_bias",
                "expert_weight",
                "expert_bias",
            )
            if getattr(self, name) is not None
        }

    def serialize(self) -> dict[str, Any]:
        """Serialize the map to a dict."""
        data = self._expert_record(super().serialize(), self.n_experts, self.n_elements)
        data["@variables"].update(self._expert_variables())
        return data

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> Any:
        """
        Deserialize the map from a dict.

        The plain class restores the map of the record, and the MoE
        weights of the record are loaded onto it.
        """
        if data["@class"] != cls.__name__:
            raise ValueError(f"Invalid class for {cls.__name__}: {data['@class']}")
        config = cls._record_config(data)
        variables = data["@variables"]
        obj = super().deserialize({**data, "@class": cls.PLAIN.__name__})
        obj._init_experts(config["n_experts"], config["n_elements"])
        prec = PRECISION_DICT[obj.precision.lower()]
        for name in obj._expert_variables():
            if name in variables:
                setattr(obj, name, np.asarray(variables[name], dtype=prec))
        return obj


class ExpertContainer:
    """
    Module that owns expert maps, directly or in member modules.

    A subclass records in ``experts`` the number of experts and of element
    increments ``[n_experts, n_elements]`` of its maps, or None for plain
    maps, and names in :meth:`_expert_members` the members that are expert
    maps, sequences of expert maps or containers of them. The order of the
    members is the order of the task increments that a route hands to the
    container.
    """

    experts: list[int] | None

    def _expert_members(self) -> list[str]:
        """Names of the members that hold expert maps, in order."""
        raise NotImplementedError

    def expert_maps(self) -> list[Any]:
        """The expert maps of the container, in the order of their increments."""
        if self.experts is None:
            return []
        maps: list[Any] = []
        for name in self._expert_members():
            member = getattr(self, name)
            if isinstance(member, ExpertContainer):
                maps.extend(member.expert_maps())
            elif isinstance(member, NativeOP):
                maps.append(member)
            else:
                maps.extend(member)
        return maps

    def merge_experts(
        self,
        task: Sequence[tuple[np.ndarray, np.ndarray | None]],
        coeff: np.ndarray | None,
        keep: Sequence[int],
    ) -> None:
        """
        Merge every expert map of the container, in place.

        Parameters
        ----------
        task
            Task increments of the maps of :meth:`expert_maps`.
        coeff
            Coefficients of the experts of every element slot with shape
            (1 + n_elements, K); None without experts.
        keep
            Element slots that keep an increment (see
            :meth:`ExpertWeights.merge`).
        """
        if self.experts is None:
            return
        offset = 0
        for name in self._expert_members():
            member = getattr(self, name)
            if isinstance(member, ExpertContainer):
                count = len(member.expert_maps())
                member.merge_experts(task[offset : offset + count], coeff, keep)
            elif isinstance(member, NativeOP):
                count = 1
                setattr(self, name, member.merge(task[offset], coeff, keep))
            else:
                count = len(member)
                for index, increment in enumerate(task[offset : offset + count]):
                    member[index] = member[index].merge(increment, coeff, keep)
            offset += count
        self.experts = [0, len(keep)] if len(keep) else None


# Expert variant of every plain map, which :func:`linear_with_experts` builds.
# A map registers its variant in the module that defines it.
EXPERT_VARIANTS: dict[type, type] = {}


def linear_with_experts(
    cls: type, experts: Sequence[int] | None, *args: Any, **kwargs: Any
) -> NativeOP:
    """
    Build a linear map, as its expert variant for a component with experts.

    Parameters
    ----------
    cls
        The plain map class, which is a key of :data:`EXPERT_VARIANTS`.
    experts
        Number of experts and of element increments ``(n_experts,
        n_elements)`` of the maps of the component; None builds the plain map.
    *args, **kwargs
        Arguments of ``cls``.

    Returns
    -------
    NativeOP
        The plain map or its expert variant.
    """
    if experts is None:
        return cls(*args, **kwargs)
    n_experts, n_elements = experts
    return EXPERT_VARIANTS[cls](
        *args, n_experts=n_experts, n_elements=n_elements, **kwargs
    )


def map_args(route: Any, index: int) -> tuple[Any, ...]:
    """
    Positional expert arguments of one map of a container.

    Parameters
    ----------
    route
        Route of the container (a
        :class:`~deepmd.dpmodel.utils.expert_route.UnitRoute`), with the
        weights of its expert maps; None for a container of plain maps.
    index
        Position of the map among the expert maps of the container.

    Returns
    -------
    tuple
        The weights of the map, or nothing for a plain map.
    """
    return () if route is None else (route.map(index),)


def init_task_increments(
    maps: Sequence[ExpertWeights],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Task increments of the expert maps of a component, all zero.

    Parameters
    ----------
    maps
        The expert maps of the component in their order.

    Returns
    -------
    weights : list[np.ndarray]
        Increment of the weight of every map.
    biases : list[np.ndarray]
        Increment of the bias of every map that has one, in the order of the
        maps.
    """
    weights, biases = [], []
    for layer in maps:
        prec = PRECISION_DICT[layer.precision.lower()]
        weight_shape, bias_shape = layer.task_shapes()
        weights.append(np.zeros(weight_shape, dtype=prec))
        if bias_shape is not None:
            biases.append(np.zeros(bias_shape, dtype=prec))
    return weights, biases


def pair_task_increments(
    maps: Sequence[ExpertWeights], weights: Sequence[Any], biases: Sequence[Any]
) -> list[tuple[Any, Any]]:
    """
    Task increment ``(weight, bias)`` of every expert map of a component.

    Parameters
    ----------
    maps
        The expert maps of the component in their order.
    weights, biases
        The increments as :func:`init_task_increments` lays them out.

    Returns
    -------
    list[tuple]
        The increment of every map; the bias is None for a map without one.
    """
    remaining = iter(biases)
    return [
        (weight, next(remaining) if layer.has_bias() else None)
        for layer, weight in zip(maps, weights, strict=True)
    ]


def task_variables(weights: Sequence[Any], biases: Sequence[Any]) -> dict[str, Any]:
    """Task increments of a component as serialized variables."""
    variables = {
        f"task_weight.{index}": to_numpy_array(value)
        for index, value in enumerate(weights)
    }
    variables.update(
        {
            f"task_bias.{index}": to_numpy_array(value)
            for index, value in enumerate(biases)
        }
    )
    return variables


def load_task_variables(
    maps: Sequence[ExpertWeights], variables: dict[str, Any]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Task increments of a component from serialized variables.

    Parameters
    ----------
    maps
        The expert maps of the component in their order.
    variables
        Variables holding the increments under the keys of
        :func:`task_variables`.

    Returns
    -------
    weights, biases : list[np.ndarray]
        The increments as :func:`init_task_increments` lays them out.
    """
    weights, biases = init_task_increments(maps)
    return (
        [
            np.asarray(variables[f"task_weight.{index}"], dtype=value.dtype)
            for index, value in enumerate(weights)
        ],
        [
            np.asarray(variables[f"task_bias.{index}"], dtype=value.dtype)
            for index, value in enumerate(biases)
        ],
    )
