# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-neutral neighbor representation contract.

The contract separates *how* a model consumes neighbors (dense fixed-capacity
lists vs. carry-all graphs) from the legacy ``sel`` / ``get_sel()`` ABI.
Descriptors and models expose it through plugins so preprocessing can decide
whether neighbor-capacity discovery is required before construction.
"""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    Any,
    Literal,
)

NeighborRepresentation = Literal["dense", "graph"]

# Construction stand-in for graph-native descriptors that still size mean/std
# buffers with ``nnei`` until the companion sel-decoupling work lands.
# Width 1 matches the intended graph-native per-type statistic shape.
GRAPH_NATIVE_CONSTRUCTION_SEL = 1

NEIGHBOR_CONTRACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class NeighborContract:
    r"""Versioned neighbor-representation capability.

    Parameters
    ----------
    version
        Schema version of this contract record.
    representation
        ``\"dense\"`` for fixed-capacity neighbor lists; ``\"graph\"`` for
        carry-all ``NeighborGraph`` execution.
    requires_capacity
        Whether neighbor-capacity discovery (``update_sel`` / auto-sel) must
        run before construction. Graph-native models set this to ``False``.
    capacity
        Explicit per-type (or mixed-type single) neighbor capacities when the
        dense ABI needs them. ``None`` for graph-native models - metadata must
        not invent a dummy capacity.
    """

    version: int = NEIGHBOR_CONTRACT_VERSION
    representation: NeighborRepresentation = "dense"
    requires_capacity: bool = True
    capacity: tuple[int, ...] | None = None

    @classmethod
    def dense(
        cls,
        capacity: list[int] | tuple[int, ...] | int | None = None,
        *,
        requires_capacity: bool | None = None,
    ) -> NeighborContract:
        """Build a dense-list contract.

        An explicit integer capacity means discovery is already settled.
        ``None`` (or an auto-sel marker handled by callers) means discovery is
        still required.
        """
        normalized = _normalize_capacity(capacity)
        if requires_capacity is None:
            requires_capacity = normalized is None
        return cls(
            version=NEIGHBOR_CONTRACT_VERSION,
            representation="dense",
            requires_capacity=bool(requires_capacity),
            capacity=normalized,
        )

    @classmethod
    def graph(cls, *, requires_capacity: bool = False) -> NeighborContract:
        """Build a carry-all graph contract with no neighbor capacity."""
        return cls(
            version=NEIGHBOR_CONTRACT_VERSION,
            representation="graph",
            requires_capacity=bool(requires_capacity),
            capacity=None,
        )

    @classmethod
    def from_legacy_sel(
        cls,
        sel: list[int] | tuple[int, ...] | int | str | None,
    ) -> NeighborContract:
        """Reconstruct a dense contract from a legacy ``sel`` value."""
        if sel is None or _is_auto_sel(sel):
            return cls.dense(None, requires_capacity=True)
        return cls.dense(sel, requires_capacity=False)

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> NeighborContract:
        """Read a contract from export/inference metadata.

        New archives carry ``neighbor_contract``. Legacy archives only have
        ``sel`` / ``nnei`` and are treated as dense.
        """
        raw = metadata.get("neighbor_contract")
        if raw is not None:
            return cls.from_dict(raw)
        if "sel" in metadata:
            return cls.from_legacy_sel(metadata["sel"])
        if "nnei" in metadata:
            return cls.dense(int(metadata["nnei"]), requires_capacity=False)
        return cls.dense(None, requires_capacity=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NeighborContract:
        """Deserialize a contract dictionary."""
        representation = data.get("representation", "dense")
        if representation not in ("dense", "graph"):
            raise ValueError(
                f"Unsupported neighbor representation {representation!r}; "
                "expected 'dense' or 'graph'."
            )
        capacity = data.get("capacity")
        return cls(
            version=int(data.get("version", NEIGHBOR_CONTRACT_VERSION)),
            representation=representation,
            requires_capacity=bool(
                data.get(
                    "requires_capacity",
                    representation == "dense" and capacity is None,
                )
            ),
            capacity=_normalize_capacity(capacity),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for export metadata."""
        return {
            "version": int(self.version),
            "representation": self.representation,
            "requires_capacity": bool(self.requires_capacity),
            "capacity": list(self.capacity) if self.capacity is not None else None,
        }

    @property
    def is_graph(self) -> bool:
        """Whether this contract describes a carry-all graph model."""
        return self.representation == "graph"

    @property
    def is_dense(self) -> bool:
        """Whether this contract describes a fixed-capacity dense model."""
        return self.representation == "dense"

    def legacy_sel(self) -> list[int]:
        """Return a dense ``sel`` list for legacy adapters.

        Graph contracts have no capacity; callers that still need a list must
        not invent one silently — this raises instead.
        """
        if self.capacity is None:
            raise ValueError(
                "NeighborContract has no capacity to expose as legacy sel "
                f"(representation={self.representation!r}, "
                f"requires_capacity={self.requires_capacity})."
            )
        return list(self.capacity)

    def merge(self, other: NeighborContract) -> NeighborContract:
        """Combine contracts from composed descriptors/models.

        Composites must agree on representation. Dense capacities take the
        element-wise maximum when both sides publish one; discovery remains
        required if either side still needs it.
        """
        if self.representation != other.representation:
            raise ValueError(
                "Incompatible neighbor contracts in a composed model: "
                f"{self.representation!r} vs {other.representation!r}. "
                "Every child must share one representation (dense or graph)."
            )
        if self.is_graph:
            # Preserve OR of requires_capacity so a graph child that still
            # needs discovery (e.g. DPA1 sel:auto) is not silently dropped.
            return NeighborContract.graph(
                requires_capacity=self.requires_capacity or other.requires_capacity
            )
        capacity = _merge_capacities(self.capacity, other.capacity)
        # Discovery remains required if either side still needs it, even when
        # the merged capacity is non-None (sibling may still be auto-sel).
        requires_capacity = self.requires_capacity or other.requires_capacity
        return NeighborContract.dense(capacity, requires_capacity=requires_capacity)


def is_auto_sel(sel: Any) -> bool:
    """Return whether ``sel`` is an auto-selection marker."""
    return _is_auto_sel(sel)


def _is_auto_sel(sel: Any) -> bool:
    if not isinstance(sel, str):
        return False
    return sel.split(":", 1)[0] == "auto"


def _normalize_capacity(
    capacity: list[int] | tuple[int, ...] | int | None,
) -> tuple[int, ...] | None:
    if capacity is None:
        return None
    if isinstance(capacity, int):
        return (int(capacity),)
    return tuple(int(x) for x in capacity)


def _merge_capacities(
    left: tuple[int, ...] | None,
    right: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left
    if len(left) == 1 and len(right) != 1:
        left = left * len(right)
    if len(right) == 1 and len(left) != 1:
        right = right * len(left)
    if len(left) != len(right):
        raise ValueError(
            "Incompatible dense neighbor capacities in a composed model: "
            f"{list(left)} vs {list(right)}."
        )
    return tuple(max(a, b) for a, b in zip(left, right, strict=True))


def dense_contract_from_sel_field(sel: Any) -> NeighborContract:
    """Interpret a descriptor ``sel`` config field as a dense contract."""
    return NeighborContract.from_legacy_sel(sel)


def graph_eligible_tebd_mode(tebd_input_mode: str | None) -> bool:
    """Whether a DPA1/DPA2-style tebd mode is graph-lower eligible."""
    return (tebd_input_mode or "concat") in ("concat", "strip")


def ensure_construction_sel(
    local_jdata: dict[str, Any],
    *,
    placeholder: int = GRAPH_NATIVE_CONSTRUCTION_SEL,
) -> dict[str, Any]:
    """Replace missing/auto ``sel`` with a construction placeholder.

    Used for graph-native descriptors that still allocate mean/std buffers
    with an ``nnei`` axis. The placeholder is not a discovered capacity and
    must not appear in exported graph metadata.
    """
    out = dict(local_jdata)
    sel = out.get("sel")
    if sel is None or _is_auto_sel(sel):
        out["sel"] = int(placeholder)
    return out
