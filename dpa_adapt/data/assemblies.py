# SPDX-License-Identifier: LGPL-3.0-or-later
"""High-level assembly dataset writer for grouped property training.

The DeepMD system stores only tensors needed at train time.  Scientific
semantics, generation provenance, source paths, roles, blocks, and condition
schemas live in the adapt manifest so the user-facing API can evolve without
turning a DeepMD ``set.*`` directory into a metadata dump.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from dpa_adapt.data.errors import DPADataError

GROUP_ID_KEY = "group_id"
WEIGHT_KEY = "weight"
POOL_MASK_KEY = "pool_mask"
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class SiteSelector:
    """Declarative site selection spec stored in manifest provenance."""

    mode: str
    value: Any

    @classmethod
    def indices(cls, indices: Sequence[int]) -> "SiteSelector":
        return cls("indices", [int(i) for i in indices])

    @classmethod
    def element(cls, element: str) -> "SiteSelector":
        return cls("element", str(element))

    @classmethod
    def tag(cls, tag: str) -> "SiteSelector":
        return cls("tag", str(tag))

    @classmethod
    def top_layer(cls, element: str | None = None, topk: int | None = None) -> "SiteSelector":
        value: dict[str, Any] = {}
        if element is not None:
            value["element"] = element
        if topk is not None:
            value["topk"] = int(topk)
        return cls("top_layer", value)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "value": self.value}


@dataclass(frozen=True)
class SubstitutionSpec:
    """Declarative substitution/doping provenance for manifest storage."""

    sites: SiteSelector
    composition: Mapping[str, float]
    mode: str = "random_by_fraction"
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "mode": self.mode,
            "sites": self.sites.to_dict(),
            "composition": {str(k): float(v) for k, v in self.composition.items()},
        }
        if self.seed is not None:
            data["seed"] = int(self.seed)
        return data


@dataclass(frozen=True)
class PoolMask:
    """Helpers for constructing frame-level pooling masks."""

    include: Sequence[int] | None = None
    exclude: Sequence[int] | None = None

    @classmethod
    def all(cls) -> "PoolMask":
        return cls()

    @classmethod
    def exclude_indices(cls, indices: Sequence[int]) -> "PoolMask":
        return cls(exclude=[int(i) for i in indices])

    @classmethod
    def include_indices(cls, indices: Sequence[int]) -> "PoolMask":
        return cls(include=[int(i) for i in indices])

    def as_array(self, natoms: int) -> np.ndarray:
        mask = np.ones(natoms, dtype=np.float64)
        if self.include is not None:
            mask[:] = 0.0
            mask[list(self.include)] = 1.0
        if self.exclude is not None:
            mask[list(self.exclude)] = 0.0
        return mask


@dataclass
class ComponentSpec:
    """One structure/component that becomes one DeepMD frame."""

    coords: np.ndarray
    symbols: list[str]
    box: np.ndarray | None = None
    weight: float = 1.0
    pool_mask: np.ndarray | PoolMask | None = None
    role: str | None = None
    block: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_arrays(
        cls,
        coords: Sequence[Sequence[float]],
        symbols: Sequence[str],
        *,
        box: Sequence[Sequence[float]] | Sequence[float] | None = None,
        weight: float = 1.0,
        pool_mask: Sequence[float] | PoolMask | None = None,
        role: str | None = None,
        block: str | None = None,
        source: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ComponentSpec":
        return cls(
            coords=np.asarray(coords, dtype=np.float64),
            symbols=[str(s) for s in symbols],
            box=None if box is None else np.asarray(box, dtype=np.float64),
            weight=float(weight),
            pool_mask=pool_mask if isinstance(pool_mask, PoolMask) or pool_mask is None else np.asarray(pool_mask, dtype=np.float64),
            role=role,
            block=block,
            source=source,
            metadata=dict(metadata or {}),
        )

    def normalized_box(self) -> np.ndarray:
        if self.box is None:
            return np.eye(3, dtype=np.float64) * 100.0
        box = np.asarray(self.box, dtype=np.float64)
        if box.shape == (9,):
            return box.reshape(3, 3)
        if box.shape != (3, 3):
            raise DPADataError(f"box has shape {box.shape}; expected (3,3) or (9,).")
        return box

    def normalized_pool_mask(self) -> np.ndarray:
        natoms = len(self.symbols)
        if self.pool_mask is None:
            return np.ones(natoms, dtype=np.float64)
        if isinstance(self.pool_mask, PoolMask):
            return self.pool_mask.as_array(natoms)
        mask = np.asarray(self.pool_mask, dtype=np.float64).reshape(-1)
        if mask.shape != (natoms,):
            raise DPADataError(
                f"pool_mask has shape {mask.shape}; expected ({natoms},)."
            )
        return mask

    def validate(self) -> None:
        if self.coords.shape != (len(self.symbols), 3):
            raise DPADataError(
                f"coords has shape {self.coords.shape}; expected "
                f"({len(self.symbols)}, 3)."
            )
        self.normalized_box()
        self.normalized_pool_mask()


@dataclass
class GroupSpec:
    """A group-level labeled sample made of one or more components."""

    key: str
    label: float | Sequence[float]
    components: list[ComponentSpec] = field(default_factory=list)
    conditions: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_component(self, component: ComponentSpec, **overrides: Any) -> ComponentSpec:
        for key, value in overrides.items():
            if not hasattr(component, key):
                raise TypeError(f"Unknown ComponentSpec field: {key}")
            setattr(component, key, value)
        self.components.append(component)
        return component


class AssemblyDatasetBuilder:
    """Build grouped DeepMD data plus an adapt manifest.

    The current writer is intentionally conservative: each group is written as one
    DeepMD system, and all components within a group must have identical atom
    count and symbol order.  This matches the DeepMD MVP while keeping richer
    assembly semantics in ``manifest.json``.
    """

    def __init__(
        self,
        *,
        property_name: str = "property",
        type_map: Sequence[str] | None = None,
        schema: str = "dpa_adapt.assembly.v1",
    ) -> None:
        self.property_name = str(property_name)
        self.type_map = [str(t) for t in type_map] if type_map is not None else None
        self.schema = schema
        self.groups: list[GroupSpec] = []
        self.condition_schema: list[dict[str, Any]] = []

    def group(
        self,
        *,
        key: str | None = None,
        label: float | Sequence[float],
        conditions: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GroupSpec:
        group = GroupSpec(
            key=str(key if key is not None else f"group_{len(self.groups)}"),
            label=label,
            conditions={str(k): float(v) for k, v in (conditions or {}).items()},
            metadata=dict(metadata or {}),
        )
        self.groups.append(group)
        return group

    def set_condition_schema(self, schema: Sequence[Mapping[str, Any]]) -> None:
        self.condition_schema = [dict(item) for item in schema]

    def write_deepmd_npy(
        self,
        out: str | Path,
        *,
        system_dir: str = "systems",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        out_path = Path(out)
        if out_path.exists() and any(out_path.iterdir()) and not overwrite:
            raise DPADataError(f"Output directory is not empty: {out_path}")
        out_path.mkdir(parents=True, exist_ok=True)
        systems_root = out_path / system_dir
        systems_root.mkdir(parents=True, exist_ok=True)

        manifest_groups = []
        systems: list[str] = []
        for group_idx, group in enumerate(self.groups):
            system_path = systems_root / _safe_name(group.key, fallback=f"group_{group_idx}")
            _write_group_system(
                group,
                group_idx=group_idx,
                system_path=system_path,
                property_name=self.property_name,
                type_map=self.type_map,
            )
            systems.append(str(system_path.relative_to(out_path)))
            manifest_groups.append(
                _group_manifest(group, group_idx, str(system_path.relative_to(out_path)))
            )

        manifest = {
            "schema": self.schema,
            "property_name": self.property_name,
            "system_dir": system_dir,
            "tensor_fields": {
                "group_id": GROUP_ID_KEY,
                "weight": WEIGHT_KEY,
                "pool_mask": POOL_MASK_KEY,
                "label": self.property_name,
                "conditions": "fparam" if self._has_conditions() else None,
            },
            "type_map": self.type_map,
            "condition_schema": self.condition_schema,
            "groups": manifest_groups,
        }
        manifest_path = out_path / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "output_dir": str(out_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "systems": systems,
            "n_groups": len(self.groups),
        }

    def _has_conditions(self) -> bool:
        return any(group.conditions for group in self.groups)


def write_grouped_deepmd(
    groups: Iterable[GroupSpec],
    out: str | Path,
    *,
    property_name: str = "property",
    type_map: Sequence[str] | None = None,
    condition_schema: Sequence[Mapping[str, Any]] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write grouped data from pre-built :class:`GroupSpec` objects."""
    builder = AssemblyDatasetBuilder(property_name=property_name, type_map=type_map)
    builder.groups.extend(groups)
    if condition_schema is not None:
        builder.set_condition_schema(condition_schema)
    return builder.write_deepmd_npy(out, overwrite=overwrite)


def _write_group_system(
    group: GroupSpec,
    *,
    group_idx: int,
    system_path: Path,
    property_name: str,
    type_map: Sequence[str] | None,
) -> None:
    if not group.components:
        raise DPADataError(f"Group {group.key!r} has no components.")
    for component in group.components:
        component.validate()

    symbols = group.components[0].symbols
    natoms = len(symbols)
    for component in group.components[1:]:
        if component.symbols != symbols:
            raise DPADataError(
                f"Group {group.key!r} components must have identical symbol order."
            )

    resolved_type_map = list(type_map) if type_map is not None else _stable_unique(symbols)
    type_index = {el: ii for ii, el in enumerate(resolved_type_map)}
    missing = [sym for sym in symbols if sym not in type_index]
    if missing:
        raise DPADataError(
            f"type_map is missing symbols for group {group.key!r}: {sorted(set(missing))}"
        )

    set_dir = system_path / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)
    (system_path / "type_map.raw").write_text(
        "".join(f"{el}\n" for el in resolved_type_map), encoding="utf-8"
    )
    (system_path / "type.raw").write_text(
        "".join(f"{type_index[sym]}\n" for sym in symbols), encoding="utf-8"
    )

    nframes = len(group.components)
    coord = np.stack([c.coords.reshape(natoms * 3) for c in group.components])
    box = np.stack([c.normalized_box().reshape(9) for c in group.components])
    group_id = np.full((nframes,), int(group_idx), dtype=np.int64)
    weight = np.asarray([c.weight for c in group.components], dtype=np.float64)
    pool_mask = np.stack([c.normalized_pool_mask() for c in group.components])
    label = np.asarray(group.label, dtype=np.float64).reshape(1, -1)
    label = np.repeat(label, nframes, axis=0)

    np.save(set_dir / "coord.npy", coord)
    np.save(set_dir / "box.npy", box)
    np.save(set_dir / f"{property_name}.npy", label)
    np.save(set_dir / f"{GROUP_ID_KEY}.npy", group_id)
    np.save(set_dir / f"{WEIGHT_KEY}.npy", weight)
    np.save(set_dir / f"{POOL_MASK_KEY}.npy", pool_mask)
    if group.conditions:
        keys = sorted(group.conditions)
        fparam = np.asarray([[group.conditions[k] for k in keys]], dtype=np.float64)
        np.save(set_dir / "fparam.npy", np.repeat(fparam, nframes, axis=0))


def _group_manifest(group: GroupSpec, group_idx: int, system: str) -> dict[str, Any]:
    components = []
    for frame, component in enumerate(group.components):
        item = {
            "frame": frame,
            "weight": float(component.weight),
            "role": component.role,
            "block": component.block,
            "source": component.source,
            "pool_mask_excluded": np.where(component.normalized_pool_mask() == 0)[0].astype(int).tolist(),
            "metadata": component.metadata,
        }
        components.append({k: v for k, v in item.items() if v is not None and v != {}})
    return {
        "group_id": int(group_idx),
        "key": group.key,
        "label": np.asarray(group.label, dtype=float).reshape(-1).tolist(),
        "system": system,
        "conditions": group.conditions,
        "metadata": group.metadata,
        "components": components,
    }


def _safe_name(raw: str, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", raw).strip("._")
    return name or fallback


def _stable_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
