# SPDX-License-Identifier: LGPL-3.0-or-later
"""High-level assembly dataset writer for grouped property training.

The DeepMD system stores only tensors needed at train time.  Scientific
semantics, generation provenance, source paths, roles, blocks, and fparam
schemas live in the adapt manifest so the user-facing API can evolve without
turning a DeepMD ``set.*`` directory into a metadata dump.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from dpa_adapt.data.errors import DPADataError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

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
    def indices(cls, indices: Sequence[int]) -> SiteSelector:
        return cls("indices", [int(i) for i in indices])

    @classmethod
    def element(cls, element: str) -> SiteSelector:
        return cls("element", str(element))

    @classmethod
    def tag(cls, tag: str) -> SiteSelector:
        return cls("tag", str(tag))

    @classmethod
    def top_layer(cls, element: str | None = None, topk: int | None = None) -> SiteSelector:
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
    def all(cls) -> PoolMask:
        return cls()

    @classmethod
    def exclude_indices(cls, indices: Sequence[int]) -> PoolMask:
        return cls(exclude=[int(i) for i in indices])

    @classmethod
    def include_indices(cls, indices: Sequence[int]) -> PoolMask:
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
    ) -> ComponentSpec:
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
        mask = self.normalized_pool_mask()
        if float(mask.sum()) == 0.0:
            raise DPADataError(
                "pool_mask is all-zero: every atom of this component is excluded "
                "from pooling, giving an undefined frame embedding. An all-masked "
                "frame is a data bug, not a numerical edge case."
            )


@dataclass
class GroupSpec:
    """A labeled group made of one or more component frames."""

    key: str
    label: float | Sequence[float]
    components: list[ComponentSpec] = field(default_factory=list)
    fparam: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_component(self, component: ComponentSpec, **overrides: Any) -> ComponentSpec:
        for key, value in overrides.items():
            if not hasattr(component, key):
                raise TypeError(f"Unknown ComponentSpec field: {key}")
            setattr(component, key, value)
        self.components.append(component)
        return component

    def add(
        self,
        coords: Sequence[Sequence[float]],
        symbols: Sequence[str],
        *,
        weight: float = 1.0,
        pool_mask: Sequence[float] | PoolMask | None = None,
        block: str | None = None,
        box: Sequence[Sequence[float]] | Sequence[float] | None = None,
        role: str | None = None,
    ) -> ComponentSpec:
        """Build a component from ``coords`` + ``symbols`` and append it.

        Sugar over ``add_component(ComponentSpec.from_arrays(...))``.
        """
        component = ComponentSpec.from_arrays(
            coords,
            symbols,
            box=box,
            weight=weight,
            pool_mask=pool_mask,
            block=block,
            role=role,
        )
        self.components.append(component)
        return component


class Assembly:
    """Build assembly DeepMD data plus an adapt manifest.

    Each group is written as one DeepMD system.  Components within a group may
    differ in size and composition: every frame is padded up to the group's max
    atom count with virtual atoms (real type -1) and stored in the DeepMD
    ``mixed_type`` layout, with padding atoms masked out of pooling.  Richer
    assembly semantics live in ``manifest.json``.
    """

    def __init__(
        self,
        *,
        target: str = "property",
        type_map: Sequence[str] | None = None,
        schema: str = "dpa_adapt.assembly.v1",
    ) -> None:
        self.property_name = str(target)
        self.type_map = [str(t) for t in type_map] if type_map is not None else None
        self.schema = schema
        self.groups: list[GroupSpec] = []
        self.fparam_schema: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # scenario constructors (facade)
    # ------------------------------------------------------------------

    @classmethod
    def from_polymer_csv(
        cls,
        path: str | Path,
        *,
        target: str = "cloud_point",
        **kwargs: Any,
    ) -> Any:
        """Build a grouped dataset from a cloud-point polymer CSV.

        Returns a polymer builder exposing ``.write(path)`` (see
        :meth:`dpa_adapt.grouped._polymer.PolymerBuilder.from_csv`).
        """
        from dpa_adapt.grouped._polymer import PolymerBuilder

        return PolymerBuilder.from_csv(path, target=target, **kwargs)

    @classmethod
    def mark_existing(
        cls,
        data: Any,
        *,
        target: str = "property",
        group_by: str | int = "system",
        **kwargs: Any,
    ) -> Any:
        """Retrofit existing mixed-type deepmd/npy *in place* with grouped
        markers (``group_id`` / ``pool_mask``); see
        :func:`dpa_adapt.grouped._convert.add_group_markers`.
        """
        from dpa_adapt.grouped._convert import add_group_markers

        return add_group_markers(data, group_by=group_by, property_name=target, **kwargs)

    def group(
        self,
        *,
        key: str | None = None,
        label: float | Sequence[float],
        fparam: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GroupSpec:
        group = GroupSpec(
            key=str(key if key is not None else f"group_{len(self.groups)}"),
            label=label,
            fparam={str(k): float(v) for k, v in (fparam or {}).items()},
            metadata=dict(metadata or {}),
        )
        self.groups.append(group)
        return group

    def set_fparam_schema(self, schema: Sequence[Mapping[str, Any]]) -> None:
        self.fparam_schema = [dict(item) for item in schema]

    def write(
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

        resolved_type_map = self._resolved_type_map()

        manifest_groups = []
        systems: list[str] = []
        for group_idx, group in enumerate(self.groups):
            system_path = systems_root / _safe_name(group.key, fallback=f"group_{group_idx}")
            _write_group_system(
                group,
                group_idx=group_idx,
                system_path=system_path,
                property_name=self.property_name,
                type_map=resolved_type_map,
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
                "fparam": "fparam" if self._has_fparam() else None,
            },
            "type_map": resolved_type_map,
            "fparam_schema": self.fparam_schema,
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

    def _resolved_type_map(self) -> list[str] | None:
        if self.type_map is not None:
            return list(self.type_map)
        symbols = [
            sym
            for group in self.groups
            for component in group.components
            for sym in component.symbols
        ]
        return _stable_unique(symbols) if symbols else None

    def _has_fparam(self) -> bool:
        return any(group.fparam for group in self.groups)


def write_grouped_deepmd(
    groups: Iterable[GroupSpec],
    out: str | Path,
    *,
    target: str = "property",
    type_map: Sequence[str] | None = None,
    fparam_schema: Sequence[Mapping[str, Any]] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write assembly data from pre-built :class:`GroupSpec` objects."""
    builder = Assembly(target=target, type_map=type_map)
    builder.groups.extend(groups)
    if fparam_schema is not None:
        builder.set_fparam_schema(fparam_schema)
    return builder.write(out, overwrite=overwrite)


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

    # Components in a group may differ in size and composition (e.g. OER
    # O*/OH*/OOH*).  Pad every frame up to the group's max atom count with
    # virtual atoms (real type -1) and emit the DeepMD ``mixed_type`` layout so
    # one shared descriptor can consume the whole group in a single system.
    # Padding atoms are excluded from pooling (``pool_mask`` = 0) and ignored by
    # the neighbor list (type < 0), so they never affect real-atom embeddings.
    natoms = max(len(c.symbols) for c in group.components)
    all_symbols = [sym for c in group.components for sym in c.symbols]
    resolved_type_map = (
        list(type_map) if type_map is not None else _stable_unique(all_symbols)
    )
    type_index = {el: ii for ii, el in enumerate(resolved_type_map)}
    missing = sorted({sym for sym in all_symbols if sym not in type_index})
    if missing:
        raise DPADataError(
            f"type_map is missing symbols for group {group.key!r}: {missing}"
        )

    set_dir = system_path / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)
    (system_path / "type_map.raw").write_text(
        "".join(f"{el}\n" for el in resolved_type_map), encoding="utf-8"
    )
    # ``mixed_type`` placeholder: a uniform (all-zero) ``type.raw`` makes
    # DeepMD's per-atom sort a no-op, keeping coord/pool_mask/real_atom_types
    # aligned in written order.  Real per-frame types live in real_atom_types.
    (system_path / "type.raw").write_text("0\n" * natoms, encoding="utf-8")

    nframes = len(group.components)
    coord = np.zeros((nframes, natoms * 3), dtype=np.float64)
    real_atom_types = np.full((nframes, natoms), -1, dtype=np.int32)
    pool_mask = np.zeros((nframes, natoms), dtype=np.float64)
    for frame, component in enumerate(group.components):
        n_i = len(component.symbols)
        coord[frame, : n_i * 3] = component.coords.reshape(n_i * 3)
        real_atom_types[frame, :n_i] = [type_index[sym] for sym in component.symbols]
        pool_mask[frame, :n_i] = component.normalized_pool_mask()
        # Task D: place padding (virtual, real_atom_types == -1) atoms at a large,
        # spread-out non-physical offset so they lie outside every real atom's
        # cutoff -- and each other's -- even on a backend whose neighbor list does
        # NOT relocate atype<0.  (The pt backend also masks atype<0 in nlist, so
        # for the pt/PBC path this is defensive; see the data-format note below.)
        box_diag = float(np.linalg.norm(component.normalized_box().sum(axis=0)))
        for pad_index in range(natoms - n_i):
            offset = box_diag + 100.0 * (pad_index + 1)
            start = (n_i + pad_index) * 3
            coord[frame, start : start + 3] = offset

    box = np.stack([c.normalized_box().reshape(9) for c in group.components])
    group_id = np.full((nframes,), int(group_idx), dtype=np.int64)
    weight = np.asarray([c.weight for c in group.components], dtype=np.float64)
    label = np.asarray(group.label, dtype=np.float64).reshape(1, -1)
    label = np.repeat(label, nframes, axis=0)

    # Data-format notes:
    #  - ``role`` (e.g. repeat_unit, solvent) is CONSTRUCTION-TIME metadata: it
    #    informs weight/pool_mask generation in Assembly readers and is NOT
    #    serialized to npy or consumed by the model (only kept in the manifest).
    #  - Padding atoms (real_atom_types == -1) carry non-physical coords by
    #    construction (large offset above).  On the pt backend the nlist also
    #    relocates atype<0, so for periodic systems -- where the offset may wrap
    #    back into the cell -- that nlist masking is the guarantee; the offset is
    #    the fallback for PBC-off / other backends.
    np.save(set_dir / "coord.npy", coord)
    np.save(set_dir / "box.npy", box)
    np.save(set_dir / "real_atom_types.npy", real_atom_types)
    np.save(set_dir / f"{property_name}.npy", label)
    np.save(set_dir / f"{GROUP_ID_KEY}.npy", group_id)
    np.save(set_dir / f"{WEIGHT_KEY}.npy", weight)
    np.save(set_dir / f"{POOL_MASK_KEY}.npy", pool_mask)
    if group.fparam:
        keys = sorted(group.fparam)
        fparam = np.asarray([[group.fparam[k] for k in keys]], dtype=np.float64)
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
        "fparam": group.fparam,
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
