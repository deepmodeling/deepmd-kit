# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build grouped DeepMD data for polymer property prediction.

A polymer is one **group**: each repeating unit / end group becomes one
component (frame), weighted by its mole fraction (end groups by their computed
share).  Per-polymer non-structural context (Mn, salts, pH, concentration) is
standardized and written as ``fparam.npy`` -- a per-group side-feature vector the
grouped ``group_property`` head concatenates after aggregation.

Thin wrapper over :class:`dpa_adapt.grouped.Assembly`:

    PolymerBuilder.from_csv("cloud_points_data.csv", target="cloud_point").write(PATH)

SMILES are embedded to 3D once each (RDKit ETKDG+MMFF, open valences capped),
the type_map is the element union across all monomers, and every auxiliary npy
(coord / real_atom_types / pool_mask / weight / group_id / fparam / label) is
written automatically.
"""

from __future__ import (
    annotations,
)

import json
import math
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import (
    dataclass,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np

from dpa_adapt.data.errors import (
    DPADataError,
)

MN_KEY = "mn_log"
SALT_PREFIX = "salt:"


def _isnan(value: object) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return value is None


def _as_unit_pairs(
    units: Mapping[str, float] | Sequence[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Normalize ``units`` to a list of ``(smiles, mole_fraction)`` pairs."""
    if isinstance(units, Mapping):
        items = list(units.items())
    else:
        items = [tuple(item) for item in units]
    pairs: list[tuple[str, float]] = []
    for smiles, frac in items:
        if smiles is None or _isnan(frac):
            continue
        pairs.append((str(smiles), float(frac)))
    if not pairs:
        raise DPADataError("A polymer needs at least one repeating unit.")
    return pairs


@dataclass
class _PolymerRow:
    units: list[tuple[str, float]]
    ends: list[str]
    mol_weight: float | None
    scalars: dict[str, float]
    salts: dict[str, float]
    target: float
    key: str


class PolymerBuilder:
    """Accumulate polymers and write them as grouped DeepMD systems."""

    def __init__(
        self,
        target: str = "cloud_point",
        *,
        type_map: Sequence[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.target = str(target)
        self.type_map = list(type_map) if type_map is not None else None
        self.seed = int(seed)
        self._rows: list[_PolymerRow] = []

    # ------------------------------------------------------------------
    # low-level primitive
    # ------------------------------------------------------------------

    def add(
        self,
        *,
        units: Mapping[str, float] | Sequence[tuple[str, float]],
        target: float,
        ends: Sequence[str] | None = None,
        mol_weight: float | None = None,
        fparam: Mapping[str, Any] | None = None,
        key: str | None = None,
    ) -> PolymerBuilder:
        """Add one polymer (becomes one group).

        Parameters
        ----------
        units
            Repeating units as ``{smiles: mole_fraction}`` or ``[(smiles, frac)]``.
        target
            Group-level label (e.g. cloud point).
        ends
            End-group SMILES; weighted by the computed end share (needs
            *mol_weight*), else by the mean unit fraction.
        mol_weight
            Number-average molar mass (Mn): used for the end share and, as
            ``log10(Mn)``, as a side feature.
        fparam
            Per-polymer context.  Scalar entries become fparam columns; a nested
            ``"salts": {name: molar_conc}`` becomes one-hot concentration columns.
        key
            Optional system name; defaults to ``polymer_{i}``.
        """
        conds = dict(fparam or {})
        salts_raw = conds.pop("salts", None) or {}
        scalars = {
            str(k): float(v)
            for k, v in conds.items()
            if v is not None and not _isnan(v)
        }
        salts = {
            str(name): float(conc)
            for name, conc in salts_raw.items()
            if name is not None and conc is not None and not _isnan(conc)
        }
        self._rows.append(
            _PolymerRow(
                units=_as_unit_pairs(units),
                ends=[str(s) for s in (ends or []) if s is not None and not _isnan(s)],
                mol_weight=None
                if mol_weight is None or _isnan(mol_weight)
                else float(mol_weight),
                scalars=scalars,
                salts=salts,
                target=float(target),
                key=str(key) if key is not None else f"polymer_{len(self._rows)}",
            )
        )
        return self

    # ------------------------------------------------------------------
    # CSV ingestion (standard cloud-point schema)
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        target: str = "cloud_point",
        sep: str = ";",
        decimal: str = ",",
        seed: int = 42,
    ) -> PolymerBuilder:
        """Ingest the standard cloud-point polymer CSV.

        Recognizes ``SMILES_repeating_unitA..E`` + ``molpercent_repeating_unitA..E``,
        ``SMILES_start_group`` / ``SMILES_end_group``, ``Mn``, ``pH``,
        ``polymer_concentration_wpercent`` (or ``_mass_conc``), ``additive1/2`` +
        ``additive*_concentration_molar``, and ``{target}``.  Rows missing Mn or
        the target are skipped.
        """
        import pandas as pd

        df = pd.read_csv(str(path), sep=sep, decimal=decimal, encoding="utf8")
        builder = cls(target=target, seed=seed)

        def cell(row: Any, col: str) -> Any:
            return row[col] if col in df.columns else None

        for idx, row in df.iterrows():
            if _isnan(cell(row, target)) or _isnan(cell(row, "Mn")):
                continue
            units: list[tuple[str, float]] = []
            for suffix in ("A", "B", "C", "D", "E"):
                smiles = cell(row, f"SMILES_repeating_unit{suffix}")
                frac = cell(row, f"molpercent_repeating_unit{suffix}")
                if smiles is None or _isnan(smiles) or _isnan(frac):
                    continue
                units.append((str(smiles), float(frac)))
            if not units:
                continue

            ends = [
                str(cell(row, c))
                for c in ("SMILES_start_group", "SMILES_end_group")
                if cell(row, c) is not None and not _isnan(cell(row, c))
            ]

            conc = cell(row, "polymer_concentration_wpercent")
            if _isnan(conc):
                conc = cell(row, "polymer_concentration_mass_conc")
            pH = cell(row, "pH")
            fparam: dict[str, Any] = {
                "pH": 7.0 if _isnan(pH) else float(pH),
            }
            if not _isnan(conc):
                fparam["conc"] = float(conc)

            salts: dict[str, float] = {}
            for name_col, conc_col in (
                ("additive1", "additive1_concentration_molar"),
                ("additive2", "additive2_concentration_molar"),
            ):
                name = cell(row, name_col)
                c = cell(row, conc_col)
                if name is not None and not _isnan(name) and not _isnan(c):
                    salts[str(name)] = salts.get(str(name), 0.0) + float(c)
            if salts:
                fparam["salts"] = salts

            builder.add(
                units=units,
                ends=ends,
                mol_weight=float(cell(row, "Mn")),
                fparam=fparam,
                target=float(cell(row, target)),
                key=f"row_{idx}",
            )
        if not builder._rows:
            raise DPADataError(f"No usable polymer rows parsed from {path!r}.")
        return builder

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(
        self,
        out: str | Path,
        *,
        overwrite: bool = False,
        scaler: dict[str, Any] | str | Path | None = None,
    ) -> dict[str, Any]:
        """Embed, standardize, and write grouped DeepMD data.

        ``scaler`` reuses a previously written scaler (dict or path to the
        ``polymer_scaler.json`` written by an earlier ``write``) so a validation
        split is standardized with the training statistics.  When ``None`` the
        scaler is fit on these rows and saved next to the data.
        """
        from dpa_adapt.grouped._core import (
            Assembly,
            ComponentSpec,
        )

        if not self._rows:
            raise DPADataError("PolymerBuilder is empty; call add()/from_csv() first.")

        coords_by_smiles = self._embed_all()
        type_map = self.type_map or self._collect_type_map(coords_by_smiles)

        loaded_scaler = _load_scaler(scaler)
        schema = loaded_scaler["columns"] if loaded_scaler else self._build_schema()
        stats = loaded_scaler["stats"] if loaded_scaler else self._fit_stats(schema)

        builder = Assembly(target=self.target, type_map=type_map)
        for row in self._rows:
            fparam_vec = self._standardized_fparam(row, schema, stats)
            group = builder.group(key=row.key, label=row.target, fparam=fparam_vec)
            end_w = _end_share(row.ends, row.units, row.mol_weight, coords_by_smiles)
            for smiles, frac in row.units:
                symbols, xyz = coords_by_smiles[smiles]
                group.add_component(
                    ComponentSpec.from_arrays(xyz, symbols, weight=frac, block="rep")
                )
            for smiles in row.ends:
                symbols, xyz = coords_by_smiles[smiles]
                group.add_component(
                    ComponentSpec.from_arrays(xyz, symbols, weight=end_w, block="end")
                )

        result = builder.write(out, overwrite=overwrite)

        scaler_out = {"columns": schema, "stats": stats}
        scaler_path = Path(result["output_dir"]) / "polymer_scaler.json"
        scaler_path.write_text(json.dumps(scaler_out, indent=2), encoding="utf-8")
        result["fparam_dim"] = len(schema)
        result["scaler"] = scaler_out
        result["type_map"] = type_map
        # Systems live under ``<out>/systems/*``; hand back a ready-to-train glob.
        result["train_glob"] = str(Path(result["output_dir"]) / "systems" / "*")
        return result

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _embed_all(self) -> dict[str, tuple[list[str], np.ndarray]]:
        from dpa_adapt.data.smiles import (
            smiles_to_3d_coords,
        )

        smiles_set = {s for row in self._rows for s, _ in row.units}
        smiles_set.update(s for row in self._rows for s in row.ends)
        cache: dict[str, tuple[list[str], np.ndarray]] = {}
        for smiles in sorted(smiles_set):
            symbols, xyz = smiles_to_3d_coords(smiles, random_seed=self.seed)
            cache[smiles] = (symbols, np.asarray(xyz, dtype=np.float64))
        return cache

    @staticmethod
    def _collect_type_map(
        coords_by_smiles: dict[str, tuple[list[str], np.ndarray]],
    ) -> list[str]:
        seen: dict[str, None] = {}
        for symbols, _ in coords_by_smiles.values():
            for sym in symbols:
                seen.setdefault(sym, None)
        return list(seen)

    def _build_schema(self) -> list[str]:
        """Ordered fparam column names: mn_log, scalar fparam fields, salt one-hots."""
        columns: list[str] = [MN_KEY]
        scalar_keys: set[str] = set()
        salt_names: set[str] = set()
        for row in self._rows:
            scalar_keys.update(row.scalars)
            salt_names.update(row.salts)
        columns.extend(sorted(scalar_keys))
        columns.extend(f"{SALT_PREFIX}{name}" for name in sorted(salt_names))
        return columns

    def _raw_vector(self, row: _PolymerRow, schema: Sequence[str]) -> np.ndarray:
        vec = np.zeros(len(schema), dtype=np.float64)
        for i, col in enumerate(schema):
            if col == MN_KEY:
                vec[i] = math.log10(row.mol_weight) if row.mol_weight else 0.0
            elif col.startswith(SALT_PREFIX):
                vec[i] = row.salts.get(col[len(SALT_PREFIX) :], 0.0)
            else:
                vec[i] = row.scalars.get(col, 0.0)
        return vec

    def _fit_stats(self, schema: Sequence[str]) -> dict[str, list[float]]:
        matrix = np.vstack([self._raw_vector(row, schema) for row in self._rows])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std == 0] = 1.0
        return {"mean": mean.tolist(), "std": std.tolist()}

    def _standardized_fparam(
        self,
        row: _PolymerRow,
        schema: Sequence[str],
        stats: Mapping[str, Sequence[float]],
    ) -> dict[str, float]:
        raw = self._raw_vector(row, schema)
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        z = (raw - mean) / std
        return {col: float(z[i]) for i, col in enumerate(schema)}


def _load_scaler(scaler: dict[str, Any] | str | Path | None) -> dict[str, Any] | None:
    if scaler is None:
        return None
    if isinstance(scaler, (str, Path)):
        return json.loads(Path(scaler).read_text(encoding="utf-8"))
    return dict(scaler)


def _molar_weight(
    smiles: str, cache: Mapping[str, tuple[list[str], np.ndarray]]
) -> float:
    """Approximate molar weight from the embedded atom symbols."""
    from rdkit import (
        Chem,
    )

    periodic_table = Chem.GetPeriodicTable()
    symbols, _ = cache[smiles]
    return float(
        sum(
            periodic_table.GetAtomicWeight(periodic_table.GetAtomicNumber(symbol))
            for symbol in symbols
        )
    )


def _end_share(
    ends: Sequence[str],
    units: Sequence[tuple[str, float]],
    mol_weight: float | None,
    cache: Mapping[str, tuple[list[str], np.ndarray]],
) -> float:
    """Mole-fraction share carried by each end group (mirrors calc_ends_share).

    Falls back to the mean unit fraction when Mn is unavailable.
    """
    if not ends:
        return 0.0
    if not mol_weight:
        return float(np.mean([frac for _, frac in units]))
    total_minus = mol_weight - sum(_molar_weight(s, cache) for s in ends)
    total_moles = 0.0
    for smiles, frac in units:
        total_moles += (total_minus * frac) / _molar_weight(smiles, cache)
    return 1.0 / total_moles if total_moles > 0 else 0.0
