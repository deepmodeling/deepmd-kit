# SPDX-License-Identifier: LGPL-3.0-or-later
"""Post-training prediction calibration."""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
    field,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )

import numpy as np

_GROUP_STAT_NAMES = (
    "n_frames",
    "weight_sum",
    "weight_mean",
    "weight_std",
    "weight_min",
    "weight_max",
    "weight_eff_n",
    "weight_l2",
)


@dataclass
class Calibrator:
    """Post-training prediction correction.

    Supported feature sources:

    - ``"prediction"``: raw model prediction columns.
    - ``"fparam"``: per-frame or per-group ``set.*/fparam.npy`` columns.
    - ``"group_stats"``: grouped marker statistics derived from
      ``group_id.npy`` and ``weight.npy``; ordinary non-grouped data gets the
      one-frame/unweighted equivalent.
    """

    method: str = "ridge"
    alpha: float = 1.0
    features: Sequence[str] = ("prediction",)
    model: object | None = field(default=None, init=False, repr=False)
    task_dim: int | None = field(default=None, init=False)
    feature_names_: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        self.method = str(self.method).lower()
        if self.method not in {"ridge", "linear"}:
            raise ValueError(
                f"Calibrator method must be 'ridge' or 'linear'; got {self.method!r}."
            )
        self.features = tuple(self.features)
        if not self.features:
            raise ValueError("Calibrator requires at least one feature.")
        supported = {"prediction", "fparam", "group_stats"}
        unsupported = sorted(set(self.features) - supported)
        if unsupported:
            raise ValueError(
                f"Unsupported Calibrator feature sources: {unsupported}. "
                f"Supported sources are {sorted(supported)}."
            )
        if self.alpha < 0.0:
            raise ValueError(f"alpha must be non-negative; got {self.alpha}.")

    def fit_from_arrays(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        *,
        data: str | list[str] | None = None,
        fmt: str | None = None,
    ) -> Calibrator:
        x, names = self._build_features(predictions, data=data, fmt=fmt)
        y = np.asarray(labels, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Calibration row mismatch: features have {x.shape[0]} rows, "
                f"labels have {y.shape[0]} rows."
            )

        from sklearn.linear_model import (
            LinearRegression,
            Ridge,
        )

        if self.method == "ridge":
            self.model = Ridge(alpha=float(self.alpha))
        else:
            self.model = LinearRegression()
        self.model.fit(x, y)
        self.task_dim = int(y.shape[1])
        self.feature_names_ = tuple(names)
        return self

    def fit(
        self, model: object, data: str | list[str], fmt: str | None = None
    ) -> Calibrator:
        result = model.evaluate(data, fmt=fmt, calibrated=False)
        return self.fit_from_arrays(
            result.predictions,
            result.labels,
            data=data,
            fmt=fmt,
        )

    def predict_from_arrays(
        self,
        predictions: np.ndarray,
        *,
        data: str | list[str] | None = None,
        fmt: str | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator.predict() called before fit().")
        x, names = self._build_features(predictions, data=data, fmt=fmt)
        if self.feature_names_ and tuple(names) != self.feature_names_:
            raise ValueError(
                "Calibration feature columns differ from fit-time columns.\n"
                f"  fit     : {list(self.feature_names_)}\n"
                f"  predict : {names}"
            )
        out = np.asarray(self.model.predict(x), dtype=float)
        task_dim = self.task_dim or (1 if out.ndim == 1 else out.shape[-1])
        return out.reshape(-1, task_dim)

    def predict(
        self, model: object, data: str | list[str], fmt: str | None = None
    ) -> np.ndarray:
        result = model.predict(data, fmt=fmt, calibrated=False)
        return self.predict_from_arrays(result.predictions, data=data, fmt=fmt)

    def _build_features(
        self,
        predictions: np.ndarray,
        *,
        data: str | list[str] | None,
        fmt: str | None,
    ) -> tuple[np.ndarray, list[str]]:
        pred = np.asarray(predictions, dtype=float)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        pred = pred.reshape(pred.shape[0], -1)

        parts: list[np.ndarray] = []
        names: list[str] = []
        if "prediction" in self.features:
            parts.append(pred)
            names.extend([f"prediction_{i}" for i in range(pred.shape[1])])

        need_data = any(source in self.features for source in ("fparam", "group_stats"))
        rows = None
        if need_data:
            if data is None:
                raise ValueError(
                    "Calibrator features 'fparam' and 'group_stats' require data."
                )
            rows = _feature_rows_from_data(data, fmt=fmt, expected_rows=pred.shape[0])

        if "fparam" in self.features:
            assert rows is not None
            if rows.fparam is None:
                raise ValueError(
                    "Calibrator feature 'fparam' requested, but no fparam.npy was found."
                )
            parts.append(rows.fparam)
            names.extend([f"fparam_{i}" for i in range(rows.fparam.shape[1])])

        if "group_stats" in self.features:
            assert rows is not None
            parts.append(rows.group_stats)
            names.extend(_GROUP_STAT_NAMES)

        if not parts:
            raise ValueError("No calibration features were selected.")
        return np.concatenate(parts, axis=1), names


@dataclass(frozen=True)
class _FeatureRows:
    fparam: np.ndarray | None
    group_stats: np.ndarray


def _feature_rows_from_data(
    data: str | list[str], *, fmt: str | None, expected_rows: int
) -> _FeatureRows:
    from dpa_adapt.data.loader import (
        _get_source,
        load_data,
    )

    systems = load_data(data, fmt=fmt)
    grouped_records = []
    frame_fparams = []
    frame_stats = []
    any_grouped = False
    any_fparam = False
    next_group_id = 0

    for system in systems:
        source = _get_source(system)
        if source is None:
            raise ValueError(
                "Calibration fparam/group_stats require filesystem-backed deepmd/npy data."
            )
        source_path = Path(source)
        local_to_global: dict[int, int] = {}
        for set_dir in sorted(source_path.glob("set.*")):
            nframes, natoms = _set_shape(set_dir)
            gid_path = set_dir / "group_id.npy"
            if gid_path.is_file():
                group_ids = np.load(gid_path).reshape(-1).astype(np.int64)
                if group_ids.shape != (nframes,):
                    raise ValueError(
                        f"{gid_path} shape {group_ids.shape}; expected ({nframes},)."
                    )
                any_grouped = True
            else:
                group_ids = np.arange(nframes, dtype=np.int64)

            weight_path = set_dir / "weight.npy"
            weights = (
                np.asarray(np.load(weight_path), dtype=float).reshape(-1)
                if weight_path.is_file()
                else np.ones(nframes, dtype=float)
            )
            if weights.shape != (nframes,):
                raise ValueError(
                    f"{weight_path} shape {weights.shape}; expected ({nframes},)."
                )

            fp_path = set_dir / "fparam.npy"
            fparam = None
            if fp_path.is_file():
                fparam = np.asarray(np.load(fp_path), dtype=float).reshape(nframes, -1)
                any_fparam = True

            for frame in range(nframes):
                frame_stats.append(_stats_for_weights(weights[frame : frame + 1]))
                if fparam is not None:
                    frame_fparams.append(fparam[frame])

            for local_gid in group_ids:
                if int(local_gid) not in local_to_global:
                    local_to_global[int(local_gid)] = next_group_id
                    next_group_id += 1
            for frame, local_gid in enumerate(group_ids):
                grouped_records.append(
                    (
                        local_to_global[int(local_gid)],
                        float(weights[frame]),
                        None if fparam is None else fparam[frame],
                    )
                )

    # ``any_fparam`` only turns True once a set carrying fparam.npy has been
    # read, so a set *before* that one contributes no fparam row and cannot be
    # caught while looping. Compare the collected row counts once every system
    # has been read, otherwise the gap only surfaces as an opaque concatenate
    # shape error further down.
    if any_fparam and len(frame_fparams) != len(frame_stats):
        raise ValueError(
            "fparam.npy must be present for every set when Calibrator uses fparam."
        )

    if any_grouped:
        group_ids = sorted({record[0] for record in grouped_records})
        group_stats = []
        group_fparams = []
        for gid in group_ids:
            records = [record for record in grouped_records if record[0] == gid]
            weights = np.asarray([record[1] for record in records], dtype=float)
            group_stats.append(_stats_for_weights(weights))
            fps = [record[2] for record in records if record[2] is not None]
            if fps:
                arr = np.asarray(fps, dtype=float)
                first = arr[0]
                if not np.allclose(arr, first[None, :], atol=1e-8):
                    raise ValueError(
                        f"fparam must be constant within grouped calibration data; group {gid} differs."
                    )
                group_fparams.append(first)
            elif any_fparam:
                raise ValueError(
                    "fparam.npy must be present for every grouped frame when Calibrator uses fparam."
                )
        rows = _FeatureRows(
            fparam=np.asarray(group_fparams, dtype=float) if any_fparam else None,
            group_stats=np.asarray(group_stats, dtype=float),
        )
        if rows.group_stats.shape[0] == expected_rows:
            return rows

    frame_rows = _FeatureRows(
        fparam=np.asarray(frame_fparams, dtype=float) if any_fparam else None,
        group_stats=np.asarray(frame_stats, dtype=float),
    )
    if frame_rows.group_stats.shape[0] == expected_rows:
        return frame_rows

    raise ValueError(
        f"Could not align calibration features to predictions: got "
        f"{frame_rows.group_stats.shape[0]} frame rows"
        + (f" and {rows.group_stats.shape[0]} group rows" if any_grouped else "")
        + f", expected {expected_rows}."
    )


def _set_shape(set_dir: Path) -> tuple[int, int]:
    real_path = set_dir / "real_atom_types.npy"
    if real_path.is_file():
        arr = np.load(real_path, mmap_mode="r")
        return int(arr.shape[0]), int(arr.shape[1])
    coord = np.load(set_dir / "coord.npy", mmap_mode="r")
    return int(coord.shape[0]), int(coord.shape[1] // 3)


def _stats_for_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    l2_sq = float(np.sum(w**2))
    weight_sum = float(np.sum(w))
    eff_n = float((weight_sum**2) / l2_sq) if l2_sq > 1e-12 else 0.0
    return np.asarray(
        [
            float(len(w)),
            weight_sum,
            float(np.mean(w)) if len(w) else 0.0,
            float(np.std(w)) if len(w) else 0.0,
            float(np.min(w)) if len(w) else 0.0,
            float(np.max(w)) if len(w) else 0.0,
            eff_n,
            float(np.sqrt(l2_sq)),
        ],
        dtype=float,
    )
