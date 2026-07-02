# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent helpers for full validation during training."""

from __future__ import (
    annotations,
)

import logging
import re
import shutil
import traceback
from abc import (
    ABC,
    abstractmethod,
)
from contextlib import (
    nullcontext,
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

from deepmd.utils.argcheck import (
    normalize_full_validation_metric,
    resolve_full_validation_start_step,
)
from deepmd.utils.eval_metrics import (
    FULL_VALIDATION_METRIC_FAMILY_BY_KEY,
    FULL_VALIDATION_METRIC_KEY_MAP,
)

log = logging.getLogger(__name__)

LOG_COLUMN_ORDER = [
    ("E_MAE", "mae_e_per_atom"),
    ("E_RMSE", "rmse_e_per_atom"),
    ("F_MAE", "mae_f"),
    ("F_RMSE", "rmse_f"),
    ("V_MAE", "mae_v_per_atom"),
    ("V_RMSE", "rmse_v_per_atom"),
]

TOPK_RECORDS_INFO_KEY = "full_validation_topk_records"
BEST_METRIC_NAME_INFO_KEY = "full_validation_metric"
STALE_FULL_VALIDATION_INFO_KEYS = (
    "full_validation_best_metric",
    "full_validation_best_step",
    "full_validation_best_path",
    "full_validation_best_records",
)
BEST_CKPT_PREFIX = "best.ckpt"
VAL_LOG_SIGNIFICANT_DIGITS = 5
VAL_LOG_COLUMN_GAP = "   "
VAL_LOG_HEADER_PREFIX = "# "
VAL_LOG_DATA_PREFIX = "  "
METRIC_LOG_UNIT_MAP = {
    "e": ("meV/atom", 1000.0),
    "f": ("meV/Å", 1000.0),
    "v": ("meV/atom", 1000.0),
}


@dataclass(frozen=True)
class FullValidationResult:
    """Result of one full validation run."""

    display_step: int
    metrics: dict[str, float]
    selected_metric_key: str
    selected_metric_value: float
    saved_best_path: str | None


@dataclass(order=True, frozen=True)
class BestCheckpointRecord:
    """One best-checkpoint record ordered by metric then step."""

    metric: float
    step: int


def build_best_checkpoint_glob(
    best_checkpoint_prefix: str,
    best_checkpoint_suffix: str,
) -> str:
    """Build the glob pattern for managed best checkpoints."""
    return f"{best_checkpoint_prefix}-*.t-*{best_checkpoint_suffix}"


def build_best_checkpoint_pattern(
    best_checkpoint_prefix: str,
    best_checkpoint_suffix: str,
) -> re.Pattern[str]:
    """Build the regex pattern for managed best checkpoints."""
    return re.compile(
        rf"^{re.escape(best_checkpoint_prefix)}-(\d+)\.t-(\d+)"
        rf"{re.escape(best_checkpoint_suffix)}$"
    )


def resolve_best_checkpoint_dir(
    validating_params: dict[str, Any],
    save_ckpt: str,
) -> Path:
    """Resolve the directory for full-validation best checkpoints."""
    save_best_dir = validating_params.get("save_best_dir")
    if save_best_dir:
        return Path(save_best_dir)
    return Path(save_ckpt).parent


def parse_validation_metric(metric: str) -> tuple[str, str]:
    """Parse the configured full validation metric."""
    normalized_metric = normalize_full_validation_metric(metric)
    if normalized_metric not in FULL_VALIDATION_METRIC_KEY_MAP:
        supported_metrics = ", ".join(
            item.upper() for item in FULL_VALIDATION_METRIC_KEY_MAP
        )
        raise ValueError(
            "validating.validation_metric must be one of "
            f"{supported_metrics}, got {metric!r}."
        )
    return normalized_metric, FULL_VALIDATION_METRIC_KEY_MAP[normalized_metric]


def format_metric_for_log(
    metric_name: str, metric_value: float
) -> tuple[str, float, str]:
    """Format a full validation metric for user-facing logging."""
    metric_family, metric_kind = metric_name.split(":")
    metric_unit, metric_scale = METRIC_LOG_UNIT_MAP[metric_family]
    metric_label = f"{metric_family.upper()}:{metric_kind.upper()}"
    return metric_label, metric_value * metric_scale, metric_unit


def format_metric_value_for_table(
    metric_key: str, metric_value: float
) -> tuple[float, str]:
    """Format one table metric value and its unit for `val.log`."""
    metric_family = FULL_VALIDATION_METRIC_FAMILY_BY_KEY.get(metric_key)
    if metric_family is None:
        raise ValueError(f"Unknown full validation metric key: {metric_key}")
    metric_unit, metric_scale = METRIC_LOG_UNIT_MAP[metric_family]
    return metric_value * metric_scale, metric_unit


def format_metric_number_for_log(metric_value: float) -> str:
    """Format one metric value for `val.log` and best-save messages."""
    if np.isnan(metric_value):
        return "nan"
    if np.isposinf(metric_value):
        return "inf"
    if np.isneginf(metric_value):
        return "-inf"
    if metric_value == 0.0:
        return "0"
    abs_value = abs(metric_value)
    if abs_value < np.finfo(float).tiny:
        return "0"
    decimals = VAL_LOG_SIGNIFICANT_DIGITS - int(np.floor(np.log10(abs_value))) - 1
    if decimals > 16:
        return f"{metric_value:.{VAL_LOG_SIGNIFICANT_DIGITS - 1}e}"
    rounded_value = round(metric_value, decimals)
    if rounded_value == 0.0:
        rounded_value = 0.0
    if decimals > 0:
        return f"{rounded_value:.{decimals}f}"
    return f"{rounded_value:.0f}"


class FullValidatorBase(ABC):
    """Run independent full validation during backend-specific training."""

    def __init__(
        self,
        *,
        validating_params: dict[str, Any],
        state_store: dict[str, Any],
        num_steps: int,
        rank: int,
        restart_training: bool,
        checkpoint_dir: Path | None = None,
        full_val_file: str | Path | None = None,
        best_checkpoint_prefix: str = BEST_CKPT_PREFIX,
        best_checkpoint_suffix: str,
        metric_name_info_key: str = BEST_METRIC_NAME_INFO_KEY,
        topk_records_info_key: str = TOPK_RECORDS_INFO_KEY,
        stale_state_keys: tuple[str, ...] = STALE_FULL_VALIDATION_INFO_KEYS,
        emit_best_save_log: bool = True,
    ) -> None:
        self.state_store = state_store
        self.rank = rank
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else Path(".")
        )
        self.metric_name_info_key = metric_name_info_key
        self.topk_records_info_key = topk_records_info_key
        self.stale_state_keys = stale_state_keys
        self.best_checkpoint_prefix = best_checkpoint_prefix
        self.best_checkpoint_suffix = best_checkpoint_suffix
        self.best_checkpoint_glob = build_best_checkpoint_glob(
            best_checkpoint_prefix, best_checkpoint_suffix
        )
        self.best_checkpoint_pattern = build_best_checkpoint_pattern(
            best_checkpoint_prefix, best_checkpoint_suffix
        )
        self.emit_best_save_log = emit_best_save_log

        self.full_validation = bool(validating_params.get("full_validation", False))
        self.validation_freq = int(validating_params.get("validation_freq", 5000))
        self.save_best = bool(validating_params.get("save_best", True))
        self.max_best_ckpt = int(validating_params.get("max_best_ckpt", 1))
        self.metric_name, self.metric_key = parse_validation_metric(
            str(validating_params.get("validation_metric", "E:MAE"))
        )
        resolved_log_file = (
            full_val_file
            if full_val_file is not None
            else validating_params.get("full_val_file", "val.log")
        )
        self.full_val_file = Path(resolved_log_file)
        self.start_step = resolve_full_validation_start_step(
            validating_params.get("full_val_start", 0.5),
            num_steps,
        )
        self.enabled = (
            self.full_validation
            and self.start_step is not None
            and self.start_step <= num_steps
        )
        self.step_column_width = max(len("step"), len(str(num_steps)))
        self._write_mode = "a" if restart_training else "w"
        self._should_write_header = not (
            restart_training and self.full_val_file.exists()
        )
        self.table_column_specs = []
        for column_name, metric_key in LOG_COLUMN_ORDER:
            _, metric_unit = format_metric_value_for_table(metric_key, 1.0)
            header_label = f"{column_name}({metric_unit})"
            self.table_column_specs.append(
                (metric_key, header_label, max(len(header_label), 18))
            )

        self.topk_records = self._load_topk_records()
        self._sync_state_store()
        if self.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._initialize_best_checkpoints(restart_training=restart_training)

    def should_run(self, display_step: int) -> bool:
        """Check whether the current step should trigger full validation."""
        if not self.enabled or self.start_step is None:
            return False
        if display_step < self.start_step:
            return False
        return (display_step - self.start_step) % self.validation_freq == 0

    def run(
        self,
        *,
        step_id: int,
        display_step: int,
        lr: float,
        save_checkpoint: Any,
    ) -> FullValidationResult | None:
        """Run full validation if the current step is due."""
        if not self.should_run(display_step):
            return None

        result: FullValidationResult | None = None
        caught_exception: Exception | None = None
        error_message = None
        save_path: str | None = None
        if self.rank == 0:
            try:
                result = self._evaluate(display_step)
                save_path = result.saved_best_path
            except Exception as exc:
                caught_exception = exc
                error_message = (
                    "Full validation failed during evaluation:\n"
                    f"{traceback.format_exc()}"
                )

        self._raise_if_error(error_message, caught_exception)

        if save_path is not None and self.rank == 0:
            try:
                save_checkpoint(Path(save_path), lr=lr, step=step_id)
                self._reconcile_best_checkpoints()
            except Exception as exc:
                caught_exception = exc
                error_message = (
                    "Full validation failed while saving the best checkpoint:\n"
                    f"{traceback.format_exc()}"
                )
            else:
                error_message = None
                caught_exception = None

        self._raise_if_error(error_message, caught_exception)

        if self.rank == 0:
            try:
                self._log_result(result)
            except Exception as exc:
                caught_exception = exc
                error_message = (
                    "Full validation failed while writing logs:\n"
                    f"{traceback.format_exc()}"
                )
            else:
                error_message = None
                caught_exception = None

        self._raise_if_error(error_message, caught_exception)

        return result if self.rank == 0 else None

    def _evaluate(self, display_step: int) -> FullValidationResult:
        """Evaluate all validation systems and update best state."""
        with self.evaluation_context():
            metrics = self.evaluate_all_systems()

        if self.metric_key not in metrics or np.isnan(metrics[self.metric_key]):
            raise RuntimeError(
                "The selected full validation metric is unavailable on the "
                f"validation dataset: {self.metric_name.upper()}."
            )

        selected_metric_value = float(metrics[self.metric_key])
        saved_best_path = self._update_best_state(
            display_step=display_step,
            selected_metric_value=selected_metric_value,
        )
        return FullValidationResult(
            display_step=display_step,
            metrics=metrics,
            selected_metric_key=self.metric_key,
            selected_metric_value=selected_metric_value,
            saved_best_path=saved_best_path,
        )

    def evaluation_context(self) -> Any:
        """Return a context manager active during model evaluation."""
        return nullcontext()

    @abstractmethod
    def evaluate_all_systems(self) -> dict[str, float]:
        """Evaluate every validation system and aggregate metrics."""

    def _update_best_state(
        self,
        *,
        display_step: int,
        selected_metric_value: float,
    ) -> str | None:
        """Update the top-K records and return the checkpoint path to save."""
        candidate = BestCheckpointRecord(
            metric=selected_metric_value,
            step=display_step,
        )
        updated_records = [
            record for record in self.topk_records if record.step != display_step
        ]
        updated_records.append(candidate)
        updated_records.sort()
        updated_records = updated_records[: self.max_best_ckpt]
        if candidate not in updated_records:
            return None

        self.topk_records = updated_records
        self._sync_state_store()
        if not self.save_best:
            return None
        candidate_rank = self.topk_records.index(candidate) + 1
        return str(self._best_checkpoint_path(display_step, candidate_rank))

    def _sync_state_store(self) -> None:
        """Synchronize top-K validation state into the configured state store."""
        for key in self.stale_state_keys:
            self.state_store.pop(key, None)
        self.state_store[self.metric_name_info_key] = self.metric_name
        self.state_store[self.topk_records_info_key] = [
            {"metric": record.metric, "step": record.step}
            for record in self.topk_records
        ]

    def _load_topk_records(self) -> list[BestCheckpointRecord]:
        """Load top-K records from the configured state store."""
        if self.state_store.get(self.metric_name_info_key) != self.metric_name:
            return []
        raw_records = self.state_store.get(self.topk_records_info_key, [])
        if not isinstance(raw_records, list):
            return []
        records = []
        for raw_record in raw_records:
            if not isinstance(raw_record, dict):
                continue
            if "metric" not in raw_record or "step" not in raw_record:
                continue
            records.append(
                BestCheckpointRecord(
                    metric=float(raw_record["metric"]),
                    step=int(raw_record["step"]),
                )
            )
        records.sort()
        return records[: self.max_best_ckpt]

    def _best_checkpoint_name(self, step: int, rank: int) -> str:
        """Build the best-checkpoint filename for one step."""
        return f"{self.best_checkpoint_prefix}-{step}.t-{rank}{self.best_checkpoint_suffix}"

    def _best_checkpoint_path(self, step: int, rank: int) -> Path:
        """Build the best-checkpoint path for one step."""
        return self.checkpoint_dir / self._best_checkpoint_name(step, rank)

    def _list_best_checkpoints(self) -> list[Path]:
        """List all managed best checkpoints in the checkpoint directory."""
        best_checkpoints = [
            path
            for path in self.checkpoint_dir.glob(self.best_checkpoint_glob)
            if path.exists() and not path.is_symlink()
        ]
        best_checkpoints.sort(key=lambda path: path.stat().st_mtime)
        return best_checkpoints

    @staticmethod
    def _remove_checkpoint_path(path: Path) -> None:
        """Remove one managed checkpoint path, file or directory."""
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    def _expected_topk_checkpoint_names(self) -> dict[int, str]:
        """Return the expected checkpoint filename for each retained step."""
        return {
            record.step: self._best_checkpoint_name(record.step, rank)
            for rank, record in enumerate(self.topk_records, start=1)
        }

    def _reconcile_best_checkpoints(self) -> None:
        """Rename retained best checkpoints to ranked names and delete stale ones."""
        expected_names = self._expected_topk_checkpoint_names()
        current_files = self._list_best_checkpoints()
        files_by_step: dict[int, list[Path]] = {}
        stale_files: list[Path] = []
        for checkpoint_path in current_files:
            match = self.best_checkpoint_pattern.match(checkpoint_path.name)
            if match is None:
                stale_files.append(checkpoint_path)
                continue
            step = int(match.group(1))
            files_by_step.setdefault(step, []).append(checkpoint_path)

        temp_moves: list[tuple[Path, Path]] = []
        for step, checkpoint_paths in files_by_step.items():
            expected_name = expected_names.get(step)
            if expected_name is None:
                stale_files.extend(checkpoint_paths)
                continue

            keep_path = next(
                (
                    checkpoint_path
                    for checkpoint_path in checkpoint_paths
                    if checkpoint_path.name == expected_name
                ),
                checkpoint_paths[0],
            )
            for checkpoint_path in checkpoint_paths:
                if checkpoint_path != keep_path:
                    stale_files.append(checkpoint_path)
            if keep_path.name != expected_name:
                temp_path = keep_path.with_name(f"{keep_path.name}.tmp")
                keep_path.rename(temp_path)
                temp_moves.append((temp_path, keep_path.with_name(expected_name)))

        for checkpoint_path in stale_files:
            self._remove_checkpoint_path(checkpoint_path)
        for temp_path, final_path in temp_moves:
            self._remove_checkpoint_path(final_path)
            temp_path.rename(final_path)

    def _initialize_best_checkpoints(self, restart_training: bool) -> None:
        """Align on-disk best checkpoints with the current training mode."""
        if restart_training and self.save_best and self.topk_records:
            self._reconcile_best_checkpoints()
            return
        for checkpoint_path in self._list_best_checkpoints():
            self._remove_checkpoint_path(checkpoint_path)

    def _raise_if_error(
        self,
        error_message: str | None,
        local_exception: Exception | None = None,
    ) -> None:
        """Raise a full-validation error if one occurred."""
        propagated_error = self.propagate_error(error_message)
        if propagated_error is None:
            return
        if local_exception is not None:
            raise RuntimeError(propagated_error) from local_exception
        raise RuntimeError(propagated_error)

    def propagate_error(self, error_message: str | None) -> str | None:
        """Propagate a rank-0 full-validation error to backend peers if needed."""
        return error_message

    def _log_result(self, result: FullValidationResult | None) -> None:
        """Log and persist full validation results on rank 0."""
        if result is None:
            raise ValueError("Full validation logging requires a result on rank 0.")
        self._write_log_file(result)
        if self.emit_best_save_log and result.saved_best_path is not None:
            metric_label, metric_value, metric_unit = format_metric_for_log(
                self.metric_name, result.selected_metric_value
            )
            log.info(
                f"Saved best model to {result.saved_best_path} "
                f"with {metric_label} = {format_metric_number_for_log(metric_value)} "
                f"{metric_unit}"
            )

    def _write_log_file(self, result: FullValidationResult) -> None:
        """Append one full validation entry to the dedicated log file."""
        with self.full_val_file.open(self._write_mode, buffering=1) as fout:
            if self._should_write_header:
                header = VAL_LOG_HEADER_PREFIX + f"{'step':^{self.step_column_width}s}"
                for _, header_label, column_width in self.table_column_specs:
                    header += VAL_LOG_COLUMN_GAP + f"{header_label:^{column_width}s}"
                header += "\n"
                header += (
                    "# E uses per-atom energy, F uses component-wise force errors, "
                    "and V uses virial normalized by natoms.\n"
                )
                fout.write(header)
                self._should_write_header = False
                self._write_mode = "a"

            line = (
                VAL_LOG_DATA_PREFIX
                + f"{result.display_step:^{self.step_column_width}d}"
            )
            for metric_key, _, column_width in self.table_column_specs:
                metric_value = result.metrics.get(metric_key, float("nan"))
                if not np.isnan(metric_value):
                    metric_value, _ = format_metric_value_for_table(
                        metric_key, metric_value
                    )
                metric_text = format_metric_number_for_log(metric_value)
                line += VAL_LOG_COLUMN_GAP + f"{metric_text:^{column_width}s}"
            line += "\n"
            fout.write(line)
            if result.saved_best_path is not None:
                metric_label, metric_value, metric_unit = format_metric_for_log(
                    self.metric_name, result.selected_metric_value
                )
                fout.write(
                    "# saved best checkpoint: "
                    f"{result.saved_best_path} ({metric_label} = "
                    f"{format_metric_number_for_log(metric_value)} {metric_unit})\n"
                )
