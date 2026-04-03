#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import (
    annotations,
)

import logging
import re
import traceback
from dataclasses import (
    dataclass,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import torch
import torch.distributed as dist

from deepmd.dpmodel.common import PRECISION_DICT as NP_PRECISION_DICT
from deepmd.pt.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.pt.utils.dataset import (
    DeepmdDataSetForLoader,
)
from deepmd.pt.utils.env import (
    DEVICE,
    GLOBAL_PT_FLOAT_PRECISION,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)
from deepmd.utils.argcheck import (
    normalize_full_validation_metric,
)
from deepmd.utils.weight_avg import (
    weighted_average,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from deepmd.utils.data import (
        DeepmdData,
    )

METRIC_KEY_MAP = {
    "e:mae": "mae_e_per_atom",
    "e:rmse": "rmse_e_per_atom",
    "f:mae": "mae_f_vector",
    "f:rmse": "rmse_f_vector",
    "v:mae": "mae_v_per_atom",
    "v:rmse": "rmse_v_per_atom",
}
LOG_COLUMN_ORDER = [
    ("E_MAE", "mae_e_per_atom"),
    ("E_RMSE", "rmse_e_per_atom"),
    ("F_MAE", "mae_f_vector"),
    ("F_RMSE", "rmse_f_vector"),
    ("V_MAE", "mae_v_per_atom"),
    ("V_RMSE", "rmse_v_per_atom"),
]

TOPK_RECORDS_INFO_KEY = "full_validation_topk_records"
BEST_METRIC_NAME_INFO_KEY = "full_validation_metric"
BEST_CKPT_GLOB = "best.ckpt-*.t-*.pt"
BEST_CKPT_PATTERN = re.compile(r"^best\.ckpt-(\d+)\.t-(\d+)\.pt$")
STALE_FULL_VALIDATION_INFO_KEYS = (
    "full_validation_best_metric",
    "full_validation_best_step",
    "full_validation_best_path",
    "full_validation_best_records",
)
BATCH_SIZE_LOGGER_NAME = "deepmd.utils.batch_size"
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


def resolve_full_validation_start_step(
    full_val_start: float, num_steps: int
) -> int | None:
    """Resolve the first step at which full validation becomes active."""
    start_value = float(full_val_start)
    if start_value == 1.0:
        return None
    if 0.0 <= start_value < 1.0:
        return int(num_steps * start_value)
    return int(start_value)


def parse_validation_metric(metric: str) -> tuple[str, str]:
    """Parse the configured full validation metric."""
    normalized_metric = normalize_full_validation_metric(metric)
    if normalized_metric not in METRIC_KEY_MAP:
        supported_metrics = ", ".join(item.upper() for item in METRIC_KEY_MAP)
        raise ValueError(
            "validating.validation_metric must be one of "
            f"{supported_metrics}, got {metric!r}."
        )
    return normalized_metric, METRIC_KEY_MAP[normalized_metric]


def _compute_system_metrics(
    prediction: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    natoms: int,
    has_pbc: bool,
) -> dict[str, tuple[float, float]]:
    """Compute per-system full validation metrics."""
    metrics: dict[str, tuple[float, float]] = {}
    find_energy = bool(test_data.get("find_energy", 0.0))
    find_force = bool(test_data.get("find_force", 0.0))
    find_virial = bool(test_data.get("find_virial", 0.0))

    if find_energy:
        diff_e = prediction["energy"].reshape(-1, 1) - test_data["energy"].reshape(
            -1, 1
        )
        mae_e_per_atom = float(np.mean(np.abs(diff_e)) / natoms)
        rmse_e_per_atom = float(np.sqrt(np.mean(diff_e * diff_e)) / natoms)
        metrics["mae_e_per_atom"] = (mae_e_per_atom, float(diff_e.size))
        metrics["rmse_e_per_atom"] = (rmse_e_per_atom, float(diff_e.size))

    if find_force:
        diff_f = prediction["force"].reshape(-1, 3) - test_data["force"].reshape(-1, 3)
        diff_f_norm = np.linalg.vector_norm(diff_f, axis=1)
        mae_f_vector = float(np.mean(diff_f_norm))
        rmse_f_vector = float(np.sqrt(np.mean(diff_f_norm * diff_f_norm)))
        metrics["mae_f_vector"] = (mae_f_vector, float(diff_f_norm.size))
        metrics["rmse_f_vector"] = (rmse_f_vector, float(diff_f_norm.size))

    if has_pbc and find_virial:
        diff_v = prediction["virial"].reshape(-1, 9) - test_data["virial"].reshape(
            -1, 9
        )
        mae_v_per_atom = float(np.mean(np.abs(diff_v)) / natoms)
        rmse_v_per_atom = float(np.sqrt(np.mean(diff_v * diff_v)) / natoms)
        metrics["mae_v_per_atom"] = (mae_v_per_atom, float(diff_v.size))
        metrics["rmse_v_per_atom"] = (rmse_v_per_atom, float(diff_v.size))

    return metrics


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
    if metric_key.endswith("_e_per_atom"):
        metric_family = "e"
    elif metric_key.endswith("_f_vector"):
        metric_family = "f"
    elif metric_key.endswith("_v_per_atom"):
        metric_family = "v"
    else:
        raise ValueError(f"Unknown full validation metric key: {metric_key}")
    metric_unit, metric_scale = METRIC_LOG_UNIT_MAP[metric_family]
    return metric_value * metric_scale, metric_unit


def format_metric_number_for_log(metric_value: float) -> str:
    """Format one metric value for `val.log` and best-save messages."""
    if np.isnan(metric_value):
        return "nan"
    if metric_value == 0.0:
        return "0"
    abs_value = abs(metric_value)
    decimals = VAL_LOG_SIGNIFICANT_DIGITS - int(np.floor(np.log10(abs_value))) - 1
    rounded_value = round(metric_value, decimals)
    if rounded_value == 0.0:
        rounded_value = 0.0
    if decimals > 0:
        return f"{rounded_value:.{decimals}f}"
    return f"{rounded_value:.0f}"


class SilentAutoBatchSize(AutoBatchSize):
    """Auto batch size that does not emit adjustment logs."""

    def __init__(
        self,
        initial_batch_size: int = 1024,
        factor: float = 2.0,
    ) -> None:
        batch_size_log = logging.getLogger(BATCH_SIZE_LOGGER_NAME)
        old_disabled = batch_size_log.disabled
        batch_size_log.disabled = True
        try:
            super().__init__(
                initial_batch_size=initial_batch_size,
                factor=factor,
            )
        finally:
            batch_size_log.disabled = old_disabled

    def _adjust_batch_size(self, factor: float) -> None:
        self.current_batch_size = int(self.current_batch_size * factor)


class FullValidator:
    """Run independent full validation during training."""

    def __init__(
        self,
        *,
        validating_params: dict[str, Any],
        validation_data: Any,
        model: torch.nn.Module,
        train_infos: dict[str, Any],
        num_steps: int,
        rank: int,
        zero_stage: int,
        restart_training: bool,
    ) -> None:
        self.validation_data = validation_data
        self.model = model
        self.train_infos = train_infos
        self.rank = rank
        self.zero_stage = zero_stage
        self.is_distributed = dist.is_available() and dist.is_initialized()

        self.full_validation = bool(validating_params.get("full_validation", False))
        self.validation_freq = int(validating_params.get("validation_freq", 5000))
        self.save_best = bool(validating_params.get("save_best", True))
        self.max_best_ckpt = int(validating_params.get("max_best_ckpt", 1))
        self.metric_name, self.metric_key = parse_validation_metric(
            str(validating_params.get("validation_metric", "E:MAE"))
        )
        self.full_val_file = Path(validating_params.get("full_val_file", "val.log"))
        self.start_step = resolve_full_validation_start_step(
            validating_params.get("full_val_start", 0.0),
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
        self.auto_batch_size = SilentAutoBatchSize()
        self.table_column_specs = []
        for column_name, metric_key in LOG_COLUMN_ORDER:
            _, metric_unit = format_metric_value_for_table(metric_key, 1.0)
            header_label = f"{column_name}({metric_unit})"
            self.table_column_specs.append(
                (metric_key, header_label, max(len(header_label), 18))
            )

        self.topk_records = self._load_topk_records()
        self._sync_train_infos()
        if self.rank == 0:
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

        if self.is_distributed:
            dist.barrier()

        result: FullValidationResult | None = None
        caught_exception: Exception | None = None
        error_message = None
        save_path = [None]
        if self.rank == 0:
            try:
                result = self._evaluate(display_step)
                save_path[0] = result.saved_best_path
            except Exception as exc:
                caught_exception = exc
                error_message = (
                    "Full validation failed on rank 0 during evaluation:\n"
                    f"{traceback.format_exc()}"
                )

        self._raise_if_distributed_error(error_message, caught_exception)

        if self.is_distributed:
            dist.broadcast_object_list(save_path, src=0)

        if save_path[0] is not None:
            try:
                if not self.is_distributed or self.zero_stage == 0:
                    if self.rank == 0:
                        save_checkpoint(Path(save_path[0]), lr=lr, step=step_id)
                else:
                    save_checkpoint(Path(save_path[0]), lr=lr, step=step_id)
                if self.rank == 0:
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

            self._raise_if_distributed_error(error_message, caught_exception)

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

        self._raise_if_distributed_error(error_message, caught_exception)

        if self.is_distributed:
            dist.barrier()

        return result if self.rank == 0 else None

    def _evaluate(self, display_step: int) -> FullValidationResult:
        """Evaluate all validation systems and update best state."""
        # === Step 1. Switch to Evaluation Mode ===
        was_training = bool(getattr(self.model, "training", True))
        self.model.eval()
        try:
            # === Step 2. Evaluate All Systems ===
            metrics = self.evaluate_all_systems()
        finally:
            self.model.train(was_training)

        if self.metric_key not in metrics or np.isnan(metrics[self.metric_key]):
            raise RuntimeError(
                "The selected full validation metric is unavailable on the "
                f"validation dataset: {self.metric_name.upper()}."
            )

        # === Step 3. Update Best Tracking ===
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

    def evaluate_all_systems(self) -> dict[str, float]:
        """Evaluate every validation system and aggregate metrics."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        system_metrics = []
        for dataset in self.validation_data.systems:
            if not isinstance(dataset, DeepmdDataSetForLoader):
                raise TypeError(
                    "Full validation expects each dataset in validation_data.systems "
                    f"to be DeepmdDataSetForLoader, got {type(dataset)!r}."
                )
            system_metrics.append(self._evaluate_system(dataset.data_system))

        aggregated = weighted_average([metric for metric in system_metrics if metric])
        return {
            metric_key: float(aggregated[metric_key])
            for _, metric_key in LOG_COLUMN_ORDER
            if metric_key in aggregated
        }

    def _evaluate_system(
        self, data_system: DeepmdData
    ) -> dict[str, tuple[float, float]]:
        """Evaluate one validation system."""
        test_data = data_system.get_test()
        natoms = int(test_data["type"].shape[1])
        nframes = int(test_data["coord"].shape[0])
        prediction = self._predict_outputs(
            coord=test_data["coord"].reshape(nframes, -1),
            atom_types=test_data["type"],
            box=test_data["box"] if data_system.pbc else None,
            fparam=test_data["fparam"]
            if bool(test_data.get("find_fparam", 0.0))
            else None,
            aparam=test_data["aparam"] if self.model.get_dim_aparam() > 0 else None,
            natoms=natoms,
            nframes=nframes,
        )
        return _compute_system_metrics(
            prediction=prediction,
            test_data=test_data,
            natoms=natoms,
            has_pbc=data_system.pbc,
        )

    def _predict_outputs(
        self,
        *,
        coord: np.ndarray,
        atom_types: np.ndarray,
        box: np.ndarray | None,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        natoms: int,
        nframes: int,
    ) -> dict[str, np.ndarray]:
        """Predict energy, force, and virial for the full validation batch."""

        def predict_batch(
            coord_batch: np.ndarray,
            atom_types_batch: np.ndarray,
            box_batch: np.ndarray | None,
            fparam_batch: np.ndarray | None,
            aparam_batch: np.ndarray | None,
        ) -> dict[str, np.ndarray]:
            coord_input = torch.tensor(
                coord_batch.reshape(-1, natoms, 3).astype(
                    NP_PRECISION_DICT[
                        RESERVED_PRECISION_DICT[GLOBAL_PT_FLOAT_PRECISION]
                    ]
                ),
                dtype=GLOBAL_PT_FLOAT_PRECISION,
                device=DEVICE,
            )
            type_input = torch.tensor(
                atom_types_batch.astype(np.int64),
                dtype=torch.long,
                device=DEVICE,
            )
            if box_batch is not None:
                box_input = torch.tensor(
                    box_batch.reshape(-1, 3, 3).astype(
                        NP_PRECISION_DICT[
                            RESERVED_PRECISION_DICT[GLOBAL_PT_FLOAT_PRECISION]
                        ]
                    ),
                    dtype=GLOBAL_PT_FLOAT_PRECISION,
                    device=DEVICE,
                )
            else:
                box_input = None
            if fparam_batch is not None:
                fparam_input = to_torch_tensor(
                    fparam_batch.reshape(-1, self.model.get_dim_fparam())
                )
            else:
                fparam_input = None
            if aparam_batch is not None:
                aparam_input = to_torch_tensor(
                    aparam_batch.reshape(-1, natoms, self.model.get_dim_aparam())
                )
            else:
                aparam_input = None

            batch_output = self.model(
                coord_input,
                type_input,
                box=box_input,
                fparam=fparam_input,
                aparam=aparam_input,
            )
            if isinstance(batch_output, tuple):
                batch_output = batch_output[0]

            return {
                "energy": batch_output["energy"].detach().cpu().numpy().reshape(-1, 1),
                "force": batch_output["force"]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, natoms * 3),
                "virial": batch_output["virial"].detach().cpu().numpy().reshape(-1, 9),
            }

        prediction = self.auto_batch_size.execute_all(
            predict_batch,
            nframes,
            natoms,
            coord,
            atom_types,
            box,
            fparam,
            aparam,
        )
        return {
            "energy": prediction["energy"],
            "force": prediction["force"],
            "virial": prediction["virial"],
        }

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
        self._sync_train_infos()
        if not self.save_best:
            return None
        candidate_rank = self.topk_records.index(candidate) + 1
        return self._best_checkpoint_name(display_step, candidate_rank)

    def _sync_train_infos(self) -> None:
        """Synchronize top-K validation state into train infos."""
        for key in STALE_FULL_VALIDATION_INFO_KEYS:
            self.train_infos.pop(key, None)
        self.train_infos[BEST_METRIC_NAME_INFO_KEY] = self.metric_name
        self.train_infos[TOPK_RECORDS_INFO_KEY] = [
            {"metric": record.metric, "step": record.step}
            for record in self.topk_records
        ]

    def _load_topk_records(self) -> list[BestCheckpointRecord]:
        """Load top-K records from train infos for the current metric."""
        if self.train_infos.get(BEST_METRIC_NAME_INFO_KEY) != self.metric_name:
            return []
        raw_records = self.train_infos.get(TOPK_RECORDS_INFO_KEY, [])
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
        return f"best.ckpt-{step}.t-{rank}.pt"

    def _list_best_checkpoints(self) -> list[Path]:
        """List all managed best checkpoints in the working directory."""
        best_checkpoints = [
            path
            for path in Path(".").glob(BEST_CKPT_GLOB)
            if path.is_file() and not path.is_symlink()
        ]
        best_checkpoints.sort(key=lambda path: path.stat().st_mtime)
        return best_checkpoints

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
            match = BEST_CKPT_PATTERN.match(checkpoint_path.name)
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
            checkpoint_path.unlink(missing_ok=True)
        for temp_path, final_path in temp_moves:
            final_path.unlink(missing_ok=True)
            temp_path.rename(final_path)

    def _initialize_best_checkpoints(self, restart_training: bool) -> None:
        """Align on-disk best checkpoints with the current training mode."""
        if restart_training and self.save_best and self.topk_records:
            self._reconcile_best_checkpoints()
            return
        for checkpoint_path in self._list_best_checkpoints():
            checkpoint_path.unlink(missing_ok=True)

    def _raise_if_distributed_error(
        self,
        local_error_message: str | None,
        local_exception: Exception | None = None,
    ) -> None:
        """Propagate a local error to all ranks and raise consistently."""
        error_message = local_error_message
        if self.is_distributed:
            gathered_errors = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_errors, local_error_message)
            error_message = next(
                (message for message in gathered_errors if message is not None), None
            )
        if error_message is None:
            return
        if local_exception is not None:
            raise RuntimeError(error_message) from local_exception
        raise RuntimeError(error_message)

    def _log_result(self, result: FullValidationResult | None) -> None:
        """Log and persist full validation results on rank 0."""
        assert result is not None
        self._write_log_file(result)
        if result.saved_best_path is not None:
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
                    "# E uses per-atom energy, F uses per-atom force-vector L2 "
                    "norms, and V uses virial normalized by natoms.\n"
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
