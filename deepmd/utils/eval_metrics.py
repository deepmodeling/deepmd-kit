# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)

import numpy as np

FULL_VALIDATION_METRIC_KEY_MAP = {
    "e:mae": "mae_e_per_atom",
    "e:rmse": "rmse_e_per_atom",
    "f:mae": "mae_f",
    "f:rmse": "rmse_f",
    "v:mae": "mae_v_per_atom",
    "v:rmse": "rmse_v_per_atom",
}
FULL_VALIDATION_WEIGHTED_METRIC_KEYS = {
    "energy_per_atom": ("mae_e_per_atom", "rmse_e_per_atom"),
    "force": ("mae_f", "rmse_f"),
    "virial_per_atom": ("mae_v_per_atom", "rmse_v_per_atom"),
}
FULL_VALIDATION_METRIC_FAMILY_BY_KEY = {
    "mae_e_per_atom": "e",
    "rmse_e_per_atom": "e",
    "mae_f": "f",
    "rmse_f": "f",
    "mae_v_per_atom": "v",
    "rmse_v_per_atom": "v",
}
DP_TEST_WEIGHTED_METRIC_KEYS = {
    "energy": ("mae_e", "rmse_e"),
    "energy_per_atom": ("mae_ea", "rmse_ea"),
    "force": ("mae_f", "rmse_f"),
    "virial": ("mae_v", "rmse_v"),
    "virial_per_atom": ("mae_va", "rmse_va"),
}
DP_TEST_SPIN_WEIGHTED_METRIC_KEYS = {
    "force_real": ("mae_fr", "rmse_fr"),
    "force_magnetic": ("mae_fm", "rmse_fm"),
}
DP_TEST_WEIGHTED_FORCE_METRIC_KEYS = ("mae_fw", "rmse_fw")
DP_TEST_HESSIAN_METRIC_KEYS = ("mae_h", "rmse_h")


def mae(diff: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return float(np.mean(np.abs(diff)))


def rmse(diff: np.ndarray) -> float:
    """Calculate root mean square error."""
    return float(np.sqrt(np.mean(diff * diff)))


@dataclass(frozen=True)
class ErrorStat:
    """One weighted MAE/RMSE pair."""

    mae: float
    rmse: float
    weight: float

    def as_weighted_average_errors(
        self,
        mae_key: str,
        rmse_key: str,
    ) -> dict[str, tuple[float, float]]:
        """Convert one metric pair into `weighted_average` inputs."""
        return {
            mae_key: (self.mae, self.weight),
            rmse_key: (self.rmse, self.weight),
        }


@dataclass(frozen=True)
class EnergyTypeEvalMetrics:
    """Shared energy-type metrics for one evaluation batch or system."""

    energy: ErrorStat | None = None
    energy_per_atom: ErrorStat | None = None
    force: ErrorStat | None = None
    virial: ErrorStat | None = None
    virial_per_atom: ErrorStat | None = None

    def as_weighted_average_errors(
        self,
        metric_keys: dict[str, tuple[str, str]],
    ) -> dict[str, tuple[float, float]]:
        """Project shared metrics into caller-specific error dict keys."""
        errors: dict[str, tuple[float, float]] = {}
        for metric_name, (mae_key, rmse_key) in metric_keys.items():
            metric = getattr(self, metric_name)
            if metric is not None:
                errors.update(metric.as_weighted_average_errors(mae_key, rmse_key))
        return errors


@dataclass(frozen=True)
class SpinForceEvalMetrics:
    """Shared spin-force metrics for one evaluation batch or system."""

    force_real: ErrorStat | None = None
    force_magnetic: ErrorStat | None = None

    def as_weighted_average_errors(
        self,
        metric_keys: dict[str, tuple[str, str]],
    ) -> dict[str, tuple[float, float]]:
        """Project shared spin metrics into caller-specific error dict keys."""
        errors: dict[str, tuple[float, float]] = {}
        for metric_name, (mae_key, rmse_key) in metric_keys.items():
            metric = getattr(self, metric_name)
            if metric is not None:
                errors.update(metric.as_weighted_average_errors(mae_key, rmse_key))
        return errors


def compute_error_stat(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    scale: float = 1.0,
) -> ErrorStat:
    """Compute one MAE/RMSE pair from aligned prediction and reference arrays."""
    diff = prediction - reference
    return ErrorStat(
        mae=mae(diff) * scale,
        rmse=rmse(diff) * scale,
        weight=float(diff.size),
    )


def compute_weighted_error_stat(
    prediction: np.ndarray,
    reference: np.ndarray,
    weight: np.ndarray,
) -> ErrorStat:
    """Compute weighted MAE/RMSE from aligned prediction and reference arrays."""
    diff = prediction - reference
    weight_sum = float(np.sum(weight))
    if weight_sum <= 0.0:
        return ErrorStat(mae=0.0, rmse=0.0, weight=weight_sum)
    return ErrorStat(
        mae=float(np.sum(np.abs(diff) * weight) / weight_sum),
        rmse=float(np.sqrt(np.sum(diff * diff * weight) / weight_sum)),
        weight=weight_sum,
    )


def compute_energy_type_metrics(
    prediction: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    natoms: int,
    has_pbc: bool,
) -> EnergyTypeEvalMetrics:
    """Compute shared energy-type metrics for one evaluation dataset."""
    energy = None
    energy_per_atom = None
    force = None
    virial = None
    virial_per_atom = None

    if bool(test_data.get("find_energy", 0.0)):
        energy = compute_error_stat(
            prediction["energy"].reshape(-1, 1),
            test_data["energy"].reshape(-1, 1),
        )
        energy_per_atom = compute_error_stat(
            prediction["energy"].reshape(-1, 1),
            test_data["energy"].reshape(-1, 1),
            scale=1.0 / natoms,
        )

    if bool(test_data.get("find_force", 0.0)):
        force = compute_error_stat(
            prediction["force"].reshape(-1),
            test_data["force"].reshape(-1),
        )

    if has_pbc and bool(test_data.get("find_virial", 0.0)):
        virial = compute_error_stat(
            prediction["virial"].reshape(-1, 9),
            test_data["virial"].reshape(-1, 9),
        )
        virial_per_atom = compute_error_stat(
            prediction["virial"].reshape(-1, 9),
            test_data["virial"].reshape(-1, 9),
            scale=1.0 / natoms,
        )

    return EnergyTypeEvalMetrics(
        energy=energy,
        energy_per_atom=energy_per_atom,
        force=force,
        virial=virial,
        virial_per_atom=virial_per_atom,
    )


def compute_spin_force_metrics(
    force_real_prediction: np.ndarray,
    force_real_reference: np.ndarray,
    *,
    force_magnetic_prediction: np.ndarray | None = None,
    force_magnetic_reference: np.ndarray | None = None,
) -> SpinForceEvalMetrics:
    """Compute spin-aware force metrics from aligned real and magnetic forces."""
    force_real = compute_error_stat(force_real_prediction, force_real_reference)
    force_magnetic = None
    if force_magnetic_prediction is not None or force_magnetic_reference is not None:
        if force_magnetic_prediction is None or force_magnetic_reference is None:
            raise ValueError(
                "Spin magnetic force metrics require both prediction and reference."
            )
        force_magnetic = compute_error_stat(
            force_magnetic_prediction,
            force_magnetic_reference,
        )
    return SpinForceEvalMetrics(
        force_real=force_real,
        force_magnetic=force_magnetic,
    )
