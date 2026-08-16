# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
)

import numpy as np

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

FULL_VALIDATION_WEIGHTED_METRIC_KEYS = {
    "energy_per_atom": ("mae_e_per_atom", "rmse_e_per_atom"),
    "force": ("mae_f", "rmse_f"),
    "virial_per_atom": ("mae_v_per_atom", "rmse_v_per_atom"),
    "stress": ("mae_s", "rmse_s"),
}
# Spin full validation splits the force term into real and magnetic parts, so
# it projects every shared metric except the plain force.
SPIN_FULL_VALIDATION_WEIGHTED_METRIC_KEYS = {
    "energy_per_atom": ("mae_e_per_atom", "rmse_e_per_atom"),
    "virial_per_atom": ("mae_v_per_atom", "rmse_v_per_atom"),
    "stress": ("mae_s", "rmse_s"),
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
    stress: ErrorStat | None = None

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


def _compute_stress_error_stat(
    virial_prediction: np.ndarray,
    virial_reference: np.ndarray,
    box: np.ndarray | None,
) -> ErrorStat | None:
    """Compute the stress error of one evaluation dataset.

    Stress is the negated virial divided by the cell volume, the
    tensile-positive convention ``dp test`` reports. A frame whose cell is
    singular carries no stress and is dropped rather than contributing a
    divergent entry.

    Parameters
    ----------
    virial_prediction : np.ndarray
        Predicted virial with shape (nframes, 9), in eV.
    virial_reference : np.ndarray
        Reference virial with shape (nframes, 9), in eV.
    box : np.ndarray | None
        Cell vectors with shape (nframes, 9) in Angstrom, or ``None`` when the
        caller supplies no cell.

    Returns
    -------
    ErrorStat | None
        Stress error in eV/Angstrom^3, or ``None`` when no cell is supplied or
        no frame has a non-singular cell.
    """
    if box is None:
        return None
    volume = np.abs(np.linalg.det(np.asarray(box).reshape(-1, 3, 3)))
    finite = volume > 0.0
    if not np.any(finite):
        return None
    scale = -1.0 / volume[finite, None]
    return compute_error_stat(
        virial_prediction[finite] * scale, virial_reference[finite] * scale
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
    stress = None

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
        virial_prediction = prediction["virial"].reshape(-1, 9)
        virial_reference = test_data["virial"].reshape(-1, 9)
        virial = compute_error_stat(virial_prediction, virial_reference)
        virial_per_atom = compute_error_stat(
            virial_prediction, virial_reference, scale=1.0 / natoms
        )
        stress = _compute_stress_error_stat(
            virial_prediction, virial_reference, test_data.get("box")
        )

    return EnergyTypeEvalMetrics(
        energy=energy,
        energy_per_atom=energy_per_atom,
        force=force,
        virial=virial,
        virial_per_atom=virial_per_atom,
        stress=stress,
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


def _spin_force_metrics_from_prediction(
    prediction: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
) -> SpinForceEvalMetrics:
    """Align predicted and reference forces into real and magnetic subsets.

    Real forces cover all atoms, while magnetic forces are restricted to the
    magnetic atoms selected by the boolean ``mask_mag`` of shape
    ``(nframes, natoms)``. The magnetic term is produced only when
    ``find_force_mag`` is set and both prediction and reference magnetic
    forces are present, matching the ``dp test`` spin convention.

    Parameters
    ----------
    prediction : dict[str, np.ndarray]
        Model predictions containing ``force`` and, for the magnetic term,
        ``force_mag`` and ``mask_mag``.
    test_data : dict[str, np.ndarray]
        Reference labels and ``find_*`` availability flags for one system.

    Returns
    -------
    SpinForceEvalMetrics
        The real-atom and (optionally) magnetic-atom force errors.
    """
    force_real_prediction = prediction["force"].reshape(-1, 3)
    force_real_reference = test_data["force"].reshape(-1, 3)
    has_force_mag = (
        bool(test_data.get("find_force_mag", 0.0))
        and "force_mag" in prediction
        and "force_mag" in test_data
    )
    if not has_force_mag:
        return compute_spin_force_metrics(
            force_real_prediction=force_real_prediction,
            force_real_reference=force_real_reference,
        )
    magnetic_mask = prediction["mask_mag"].reshape(-1).astype(bool)
    return compute_spin_force_metrics(
        force_real_prediction=force_real_prediction,
        force_real_reference=force_real_reference,
        force_magnetic_prediction=prediction["force_mag"].reshape(-1, 3)[magnetic_mask],
        force_magnetic_reference=test_data["force_mag"].reshape(-1, 3)[magnetic_mask],
    )


def compute_full_validation_energy_metrics(
    prediction: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    natoms: int,
    has_pbc: bool,
) -> dict[str, tuple[float, float]]:
    """Compute energy-type full validation metrics for one system.

    Parameters
    ----------
    prediction : dict[str, np.ndarray]
        Model predictions containing ``energy``, ``force`` and optionally
        ``virial``.
    test_data : dict[str, np.ndarray]
        Reference labels and ``find_*`` availability flags for one system.
    natoms : int
        The number of atoms per frame, used for per-atom normalization.
    has_pbc : bool
        Whether the system is periodic, gating the second-rank metrics.

    Returns
    -------
    dict[str, tuple[float, float]]
        Weighted-average-ready ``(value, weight)`` pairs keyed by metric.
    """
    metrics = compute_energy_type_metrics(prediction, test_data, natoms, has_pbc)
    return metrics.as_weighted_average_errors(FULL_VALIDATION_WEIGHTED_METRIC_KEYS)


def compute_full_validation_spin_metrics(
    prediction: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    natoms: int,
    has_pbc: bool,
) -> dict[str, tuple[float, float]]:
    """Compute spin-energy full validation metrics for one system.

    The energy and second-rank terms come from the shared energy-type metrics.
    Forces replace the shared plain-force term with a real-atom term over all
    atoms and a magnetic term over the magnetic atoms selected by ``mask_mag``.

    Parameters
    ----------
    prediction : dict[str, np.ndarray]
        Model predictions containing ``energy``, ``force``, ``force_mag`` and
        the boolean ``mask_mag``.
    test_data : dict[str, np.ndarray]
        Reference labels and ``find_*`` availability flags for one system.
    natoms : int
        The number of atoms per frame, used for per-atom normalization.
    has_pbc : bool
        Whether the system is periodic, gating the second-rank metrics.

    Returns
    -------
    dict[str, tuple[float, float]]
        Weighted-average-ready ``(value, weight)`` pairs keyed by metric.
    """
    metrics = compute_energy_type_metrics(prediction, test_data, natoms, has_pbc)
    errors = metrics.as_weighted_average_errors(
        SPIN_FULL_VALIDATION_WEIGHTED_METRIC_KEYS
    )
    if bool(test_data.get("find_force", 0.0)):
        spin_metrics = _spin_force_metrics_from_prediction(prediction, test_data)
        errors.update(
            spin_metrics.as_weighted_average_errors(DP_TEST_SPIN_WEIGHTED_METRIC_KEYS)
        )
    return errors


@dataclass(frozen=True)
class MetricFamily:
    """One quantity a full validation profile reports, as MAE and RMSE.

    Families sharing a loss prefactor pair are alternative presentations of the
    same trained quantity, such as the second-rank response reported either as
    stress or as per-atom virial. The log table shows exactly one of them.

    Attributes
    ----------
    token : str
        Family identifier used in ``validation_metric``, such as ``"e"``.
    mae_key : str
        Internal metric key carrying the mean absolute error.
    rmse_key : str
        Internal metric key carrying the root mean square error.
    unit : tuple[str, float]
        Display unit and the factor converting an internal value into it.
    prefactors : tuple[str, str]
        Loss prefactor keys that must both be active for the family to be
        trainable.
    """

    token: str
    mae_key: str
    rmse_key: str
    unit: tuple[str, float]
    prefactors: tuple[str, str]

    def metrics(self) -> tuple[tuple[str, str], ...]:
        """Return the ``(kind, metric_key)`` pairs this family contributes."""
        return (("mae", self.mae_key), ("rmse", self.rmse_key))


@dataclass(frozen=True)
class FullValidationMetricProfile:
    """Metric family definition for one full validation model class.

    Bundles every aspect that differs between energy-type and spin-energy full
    validation so the validator stays data-driven instead of branching on the
    model class. Every selectable metric, display unit and loss prefactor pair
    derives from ``families``, so each quantity is declared exactly once.

    Attributes
    ----------
    name : str
        Profile identifier, either ``"energy"`` or ``"spin"``.
    families : tuple[MetricFamily, ...]
        Reported quantities in table order. Where several families share a
        loss prefactor pair, the first one is the default presentation.
    needs_spin : bool
        Whether the profile requires spin input and magnetic-force outputs.
    log_header_note : str
        One-line legend describing the metric columns.
    compute_system_metrics : Callable
        Routine computing weighted metric pairs for one system, with signature
        ``(prediction, test_data, natoms, has_pbc) -> dict``.
    """

    name: str
    families: tuple[MetricFamily, ...]
    needs_spin: bool
    log_header_note: str
    compute_system_metrics: Callable[
        [dict[str, np.ndarray], dict[str, np.ndarray], int, bool],
        dict[str, tuple[float, float]],
    ]

    @property
    def metric_key_map(self) -> dict[str, str]:
        """Map a normalized metric token to its internal metric key."""
        return {
            f"{family.token}:{kind}": key
            for family in self.families
            for kind, key in family.metrics()
        }

    @property
    def metric_family_by_key(self) -> dict[str, str]:
        """Map an internal metric key back to its family identifier."""
        return {
            key: family.token for family in self.families for _, key in family.metrics()
        }

    @property
    def unit_by_family(self) -> dict[str, tuple[str, float]]:
        """Map a family identifier to its ``(display_unit, scale)``."""
        return {family.token: family.unit for family in self.families}

    @property
    def prefactor_by_metric(self) -> dict[str, tuple[str, str]]:
        """Map a normalized metric token to its loss prefactor keys."""
        return {
            f"{family.token}:{kind}": family.prefactors
            for family in self.families
            for kind, _ in family.metrics()
        }

    def columns(self, metric: str) -> tuple[tuple[str, str], ...]:
        """Return the ``val.log`` table layout for a selected metric.

        The table carries one column pair per trained quantity. Where several
        families present the same quantity, the selected family wins and the
        first declared family is the fallback, so no quantity is reported
        twice and the selected metric is always available to the
        best-checkpoint selector.

        Parameters
        ----------
        metric : str
            Normalized ``validation_metric`` token, such as ``"v:rmse"``.

        Returns
        -------
        tuple[tuple[str, str], ...]
            Ordered ``(header_label, metric_key)`` pairs.
        """
        selected = metric.split(":")[0]
        shown: dict[tuple[str, str], MetricFamily] = {}
        for family in self.families:
            if family.prefactors not in shown or family.token == selected:
                shown[family.prefactors] = family
        return tuple(
            (f"{family.token.upper()}_{kind.upper()}", key)
            for family in shown.values()
            for kind, key in family.metrics()
        )


#: Per-atom energy, reported by every profile.
_ENERGY_FAMILY = MetricFamily(
    token="e",
    mae_key="mae_e_per_atom",
    rmse_key="rmse_e_per_atom",
    unit=("meV/atom", 1000.0),
    prefactors=("start_pref_e", "limit_pref_e"),
)
#: The second-rank response as stress, the default presentation. Declared
#: ahead of the per-atom virial so the table shows stress unless
#: ``validation_metric`` selects the virial instead.
_STRESS_FAMILY = MetricFamily(
    token="s",
    mae_key="mae_s",
    rmse_key="rmse_s",
    unit=("meV/Å³", 1000.0),
    prefactors=("start_pref_v", "limit_pref_v"),
)
#: The second-rank response as virial normalized by the atom count.
_VIRIAL_FAMILY = MetricFamily(
    token="v",
    mae_key="mae_v_per_atom",
    rmse_key="rmse_v_per_atom",
    unit=("meV/atom", 1000.0),
    prefactors=("start_pref_v", "limit_pref_v"),
)
#: Legend fragment shared by every profile that reports the second-rank term.
_SECOND_RANK_NOTE = (
    "the second-rank column is S, stress as the negated virial divided by the "
    "cell volume, or V, virial normalized by natoms, following "
    "`validation_metric`.\n"
)

ENERGY_FULL_VALIDATION_PROFILE = FullValidationMetricProfile(
    name="energy",
    families=(
        _ENERGY_FAMILY,
        MetricFamily(
            token="f",
            mae_key="mae_f",
            rmse_key="rmse_f",
            unit=("meV/Å", 1000.0),
            prefactors=("start_pref_f", "limit_pref_f"),
        ),
        _STRESS_FAMILY,
        _VIRIAL_FAMILY,
    ),
    needs_spin=False,
    log_header_note=(
        "# E uses per-atom energy, F uses component-wise force errors, and "
        + _SECOND_RANK_NOTE
    ),
    compute_system_metrics=compute_full_validation_energy_metrics,
)

SPIN_FULL_VALIDATION_PROFILE = FullValidationMetricProfile(
    name="spin",
    families=(
        _ENERGY_FAMILY,
        MetricFamily(
            token="fr",
            mae_key="mae_fr",
            rmse_key="rmse_fr",
            unit=("meV/Å", 1000.0),
            prefactors=("start_pref_fr", "limit_pref_fr"),
        ),
        MetricFamily(
            token="fm",
            mae_key="mae_fm",
            rmse_key="rmse_fm",
            unit=("meV/μB", 1000.0),
            prefactors=("start_pref_fm", "limit_pref_fm"),
        ),
        _STRESS_FAMILY,
        _VIRIAL_FAMILY,
    ),
    needs_spin=True,
    log_header_note=(
        "# E uses per-atom energy, FR uses component-wise real-atom force "
        "errors, FM uses magnetic-atom force errors, and " + _SECOND_RANK_NOTE
    ),
    compute_system_metrics=compute_full_validation_spin_metrics,
)

FULL_VALIDATION_PROFILES: dict[str, FullValidationMetricProfile] = {
    ENERGY_FULL_VALIDATION_PROFILE.name: ENERGY_FULL_VALIDATION_PROFILE,
    SPIN_FULL_VALIDATION_PROFILE.name: SPIN_FULL_VALIDATION_PROFILE,
}
