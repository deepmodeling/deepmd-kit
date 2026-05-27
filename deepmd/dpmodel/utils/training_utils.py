# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from collections.abc import (
    Iterable,
)

import numpy as np

log = logging.getLogger(__name__)


def compute_total_numb_batch(
    numb_batches: Iterable[int],
    sampler_weights: np.ndarray,
) -> int:
    """Compute total number of batches considering sampler weights.

    Parameters
    ----------
    numb_batches : Iterable[int]
        Number of batches for each data system.
    sampler_weights : np.ndarray
        Sampling weights for each data system.

    Returns
    -------
    int
        Total number of batches.

    Raises
    ------
    ValueError
        If input validation fails.
    """
    weights = np.asarray(sampler_weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("Sampler weights must be 1D.")
    if weights.size == 0:
        raise ValueError("Sampler weights are empty.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Sampler weights must be finite.")
    if np.any(weights < 0.0):
        raise ValueError("Sampler weights must be non-negative.")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("Sampler weights must sum to a positive value.")
    probs = weights / weight_sum
    nbatches = np.asarray(numb_batches, dtype=np.float64)
    if nbatches.ndim != 1:
        raise ValueError("Number of batches must be 1D.")
    if nbatches.size == 0:
        raise ValueError("Number of batches is empty.")
    if not np.all(np.isfinite(nbatches)):
        raise ValueError("Number of batches must be finite.")
    if np.any(nbatches < 0.0):
        raise ValueError("Number of batches must be non-negative.")
    if nbatches.shape[0] != probs.shape[0]:
        raise ValueError("Number of batches and sampler weights must match.")
    valid = probs > 0.0
    if not np.any(valid):
        raise ValueError(
            "Sampler probabilities must contain at least one positive entry."
        )
    return int(np.ceil(np.max(nbatches[valid] / probs[valid])))


def resolve_model_prob(
    model_keys: list[str],
    model_prob_config: dict[str, float] | None,
    model_training_data: dict[str, object],
    rank: int = 0,
) -> np.ndarray:
    """Resolve model training probability for multi-task training.

    Parameters
    ----------
    model_keys : list[str]
        List of model keys.
    model_prob_config : dict[str, float] | None
        User-specified model probabilities. If None, use data size.
    model_training_data : dict[str, object]
        Training data for each model.
    rank : int, optional
        Process rank for distributed training, by default 0.

    Returns
    -------
    np.ndarray
        Normalized model probabilities.

    Raises
    ------
    ValueError
        If input validation fails.
    """
    model_prob = np.zeros(len(model_keys), dtype=np.float64)
    if model_prob_config:
        missing = [k for k in model_keys if k not in model_prob_config]
        if missing:
            raise ValueError(
                f"training.model_prob must specify all tasks; missing: {missing}"
            )
        for ii, model_key in enumerate(model_keys):
            if model_key in model_prob_config:
                model_prob[ii] = float(model_prob_config[model_key])
    else:
        if rank == 0:
            log.info(
                "training.model_prob is not set or empty; defaulting to the "
                "number of systems per task."
            )
        for ii, model_key in enumerate(model_keys):
            model_prob[ii] = float(len(model_training_data[model_key]))
    if not np.all(np.isfinite(model_prob)):
        raise ValueError("Model prob must be finite.")
    if np.any(model_prob < 0.0):
        raise ValueError("Model prob must be non-negative.")
    sum_prob = float(np.sum(model_prob))
    if sum_prob <= 0.0:
        raise ValueError("Sum of model prob must be larger than 0!")
    return model_prob / sum_prob


def resolve_model_prob_from_epochs(
    model_keys: list[str],
    num_epoch_dict_config: dict[str, float],
    per_task_total: np.ndarray,
) -> tuple[np.ndarray, int, dict[str, float]]:
    """Resolve model probability and training steps from epoch configuration.

    Parameters
    ----------
    model_keys : list[str]
        List of model keys.
    num_epoch_dict_config : dict[str, float]
        Target epochs for each task.
    per_task_total : np.ndarray
        Total batches per task.

    Returns
    -------
    tuple[np.ndarray, int, dict[str, float]]
        Model probabilities, total training steps, and per-task steps.

    Raises
    ------
    ValueError
        If input validation fails.
    """
    if not num_epoch_dict_config:
        raise ValueError("training.num_epoch_dict must be set for multi-task epochs.")
    missing = [k for k in model_keys if k not in num_epoch_dict_config]
    if missing:
        raise ValueError(
            f"training.num_epoch_dict must specify all tasks; missing: {missing}"
        )
    epoch_targets = np.zeros(len(model_keys), dtype=np.float64)
    for ii, model_key in enumerate(model_keys):
        epoch_value = num_epoch_dict_config[model_key]
        if epoch_value is None:
            raise ValueError(
                f"training.num_epoch_dict['{model_key}'] must be positive."
            )
        epoch_value = float(epoch_value)
        if not np.isfinite(epoch_value) or epoch_value <= 0.0:
            raise ValueError(
                f"training.num_epoch_dict['{model_key}'] must be positive, got {epoch_value}."
            )
        epoch_targets[ii] = epoch_value
    per_task_total = np.asarray(per_task_total, dtype=np.float64)
    if per_task_total.ndim != 1:
        raise ValueError("Per-task total batches must be 1D.")
    if per_task_total.shape[0] != epoch_targets.shape[0]:
        raise ValueError("Per-task totals and epoch targets must match.")
    if not np.all(np.isfinite(per_task_total)):
        raise ValueError("Per-task total batches must be finite.")
    if np.any(per_task_total <= 0.0):
        raise ValueError("Per-task total batches must be positive.")
    per_task_steps = per_task_total * epoch_targets
    total_target_steps = float(np.sum(per_task_steps))
    if total_target_steps <= 0.0:
        raise ValueError("Sum of target steps must be positive.")
    model_prob = per_task_steps / total_target_steps
    num_steps = int(np.ceil(total_target_steps))
    per_task_steps_map = {
        model_key: float(per_task_steps[ii]) for ii, model_key in enumerate(model_keys)
    }
    return model_prob, num_steps, per_task_steps_map
