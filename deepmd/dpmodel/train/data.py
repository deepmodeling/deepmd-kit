# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared training-data helpers for backend entrypoints."""

from __future__ import (
    annotations,
)

import inspect
from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

from .trainer import (
    DEFAULT_TASK_KEY,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
        Mapping,
    )


@dataclass(frozen=True)
class TrainingTaskConfig:
    """Normalized config view for one training task."""

    key: str
    model_params: Mapping[str, Any]
    training_data_params: Mapping[str, Any]
    validation_data_params: Mapping[str, Any] | None
    stat_file: str | None
    valid_numb_batch: int


def iter_training_task_configs(
    config: Mapping[str, Any],
) -> Iterator[TrainingTaskConfig]:
    """Yield task configs, treating single-task input as one ``Default`` task."""
    model_params = config["model"]
    training_params = config["training"]
    if "model_dict" not in model_params:
        validation_data_params = training_params.get("validation_data")
        yield TrainingTaskConfig(
            key=DEFAULT_TASK_KEY,
            model_params=model_params,
            training_data_params=training_params["training_data"],
            validation_data_params=validation_data_params,
            stat_file=training_params.get("stat_file"),
            valid_numb_batch=_valid_numb_batch(validation_data_params),
        )
        return

    data_dict = training_params["data_dict"]
    for task_key, task_model_params in model_params["model_dict"].items():
        task_data_params = data_dict[task_key]
        validation_data_params = task_data_params.get("validation_data")
        yield TrainingTaskConfig(
            key=task_key,
            model_params=task_model_params,
            training_data_params=task_data_params["training_data"],
            validation_data_params=validation_data_params,
            stat_file=task_data_params.get("stat_file"),
            valid_numb_batch=_valid_numb_batch(validation_data_params),
        )


def make_task_maps(
    config: Mapping[str, Any],
    factory: Callable[[TrainingTaskConfig], tuple[Any, Any | None, Any | None]],
) -> tuple[dict[str, Any], dict[str, Any | None], dict[str, Any | None]]:
    """Build training, validation, and stat maps from normalized task configs."""
    training_data: dict[str, Any] = {}
    validation_data: dict[str, Any | None] = {}
    stat_data: dict[str, Any | None] = {}
    for task_config in iter_training_task_configs(config):
        train_item, valid_item, stat_item = factory(task_config)
        training_data[task_config.key] = train_item
        validation_data[task_config.key] = valid_item
        stat_data[task_config.key] = stat_item
    return training_data, validation_data, stat_data


def print_data_summaries(
    training_data: Mapping[str, Any],
    validation_data: Mapping[str, Any | None],
    *,
    probabilities: Mapping[str, float] | None = None,
) -> None:
    """Print train/validation data summaries for one or more tasks."""
    multi_task = len(training_data) > 1
    for task_key, data in training_data.items():
        name = f"training data({task_key})" if multi_task else "training"
        _print_summary(data, name, _task_probability(probabilities, task_key))
        valid_data = validation_data.get(task_key)
        if valid_data is not None:
            name = f"validation data({task_key})" if multi_task else "validation"
            _print_summary(valid_data, name, None)


def _valid_numb_batch(validation_data_params: Mapping[str, Any] | None) -> int:
    if validation_data_params is None:
        return 1
    return max(int(validation_data_params.get("numb_btch", 1)), 1)


def _task_probability(
    probabilities: Mapping[str, float] | None,
    task_key: str,
) -> list[float] | None:
    if probabilities is None or task_key not in probabilities:
        return None
    return [float(probabilities[task_key])]


def _print_summary(data: Any, name: str, prob: list[float] | None) -> None:
    printer = data.print_summary
    try:
        signature = inspect.signature(printer)
    except (TypeError, ValueError):
        printer(name, prob)
        return
    try:
        signature.bind(name, prob)
    except TypeError as exc:
        try:
            signature.bind(name)
        except TypeError:
            raise exc from None
        printer(name)
    else:
        printer(name, prob)
