# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generic weighted many-to-one aggregation utilities."""

from __future__ import annotations

import numpy as np

from dpa_adapt.data.errors import DPADataError


def aggregate_weighted_groups(
    features: np.ndarray,
    group_ids: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate row features into one weighted vector per group.

    Parameters
    ----------
    features
        Per-item feature matrix with shape ``(n_items, feature_dim)``.
    group_ids
        Integer group id for each feature row, shape ``(n_items,)``.
    weights
        Weight for each feature row within its group, shape ``(n_items,)``.
    labels
        Per-item labels, shape ``(n_items,)`` or ``(n_items, task_dim)``.
        The first label encountered for each group is used.

    Returns
    -------
    embeddings, labels, group_ids
        Group-level embeddings, group labels, and sorted group ids.
    """
    features = np.asarray(features)
    group_ids = np.asarray(group_ids)
    weights = np.asarray(weights, dtype=float)
    labels = np.asarray(labels)

    if features.ndim != 2:
        raise DPADataError(
            f"features has shape {features.shape}; expected (n_items, feature_dim)."
        )
    n_items = features.shape[0]
    if group_ids.shape != (n_items,):
        raise DPADataError(
            f"group_ids has shape {group_ids.shape}; expected ({n_items},)."
        )
    if weights.shape != (n_items,):
        raise DPADataError(
            f"weights has shape {weights.shape}; expected ({n_items},)."
        )
    if labels.shape[0] != n_items:
        raise DPADataError(
            f"labels has {labels.shape[0]} rows; expected {n_items}."
        )
    if n_items == 0:
        raise DPADataError("Cannot aggregate an empty feature matrix.")

    ordered_ids = np.array(sorted(np.unique(group_ids.astype(np.int64))), dtype=np.int64)
    embeddings = []
    grouped_labels = []
    for group_id in ordered_ids:
        mask = group_ids == group_id
        embeddings.append(np.sum(features[mask] * weights[mask, None], axis=0))
        grouped_labels.append(labels[mask][0])

    label_arr = np.asarray(grouped_labels)
    if label_arr.ndim == 2 and label_arr.shape[1] == 1:
        label_arr = label_arr.reshape(-1)
    return np.vstack(embeddings), label_arr, ordered_ids
