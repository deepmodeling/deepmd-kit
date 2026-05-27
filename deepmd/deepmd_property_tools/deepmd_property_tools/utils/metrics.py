# SPDX-License-Identifier: LGPL-3.0-or-later
"""Simple regression metrics."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }
