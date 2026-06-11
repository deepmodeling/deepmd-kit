# SPDX-License-Identifier: LGPL-3.0-or-later
"""Lebedev quadrature data loader for S2 projections."""

from __future__ import (
    annotations,
)

from pathlib import (
    Path,
)

import numpy as np

# See: https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html
LEBEDEV_RULES_FILE = Path(__file__).with_name("lebedev_rules.npz")
LEBEDEV_PRECISION_TO_NPOINTS = {
    3: 6,
    5: 14,
    7: 26,
    9: 38,
    11: 50,
    13: 74,
    15: 86,
    17: 110,
    19: 146,
    21: 170,
    23: 194,
    25: 230,
    27: 266,
    29: 302,
    31: 350,
    35: 434,
    41: 590,
    47: 770,
    53: 974,
    59: 1202,
    65: 1454,
    71: 1730,
    77: 2030,
    83: 2354,
    89: 2702,
    95: 3074,
    101: 3470,
    107: 3890,
    113: 4334,
    119: 4802,
    125: 5294,
    131: 5810,
}


def load_lebedev_rule(precision: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one Lebedev rule from the packaged compressed data file.

    Parameters
    ----------
    precision
        Algebraic precision of the requested Lebedev rule.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Cartesian unit points with shape ``(A, 3)`` and normalized weights with
        shape ``(A,)``. The weights sum to one, so the sphere integral is
        ``4*pi*sum(weights*f(points))``.
    """
    rule_key = f"{int(precision):03d}"
    if not LEBEDEV_RULES_FILE.exists():
        raise FileNotFoundError(
            f"Lebedev quadrature data file is missing: {LEBEDEV_RULES_FILE}"
        )
    with np.load(LEBEDEV_RULES_FILE) as rules:
        point_key = f"points_{rule_key}"
        weight_key = f"weights_{rule_key}"
        if point_key not in rules or weight_key not in rules:
            raise ValueError(
                f"Lebedev rule with precision {precision} is not packaged; "
                f"available precisions: {sorted(LEBEDEV_PRECISION_TO_NPOINTS)}"
            )
        points = rules[point_key]
        weights = rules[weight_key]
    return points, weights
