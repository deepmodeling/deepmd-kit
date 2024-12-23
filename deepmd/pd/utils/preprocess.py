# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

import paddle

log = logging.getLogger(__name__)


def compute_smooth_weight(distance, rmin: float, rmax: float):
    """Compute smooth weight for descriptor elements."""
    if rmin >= rmax:
        raise ValueError("rmin should be less than rmax.")
    min_mask = distance <= rmin
    max_mask = distance >= rmax
    mid_mask = paddle.logical_not(paddle.logical_or(min_mask, max_mask))
    uu = (distance - rmin) / (rmax - rmin)
    vv = uu * uu * uu * (-6 * uu * uu + 15 * uu - 10) + 1
    return vv * mid_mask.astype(vv.dtype) + min_mask.astype(vv.dtype)
