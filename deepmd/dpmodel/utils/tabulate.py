# SPDX-License-Identifier: LGPL-3.0-or-later
import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)


def tabulate_fusion(
    table: Array,
    table_info: Array,
    em_x: Array,
    last_layer_size: int,
    reference: Array | None = None,
) -> Array:
    """Evaluate tabulated embedding-net values with C1 extrapolation."""
    if reference is None:
        xp = array_api_compat.array_namespace(em_x)
        reference = em_x
    else:
        xp = array_api_compat.array_namespace(em_x, reference)
    device = array_api_compat.device(reference)
    table = xp.asarray(table[...], dtype=reference.dtype, device=device)
    table_info = xp.asarray(table_info[...], dtype=reference.dtype, device=device)

    nloc, nnei = em_x.shape[:2]
    xx = xp.reshape(em_x, (nloc, nnei))
    lower = table_info[0]
    upper = table_info[1]
    table_max = table_info[2]
    stride0 = table_info[3]
    stride1 = table_info[4]

    zeros = xp.zeros(xx.shape, dtype=xp.int64, device=device)
    nspline = table.shape[0]
    last_idx = xp.full(xx.shape, nspline - 1, dtype=xp.int64, device=device)
    first_stride = xp.astype(xp.floor((upper - lower) / stride0), xp.int64)
    first_stride_value = xp.astype(first_stride, reference.dtype)

    first_idx = xp.astype(xp.floor((xx - lower) / stride0), xp.int64)
    second_idx = first_stride + xp.astype(xp.floor((xx - upper) / stride1), xp.int64)
    table_idx = xp.where(
        xx < lower,
        zeros,
        xp.where(
            xx < upper,
            first_idx,
            xp.where(xx < table_max, second_idx, last_idx),
        ),
    )
    table_idx = xp.minimum(xp.maximum(table_idx, zeros), last_idx)

    table_idx_value = xp.astype(table_idx, reference.dtype)
    dx_first = xx - (table_idx_value * stride0 + lower)
    dx_second = xx - ((table_idx_value - first_stride_value) * stride1 + upper)
    dx_high = table_max - (
        (xp.astype(last_idx, reference.dtype) - first_stride_value) * stride1 + upper
    )
    dx = xp.where(
        xx < lower,
        xp.zeros_like(xx),
        xp.where(
            xx < upper,
            dx_first,
            xp.where(xx < table_max, dx_second, dx_high),
        ),
    )
    extrapolate_delta = xp.where(
        xx < lower,
        xx - lower,
        xp.where(xx >= table_max, xx - table_max, xp.zeros_like(xx)),
    )

    coeff = xp.take(table, xp.reshape(table_idx, (-1,)), axis=0)
    coeff = xp.reshape(coeff, (nloc, nnei, last_layer_size, 6))
    dx = xp.reshape(dx, (nloc, nnei, 1))
    values = (
        coeff[..., 0]
        + (
            coeff[..., 1]
            + (
                coeff[..., 2]
                + (coeff[..., 3] + (coeff[..., 4] + coeff[..., 5] * dx) * dx) * dx
            )
            * dx
        )
        * dx
    )
    values_grad = (
        coeff[..., 1]
        + (
            2 * coeff[..., 2]
            + (3 * coeff[..., 3] + (4 * coeff[..., 4] + 5 * coeff[..., 5] * dx) * dx)
            * dx
        )
        * dx
    )
    extrapolate_delta = xp.reshape(extrapolate_delta, (nloc, nnei, 1))
    return values + values_grad * extrapolate_delta
