# SPDX-License-Identifier: LGPL-3.0-or-later
"""Safe versions of some functions that have problematic gradients.

Check https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
for more information.
"""

import array_api_compat


def safe_for_sqrt(x):
    """Safe version of sqrt that has a gradient of 0 at x = 0."""
    xp = array_api_compat.array_namespace(x)
    mask = x > 0.0
    return xp.where(mask, xp.sqrt(xp.where(mask, x, xp.ones_like(x))), xp.zeros_like(x))


def safe_for_vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
    """Safe version of sqrt that has a gradient of 0 at x = 0."""
    xp = array_api_compat.array_namespace(x)
    mask = xp.sum(xp.square(x), axis=axis, keepdims=True) > 0
    if keepdims:
        mask_squeezed = mask
    else:
        mask_squeezed = xp.squeeze(mask, axis=axis)
    return xp.where(
        mask_squeezed,
        xp.linalg.vector_norm(
            xp.where(mask, x, xp.ones_like(x)), axis=axis, keepdims=keepdims, ord=ord
        ),
        xp.zeros_like(mask_squeezed, dtype=x.dtype),
    )
