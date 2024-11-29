# SPDX-License-Identifier: LGPL-3.0-or-later

# This file is used to implement some paddle functions with composite API,
# so as to support high-order differentation when double-backward is needed.
# For example: [norm] --decomposition--> [multiply, power, sum]
# This file will be removed when implmented functions are decomposed into primitive
# function in Paddle framework in the future.

from __future__ import (
    annotations,
)

import numpy as np
import paddle

__all__ = [
    "masked_add_",
    "numel",
    "scatter_reduce",
    "sec",
]


def scatter_reduce_decomp(
    input: paddle.Tensor,
    axis: int,
    index: paddle.Tensor,
    src: paddle.Tensor,
    reduce: str,
) -> paddle.Tensor:
    """Forward decompsition function of scatter_reduce.

    Parameters
    ----------
    input : paddle.Tensor
        Input tensor.
    axis : int
        The axis along which to index.
    index : paddle.Tensor
        The indices of elements to scatter and reduce.
    src : paddle.Tensor
        The source elements to scatter and reduce.
    reduce : str
        The reduction operation to apply for non-unique indices.
        Supported modes: ("sum", "prod", "mean", "amax", "amin").

    Returns
    -------
    paddle.Tensor
        Computed output.
    """
    # reduce: "sum", "prod", "mean", "amax", "amin"
    if reduce == "sum":
        output = input.put_along_axis(
            indices=index, values=src, axis=axis, reduce="add"
        )
    elif reduce == "mean":
        output = input.put_along_axis(
            indices=index, values=src, axis=axis, reduce="mean"
        )
    elif reduce == "prod":
        output = input.put_along_axis(
            indices=index, values=src, axis=axis, reduce="mul"
        )
    else:
        raise NotImplementedError("only support mode in ['sum', 'prod', 'mean']!")
    return output


def sec(length: int, size: int) -> list[int]:
    """Auxiliary function for decomposed functions.

    If length is not divisible by size, the last chunk will be smaller.

    Parameters
    ----------
    length : int
        Length to be chunked.
    size : int
        Chunk size.

    Returns
    -------
    list[int]
        Chunked output list.
    """
    assert length > 0
    assert size > 0
    if length % size == 0:
        return [size] * (length // size)
    return [size] * (length // size) + [length % size]


def masked_add__decomp(
    x: paddle.Tensor, mask: paddle.Tensor, v: paddle.Tensor
) -> paddle.Tensor:
    """Forward decompsition function of masked_add_(inplace operator).

    Parameters
    ----------
    x : paddle.Tensor
        Input tensor.
    mask : paddle.Tensor
        Mask tensor.
    v : paddle.Tensor
        Value to add.

    Returns
    -------
    paddle.Tensor
        Computed output.
    """
    assert mask.dtype == paddle.bool, f"mask must be bool type, but got {mask.dtype}"
    # indices is bool mask
    mask_coord = paddle.concat(
        paddle.nonzero(mask, as_tuple=True),
        axis=1,
    )  # [nz, dim]
    if not paddle.is_tensor(v):
        v = paddle.full([mask_coord.shape[0]], v, dtype=x.dtype)
    t = paddle.scatter_nd_add(
        x,
        mask_coord,
        v,
    )
    paddle.assign(t, x)  # inplace update
    return x


def numel(x: paddle.Tensor) -> int:
    if paddle.in_dynamic_mode():
        return np.prod(x.shape)

    return paddle.numel(x)


# alias for decomposed functions for convinience
masked_add_ = masked_add__decomp
scatter_reduce = scatter_reduce_decomp
