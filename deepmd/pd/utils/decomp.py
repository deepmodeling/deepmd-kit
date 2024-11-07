# SPDX-License-Identifier: LGPL-3.0-or-later

# This file is used to implement some paddle functions with composite API,
# so as to support high-order differentation when double-backward is needed.
# For example: [norm] --decomposition--> [multiply, power, sum]
# This file will be removed when implmented functions are decomposed into primitive
# function in Paddle framework in the future.

from __future__ import (
    annotations,
)

import paddle

__all__ = [
    "softmax",
    "norm",
    "take_along_axis",
    "scatter_reduce",
    "sec",
    "masked_add_",
]


# decomposition for forward function
def softmax_decomp(x: paddle.Tensor, axis: int = -1) -> paddle.Tensor:
    """Forward decompsition function of softmax.

    Parameters
    ----------
    x : paddle.Tensor
        Input.
    axis : int, defaults: -1.
        A dimension along which softmax will be computed.

    Returns
    -------
    paddle.Tensor
        Computed output.
    """
    x_max = paddle.max(x, axis=axis, keepdim=True)
    x = x - x_max
    return paddle.exp(x) / paddle.sum(paddle.exp(x), axis=axis, keepdim=True)


def norm_decomp(
    x: paddle.Tensor, p: float = 2, axis: bool = -1, keepdim: bool = False
) -> paddle.Tensor:
    """Forward decompsition function of norm.

    Parameters
    ----------
    x : paddle.Tensor
        Input
    p : float, default: 2
        Order of norm
    axis : bool, default: -1
        Dimensions over which to compute the vector or matrix norm
    keepdim : bool, default: False
        If set to True, the reduced dimensions are retained in the result as dimensions
        with size one

    Returns
    -------
    paddle.Tensor
        A real-valued tensor, even when A is complex.
    """
    if p == 2 or p == 2.0:
        # clip for negative indexing, or 1/(0^(k-1)) will cause inf in backward
        return (x * x).sum(axis=axis, keepdim=keepdim).clip(1e-12) ** 0.5
    return (x.abs()**p).sum(axis=axis, keepdim=keepdim) ** (1 / p)


def take_along_axis_decomp(
    x: paddle.Tensor, indices: paddle.Tensor, axis: int, broadcast: bool = True
) -> paddle.Tensor:
    """Forward decompsition function of take_along_axis.

    Parameters
    ----------
    x : paddle.Tensor
        The input tensor.
    indices : paddle.Tensor
        Indices to take along each 1d slice of array.
    axis : int
        The axis to take 1d slices along.
    broadcast : bool, default: True
        Whether the indices broadcast.

    Returns
    -------
    paddle.Tensor
        Computed output.
    """
    # manually contruct indices for gather_nd(ind_gather_nd.ndim == indices.ndim + 1,
    # the lsat 1 represents the number of dimension(s) of indices)
    ind_gather_nd = paddle.stack(
        paddle.meshgrid(*[paddle.arange(v) for v in indices.shape], indexing="ij"),
        axis=-1,
    )
    ind_gather_nd[..., axis] = indices
    # compute output using constructed indices via gather_nd
    out = paddle.gather_nd(x, ind_gather_nd)
    return out


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
        input.put_along_axis_(indices=index, values=src, axis=axis, reduce="add")
    elif reduce == "mean":
        input.put_along_axis_(indices=index, values=src, axis=axis, reduce="add")
        dst_div = paddle.ones_like(input).put_along_axis(
            indices=index,
            values=paddle.to_tensor(1.0, dtype=input.dtype),
            axis=axis,
            reduce="add",
        )
        input = input / dst_div
    elif reduce == "prod":
        input = input.put_along_axis(indices=index, values=src, axis=axis, reduce="mul")
    else:
        raise NotImplementedError("only support mode in ['sum', 'prod', 'mean']!")
    return input


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


def normalize_decomp(
    x: paddle.Tensor,
    p: float = 2,
    axis: int = 1,
    epsilon: float = 1e-12,
) -> paddle.Tensor:
    """Forward decompsition function of normalize.

    Parameters
    ----------
    x : paddle.Tensor
        Input tensor.
    p : float, optional
        Order of the norm, default: 2
    axis : int, optional
        Axis on which to perform normalization, default: 1
    epsilon : float, optional
        Epislon value, default: 1e-12

    Returns
    -------
    paddle.Tensor
        Computed output.
    """
    return x / (norm(x, p=p, axis=axis, keepdim=True).clip(min=epsilon))


# alias for decomposed functions for convinience
normalize = normalize_decomp
masked_add_ = masked_add__decomp
scatter_reduce = scatter_reduce_decomp
take_along_axis = take_along_axis_decomp
norm = norm_decomp
softmax = softmax_decomp
