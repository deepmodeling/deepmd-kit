# SPDX-License-Identifier: LGPL-3.0-or-later

# This file is used to implement some paddle functions with composite APi,
# so as to support high-order differentation when double-backward is needed.

# This file will be removed when implmented functions are decomposed into primitive
# function in Paddle framework in the future.


import paddle

__all__ = [
    "norm",
    "take_along_axis",
    "scatter_reduce",
    "sec",
    "masked_add_",
]


def norm(
    x: paddle.Tensor, p: float = 2, axis: bool = -1, keepdim: bool = False
) -> paddle.Tensor:
    if p == 2 or p == 2.0:
        return (x * x).sum(axis=axis, keepdim=keepdim) ** 0.5
    return (x**p).sum(axis=axis, keepdim=keepdim) ** (1 / p)


def take_along_axis(
    x: paddle.Tensor, indices: paddle.Tensor, axis: int, broadcast: bool = True
):
    """Broadcast no used now."""
    # manually contruct indices for gather_nd(ind_gather_nd.ndim == indices.ndim + 1, the lsat 1 represents the number of dimension(s) of indices)
    ind_gather_nd = paddle.stack(
        paddle.meshgrid(*[paddle.arange(v) for v in indices.shape], indexing="ij"),
        axis=-1,
    )
    ind_gather_nd[..., axis] = indices
    # compute output using constructed indices via gather_nd
    out = paddle.gather_nd(x, ind_gather_nd)
    return out


def scatter_reduce(
    input: paddle.Tensor,
    axis: int,
    index: paddle.Tensor,
    src: paddle.Tensor,
    reduce: str,
) -> paddle.Tensor:
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


def sec(l: int, size: int) -> list[int]:
    assert l > 0
    assert size > 0
    if l % size == 0:
        return [size] * (l // size)
    return [size] * (l // size) + [l % size]


def masked_add_(
    x: paddle.Tensor, mask: paddle.Tensor, v: paddle.Tensor
) -> paddle.Tensor:
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


def normalize(
    x: paddle.Tensor,
    p: float = 2,
    axis: int = 1,
    epsilon: float = 1e-12,
) -> paddle.Tensor:
    return x / (norm(x, p=p, axis=axis, keepdim=True).clip(min=epsilon))
