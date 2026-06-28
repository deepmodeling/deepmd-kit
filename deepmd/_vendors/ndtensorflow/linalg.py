# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import math
from collections import (
    namedtuple,
)

import tensorflow as tf

from ._array import (
    Array,
)
from ._namespace import (
    _is_complex,
    _moveaxis,
    _promote_two,
    _real_dtype_for,
    _shape_tuple,
    _unwrap,
    count_nonzero,
    finfo,
    matmul,
    matrix_transpose,
)
from ._namespace import sum as xp_sum
from ._namespace import (
    tensordot,
    vecdot,
)

EighResult = namedtuple("EighResult", ["eigenvalues", "eigenvectors"])
EigResult = namedtuple("EigResult", ["eigenvalues", "eigenvectors"])
QRResult = namedtuple("QRResult", ["Q", "R"])
SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])
SVDResult = namedtuple("SVDResult", ["U", "S", "Vh"])


def _wrap(x: tf.Tensor) -> Array:
    return Array._from_tensor(x)


def _replace_nonfinite(x: tf.Tensor) -> tf.Tensor:
    if _is_complex(x.dtype):
        finite = tf.math.is_finite(tf.math.real(x)) & tf.math.is_finite(tf.math.imag(x))
    else:
        finite = tf.math.is_finite(x)
    return tf.where(finite, x, tf.zeros((), dtype=x.dtype))


def outer(x1: Array, x2: Array, /) -> Array:
    x1_, x2_ = _promote_two(x1, x2)
    return _wrap(tf.reshape(x1_, (-1, 1)) * tf.reshape(x2_, (1, -1)))


def cross(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    x1_, x2_ = _promote_two(x1, x2)
    if (x1_.shape[axis] is not None and x1_.shape[axis] != 3) or (
        x2_.shape[axis] is not None and x2_.shape[axis] != 3
    ):
        raise ValueError("cross product axis must have size 3")
    x1_ = _moveaxis(x1_, axis, -1)
    x2_ = _moveaxis(x2_, axis, -1)
    shape = tf.broadcast_static_shape(x1_.shape, x2_.shape)
    shape = (
        tf.broadcast_dynamic_shape(tf.shape(x1_), tf.shape(x2_))
        if not shape.is_fully_defined()
        else shape
    )
    x1_, x2_ = tf.broadcast_to(x1_, shape), tf.broadcast_to(x2_, shape)
    return _wrap(_moveaxis(tf.linalg.cross(x1_, x2_), -1, axis))


def eigh(x: Array, /) -> EighResult:
    values, vectors = tf.linalg.eigh(_unwrap(x))
    return EighResult(_wrap(values), _wrap(vectors))


def eig(x: Array, /) -> EigResult:
    values, vectors = tf.linalg.eig(_unwrap(x))
    return EigResult(_wrap(values), _wrap(vectors))


def eigvals(x: Array, /) -> Array:
    values, _ = tf.linalg.eig(_unwrap(x))
    return _wrap(values)


def eigvalsh(x: Array, /) -> Array:
    return _wrap(tf.linalg.eigvalsh(_unwrap(x)))


def det(x: Array, /) -> Array:
    return _wrap(tf.linalg.det(_unwrap(x)))


def inv(x: Array, /) -> Array:
    return _wrap(tf.linalg.inv(_unwrap(x)))


def qr(x: Array, /, *, mode: str = "reduced") -> QRResult:
    if mode not in ("reduced", "complete"):
        raise ValueError("mode must be 'reduced' or 'complete'")
    res = tf.linalg.qr(_replace_nonfinite(_unwrap(x)), full_matrices=mode == "complete")
    return QRResult(_wrap(_replace_nonfinite(res.q)), _wrap(_replace_nonfinite(res.r)))


def slogdet(x: Array, /) -> SlogdetResult:
    res = tf.linalg.slogdet(_unwrap(x))
    return SlogdetResult(_wrap(res.sign), _wrap(tf.math.real(res.log_abs_determinant)))


def svd(x: Array, /, *, full_matrices: bool = True) -> SVDResult:
    s, u, v = tf.linalg.svd(_unwrap(x), full_matrices=full_matrices, compute_uv=True)
    vh = tf.linalg.matrix_transpose(tf.math.conj(v))
    return SVDResult(_wrap(u), _wrap(s), _wrap(vh))


def cholesky(x: Array, /, *, upper: bool = False) -> Array:
    out = tf.linalg.cholesky(_unwrap(x))
    if upper:
        out = tf.linalg.matrix_transpose(out)
        if _is_complex(out.dtype):
            out = tf.math.conj(out)
    return _wrap(out)


def matrix_rank(x: Array, /, *, rtol: float | Array | None = None) -> Array:
    tensor = _unwrap(x)
    if tensor.shape.rank < 2:
        raise ValueError(
            "1-dimensional array given. Array must be at least two-dimensional"
        )
    s = _unwrap(svdvals(x))
    if rtol is None:
        tol = (
            tf.reduce_max(s, axis=-1, keepdims=True)
            * max(tensor.shape[-2:])
            * finfo(s.dtype).eps
        )
    else:
        tol = (
            tf.reduce_max(s, axis=-1, keepdims=True)
            * tf.cast(_unwrap(rtol), s.dtype)[..., tf.newaxis]
        )
    return count_nonzero(Array._from_tensor(s > tol), axis=-1)


def pinv(x: Array, /, *, rtol: float | Array | None = None) -> Array:
    tensor = _unwrap(x)
    s, u, v = tf.linalg.svd(tensor, full_matrices=False, compute_uv=True)
    if rtol is None:
        rtol = max(tensor.shape[-2:]) * finfo(tensor.dtype).eps
    rtol_ = tf.cast(_unwrap(rtol), s.dtype)
    if rtol_.shape.rank != 0:
        rtol_ = rtol_[..., tf.newaxis]
    cutoff = tf.reduce_max(s, axis=-1, keepdims=True) * rtol_
    s_inv = tf.where(s > cutoff, tf.math.reciprocal(s), tf.zeros((), dtype=s.dtype))
    v_scaled = v * tf.cast(s_inv[..., tf.newaxis, :], v.dtype)
    return _wrap(
        tf.linalg.matmul(v_scaled, tf.linalg.matrix_transpose(tf.math.conj(u)))
    )


def matrix_norm(
    x: Array,
    /,
    *,
    keepdims: bool = False,
    ord: int | float | str | None = "fro",
) -> Array:
    tensor = _unwrap(x)
    out_dtype = _real_dtype_for(tensor.dtype)
    abs_x = tf.cast(tf.abs(tensor), out_dtype)

    if ord in (None, "fro"):
        out = tf.sqrt(tf.reduce_sum(tf.square(abs_x), axis=(-2, -1)))
    elif ord == 1:
        out = tf.reduce_max(tf.reduce_sum(abs_x, axis=-2), axis=-1)
    elif ord == -1:
        out = tf.reduce_min(tf.reduce_sum(abs_x, axis=-2), axis=-1)
    elif ord == float("inf"):
        out = tf.reduce_max(tf.reduce_sum(abs_x, axis=-1), axis=-1)
    elif ord == -float("inf"):
        out = tf.reduce_min(tf.reduce_sum(abs_x, axis=-1), axis=-1)
    elif ord in (2, -2, "nuc"):
        s = tf.linalg.svd(tensor, compute_uv=False)
        if ord == 2:
            out = tf.reduce_max(s, axis=-1)
        elif ord == -2:
            out = tf.reduce_min(s, axis=-1)
        else:
            out = tf.reduce_sum(s, axis=-1)
    else:
        raise ValueError(f"unsupported matrix norm order: {ord!r}")

    out = tf.cast(out, out_dtype)
    if keepdims:
        out = tf.reshape(out, _shape_tuple(tensor)[:-2] + (1, 1))
    return _wrap(out)


def matrix_power(x: Array, n: int, /) -> Array:
    tensor = _unwrap(x)
    if n == 0:
        eye = tf.eye(tensor.shape[-1], dtype=tensor.dtype)
        return _wrap(tf.broadcast_to(eye, _shape_tuple(tensor)))
    if n < 0:
        tensor = tf.linalg.inv(tensor)
        n = -n
    result = tensor
    for _ in range(n - 1):
        result = tf.linalg.matmul(result, tensor)
    return _wrap(result)


def solve(x1: Array, x2: Array, /) -> Array:
    x1_, x2_ = _promote_two(x1, x2)
    squeeze = False
    if x2_.shape.rank == 1:
        stack_shape = _shape_tuple(x1_)[:-2]
        x2_ = tf.reshape(x2_, (1,) * len(stack_shape) + _shape_tuple(x2_) + (1,))
        x2_ = tf.broadcast_to(x2_, stack_shape + (_shape_tuple(x1_)[-1], 1))
        squeeze = True
    else:
        stack_shape = tuple(
            tf.broadcast_static_shape(
                tf.TensorShape(_shape_tuple(x1_)[:-2]),
                tf.TensorShape(_shape_tuple(x2_)[:-2]),
            ).as_list()
        )
        x1_ = tf.broadcast_to(x1_, stack_shape + _shape_tuple(x1_)[-2:])
        x2_ = tf.broadcast_to(x2_, stack_shape + _shape_tuple(x2_)[-2:])
    out = tf.linalg.solve(x1_, x2_)
    return _wrap(tf.squeeze(out, axis=-1) if squeeze else out)


def svdvals(x: Array, /) -> Array:
    return _wrap(tf.linalg.svd(_unwrap(x), compute_uv=False))


def diagonal(x: Array, /, *, offset: int = 0) -> Array:
    tensor = _unwrap(x)
    if tensor.shape.rank < 2:
        raise ValueError("x must be at least 2-dimensional for diagonal")
    if offset >= tensor.shape[-1] or offset <= -tensor.shape[-2]:
        return _wrap(tf.zeros(_shape_tuple(tensor)[:-2] + (0,), dtype=tensor.dtype))
    return _wrap(tf.linalg.diag_part(tensor, k=offset))


def trace(x: Array, /, *, offset: int = 0, dtype: tf.DType | None = None) -> Array:
    return xp_sum(diagonal(x, offset=offset), axis=-1, dtype=dtype)


def vector_norm(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: int | float = 2,
) -> Array:
    tensor = _unwrap(x)
    out_dtype = _real_dtype_for(tensor.dtype)
    if axis == ():
        return _wrap(
            tf.cast(tensor != 0, out_dtype)
            if ord == 0
            else tf.cast(tf.abs(tensor), out_dtype)
        )

    if axis is None:
        x_ = tf.reshape(tensor, (-1,))
        axis_ = 0
    elif isinstance(axis, tuple):
        axes = tuple(a + tensor.shape.rank if a < 0 else a for a in axis)
        rest = tuple(i for i in range(tensor.shape.rank) if i not in axes)
        x_ = tf.transpose(tensor, axes + rest)
        axis_size = math.prod(tensor.shape[a] for a in axes)
        x_ = tf.reshape(x_, (axis_size, *[tensor.shape[i] for i in rest]))
        axis_ = 0
    else:
        x_ = tensor
        axis_ = axis

    abs_x = tf.cast(tf.abs(x_), out_dtype)
    if ord == 0:
        out = tf.cast(
            count_nonzero(Array._from_tensor(x_), axis=axis_).unwrap(), out_dtype
        )
    elif ord == 1:
        out = tf.reduce_sum(abs_x, axis=axis_)
    elif ord == 2:
        out = tf.sqrt(tf.reduce_sum(tf.square(abs_x), axis=axis_))
    elif ord == float("inf"):
        out = tf.reduce_max(abs_x, axis=axis_)
    elif ord == -float("inf"):
        out = tf.reduce_min(abs_x, axis=axis_)
    else:
        p = tf.cast(ord, out_dtype)
        out = tf.pow(tf.reduce_sum(tf.pow(abs_x, p), axis=axis_), 1 / p)

    if keepdims:
        shape = list(_shape_tuple(tensor))
        axes = (
            range(tensor.shape.rank)
            if axis is None
            else (axis if isinstance(axis, tuple) else (axis,))
        )
        for a in axes:
            shape[a] = 1
        out = tf.reshape(out, shape)
    return _wrap(out)


def norm(
    x: Array,
    ord: int | float | str | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    if isinstance(axis, tuple) and len(axis) == 2:
        return matrix_norm(x, ord="fro" if ord is None else ord, keepdims=keepdims)
    return vector_norm(x, ord=2 if ord is None else ord, axis=axis, keepdims=keepdims)


__all__ = [
    "EigResult",
    "EighResult",
    "QRResult",
    "SVDResult",
    "SlogdetResult",
    "cholesky",
    "cross",
    "diagonal",
    "det",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "inv",
    "matmul",
    "matrix_norm",
    "norm",
    "matrix_power",
    "matrix_rank",
    "matrix_transpose",
    "outer",
    "pinv",
    "qr",
    "slogdet",
    "solve",
    "svd",
    "svdvals",
    "tensordot",
    "trace",
    "vecdot",
    "vector_norm",
]


def __dir__() -> list[str]:
    return __all__
