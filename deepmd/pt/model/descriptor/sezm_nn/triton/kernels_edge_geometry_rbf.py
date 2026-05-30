# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN201, ANN202
"""Triton kernels for the SeZM edge geometry/RBF chain.

This file implements the standard non-compile path hot segment:

``coord_gather -> edge_vec -> edge_len -> inner_clamp -> edge_env -> edge_rbf``

The kernels intentionally stop before Wigner-D construction so the existing eager
quaternion/Wigner path remains unchanged.
"""

from __future__ import (
    annotations,
)

import triton
import triton.language as tl


@triton.jit
def _pow_int(x, power: tl.constexpr):
    """Raise ``x`` to a small compile-time integer power."""
    out = x * 0.0 + 1.0
    for _ in tl.static_range(power):
        out = out * x
    return out


@triton.jit
def _safe_sinc_no_pi(x):
    """Compute ``sin(x) / x`` with a short Taylor branch near zero."""
    x2 = x * x
    approx = 1.0 - x2 / 6.0 + (x2 * x2) / 120.0
    regular = tl.sin(x) / x
    return tl.where(tl.abs(x) < 1.0e-4, approx, regular)


@triton.jit
def _safe_sinc_grad_no_pi(x):
    """Compute ``d/dx [sin(x) / x]`` with a short Taylor branch near zero."""
    x2 = x * x
    approx = -x / 3.0 + (x * x2) / 30.0
    regular = (x * tl.cos(x) - tl.sin(x)) / x2
    return tl.where(tl.abs(x) < 1.0e-4, approx, regular)


@triton.jit
def _compute_cutoff_envelope(
    r,
    rcut,
    a,
    b,
    c,
    d,
    exponent: tl.constexpr,
):
    """Evaluate the C^3 cutoff envelope on one distance vector."""
    x = tl.maximum(0.0, tl.minimum(r / rcut, 1.0))
    poly = a + x * (b + x * (c + x * d))
    env = 1.0 + _pow_int(x, exponent) * poly
    return tl.where(x < 1.0, env, 0.0)


@triton.jit
def _compute_cutoff_envelope_grad(
    r,
    rcut,
    a,
    b,
    c,
    d,
    exponent: tl.constexpr,
):
    """Evaluate ``d envelope / d r`` on one distance vector."""
    x = tl.maximum(0.0, tl.minimum(r / rcut, 1.0))
    poly = a + x * (b + x * (c + x * d))
    poly_grad = b + 2.0 * c * x + 3.0 * d * x * x
    if exponent == 1:
        leading = poly
    else:
        leading = float(exponent) * _pow_int(x, exponent - 1) * poly
    grad_x = leading + _pow_int(x, exponent) * poly_grad
    return tl.where(x < 1.0, grad_x / rcut, 0.0)


@triton.jit
def _apply_inner_clamp(
    raw_len,
    r_inner,
    r_outer,
):
    """Apply the septic Hermite inner clamp."""
    delta = r_outer - r_inner
    t = tl.maximum(0.0, tl.minimum((raw_len - r_inner) / delta, 1.0))
    t2 = t * t
    t4 = t2 * t2
    h = t4 * (20.0 + t * (-45.0 + t * (36.0 - 10.0 * t)))
    clamped = r_inner + delta * h
    return tl.where(raw_len >= r_outer, raw_len, clamped)


@triton.jit
def _apply_inner_clamp_grad(
    raw_len,
    r_inner,
    r_outer,
):
    """Evaluate ``d clamp / d raw_len`` for the septic Hermite inner clamp."""
    delta = r_outer - r_inner
    t = tl.maximum(0.0, tl.minimum((raw_len - r_inner) / delta, 1.0))
    t2 = t * t
    t3 = t2 * t
    grad = t3 * (80.0 + t * (-225.0 + t * (216.0 - 70.0 * t)))
    return tl.where(raw_len >= r_outer, 1.0, grad)


@triton.jit
def edge_geometry_rbf_forward_kernel(
    coord_ptr,
    center_index_ptr,
    neighbor_index_ptr,
    freq_ptr,
    edge_vec_ptr,
    edge_len_ptr,
    edge_env_ptr,
    edge_rbf_ptr,
    num_edges,
    n_radial,
    coord_stride_n,
    coord_stride_c,
    edge_vec_stride_e,
    edge_vec_stride_c,
    edge_rbf_stride_e,
    edge_rbf_stride_r,
    eps,
    rcut,
    edge_env_a,
    edge_env_b,
    edge_env_c,
    edge_env_d,
    radial_env_a,
    radial_env_b,
    radial_env_c,
    radial_env_d,
    r_inner,
    r_outer,
    EDGE_ENV_EXPONENT: tl.constexpr,
    RADIAL_ENV_EXPONENT: tl.constexpr,
    HAS_INNER_CLAMP: tl.constexpr,
    BLOCK_EDGE: tl.constexpr,
    BLOCK_RADIAL: tl.constexpr,
):
    """Compute the fused edge geometry/RBF chain for one edge/radial tile."""
    pid_edge = tl.program_id(0)
    pid_radial = tl.program_id(1)

    edge_offsets = pid_edge * BLOCK_EDGE + tl.arange(0, BLOCK_EDGE)
    radial_offsets = pid_radial * BLOCK_RADIAL + tl.arange(0, BLOCK_RADIAL)
    edge_mask = edge_offsets < num_edges
    radial_mask = radial_offsets < n_radial
    first_radial_mask = edge_mask & (pid_radial == 0)

    center_index = tl.load(center_index_ptr + edge_offsets, mask=edge_mask, other=0)
    neighbor_index = tl.load(neighbor_index_ptr + edge_offsets, mask=edge_mask, other=0)

    center_x = tl.load(
        coord_ptr + center_index * coord_stride_n + 0 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    center_y = tl.load(
        coord_ptr + center_index * coord_stride_n + 1 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    center_z = tl.load(
        coord_ptr + center_index * coord_stride_n + 2 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    neighbor_x = tl.load(
        coord_ptr + neighbor_index * coord_stride_n + 0 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    neighbor_y = tl.load(
        coord_ptr + neighbor_index * coord_stride_n + 1 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    neighbor_z = tl.load(
        coord_ptr + neighbor_index * coord_stride_n + 2 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )

    diff_x = neighbor_x - center_x
    diff_y = neighbor_y - center_y
    diff_z = neighbor_z - center_z
    raw_len = tl.sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + eps * eps)

    if HAS_INNER_CLAMP:
        clamped_len = _apply_inner_clamp(raw_len, r_inner, r_outer)
        scale = clamped_len / raw_len
        edge_vec_x = diff_x * scale
        edge_vec_y = diff_y * scale
        edge_vec_z = diff_z * scale
        edge_len = clamped_len
    else:
        edge_vec_x = diff_x
        edge_vec_y = diff_y
        edge_vec_z = diff_z
        edge_len = raw_len

    edge_env = _compute_cutoff_envelope(
        edge_len,
        rcut,
        edge_env_a,
        edge_env_b,
        edge_env_c,
        edge_env_d,
        exponent=EDGE_ENV_EXPONENT,
    )
    radial_env = _compute_cutoff_envelope(
        edge_len,
        rcut,
        radial_env_a,
        radial_env_b,
        radial_env_c,
        radial_env_d,
        exponent=RADIAL_ENV_EXPONENT,
    )

    tl.store(
        edge_vec_ptr + edge_offsets * edge_vec_stride_e + 0 * edge_vec_stride_c,
        edge_vec_x,
        mask=first_radial_mask,
    )
    tl.store(
        edge_vec_ptr + edge_offsets * edge_vec_stride_e + 1 * edge_vec_stride_c,
        edge_vec_y,
        mask=first_radial_mask,
    )
    tl.store(
        edge_vec_ptr + edge_offsets * edge_vec_stride_e + 2 * edge_vec_stride_c,
        edge_vec_z,
        mask=first_radial_mask,
    )
    tl.store(edge_len_ptr + edge_offsets, edge_len, mask=first_radial_mask)
    tl.store(edge_env_ptr + edge_offsets, edge_env, mask=first_radial_mask)

    freqs = tl.load(freq_ptr + radial_offsets, mask=radial_mask, other=0.0)
    phase = edge_len[:, None] * freqs[None, :]
    raw = freqs[None, :] * _safe_sinc_no_pi(phase)
    edge_rbf = raw * radial_env[:, None]
    tl.store(
        edge_rbf_ptr
        + edge_offsets[:, None] * edge_rbf_stride_e
        + radial_offsets[None, :] * edge_rbf_stride_r,
        edge_rbf,
        mask=edge_mask[:, None] & radial_mask[None, :],
    )


@triton.jit
def edge_geometry_rbf_bwd_accum_kernel(
    grad_edge_len_ptr,
    grad_edge_env_ptr,
    grad_edge_rbf_ptr,
    coord_ptr,
    center_index_ptr,
    neighbor_index_ptr,
    freq_ptr,
    grad_r_total_ptr,
    grad_freq_ptr,
    num_edges,
    n_radial,
    coord_stride_n,
    coord_stride_c,
    grad_edge_rbf_stride_e,
    grad_edge_rbf_stride_r,
    eps,
    rcut,
    edge_env_a,
    edge_env_b,
    edge_env_c,
    edge_env_d,
    radial_env_a,
    radial_env_b,
    radial_env_c,
    radial_env_d,
    r_inner,
    r_outer,
    EDGE_ENV_EXPONENT: tl.constexpr,
    RADIAL_ENV_EXPONENT: tl.constexpr,
    HAS_INNER_CLAMP: tl.constexpr,
    BLOCK_EDGE: tl.constexpr,
    BLOCK_RADIAL: tl.constexpr,
):
    """Accumulate scalar distance gradients and frequency gradients."""
    pid_edge = tl.program_id(0)
    pid_radial = tl.program_id(1)

    edge_offsets = pid_edge * BLOCK_EDGE + tl.arange(0, BLOCK_EDGE)
    radial_offsets = pid_radial * BLOCK_RADIAL + tl.arange(0, BLOCK_RADIAL)
    edge_mask = edge_offsets < num_edges
    radial_mask = radial_offsets < n_radial

    center_index = tl.load(center_index_ptr + edge_offsets, mask=edge_mask, other=0)
    neighbor_index = tl.load(neighbor_index_ptr + edge_offsets, mask=edge_mask, other=0)

    center_x = tl.load(
        coord_ptr + center_index * coord_stride_n + 0 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    center_y = tl.load(
        coord_ptr + center_index * coord_stride_n + 1 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    center_z = tl.load(
        coord_ptr + center_index * coord_stride_n + 2 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    neighbor_x = tl.load(
        coord_ptr + neighbor_index * coord_stride_n + 0 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    neighbor_y = tl.load(
        coord_ptr + neighbor_index * coord_stride_n + 1 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    neighbor_z = tl.load(
        coord_ptr + neighbor_index * coord_stride_n + 2 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )

    diff_x = neighbor_x - center_x
    diff_y = neighbor_y - center_y
    diff_z = neighbor_z - center_z
    raw_len = tl.sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + eps * eps)

    if HAS_INNER_CLAMP:
        edge_len = _apply_inner_clamp(raw_len, r_inner, r_outer)
    else:
        edge_len = raw_len

    radial_env = _compute_cutoff_envelope(
        edge_len,
        rcut,
        radial_env_a,
        radial_env_b,
        radial_env_c,
        radial_env_d,
        exponent=RADIAL_ENV_EXPONENT,
    )
    radial_env_grad = _compute_cutoff_envelope_grad(
        edge_len,
        rcut,
        radial_env_a,
        radial_env_b,
        radial_env_c,
        radial_env_d,
        exponent=RADIAL_ENV_EXPONENT,
    )

    grad_edge_rbf = tl.load(
        grad_edge_rbf_ptr
        + edge_offsets[:, None] * grad_edge_rbf_stride_e
        + radial_offsets[None, :] * grad_edge_rbf_stride_r,
        mask=edge_mask[:, None] & radial_mask[None, :],
        other=0.0,
    )
    freqs = tl.load(freq_ptr + radial_offsets, mask=radial_mask, other=0.0)
    phase = edge_len[:, None] * freqs[None, :]
    raw = freqs[None, :] * _safe_sinc_no_pi(phase)
    raw_grad_r = freqs[None, :] * freqs[None, :] * _safe_sinc_grad_no_pi(phase)
    radial_grad_r = raw_grad_r * radial_env[:, None] + raw * radial_env_grad[:, None]
    grad_rbf_to_r = tl.sum(grad_edge_rbf * radial_grad_r, axis=1)
    tl.atomic_add(grad_r_total_ptr + edge_offsets, grad_rbf_to_r, mask=edge_mask)

    grad_freq = tl.sum(grad_edge_rbf * (radial_env[:, None] * tl.cos(phase)), axis=0)
    tl.atomic_add(grad_freq_ptr + radial_offsets, grad_freq, mask=radial_mask)

    if pid_radial == 0:
        grad_edge_len = tl.load(
            grad_edge_len_ptr + edge_offsets, mask=edge_mask, other=0.0
        )
        grad_edge_env = tl.load(
            grad_edge_env_ptr + edge_offsets, mask=edge_mask, other=0.0
        )
        edge_env_grad = _compute_cutoff_envelope_grad(
            edge_len,
            rcut,
            edge_env_a,
            edge_env_b,
            edge_env_c,
            edge_env_d,
            exponent=EDGE_ENV_EXPONENT,
        )
        base = grad_edge_len + grad_edge_env * edge_env_grad
        tl.atomic_add(grad_r_total_ptr + edge_offsets, base, mask=edge_mask)


@triton.jit
def edge_geometry_rbf_bwd_coord_kernel(
    grad_edge_vec_ptr,
    grad_r_total_ptr,
    coord_ptr,
    center_index_ptr,
    neighbor_index_ptr,
    grad_coord_ptr,
    num_edges,
    coord_stride_n,
    coord_stride_c,
    grad_edge_vec_stride_e,
    grad_edge_vec_stride_c,
    grad_coord_stride_n,
    grad_coord_stride_c,
    eps,
    r_inner,
    r_outer,
    HAS_INNER_CLAMP: tl.constexpr,
    BLOCK_EDGE: tl.constexpr,
):
    """Backpropagate the fused geometry/RBF chain into flat coordinates."""
    pid_edge = tl.program_id(0)
    edge_offsets = pid_edge * BLOCK_EDGE + tl.arange(0, BLOCK_EDGE)
    edge_mask = edge_offsets < num_edges

    center_index = tl.load(center_index_ptr + edge_offsets, mask=edge_mask, other=0)
    neighbor_index = tl.load(neighbor_index_ptr + edge_offsets, mask=edge_mask, other=0)

    center_x = tl.load(
        coord_ptr + center_index * coord_stride_n + 0 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    center_y = tl.load(
        coord_ptr + center_index * coord_stride_n + 1 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    center_z = tl.load(
        coord_ptr + center_index * coord_stride_n + 2 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    neighbor_x = tl.load(
        coord_ptr + neighbor_index * coord_stride_n + 0 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    neighbor_y = tl.load(
        coord_ptr + neighbor_index * coord_stride_n + 1 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    neighbor_z = tl.load(
        coord_ptr + neighbor_index * coord_stride_n + 2 * coord_stride_c,
        mask=edge_mask,
        other=0.0,
    )

    diff_x = neighbor_x - center_x
    diff_y = neighbor_y - center_y
    diff_z = neighbor_z - center_z
    raw_len = tl.sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + eps * eps)

    if HAS_INNER_CLAMP:
        edge_len = _apply_inner_clamp(raw_len, r_inner, r_outer)
        clamp_grad = _apply_inner_clamp_grad(raw_len, r_inner, r_outer)
        scale = edge_len / raw_len
    else:
        edge_len = raw_len
        clamp_grad = raw_len * 0.0 + 1.0
        scale = raw_len * 0.0 + 1.0

    grad_edge_vec_x = tl.load(
        grad_edge_vec_ptr
        + edge_offsets * grad_edge_vec_stride_e
        + 0 * grad_edge_vec_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    grad_edge_vec_y = tl.load(
        grad_edge_vec_ptr
        + edge_offsets * grad_edge_vec_stride_e
        + 1 * grad_edge_vec_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    grad_edge_vec_z = tl.load(
        grad_edge_vec_ptr
        + edge_offsets * grad_edge_vec_stride_e
        + 2 * grad_edge_vec_stride_c,
        mask=edge_mask,
        other=0.0,
    )
    grad_r_total = tl.load(grad_r_total_ptr + edge_offsets, mask=edge_mask, other=0.0)

    dot_grad_vec = (
        grad_edge_vec_x * diff_x + grad_edge_vec_y * diff_y + grad_edge_vec_z * diff_z
    )
    inv_raw_len = 1.0 / raw_len
    scalar = grad_r_total * clamp_grad + dot_grad_vec * (
        (clamp_grad * raw_len - edge_len) * inv_raw_len * inv_raw_len
    )
    grad_diff_common = scalar * inv_raw_len
    grad_diff_x = grad_edge_vec_x * scale + diff_x * grad_diff_common
    grad_diff_y = grad_edge_vec_y * scale + diff_y * grad_diff_common
    grad_diff_z = grad_edge_vec_z * scale + diff_z * grad_diff_common

    tl.atomic_add(
        grad_coord_ptr + neighbor_index * grad_coord_stride_n + 0 * grad_coord_stride_c,
        grad_diff_x,
        mask=edge_mask,
    )
    tl.atomic_add(
        grad_coord_ptr + neighbor_index * grad_coord_stride_n + 1 * grad_coord_stride_c,
        grad_diff_y,
        mask=edge_mask,
    )
    tl.atomic_add(
        grad_coord_ptr + neighbor_index * grad_coord_stride_n + 2 * grad_coord_stride_c,
        grad_diff_z,
        mask=edge_mask,
    )
    tl.atomic_add(
        grad_coord_ptr + center_index * grad_coord_stride_n + 0 * grad_coord_stride_c,
        -grad_diff_x,
        mask=edge_mask,
    )
    tl.atomic_add(
        grad_coord_ptr + center_index * grad_coord_stride_n + 1 * grad_coord_stride_c,
        -grad_diff_y,
        mask=edge_mask,
    )
    tl.atomic_add(
        grad_coord_ptr + center_index * grad_coord_stride_n + 2 * grad_coord_stride_c,
        -grad_diff_z,
        mask=edge_mask,
    )
