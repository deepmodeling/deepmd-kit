# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SO(3) packed-index and projection helpers for SeZM.

This module defines the packed `(l, m)` indexing helpers and the projection
utilities used by the SeZM equivariant operators.
"""

from __future__ import (
    annotations,
)

import torch


def get_so3_dim_of_lmax(lmax: int) -> int:
    """
    Return SO(3) representation dimension for given lmax.

    The dimension equals::

        sum_{l<=lmax} (2l+1) = (lmax+1)^2

    which is the number of spherical harmonics basis functions.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.

    Returns
    -------
    int
        The SO(3) dimension D = (lmax+1)^2.
    """
    return int((int(lmax) + 1) ** 2)


def map_degree_idx(lmax: int, *, device: torch.device) -> torch.Tensor:
    """
    Build degree (l) index for each position in the packed (l, m) layout.

    For each spherical harmonic coefficient position in the packed tensor,
    returns the corresponding angular momentum quantum number l.

    Examples
    --------
    For lmax=2, the packed layout has D=9 positions:
    - Position 0: l=0, m=0
    - Positions 1-3: l=1, m=-1,0,+1
    - Positions 4-8: l=2, m=-2,-1,0,+1,+2

    Returns: [0, 1,1,1, 2,2,2,2,2]

    Parameters
    ----------
    lmax
        Maximum angular momentum degree.
    device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Integer tensor with shape (D,), where D=(lmax+1)^2.
        Each element is the l value for that position.
    """
    lmax = int(lmax)
    counts = torch.tensor(
        [2 * degree + 1 for degree in range(lmax + 1)],
        device=device,
        dtype=torch.long,
    )
    return torch.repeat_interleave(
        torch.arange(lmax + 1, device=device, dtype=torch.long), counts
    )


def project_D_to_m(
    D_full: torch.Tensor,
    coeff_index_m: torch.Tensor,
    ebed_dim_full: int,
    cache: dict[str, torch.Tensor] | None,
    key_lmax: int,
    key_mmax: int,
) -> torch.Tensor:
    """
    Row-project block-diagonal Wigner-D to the m-major truncated layout.

    Parameters
    ----------
    D_full
        Block-diagonal Wigner-D with shape (E, D, D).
    coeff_index_m
        Indices for m-major reduced layout with shape (D_m_trunc,).
    ebed_dim_full
        Full SO(3) dimension D_full = (lmax+1)^2 to slice the block.
    cache
        Optional cache mapping (lmax, mmax) -> projected matrix.
    key_lmax
        lmax used to build coeff_index_m (cache key).
    key_mmax
        mmax used to build coeff_index_m (cache key).

    Returns
    -------
    torch.Tensor
        Projected rotation matrix with shape (E, D_m_trunc, D).

    Examples
    --------
    For lmax=2, mmax=1 (D=9, D_m_trunc=7), coeff_index_m selects
    [0,2,6,1,5,3,7] in packed (l,m) order. The returned tensor keeps only those
    rows of ``D_full`` while retaining all columns, so that rotating and truncating
    is done in a single bmm: ``x_local = D_to_m @ x_global``.
    """
    cache_key = f"{int(key_lmax)}:{int(key_mmax)}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    D_block = D_full[:, :ebed_dim_full, :ebed_dim_full]
    proj = D_block.index_select(1, coeff_index_m)
    if cache is not None:
        cache[cache_key] = proj
    return proj


def project_Dt_from_m(
    Dt_full: torch.Tensor,
    coeff_index_m: torch.Tensor,
    ebed_dim_full: int,
    cache: dict[str, torch.Tensor] | None,
    key_lmax: int,
    key_mmax: int,
) -> torch.Tensor:
    """
    Column-project block-diagonal Wigner-D^T for inverse rotation.

    Parameters
    ----------
    Dt_full
        Block-diagonal Wigner-D^T with shape (E, D, D).
    coeff_index_m
        Indices for m-major reduced layout with shape (D_m_trunc,).
    ebed_dim_full
        Full SO(3) dimension D_full = (lmax+1)^2 to slice the block.
    cache
        Optional cache mapping (lmax, mmax) -> projected matrix.
    key_lmax
        lmax used to build coeff_index_m (cache key).
    key_mmax
        mmax used to build coeff_index_m (cache key).

    Returns
    -------
    torch.Tensor
        Projected inverse rotation matrix with shape (E, D, D_m_trunc).

    Examples
    --------
    Continuing lmax=2, mmax=1, the projection selects the same column subset
    [0,2,6,1,5,3,7] from ``Dt_full``. This enables inverse rotation with missing
    coefficients implicitly zeroed: ``x_global = Dt_from_m @ x_local``.
    """
    cache_key = f"{int(key_lmax)}:{int(key_mmax)}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    Dt_block = Dt_full[:, :ebed_dim_full, :ebed_dim_full]
    proj = Dt_block.index_select(2, coeff_index_m)
    if cache is not None:
        cache[cache_key] = proj
    return proj


def so3_packed_index(degree: int, m: int) -> int:
    """
    Compute packed (l, m) index for real spherical harmonics layout.

    The packed layout is l-primary with m ordered as ``-l..+l`` inside each l-block.
    The index formula is::

        idx(l, m) = l^2 + l + m

    Parameters
    ----------
    degree
        Degree l.
    m
        Order m, must satisfy ``-l <= m <= l``.

    Returns
    -------
    int
        Packed index.
    """
    degree = int(degree)
    m = int(m)
    return degree * degree + degree + m


def build_l_major_index(lmax: int, mmax: int, *, device: torch.device) -> torch.Tensor:
    """
    Build coefficient indices for l-major layout truncated by mmax.

    The returned indices select coefficients with ``|m| <= min(mmax, l)`` in the
    standard packed (l, m) layout. The order is l-major:

    - l = 0..lmax
    - within each l, m = -min(mmax, l) .. +min(mmax, l)

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum order (|m|). Must satisfy ``0 <= mmax <= lmax``.
    device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Long tensor of indices with shape (D_m_trunc,), selecting coefficients
        from the full packed layout with D=(lmax+1)^2, where D_m_trunc is
        the number of coefficients kept under ``|m| <= min(mmax, l)``.

    Examples
    --------
    For lmax=2, mmax=1:
    - Full packed layout: l=0(0), l=1(1-3), l=2(4-8)
    - Truncated by mmax=1: skip (l=2, m=±2) at indices 4,8
    - Returns: [0, 1, 2, 3, 5, 6, 7]
    """
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    if lmax_i < 0:
        raise ValueError("`lmax` must be non-negative")
    if mmax_i < 0:
        raise ValueError("`mmax` must be non-negative")
    if mmax_i > lmax_i:
        raise ValueError("`mmax` must be <= `lmax`")

    indices: list[int] = []
    for degree in range(lmax_i + 1):
        m_keep = min(mmax_i, degree)
        for m in range(-m_keep, m_keep + 1):
            indices.append(so3_packed_index(degree, m))
    return torch.tensor(indices, device=device, dtype=torch.long)


def build_m_major_index(lmax: int, mmax: int, *, device: torch.device) -> torch.Tensor:
    """
    Build coefficient indices for m-major layout truncated by mmax.

    This layout minimizes rotation cost and avoids gather-heavy indexing:

    - m = 0: l = 0..lmax (single coefficient per l)
    - for each m = 1..mmax:
        - negative part: l = m..lmax, coefficient (l, -m)
        - positive part: l = m..lmax, coefficient (l, +m)

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum order (|m|). Must satisfy ``0 <= mmax <= lmax``.
    device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Long tensor of indices with shape (D_m_trunc,), selecting coefficients
        from the full packed layout with D=(lmax+1)^2, where D_m_trunc is
        the number of coefficients kept under ``|m| <= min(mmax, l)``.

    Examples
    --------
    For lmax=2, mmax=1:
    - m=0 group: (l=0,m=0)→0, (l=1,m=0)→2, (l=2,m=0)→6
    - m=1 neg group: (l=1,m=-1)→1, (l=2,m=-1)→5
    - m=1 pos group: (l=1,m=+1)→3, (l=2,m=+1)→7
    - Returns: [0, 2, 6, 1, 5, 3, 7]
    """
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    if lmax_i < 0:
        raise ValueError("`lmax` must be non-negative")
    if mmax_i < 0:
        raise ValueError("`mmax` must be non-negative")
    if mmax_i > lmax_i:
        raise ValueError("`mmax` must be <= `lmax`")

    indices: list[int] = []
    # === Step 1. m = 0 group (l = 0..lmax) ===
    for degree in range(lmax_i + 1):
        indices.append(so3_packed_index(degree, 0))

    # === Step 2. m > 0 groups (neg then pos) ===
    for m in range(1, mmax_i + 1):
        for degree in range(m, lmax_i + 1):
            indices.append(so3_packed_index(degree, -m))
        for degree in range(m, lmax_i + 1):
            indices.append(so3_packed_index(degree, m))

    return torch.tensor(indices, device=device, dtype=torch.long)


def build_m_major_l_index(
    lmax: int, mmax: int, *, device: torch.device
) -> torch.Tensor:
    """
    Build degree (l) index aligned with `build_m_major_index`.

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum order (|m|). Must satisfy ``0 <= mmax <= lmax``.
    device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Long tensor of degrees with shape (D_m_trunc,). Entry i is the degree
        l for the i-th coefficient in the m-major layout.

    Examples
    --------
    For lmax=2, mmax=1:
    - m=0 group: l=0,1,2
    - m=1 neg group: l=1,2
    - m=1 pos group: l=1,2
    - Returns: [0, 1, 2, 1, 2, 1, 2]
    """
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    if lmax_i < 0:
        raise ValueError("`lmax` must be non-negative")
    if mmax_i < 0:
        raise ValueError("`mmax` must be non-negative")
    if mmax_i > lmax_i:
        raise ValueError("`mmax` must be <= `lmax`")

    degrees: list[int] = []
    # === Step 1. m = 0 group ===
    for degree in range(lmax_i + 1):
        degrees.append(degree)

    # === Step 2. m > 0 groups (neg then pos) ===
    for m in range(1, mmax_i + 1):
        for degree in range(m, lmax_i + 1):
            degrees.append(degree)
        for degree in range(m, lmax_i + 1):
            degrees.append(degree)

    return torch.tensor(degrees, device=device, dtype=torch.long)


def build_rotate_inv_rescale(
    lmax: int,
    mmax: int,
    degree_index: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build reduced-layout inverse-rotation rescale factors.

    When ``mmax < lmax``, the reduced local layout keeps only ``2*mmax+1`` orders
    for each degree ``l > mmax``. The inverse rotation rescales those truncated
    degrees by ``sqrt((2*l+1)/(2*mmax+1))`` so the reduced representation matches
    the amplitude expected by the full SO(3) basis.

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum order (|m|). Must satisfy ``0 <= mmax <= lmax``.
    degree_index
        Degree index aligned with the reduced coefficient layout, typically
        returned by ``build_m_major_l_index``.
    device
        Device for the returned tensor.
    dtype
        Floating-point dtype for the returned tensor.

    Returns
    -------
    torch.Tensor
        Rescale vector with shape (D_m_trunc,), aligned with the reduced
        coefficient layout.
    """
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    if lmax_i < 0:
        raise ValueError("`lmax` must be non-negative")
    if mmax_i < 0:
        raise ValueError("`mmax` must be non-negative")
    if mmax_i > lmax_i:
        raise ValueError("`mmax` must be <= `lmax`")

    degrees = degree_index.to(device=device, dtype=torch.long)
    rescale = torch.ones(degrees.shape[0], device=device, dtype=dtype)
    if mmax_i == lmax_i:
        return rescale

    mask = degrees > mmax_i
    if mask.any():
        denom = float(2 * mmax_i + 1)
        degree_values = degrees[mask].to(dtype=dtype)
        rescale[mask] = torch.sqrt((2.0 * degree_values + 1.0) / denom)
    return rescale
