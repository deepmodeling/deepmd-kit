# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Real spherical harmonics matching the e3nn convention used by SeZM.

This module is a torch-free, pure-numpy replacement for the exact e3nn call
used by the SeZM Lebedev S2-grid projection code
(``deepmd/pt/model/descriptor/sezm_nn/projection.py``)::

    e3nn.o3.spherical_harmonics(
        list(range(lmax + 1)), vecs, normalize=True, normalization="norm"
    )

Empirically verified e3nn convention facts (probed on basis vectors and
random points, matching to machine precision):

- Output layout: for each degree ``l`` from 0 to ``lmax``, a block of
  ``2l + 1`` components ordered ``m = -l, ..., +l``; component ``l*l + l + m``
  is the order-``m`` harmonic.
- Axis convention: e3nn uses ``y`` as the polar axis. The ``l = 1`` block
  under ``normalization="norm"`` is exactly the unit input vector
  ``(x, y, z)`` in ``m = (-1, 0, +1)`` order, i.e. the standard z-polar real
  spherical harmonics evaluated with axes ``(x_s, y_s, z_s) = (z, x, y)``.
- Phase convention: no Condon-Shortley phase. Negative orders carry
  ``sin(m * phi)``, positive orders ``cos(m * phi)`` with
  ``phi = atan2(x, z)``.
- Normalization ``"norm"``: ``Y_lm = sqrt(4*pi / (2l+1)) * Y_lm^{orthonormal}``
  so that ``sum_m Y_lm(v)^2 = 1`` for every unit vector ``v``.
- ``normalize=True``: input vectors are normalized internally before
  evaluation, making the output invariant to the input vector length.
- Zero vectors: e3nn clamps the norm instead of dividing by zero, so a zero
  input vector yields ``[Y00, 0, 0, ...] = [1, 0, 0, ...]``. This module
  reproduces that exactly (no NaN, no runtime warning).
"""

import numpy as np

__all__ = ["real_spherical_harmonics"]


def real_spherical_harmonics(vecs: np.ndarray, lmax: int) -> np.ndarray:
    r"""Evaluate real spherical harmonics in the e3nn ``"norm"`` convention.

    Exactly matches ``e3nn.o3.spherical_harmonics(list(range(lmax + 1)),
    vecs, normalize=True, normalization="norm")`` (see module docstring).

    For :math:`m>0`, using the e3nn polar-axis convention, the components are

    .. math::

       Y_{l,-m}=\sqrt{2}\,N_{lm}P_l^m(\cos\theta)\sin(m\phi),\qquad
       Y_{l,m}=\sqrt{2}\,N_{lm}P_l^m(\cos\theta)\cos(m\phi),

    with :math:`Y_{l,0}=N_{l0}P_l(\cos\theta)` and normalization chosen so
    that :math:`\sum_{m=-l}^lY_{lm}(\hat{\mathbf r})^2=1`.

    Parameters
    ----------
    vecs
        Input vectors with shape ``(..., 3)``. They do not need to be
        normalized; vectors are normalized internally (e3nn
        ``normalize=True``). Zero vectors map to ``[1, 0, 0, ...]``
        exactly as in e3nn (which clamps the norm before dividing).
    lmax
        Maximum angular degree, any non-negative integer.

    Returns
    -------
    np.ndarray
        Real spherical harmonics with shape ``(..., (lmax + 1) ** 2)`` in
        float64, with each degree-``l`` block ordered ``m = -l, ..., +l``.
    """
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")
    vecs = np.asarray(vecs, dtype=np.float64)
    if vecs.ndim == 0 or vecs.shape[-1] != 3:
        raise ValueError(f"vecs must have shape (..., 3), got {vecs.shape}")
    lead_shape = vecs.shape[:-1]
    vecs = vecs.reshape(-1, 3)
    # normalize=True: e3nn normalizes the input vectors internally.
    # e3nn clamps the norm, so zero vectors stay zero instead of becoming
    # NaN; the corresponding l >= 1 output components are zeroed below.
    norm = np.linalg.norm(vecs, axis=-1, keepdims=True)
    zero_mask = norm == 0.0
    vecs = vecs / np.where(zero_mask, 1.0, norm)
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    # e3nn polar axis is y: standard axes (x_s, y_s, z_s) = (z, x, y)
    cos_theta = np.clip(y, -1.0, 1.0)
    sin_theta = np.hypot(x, z)
    phi = np.arctan2(x, z)

    nbatch = vecs.shape[0]
    # Fully normalized associated Legendre functions (Condon-Shortley-free):
    #   pbar[l][m] = sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!) * P_l^m(cos_theta)
    # computed with the standard stable recurrences on the normalized
    # functions (no factorials, no overflow for large lmax). See DLMF
    # §14.10 (https://dlmf.nist.gov/14.10) and Holmes & Featherstone,
    # J. Geodesy 76, 279-299 (2002), doi:10.1007/s00190-002-0216-2.
    pbar = [[None] * (ll + 1) for ll in range(lmax + 1)]
    pbar[0][0] = np.full(nbatch, np.sqrt(1.0 / (4.0 * np.pi)), dtype=np.float64)
    # diagonal: pbar[m][m]
    for m in range(1, lmax + 1):
        pbar[m][m] = (
            np.sqrt((2.0 * m + 1.0) / (2.0 * m)) * sin_theta * pbar[m - 1][m - 1]
        )
    # first off-diagonal: pbar[m+1][m]
    for m in range(lmax):
        pbar[m + 1][m] = np.sqrt(2.0 * m + 3.0) * cos_theta * pbar[m][m]
    # remaining: three-term recurrence in l
    for m in range(lmax + 1):
        for ll in range(m + 2, lmax + 1):
            a_lm = np.sqrt((4.0 * ll * ll - 1.0) / (ll * ll - m * m))
            b_lm = np.sqrt(((ll - 1.0) ** 2 - m * m) / (4.0 * (ll - 1.0) ** 2 - 1.0))
            pbar[ll][m] = a_lm * (cos_theta * pbar[ll - 1][m] - b_lm * pbar[ll - 2][m])

    out = np.zeros((nbatch, (lmax + 1) ** 2), dtype=np.float64)
    sqrt2 = np.sqrt(2.0)
    # sin(m*phi) / cos(m*phi): computed once per order m, reused across l
    sin_mphi = [None] + [np.sin(m * phi) for m in range(1, lmax + 1)]
    cos_mphi = [None] + [np.cos(m * phi) for m in range(1, lmax + 1)]
    for ll in range(lmax + 1):
        # normalization="norm": scale orthonormal SH by sqrt(4*pi/(2l+1))
        scale = np.sqrt(4.0 * np.pi / (2.0 * ll + 1.0))
        center = ll * ll + ll
        out[:, center] = scale * pbar[ll][0]
        for m in range(1, ll + 1):
            base = sqrt2 * scale * pbar[ll][m]
            out[:, center - m] = base * sin_mphi[m]
            out[:, center + m] = base * cos_mphi[m]
    # e3nn zero-vector behavior: the clamped-norm zero vector evaluates the
    # degree-l polynomials at the origin, giving exactly [1, 0, 0, ...].
    # The theta/phi recurrences above are ill-defined there, so enforce it.
    zero_out = np.zeros((lmax + 1) ** 2, dtype=np.float64)
    zero_out[0] = 1.0
    out = np.where(zero_mask, zero_out, out)
    return out.reshape(*lead_shape, (lmax + 1) ** 2)
