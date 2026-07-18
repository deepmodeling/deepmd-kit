# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202
"""Universal fused smooth environment-matrix kernel (``DP_TRITON_INFER >= 1``).

The smooth environment matrix is the shared front end of every ``se``-family
descriptor (``se_a``, ``se_r``, ``se_t``, ``se_atten`` / DPA1, and the DPA2/DPA3
representation blocks): from the neighbor list it forms, per center ``i`` and
neighbor slot ``n`` (with relative vector ``d = r_j - r_i`` and distance
``L = |d|``), the normalized quartet

    env[0]     = (s(L) / (L + p)            - avg[0]) / std[0]
    env[1..3]  = (s(L) * d / (L + p)^2      - avg[1..3]) / std[1..3]

where ``s`` is the smooth switch (quintic ``compute_smooth_weight`` or the
exponential ``compute_exp_sw``), ``p`` the division protection, and ``avg``/
``std`` the per-type running statistics.  It also returns the relative vector
``diff = d`` and the switch value ``sw = s(L)``; invalid / padding slots
(``nlist < 0``) carry a zero switch, a zero ``diff``, and hence ``env = -avg/std``
(the switch multiplies the raw quartet before the affine normalization).

The eager path expresses this as a chain of roughly ten memory-bound tensor ops
whose autograd backward -- the force path -- expands into as many gather / norm /
switch / division gradient kernels; on a 4k-atom cell that backward dominates the
front end (~0.75 ms).  This module fuses the forward into a single node-parallel
kernel and replaces the autograd backward with a closed-form one: one kernel
evaluates the ``d env / d d`` Jacobian (contracted against the upstream grads of
all three outputs) into a per-edge ``d`` gradient, and a single scatter-add sends
it to the neighbor (``+``) and center (``-``) coordinates.

The analytic Jacobian per slot, with ``q = L + p``, ``W = s(L)`` (masked),
``W' = s'(L)`` (masked), ``g_raw = g_env / std``, and ``G = <g_raw[1..3], d>``::

    g_d = (d / L) * [ g_raw[0] (W'/q - W/q^2) + G (W'/q^2 - 2 W/q^3) ]
        + (W / q^2) * g_raw[1..3]
        + g_diff
        + g_sw * W' * (d / L)

The kernel is inference-only (``register_autograd`` provides the first-order
backward used for forces; higher-order / training differentiation keeps the eager
path) and is registered as a ``triton_op`` so it is captured as a single opaque
node under ``make_fx`` / ``torch.export`` (the ``pt_expt`` backend).  Off CUDA or
with Triton unavailable it transparently falls back to the validated eager
reference, so it is a drop-in for the descriptors' ``prod_env_mat`` front end.
"""

from __future__ import (
    annotations,
)

import torch
from torch import (
    Tensor,
)
from torch.library import (
    triton_op,
    wrap_triton,
)

from deepmd.dpmodel.utils.safe_gradient import (
    safe_for_vector_norm,
)
from deepmd.kernels.utils import (
    triton_infer_level,
)

# Constant of the exponential switch ``compute_exp_sw`` (``a = C / rmin``).
_EXP_SWITCH_C = 20.0

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

__all__ = [
    "TRITON_AVAILABLE",
    "edge_env_mat",
    "env_mat",
]


if TRITON_AVAILABLE:

    @triton.jit
    def _switch(L, rmin, rmax, sw_c, USE_EXP: tl.constexpr):
        """Smooth switch ``s(L)`` and its derivative ``s'(L)`` (clamped window).

        Matches ``compute_smooth_weight`` (quintic) / ``compute_exp_sw``
        (double exponential); the derivative is zero outside the smooth window,
        consistent with the ``clamp`` in the eager definitions.
        """
        if USE_EXP:
            a = sw_c / rmin
            inside = (L > 0.0) & (L < rmax)
            lc = tl.minimum(tl.maximum(L, 0.0), rmax)
            xarg = a * (lc - rmin)
            e = tl.exp(xarg)
            w = tl.exp(-e)
            # d(exp(-e))/dL = -a e w = -a exp(xarg - e); the fused exponent stays
            # finite as e -> inf (xarg - e -> -inf), avoiding the 0*inf NaN that
            # the factored ``-a e w`` produces once ``e`` overflows in fp32.
            dw = tl.where(inside, -a * tl.exp(xarg - e), 0.0)
        else:
            inside = (L > rmin) & (L < rmax)
            u = (L - rmin) / (rmax - rmin)
            u = tl.minimum(tl.maximum(u, 0.0), 1.0)
            u2 = u * u
            w = u2 * u * (-6.0 * u2 + 15.0 * u - 10.0) + 1.0
            dw = tl.where(
                inside,
                (-30.0 * u2 * u2 + 60.0 * u2 * u - 30.0 * u2) / (rmax - rmin),
                0.0,
            )
        return w, dw

    @triton.jit
    def _env_mat_fwd_kernel(
        coord_ptr,  # (nf, nall, 3)
        nlist_ptr,  # (nf, nloc, nnei) int
        atype_ptr,  # (nf, nloc) int
        mean_ptr,  # (ntypes, nnei, C)
        std_ptr,  # (ntypes, nnei, C)
        env_ptr,  # (nf, nloc, nnei, C)
        diff_ptr,  # (nf, nloc, nnei, 3)
        sw_ptr,  # (nf, nloc, nnei)
        rcut,
        rcut_smth,
        protection,
        sw_c,
        nloc,
        nnei,
        nall,
        C: tl.constexpr,
        RADIAL: tl.constexpr,
        USE_EXP: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One program per center; a lane per neighbor slot (padded to BLOCK_N)."""
        pid = tl.program_id(0)
        f = pid // nloc
        i = pid % nloc
        n = tl.arange(0, BLOCK_N)
        nmask = n < nnei

        ci = (f * nall + i) * 3
        cix = tl.load(coord_ptr + ci + 0)
        ciy = tl.load(coord_ptr + ci + 1)
        ciz = tl.load(coord_ptr + ci + 2)
        ti = tl.load(atype_ptr + f * nloc + i)

        jj = tl.load(nlist_ptr + pid * nnei + n, mask=nmask, other=-1)
        m = jj >= 0
        js = tl.where(m, jj, 0)
        cj = (f * nall + js) * 3
        cjx = tl.load(coord_ptr + cj + 0, mask=nmask, other=0.0)
        cjy = tl.load(coord_ptr + cj + 1, mask=nmask, other=0.0)
        cjz = tl.load(coord_ptr + cj + 2, mask=nmask, other=0.0)
        dx = cjx - cix
        dy = cjy - ciy
        dz = cjz - ciz
        length = tl.sqrt(dx * dx + dy * dy + dz * dz)
        # invalid slots take L=1 to keep q finite; the zero switch nulls the raw
        # quartet and the diff, so their contribution vanishes regardless.
        lsafe = tl.where(m, length, 1.0)
        q = lsafe + protection
        sw_raw, _ = _switch(lsafe, rcut_smth, rcut, sw_c, USE_EXP)
        w = tl.where(m, sw_raw, 0.0)
        inv_q = 1.0 / q

        base = pid * nnei * C + n * C
        mb = ti * nnei * C + n * C
        raw0 = w * inv_q
        mean0 = tl.load(mean_ptr + mb, mask=nmask, other=0.0)
        std0 = tl.load(std_ptr + mb, mask=nmask, other=1.0)
        tl.store(env_ptr + base, (raw0 - mean0) / std0, mask=nmask)
        tl.store(sw_ptr + pid * nnei + n, w, mask=nmask)

        db = pid * nnei * 3 + n * 3
        mf = tl.where(m, 1.0, 0.0)
        tl.store(diff_ptr + db + 0, dx * mf, mask=nmask)
        tl.store(diff_ptr + db + 1, dy * mf, mask=nmask)
        tl.store(diff_ptr + db + 2, dz * mf, mask=nmask)

        if not RADIAL:
            inv_q2 = inv_q * inv_q
            wq2 = w * inv_q2
            for c in tl.static_range(3):
                dc = tl.where(c == 0, dx, tl.where(c == 1, dy, dz))
                meanc = tl.load(mean_ptr + mb + (c + 1), mask=nmask, other=0.0)
                stdc = tl.load(std_ptr + mb + (c + 1), mask=nmask, other=1.0)
                tl.store(
                    env_ptr + base + (c + 1), (wq2 * dc - meanc) / stdc, mask=nmask
                )

    @triton.jit
    def _env_mat_bwd_kernel(
        coord_ptr,  # (nf, nall, 3)
        nlist_ptr,  # (nf, nloc, nnei)
        atype_ptr,  # (nf, nloc)
        std_ptr,  # (ntypes, nnei, C)
        genv_ptr,  # (nf, nloc, nnei, C)
        gdiff_ptr,  # (nf, nloc, nnei, 3)
        gsw_ptr,  # (nf, nloc, nnei)
        grad_ptr,  # (nf, nall, 3) output: grad wrt coordinates (zero-initialized)
        rcut,
        rcut_smth,
        protection,
        sw_c,
        nloc,
        nnei,
        nall,
        C: tl.constexpr,
        RADIAL: tl.constexpr,
        USE_EXP: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Closed-form ``d out / d d`` scattered to the coordinate gradient.

        The per-slot ``d`` gradient goes to the neighbor coordinate (``+``, an
        atomic add since neighbors are shared across centers) and, summed over
        the slots, to the center coordinate (``-``); fusing the scatter here
        avoids a chain of eager tensor ops (the eager backward's dominant
        launch-overhead cost).
        """
        pid = tl.program_id(0)
        f = pid // nloc
        i = pid % nloc
        n = tl.arange(0, BLOCK_N)
        nmask = n < nnei

        ci = (f * nall + i) * 3
        cix = tl.load(coord_ptr + ci + 0)
        ciy = tl.load(coord_ptr + ci + 1)
        ciz = tl.load(coord_ptr + ci + 2)
        ti = tl.load(atype_ptr + f * nloc + i)

        jj = tl.load(nlist_ptr + pid * nnei + n, mask=nmask, other=-1)
        m = jj >= 0
        js = tl.where(m, jj, 0)
        cj = (f * nall + js) * 3
        dx = tl.load(coord_ptr + cj + 0, mask=nmask, other=0.0) - cix
        dy = tl.load(coord_ptr + cj + 1, mask=nmask, other=0.0) - ciy
        dz = tl.load(coord_ptr + cj + 2, mask=nmask, other=0.0) - ciz
        length = tl.sqrt(dx * dx + dy * dy + dz * dz)
        lsafe = tl.where(m, length, 1.0)
        q = lsafe + protection
        sw_raw, dsw_raw = _switch(lsafe, rcut_smth, rcut, sw_c, USE_EXP)
        w = tl.where(m, sw_raw, 0.0)
        wp = tl.where(m, dsw_raw, 0.0)
        inv_q = 1.0 / q
        inv_q2 = inv_q * inv_q
        inv_q3 = inv_q2 * inv_q
        inv_l = tl.where(m, 1.0 / lsafe, 0.0)

        mb = ti * nnei * C + n * C
        gb = pid * nnei * C + n * C
        std0 = tl.load(std_ptr + mb, mask=nmask, other=1.0)
        g0 = tl.load(genv_ptr + gb, mask=nmask, other=0.0) / std0

        gvx = tl.zeros((BLOCK_N,), dtype=g0.dtype)
        gvy = tl.zeros((BLOCK_N,), dtype=g0.dtype)
        gvz = tl.zeros((BLOCK_N,), dtype=g0.dtype)
        gdot = tl.zeros((BLOCK_N,), dtype=g0.dtype)
        wq2 = w * inv_q2
        if not RADIAL:
            std1 = tl.load(std_ptr + mb + 1, mask=nmask, other=1.0)
            std2 = tl.load(std_ptr + mb + 2, mask=nmask, other=1.0)
            std3 = tl.load(std_ptr + mb + 3, mask=nmask, other=1.0)
            gvx = tl.load(genv_ptr + gb + 1, mask=nmask, other=0.0) / std1
            gvy = tl.load(genv_ptr + gb + 2, mask=nmask, other=0.0) / std2
            gvz = tl.load(genv_ptr + gb + 3, mask=nmask, other=0.0) / std3
            gdot = gvx * dx + gvy * dy + gvz * dz

        gsw = tl.load(gsw_ptr + pid * nnei + n, mask=nmask, other=0.0)
        # coef multiplies d/L: the g_env[0] + g_env[1..3] + g_sw radial terms.
        coef = (
            g0 * (wp * inv_q - w * inv_q2)
            + gdot * (wp * inv_q2 - 2.0 * w * inv_q3)
            + gsw * wp
        ) * inv_l

        db = pid * nnei * 3 + n * 3
        gdfx = tl.load(gdiff_ptr + db + 0, mask=nmask, other=0.0)
        gdfy = tl.load(gdiff_ptr + db + 1, mask=nmask, other=0.0)
        gdfz = tl.load(gdiff_ptr + db + 2, mask=nmask, other=0.0)
        mf = tl.where(m, 1.0, 0.0)
        gdx = (coef * dx + wq2 * gvx + gdfx) * mf
        gdy = (coef * dy + wq2 * gvy + gdfy) * mf
        gdz = (coef * dz + wq2 * gvz + gdfz) * mf
        # scatter d(d = r_j - r_i): +grad to the neighbor row, -sum to the center.
        smask = nmask & m
        cj = (f * nall + js) * 3
        tl.atomic_add(grad_ptr + cj + 0, gdx, mask=smask)
        tl.atomic_add(grad_ptr + cj + 1, gdy, mask=smask)
        tl.atomic_add(grad_ptr + cj + 2, gdz, mask=smask)
        tl.atomic_add(grad_ptr + ci + 0, -tl.sum(gdx, axis=0))
        tl.atomic_add(grad_ptr + ci + 1, -tl.sum(gdy, axis=0))
        tl.atomic_add(grad_ptr + ci + 2, -tl.sum(gdz, axis=0))

    @triton.jit
    def _edge_env_mat_fwd_kernel(
        edge_vec_ptr,  # (E, 3)
        ctype_ptr,  # (E,) int center-atom type
        emask_ptr,  # (E,) int valid-edge flag (1 valid, 0 padding)
        mean_ptr,  # (ntypes, 4) slot-independent
        std_ptr,  # (ntypes, 4)
        env_ptr,  # (E, 4)
        sw_ptr,  # (E,) smooth switch, zeroed on padding
        rcut,
        rcut_smth,
        protection,
        sw_c,
        n_edge,
        BLOCK_E: tl.constexpr,
    ):
        """Graph-native (slot-free) environment matrix; one lane per edge.

        The quintic switch is not masked in the env quartet: padding edges
        (``valid == 0``) instead take ``length + 1`` so the switch decays past
        the cutoff, matching the dense ``length + ~mask`` guard.  The quartet
        rows carry nonzero values on padding and are masked downstream by the
        edge mask.  The separately emitted switch ``sw`` is zeroed on padding,
        mirroring the dense ``weight * mask`` (it feeds the strip type-pair gate).
        """
        pid = tl.program_id(0)
        e = pid * BLOCK_E + tl.arange(0, BLOCK_E)
        m = e < n_edge
        dx = tl.load(edge_vec_ptr + e * 3 + 0, mask=m, other=0.0)
        dy = tl.load(edge_vec_ptr + e * 3 + 1, mask=m, other=0.0)
        dz = tl.load(edge_vec_ptr + e * 3 + 2, mask=m, other=0.0)
        ct = tl.load(ctype_ptr + e, mask=m, other=0)
        valid = tl.load(emask_ptr + e, mask=m, other=0)
        length = tl.sqrt(dx * dx + dy * dy + dz * dz)
        length = length + tl.where(valid != 0, 0.0, 1.0)
        denom = length + protection
        sw, _ = _switch(length, rcut_smth, rcut, sw_c, 0)
        inv_d = 1.0 / denom
        inv_d2 = inv_d * inv_d
        mb = ct * 4
        m0 = tl.load(mean_ptr + mb + 0, mask=m, other=0.0)
        s0 = tl.load(std_ptr + mb + 0, mask=m, other=1.0)
        tl.store(env_ptr + e * 4 + 0, (sw * inv_d - m0) / s0, mask=m)
        tl.store(sw_ptr + e, tl.where(valid != 0, sw, 0.0), mask=m)
        for c in tl.static_range(3):
            dc = tl.where(c == 0, dx, tl.where(c == 1, dy, dz))
            mc = tl.load(mean_ptr + mb + (c + 1), mask=m, other=0.0)
            sc = tl.load(std_ptr + mb + (c + 1), mask=m, other=1.0)
            tl.store(env_ptr + e * 4 + (c + 1), (sw * dc * inv_d2 - mc) / sc, mask=m)

    @triton.jit
    def _edge_env_mat_bwd_kernel(
        edge_vec_ptr,  # (E, 3)
        ctype_ptr,  # (E,)
        emask_ptr,  # (E,)
        std_ptr,  # (ntypes, 4)
        genv_ptr,  # (E, 4)
        gsw_ptr,  # (E,) upstream gradient of the switch (strip gate)
        gedge_ptr,  # (E, 3) output: grad wrt edge_vec
        rcut,
        rcut_smth,
        protection,
        sw_c,
        n_edge,
        BLOCK_E: tl.constexpr,
    ):
        """Closed-form ``d env / d edge_vec`` per edge; edge_vec is the leaf, so
        no scatter -- each edge writes its own gradient row directly.

        The env quartet and the separately emitted switch ``sw`` share the same
        radial dependence on ``|edge_vec|``, so the ``sw`` cotangent folds into
        the common radial coefficient as ``g_sw * s'(L)``. The ``inv_l`` guard
        (zero at a zero-length padding edge) keeps that term finite.
        """
        pid = tl.program_id(0)
        e = pid * BLOCK_E + tl.arange(0, BLOCK_E)
        m = e < n_edge
        dx = tl.load(edge_vec_ptr + e * 3 + 0, mask=m, other=0.0)
        dy = tl.load(edge_vec_ptr + e * 3 + 1, mask=m, other=0.0)
        dz = tl.load(edge_vec_ptr + e * 3 + 2, mask=m, other=0.0)
        ct = tl.load(ctype_ptr + e, mask=m, other=0)
        valid = tl.load(emask_ptr + e, mask=m, other=0)
        length = tl.sqrt(dx * dx + dy * dy + dz * dz)
        # safe-norm gradient: zero at a zero-length (padding) edge.
        inv_l = tl.where(length > 0.0, 1.0 / length, 0.0)
        length = length + tl.where(valid != 0, 0.0, 1.0)
        denom = length + protection
        sw, dsw = _switch(length, rcut_smth, rcut, sw_c, 0)
        inv_d = 1.0 / denom
        inv_d2 = inv_d * inv_d
        inv_d3 = inv_d2 * inv_d
        mb = ct * 4
        s0 = tl.load(std_ptr + mb + 0, mask=m, other=1.0)
        s1 = tl.load(std_ptr + mb + 1, mask=m, other=1.0)
        s2 = tl.load(std_ptr + mb + 2, mask=m, other=1.0)
        s3 = tl.load(std_ptr + mb + 3, mask=m, other=1.0)
        g0 = tl.load(genv_ptr + e * 4 + 0, mask=m, other=0.0) / s0
        gvx = tl.load(genv_ptr + e * 4 + 1, mask=m, other=0.0) / s1
        gvy = tl.load(genv_ptr + e * 4 + 2, mask=m, other=0.0) / s2
        gvz = tl.load(genv_ptr + e * 4 + 3, mask=m, other=0.0) / s3
        gsw = tl.load(gsw_ptr + e, mask=m, other=0.0)
        gdot = gvx * dx + gvy * dy + gvz * dz
        coef = (
            g0 * (dsw * inv_d - sw * inv_d2)
            + gdot * (dsw * inv_d2 - 2.0 * sw * inv_d3)
            + gsw * dsw
        ) * inv_l
        swq2 = sw * inv_d2
        tl.store(gedge_ptr + e * 3 + 0, coef * dx + swq2 * gvx, mask=m)
        tl.store(gedge_ptr + e * 3 + 1, coef * dy + swq2 * gvy, mask=m)
        tl.store(gedge_ptr + e * 3 + 2, coef * dz + swq2 * gvz, mask=m)


def _switch_reference(
    length: Tensor, rmin: float, rmax: float, use_exp: bool
) -> tuple[Tensor, Tensor]:
    """Eager smooth switch and its derivative (clamped-window semantics)."""
    if use_exp:
        a = _EXP_SWITCH_C / rmin
        inside = (length > 0.0) & (length < rmax)
        lc = length.clamp(0.0, rmax)
        xarg = a * (lc - rmin)
        e = torch.exp(xarg)
        w = torch.exp(-e)
        # -a e w = -a exp(xarg - e): the fused exponent stays finite as e -> inf,
        # avoiding the 0*inf NaN of the factored form once e overflows in fp32.
        dw = torch.where(inside, -a * torch.exp(xarg - e), torch.zeros_like(length))
    else:
        inside = (length > rmin) & (length < rmax)
        u = ((length - rmin) / (rmax - rmin)).clamp(0.0, 1.0)
        u2 = u * u
        w = u2 * u * (-6.0 * u2 + 15.0 * u - 10.0) + 1.0
        dw = torch.where(
            inside,
            (-30.0 * u2 * u2 + 60.0 * u2 * u - 30.0 * u2) / (rmax - rmin),
            torch.zeros_like(length),
        )
    return w, dw


def _geometry(
    coord: Tensor, nlist: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Shared prologue: masked neighbor index, relative vector, and safe length.

    Returns ``(mask, j_safe, d, length_safe, center_index)`` with ``d`` the
    ``(nf, nloc, nnei, 3)`` relative vector and ``length_safe`` its norm with
    invalid slots pinned to 1 (kept finite; nulled downstream by the mask).
    """
    nf, nloc, nnei = nlist.shape
    mask = nlist >= 0
    j = torch.where(mask, nlist, torch.zeros_like(nlist))
    ci = coord[:, :nloc]
    cj = torch.gather(coord, 1, j.reshape(nf, -1, 1).expand(-1, -1, 3)).reshape(
        nf, nloc, nnei, 3
    )
    d = cj - ci[:, :, None, :]
    length = torch.linalg.norm(d, dim=-1)
    length_safe = torch.where(mask, length, torch.ones_like(length))
    return mask, j, d, length_safe, ci


def _env_mat_reference(
    coord: Tensor,
    nlist: Tensor,
    atype: Tensor,
    mean: Tensor,
    stddev: Tensor,
    rcut: float,
    rcut_smth: float,
    radial_only: bool,
    protection: float,
    use_exp_switch: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Eager environment matrix, numerically identical to ``prod_env_mat``."""
    mask, _, d, length_safe, _ = _geometry(coord, nlist)
    q = length_safe + protection
    sw_raw, _ = _switch_reference(length_safe, rcut_smth, rcut, use_exp_switch)
    w = (sw_raw * mask).unsqueeze(-1)
    t0 = 1.0 / q.unsqueeze(-1)
    if radial_only:
        raw = t0 * w
    else:
        raw = torch.cat([t0, d / q.unsqueeze(-1) ** 2], dim=-1) * w
    env = (raw - mean[atype]) / stddev[atype]
    return env, d * mask.unsqueeze(-1), (sw_raw * mask).unsqueeze(-1)


def _env_mat_grad_coord_reference(
    coord: Tensor,
    nlist: Tensor,
    atype: Tensor,
    stddev: Tensor,
    g_env: Tensor,
    g_diff: Tensor,
    g_sw: Tensor,
    rcut: float,
    rcut_smth: float,
    radial_only: bool,
    protection: float,
    use_exp_switch: bool,
) -> Tensor:
    """Eager closed-form ``grad_coord`` for the three env-matrix outputs."""
    nf, nloc, nnei = nlist.shape
    nall = coord.shape[1]
    mask, j, d, length_safe, _ = _geometry(coord, nlist)
    q = length_safe + protection
    sw_raw, dsw_raw = _switch_reference(length_safe, rcut_smth, rcut, use_exp_switch)
    w = sw_raw * mask
    wp = dsw_raw * mask
    inv_l = torch.where(mask, 1.0 / length_safe, torch.zeros_like(length_safe))
    g_raw = g_env / stddev[atype]
    g0 = g_raw[..., 0]
    if radial_only:
        gdot = torch.zeros_like(g0)
        gv = torch.zeros_like(d)
    else:
        gv = g_raw[..., 1:4]
        gdot = (gv * d).sum(-1)
    coef = (
        g0 * (wp / q - w / q**2)
        + gdot * (wp / q**2 - 2.0 * w / q**3)
        + g_sw[..., 0] * wp
    ) * inv_l
    g_d = coef[..., None] * d + (w / q**2)[..., None] * gv + g_diff
    g_d = g_d * mask[..., None]

    grad = torch.zeros(nf, nall, 3, dtype=coord.dtype, device=coord.device)
    grad = grad.scatter_add(
        1, j.reshape(nf, -1, 1).expand(-1, -1, 3), g_d.reshape(nf, -1, 3)
    )
    grad = grad.index_add(
        1, torch.arange(nloc, dtype=torch.int64, device=coord.device), -g_d.sum(2)
    )
    return grad


def _use_triton(coord: Tensor) -> bool:
    return TRITON_AVAILABLE and coord.is_cuda


def _launch(coord: Tensor, nlist: Tensor, nnei: int):
    nf, nloc = nlist.shape[0], nlist.shape[1]
    return (nf * nloc,), triton.next_power_of_2(max(nnei, 1))


def _env_mat_fwd_impl(
    coord: Tensor,
    nlist: Tensor,
    atype: Tensor,
    mean: Tensor,
    stddev: Tensor,
    rcut: float,
    rcut_smth: float,
    radial_only: bool,
    protection: float,
    use_exp_switch: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    if not _use_triton(coord):
        return _env_mat_reference(
            coord,
            nlist,
            atype,
            mean,
            stddev,
            rcut,
            rcut_smth,
            radial_only,
            protection,
            use_exp_switch,
        )
    nf, nloc, nnei = nlist.shape
    nall = coord.shape[1]
    channels = 1 if radial_only else 4
    env = torch.empty(nf, nloc, nnei, channels, dtype=coord.dtype, device=coord.device)
    diff = torch.empty(nf, nloc, nnei, 3, dtype=coord.dtype, device=coord.device)
    sw = torch.empty(nf, nloc, nnei, dtype=coord.dtype, device=coord.device)
    grid, block_n = _launch(coord, nlist, nnei)
    wrap_triton(_env_mat_fwd_kernel)[grid](
        coord.contiguous(),
        nlist.contiguous(),
        atype.contiguous(),
        mean.contiguous(),
        stddev.contiguous(),
        env,
        diff,
        sw,
        float(rcut),
        float(rcut_smth),
        float(protection),
        _EXP_SWITCH_C,
        nloc,
        nnei,
        nall,
        C=channels,
        RADIAL=radial_only,
        USE_EXP=use_exp_switch,
        BLOCK_N=block_n,
    )
    return env, diff, sw.unsqueeze(-1)


def _env_mat_grad_impl(
    coord: Tensor,
    nlist: Tensor,
    atype: Tensor,
    stddev: Tensor,
    g_env: Tensor,
    g_diff: Tensor,
    g_sw: Tensor,
    rcut: float,
    rcut_smth: float,
    radial_only: bool,
    protection: float,
    use_exp_switch: bool,
) -> Tensor:
    """Coordinate gradient ``(nf, nall, 3)`` scattered from the env-mat outputs."""
    if not _use_triton(coord):
        return _env_mat_grad_coord_reference(
            coord,
            nlist,
            atype,
            stddev,
            g_env,
            g_diff,
            g_sw,
            rcut,
            rcut_smth,
            radial_only,
            protection,
            use_exp_switch,
        )
    nf, nloc, nnei = nlist.shape
    nall = coord.shape[1]
    channels = 1 if radial_only else 4
    grad = torch.zeros(nf, nall, 3, dtype=coord.dtype, device=coord.device)
    grid, block_n = _launch(coord, nlist, nnei)
    wrap_triton(_env_mat_bwd_kernel)[grid](
        coord.contiguous(),
        nlist.contiguous(),
        atype.contiguous(),
        stddev.contiguous(),
        g_env.contiguous(),
        g_diff.contiguous(),
        g_sw.contiguous(),
        grad,
        float(rcut),
        float(rcut_smth),
        float(protection),
        _EXP_SWITCH_C,
        nloc,
        nnei,
        nall,
        C=channels,
        RADIAL=radial_only,
        USE_EXP=use_exp_switch,
        BLOCK_N=block_n,
    )
    return grad


_fwd_op = triton_op("deepmd_triton::env_mat", mutates_args=())(_env_mat_fwd_impl)
_grad_op = triton_op("deepmd_triton::env_mat_grad", mutates_args=())(_env_mat_grad_impl)


@_fwd_op.register_fake
def _(
    coord,
    nlist,
    atype,
    mean,
    stddev,
    rcut,
    rcut_smth,
    radial_only,
    protection,
    use_exp_switch,
):
    nf, nloc, nnei = nlist.shape
    channels = 1 if radial_only else 4
    return (
        coord.new_empty((nf, nloc, nnei, channels)),
        coord.new_empty((nf, nloc, nnei, 3)),
        coord.new_empty((nf, nloc, nnei, 1)),
    )


@_grad_op.register_fake
def _(
    coord,
    nlist,
    atype,
    stddev,
    g_env,
    g_diff,
    g_sw,
    rcut,
    rcut_smth,
    radial_only,
    protection,
    use_exp_switch,
):
    return coord.new_empty((coord.shape[0], coord.shape[1], 3))


def _fwd_setup_context(ctx, inputs, output):
    (
        coord,
        nlist,
        atype,
        mean,
        stddev,
        rcut,
        rcut_smth,
        radial_only,
        protection,
        use_exp_switch,
    ) = inputs
    ctx.save_for_backward(coord, nlist, atype, stddev)
    ctx.cfg = (rcut, rcut_smth, radial_only, protection, use_exp_switch)


def _fwd_backward(ctx, g_env, g_diff, g_sw):
    coord, nlist, atype, stddev = ctx.saved_tensors
    rcut, rcut_smth, radial_only, protection, use_exp_switch = ctx.cfg
    grad = _grad_op(
        coord,
        nlist,
        atype,
        stddev,
        g_env,
        g_diff,
        g_sw,
        rcut,
        rcut_smth,
        radial_only,
        protection,
        use_exp_switch,
    )
    return grad, None, None, None, None, None, None, None, None, None


_fwd_op.register_autograd(_fwd_backward, setup_context=_fwd_setup_context)


def env_mat(
    coord: Tensor,
    nlist: Tensor,
    atype: Tensor,
    mean: Tensor,
    stddev: Tensor,
    rcut: float,
    rcut_smth: float,
    radial_only: bool = False,
    protection: float = 0.0,
    use_exp_switch: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fused smooth environment matrix, a drop-in for ``prod_env_mat``.

    Parameters
    ----------
    coord : Tensor
        Extended coordinates with shape (nf, nall, 3) or (nf, nall * 3).
    nlist : Tensor
        Neighbor indices with shape (nf, nloc, nnei); ``< 0`` marks padding.
    atype : Tensor
        Local atom types with shape (nf, nloc), indexing ``mean`` / ``stddev``.
    mean : Tensor
        Per-type running average with shape (ntypes, nnei, 4) -- or (ntypes,
        nnei, 1) when ``radial_only``.
    stddev : Tensor
        Per-type running standard deviation, same shape as ``mean``.
    rcut : float
        Neighbor cutoff radius, in the coordinate length unit.
    rcut_smth : float
        Inner radius where the smooth switch starts to decay.
    radial_only : bool
        Whether to emit only the radial channel (shape (..., 1)); otherwise the
        full quartet (shape (..., 4)).
    protection : float
        Additive protection on the inverse distance, guarding division by zero.
    use_exp_switch : bool
        Whether to use the exponential switch instead of the quintic one.

    Returns
    -------
    tuple of Tensor
        ``(env_mat, diff, switch)`` with shapes (nf, nloc, nnei, 4 or 1),
        (nf, nloc, nnei, 3), and (nf, nloc, nnei, 1).

    Notes
    -----
    Routes to the Triton operator at ``DP_TRITON_INFER >= 1`` on CUDA; elsewhere
    (CPU, Triton absent, or under a double-backward / training graph) it uses the
    eager reference so results are identical. The registered backward is
    first-order (the force path); training keeps the eager autograd chain.

    The scalar hyper-parameters are passed as kernel arguments (fp32) rather than
    a device tensor: a device tensor built from Python scalars forces a
    synchronizing host-to-device copy that drains the launch queue every call and
    dominates the eager cost. In fp32 they are exact; in fp64 they are exact for
    values representable in fp32 (typical ``rcut`` / ``rcut_smth`` / zero
    ``protection``) and otherwise perturb the result at ~1e-8.
    """
    if coord.dim() == 2:
        coord = coord.reshape(coord.shape[0], -1, 3)
    # Level gate only: the operator is called unconditionally so that it is
    # captured as an opaque node under a CPU ``make_fx`` trace (the pt_expt
    # freeze), while its implementation resolves the CUDA kernel vs. the eager
    # reference per the runtime device.
    if triton_infer_level() >= 1 and TRITON_AVAILABLE:
        return _fwd_op(
            coord,
            nlist,
            atype,
            mean,
            stddev,
            rcut,
            rcut_smth,
            radial_only,
            protection,
            use_exp_switch,
        )
    return _env_mat_reference(
        coord,
        nlist,
        atype,
        mean,
        stddev,
        rcut,
        rcut_smth,
        radial_only,
        protection,
        use_exp_switch,
    )


# ======================================================================
# Graph-native (edge-stream) environment matrix
# ======================================================================
# The edge form is the slot-free analogue used by the graph lower: the relative
# vector ``edge_vec = r_j - r_i`` is given directly (no neighbor gather), the
# per-type statistics are indexed by the center (dst) atom only, and the smooth
# switch is quintic. The backward is w.r.t. ``edge_vec`` (the autograd leaf on
# the graph path), so no scatter is needed -- each edge owns its gradient row.

# Edges per program block; the kernel is memory-bound, so this is not tuned.
_EDGE_BLOCK = 256


def _edge_env_mat_reference(
    edge_vec: Tensor,
    center_type: Tensor,
    davg: Tensor,
    dstd: Tensor,
    rcut: float,
    rcut_smth: float,
    protection: float,
    edge_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Eager per-edge environment matrix, identical to the dpmodel reference.

    Returns ``(env, sw)`` with ``sw`` (E, 1) the switch zeroed on padding
    edges. The safe norm keeps the autograd force gradient finite at a
    zero-length padding edge.
    """
    length = safe_for_vector_norm(edge_vec, axis=-1, keepdims=True)
    if edge_mask is not None:
        length = length + (~edge_mask.bool())[:, None].to(length.dtype)
    else:
        length = torch.where(length < 1e-10, torch.ones_like(length), length)
    denom = length + protection
    sw, _ = _switch_reference(length, rcut_smth, rcut, False)
    em = torch.cat([1.0 / denom, edge_vec / denom**2], dim=-1) * sw
    env = (em - davg[center_type]) / dstd[center_type]
    if edge_mask is not None:
        sw = sw * edge_mask[:, None].to(sw.dtype)
    return env, sw


def _edge_env_mat_grad_reference(
    edge_vec: Tensor,
    center_type: Tensor,
    dstd: Tensor,
    g_env: Tensor,
    g_sw: Tensor,
    rcut: float,
    rcut_smth: float,
    protection: float,
    edge_mask: Tensor | None,
) -> Tensor:
    """Eager closed-form ``grad_edge_vec`` for the per-edge environment matrix.

    ``g_sw`` (E, 1) is the upstream gradient of the switch output; it enters the
    shared radial coefficient as ``g_sw * s'(L)`` (mirroring the fused kernel).
    """
    d = edge_vec
    length_raw = safe_for_vector_norm(d, axis=-1)
    inv_l = torch.where(
        length_raw > 0.0, 1.0 / length_raw, torch.zeros_like(length_raw)
    )
    if edge_mask is not None:
        length = length_raw + (~edge_mask.bool()).to(d.dtype)
    else:
        length = torch.where(
            length_raw < 1e-10, torch.ones_like(length_raw), length_raw
        )
    denom = length + protection
    sw, dsw = _switch_reference(length, rcut_smth, rcut, False)
    g_raw = g_env / dstd[center_type]
    g0 = g_raw[:, 0]
    gv = g_raw[:, 1:4]
    gdot = (gv * d).sum(-1)
    coef = (
        g0 * (dsw / denom - sw / denom**2)
        + gdot * (dsw / denom**2 - 2.0 * sw / denom**3)
        + g_sw[:, 0] * dsw
    ) * inv_l
    return coef[:, None] * d + (sw / denom**2)[:, None] * gv


def _edge_env_mat_fwd_impl(
    edge_vec: Tensor,
    center_type: Tensor,
    edge_mask: Tensor,
    davg: Tensor,
    dstd: Tensor,
    rcut: float,
    rcut_smth: float,
    protection: float,
) -> tuple[Tensor, Tensor]:
    if not _use_triton(edge_vec):
        return _edge_env_mat_reference(
            edge_vec, center_type, davg, dstd, rcut, rcut_smth, protection, edge_mask
        )
    n_edge = edge_vec.shape[0]
    env = torch.empty(n_edge, 4, dtype=edge_vec.dtype, device=edge_vec.device)
    sw = torch.empty(n_edge, dtype=edge_vec.dtype, device=edge_vec.device)
    grid = (triton.cdiv(max(n_edge, 1), _EDGE_BLOCK),)
    wrap_triton(_edge_env_mat_fwd_kernel)[grid](
        edge_vec.contiguous(),
        center_type.contiguous(),
        edge_mask.to(torch.int32).contiguous(),
        davg.contiguous(),
        dstd.contiguous(),
        env,
        sw,
        float(rcut),
        float(rcut_smth),
        float(protection),
        _EXP_SWITCH_C,
        n_edge,
        BLOCK_E=_EDGE_BLOCK,
    )
    return env, sw.unsqueeze(-1)


def _edge_env_mat_grad_impl(
    edge_vec: Tensor,
    center_type: Tensor,
    edge_mask: Tensor,
    dstd: Tensor,
    g_env: Tensor,
    g_sw: Tensor,
    rcut: float,
    rcut_smth: float,
    protection: float,
) -> Tensor:
    if not _use_triton(edge_vec):
        return _edge_env_mat_grad_reference(
            edge_vec,
            center_type,
            dstd,
            g_env,
            g_sw,
            rcut,
            rcut_smth,
            protection,
            edge_mask,
        )
    n_edge = edge_vec.shape[0]
    gedge = torch.empty(n_edge, 3, dtype=edge_vec.dtype, device=edge_vec.device)
    grid = (triton.cdiv(max(n_edge, 1), _EDGE_BLOCK),)
    wrap_triton(_edge_env_mat_bwd_kernel)[grid](
        edge_vec.contiguous(),
        center_type.contiguous(),
        edge_mask.to(torch.int32).contiguous(),
        dstd.contiguous(),
        g_env.contiguous(),
        g_sw.reshape(-1).contiguous(),
        gedge,
        float(rcut),
        float(rcut_smth),
        float(protection),
        _EXP_SWITCH_C,
        n_edge,
        BLOCK_E=_EDGE_BLOCK,
    )
    return gedge


_edge_fwd_op = triton_op("deepmd_triton::edge_env_mat", mutates_args=())(
    _edge_env_mat_fwd_impl
)
_edge_grad_op = triton_op("deepmd_triton::edge_env_mat_grad", mutates_args=())(
    _edge_env_mat_grad_impl
)


@_edge_fwd_op.register_fake
def _(edge_vec, center_type, edge_mask, davg, dstd, rcut, rcut_smth, protection):
    return (
        edge_vec.new_empty((edge_vec.shape[0], 4)),
        edge_vec.new_empty((edge_vec.shape[0], 1)),
    )


@_edge_grad_op.register_fake
def _(edge_vec, center_type, edge_mask, dstd, g_env, g_sw, rcut, rcut_smth, protection):
    return edge_vec.new_empty((edge_vec.shape[0], 3))


def _edge_setup_context(ctx, inputs, output):
    edge_vec, center_type, edge_mask, davg, dstd, rcut, rcut_smth, protection = inputs
    ctx.save_for_backward(edge_vec, center_type, edge_mask, dstd)
    ctx.cfg = (rcut, rcut_smth, protection)


def _edge_backward(ctx, g_env, g_sw):
    edge_vec, center_type, edge_mask, dstd = ctx.saved_tensors
    rcut, rcut_smth, protection = ctx.cfg
    g_edge = _edge_grad_op(
        edge_vec,
        center_type,
        edge_mask,
        dstd,
        g_env,
        g_sw,
        rcut,
        rcut_smth,
        protection,
    )
    return g_edge, None, None, None, None, None, None, None


_edge_fwd_op.register_autograd(_edge_backward, setup_context=_edge_setup_context)


def edge_env_mat(
    edge_vec: Tensor,
    center_type: Tensor,
    davg: Tensor,
    dstd: Tensor,
    rcut: float,
    rcut_smth: float,
    protection: float = 0.0,
    edge_mask: Tensor | None = None,
    return_sw: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Fused per-edge environment matrix, a drop-in for the dpmodel edge form.

    Parameters
    ----------
    edge_vec : Tensor
        Relative vectors ``r_src - r_dst`` with shape (E, 3); padding edges must
        carry a zero vector.
    center_type : Tensor
        Center (dst) atom type per edge with shape (E,), indexing the statistics.
    davg : Tensor
        Per-center-type mean with shape (ntypes, 4) (slot-independent).
    dstd : Tensor
        Per-center-type standard deviation, same shape as ``davg``.
    rcut : float
        Neighbor cutoff radius.
    rcut_smth : float
        Inner radius where the smooth (quintic) switch starts to decay.
    protection : float
        Additive protection on the inverse distance.
    edge_mask : Tensor or None
        Boolean valid-edge mask with shape (E,); invalid edges take
        ``length + 1`` (matching the dense ``length + ~mask`` guard). When
        ``None`` a zero-length threshold guard is used and the eager reference
        serves the request.
    return_sw : bool
        When ``True``, also return the per-edge smooth switch (E, 1), zeroed on
        padding edges (the strip type-pair gate consumes it). Mirrors the dense
        :func:`env_mat`, which always returns the switch.

    Returns
    -------
    Tensor
        The per-edge environment matrix with shape (E, 4); or, when
        ``return_sw`` is set, the tuple ``(env, sw)`` with ``sw`` of shape
        (E, 1).

    Notes
    -----
    Routes to the Triton operator at ``DP_TRITON_INFER >= 1`` when an
    ``edge_mask`` is supplied (the graph path always provides one); it is called
    unconditionally there so a CPU ``make_fx`` trace captures it as an opaque
    node, while the implementation resolves the CUDA kernel vs. the eager
    reference per the runtime device. The registered backward differentiates
    ``edge_vec`` (the graph-path force leaf) and folds the switch cotangent when
    ``return_sw`` feeds a downstream gradient.
    """
    if triton_infer_level() >= 1 and TRITON_AVAILABLE and edge_mask is not None:
        env, sw = _edge_fwd_op(
            edge_vec, center_type, edge_mask, davg, dstd, rcut, rcut_smth, protection
        )
    else:
        env, sw = _edge_env_mat_reference(
            edge_vec, center_type, davg, dstd, rcut, rcut_smth, protection, edge_mask
        )
    return (env, sw) if return_sw else env
