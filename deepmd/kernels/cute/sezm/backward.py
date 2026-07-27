# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN201, ANN202, ANN204, ANN205
"""
CuTe-DSL fused recompute-backward kernel for the SeZM SO(2) value path.

Given the upstream gradients ``g_out`` (w.r.t. the pre-focus-compete local
features ``x_local``) and ``g_fgate`` (w.r.t. the pre-mixing ``l = 0`` scalar),
one bucketed kernel recomputes the forward value path from the small saved
inputs (``x``, ``D_to_m``, ``Kc``) entirely on chip and backpropagates it,
emitting the position-path gradients that carry the ``edge_vec`` -> force
dependence::

    grad_x       node-feature gradient, scattered to source nodes
    grad_D_to_m  Wigner-rotation gradient, atomically summed over focus
    grad_Kc      radial degree-kernel gradient, atomically summed over focus

The weights are frozen on the inference force path. No ``E x D_m x C``
intermediate reaches DRAM: the kernel recomputes the forward storing only the two
gated-layer pre-activations in shared memory, then backpropagates the residual
stack (gated-activation backward in registers/smem), the radial degree mix, and
the rotation. This kernel is specialized to the three-layer ``[gated, gated,
identity]`` mixing stack of the deployed configuration.

Buffers per CTA (one focus of a bucket of ``B`` edges): four ``B x (D_m*Cf)``
scratch tensors plus one ``max_block^2`` weight/gate scratch, inside the sm_90
shared-memory limit at ``B = 16``. All accumulation is fp32 IEEE.
"""

from __future__ import (
    annotations,
)

import torch

from .forward import (
    SEZM_CUTE_AVAILABLE,
)

if SEZM_CUTE_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.math as cmath
    from cutlass.cute.runtime import (
        from_dlpack,
    )

    from .forward import (
        ForwardRunner,
    )

    class BackwardProgram:
        """Bucketed fused recompute-backward program for the SO(2) value path.

        Parameters
        ----------
        lmax, mmax, cf, n_focus, n_layers, bucket, threads, rb, rn
            Kernel configuration (see :class:`.forward.ForwardProgram`).
        """

        def __init__(
            self,
            *,
            lmax: int,
            mmax: int,
            cf: int,
            n_focus: int,
            n_layers: int,
            bucket: int,
            threads: int,
            rb: int,
            rn: int,
        ) -> None:
            self.lmax, self.mmax, self.cf, self.nf, self.nl = (
                lmax,
                mmax,
                cf,
                n_focus,
                n_layers,
            )
            self._B, self._T, self._RB, self._RN = bucket, threads, rb, rn
            self.D = (lmax + 1) ** 2
            self.Dm = (lmax + 1) + sum(2 * (lmax - m + 1) for m in range(1, mmax + 1))
            self.Cw = n_focus * cf
            self.gate_out = lmax * cf
            self.ngroup = 1 + 2 * mmax
            self.FLAT = self.Dm * cf
            groups = [lmax + 1] + [2 * (lmax - m + 1) for m in range(1, mmax + 1)]
            self._blocks: list[tuple[int, int]] = []
            off = 0
            for g in groups:
                self._blocks.append((off, g * cf))
                off += g * cf
            self._max_sb = max(sb for _, sb in self._blocks)
            self._scr = max(
                self._max_sb * self._max_sb,
                cf * self.gate_out + 2 * bucket * self.gate_out,
            )
            assert self._B % rb == 0
            for _, sb in self._blocks:
                assert sb % rn == 0

        @cute.jit
        def __call__(
            self,
            mGout,
            mGfg,
            mX,
            mSrc,
            mDtoM,
            mKc,
            mCB,
            mW,
            mGW,
            mExpand,
            mGx,
            mGD,
            mGKc,
            n_edge: cutlass.Int32,
            n_bucket: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mGout,
                mGfg,
                mX,
                mSrc,
                mDtoM,
                mKc,
                mCB,
                mW,
                mGW,
                mExpand,
                mGx,
                mGD,
                mGKc,
                n_edge,
            ).launch(grid=[n_bucket, self.nf, 1], block=[self._T, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            mGout,
            mGfg,
            mX,
            mSrc,
            mDtoM,
            mKc,
            mCB,
            mW,
            mGW,
            mExpand,
            mGx,
            mGD,
            mGKc,
            n_edge: cutlass.Int32,
        ):
            D = cutlass.const_expr(self.D)
            Dm = cutlass.const_expr(self.Dm)
            CF = cutlass.const_expr(self.cf)
            FLAT = cutlass.const_expr(self.FLAT)
            B = cutlass.const_expr(self._B)
            T = cutlass.const_expr(self._T)

            tidx, _, _ = cute.arch.thread_idx()
            bucket, focus, _ = cute.arch.block_idx()
            e0 = bucket * B

            smem = cutlass.utils.SmemAllocator()
            b0 = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((B, FLAT), stride=(FLAT, 1)), 16
            )
            b1 = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((B, FLAT), stride=(FLAT, 1)), 16
            )
            b2 = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((B, FLAT), stride=(FLAT, 1)), 16
            )
            b3 = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((B, FLAT), stride=(FLAT, 1)), 16
            )
            s_w = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((cutlass.const_expr(self._scr),), stride=(1,)),
                16,
            )

            # === Step 1. Forward recompute: store z_0 -> b1, z_1 -> b2 ===
            # b0 carries the running layer input h_l; b3 is the activation temp.
            i = tidx
            while i < B * FLAT:
                b = i // FLAT
                rem = i % FLAT
                dm = rem // CF
                c = rem % CF
                e = e0 + b
                eq = e if e < n_edge else n_edge - 1
                src = mSrc[eq]
                acc = cutlass.Float32(0.0)
                for k in cutlass.range_constexpr(D):
                    acc += mDtoM[eq, dm, k] * mX[src, k, focus * CF + c]
                b1[b, rem] = acc
                i += T
            cute.arch.sync_threads()
            i = tidx
            while i < B * FLAT:
                b = i // FLAT
                rem = i % FLAT
                o = rem // CF
                c = rem % CF
                e = e0 + b
                eq = e if e < n_edge else n_edge - 1
                acc = cutlass.Float32(0.0)
                for ii in cutlass.range_constexpr(Dm):
                    acc += mKc[eq, o, ii] * b1[b, ii * CF + c]
                b0[b, rem] = acc * mCB[focus * CF + c]
                i += T
            cute.arch.sync_threads()
            for lyr in cutlass.range_constexpr(self.nl - 1):
                zbuf = b1 if lyr == 0 else b2
                self._gemm_fwd(b0, zbuf, s_w, mW, lyr, focus, tidx)
                self._gated_fwd(zbuf, b3, s_w, mGW, mExpand, lyr, focus, tidx)
                i = tidx
                while i < B * FLAT:
                    b = i // FLAT
                    rem = i % FLAT
                    b0[b, rem] = b0[b, rem] + b3[b, rem]
                    i += T
                cute.arch.sync_threads()

            # === Step 2. Reverse the residual stack; b0 accumulates grad_h ===
            i = tidx
            while i < B * FLAT:
                b = i // FLAT
                rem = i % FLAT
                o = rem // CF
                c = rem % CF
                b0[b, rem] = mGout[e0 + b, focus, o, c]
                i += T
            cute.arch.sync_threads()
            # layer 2 (identity): grad_z = grad_out; grad_h += W_2^T @ grad_z
            self._gemm_bwd(b0, b3, s_w, mW, self.nl - 1, focus, tidx)
            i = tidx
            while i < B * FLAT:
                b = i // FLAT
                rem = i % FLAT
                b0[b, rem] = b0[b, rem] + b3[b, rem]
                i += T
            cute.arch.sync_threads()
            # layer 1 (gated): grad_z_1 -> b2 in place; grad_h += W_1^T @ grad_z_1
            self._gated_bwd(b2, b0, s_w, mGW, mExpand, 1, focus, tidx)
            self._gemm_bwd(b2, b3, s_w, mW, 1, focus, tidx)
            i = tidx
            while i < B * FLAT:
                b = i // FLAT
                rem = i % FLAT
                b0[b, rem] = b0[b, rem] + b3[b, rem]
                i += T
            cute.arch.sync_threads()
            # layer 0 (gated): grad_z_0 -> b1 in place; grad_h += W_0^T @ grad_z_0
            self._gated_bwd(b1, b0, s_w, mGW, mExpand, 0, focus, tidx)
            self._gemm_bwd(b1, b3, s_w, mW, 0, focus, tidx)
            i = tidx
            while i < B * FLAT:
                b = i // FLAT
                rem = i % FLAT
                b0[b, rem] = b0[b, rem] + b3[b, rem]
                i += T
            cute.arch.sync_threads()
            # add the focus-competition gradient into the l=0 row
            i = tidx
            while i < B * CF:
                b = i // CF
                c = i % CF
                b0[b, c] = b0[b, c] + mGfg[e0 + b, focus, c]
                i += T
            cute.arch.sync_threads()

            # === Step 3. Radial + rotation backward; b0 holds grad_h0 ===
            # recompute x_rot -> b1
            i = tidx
            while i < B * FLAT:
                b = i // FLAT
                rem = i % FLAT
                dm = rem // CF
                c = rem % CF
                e = e0 + b
                eq = e if e < n_edge else n_edge - 1
                src = mSrc[eq]
                acc = cutlass.Float32(0.0)
                for k in cutlass.range_constexpr(D):
                    acc += mDtoM[eq, dm, k] * mX[src, k, focus * CF + c]
                b1[b, rem] = acc
                i += T
            cute.arch.sync_threads()
            # grad_Kc[o, ii] = sum_c channel_basis[c] * grad_h0[o, c] * x_rot[ii, c]
            i = tidx
            while i < B * Dm * Dm:
                b = i // (Dm * Dm)
                rem = i % (Dm * Dm)
                o = rem // Dm
                ii = rem % Dm
                e = e0 + b
                acc = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(CF):
                    acc += mCB[focus * CF + c] * b0[b, o * CF + c] * b1[b, ii * CF + c]
                cute.arch.atomic_add(mGKc.iterator + mGKc.layout((e, o, ii)), acc)
                i += T
            # grad_x_rot[ii, c] = channel_basis[c] * sum_o Kc[o, ii] * grad_h0[o, c] -> b2
            i = tidx
            while i < B * Dm * CF:
                b = i // (Dm * CF)
                rem = i % (Dm * CF)
                ii = rem // CF
                c = rem % CF
                e = e0 + b
                eq = e if e < n_edge else n_edge - 1
                acc = cutlass.Float32(0.0)
                for o in cutlass.range_constexpr(Dm):
                    acc += mKc[eq, o, ii] * b0[b, o * CF + c]
                b2[b, ii * CF + c] = acc * mCB[focus * CF + c]
                i += T
            cute.arch.sync_threads()
            # grad_x[src, k, c] += sum_ii D_to_m[ii, k] * grad_x_rot[ii, c]
            i = tidx
            while i < B * D * CF:
                b = i // (D * CF)
                rem = i % (D * CF)
                k = rem // CF
                c = rem % CF
                e = e0 + b
                eq = e if e < n_edge else n_edge - 1
                src = mSrc[eq]
                acc = cutlass.Float32(0.0)
                for ii in cutlass.range_constexpr(Dm):
                    acc += mDtoM[eq, ii, k] * b2[b, ii * CF + c]
                cute.arch.atomic_add(
                    mGx.iterator + mGx.layout((src, k, focus * CF + c)), acc
                )
                i += T
            # grad_D_to_m[ii, k] = sum_c grad_x_rot[ii, c] * x_src[k, c]
            i = tidx
            while i < B * Dm * D:
                b = i // (Dm * D)
                rem = i % (Dm * D)
                ii = rem // D
                k = rem % D
                e = e0 + b
                eq = e if e < n_edge else n_edge - 1
                src = mSrc[eq]
                acc = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(CF):
                    acc += b2[b, ii * CF + c] * mX[src, k, focus * CF + c]
                cute.arch.atomic_add(mGD.iterator + mGD.layout((e, ii, k)), acc)
                i += T

        @cute.jit
        def _gemm_fwd(self, hbuf, zbuf, s_w, mW, lyr, focus, tidx):
            """Recompute ``zbuf = hbuf @ W[lyr, focus]`` (block-diagonal)."""
            T = cutlass.const_expr(self._T)
            RB = cutlass.const_expr(self._RB)
            RN = cutlass.const_expr(self._RN)
            B = cutlass.const_expr(self._B)
            for ob, sb in cutlass.const_expr(self._blocks):
                j = tidx
                while j < sb * sb:
                    s_w[(j // sb) * sb + (j % sb)] = mW[
                        lyr, focus, ob + (j // sb), ob + (j % sb)
                    ]
                    j += T
                cute.arch.sync_threads()
                n_mt = cutlass.const_expr((B // RB) * (sb // RN))
                racc = cute.make_rmem_tensor(
                    cute.make_layout((RB, RN)), cutlass.Float32
                )
                mt = tidx
                while mt < n_mt:
                    bi = (mt // (sb // RN)) * RB
                    nj = (mt % (sb // RN)) * RN
                    for r in cutlass.range_constexpr(RB):
                        for s in cutlass.range_constexpr(RN):
                            racc[r, s] = cutlass.Float32(0.0)
                    for k in cutlass.range(sb):
                        for r in cutlass.range_constexpr(RB):
                            a_rk = hbuf[bi + r, ob + k]
                            for s in cutlass.range_constexpr(RN):
                                racc[r, s] += a_rk * s_w[k * sb + nj + s]
                    for r in cutlass.range_constexpr(RB):
                        for s in cutlass.range_constexpr(RN):
                            zbuf[bi + r, ob + nj + s] = racc[r, s]
                    mt += T
                cute.arch.sync_threads()

        @cute.jit
        def _gemm_bwd(self, gzbuf, ghbuf, s_w, mW, lyr, focus, tidx):
            """Compute ``ghbuf = W[lyr, focus]^T @ gzbuf`` (block-diagonal)."""
            T = cutlass.const_expr(self._T)
            RB = cutlass.const_expr(self._RB)
            RN = cutlass.const_expr(self._RN)
            B = cutlass.const_expr(self._B)
            for ob, sb in cutlass.const_expr(self._blocks):
                j = tidx
                while j < sb * sb:
                    s_w[(j // sb) * sb + (j % sb)] = mW[
                        lyr, focus, ob + (j // sb), ob + (j % sb)
                    ]
                    j += T
                cute.arch.sync_threads()
                n_mt = cutlass.const_expr((B // RB) * (sb // RN))
                racc = cute.make_rmem_tensor(
                    cute.make_layout((RB, RN)), cutlass.Float32
                )
                mt = tidx
                while mt < n_mt:
                    bi = (mt // (sb // RN)) * RB
                    kj = (mt % (sb // RN)) * RN  # W in-index (grad_h output column)
                    for r in cutlass.range_constexpr(RB):
                        for s in cutlass.range_constexpr(RN):
                            racc[r, s] = cutlass.Float32(0.0)
                    for n in cutlass.range(sb):  # sum over the W out-index
                        for r in cutlass.range_constexpr(RB):
                            gz_rn = gzbuf[bi + r, ob + n]
                            for s in cutlass.range_constexpr(RN):
                                racc[r, s] += gz_rn * s_w[(kj + s) * sb + n]
                    for r in cutlass.range_constexpr(RB):
                        for s in cutlass.range_constexpr(RN):
                            ghbuf[bi + r, ob + kj + s] = racc[r, s]
                    mt += T
                cute.arch.sync_threads()

        @cute.jit
        def _gated_fwd(self, zbuf, abuf, s_w, mGW, mExpand, lyr, focus, tidx):
            """Recompute ``abuf = GatedActivation(zbuf)`` (silu l=0, gate l>0)."""
            CF = cutlass.const_expr(self.cf)
            GO = cutlass.const_expr(self.gate_out)
            Dm = cutlass.const_expr(self.Dm)
            B = cutlass.const_expr(self._B)
            T = cutlass.const_expr(self._T)
            SIG_OFF = cutlass.const_expr(self.cf * self.gate_out)
            j = tidx
            while j < CF * GO:
                s_w[(j // GO) * GO + (j % GO)] = mGW[lyr, focus, j // GO, j % GO]
                j += T
            cute.arch.sync_threads()
            j = tidx
            while j < B * GO:
                b = j // GO
                o = j % GO
                acc = cutlass.Float32(0.0)
                for ii in cutlass.range_constexpr(CF):
                    acc += zbuf[b, ii] * s_w[ii * GO + o]
                s_w[SIG_OFF + b * GO + o] = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cmath.exp(-acc)
                )
                j += T
            cute.arch.sync_threads()
            j = tidx
            while j < B * CF:
                b = j // CF
                c = j % CF
                z = zbuf[b, c]
                abuf[b, c] = z / (cutlass.Float32(1.0) + cmath.exp(-z))
                j += T
            j = tidx
            while j < B * (Dm - 1) * CF:
                b = j // ((Dm - 1) * CF)
                rem = j % ((Dm - 1) * CF)
                d1 = rem // CF
                c = rem % CF
                lidx = mExpand[d1]
                abuf[b, (d1 + 1) * CF + c] = (
                    zbuf[b, (d1 + 1) * CF + c] * s_w[SIG_OFF + b * GO + lidx * CF + c]
                )
                j += T
            cute.arch.sync_threads()

        @cute.jit
        def _gated_bwd(self, zbuf, gabuf, s_w, mGW, mExpand, lyr, focus, tidx):
            """Backprop the gated activation in place: ``zbuf`` (z_l) -> grad_z_l.

            With ``g_a = gabuf`` the incoming gradient and the recomputed gate
            sigmoids ``sig``::

                grad_z[dm, c] = g_a[dm, c] * sig[expand[dm-1], c]              (dm > 0)
                g_sig[L, c]   = sum_{dm: expand[dm-1]=L} g_a[dm, c] * z[dm, c]
                grad_z[0, i]  = g_a[0, i] * silu'(z[0, i])
                                + sum_{o'} Wg[i, o'] * g_sig * sig*(1-sig)
            """
            CF = cutlass.const_expr(self.cf)
            GO = cutlass.const_expr(self.gate_out)
            Dm = cutlass.const_expr(self.Dm)
            LMAX = cutlass.const_expr(self.lmax)
            NG = cutlass.const_expr(self.ngroup)
            B = cutlass.const_expr(self._B)
            T = cutlass.const_expr(self._T)
            SIG_OFF = cutlass.const_expr(self.cf * self.gate_out)
            GGL_OFF = cutlass.const_expr(self.cf * self.gate_out + B * self.gate_out)
            j = tidx
            while j < CF * GO:
                s_w[(j // GO) * GO + (j % GO)] = mGW[lyr, focus, j // GO, j % GO]
                j += T
            cute.arch.sync_threads()
            j = tidx
            while j < B * GO:
                b = j // GO
                o = j % GO
                acc = cutlass.Float32(0.0)
                for ii in cutlass.range_constexpr(CF):
                    acc += zbuf[b, ii] * s_w[ii * GO + o]
                s_w[SIG_OFF + b * GO + o] = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cmath.exp(-acc)
                )
                j += T
            cute.arch.sync_threads()
            # g_gl[L, c] = (sum over the (1 + 2*mmax) |m| groups) * sigmoid'(gate).
            # For degree l = L + 1 the contributing coefficients are dm = 1 + L + k*lmax.
            j = tidx
            while j < B * GO:
                b = j // GO
                o = j % GO
                gate_l = o // CF
                c = o % CF
                gsig = cutlass.Float32(0.0)
                for kk in cutlass.range_constexpr(NG):
                    dm = 1 + gate_l + kk * LMAX
                    gsig += gabuf[b, dm * CF + c] * zbuf[b, dm * CF + c]
                s = s_w[SIG_OFF + b * GO + o]
                s_w[GGL_OFF + b * GO + o] = gsig * s * (cutlass.Float32(1.0) - s)
                j += T
            cute.arch.sync_threads()
            # grad_z[dm>0]: reads sig, writes disjoint from the l=0 slice.
            j = tidx
            while j < B * (Dm - 1) * CF:
                b = j // ((Dm - 1) * CF)
                rem = j % ((Dm - 1) * CF)
                d1 = rem // CF
                c = rem % CF
                lidx = mExpand[d1]
                zbuf[b, (d1 + 1) * CF + c] = (
                    gabuf[b, (d1 + 1) * CF + c] * s_w[SIG_OFF + b * GO + lidx * CF + c]
                )
                j += T
            # grad_z[0]: reads z[0, i] before overwriting it.
            j = tidx
            while j < B * CF:
                b = j // CF
                ii = j % CF
                z0 = zbuf[b, ii]
                sg = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cmath.exp(-z0))
                silup = sg + z0 * sg * (cutlass.Float32(1.0) - sg)
                acc = gabuf[b, ii] * silup
                for o in cutlass.range_constexpr(GO):
                    acc += s_w[ii * GO + o] * s_w[GGL_OFF + b * GO + o]
                zbuf[b, ii] = acc
                j += T
            cute.arch.sync_threads()

    class BackwardRunner:
        """Compile-once driver for :class:`BackwardProgram` (force-path gradients).

        Parameters
        ----------
        weights
            Packed weights (see :class:`.forward.ForwardRunner`).
        lmax, mmax, cf, n_focus, n_layers, bucket, threads, rb, rn
            Kernel configuration.
        """

        def __init__(
            self,
            weights,
            *,
            lmax: int,
            mmax: int,
            cf: int,
            n_focus: int,
            n_layers: int,
            bucket: int,
            threads: int,
            rb: int,
            rn: int,
        ) -> None:
            self.op = BackwardProgram(
                lmax=lmax,
                mmax=mmax,
                cf=cf,
                n_focus=n_focus,
                n_layers=n_layers,
                bucket=bucket,
                threads=threads,
                rb=rb,
                rn=rn,
            )
            self._B = bucket
            self.nf, self.cf, self.Dm, self.D = n_focus, cf, self.op.Dm, self.op.D
            self._compiled = None
            self._stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            fr = ForwardRunner(
                weights,
                lmax=lmax,
                mmax=mmax,
                cf=cf,
                n_focus=n_focus,
                n_layers=n_layers,
                bucket=bucket,
                threads=threads,
                rb=rb,
                rn=rn,
            )
            self.m_w, self.m_gw, self.m_cb, self.m_expand = (
                fr.m_w,
                fr.m_gw,
                fr.m_cb,
                fr.m_expand,
            )

        @staticmethod
        def _dyn(t: torch.Tensor, leading: int):
            return from_dlpack(t, assumed_align=16).mark_layout_dynamic(
                leading_dim=leading
            )

        def __call__(
            self,
            x: torch.Tensor,
            src: torch.Tensor,
            d_to_m: torch.Tensor,
            kc: torch.Tensor,
            g_out: torch.Tensor,
            g_fgate: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Emit the value-path gradients.

            Parameters
            ----------
            x, src, d_to_m, kc
                Forward inputs (see :meth:`.forward.ForwardRunner.__call__`).
            g_out : torch.Tensor
                Gradient w.r.t. ``x_local`` with shape (E, F, D_m, Cf).
            g_fgate : torch.Tensor
                Gradient w.r.t. ``focus_gate`` with shape (E, F, Cf).

            Returns
            -------
            grad_x : torch.Tensor
                Node-feature gradient with shape (N, D, C_wide).
            grad_d_to_m : torch.Tensor
                Wigner-rotation gradient with shape (E, D_m, D).
            grad_kc : torch.Tensor
                Radial degree-kernel gradient with shape (E, D_m, D_m).
            """
            n_edge = src.shape[0]
            n_bucket = (n_edge + self._B - 1) // self._B
            n_pad = n_bucket * self._B
            s32 = src.to(torch.int32)
            g_out_p, g_fg_p = g_out, g_fgate
            if n_pad > n_edge:
                s32 = torch.cat([s32, s32.new_zeros(n_pad - n_edge)])
                g_out_p = torch.cat(
                    [g_out, g_out.new_zeros(n_pad - n_edge, *g_out.shape[1:])]
                )
                g_fg_p = torch.cat(
                    [g_fgate, g_fgate.new_zeros(n_pad - n_edge, *g_fgate.shape[1:])]
                )
            grad_x = torch.zeros_like(x)
            grad_d = torch.zeros(
                n_pad, self.Dm, self.D, device=x.device, dtype=torch.float32
            )
            grad_kc = torch.zeros(
                n_pad, self.Dm, self.Dm, device=x.device, dtype=torch.float32
            )
            views = (
                self._dyn(g_out_p, 3),
                self._dyn(g_fg_p, 2),
                self._dyn(x, 2),
                self._dyn(s32, 0),
                self._dyn(d_to_m, 2),
                self._dyn(kc, 2),
                from_dlpack(self.m_cb, assumed_align=16),
                from_dlpack(self.m_w, assumed_align=16),
                from_dlpack(self.m_gw, assumed_align=16),
                from_dlpack(self.m_expand, assumed_align=16),
                self._dyn(grad_x, 2),
                self._dyn(grad_d, 2),
                self._dyn(grad_kc, 2),
            )
            args = (*views, cutlass.Int32(n_edge), cutlass.Int32(n_bucket))
            if self._compiled is None:
                self._compiled = cute.compile(self.op, *args, stream=self._stream)
            self._compiled(*args, stream=self._stream)
            return grad_x, grad_d[:n_edge], grad_kc[:n_edge]
