# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN201, ANN204, ANN205
"""
CuTe-DSL fused forward kernel for the SeZM SO(2) value path.

One bucketed kernel folds the entire per-edge value path of ``SO2Convolution``
into a single launch::

    gather x[src] -> rotate_to_local (D_to_m)                 [prologue, in smem]
      -> radial degree mix (Kc, channel_basis)                [prologue, in smem]
      -> 3x (block-diagonal SO2Linear + GatedActivation + residual)   [in smem]
      -> x_local (E, F, D_m, Cf)                              [+ pre-mixing l=0 scalar]

The block-diagonal ``SO2Linear`` weight is staged in shared memory once per
bucket and reused across the bucket's ``B`` edges (register-blocked ``RB x RN``
FMA micro-tile), so the ``E x D_m x C`` intermediates of all three mixing layers
stay resident on chip and never reach DRAM. The focus competition (a per-edge
softmax of the pre-mixing ``l = 0`` feature) is applied outside the kernel from
the returned ``focus_gate`` scalar.

Grid ``(n_bucket, n_focus)``; one CTA owns ``B`` edges of one focus stream.
All accumulation is fp32 IEEE (no TF32) to keep the potential-energy surface
smooth.
"""

from __future__ import (
    annotations,
)

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.math as cmath
    from cutlass.cute.runtime import (
        from_dlpack,
    )

    SEZM_CUTE_AVAILABLE = True
except Exception:  # pragma: no cover - import guard for non-CuTe environments
    SEZM_CUTE_AVAILABLE = False


if SEZM_CUTE_AVAILABLE:

    class ForwardProgram:
        """Bucketed fused forward program for the SO(2) value path.

        Parameters
        ----------
        lmax : int
            Maximum spherical harmonic degree.
        mmax : int
            Maximum SO(2) order retained in the reduced layout.
        cf : int
            Per-focus channel width ``Cf``.
        n_focus : int
            Number of focus streams ``F``.
        n_layers : int
            Number of SO(2) mixing layers.
        bucket : int
            Edges processed per CTA ``B``.
        threads : int
            Threads per CTA.
        rb, rn : int
            Register micro-tile dimensions (``RB`` bucket rows, ``RN`` output
            columns per thread) of the block-diagonal GEMM.
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
            self.FLAT = self.Dm * cf
            # Block-diagonal |m| block widths in the flattened coeff*channel axis:
            # m = 0 spans (lmax + 1) coefficients, each |m| > 0 spans 2*(lmax-m+1).
            groups = [lmax + 1] + [2 * (lmax - m + 1) for m in range(1, mmax + 1)]
            self._blocks: list[tuple[int, int]] = []
            off = 0
            for g in groups:
                self._blocks.append((off, g * cf))
                off += g * cf
            self._max_sb = max(sb for _, sb in self._blocks)
            # Shared scratch reused for the resident weight block and, during the
            # gated activation, the gate weight plus per-edge sigmoid buffer.
            self._scr = max(
                self._max_sb * self._max_sb, cf * self.gate_out + bucket * self.gate_out
            )
            assert self._B % rb == 0
            for _, sb in self._blocks:
                assert sb % rn == 0

        @cute.jit
        def __call__(
            self,
            mX,
            mSrc,
            mDtoM,
            mKc,
            mCB,
            mW,
            mGW,
            mExpand,
            mOut,
            mFocusGate,
            n_edge: cutlass.Int32,
            n_bucket: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mX, mSrc, mDtoM, mKc, mCB, mW, mGW, mExpand, mOut, mFocusGate, n_edge
            ).launch(grid=[n_bucket, self.nf, 1], block=[self._T, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            mX,
            mSrc,
            mDtoM,
            mKc,
            mCB,
            mW,
            mGW,
            mExpand,
            mOut,
            mFocusGate,
            n_edge: cutlass.Int32,
        ):
            D = cutlass.const_expr(self.D)
            Dm = cutlass.const_expr(self.Dm)
            CF = cutlass.const_expr(self.cf)
            FLAT = cutlass.const_expr(self.FLAT)
            GO = cutlass.const_expr(self.gate_out)
            B = cutlass.const_expr(self._B)
            T = cutlass.const_expr(self._T)
            RB = cutlass.const_expr(self._RB)
            RN = cutlass.const_expr(self._RN)
            SCR = cutlass.const_expr(self._scr)
            GATE_OFF = cutlass.const_expr(self.cf * self.gate_out)
            NGATE = cutlass.const_expr(self.nl - 1)

            tidx, _, _ = cute.arch.thread_idx()
            bucket, focus, _ = cute.arch.block_idx()
            e0 = bucket * B

            smem = cutlass.utils.SmemAllocator()
            s_cur = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((B, FLAT), stride=(FLAT, 1)), 16
            )
            s_tmp = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((B, FLAT), stride=(FLAT, 1)), 16
            )
            s_scr = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((SCR,), stride=(1,)), 16
            )

            # === Step 1. rotate_to_local: s_tmp[b, dm*Cf+c] = sum_k D_to_m x[src] ===
            # Padding edges (e >= n_edge) clamp their read index; their output rows
            # are sliced off by the caller.
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
                s_tmp[b, rem] = acc
                i += T
            cute.arch.sync_threads()

            # === Step 2. radial degree mix: s_cur = channel_basis * (Kc @ x_rot) ===
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
                    acc += mKc[eq, o, ii] * s_tmp[b, ii * CF + c]
                s_cur[b, rem] = acc * mCB[focus * CF + c]
                i += T
            cute.arch.sync_threads()

            # === Step 3. Emit the pre-mixing l=0 scalar for the focus competition ===
            i = tidx
            while i < B * CF:
                b = i // CF
                c = i % CF
                mFocusGate[e0 + b, focus, c] = s_cur[b, c]
                i += T
            cute.arch.sync_threads()

            # === Step 4. Multi-layer gated SO(2) mixing (block-diagonal, residual) ===
            for lyr in cutlass.range_constexpr(self.nl):
                # --- SO2Linear: s_tmp = s_cur @ W[lyr, focus] (per |m| block) ---
                for ob, sb in cutlass.const_expr(self._blocks):
                    j = tidx
                    while j < sb * sb:
                        k = j // sb
                        n = j % sb
                        s_scr[k * sb + n] = mW[lyr, focus, ob + k, ob + n]
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
                                a_rk = s_cur[bi + r, ob + k]
                                for s in cutlass.range_constexpr(RN):
                                    racc[r, s] += a_rk * s_scr[k * sb + nj + s]
                        for r in cutlass.range_constexpr(RB):
                            for s in cutlass.range_constexpr(RN):
                                s_tmp[bi + r, ob + nj + s] = racc[r, s]
                        mt += T
                    cute.arch.sync_threads()

                # --- GatedActivation (gated layers) or identity (last layer) ---
                if lyr < NGATE:
                    # gate FocusLinear weight -> s_scr[0 : Cf*GO]
                    j = tidx
                    while j < CF * GO:
                        ii = j // GO
                        o = j % GO
                        s_scr[ii * GO + o] = mGW[lyr, focus, ii, o]
                        j += T
                    cute.arch.sync_threads()
                    # gate sigmoids from the l=0 scalar -> s_scr[GATE_OFF + b*GO + o]
                    j = tidx
                    while j < B * GO:
                        b = j // GO
                        o = j % GO
                        acc = cutlass.Float32(0.0)
                        for ii in cutlass.range_constexpr(CF):
                            acc += s_tmp[b, ii] * s_scr[ii * GO + o]
                        s_scr[GATE_OFF + b * GO + o] = cutlass.Float32(1.0) / (
                            cutlass.Float32(1.0) + cmath.exp(-acc)
                        )
                        j += T
                    cute.arch.sync_threads()
                    # l=0: silu(z) = z / (1 + exp(-z))
                    j = tidx
                    while j < B * CF:
                        b = j // CF
                        c = j % CF
                        z = s_tmp[b, c]
                        s_tmp[b, c] = z / (cutlass.Float32(1.0) + cmath.exp(-z))
                        j += T
                    # l>0: z * sigmoid(gate[expand[dm-1]])
                    j = tidx
                    while j < B * (Dm - 1) * CF:
                        b = j // ((Dm - 1) * CF)
                        rem = j % ((Dm - 1) * CF)
                        d1 = rem // CF
                        c = rem % CF
                        lidx = mExpand[d1]
                        z = s_tmp[b, (d1 + 1) * CF + c]
                        s_tmp[b, (d1 + 1) * CF + c] = (
                            z * s_scr[GATE_OFF + b * GO + lidx * CF + c]
                        )
                        j += T
                    cute.arch.sync_threads()

                # --- residual add: s_cur += activation(s_tmp) ---
                i = tidx
                while i < B * FLAT:
                    b = i // FLAT
                    rem = i % FLAT
                    s_cur[b, rem] = s_cur[b, rem] + s_tmp[b, rem]
                    i += T
                cute.arch.sync_threads()

            # === Step 5. Write the pre-focus-compete local features (E, F, D_m, Cf) ===
            i = tidx
            while i < B * FLAT:
                b = i // FLAT
                rem = i % FLAT
                o = rem // CF
                c = rem % CF
                mOut[e0 + b, focus, o, c] = s_cur[b, rem]
                i += T

    class ForwardRunner:
        """Compile-once driver for :class:`ForwardProgram`.

        Prepares the static packed weights on construction and dispatches the
        compiled kernel over dynamic edge counts.

        Parameters
        ----------
        weights
            Packed weights exposing ``so2_w`` (L, F, D_m*Cf, D_m*Cf),
            ``gate_w`` (L, Cf, F, lmax*Cf), ``has_gate`` (L,), and
            ``channel_basis`` (C_wide,).
        lmax, mmax, cf, n_focus, n_layers, bucket, threads, rb, rn
            Kernel configuration (see :class:`ForwardProgram`).
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
            self.op = ForwardProgram(
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
            self.nf, self.cf, self.Dm = n_focus, cf, self.op.Dm
            self._compiled = None
            self._stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            self._pack(weights, lmax, cf, n_focus, n_layers)

        def _pack(self, w, lmax: int, cf: int, nf: int, nl: int) -> None:
            dev = w.so2_w.device
            self.m_w = w.so2_w.detach().contiguous()
            gate = torch.zeros(nl, nf, cf, lmax * cf, device=dev, dtype=torch.float32)
            for layer in range(nl):
                if bool(w.has_gate[layer]):
                    gate[layer] = w.gate_w[layer].detach().permute(1, 0, 2).contiguous()
            self.m_gw = gate.contiguous()
            self.m_cb = w.channel_basis.detach().contiguous()
            # m-major degree index l(dm); the gate expand maps dm>0 -> (l-1).
            l_index = list(range(lmax + 1))
            for m in range(1, self.op.mmax + 1):
                l_index += list(range(m, lmax + 1)) * 2
            self.m_expand = torch.tensor(
                [li - 1 for li in l_index[1:]], device=dev, dtype=torch.int32
            ).contiguous()

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
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Run the fused forward.

            Parameters
            ----------
            x : torch.Tensor
                Node features with shape (N, D, C_wide).
            src : torch.Tensor
                Per-edge source-node indices with shape (E,).
            d_to_m : torch.Tensor
                Row-projected Wigner-D with shape (E, D_m, D).
            kc : torch.Tensor
                Radial degree kernel with shape (E, D_m, D_m).

            Returns
            -------
            x_local : torch.Tensor
                Pre-focus-compete local features with shape (E, F, D_m, Cf).
            focus_gate : torch.Tensor
                Pre-mixing l=0 scalar with shape (E, F, Cf).
            """
            n_edge = src.shape[0]
            n_bucket = (n_edge + self._B - 1) // self._B
            n_pad = n_bucket * self._B
            s32 = src.to(torch.int32)
            if n_pad > n_edge:
                s32 = torch.cat([s32, s32.new_zeros(n_pad - n_edge)])
            out = x.new_empty(n_pad, self.nf, self.Dm, self.cf)
            fgate = x.new_empty(n_pad, self.nf, self.cf)
            views = (
                self._dyn(x, 2),
                self._dyn(s32, 0),
                self._dyn(d_to_m, 2),
                self._dyn(kc, 2),
                from_dlpack(self.m_cb, assumed_align=16),
                from_dlpack(self.m_w, assumed_align=16),
                from_dlpack(self.m_gw, assumed_align=16),
                from_dlpack(self.m_expand, assumed_align=16),
                self._dyn(out, 3),
                self._dyn(fgate, 2),
            )
            args = (*views, cutlass.Int32(n_edge), cutlass.Int32(n_bucket))
            if self._compiled is None:
                self._compiled = cute.compile(self.op, *args, stream=self._stream)
            self._compiled(*args, stream=self._stream)
            return out[:n_edge], fgate[:n_edge]
