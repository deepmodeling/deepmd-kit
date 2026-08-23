// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Shared host entries of the SeZM training kernels.
//
// The fused SO(2) value-path operator composes these traversals inside its
// own backward and second order; they live in one named namespace so the
// composition is plain C++ calls rather than dispatcher round-trips.

#pragma once

#include <ATen/ATen.h>

#include <tuple>

namespace dpa4_sezm {

// Whole-stack gated-mixing forward: (x_local, z_all, u_final).
std::tuple<at::Tensor, at::Tensor, at::Tensor> mixing_fwd(
    const at::Tensor& u0,
    const at::Tensor& alpha,
    const at::Tensor& w0_all,
    const at::Tensor& w1_all,
    const at::Tensor& gw_all,
    int64_t lmax,
    int64_t focus_dim,
    bool apply_alpha);

// Whole-stack gated-mixing first-order backward; ``with_weights`` selects
// the weight-gradient contractions and ``keep_state`` retains the per-layer
// surfaces the second order linearizes around.
std::tuple<at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor>
mixing_bwd(const at::Tensor& grad_out,
           const at::Tensor& x_local,
           const at::Tensor& z_all,
           const at::Tensor& u_final,
           const at::Tensor& alpha,
           const at::Tensor& w0t_all,
           const at::Tensor& w1t_all,
           const at::Tensor& gw_all,
           const at::Tensor& gwt_all,
           const c10::optional<at::Tensor>& u0,
           const c10::optional<at::Tensor>& grad_z_up,
           const c10::optional<at::Tensor>& grad_u_up,
           int64_t lmax,
           int64_t focus_dim,
           bool apply_alpha,
           bool with_weights,
           bool keep_state);

// Second order of the mixing traversal for the force-loss regime. When the
// first-order backward retained its per-layer surfaces they arrive through
// the ``kept_*`` slots and no replay runs; otherwise the traversal is
// replayed internally and its input gradient rides out as the trailing
// output, so a caller that needs both differentiations pays for one
// traversal either way.
std::tuple<at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor>
mixing_bwd2(const at::Tensor& grad_out,
            const at::Tensor& x_local,
            const at::Tensor& z_all,
            const at::Tensor& u_final,
            const at::Tensor& alpha,
            const at::Tensor& w0t_all,
            const at::Tensor& w1t_all,
            const at::Tensor& gw_all,
            const at::Tensor& gwt_all,
            const c10::optional<at::Tensor>& u0,
            const at::Tensor& h_u0,
            const c10::optional<at::Tensor>& h_alpha,
            const c10::optional<at::Tensor>& grad_z_up,
            const c10::optional<at::Tensor>& grad_u_up,
            const c10::optional<at::Tensor>& kept_upstream,
            const c10::optional<at::Tensor>& kept_grad_z,
            const c10::optional<at::Tensor>& kept_grad_logit,
            const c10::optional<at::Tensor>& ggout_init,
            int64_t lmax,
            int64_t focus_dim,
            bool apply_alpha);

// Fused gather + block-diagonal Wigner rotation + radial degree mixing,
// focus-major output (F, E, ROW).
at::Tensor rotate_mix_fwd(const at::Tensor& x,
                          const at::Tensor& src,
                          const at::Tensor& wigner,
                          const at::Tensor& kc,
                          const at::Tensor& cb,
                          int64_t lmax,
                          int64_t n_focus,
                          int64_t rank);

// Paired forward for the second order: one traversal produces the rotated
// input u0 and the upstream cotangent of the rotation backward,
// h_gu0 = M(kc) R(wig) h_e + M(kc) R(h_gwig) x + M(h_gkc) R(wig) x, with
// the node cotangent h_gx gathered onto edges in place.
std::tuple<at::Tensor, at::Tensor> rotate_mix_fwd_pair(
    const at::Tensor& x,
    const at::Tensor& h_gx,
    const at::Tensor& src,
    const at::Tensor& wigner,
    const c10::optional<at::Tensor>& h_gwig,
    const at::Tensor& kc,
    const c10::optional<at::Tensor>& h_gkc,
    const at::Tensor& cb,
    int64_t lmax,
    int64_t n_focus,
    int64_t rank);

// First-order backward of the fused front end: per-edge node gradient (the
// caller segment-sums it), Wigner gradient on the structural non-zeros, the
// degree-kernel gradient, and the channel-basis gradient (zero-shaped for
// the basis-free rank-0 form).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> rotate_mix_bwd(
    const at::Tensor& grad_u,
    const at::Tensor& x,
    const at::Tensor& src,
    const at::Tensor& wigner,
    const at::Tensor& kc,
    const at::Tensor& cb,
    int64_t lmax,
    int64_t n_focus,
    int64_t rank);

// Rotation curvature for the second order: the three multilinear re-entries
// of the rotation backward against the shared upstream, merged into one
// traversal. Returns the per-edge node curvature (zero-shaped when neither
// the Wigner nor the kernel cotangent is present), the Wigner curvature,
// the kernel curvature, and the channel-basis curvature.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> rotate_mix_bwd2(
    const at::Tensor& grad_u,
    const at::Tensor& x,
    const at::Tensor& h_gx,
    const at::Tensor& src,
    const at::Tensor& wigner,
    const c10::optional<at::Tensor>& h_gwig,
    const at::Tensor& kc,
    const c10::optional<at::Tensor>& h_gkc,
    const at::Tensor& cb,
    int64_t lmax,
    int64_t n_focus,
    int64_t rank);

// Contention-free CSR segment sum over the leading axis.
at::Tensor segment_sum_csr(const at::Tensor& rows,
                           const at::Tensor& order,
                           const at::Tensor& row_ptr);

}  // namespace dpa4_sezm
