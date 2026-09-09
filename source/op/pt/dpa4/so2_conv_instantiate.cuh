// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Instantiation body of the fused DPA4 / SeZM SO(2) convolution for one
// ``(direction, lmax, focus width)`` triple. Included once per translation unit
// with ``DPA4_CONV_L``, ``DPA4_CONV_CF`` and ``DPA4_CONV_BACKWARD`` defined.
//
// The two directions are separate units because the build wall time is set by
// the largest one, and the backward carries roughly twice the unrolled body.

#include "so2_conv_kernel.cuh"
#include "so2_conv_launch.h"

#ifndef DPA4_CONV_L
#error "DPA4_CONV_L must name the spherical-harmonic degree of this unit"
#endif
#ifndef DPA4_CONV_CF
#error "DPA4_CONV_CF must name the focus width of this unit"
#endif
#ifndef DPA4_CONV_BACKWARD
#error "DPA4_CONV_BACKWARD must select the direction of this unit"
#endif

namespace dpa4 {

/// Shared memory per multiprocessor of the running device, cached per device.
///
/// The launch policy reads the actual device rather than the architecture
/// name: variants of one architecture ship different shared-memory sizes and
/// the same binary serves all of them.
static int smem_per_multiprocessor() {
  constexpr int kMaxDevices = 64;
  static int cache[kMaxDevices] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= kMaxDevices) {
    return 0;
  }
  if (cache[dev] == 0) {
    int value = 0;
    cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                           dev);
    cache[dev] = value;
  }
  return cache[dev];
}

/// Opt a kernel into the shared-memory carveout above the 48 KB default.
///
/// The preferred carveout is the smallest fraction of the multiprocessor's
/// shared memory that seats the tile's resident-block target (never below
/// two); the remainder stays available as L1. Asking for the maximum
/// unconditionally wastes L1 on parts with large shared memory.
template <typename Kernel>
static cudaError_t enable_dynamic_smem(Kernel kernel, int bytes, int blocks) {
  const cudaError_t status =
      cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (status != cudaSuccess) {
    return status;
  }
  const long per_sm = smem_per_multiprocessor();
  if (per_sm > 0) {
    const long want = static_cast<long>(blocks < 2 ? 2 : blocks) * bytes;
    const int pct = static_cast<int>((want * 100 + per_sm - 1) / per_sm);
    cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         pct > 100 ? 100 : pct);
  }
  return cudaSuccess;
}

/// Fail with the shape, the requested size and the CUDA error of one launch
/// step. The check lives at the launch rather than at the operator's tail so a
/// failure names its shape; a trailing ``cudaGetLastError`` also catches errors
/// drifting in from earlier unchecked launches and misattributes them.
void report_launch_failure(int lmax, int focus_dim, int bytes, int error);

template <int L, int CF>
void conv_forward_launch(const ConvArgs& args,
                         int n_node,
                         float* out,
                         float* pre_gate,
                         cudaStream_t stream) {
  using T = typename ConvLaunch<L, CF>::Tile;
  const int smem = ConvSmem<L, CF, T>::bytes(args.kc_len, args.n_head, false);
  cudaError_t status =
      enable_dynamic_smem(so2_conv_fwd_kernel<L, CF, T>, smem, T::OCC);
  if (status == cudaSuccess) {
    so2_conv_fwd_kernel<L, CF, T>
        <<<dim3(static_cast<unsigned>(n_node)), dim3(T::NT), smem, stream>>>(
            args, out, pre_gate);
    status = cudaGetLastError();
  }
  if (status != cudaSuccess) {
    report_launch_failure(L, CF, smem, static_cast<int>(status));
  }
}

template <int L, int CF>
void conv_backward_launch(const ConvArgs& args,
                          int n_node,
                          const float* g_out,
                          const float* w0t,
                          const float* w1t,
                          const float* gwt,
                          float* g_x,
                          float* g_quat,
                          float* g_kc,
                          float* g_alpha,
                          cudaStream_t stream) {
  using T = typename ConvLaunch<L, CF>::Tile;
  const int smem = ConvSmem<L, CF, T>::bytes(args.kc_len, args.n_head, true);
  cudaError_t status =
      enable_dynamic_smem(so2_conv_bwd_kernel<L, CF, T>, smem, T::OCC);
  if (status == cudaSuccess) {
    so2_conv_bwd_kernel<L, CF, T>
        <<<dim3(static_cast<unsigned>(n_node)), dim3(T::NT), smem, stream>>>(
            args, g_out, w0t, w1t, gwt, g_x, g_quat, g_kc, g_alpha);
    status = cudaGetLastError();
  }
  if (status != cudaSuccess) {
    report_launch_failure(L, CF, smem, static_cast<int>(status));
  }
}

#if DPA4_CONV_BACKWARD
template void conv_backward_launch<DPA4_CONV_L, DPA4_CONV_CF>(const ConvArgs&,
                                                              int,
                                                              const float*,
                                                              const float*,
                                                              const float*,
                                                              const float*,
                                                              float*,
                                                              float*,
                                                              float*,
                                                              float*,
                                                              cudaStream_t);
#else
template void conv_forward_launch<DPA4_CONV_L, DPA4_CONV_CF>(
    const ConvArgs&, int, float*, float*, cudaStream_t);
#endif

}  // namespace dpa4
