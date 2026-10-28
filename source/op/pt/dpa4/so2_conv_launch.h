// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Launch policy and per-shape entry points of the fused DPA4 / SeZM SO(2)
// convolution.
//
// The kernels are templated on the spherical-harmonic degree and the focus
// width because both size register-resident state. Every other configuration
// dimension -- layer count, radial-mixer rank, focus-stream count, attention
// head count -- is a runtime argument, which keeps the instantiation count at
// ``lmax x focus_width`` rather than the product of the whole design family.
//
// Each ``(lmax, focus width)`` pair is explicitly instantiated in its own
// translation unit (``so2_conv_c{32,64}_l{1..6}.cu``) so the twelve
// instantiations compile in parallel; a single unit per width serializes six
// degrees behind one ``nvcc`` invocation and dominates the build.

#pragma once

#include <cuda_runtime.h>

#include "so2_conv.cuh"

namespace dpa4 {

/// Focus widths with an instantiation.
constexpr int kFocusDim32 = 32;
constexpr int kFocusDim64 = 64;

/// Weight-panel budget in floats. The double-buffered panel is the largest
/// shared-memory allocation, so the budget is what holds several blocks
/// resident.

/// Weight-panel depth: reduction rows staged at once.
///
/// A panel costs two barriers, so a deeper one is better until its staging
/// displaces a resident block. Where that tips is a measured property of the
/// shape: the panel costs ``PK * NMAX`` floats, but what it competes against is
/// the tile's register footprint, which grows with the degree as well. The
/// thresholds below reproduce the measured optimum of every instantiated
/// shape -- 32 for ``nano``, 16 for ``mini``, 8 for ``neo`` and ``air``, 4 for
/// ``plus`` and ``pro`` -- and the staged floats they imply stay between 1536
/// and 3072 throughout.
///
/// The two ends are worth stating because both were measured: halving the
/// depth of ``mini`` costs 8 percent in barriers, and doubling the depth of
/// ``air`` past this point costs 49 percent in residency.
constexpr int panel_depth_of(int nmax) {
  if (nmax <= 64) {
    return 32;
  }
  if (nmax <= 128) {
    return 16;
  }
  if (nmax <= 384) {
    return 8;
  }
  return 4;
}

/// Launch tile of one ``(lmax, focus width)`` instantiation.
///
/// The activation tile costs ``TM * RB`` registers per thread and every weight
/// load feeds ``TM`` products, so ``TM`` trades occupancy against weight
/// traffic. It is capped by degree: the reduced row grows as ``(3 * lmax + 1)``
/// column groups, and holding eight of those rows past degree two would exhaust
/// the register file.
///
/// Hardware dependence. Correctness never depends on these constants; the
/// numbers below are a measured optimum for one part and the launch layer
/// already adapts what an API exposes at run time (the shared-memory carveout
/// follows the device's per-multiprocessor size). What must be retuned per
/// part, and against which resource:
///
/// - ``TM`` thresholds: the 64 K-register file of the multiprocessor.
/// - ``PK`` (weight-panel depth): shared memory per block against barrier
///   count; ``air`` wants twice the depth ``plus`` does on this part.
/// - ``OCC`` (resident-block target): the register file again; forcing it on a
///   tile whose natural footprint is 250+ registers makes the assembler trade
///   scheduling freedom for residency, which pays on some shapes only.
///
/// Measured on an RTX PRO 6000 Blackwell (100 KB shared memory and 64 K
/// registers per multiprocessor, 117 float32 TFLOP/s). The same point was swept
/// again on an H20 (228 KB shared memory, 40 TFLOP/s), whose balance is the
/// opposite: every alternative lost there as well, and the best of them was
/// within half a percent of this one, so one policy serves both parts and the
/// occupancy of the tile, not its shared-memory footprint, is what the
/// convolution is sensitive to.
///
/// Retune by sweeping the constants with ``-DDPA4_TILE_<name>=<value>`` on the
/// build, timing with ``debug/cuda_bench/check_conv.py --skip-check`` and
/// confirming in the model graph with ``debug/cuda_bench/compare_paths.py``;
/// standalone and in-graph optima differ (variable neighbor degrees), so the
/// model graph has the final word.
// Every constant of the policy below can be pinned from the build system with
// ``-DDPA4_TILE_<name>=<value>``, which is what lets a sweep explore the space
// without editing this header. Unpinned, the shape-dependent defaults apply.
#ifndef DPA4_TILE_TM
#define DPA4_TILE_TM 0
#endif
#ifndef DPA4_TILE_WARPS
#define DPA4_TILE_WARPS 0
#endif
#ifndef DPA4_TILE_PK
#define DPA4_TILE_PK 0
#endif
#ifndef DPA4_TILE_OCC
#define DPA4_TILE_OCC 0
#endif

template <int L, int CF>
struct ConvLaunch {
  static constexpr int RB = (3 * L + 1) * (CF / kWarp);
  static constexpr int NMAX = (2 * L > L + 1 ? 2 * L : L + 1) * CF;
  // Four activation rows per warp, so every staged weight feeds four products,
  // as far as the register file carries it: the activation tile costs
  // ``TM * RB`` registers per thread, and past ``RB = 26`` the wide tile
  // spills more than its arithmetic density returns (``pro``, at 32, loses a
  // third to it).
  static constexpr int TM = DPA4_TILE_TM ? DPA4_TILE_TM : ((RB <= 26) ? 4 : 2);
  static constexpr int WARPS = DPA4_TILE_WARPS ? DPA4_TILE_WARPS : 4;
  static constexpr int PK = DPA4_TILE_PK ? DPA4_TILE_PK : panel_depth_of(NMAX);
  // The residency target trades warps to hide latency against registers per
  // thread, and the optimum follows the activation tile. The narrow rows sit
  // at four blocks; the middle keeps three; the wide rows keep the ``TM = 4``
  // tile only by dropping to two, which is worth 13 percent on ``air`` over
  // the narrow tile at three; and the widest row cannot hold that tile at any
  // residency, so it returns to the narrow tile at three.
  static constexpr int OCC =
      DPA4_TILE_OCC ? DPA4_TILE_OCC
                    : ((RB <= 8) ? 4 : ((RB <= 14) ? 3 : ((RB <= 26) ? 2 : 3)));
  using Tile = ConvTile<TM, WARPS, PK, OCC>;
};

/// Whether an instantiation exists for this shape.
bool conv_shape_instantiated(int lmax, int focus_dim);

/// Launch one forward evaluation. Defined by the ``(L, CF)`` translation unit.
template <int L, int CF>
void conv_forward_launch(const ConvArgs& args,
                         int n_node,
                         float* out,
                         float* pre_gate,
                         cudaStream_t stream);

/// Launch one backward evaluation. Defined by the ``(L, CF)`` translation unit.
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
                          cudaStream_t stream);

/// Expand ``macro(L, CF)`` over every instantiated shape.
#define DPA4_CONV_FOR_EACH_SHAPE(macro)                                \
  macro(1, 32) macro(2, 32) macro(3, 32) macro(4, 32) macro(5, 32)     \
      macro(6, 32) macro(1, 64) macro(2, 64) macro(3, 64) macro(4, 64) \
          macro(5, 64) macro(6, 64)

#define DPA4_CONV_DECLARE(LV, CFV)                                    \
  extern template void conv_forward_launch<LV, CFV>(                  \
      const ConvArgs&, int, float*, float*, cudaStream_t);            \
  extern template void conv_backward_launch<LV, CFV>(                 \
      const ConvArgs&, int, const float*, const float*, const float*, \
      const float*, float*, float*, float*, float*, cudaStream_t);

DPA4_CONV_FOR_EACH_SHAPE(DPA4_CONV_DECLARE)

#undef DPA4_CONV_DECLARE

}  // namespace dpa4
