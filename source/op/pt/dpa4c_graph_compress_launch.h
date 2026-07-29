// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Host-side launch interface of the compressed DPA4C descriptor.
//
// The kernel templates are instantiated once per scalar width in a dedicated
// translation unit, which keeps the compile time of the angular-degree and
// topology specializations bounded and parallel. The dispatching translation
// unit only sees the plain argument bundle declared here.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace deepmd_dpa4c {

constexpr int kWarpSize = 32;

/// Element type of the topology indices, folded to two widths because a
/// signed and an unsigned 32-bit index share their representation over the
/// non-negative range that node and edge identifiers occupy.
enum class IndexKind : int { Bits32 = 0, Bits64 = 1 };

/// Complete argument bundle of one descriptor launch.
///
/// Pointers that a given direction does not read stay null. All tensors are
/// contiguous and, except for the topology and the coupling layout, fp32.
struct Arguments {
  long node_count = 0;
  long edge_count = 0;
  int lmax = 2;
  int interval_count = 0;
  int type_count = 0;
  int table_width = 0;
  int radial_modes = 0;
  // First node of the launched run within the system. Node-indexed buffers are
  // already offset by the caller; the atom type table is not, because neighbor
  // lookups index it with absolute source indices.
  long node_begin = 0;
  // Whether this run owns the reserved edge slots past the physical count.
  // They lie beyond every destination row, so only the run that ends at the
  // last node may clear them; an earlier run would erase the gradients that
  // later runs are about to write.
  bool clear_padding = true;
  int coupling_count = 0;
  float table_stride = 0.0f;
  float table_max = 0.0f;
  float rcut = 0.0f;
  float eps = 0.0f;
  float degree_floor = 0.0f;
  bool canonical = false;
  // Whether the native spin branch is active. The neighbour spin width is
  // derived from the degree profile, so presence is the whole runtime choice.
  bool has_spin = false;
  IndexKind index_kind = IndexKind::Bits64;

  const void* edge_index = nullptr;
  const void* destination_order = nullptr;
  const float* edge_vec = nullptr;
  const bool* edge_mask = nullptr;
  const long* destination_row_ptr = nullptr;
  const long* atype = nullptr;
  const float* table = nullptr;
  const float* pair_film = nullptr;
  const float* pair_mixing = nullptr;
  const float* type_embedding = nullptr;
  const float* readout_matrices = nullptr;
  const int* coupling_meta = nullptr;
  const int* coupling_entry = nullptr;
  const float* coupling_value = nullptr;
  const float* output_mean = nullptr;
  const float* output_inv_std = nullptr;
  const float* descriptor_gradient = nullptr;
  const float* state = nullptr;

  // === Native spin ===
  // Present together or not at all. ``spin`` is indexed by absolute node id,
  // like ``atype``, because neighbour lookups address it with source indices.
  // ``spin_type`` packs the four per-type scalars a node needs -- the fused
  // gate over reference magnitude, the bare gate that the magnetic
  // coordination family counts, and the two on-site weights -- so one node
  // reads them in a single 128-bit load.
  const float* spin = nullptr;
  const float* spin_pair = nullptr;
  const float* spin_type = nullptr;

  float* descriptor = nullptr;
  float* state_out = nullptr;
  float* moment_gradient = nullptr;
  float* edge_gradient = nullptr;
  // Per-node on-site magnetic gradient and the per-edge contribution to the
  // source node's magnetic gradient. The latter is reduced onto source nodes
  // by the shared edge force assembly, which already walks the source CSR.
  float* spin_gradient = nullptr;
  float* edge_spin_gradient = nullptr;
};

// === Compile-time descriptor profile ===

constexpr int triangular(int width) { return width * (width + 1) / 2; }

constexpr int degree_one_width(int channels) {
  int exponent = 0;
  for (int value = channels; value > 1; value >>= 1) {
    ++exponent;
  }
  const int width = 1 << ((exponent + 1) / 2);
  return width < 4 ? 4 : width;
}

constexpr int degree_two_width(int channels) {
  const int width = degree_one_width(channels) >> 1;
  return width < 4 ? 4 : width;
}

constexpr int degree_rank(int degree, int rank_one, int rank_two) {
  return degree == 1 ? rank_one : (degree == 2 ? rank_two : 1);
}

// Independent probe contractions emitted by one degree triple. Axes carrying
// equal degrees are symmetric under the symmetrized coupling, so only one
// representative of each orbit is emitted.
constexpr int triple_outputs(int l1, int l2, int l3, int k1, int k2) {
  const int r1 = degree_rank(l1, k1, k2);
  const int r2 = degree_rank(l2, k1, k2);
  const int r3 = degree_rank(l3, k1, k2);
  if (l1 == l3) {
    return r1 * (r1 + 1) * (r1 + 2) / 6;
  }
  if (l1 == l2) {
    return triangular(r1) * r3;
  }
  if (l2 == l3) {
    return r1 * triangular(r2);
  }
  return r1 * r2 * r3;
}

// Outputs emitted by every triple enumerated before the target. Passing a
// triple that cannot occur returns the complete bispectrum width.
constexpr int bispectrum_prefix(int lmax, int k1, int k2, int a, int b, int c) {
  int total = 0;
  for (int l1 = 1; l1 <= lmax; ++l1) {
    for (int l2 = l1; l2 <= lmax; ++l2) {
      for (int l3 = l2; l3 <= lmax; ++l3) {
        if (l3 > l1 + l2 || (l1 + l2 + l3) % 2 != 0) {
          continue;
        }
        if (l1 == a && l2 == b && l3 == c) {
          return total;
        }
        total += triple_outputs(l1, l2, l3, k1, k2);
      }
    }
  }
  return total;
}

// Degree triples that the sparse coupling artifact must describe, that is
// every allowed triple except the two the kernel contracts in closed form.
constexpr int coupling_record_count(int lmax) {
  int total = 0;
  for (int l1 = 1; l1 <= lmax; ++l1) {
    for (int l2 = l1; l2 <= lmax; ++l2) {
      for (int l3 = l2; l3 <= lmax; ++l3) {
        if (l3 > l1 + l2 || (l1 + l2 + l3) % 2 != 0) {
          continue;
        }
        if ((l1 == 1 && l2 == 1 && l3 == 2) ||
            (l1 == 2 && l2 == 2 && l3 == 2)) {
          continue;
        }
        ++total;
      }
    }
  }
  return total;
}

// Number of lanes that cooperate on one edge. A warp therefore keeps
// ``32 / width`` edges in flight, and the per-edge geometry, which every lane
// of a group recomputes, is amortized over that many edges. Narrowing the
// group lowers that fixed cost but raises the per-lane channel and moment
// footprint, because an angular channel beyond the group width has to be
// tiled and consumes another accumulator. Each profile therefore sits at the
// measured optimum of that trade-off on a diamond neighborhood; forward and
// backward differ because the backward carries the additional angular
// cotangents.
//
// The native spin branch carries its own pair of widths, because its channel
// count is tied to the degree-two width rather than to the scalar width and
// its five families add accumulators that the geometric optimum does not
// account for. Both were measured on the same diamond neighborhood as the
// geometric pair, independently of each other, since one governs the forward
// kernel and the other the backward kernel.
//
// The measured spin optima differ from the geometric ones in exactly two
// places. At the narrowest profile the spin forward wants one step wider,
// because the eight-fold spin payload of a two-lane group tiles into more
// accumulators than the recovered edge concurrency is worth. At ``Channels ==
// 64`` the spin backward wants one step narrower, because the bond-projected
// family adds an angular cotangent whose per-lane cost outweighs the geometry
// recompute that a wider group amortizes. A group wider than the scalar width
// leaves a tile empty and does not compile, which bounds the narrow profiles
// from above.
template <int Channels>
struct EdgeMap;

template <>
struct EdgeMap<8> {
  static constexpr int Forward = 2;
  static constexpr int Backward = 2;
  static constexpr int SpinForward = 4;
  static constexpr int SpinBackward = 2;
};
template <>
struct EdgeMap<16> {
  static constexpr int Forward = 4;
  static constexpr int Backward = 4;
  static constexpr int SpinForward = 4;
  static constexpr int SpinBackward = 4;
};
template <>
struct EdgeMap<32> {
  static constexpr int Forward = 8;
  static constexpr int Backward = 4;
  static constexpr int SpinForward = 8;
  static constexpr int SpinBackward = 4;
};
template <>
struct EdgeMap<64> {
  static constexpr int Forward = 8;
  static constexpr int Backward = 8;
  static constexpr int SpinForward = 8;
  static constexpr int SpinBackward = 4;
};
template <>
struct EdgeMap<128> {
  static constexpr int Forward = 16;
  static constexpr int Backward = 8;
  static constexpr int SpinForward = 16;
  static constexpr int SpinBackward = 8;
};

// ``HasSpin`` is deliberately without a default. It changes the moment
// layout and the descriptor width, so an instantiation that omits it would
// silently read a spin-free layout out of a spin-conditioned buffer.
template <int Channels, int Lmax, bool HasSpin>
struct Profile {
  static constexpr int C0 = Channels;
  static constexpr int C1 = degree_one_width(Channels);
  static constexpr int C2 = degree_two_width(Channels);
  static constexpr int K1 = C2;
  static constexpr int K2 = 2;

  // === Native spin widths ===
  // The neighbour spin width is derived from the degree-two width, so the
  // presence of the branch is the only new compile-time degree of freedom and
  // the instantiation matrix doubles rather than growing by a factor of four.
  static constexpr int Cs = HasSpin ? C2 : 0;
  // Reduced families: neighbour moment magnitude, magnetic coordination, the
  // isotropic neighbour spin vector, the bond-projected neighbour spin vector,
  // and the single neighbour spin quadrupole.
  static constexpr int SpinEdgeWidth = HasSpin ? 8 * Cs + 5 : 0;
  // Node-local on-site vector and quadrupole, written outside the division.
  static constexpr int SpinNodeWidth = HasSpin ? 8 : 0;
  static constexpr int SpinMomentWidth = SpinEdgeWidth + SpinNodeWidth;
  // Channel width of the joint degree-one spin block: the on-site moment, the
  // isotropic neighbour channels and the bond-projected neighbour channels.
  // The two neighbour families share one grading, so one Gram over the joint
  // block emits every admissible degree-one spin invariant.
  static constexpr int SpinVectorWidth = HasSpin ? 1 + 2 * Cs : 0;

  // Flat moment layout: degree zero, degree one, degree two, the
  // single-channel high degrees in increasing order, then the spin families.
  static constexpr int ScalarOffset = 0;
  static constexpr int VectorOffset = C0;
  static constexpr int TensorOffset = C0 + 3 * C1;
  static constexpr int HighOffset = TensorOffset + 5 * C2;
  static constexpr int High3 = Lmax >= 3 ? 7 : 0;
  static constexpr int High4 = Lmax >= 4 ? 9 : 0;
  static constexpr int HighCount = High3 + High4;
  static constexpr int SpinOffset = HighOffset + HighCount;
  static constexpr int SpinMagnitude = SpinOffset;               // Cs
  static constexpr int SpinCoordination = SpinMagnitude + Cs;    // Cs
  static constexpr int SpinVector = SpinCoordination + Cs;       // 3 * Cs
  static constexpr int SpinBond = SpinVector + 3 * Cs;           // 3 * Cs
  static constexpr int SpinTensor = SpinBond + 3 * Cs;           // 5
  static constexpr int SpinOnsiteVector = SpinTensor + 5;        // 3
  static constexpr int SpinOnsiteTensor = SpinOnsiteVector + 3;  // 5
  static constexpr int MomentWidth = SpinOffset + SpinMomentWidth;
  static constexpr int StateWidth = MomentWidth + 2;

  // Cached intermediates of the invariant readout.
  static constexpr int AlignedWidth = 3 * C1 + 5 * C2;
  static constexpr int ProbeWidth = 3 * K1 + 5 * K2;

  // Descriptor layout.
  static constexpr int Gram1 = triangular(C1);
  static constexpr int Gram2 = triangular(C2);
  static constexpr int Bis112 = triangular(K1) * K2;
  static constexpr int Bis222 = 4;
  static constexpr int Quartic = K1 * K2;
  static constexpr int OutputScalar = 0;
  static constexpr int OutputGram1 = C0;
  static constexpr int OutputGram2 = OutputGram1 + Gram1;
  static constexpr int OutputGram3 = OutputGram2 + Gram2;
  static constexpr int OutputGram4 = OutputGram3 + (Lmax >= 3 ? 1 : 0);
  static constexpr int BispectrumBase = OutputGram4 + (Lmax >= 4 ? 1 : 0);
  static constexpr int OutputBis112 =
      BispectrumBase + bispectrum_prefix(Lmax, K1, K2, 1, 1, 2);
  static constexpr int OutputBis222 =
      BispectrumBase + bispectrum_prefix(Lmax, K1, K2, 2, 2, 2);
  static constexpr int OutputQuartic =
      BispectrumBase + bispectrum_prefix(Lmax, K1, K2, 0, 0, 0);
  // Spin invariants of even spin order. The quadrupole Gram omits its
  // on-site self-term, which the identity |B_2(s)|^2 = |s|^4 makes a per-type
  // constant times the square of the vector block's on-site self-term.
  static constexpr int SpinGramVector =
      HasSpin ? triangular(SpinVectorWidth) : 0;
  static constexpr int SpinGramTensor = HasSpin ? 2 : 0;
  static constexpr int SpinCross = HasSpin ? 2 * C2 : 0;
  static constexpr int SpinDim =
      SpinGramVector + SpinGramTensor + SpinCross + 2 * Cs;
  static constexpr int OutputSpinGramVector = OutputQuartic + Quartic;
  static constexpr int OutputSpinGramTensor =
      OutputSpinGramVector + SpinGramVector;
  static constexpr int OutputSpinCross = OutputSpinGramTensor + SpinGramTensor;
  static constexpr int OutputSpinMagnitude = OutputSpinCross + SpinCross;
  static constexpr int OutputSpinCoordination = OutputSpinMagnitude + Cs;
  // The two moment divisors close the geometric block. Normalization is
  // otherwise irreversible, so without them neither the readout nor the
  // fitting network can see the effective coordination they encode.
  static constexpr int OutputDivisor = OutputQuartic + Quartic + SpinDim;
  static constexpr int OutputType = OutputDivisor + 2;
  static constexpr int OutputWidth = OutputType + C0;

  // The group width does not widen for the high angular degrees. Their
  // single-channel components add one accumulator per lane and tile, but every
  // measured widening lost more to the reduced edge concurrency than it
  // recovered in register pressure.
  static constexpr int ForwardEdgeWidth =
      HasSpin ? EdgeMap<Channels>::SpinForward : EdgeMap<Channels>::Forward;
  static constexpr int BackwardEdgeWidth =
      HasSpin ? EdgeMap<Channels>::SpinBackward : EdgeMap<Channels>::Backward;
  static constexpr int NodeWidth = 8;
  static constexpr int NodeGroups = kWarpSize / NodeWidth;
  static constexpr int Threads = kWarpSize;

  // Resident blocks the edge kernels are compiled for. A block is one warp, so
  // thirty-two of them exhaust the 65,536-register file at sixty-four
  // registers per thread, and a lower target raises the per-thread budget in
  // proportion at the cost of occupancy.
  //
  // Degree two holds its geometric working set in that budget and spills only
  // what the five spin families add, so twenty-four blocks buy eighty
  // registers, remove the spill outright and are worth 2.0% to 4.5% of the
  // step. Degree three overflows the budget on geometry alone -- it spills
  // without the spin branch and still spills at eighty and at ninety-six
  // registers -- so the occupancy it would give up buys an incomplete fix and
  // costs 7.4% to 9.6%. The relief is therefore taken only where it is
  // complete.
  static constexpr int MinBlocksPerSm = (HasSpin && Lmax == 2) ? 24 : 32;
};

/// Scalar widths that own a compiled specialization.
#define DPA4C_FOR_EACH_CHANNEL(macro) \
  macro(8) macro(16) macro(32) macro(64) macro(128)

#define DPA4C_DECLARE_CHANNEL(width)                        \
  void launch_forward_c##width(const Arguments& arguments,  \
                               cudaStream_t stream);        \
  void launch_backward_c##width(const Arguments& arguments, \
                                cudaStream_t stream);

DPA4C_FOR_EACH_CHANNEL(DPA4C_DECLARE_CHANNEL)

#undef DPA4C_DECLARE_CHANNEL

}  // namespace deepmd_dpa4c
