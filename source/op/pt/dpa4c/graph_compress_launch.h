// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Host-side launch interface of the compressed DPA4C descriptor.
//
// The kernel templates are instantiated in generated translation units
// (shard.cu.in), one per angular degree for the widest scalar widths and one
// per width for the others, which keeps the compile time of the topology
// specializations bounded and parallel. The dispatching translation unit only
// sees the plain argument bundle declared here.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace deepmd_dpa4c {

constexpr int kWarpSize = 32;

/// Element type of the topology indices, folded to two widths because a
/// signed and an unsigned 32-bit index share their representation over the
/// non-negative range that node and edge identifiers occupy.
enum class IndexKind : int { Bits32 = 0, Bits64 = 1 };

/// Words of the presence bitmap of the node types of a system, which covers
/// the types below 32 times this count (`Arguments::type_present`).
constexpr int kTypeWords = 4;

/// Mode profiles that a folded scan carries as extra features and that a
/// slotted scan holds in the slots of its types. A model with more modes
/// scans unfolded.
constexpr int kFoldedModes = 4;

/// Complete argument bundle of one descriptor launch.
///
/// Pointers that a given direction does not read stay null. All tensors are
/// contiguous. The tables, weights and per-node and per-edge features are
/// fp32; the topology and the coupling layout are integers, the node-type
/// words an integer flag, an fp32 radius and a bitmap, the force totals
/// 64-bit fixed point, and the pair energy fp64.
struct Arguments {
  long node_count = 0;
  // Node count of the whole system, from which the edge scans choose their
  // row split and their co-scheduled blocks, both of which change the
  // summation order of a row, so that every run of a tiled system takes the
  // schedule of the whole system.
  long system_node_count = 0;
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
  // Whether the canonical edge scans request the bond vectors of each row into
  // L2 as the row starts, and the slotted scans its source indices as well.
  // Set when the bond vectors of the edge storage exceed the L2 cache, as
  // those of a graph that fills the storage then stream from DRAM: a row's
  // vectors are contiguous, the lanes of a warp that scans the row request
  // one 128-byte line each until the row is covered, and the transfer then
  // overlaps the scan of the row instead of stalling the first use of every
  // line. A caller may size the storage beyond the edges.
  bool prefetch_bonds = false;
  int coupling_count = 0;
  float table_stride = 0.0f;
  float table_max = 0.0f;
  float rcut = 0.0f;
  float eps = 0.0f;
  float degree_floor = 0.0f;
  // Inner bridging switch of the edge envelope, as the two dimensionless
  // fractions of a pair's own length scale that bound its window and the
  // reciprocal of their difference. A bridged launch without a window keeps
  // the outer fraction at zero, which no edge lies below.
  float f_inner = 0.0f;
  float f_outer = 0.0f;
  float inverse_f_width = 0.0f;
  // Whether the launch belongs to a zone-bridging composition, that is whether
  // it carries a bridging window or an analytical pair potential. Only the
  // bridged specializations evaluate either.
  bool bridged = false;
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
  // Low-rank radial table of the single-type and slotted scans, null when
  // the descriptor carries none: the leading ``C1`` channels, the ``Basis``
  // functions on which the scalar-only channels are expanded, then the radial
  // modes, in the layout of ``table``. ``basis_matrix`` is the transpose of
  // the expansion, of shape ``(Basis, C0 - C1)``: channel ``c >= C1`` is the
  // sum over ``k`` of ``basis_matrix[k * (C0 - C1) + c - C1]`` times basis
  // function ``k``.
  const float* basis_table = nullptr;
  const float* basis_matrix = nullptr;
  const float* pair_film = nullptr;
  const float* pair_mixing = nullptr;
  const float* type_embedding = nullptr;
  const float* readout_matrices = nullptr;
  const int* coupling_meta = nullptr;
  const int* coupling_entry = nullptr;
  const float* coupling_value = nullptr;
  const float* output_mean = nullptr;
  const float* output_inv_std = nullptr;
  // Per-element length scale in Å, indexed by atom type over the whole type
  // table including the padding row. The sum of the two entries of a pair is
  // the distance the window fractions are measured against. Read by the
  // bridged specializations only.
  const float* contact_radius = nullptr;
  const float* descriptor_gradient = nullptr;
  const float* state = nullptr;
  // Device word that stays zero while every node of the system carries the
  // type of node zero. A canonical launch of such a system scans the bare
  // radial features; null disables that scan.
  const int* type_mismatch = nullptr;
  // Largest length-scale radius among the node types of the system, written
  // on the device by the scan of the node types that precedes the edge kernels
  // of a canonical launch, or null. No bond of a row lies inside a bridging
  // window once it is at least `f_outer` times the sum of the centre radius
  // and this radius long, so the edge scans resolve the window of a pair only
  // for the shorter bonds of the row.
  const float* largest_radius = nullptr;

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

  // === Analytical pair potential ===
  // Present together or not at all, and read by the edge backward only, which
  // is the one scan that sees an edge together with the energy cotangent of
  // its destination. ``pair_table`` holds, per ordered type pair, the four
  // amplitudes and the four decay rates of the screened Coulomb series
  // V(r) = (1/r) sum_k A_k exp(-c_k r). ``pair_seed`` is that cotangent and
  // ``pair_energy`` the fp64 node energies onto which the scan adds V/2 per
  // edge. Both are node indexed and offset by the caller, like ``descriptor``.
  const float* pair_table = nullptr;
  const double* pair_seed = nullptr;
  double* pair_energy = nullptr;

  float* descriptor = nullptr;
  float* state_out = nullptr;
  float* moment_gradient = nullptr;
  // Per-edge energy cotangent. A null pointer skips the store, which a
  // backward that assembles the force itself only needs for the per-atom
  // virial.
  float* edge_gradient = nullptr;
  // Per-node on-site magnetic gradient and the per-edge contribution to the
  // source node's magnetic gradient. The latter is reduced onto source nodes
  // either by the shared edge force assembly, which walks the source CSR, or
  // by the fused assembly below.
  float* spin_gradient = nullptr;
  float* edge_spin_gradient = nullptr;

  // === Fused force assembly ===
  // Present together or not at all. The edge backward then closes the
  // conservative force and the virial itself:
  //
  //   force[i] = sum(dst=i) g_e - sum(src=i) g_e
  //   W_i      = sum(dst=i) -g_e (x) r_e
  //
  // ``force`` and, for a spin-conditioned launch, ``spin_force`` (the source
  // totals of the magnetic edge cotangent) hold signed fixed-point sums in
  // units of ``1 / kForceFixedScale`` over the absolute node axis, like
  // ``atype``, because the source half lands on nodes of other runs; node
  // ``n`` owns the ``kForceWords`` words from ``n * kForceWords``.
  // ``node_virial`` holds the nine destination-attributed components of each
  // node of the run, offset by the caller like ``descriptor``.
  unsigned long long* force = nullptr;
  unsigned long long* spin_force = nullptr;
  float* node_virial = nullptr;

  // === Node types of the system ===
  // Bitmap of the node types of the system, `kTypeWords` device words in
  // which bit `t % 32` of word `t / 32` is set when a node carries type `t`,
  // written by the scan of the node types that also writes `type_mismatch`.
  // A canonical launch of a system of a few types slots the mode profiles
  // and the low-rank basis functions of every edge by the type of its source
  // (see `TypeSlots`). It is null, which disables the slotted scans, for a
  // launch that is not canonical, of more types than the bitmap covers, the
  // padding type included, of more than `kFoldedModes` radial modes, and of a
  // descriptor without modes and without a low-rank table.
  const unsigned* type_present = nullptr;
};

/// Resolution of the fixed-point force sums, 2^32 per unit.
///
/// Integer addition is associative, so the totals do not depend on the order
/// in which concurrent atomics land and the force is reproducible bit for bit.
/// Each contribution is rounded to the nearest multiple of 2^-32 eV/Å, an
/// absolute error of at most 1.2e-10 eV/Å that lies below its fp32 rounding
/// once it exceeds about 2e-3 eV/Å, and a node keeps a range of 2^31 eV/Å
/// (see kForceFixedLimit).
constexpr float kForceFixedScale = 4294967296.0f;

/// Fixed-point words per node of the force sums: the three components and the
/// range word, so that the 32 bytes of a node fill exactly one sector. The
/// three atomics an edge adds onto its source then land in one sector and
/// travel in one instruction, which is what bounds the source scatter of the
/// edge backward.
constexpr int kForceWords = 4;

/// Word of the fixed-point totals of a node that a contribution of
/// kForceFixedLimit or more sets; the node then reports its energy as not a
/// number, which marks the forces of its frame.
constexpr int kForceRangeWord = 3;

/// Magnitude in eV/Å from which a force contribution lies outside the range
/// of the fixed-point totals: half the range of a node, which every
/// contribution whose conversion would saturate reaches, while any two
/// contributions below it sum within the range. The check covers each
/// contribution: a total that leaves the range through three or more
/// contributions below the limit is not detected, which takes three
/// neighbours of one atom within 0.001 Å (carbon) to 0.01 Å (uranium) of it
/// under the unscreened Coulomb repulsion, and closer under any screened pair
/// term.
constexpr float kForceFixedLimit = 1073741824.0f;

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

// Functions on which the single-type and slotted scans expand the
// scalar-only channels, the channels from ``C1`` on that feed degree zero
// alone.
//
// The radial map of a trained descriptor spans far fewer independent functions
// than it has channels: on the released checkpoints the largest singular value
// of its scalar-only channels that a 32-function basis discards lies more than
// nine orders of magnitude below the leading one, and the basis reproduces
// every channel and its radial derivative to the rounding of the
// single-precision table. A scan that accumulates the basis functions instead
// of the channels reads that many table columns per edge instead of
// ``C0 - C1``, and the expansion runs once per node, or once per node type of
// the system in a slotted scan. The width is a multiple of every edge-group
// width of these profiles, so the basis columns tile the lanes evenly.
// Narrower profiles have too few scalar-only channels for an expansion
// to save reads. The compression adopts the basis only when it reproduces the
// channels of the descriptor at hand, so a model of higher rank keeps the
// direct scan.
constexpr int low_rank_basis(int channels) {
  return channels == 64 || channels == 128 ? 32 : 0;
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
// Node-group width candidates. The node kernels hold six shared arrays whose
// extent is the product of the resident node groups and the scalar width: the
// three geometric quantities of the recompute and their three cotangents. That
// footprint, not the register file, is what bounds the backward's resident
// blocks once the scalar width grows, and how soon it binds depends on the
// device's shared-memory capacity rather than on its architecture generation.
// Both widths are compiled and :func:`BackwardLauncher::prefer_wide_nodes`
// picks between them from what the occupancy API reports for the running
// device. A width is a lane count, so it must be a power of two and must not
// exceed the warp.
constexpr int kNodeLanesNarrow = 8;
constexpr int kNodeLanesWide = 16;

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

// Co-scheduled rows of the canonical edge scans.
//
// An edge reads one row of the radial table, and the row is served from L1
// only while the warps resident on a multiprocessor read nearby rows at the
// same time. A destination row is sorted by bond length, so a warp sweeps the
// table from short to long bonds, and warps that start their rows together
// stay in step. One warp per block lets the block scheduler stagger the
// resident warps until they read unrelated rows, and a wide table then
// streams from L2 once per edge. The scans of the wide profiles therefore
// give one block `Warps` consecutive destination rows, enough warps to fill a
// multiprocessor on its own, and realign them at bond-length intervals of
// `kLockstepWindow` with a block barrier. The narrow profiles hold their
// table in L1 under any schedule and keep blocks of independent rows, as does
// every system too small to give each multiprocessor two co-scheduled blocks,
// unless the split scans share its rows among several warps (`row_split`).
// The widths are the measured optima on disordered diamond carbon; the
// backward of the high-degree profiles needs the realignment one profile
// earlier because its edges carry more work between table reads. A spin
// profile realigns where its spin-free counterpart does, in blocks of fewer
// rows (`coscheduled_warps`).
template <int Channels, int Lmax, bool Backward>
struct Lockstep {
  static constexpr int Warps =
      Channels >= 128 || (Backward && Channels >= 64 && Lmax >= 3) ? 24 : 1;
};

/// Bond-length interval in Å between two realignments of the co-scheduled
/// rows of one block.
constexpr float kLockstepWindow = 4.0f;

// ``HasSpin`` is deliberately without a default. It changes the moment
// layout and the descriptor width, so an instantiation that omits it would
// silently read a spin-free layout out of a spin-conditioned buffer.
// ``NodeLanes`` overrides the node-group width; zero selects the narrow
// candidate. Only the node kernels instantiate the override, so the edge
// kernels and every existing use of ``Profile`` are unaffected.
template <int Channels, int Lmax, bool HasSpin, int NodeLanes = 0>
struct Profile {
  static constexpr int C0 = Channels;
  static constexpr int C1 = degree_one_width(Channels);
  static constexpr int C2 = degree_two_width(Channels);
  static constexpr int K1 = C2;
  static constexpr int K2 = 2;
  // Width of the low-rank expansion of the scalar-only channels, zero for a
  // profile whose single-type scan reads the channels directly. The spin
  // branch never folds, so a spin-conditioned profile carries none.
  static constexpr int Basis = HasSpin ? 0 : low_rank_basis(Channels);

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
  static constexpr int NodeWidth =
      NodeLanes != 0 ? NodeLanes : kNodeLanesNarrow;
  static constexpr int NodeGroups = kWarpSize / NodeWidth;
  static constexpr int Threads = kWarpSize;
};

/// Resident warps per multiprocessor an edge kernel is compiled for. A kernel
/// of blocks of W warps passes the count divided by W as the minimum block
/// count of its launch bounds, so the count sets the register budget per
/// thread: thirty-two warps exhaust the 65,536-register file at sixty-four
/// registers per thread, and a lower count raises the budget in proportion at
/// the cost of occupancy. Zero sets no budget and leaves the register count to
/// the compiler. The comments below count one-warp blocks, which a
/// multiprocessor holds as many of as its block limit allows.
template <int Channels, int Lmax, bool HasModes, bool HasSpin, bool Backward>
constexpr int edge_resident_blocks() {
#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 890 || (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300))
  // A multiprocessor of compute capability 8.9 or 12.x holds at most
  // twenty-four blocks, so eighty registers is the tightest budget there. It
  // serves the radial-mode kernels of sixty-four channels and more, to which
  // the compiler gives 86 to 128 registers: the budget restores the resident
  // warps those counts cost for a spill of at most 80 bytes and shortens the
  // plain kernels by 3% to 13% on 12.0. The other kernels run fastest on the
  // compiler's own count, which keeps the spin families in registers and suits
  // the narrow profiles, whose two-lane edge groups keep sixteen edges in
  // flight per warp.
  //
  // The spin kernels keep the budget where their working set fits eighty
  // registers, which on 12.0 holds for the degree-two forward kernels and for
  // the degree-two backward kernels of sixty-four channels and more. The
  // others, whose geometric state at degree three or whose spin payload on
  // lanes that own few channels exceeds eighty registers, run fastest on the
  // compiler's own count.
  if (HasSpin) {
    return Lmax == 2 && (!Backward || Channels >= 64) ? 24 : 0;
  }
  return HasModes && Channels >= 64 ? 24 : 0;
#elif defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870)
  // A multiprocessor of compute capability 7.5, 8.6 or 8.7 holds at most
  // sixteen blocks, which leaves a one-warp block 128 registers, more than
  // any edge kernel takes, so the one-warp kernels keep the compiler's own
  // count there. A co-scheduled block of twenty-four warps is bounded by its
  // thread count alone, at eighty registers, and one of twenty, at ninety-six,
  // on every architecture.
  return 0;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
  // At the thirty-two resident blocks that a multiprocessor of compute
  // capability 7.0 holds, sixty-four registers per thread, the edge backward of
  // every spin-free profile spills 40 to 224 bytes. Twenty-four blocks, eighty
  // registers, remove all but 8 to 24 bytes of that spill at degree two and
  // about halve it at degree three. On a V100 they shorten the LAMMPS step of
  // Nano by 13% to 30% on carbon, BCN and five elements alike, and that of Mini
  // by 2% to 9% on BCN and five elements and by up to 35% on carbon, whose
  // single-type scan keeps the sums of its extra feature live through the edge
  // loop; against the compiler's own count they shorten the Nano step by 1.5%
  // to 5.4% on BCN and five elements.
  //
  // The spin kernels of degree two take twenty-four blocks for the forward and
  // for the backward of sixty-four channels and more, and the compiler's own
  // count for the backward of fewer channels, at which the spin backward of
  // Nano and Mini holds its payload without spilling and the spin step of Nano
  // is 14% to 22% shorter and that of Mini up to 1%. The kernels of degree
  // three serve the generic graph path and the LAMMPS systems of more than
  // eight node types; unmeasured on 7.0, they keep the thirty-two blocks of the
  // architectures without a branch of their own.
  if (HasSpin) {
    return Lmax == 2 ? (!Backward || Channels >= 64 ? 24 : 0) : 32;
  }
  return 24;
#else
  // Measured on compute capability 9.0, whose multiprocessor holds thirty-two
  // blocks; 8.0 and 10.x take these values unmeasured. Thirty-two blocks leave
  // sixty-four registers per thread. The backward of eight channels, whose
  // two-lane edge groups keep sixteen edges in flight per warp, spills 56
  // bytes there; twenty-four blocks give it eighty registers, which remove the
  // spill, shorten the kernel by 14% and the LAMMPS step of such a model by
  // 3.3% to 5.3% at 10,296 and 100,800 atoms. The backward of sixteen channels
  // does not spill at sixty-four registers and takes 3% to 5% longer at
  // eighty, those of thirty-two and sixty-four channels take up to 5% longer
  // at eighty on a system of one element, and the forward kernels change by at
  // most 4% either way, under 1% of the step, so the other kernels keep
  // thirty-two blocks. The degree-two spin backward spills 16 to 104 bytes at
  // the eighty registers of twenty-four blocks and runs 3% to 6% faster on the
  // compiler's own count, 96 to 127 registers, while its forward runs fastest
  // at twenty-four blocks. The degree-three spin kernels keep thirty-two
  // blocks; a system of at most eight types scans them slotted, so no
  // production path has measured them.
  if (HasSpin) {
    return Lmax == 2 ? (Backward ? 0 : 24) : 32;
  }
  return Backward && Channels == 8 ? 24 : 32;
#endif
}

/// Resident warps per multiprocessor a slotted edge kernel is compiled for
/// (see `edge_resident_blocks`), for blocks of `CoscheduledWarps` co-scheduled
/// rows, or zero for blocks of independent or shared rows. The slots add one
/// set of sums per slot to the state of a lane. A co-scheduled block fills a
/// multiprocessor alone, so its rows are the resident warps, and its thread
/// count sets the register budget. The other scans that slot the mode
/// profiles keep sixteen resident warps, which the 65,536-register file of
/// every supported architecture gives 128 registers per thread, and those
/// that slot the basis functions alone keep twenty, 96 registers per thread;
/// a block of one row shared by eight warps, a width that does not divide
/// twenty, keeps sixteen there as well.
template <bool HasModes, int CoscheduledWarps>
constexpr int slot_resident_warps() {
  if constexpr (CoscheduledWarps > 0) {
    return CoscheduledWarps;
  } else {
    return HasModes ? 16 : 20;
  }
}

/// Scalar widths that own a compiled specialization. Each width, with every
/// degree that graph_compress.cu dispatches, owns instantiations in a shard
/// that source/op/pt/CMakeLists.txt generates; a width or degree is added in
/// both places, and a Linux build that misses one fails at the link of the
/// operator library.
#define DPA4C_FOR_EACH_CHANNEL(macro) \
  macro(8) macro(16) macro(32) macro(64) macro(128)

/// One direction of the descriptor for one scalar width and angular degree.
/// The angular degree is a compile-time specialization because degrees three
/// and four add moment accumulators that must not enter the register budget
/// of the production ``lmax=2`` path. Every width of DPA4C_FOR_EACH_CHANNEL
/// and every degree from 2 to 4 is instantiated, both directions of a degree
/// in one generated translation unit (shard.cu.in).
template <int Channels, int Lmax>
void launch_forward(const Arguments& arguments, cudaStream_t stream);
template <int Channels, int Lmax>
void launch_backward(const Arguments& arguments, cudaStream_t stream);

}  // namespace deepmd_dpa4c
