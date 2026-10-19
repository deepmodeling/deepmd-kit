// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Kernels of the compressed degree-wise DPA4C descriptor.
//
// The header is included by the generated shards of shard.cu.in, each of
// which instantiates both directions of the descriptor for one scalar width
// and one or more angular degrees. Keeping the instantiations separated
// bounds the compile time of any single translation unit and lets them
// proceed in parallel.

#pragma once

#include <c10/util/Exception.h>

#include <climits>
#include <map>
#include <mutex>
#include <type_traits>

#include "graph_compress.cuh"

namespace deepmd_dpa4c {

// Shared mode profiles are cached per edge group. The stride is a multiple of
// four so that each row is aligned for the vector reads of the mode residual,
// and it keeps the concurrent groups of one warp on distinct banks.
constexpr int kModeStride = kMaxRadialModes + 4;

// === Native spin block accessors ===
//
// Channel zero of each spin block is the node-local on-site channel, which is
// stored outside the reduced region because it carries no neighborhood
// normalizer. Reading both through one accessor keeps the Gram loops free of
// that distinction.

/// Flat moment coordinate of one entry of the joint degree-one spin block.
///
/// The block holds the on-site moment at channel zero, the ``Cs`` isotropic
/// neighbour channels, and the ``Cs`` bond-projected neighbour channels, in
/// that order. The two neighbour families share the grading of the block, so
/// the Gram of the whole block is what emits every admissible degree-one spin
/// invariant; resolving all three regions through one accessor keeps the Gram
/// and its VJP free of the distinction between them.
template <int Channels, int Lmax>
__device__ __forceinline__ int spin_vector_offset(int component, int channel) {
  using P = Profile<Channels, Lmax, true>;
  if (channel == 0) {
    return P::SpinOnsiteVector + component;
  }
  const int neighbor = channel - 1;
  return neighbor < P::Cs
             ? P::SpinVector + component * P::Cs + neighbor
             : P::SpinBond + component * P::Cs + (neighbor - P::Cs);
}

template <int Channels, int Lmax>
__device__ __forceinline__ float spin_vector_value(const float* moments,
                                                   int component,
                                                   int channel) {
  return moments[spin_vector_offset<Channels, Lmax>(component, channel)];
}

template <int Channels, int Lmax>
__device__ __forceinline__ float spin_tensor_value(const float* moments,
                                                   int component,
                                                   int channel) {
  using P = Profile<Channels, Lmax, true>;
  return channel == 0 ? moments[P::SpinOnsiteTensor + component]
                      : moments[P::SpinTensor + component];
}

/// Decode one retained entry of the quadrupole Gram, whose on-site self-term
/// is omitted because ``|B_2(s)|^2 = |s|^4`` makes it a per-type constant
/// times the vector block's own on-site self-term.
__device__ __forceinline__ void decode_spin_tensor_pair(int entry,
                                                        int& row,
                                                        int& column) {
  row = entry;
  column = 1;
}

// === Single-type systems ===
//
// The PairFiLM amplitude and the mode mixing are affine in the radial
// features, with coefficients fixed by the ordered type pair of the edge, and
// every moment is linear in the amplitude. In a system whose atoms share one
// type every edge carries the same pair, so the coefficients are constants:
// the scans then carry the bare radial features and the coefficients apply
// once per node, to the reduced sums in the forward and to the cotangents in
// the backward. The extra features of that formulation are the mode profiles,
// at feature index `r < radial_modes`, and the constant feature that the FiLM
// shift multiplies, at index `radial_modes`; each reaches every harmonic
// component.
//
// The folded scans pay off once the per-channel coefficient work is a sizeable
// part of an edge, which holds from thirty-two channels on.

template <int Channels, bool HasSpin, bool Canonical>
constexpr bool foldable() {
  return Canonical && !HasSpin && Channels >= 32;
}

// === Systems of a few types ===
//
// The edges of a row belong to at most as many ordered pairs as the system
// has node types, and every moment is linear in the coefficients of the pair
// of each edge. The scans of a system of at most eight types therefore
// accumulate the features that a pair coefficient multiplies separately for
// the edges of each source type, in the slot of that type, and apply the
// coefficients of the pair of each slot once per row: to the reduced sums in
// the forward and to the node cotangents in the backward. Two kinds of
// feature are slotted:
//
// - The mode profiles. The mode mixing adds `sum_r M[p, c, r] q_r` to the
//   amplitude of channel `c` of an edge of pair `p`. A slot holds one moment
//   per mode and harmonic component, and lane `l` of an edge group owns mode
//   `l % kFoldedModes` of slot `l / kFoldedModes + k * (EdgeWidth /
//   kFoldedModes)` in its slot tile `k` (`slot_owner`, `owned_slot`).
// - The basis functions of the low-rank table (see `low_rank_basis`). The
//   channels from `C1` on feed degree zero alone, and the FiLM amplitude
//   `gamma[p, c] sum_k V[k, c] b_k + beta[p, c]` of such a channel is linear
//   in the basis functions `b_k` and in the constant feature that the shift
//   multiplies. In the forward a lane holds the sums of its basis columns for
//   every slot, and lane `s` of an edge group the sum of the envelopes of the
//   edges of slot `s`; the backward reads the cotangents of the basis
//   functions and of the shift of the slot of every edge from shared memory.
//
// The channels below `C1` keep their FiLM coefficients on the edges, as the
// pairwise scans do.
//
// Every slot adds state to every lane, sums in registers and, in the
// backward, rows of cotangents in shared memory, and every edge passes the
// slots of its lanes in predicated or zero-weighted updates, so a scan pays
// for the slots it compiles whatever the system: a system of up to four types
// takes the kernels of four slots, and one of five to eight types those of
// eight (`type_slots`). The profiles that slot the mode profiles run
// the kernels of eight slots in both directions, and those that slot the
// basis functions alone in the forward only: their backward of eight slots is
// no faster than the pairwise backward, since the shared rows of eight slots
// per warp need a shared-memory configuration that takes L1 cache from the
// table reads, unless the kernel gives up resident warps. The forward of such
// a system reads the low-rank table and its backward the radial table, two
// interpolants whose knot values and slopes agree to the tolerance of the
// adoption check of the low-rank table (`build_low_rank_table`), so its forces
// follow the gradient of its energy to the accuracy of the compression.
//
// A spin profile slots its mode profiles alone, since it carries no low-rank
// table; the spin families weight the moment of every neighbour and stay on
// the edges. It has no single-type scan, so its slotted scans serve a system
// of one type as well, in a single slot.

/// Whether the edge kernels of a profile slot the features of a system of a
/// few types: a spin-free profile whose single-type scans fold and that
/// carries radial modes or admits a low-rank table, or a canonical spin
/// profile of thirty-two channels or more that carries radial modes.
template <int Channels, bool HasModes, bool HasSpin, bool Canonical>
constexpr bool slottable() {
  if constexpr (HasSpin) {
    return HasModes && Canonical && Channels >= 32;
  } else {
    return (HasModes || low_rank_basis(Channels) > 0) &&
           foldable<Channels, HasSpin, Canonical>();
  }
}

/// Slot tiles of a lane of an edge group of `EdgeWidth` lanes in a scan of
/// `Slots` slots.
template <int EdgeWidth, int Slots>
constexpr int slot_tiles() {
  return Slots * kFoldedModes / EdgeWidth;
}

/// Lane of an edge group of `EdgeWidth` lanes that holds mode `mode` of slot
/// `slot`, in its slot tile `slot / (EdgeWidth / kFoldedModes)`.
template <int EdgeWidth>
__device__ __forceinline__ int slot_owner(int slot, int mode) {
  return (slot % (EdgeWidth / kFoldedModes)) * kFoldedModes + mode;
}

/// Slot whose mode `lane % kFoldedModes` lane `lane` of an edge group of
/// `EdgeWidth` lanes holds in its slot tile `tile`.
template <int EdgeWidth>
__device__ __forceinline__ int owned_slot(int lane, int tile) {
  return lane / kFoldedModes + tile * (EdgeWidth / kFoldedModes);
}

/// The `Slots` slots of the node types of a system: byte `s` of `types` holds
/// the type of slot `s`, the types in increasing order, every byte past the
/// types of the system holds 0xFF, above every type, and `count` is the
/// number of types.
template <int Slots>
struct TypeSlots {
  using Word = std::conditional_t<(Slots > 4), unsigned long long, unsigned>;
  static_assert(Slots <= static_cast<int>(sizeof(Word)),
                "a byte of the word holds the type of each slot");
  Word types = ~Word{0};
  int count = 0;
};
static_assert(32 * kTypeWords <= 0xFF,
              "every node type of the bitmap lies below the 0xFF sentinel");

/// Number of node types of a system, or zero for a launch without their
/// bitmap (see `Arguments::type_present`).
__device__ __forceinline__ int node_type_count(const Arguments& args) {
  if (args.type_present == nullptr) {
    return 0;
  }
  int count = 0;
#pragma unroll
  for (int word = 0; word < kTypeWords; ++word) {
    count += __popc(__ldg(args.type_present + word));
  }
  return count;
}

/// Whether the system of `args` holds `fewest` to `most` node types, for a
/// launch that carries their bitmap. The test reads the bitmap alone, which
/// no pairwise or single-type scan reads, so that the kernel that it retires
/// keeps the code of a kernel without it (see `retire_unless`).
__device__ __forceinline__ bool holds_types(const Arguments& args,
                                            int fewest,
                                            int most) {
  const int count = node_type_count(args);
  return count >= fewest && count <= most;
}

/// End the calling thread unless `serves` holds; `serves` is uniform over the
/// launch. The exit is a predicated instruction of inline PTX, which the
/// optimizer treats as opaque, so the scan that follows is optimized as in a
/// kernel without the test.
__device__ __forceinline__ void retire_unless(bool serves) {
  asm volatile(
      "{\n\t.reg .pred retire;\n\t"
      "setp.eq.u32 retire, %0, 0;\n\t"
      "@retire exit;\n\t}" ::"r"(static_cast<unsigned>(serves)));
}

/// The slots of the node types of a system of at most `Slots` types.
template <int Slots>
__device__ __forceinline__ TypeSlots<Slots> load_type_slots(
    const Arguments& args) {
  using Word = typename TypeSlots<Slots>::Word;
  TypeSlots<Slots> slots;
#pragma unroll
  for (int word = 0; word < kTypeWords; ++word) {
    for (unsigned bits = __ldg(args.type_present + word); bits != 0u;
         bits &= bits - 1u) {
      const Word type = 32u * word + __ffs(bits) - 1u;
      const int shift = 8 * slots.count;
      slots.types = (slots.types & ~(Word{0xFFu} << shift)) | (type << shift);
      ++slots.count;
    }
  }
  return slots;
}

/// Request the source indices of the canonical destination row `[begin, end)`
/// into L2 when the launch streams its bonds (see `Arguments::prefetch_bonds`),
/// one 128-byte line per lane of the warp from `lane`. A slotted scan reads the
/// type of the source of every edge to select its slot, so the index heads the
/// dependency chain of the slot sums as the bond vector heads that of the
/// table reads.
template <typename index_t>
__device__ __forceinline__ void prefetch_row_sources(const Arguments& args,
                                                     const index_t* edge_index,
                                                     long begin,
                                                     long end,
                                                     int lane) {
  if (args.prefetch_bonds) {
    prefetch_lines<CacheLevel::L2>(
        reinterpret_cast<const float*>(edge_index + begin),
        (end - begin) * static_cast<long>(sizeof(index_t) / sizeof(float)),
        lane, kWarpSize);
  }
}

/// Slot of `type`, a node type of the system: the number of its types below
/// it.
template <int Slots>
__device__ __forceinline__ int type_slot(const TypeSlots<Slots>& slots,
                                         int type) {
  int slot = 0;
#pragma unroll
  for (int index = 1; index < Slots; ++index) {
    slot += type >= static_cast<int>((slots.types >> (8 * index)) & 0xFFu);
  }
  return slot;
}

/// Node type of slot `slot` of a system of more than `slot` types.
template <int Slots>
__device__ __forceinline__ int slot_type(const TypeSlots<Slots>& slots,
                                         int slot) {
  return static_cast<int>((slots.types >> (8 * slot)) & 0xFFu);
}

/// Edge body of a destination row. A row of varying pairs reads the ordered
/// pair of every edge (`Pairwise`). A single-type row folds the pair
/// coefficients onto the node and reads the channels of the radial table
/// (`Direct`), or the low-rank table of a descriptor that carries one
/// (`LowRank`, see `low_rank_basis`). A row of a system of a few types slots
/// its mode profiles and reads the radial table (`Slotted4` and `Slotted8`),
/// or slots its mode profiles and the basis functions of the low-rank table
/// (`SlottedLowRank4` and `SlottedLowRank8`), in four or eight slots. Each is
/// an instantiation of one edge body, so that each loop keeps only its own
/// state live.
///
/// An edge kernel is instantiated for one folded scan: a single-type scan,
/// which it runs on a single-type system and the pairwise scan on any other,
/// or a slotted scan, which it runs on every system it serves. Whether a
/// descriptor carries the low-rank table is known on the host, so the kernels
/// hold the low-rank instantiations only when they run them.
enum class Scan {
  Pairwise,
  Direct,
  LowRank,
  Slotted4,
  SlottedLowRank4,
  Slotted8,
  SlottedLowRank8
};

template <Scan Kind>
using ScanTag = std::integral_constant<Scan, Kind>;

/// Whether a scan of kind `kind` slots the basis functions of the low-rank
/// table.
constexpr bool slots_basis(Scan kind) {
  return kind == Scan::SlottedLowRank4 || kind == Scan::SlottedLowRank8;
}

/// Whether a scan of kind `kind` slots the features of a system of a few
/// types.
constexpr bool slots_types(Scan kind) {
  return kind == Scan::Slotted4 || kind == Scan::Slotted8 || slots_basis(kind);
}

/// Whether a scan of kind `kind` reads the low-rank table.
constexpr bool reads_low_rank(Scan kind) {
  return kind == Scan::LowRank || slots_basis(kind);
}

/// Slots of a scan of kind `kind`: eight for the slotted scans of a system
/// of five to eight node types, four for the others.
constexpr int type_slots(Scan kind) {
  return kind == Scan::Slotted8 || kind == Scan::SlottedLowRank8 ? 8 : 4;
}

/// Fewest node types of a system that the slotted scans of kind `kind` serve:
/// one more than the four slots of the other slotted scans for the scans of
/// eight slots, and for those of four two, or one for a spin profile, which
/// has no single-type scan.
template <bool HasSpin>
constexpr int fewest_types(Scan kind) {
  return type_slots(kind) > 4 ? 5 : (HasSpin ? 1 : 2);
}

/// Table columns a scan of `Kind` reads before its mode profiles or extra
/// features: the channels, or the degree-one channels followed by the basis
/// functions.
template <class P, Scan Kind>
constexpr int fold_columns() {
  return reads_low_rank(Kind) ? P::C1 + P::Basis : P::C0;
}

/// Whether the edge scans of direction `Backward` of a descriptor whose
/// single-type scan is `Fold` launch the slotted kernels of `Slots` slots
/// (see `type_slots`) beside their other kernel: for a profile that slots,
/// those of four slots when the descriptor carries the mode profiles or the
/// low-rank table, and those of eight slots as well but in the backward of a
/// descriptor without mode profiles.
template <int Channels,
          bool HasModes,
          bool HasSpin,
          bool Canonical,
          bool Backward,
          Scan Fold,
          int Slots>
constexpr bool runs_slotted() {
  return slottable<Channels, HasModes, HasSpin, Canonical>() &&
         (HasModes || Fold == Scan::LowRank) &&
         (Slots <= 4 || HasModes || !Backward);
}

/// Whether the edge kernel of folded scan `Fold` of direction `Backward`
/// retires on the systems that another kernel of its launch serves: every
/// slotted kernel, and the other kernel of a launch with slotted kernels.
template <int Channels,
          bool HasModes,
          bool HasSpin,
          bool Canonical,
          bool Backward,
          Scan Fold>
constexpr bool shares_launch() {
  return slots_types(Fold) || runs_slotted<Channels, HasModes, HasSpin,
                                           Canonical, Backward, Fold, 4>();
}

/// Whether the edge kernel of folded scan `Fold` of direction `Backward`
/// serves the system of `args`: a slotted kernel the systems of
/// `fewest_types` to `type_slots` node types, and the other kernel of its
/// launch every system that no slotted kernel of the launch serves.
template <int Channels,
          bool HasModes,
          bool HasSpin,
          bool Canonical,
          bool Backward,
          Scan Fold>
__device__ __forceinline__ bool serves_system(const Arguments& args) {
  if constexpr (slots_types(Fold)) {
    return holds_types(args, fewest_types<HasSpin>(Fold), type_slots(Fold));
  } else {
    constexpr int Most = runs_slotted<Channels, HasModes, HasSpin, Canonical,
                                      Backward, Fold, 8>()
                             ? type_slots(Scan::Slotted8)
                             : type_slots(Scan::Slotted4);
    return !holds_types(args, fewest_types<HasSpin>(Scan::Slotted4), Most);
  }
}

/// Whether the edge kernels of a profile run the low-rank single-type scan
/// for a descriptor that carries the low-rank table.
template <int Channels, int Lmax, bool HasSpin, bool Canonical>
constexpr bool folds_low_rank() {
  return Profile<Channels, Lmax, HasSpin>::Basis > 0 &&
         foldable<Channels, HasSpin, Canonical>();
}

/// Check the folded scan `Fold` of an edge kernel of direction `Backward`, and
/// retire the blocks of the kernel unless it serves the system of its launch:
/// the slotted kernels serve the systems of a few node types and the other
/// kernel of their launch every other system (see `retire_unless`).
template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          bool Canonical,
          bool Backward,
          Scan Fold>
__device__ __forceinline__ void enter_edge_scan(const Arguments& args) {
  static_assert(Fold != Scan::Pairwise &&
                    (!reads_low_rank(Fold) ||
                     folds_low_rank<Channels, Lmax, HasSpin, Canonical>()) &&
                    (!slots_types(Fold) || slots_basis(Fold) || HasModes),
                "the folded scan of a kernel is a single-type or a slotted "
                "scan, reads the low-rank table only where it folds, and "
                "slots the radial table for the mode profiles");
  if constexpr (shares_launch<Channels, HasModes, HasSpin, Canonical, Backward,
                              Fold>()) {
    retire_unless(
        serves_system<Channels, HasModes, HasSpin, Canonical, Backward, Fold>(
            args));
  }
}

/// Run `launch` with the single-type scan of the descriptor in `args`.
template <int Channels, int Lmax, bool HasSpin, bool Canonical, typename Launch>
void dispatch_fold(const Arguments& args, Launch&& launch) {
  if constexpr (folds_low_rank<Channels, Lmax, HasSpin, Canonical>()) {
    if (args.basis_table != nullptr) {
      launch(ScanTag<Scan::LowRank>{});
      return;
    }
  }
  launch(ScanTag<Scan::Direct>{});
}

/// Rows of a co-scheduled block of a spin edge scan: twenty warps, whose
/// block of 640 threads keeps 96 registers per thread for the spin payload,
/// which spills at the eighty registers of a block of `Lockstep` rows.
constexpr int kSpinLockstepRows = 20;

/// Co-scheduled width of a canonical edge scan, in warps per block: that of
/// `Lockstep`, in blocks of `kSpinLockstepRows` rows for a spin profile.
template <int Channels, int Lmax, bool HasSpin, bool Canonical, bool Backward>
constexpr int coscheduled_warps() {
  constexpr int Warps = Lockstep<Channels, Lmax, Backward>::Warps;
  if constexpr (!Canonical || Warps == 1) {
    return 1;
  } else {
    return HasSpin ? kSpinLockstepRows : Warps;
  }
}

/// Rows of a co-scheduled block of a slotted edge scan of 128 registers per
/// thread: sixteen warps, a block of 512 threads.
constexpr int kSlotLockstepRows = 16;

/// Co-scheduled width of a slotted edge scan of `Slots` slots in a system
/// that fills the device with its blocks, whose rows read the same windows of
/// their wide table at the same time. The spin-free scans of 128 channels
/// take the forward blocks of `Lockstep` and backward blocks of
/// `kSlotLockstepRows` rows, whose slots add their cotangents to the state of
/// every lane, and the spin scans of 128 channels blocks of
/// `kSpinLockstepRows` rows in both directions; the forward of eight slots
/// of 128 channels, whose slot sums exceed the registers of those blocks,
/// takes blocks of `kSlotLockstepRows` rows with or without spin. The spin
/// backward of degree three and four of sixty-four channels, which
/// co-schedules as its pairwise backward does (see `Lockstep`), takes blocks
/// of `kSlotLockstepRows` rows. The other scans hold the slots in the
/// registers of `slot_resident_warps` resident warps, in blocks of independent
/// rows.
template <int Channels, int Lmax, bool HasSpin, bool Backward, int Slots>
constexpr int slot_coscheduled_warps() {
  if constexpr (Channels >= 128 && Slots > 4 && !Backward) {
    return kSlotLockstepRows;
  } else if constexpr (HasSpin) {
    if constexpr (Channels >= 128) {
      return kSpinLockstepRows;
    } else {
      return Lockstep<Channels, Lmax, Backward>::Warps > 1 ? kSlotLockstepRows
                                                           : 1;
    }
  } else if constexpr (Channels < 128) {
    return 1;
  } else {
    return Backward ? kSlotLockstepRows
                    : Lockstep<Channels, Lmax, Backward>::Warps;
  }
}

/// Rows of a block of independent rows of a slotted or a spin edge scan,
/// each a warp of its own. A kernel that does not serve the system retires
/// its blocks, whose count sets the time it takes, and every block holds the
/// shared memory that the driver reserves per block besides its own: four
/// rows a block retire in a quarter of the time of one, and keep the shared
/// memory of a multiprocessor of resident spin scans within the configuration
/// that leaves the most L1, while the rows stay as independent as in blocks
/// of one row.
constexpr int kSlotRows = 4;

/// Whether the `Warps` rows of a block walk in the lockstep windows of the
/// co-scheduled blocks (see `Lockstep`), rather than each on its own: every
/// block of several rows but the blocks of `kSlotRows` independent rows.
template <int Warps>
constexpr bool windowed() {
  return Warps > 1 && Warps != kSlotRows;
}
static_assert(kSlotRows != Lockstep<64, 3, true>::Warps &&
                  kSlotRows != Lockstep<128, 2, false>::Warps &&
                  kSlotRows != kSlotLockstepRows &&
                  kSlotRows != kSpinLockstepRows,
              "a block of independent slotted rows is not a co-scheduled one");

/// Launch one edge scan with `Warps` destination rows per block, each row
/// shared by `Split` warps.
template <int Warps, int Split, typename Kernel>
void launch_edge_scan(Kernel kernel,
                      const Arguments& args,
                      cudaStream_t stream) {
  kernel<<<static_cast<int>((args.node_count + Warps - 1) / Warps),
           Warps * Split * kWarpSize, 0, stream>>>(args);
}

/// Throw the error of a failed CUDA query that a launch choice derives from,
/// after clearing it, so that no later launch check reports it again.
inline void check_query(cudaError_t status, const char* query) {
  if (status != cudaSuccess) {
    cudaGetLastError();
    TORCH_CHECK(false, "dpa4c: ", query,
                " failed: ", cudaGetErrorString(status));
  }
}

/// Per-device memo of a quantity that follows from the current device and the
/// compiled kernels alone. Each device answers once, and every function that
/// keeps a memo of its own keeps its own answers, so a quantity that depends
/// on a kernel is remembered per kernel and per device.
class DeviceMemo {
 public:
  template <typename Query>
  long get(const Query& query) {
    int device = 0;
    check_query(cudaGetDevice(&device), "the device query");
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto known = values_.find(device);
    if (known != values_.end()) {
      return known->second;
    }
    const long value = query(device);
    values_.emplace(device, value);
    return value;
  }

 private:
  std::mutex mutex_;
  std::map<int, long> values_;
};

/// Multiprocessors of `device`.
inline long multiprocessor_count(int device) {
  int multiprocessors = 0;
  check_query(cudaDeviceGetAttribute(&multiprocessors,
                                     cudaDevAttrMultiProcessorCount, device),
              "the multiprocessor count");
  return multiprocessors;
}

/// Warps that share one destination row: the most among `1, 2, 4, 8` whose
/// warps fit in one wave of the warps that `Kernel`, the scan of blocks of
/// `Rows` independent rows that serves the system otherwise, keeps resident
/// across the current device.
template <auto Kernel, int Rows>
int row_split(long node_count) {
  static DeviceMemo capacities;
  const long capacity = capacities.get([](int device) {
    int resident = 0;
    check_query(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &resident, Kernel, Rows * kWarpSize, 0),
                "the occupancy query of an edge scan");
    return multiprocessor_count(device) * resident * Rows;
  });
  int split = 1;
  while (split < 8 && node_count * split * 2 <= capacity) {
    split *= 2;
  }
  return split;
}

/// Whether a system fills the current device with co-scheduled blocks of
/// `Warps` rows: one such block fills a multiprocessor, so a run with fewer
/// than two blocks per multiprocessor would leave part of the device idle, and
/// narrow blocks then serve it better.
template <int Warps>
bool fills_coscheduled(long node_count) {
  if constexpr (Warps == 1) {
    return false;
  } else {
    static DeviceMemo capacities;
    const long capacity = capacities.get(
        [](int device) { return 2L * multiprocessor_count(device) * Warps; });
    return node_count >= capacity;
  }
}

/// The type every node of the system carries, or -1 when the scan has to
/// read the ordered pair of every edge. The answer is uniform over the grid.
__device__ __forceinline__ int single_type(const Arguments& args,
                                           int radial_modes) {
  if (args.type_mismatch == nullptr || radial_modes > kFoldedModes ||
      __ldg(args.type_mismatch) != 0) {
    return -1;
  }
  return static_cast<int>(__ldg(args.atype));
}

/// Value of one extra radial feature of a pair-uniform row.
__device__ __forceinline__ float extra_feature(const TableRow& row,
                                               int channels,
                                               int feature,
                                               int radial_modes,
                                               float coordinate) {
  if (feature < radial_modes) {
    return evaluate_table(row, channels + feature, coordinate);
  }
  return feature == radial_modes ? 1.0f : 0.0f;
}

/// Gather the nine reduced harmonic sums of one extra feature onto every
/// lane. After the merge of the edge groups the lane of feature index `f`
/// modulo the group width holds them in its tile `f / Width`.
template <int Width, int Tiles>
__device__ __forceinline__ void gather_extra_feature(
    const float (&extra)[Tiles][9], int feature, float (&sums)[9]) {
#pragma unroll
  for (int component = 0; component < 9; ++component) {
    float value = 0.0f;
#pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
      if (tile == feature / Width) {
        value = extra[tile][component];
      }
    }
    sums[component] = __shfl_sync(kWarpMask, value, feature % Width);
  }
}

/// Expand the low-rank sums of a single-type row into the scalar sums of the
/// channels from ``C1`` on (see `low_rank_basis`). On entry the tiles from
/// ``C1 / Width`` of ``scalar`` hold the sums of the lane's basis columns, the
/// same on every edge group after their merge; on exit every tile from
/// ``C1 / Width`` holds the sum of the lane's channel, accumulated over the
/// basis functions in increasing order.
template <class P, int Width, int Tiles>
__device__ __forceinline__ void expand_basis(
    float (&scalar)[Tiles],
    const float* __restrict__ basis_matrix,
    int base_channel) {
  constexpr int Direct = P::C1 / Width;
  constexpr int BasisTiles = P::Basis / Width;
  constexpr int Expanded = P::C0 - P::C1;
  float basis[BasisTiles];
#pragma unroll
  for (int tile = 0; tile < BasisTiles; ++tile) {
    basis[tile] = scalar[Direct + tile];
  }
  float sum[Tiles - Direct] = {};
#pragma unroll
  for (int k = 0; k < P::Basis; ++k) {
    const float value = __shfl_sync(kWarpMask, basis[k / Width], k % Width);
    const float* row = basis_matrix + k * Expanded + base_channel;
#pragma unroll
    for (int tile = 0; tile < Tiles - Direct; ++tile) {
      sum[tile] = fmaf(__ldg(row + tile * Width), value, sum[tile]);
    }
  }
#pragma unroll
  for (int tile = 0; tile < Tiles - Direct; ++tile) {
    scalar[Direct + tile] = sum[tile];
  }
}

/// Stage the expansion matrix of the low-rank basis transposed into the
/// shared `transposed` with the threads of the block: coefficient `k` of
/// channel `C1 + c` at `c * (Basis + 1) + k`. The lanes of a warp that each
/// project the cotangents of a row onto one basis function then read
/// consecutive words, where the matrix itself places them a row of `C0 - C1`
/// coefficients apart and every load touches a cache line per lane; the
/// padded rows keep the transposing stores free of bank conflicts.
template <class P>
__device__ __forceinline__ void stage_transposed_basis(
    const float* __restrict__ basis_matrix, float* transposed) {
  constexpr int Expanded = P::C0 - P::C1;
  for (int index = threadIdx.x; index < P::Basis * Expanded;
       index += blockDim.x) {
    const int k = index / Expanded;
    const int channel = index - k * Expanded;
    transposed[channel * (P::Basis + 1) + k] = __ldg(basis_matrix + index);
  }
}

/// Expansion coefficients `k` of the four channels from `C1 + first` on of
/// the low-rank basis, for `first` a multiple of four, in one 16-byte load:
/// a lane that projects onto one basis function reads its own row of the
/// matrix, so a load touches a cache line per lane whatever its width, and
/// the wide load serves four channels with it.
template <class P>
__device__ __forceinline__ float4
basis_quad(const float* __restrict__ basis_matrix, int k, int first) {
  static_assert((P::C0 - P::C1) % 4 == 0,
                "the rows of the expansion hold whole quads of channels");
  return __ldg(reinterpret_cast<const float4*>(basis_matrix +
                                               k * (P::C0 - P::C1) + first));
}

/// The coefficients of `basis_quad` from the transpose that the block staged
/// in shared memory (`stage_transposed_basis`).
template <class P>
__device__ __forceinline__ float4 staged_basis_quad(const float* transposed,
                                                    int k,
                                                    int first) {
  constexpr int Stride = P::Basis + 1;
  const float* column = transposed + first * Stride + k;
  return make_float4(column[0], column[Stride], column[2 * Stride],
                     column[3 * Stride]);
}

/// Project the scalar cotangents of the channels from ``C1`` on onto the
/// low-rank basis of a single-type row: the cotangent of basis function ``k``
/// is the sum over those channels, in increasing order, of the channel's
/// cotangent times its expansion coefficient ``k``.
template <class P>
__device__ __forceinline__ void project_basis(
    const float* scalar_gradient,
    const float* __restrict__ basis_matrix,
    float* basis_gradient,
    int lane) {
  for (int k = lane; k < P::Basis; k += kWarpSize) {
    float sum = 0.0f;
    for (int first = 0; first < P::C0 - P::C1; first += 4) {
      const float4 quad = basis_quad<P>(basis_matrix, k, first);
      const float coefficients[4] = {quad.x, quad.y, quad.z, quad.w};
#pragma unroll
      for (int offset = 0; offset < 4; ++offset) {
        sum = fmaf(coefficients[offset],
                   scalar_gradient[P::C1 + first + offset], sum);
      }
    }
    basis_gradient[k] = sum;
  }
}

// === Lockstep windows ===

/// Squared true length of the bond stored at `position`.
__device__ __forceinline__ float bond_length_squared(const float* edge_vec,
                                                     long position) {
  const float x = edge_vec[position * 3 + 0];
  const float y = edge_vec[position * 3 + 1];
  const float z = edge_vec[position * 3 + 2];
  return x * x + y * y + z * z;
}

/// End of the window that starts at `position`: the first position from there
/// on whose bond is at least as long as the window bound, or `end`. The warp
/// examines thirty-two positions at a time and answers uniformly.
__device__ __forceinline__ long window_end(long position,
                                           long end,
                                           float bound_squared,
                                           const float* edge_vec,
                                           int lane) {
  for (;;) {
    const long candidate = position + lane;
    const bool inside =
        candidate < end &&
        bond_length_squared(edge_vec, candidate) < bound_squared;
    const unsigned run = __ballot_sync(kWarpMask, inside);
    const int length = run == kWarpMask ? kWarpSize : __ffs(~run) - 1;
    position += length;
    if (length < kWarpSize) {
      return position;
    }
  }
}

/// Visit the positions of the destination row `[begin, end)` that the edge
/// group `group` of the warp `part` of the row owns, every position exactly
/// once.
///
/// A row shared by `Split` warps deals its positions in runs of `Groups`, one
/// run per warp in turn. A block of several warps, one row each, walks the
/// rows in windows of `kLockstepWindow` Å of bond length (see `Lockstep`): a
/// warp visits the positions of its row below the bound of the current
/// window, and the block crosses a barrier before the next window. A window
/// ends at the first position at or beyond its bound, so the windows partition
/// every row whatever its order, and the positions left after the last window
/// are visited after it. The windows stop at the one that holds `reach`, the
/// longest bond the radial table covers, so a bond beyond it, which no valid
/// graph holds, is visited after the last window as well. A warp without a row
/// passes empty bounds and joins the barriers.
///
/// The block walks the windows from the lowest that holds the first bond of
/// one of its rows to the highest that holds the last bond of one of them. A
/// warp visits the same positions in every window, whichever rows share its
/// block, when its row is sorted by bond length, as the canonical graphs of
/// LAMMPS and of the test builders are, and for any row while the table spans
/// at most two windows, as it does below a cutoff of 8 Å; at a longer cutoff
/// the windows of an unsorted row can follow from the rows that share its
/// block.
template <int Warps, int Groups, int Split, typename Visit>
__device__ __forceinline__ void walk_row(long begin,
                                         long end,
                                         int group,
                                         int lane,
                                         int part,
                                         const float* edge_vec,
                                         float reach,
                                         const Visit& visit) {
  constexpr float width = kLockstepWindow;
  if constexpr (Warps == 1) {
    for (long position = begin + part * Groups + group; position < end;
         position += Split * Groups) {
      visit(position);
    }
  } else {
    __shared__ int range[2];
    if (threadIdx.x == 0) {
      range[0] = INT_MAX;
      range[1] = -1;
    }
    __syncthreads();
    constexpr float inverse_width = 1.0f / width;
    if (lane == 0 && end > begin) {
      // The bound keeps the conversion in range for any length, an infinite
      // one included.
      const float last_window = reach * inverse_width;
      const auto window_of = [&](long position) {
        return static_cast<int>(fminf(
            sqrtf(bond_length_squared(edge_vec, position)) * inverse_width,
            last_window));
      };
      atomicMin(&range[0], window_of(begin));
      atomicMax(&range[1], window_of(end - 1));
    }
    __syncthreads();
    const int first = range[0];
    const int last = range[1];
    long start = begin;
    for (int window = first; window <= last; ++window) {
      const float bound = static_cast<float>(window + 1) * width;
      const long stop = window_end(start, end, bound * bound, edge_vec, lane);
      for (long position = start + group; position < stop; position += Groups) {
        visit(position);
      }
      start = stop;
      __syncthreads();
    }
    for (long position = start + group; position < end; position += Groups) {
      visit(position);
    }
  }
}

// === Split rows ===
//
// A system with fewer destination rows than the device holds resident warps
// leaves most of it idle under one warp per row, and every warp then runs
// its whole row as one latency-bound chain. The edge scans of such a system
// share each row among `Split` warps of one block instead: every warp scans
// its share of the edges into its own accumulators, the warps past the first
// publish theirs to shared memory, and the first warp adds them in warp order
// before it completes the row as a single warp would. The order of every sum
// therefore depends on `Split` alone, which is a function of the node count
// and of the occupancy the device reports for the kernel, and the result is
// reproducible on a given device.

/// Accumulators exchanged per round of a split-row merge. The exchange
/// buffer holds one round, so that it stays small next to the L1 capacity the
/// table-bound scans depend on.
constexpr int kSplitChunk = 8;

/// Shared floats of the exchange buffer of a row shared by `Split` warps.
template <int Split>
constexpr int split_buffer_floats() {
  return Split > 1 ? (Split - 1) * kSplitChunk * kWarpSize : 1;
}

/// Add the accumulators of the other warps of a shared row onto the first.
///
/// Every warp passes the same visitor, which presents its accumulators in one
/// fixed order. In rounds of `kSplitChunk` accumulators, the warps past the
/// first store theirs and the first adds them in warp order. The function
/// returns whether the calling warp carries on with the row.
template <int Split, int Count, typename Visit>
__device__ __forceinline__ bool merge_split_row(int part,
                                                int lane,
                                                float* buffer,
                                                const Visit& accumulators) {
  if constexpr (Split == 1) {
    return true;
  } else {
#pragma unroll
    for (int first = 0; first < Count; first += kSplitChunk) {
      if (part > 0) {
        int slot = 0;
        accumulators([&](float& value) {
          if (slot >= first && slot < first + kSplitChunk) {
            buffer[((part - 1) * kSplitChunk + slot - first) * kWarpSize +
                   lane] = value;
          }
          ++slot;
        });
      }
      __syncthreads();
      if (part == 0) {
        for (int other = 1; other < Split; ++other) {
          int slot = 0;
          accumulators([&](float& value) {
            if (slot >= first && slot < first + kSplitChunk) {
              value += buffer[((other - 1) * kSplitChunk + slot - first) *
                                  kWarpSize +
                              lane];
            }
            ++slot;
          });
        }
      }
      __syncthreads();
    }
    return part == 0;
  }
}

/// Synchronize the warps that share a row.
template <int Split>
__device__ __forceinline__ void sync_row() {
  if constexpr (Split > 1) {
    __syncthreads();
  } else {
    __syncwarp();
  }
}

// === Forward ===

template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          bool Bridged,
          bool Canonical,
          typename index_t,
          int Warps,
          int Split,
          Scan Fold>
__global__ __launch_bounds__(
    Warps * Split * kWarpSize,
    (slots_types(Fold)
         ? slot_resident_warps<HasModes, windowed<Warps>() ? Warps : 0>()
         : edge_resident_blocks<Channels, Lmax, HasModes, HasSpin, false>()) /
        (Warps * Split)) void forward_kernel(Arguments args) {
  using P = Profile<Channels, Lmax, HasSpin>;
  enter_edge_scan<Channels, Lmax, HasModes, HasSpin, Canonical, false, Fold>(
      args);
  constexpr bool SlotScan = slots_types(Fold);
  constexpr int EdgeWidth = P::ForwardEdgeWidth;
  constexpr int Groups = kWarpSize / EdgeWidth;
  constexpr int ChannelTiles = Channels / EdgeWidth;
  constexpr int AngularTiles = (P::C1 + EdgeWidth - 1) / EdgeWidth;
  constexpr int TensorTiles = (P::C2 + EdgeWidth - 1) / EdgeWidth;
  constexpr int HighTiles = (P::HighCount + EdgeWidth - 1) / EdgeWidth;
  static_assert(
      P::Basis == 0 || (P::C1 % EdgeWidth == 0 && P::Basis % EdgeWidth == 0),
      "the low-rank columns tile the lanes of an edge group");
  // The spin families share the neighbour width with degree two, so they
  // inherit its tiling. The single neighbour quadrupole distributes its five
  // components across lanes, exactly as the single-channel high degrees do.
  constexpr int SpinTiles = HasSpin ? TensorTiles : 0;
  constexpr int SpinTensorTiles = HasSpin ? (5 + EdgeWidth - 1) / EdgeWidth : 0;

  static_assert(Warps == 1 || Split == 1,
                "a block either holds several rows or shares one");
  const int thread = threadIdx.x % kWarpSize;
  // A block holds `Warps` rows, one warp each, or one row shared by `Split`
  // warps; `part` is the warp's share of a shared row.
  const int warp = Warps > 1 ? threadIdx.x / kWarpSize : 0;
  const int part = Split > 1 ? threadIdx.x / kWarpSize : 0;
  // Blocks of independent rows of the profiles up to degree two take their
  // rows from the last block to the first, and every other block takes its
  // rows in ascending order. The LAMMPS graph builder writes the rows in
  // ascending order, so when one run covers the system, the rows it writes
  // last are still in L2 when the forward scan starts.
  constexpr bool Windowed = windowed<Warps>();
  constexpr int WalkWarps = Windowed ? Warps : 1;
  constexpr bool Reversed = !Windowed && Lmax <= 2;
  const long block = Reversed ? static_cast<long>(gridDim.x - 1 - blockIdx.x)
                              : static_cast<long>(blockIdx.x);
  const long node = block * Warps + warp;
  // A warp past the last node of the final block owns no row: it reads node
  // zero for its constants and scans an empty row, so that it still reaches
  // every barrier of its block.
  const bool active = node < args.node_count;
  const long row_node = active ? node : 0;
  const int base_channel = thread & (EdgeWidth - 1);
  const int group = thread / EdgeWidth;
  const int leader = group * EdgeWidth;
  const unsigned mask = subwarp_mask<EdgeWidth>(leader);
  const auto* edge_index = static_cast<const index_t*>(args.edge_index);
  const auto* destination_order =
      static_cast<const index_t*>(args.destination_order);
  const int center_type =
      static_cast<int>(args.atype[args.node_begin + row_node]);
  // The centre half of every pair length scale is a node constant, like the
  // centre type it follows, and so is the length from which the window of no
  // pair of the row needs resolving.
  const float center_radius =
      Bridged ? __ldg(args.contact_radius + center_type) : 0.0f;
  const float bound = Bridged ? window_bound(args, center_radius) : 0.0f;
  const long begin = args.destination_row_ptr[row_node];
  const long end = active ? args.destination_row_ptr[row_node + 1] : begin;
  const int radial_modes = HasModes ? args.radial_modes : 0;

  float scalar[ChannelTiles] = {};
  float vector[AngularTiles][3] = {};
  float tensor[TensorTiles][5] = {};
  float high[HighTiles > 0 ? HighTiles : 1] = {};
  float scalar_mass = 0.0f;
  float angular_mass = 0.0f;
  float spin_magnitude[SpinTiles > 0 ? SpinTiles : 1] = {};
  float spin_coordination[SpinTiles > 0 ? SpinTiles : 1] = {};
  float spin_vector[SpinTiles > 0 ? SpinTiles : 1][3] = {};
  float spin_bond[SpinTiles > 0 ? SpinTiles : 1][3] = {};
  float spin_tensor[SpinTensorTiles > 0 ? SpinTensorTiles : 1] = {};

  // Explicitly aligned: the mode residual reads these rows as float4, and
  // a preceding shared array of arbitrary length would otherwise leave the
  // block on a four-byte boundary.
  __shared__ __align__(
      16) float mode_cache[Warps * Split][HasModes ? Groups * kModeStride : 1];
  float* modes = mode_cache[warp + part] + (HasModes ? group * kModeStride : 0);

  // A single-type system accumulates the bare radial features in the moment
  // registers and the extra features in `extra`, laid out as one scalar,
  // three vector and five tensor sums per feature tile.
  constexpr bool Foldable = foldable<Channels, HasSpin, Canonical>();
  constexpr int ExtraFeatures = (HasModes ? kFoldedModes : 0) + 1;
  constexpr int ExtraTiles = (ExtraFeatures + EdgeWidth - 1) / EdgeWidth;
  float extra[ExtraTiles][9] = {};
  int fold_type = -1;
  if constexpr (Foldable && !SlotScan) {
    fold_type = single_type(args, radial_modes);
  }
  const bool folded = fold_type >= 0;
  // A slotted scan accumulates the mode profiles and the basis functions of
  // the edges of each node type of its system in the slot of that type.
  static_assert(
      !SlotScan || slottable<Channels, HasModes, HasSpin, Canonical>(),
      "a slotted scan serves a profile that slots");
  constexpr bool SlotModes = SlotScan && HasModes;
  constexpr bool SlotBasis = slots_basis(Fold);
  constexpr int Slots = type_slots(Fold);
  constexpr int SlotTiles = SlotModes ? slot_tiles<EdgeWidth, Slots>() : 1;
  [[maybe_unused]] constexpr int SlotLanes = EdgeWidth / kFoldedModes;
  static_assert(
      !SlotModes || (EdgeWidth % kFoldedModes == 0 && Slots % SlotLanes == 0),
      "the modes of the slots tile the lanes of an edge group");
  [[maybe_unused]] float slot_extra[SlotTiles][9] = {};
  // The channel tiles that a scan accumulates edge by edge: every tile, or
  // for a slotted low-rank scan those below `C1`, whose FiLM stays on the
  // edges; the slots of the basis functions provide the others per row.
  constexpr int ScanTiles = SlotBasis ? P::C1 / EdgeWidth : ChannelTiles;
  constexpr int BasisTiles = SlotBasis ? P::Basis / EdgeWidth : 1;
  static_assert(!SlotBasis || (Groups <= Slots && Slots % Groups == 0 &&
                               EdgeWidth >= Slots),
                "the edge groups share the slots of the basis expansion, and "
                "the lanes of a group hold the envelope sums of the slots");
  [[maybe_unused]] float basis_slot[BasisTiles][Slots] = {};
  [[maybe_unused]] float slot_mass = 0.0f;
  TypeSlots<Slots> slots;
  if constexpr (SlotScan) {
    slots = load_type_slots<Slots>(args);
  }
  // The high degrees read the full amplitude of the leading channel, whose
  // coefficients a folded scan hoists.
  const long fixed_pair =
      static_cast<long>(center_type) * args.type_count + fold_type;
  const float2 leading_film =
      folded ? __ldg(reinterpret_cast<const float2*>(args.pair_film) +
                     fixed_pair * Channels)
             : make_float2(0.0f, 0.0f);
  const float* leading_mixing =
      HasModes && folded
          ? args.pair_mixing + fixed_pair * Channels * radial_modes
          : nullptr;

  // === Step 1. Reduce the destination row into degree-wise moments ===
  const auto scan_edge = [&](const long position, auto scan) {
    constexpr Scan Kind = decltype(scan)::value;
    constexpr bool Folded = Kind == Scan::Direct || Kind == Scan::LowRank;
    constexpr int FoldColumns = fold_columns<P, Kind>();
    constexpr int FoldTiles = FoldColumns / EdgeWidth;
    // Channel tiles whose FiLM amplitude the edges carry.
    constexpr int EdgeTiles =
        slots_basis(Kind) ? P::C1 / EdgeWidth : ChannelTiles;
    const long edge = edge_at_position<Canonical>(position, destination_order);
    if (args.edge_mask != nullptr && !args.edge_mask[edge]) {
      return;
    }
    // Every lane of the group reloads the shared edge state instead of
    // broadcasting it from a leader. The addresses are identical inside the
    // group, so the memory system serves one transaction either way, whereas
    // a leader-only branch pays the same issue slots and adds ten shuffles.
    const long source = static_cast<long>(edge_index[edge]);
    const int source_type = static_cast<int>(args.atype[source]);
    if constexpr (!Canonical) {
      if (center_type >= args.type_count - 1 ||
          source_type >= args.type_count - 1) {
        return;
      }
    }
    const EdgeGeometry geometry = load_geometry<Bridged>(
        edge, source, source_type, args.rcut, args.f_inner, args.f_outer,
        args.inverse_f_width, center_radius, bound, args.contact_radius,
        args.eps, args.edge_vec);
    const TableLocation location = locate_table<Canonical>(
        geometry.seen, args.table_stride, args.table_max, args.interval_count);
    const TableRow row =
        reads_low_rank(Kind)
            ? table_row(args.basis_table, location, FoldColumns + radial_modes)
            : table_row(args.table, location, args.table_width);
    const float coordinate = location.coordinate;
    float basis[9];
    fill_angular_basis(geometry, basis);
    const float envelope = geometry.envelope;
    const float squared = envelope * envelope;
    // Angular payload of the leading channel, held by the leader of the group,
    // and for a slotted scan its amplitude before the mode mixing.
    float leading_payload = 0.0f;
    [[maybe_unused]] float leading_value = 0.0f;

    if constexpr (Folded) {
      float leading = 0.0f;
      // A column below ``C1`` is the channel of that index; a low-rank table
      // continues with the basis functions, which feed degree zero alone.
#pragma unroll
      for (int tile = 0; tile < FoldTiles; ++tile) {
        const int channel = base_channel + tile * EdgeWidth;
        const float radial = evaluate_table(row, channel, coordinate);
        scalar[tile] = fmaf(radial, envelope, scalar[tile]);
        const float angular = radial * squared;
        if (channel < P::C1) {
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            vector[tile][component] =
                fmaf(angular, basis[1 + component], vector[tile][component]);
          }
        }
        if (channel < P::C2) {
#pragma unroll
          for (int component = 0; component < 5; ++component) {
            tensor[tile][component] =
                fmaf(angular, basis[4 + component], tensor[tile][component]);
          }
        }
        if (tile == 0) {
          leading = radial;
        }
      }
      float feature_value[ExtraTiles] = {};
#pragma unroll
      for (int tile = 0; tile < ExtraTiles; ++tile) {
        if (tile * EdgeWidth <= radial_modes) {
          const float value =
              extra_feature(row, FoldColumns, base_channel + tile * EdgeWidth,
                            radial_modes, coordinate);
          feature_value[tile] = value;
          extra[tile][0] = fmaf(value, envelope, extra[tile][0]);
          const float angular = value * squared;
#pragma unroll
          for (int component = 1; component < 9; ++component) {
            extra[tile][component] =
                fmaf(angular, basis[component], extra[tile][component]);
          }
        }
      }
      if constexpr (Lmax >= 3) {
        float film_value = fmaf(leading_film.x, leading, leading_film.y);
        if constexpr (HasModes) {
#pragma unroll
          for (int mode = 0; mode < kFoldedModes; ++mode) {
            if (mode < radial_modes) {
              const float profile =
                  __shfl_sync(mask, feature_value[mode / EdgeWidth],
                              leader + mode % EdgeWidth);
              film_value =
                  fmaf(__ldg(leading_mixing + mode), profile, film_value);
            }
          }
        }
        leading_payload = film_value * squared;
      }
    } else {
      if constexpr (HasModes && Kind == Scan::Pairwise) {
        __syncwarp(mask);
        for (int mode = base_channel; mode < radial_modes; mode += EdgeWidth) {
          modes[mode] = evaluate_table(row, Channels + mode, coordinate);
        }
        __syncwarp(mask);
      }
      const long pair = static_cast<long>(center_type) * args.type_count +
                        geometry.source_type;
      const float2* film_row = reinterpret_cast<const float2*>(
                                   args.pair_film + pair * Channels * 2) +
                               base_channel;
      [[maybe_unused]] const float* mixing_row =
          HasModes ? args.pair_mixing +
                         (pair * Channels + base_channel) * radial_modes
                   : nullptr;

      // The spin payload is hoisted out of the channel loop: every channel
      // multiplies the same conditioned neighbour moment and the same
      // projection of that moment onto the bond, and only the amplitude varies
      // with the channel.
      [[maybe_unused]] float neighbor_spin[3] = {0.0f, 0.0f, 0.0f};
      [[maybe_unused]] float neighbor_magnitude = 0.0f;
      [[maybe_unused]] float neighbor_gate = 0.0f;
      [[maybe_unused]] float bond_alignment = 0.0f;
      [[maybe_unused]] const float2* spin_row = nullptr;
      if constexpr (HasSpin) {
        const SpinTypeWeights weights =
            load_spin_type(args.spin_type, geometry.source_type);
        load_conditioned_spin(args.spin, weights.scale, geometry.source,
                              neighbor_spin);
        neighbor_magnitude = neighbor_spin[0] * neighbor_spin[0] +
                             neighbor_spin[1] * neighbor_spin[1] +
                             neighbor_spin[2] * neighbor_spin[2];
        neighbor_gate = weights.gate;
        // Component of the neighbour moment along the bond. The bond-projected
        // family is this scalar times the unit direction, which is
        // `basis[1..3]`.
        bond_alignment = neighbor_spin[0] * basis[1] +
                         neighbor_spin[1] * basis[2] +
                         neighbor_spin[2] * basis[3];
        spin_row =
            reinterpret_cast<const float2*>(args.spin_pair + pair * P::Cs * 2) +
            base_channel;
      }

      [[maybe_unused]] float spin_zero = 0.0f;
#pragma unroll
      for (int tile = 0; tile < EdgeTiles; ++tile) {
        const int channel = base_channel + tile * EdgeWidth;
        const float radial = evaluate_table(row, channel, coordinate);
        const float2 film = __ldg(film_row + tile * EdgeWidth);
        float film_value = fmaf(film.x, radial, film.y);
        if constexpr (HasModes && Kind == Scan::Pairwise) {
          accumulate_modes(mixing_row + tile * (EdgeWidth * radial_modes),
                           modes, radial_modes, film_value);
        }
        // Degree zero carries one envelope factor; every non-scalar degree
        // carries a second one.
        const float amplitude = film_value * envelope;
        const float angular = amplitude * envelope;
        scalar[tile] += amplitude;
        if (channel < P::C1) {
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            vector[tile][component] =
                fmaf(angular, basis[1 + component], vector[tile][component]);
          }
        }
        if (channel < P::C2) {
#pragma unroll
          for (int component = 0; component < 5; ++component) {
            tensor[tile][component] =
                fmaf(angular, basis[4 + component], tensor[tile][component]);
          }
        }
        if constexpr (HasSpin) {
          if (channel < P::Cs) {
            const float2 spin_film = __ldg(spin_row + tile * EdgeWidth);
            const float weight =
                fmaf(spin_film.x, radial, spin_film.y) * envelope * envelope;
            spin_magnitude[tile] =
                fmaf(weight, neighbor_magnitude, spin_magnitude[tile]);
            spin_coordination[tile] =
                fmaf(weight, neighbor_gate, spin_coordination[tile]);
            const float projected = weight * bond_alignment;
#pragma unroll
            for (int component = 0; component < 3; ++component) {
              spin_vector[tile][component] =
                  fmaf(weight, neighbor_spin[component],
                       spin_vector[tile][component]);
              spin_bond[tile][component] = fmaf(projected, basis[1 + component],
                                                spin_bond[tile][component]);
            }
            if (channel == 0) {
              spin_zero = weight;
            }
          }
        }
        if (channel == 0) {
          leading_payload = angular;
          leading_value = film_value;
        }
      }
      if constexpr (HasSpin) {
        // The single neighbour quadrupole reads the leading spin channel only,
        // so its five components are distributed across the lanes of the edge
        // group and each is evaluated on the one lane that owns it.
        const float weight = __shfl_sync(mask, spin_zero, leader);
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          if (component % EdgeWidth == base_channel) {
            spin_tensor[component / EdgeWidth] = fmaf(
                weight,
                spin_quadrupole_component(neighbor_spin[0], neighbor_spin[1],
                                          neighbor_spin[2], component),
                spin_tensor[component / EdgeWidth]);
          }
        }
      }
      if constexpr (slots_types(Kind)) {
        const int slot = type_slot(slots, geometry.source_type);
        if constexpr (HasModes) {
          // The bare profile of the mode of the lane enters the slot of the
          // source type, on the lanes that own that slot.
          const int mode = base_channel % kFoldedModes;
          const float profile =
              mode < radial_modes
                  ? evaluate_table(row, FoldColumns + mode, coordinate)
                  : 0.0f;
          const float angular = profile * squared;
#pragma unroll
          for (int tile = 0; tile < SlotTiles; ++tile) {
            if (owned_slot<EdgeWidth>(base_channel, tile) == slot) {
              slot_extra[tile][0] =
                  fmaf(profile, envelope, slot_extra[tile][0]);
#pragma unroll
              for (int component = 1; component < 9; ++component) {
                slot_extra[tile][component] = fmaf(angular, basis[component],
                                                   slot_extra[tile][component]);
              }
            }
          }
          if constexpr (Lmax >= 3) {
            // The high degrees read the full amplitude of the leading
            // channel, which the leader completes with the mode mixing of
            // the pair; lane `leader + r` of the group holds the profile of
            // mode `r`.
            const float* pair_modes =
                args.pair_mixing + pair * Channels * radial_modes;
            float value = leading_value;
#pragma unroll
            for (int mixed = 0; mixed < kFoldedModes; ++mixed) {
              if (mixed < radial_modes) {
                value = fmaf(__ldg(pair_modes + mixed),
                             __shfl_sync(mask, profile, leader + mixed), value);
              }
            }
            leading_payload = value * squared;
          }
        }
        if constexpr (slots_basis(Kind)) {
          // The bare basis functions of the lane enter the slot of the source
          // type, and the envelope enters it on the lane of that slot. Every
          // slot takes the product with its weight, the envelope for the
          // slot of the edge and zero for the others, which adds the exact
          // zero and keeps the accumulation free of branches.
          float weight[Slots];
#pragma unroll
          for (int entry = 0; entry < Slots; ++entry) {
            weight[entry] = entry == slot ? envelope : 0.0f;
          }
#pragma unroll
          for (int tile = 0; tile < BasisTiles; ++tile) {
            const float radial = evaluate_table(
                row, P::C1 + base_channel + tile * EdgeWidth, coordinate);
#pragma unroll
            for (int entry = 0; entry < Slots; ++entry) {
              basis_slot[tile][entry] =
                  fmaf(radial, weight[entry], basis_slot[tile][entry]);
            }
          }
          slot_mass += base_channel == slot ? envelope : 0.0f;
        }
      }
    }
    if constexpr (Lmax >= 3) {
      // Degrees three and above read only the leading channel, so their
      // components are distributed across the lanes of the edge group. The
      // component is iterated at compile time and selected with a lane
      // predicate, which folds the harmonic selector and keeps the tile index
      // constant.
      const float amplitude = __shfl_sync(mask, leading_payload, leader);
#pragma unroll
      for (int component = 0; component < P::HighCount; ++component) {
        if (component % EdgeWidth == base_channel) {
          high[component / EdgeWidth] =
              fmaf(amplitude, high_basis_value(geometry, component),
                   high[component / EdgeWidth]);
        }
      }
    }
    if (base_channel == 0) {
      scalar_mass += squared;
      angular_mass = fmaf(squared, squared, angular_mass);
    }
  };
  const auto pairwise_edge = [&](const long position) {
    scan_edge(position, ScanTag<Scan::Pairwise>{});
  };
  // The first warp of a shared row requests the lines of the row for every
  // warp of the row.
  if constexpr (Canonical) {
    if (part == 0) {
      prefetch_row_bonds(args, begin, end, thread);
      if constexpr (SlotScan) {
        prefetch_row_sources(args, edge_index, begin, end, thread);
      }
    }
  }
  if constexpr (SlotScan) {
    walk_row<WalkWarps, Groups, Split>(
        begin, end, group, thread, part, args.edge_vec, args.table_max,
        [&](const long position) { scan_edge(position, ScanTag<Fold>{}); });
  } else if constexpr (Foldable) {
    if (folded) {
      walk_row<WalkWarps, Groups, Split>(
          begin, end, group, thread, part, args.edge_vec, args.table_max,
          [&](const long position) { scan_edge(position, ScanTag<Fold>{}); });
    } else {
      walk_row<WalkWarps, Groups, Split>(begin, end, group, thread, part,
                                         args.edge_vec, args.table_max,
                                         pairwise_edge);
    }
  } else {
    walk_row<WalkWarps, Groups, Split>(begin, end, group, thread, part,
                                       args.edge_vec, args.table_max,
                                       pairwise_edge);
  }
  if (!active) {
    return;
  }
  // === Step 1b. Merge the warps that share the row ===
  // The accumulators that `accumulators` visits, in its order.
  constexpr int Accumulators = ScanTiles + 3 * AngularTiles + 5 * TensorTiles +
                               (Lmax >= 3 ? HighTiles : 0) + 2 + 8 * SpinTiles +
                               SpinTensorTiles +
                               (Foldable && !SlotScan ? 9 * ExtraTiles : 0) +
                               (SlotModes ? 9 * SlotTiles : 0) +
                               (SlotBasis ? BasisTiles * Slots + 1 : 0);
  __shared__ float split_buffer[split_buffer_floats<Split>()];
  const auto accumulators = [&](auto&& visit) {
#pragma unroll
    for (int tile = 0; tile < ScanTiles; ++tile) {
      visit(scalar[tile]);
    }
#pragma unroll
    for (int tile = 0; tile < AngularTiles; ++tile) {
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        visit(vector[tile][component]);
      }
    }
#pragma unroll
    for (int tile = 0; tile < TensorTiles; ++tile) {
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        visit(tensor[tile][component]);
      }
    }
    if constexpr (Lmax >= 3) {
#pragma unroll
      for (int tile = 0; tile < HighTiles; ++tile) {
        visit(high[tile]);
      }
    }
    visit(scalar_mass);
    visit(angular_mass);
    if constexpr (HasSpin) {
#pragma unroll
      for (int tile = 0; tile < SpinTiles; ++tile) {
        visit(spin_magnitude[tile]);
        visit(spin_coordination[tile]);
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          visit(spin_vector[tile][component]);
          visit(spin_bond[tile][component]);
        }
      }
#pragma unroll
      for (int tile = 0; tile < SpinTensorTiles; ++tile) {
        visit(spin_tensor[tile]);
      }
    }
    if constexpr (Foldable && !SlotScan) {
#pragma unroll
      for (int tile = 0; tile < ExtraTiles; ++tile) {
#pragma unroll
        for (int component = 0; component < 9; ++component) {
          visit(extra[tile][component]);
        }
      }
    }
    if constexpr (SlotModes) {
#pragma unroll
      for (int tile = 0; tile < SlotTiles; ++tile) {
#pragma unroll
        for (int component = 0; component < 9; ++component) {
          visit(slot_extra[tile][component]);
        }
      }
    }
    if constexpr (SlotBasis) {
#pragma unroll
      for (int tile = 0; tile < BasisTiles; ++tile) {
#pragma unroll
        for (int entry = 0; entry < Slots; ++entry) {
          visit(basis_slot[tile][entry]);
        }
      }
      visit(slot_mass);
    }
  };
  if (!merge_split_row<Split, Accumulators>(part, thread, split_buffer,
                                            accumulators)) {
    return;
  }

  // === Step 2. Merge the concurrent edge groups and normalize ===
  if constexpr (Groups > 1) {
#pragma unroll
    for (int tile = 0; tile < ScanTiles; ++tile) {
      scalar[tile] = reduce_channel_groups<EdgeWidth>(scalar[tile]);
    }
#pragma unroll
    for (int tile = 0; tile < AngularTiles; ++tile) {
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        vector[tile][component] =
            reduce_channel_groups<EdgeWidth>(vector[tile][component]);
      }
    }
#pragma unroll
    for (int tile = 0; tile < TensorTiles; ++tile) {
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        tensor[tile][component] =
            reduce_channel_groups<EdgeWidth>(tensor[tile][component]);
      }
    }
    if constexpr (Lmax >= 3) {
#pragma unroll
      for (int tile = 0; tile < HighTiles; ++tile) {
        high[tile] = reduce_channel_groups<EdgeWidth>(high[tile]);
      }
    }
    if constexpr (HasSpin) {
#pragma unroll
      for (int tile = 0; tile < SpinTiles; ++tile) {
        spin_magnitude[tile] =
            reduce_channel_groups<EdgeWidth>(spin_magnitude[tile]);
        spin_coordination[tile] =
            reduce_channel_groups<EdgeWidth>(spin_coordination[tile]);
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          spin_vector[tile][component] =
              reduce_channel_groups<EdgeWidth>(spin_vector[tile][component]);
          spin_bond[tile][component] =
              reduce_channel_groups<EdgeWidth>(spin_bond[tile][component]);
        }
      }
#pragma unroll
      for (int tile = 0; tile < SpinTensorTiles; ++tile) {
        spin_tensor[tile] = reduce_channel_groups<EdgeWidth>(spin_tensor[tile]);
      }
    }
  }
  if (folded) {
    // After the merge of the edge groups, every lane holds the row totals of
    // its channels, and the lane of each extra feature its sums. The
    // coefficients of the pair turn the bare sums into the moments of the
    // amplitude.
#pragma unroll
    for (int tile = 0; tile < ExtraTiles; ++tile) {
#pragma unroll
      for (int component = 0; component < 9; ++component) {
        extra[tile][component] =
            reduce_channel_groups<EdgeWidth>(extra[tile][component]);
      }
    }
    const long pair =
        static_cast<long>(center_type) * args.type_count + fold_type;
    const float2* film_row =
        reinterpret_cast<const float2*>(args.pair_film + pair * Channels * 2) +
        base_channel;
    if constexpr (Fold == Scan::LowRank) {
      expand_basis<P, EdgeWidth>(scalar, args.basis_matrix, base_channel);
    }
    float sums[9];
    gather_extra_feature<EdgeWidth>(extra, radial_modes, sums);
#pragma unroll
    for (int tile = 0; tile < ChannelTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      const float2 film = __ldg(film_row + tile * EdgeWidth);
      scalar[tile] = fmaf(film.x, scalar[tile], film.y * sums[0]);
      if (channel < P::C1) {
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          vector[tile][component] = fmaf(film.x, vector[tile][component],
                                         film.y * sums[1 + component]);
        }
      }
      if (channel < P::C2) {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          tensor[tile][component] = fmaf(film.x, tensor[tile][component],
                                         film.y * sums[4 + component]);
        }
      }
    }
    if constexpr (HasModes) {
      const float* mixing_row =
          args.pair_mixing + (pair * Channels + base_channel) * radial_modes;
      for (int mode = 0; mode < radial_modes; ++mode) {
        gather_extra_feature<EdgeWidth>(extra, mode, sums);
#pragma unroll
        for (int tile = 0; tile < ChannelTiles; ++tile) {
          const int channel = base_channel + tile * EdgeWidth;
          const float weight =
              __ldg(mixing_row + tile * EdgeWidth * radial_modes + mode);
          scalar[tile] = fmaf(weight, sums[0], scalar[tile]);
          if (channel < P::C1) {
#pragma unroll
            for (int component = 0; component < 3; ++component) {
              vector[tile][component] =
                  fmaf(weight, sums[1 + component], vector[tile][component]);
            }
          }
          if (channel < P::C2) {
#pragma unroll
            for (int component = 0; component < 5; ++component) {
              tensor[tile][component] =
                  fmaf(weight, sums[4 + component], tensor[tile][component]);
            }
          }
        }
      }
    }
  }
  if constexpr (SlotBasis) {
    // After the merge of the edge groups every group holds the basis sums of
    // the slots. Edge group `g` expands those of slots `g + j * Groups` into
    // the bare channel sums from `C1` on, in increasing order of the basis
    // functions, as a single-type scan does, and the FiLM scale and shift of
    // the pair of the slot turn them into its share of the moments; the merge
    // of the groups then adds the shares of the slots.
    constexpr int Direct = P::C1 / EdgeWidth;
    constexpr int Expanded = P::C0 - P::C1;
#pragma unroll
    for (int tile = 0; tile < BasisTiles; ++tile) {
#pragma unroll
      for (int entry = 0; entry < Slots; ++entry) {
        basis_slot[tile][entry] =
            reduce_channel_groups<EdgeWidth>(basis_slot[tile][entry]);
      }
    }
    slot_mass = reduce_channel_groups<EdgeWidth>(slot_mass);
    float share[ChannelTiles - Direct] = {};
#pragma unroll
    for (int round = 0; round < Slots / Groups; ++round) {
      const int slot = group + round * Groups;
      float sums[BasisTiles];
#pragma unroll
      for (int tile = 0; tile < BasisTiles; ++tile) {
        sums[tile] = basis_slot[tile][0];
#pragma unroll
        for (int entry = 1; entry < Slots; ++entry) {
          if (entry == slot) {
            sums[tile] = basis_slot[tile][entry];
          }
        }
      }
      const float mass = __shfl_sync(kWarpMask, slot_mass, leader + slot);
      float expanded[ChannelTiles - Direct] = {};
#pragma unroll
      for (int k = 0; k < P::Basis; ++k) {
        const float value =
            __shfl_sync(kWarpMask, sums[k / EdgeWidth], leader + k % EdgeWidth);
        const float* row = args.basis_matrix + k * Expanded + base_channel;
#pragma unroll
        for (int tile = 0; tile < ChannelTiles - Direct; ++tile) {
          expanded[tile] =
              fmaf(__ldg(row + tile * EdgeWidth), value, expanded[tile]);
        }
      }
      if (slot < slots.count) {
        const long pair = static_cast<long>(center_type) * args.type_count +
                          slot_type(slots, slot);
        const float2* film_row = reinterpret_cast<const float2*>(
                                     args.pair_film + pair * Channels * 2) +
                                 base_channel;
#pragma unroll
        for (int tile = 0; tile < ChannelTiles - Direct; ++tile) {
          const float2 film = __ldg(film_row + (Direct + tile) * EdgeWidth);
          share[tile] += fmaf(film.x, expanded[tile], film.y * mass);
        }
      }
    }
#pragma unroll
    for (int tile = 0; tile < ChannelTiles - Direct; ++tile) {
      scalar[Direct + tile] = reduce_channel_groups<EdgeWidth>(share[tile]);
    }
  }
  if constexpr (SlotModes) {
    // After the merge of the edge groups every group holds the moments of the
    // slots, and the mode mixing of the pair of each slot turns them into the
    // mode share of the moments of every channel.
#pragma unroll
    for (int tile = 0; tile < SlotTiles; ++tile) {
#pragma unroll
      for (int component = 0; component < 9; ++component) {
        slot_extra[tile][component] =
            reduce_channel_groups<EdgeWidth>(slot_extra[tile][component]);
      }
    }
#pragma unroll
    for (int slot = 0; slot < Slots; ++slot) {
      if (slot < slots.count) {
        const long pair = static_cast<long>(center_type) * args.type_count +
                          slot_type(slots, slot);
        const float* mixing_row =
            args.pair_mixing + (pair * Channels + base_channel) * radial_modes;
#pragma unroll
        for (int mode = 0; mode < kFoldedModes; ++mode) {
          if (mode < radial_modes) {
            float sums[9];
#pragma unroll
            for (int component = 0; component < 9; ++component) {
              sums[component] = __shfl_sync(
                  kWarpMask, slot_extra[slot / SlotLanes][component],
                  slot_owner<EdgeWidth>(slot, mode));
            }
#pragma unroll
            for (int tile = 0; tile < ChannelTiles; ++tile) {
              const int channel = base_channel + tile * EdgeWidth;
              const float weight =
                  __ldg(mixing_row + tile * EdgeWidth * radial_modes + mode);
              scalar[tile] = fmaf(weight, sums[0], scalar[tile]);
              if (channel < P::C1) {
#pragma unroll
                for (int component = 0; component < 3; ++component) {
                  vector[tile][component] = fmaf(weight, sums[1 + component],
                                                 vector[tile][component]);
                }
              }
              if (channel < P::C2) {
#pragma unroll
                for (int component = 0; component < 5; ++component) {
                  tensor[tile][component] = fmaf(weight, sums[4 + component],
                                                 tensor[tile][component]);
                }
              }
            }
          }
        }
      }
    }
  }
  // Both masses are broadcast from lane zero, so every lane forms the same
  // normalizers.
  const float scalar_total = warp_sum(scalar_mass) + args.degree_floor;
  const float angular_total = warp_sum(angular_mass) + args.degree_floor;
  const float scalar_norm = rsqrtf(scalar_total);
  const float angular_norm = rsqrtf(angular_total);
  if (thread < 2) {
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputDivisor + thread,
                     sqrtf(thread == 0 ? scalar_total : angular_total));
  }

  // === Step 3. Publish the normalized moments as the saved state ===
  // The lanes of every edge group hold the merged moments after the
  // reduction, so the leading group stores them in the flat state layout.
  float* const moments = args.state_out + node * P::StateWidth;
  if (thread < EdgeWidth) {
#pragma unroll
    for (int tile = 0; tile < ChannelTiles; ++tile) {
      moments[P::ScalarOffset + base_channel + tile * EdgeWidth] =
          scalar[tile] * scalar_norm;
    }
#pragma unroll
    for (int tile = 0; tile < AngularTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      if (channel < P::C1) {
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          moments[P::VectorOffset + component * P::C1 + channel] =
              vector[tile][component] * angular_norm;
        }
      }
    }
#pragma unroll
    for (int tile = 0; tile < TensorTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      if (channel < P::C2) {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          moments[P::TensorOffset + component * P::C2 + channel] =
              tensor[tile][component] * angular_norm;
        }
      }
    }
    if constexpr (Lmax >= 3) {
#pragma unroll
      for (int component = 0; component < P::HighCount; ++component) {
        if (component % EdgeWidth == base_channel) {
          moments[P::HighOffset + component] =
              high[component / EdgeWidth] * angular_norm;
        }
      }
    }
    if constexpr (HasSpin) {
      // The reduced spin families carry the same squared envelope as every
      // non-scalar geometric moment and therefore share its normalizer.
#pragma unroll
      for (int tile = 0; tile < SpinTiles; ++tile) {
        const int channel = base_channel + tile * EdgeWidth;
        if (channel < P::Cs) {
          moments[P::SpinMagnitude + channel] =
              spin_magnitude[tile] * angular_norm;
          moments[P::SpinCoordination + channel] =
              spin_coordination[tile] * angular_norm;
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            moments[P::SpinVector + component * P::Cs + channel] =
                spin_vector[tile][component] * angular_norm;
            moments[P::SpinBond + component * P::Cs + channel] =
                spin_bond[tile][component] * angular_norm;
          }
        }
      }
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        if (component % EdgeWidth == base_channel) {
          moments[P::SpinTensor + component] =
              spin_tensor[component / EdgeWidth] * angular_norm;
        }
      }
    }
  }
  if constexpr (HasSpin) {
    // The on-site families are node local. They are written after the
    // division so that an invariant pairing an on-site channel with a
    // neighbour channel carries exactly one neighborhood normalizer.
    if (thread == 0) {
      const SpinTypeWeights weights =
          load_spin_type(args.spin_type, center_type);
      float center_spin[3];
      load_conditioned_spin(args.spin, weights.scale, args.node_begin + node,
                            center_spin);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        moments[P::SpinOnsiteVector + component] =
            weights.vector * center_spin[component];
      }
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        moments[P::SpinOnsiteTensor + component] =
            weights.quadrupole *
            spin_quadrupole_component(center_spin[0], center_spin[1],
                                      center_spin[2], component);
      }
    }
  }
  if (thread == 0) {
    moments[P::MomentWidth] = scalar_norm;
    moments[P::MomentWidth + 1] = angular_norm;
  }
}

// === Forward readout ===

// The invariant readout of the nodes whose moments the edge scan published:
// the degree alignment, the probe projections and every invariant block of
// the descriptor except the two divisors, which the scan emits itself. One
// warp owns one node, as in the scan. The readout is a launch of its own
// because the radial table rows that bound the scan are served by L1, whose
// capacity follows the carveout the scan's shared memory selects; the readout
// buffers of a resident node would double that carveout at full occupancy.
template <int Channels, int Lmax, bool HasSpin>
__global__ __launch_bounds__(kWarpSize) void forward_readout_kernel(
    Arguments args) {
  using P = Profile<Channels, Lmax, HasSpin>;
  const int thread = threadIdx.x;
  const long node = blockIdx.x;
  if (node >= args.node_count) {
    return;
  }
  const int center_type = static_cast<int>(args.atype[args.node_begin + node]);
  __shared__ float moments[P::MomentWidth];
  stage_shared<P::MomentWidth, P::Threads>(
      moments, args.state_out + node * P::StateWidth, thread);
  __syncthreads();

  // === Step 1. Align and project the wide degrees ===
  __shared__ float aligned[P::AlignedWidth];
  if (thread < P::C1) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C1; ++input) {
        value = fmaf(moments[P::VectorOffset + component * P::C1 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 0,
                                                    input, thread),
                     value);
      }
      aligned[component * P::C1 + thread] = value;
    }
  }
  if (thread < P::C2) {
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C2; ++input) {
        value = fmaf(moments[P::TensorOffset + component * P::C2 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 2,
                                                    input, thread),
                     value);
      }
      aligned[3 * P::C1 + component * P::C2 + thread] = value;
    }
  }
  __syncthreads();

  __shared__ float probes[P::ProbeWidth];
  if (thread < P::K1) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C1; ++input) {
        value = fmaf(aligned[component * P::C1 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 4,
                                                    input, thread),
                     value);
      }
      probes[component * P::K1 + thread] = value;
    }
  }
  if (thread < P::K2) {
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C2; ++input) {
        value = fmaf(aligned[3 * P::C1 + component * P::C2 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 6,
                                                    input, thread),
                     value);
      }
      probes[3 * P::K1 + component * P::K2 + thread] = value;
    }
  }
  __syncthreads();

  // === Step 2. Emit the invariant blocks ===
  for (int channel = thread; channel < Channels; channel += P::Threads) {
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputScalar + channel,
                     moments[P::ScalarOffset + channel]);
    store_descriptor(
        args.descriptor, args.output_mean, args.output_inv_std, node,
        P::OutputWidth, P::OutputType + channel,
        __ldg(args.type_embedding + static_cast<long>(center_type) * Channels +
              channel));
  }

  for (int pair = thread; pair < P::Gram1; pair += P::Threads) {
    int row, column;
    decode_upper_pair(pair, P::C1, row, column);
    float value = 0.0f;
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      value = fmaf(aligned[component * P::C1 + row],
                   aligned[component * P::C1 + column], value);
    }
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputGram1 + pair,
                     (row == column ? 1.0f : kSqrtTwo) * value);
  }
  for (int pair = thread; pair < P::Gram2; pair += P::Threads) {
    int row, column;
    decode_upper_pair(pair, P::C2, row, column);
    float value = 0.0f;
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      value = fmaf(aligned[3 * P::C1 + component * P::C2 + row],
                   aligned[3 * P::C1 + component * P::C2 + column], value);
    }
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputGram2 + pair,
                     (row == column ? 1.0f : kSqrtTwo) * value);
  }
  if constexpr (Lmax >= 3) {
    if (thread < Lmax - 2) {
      const int offset = P::HighOffset + (thread == 0 ? 0 : P::High3);
      const int count = 2 * (3 + thread) + 1;
      float value = 0.0f;
      for (int component = 0; component < count; ++component) {
        const float moment = moments[offset + component];
        value = fmaf(moment, moment, value);
      }
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputGram3 + thread, value);
    }
  }

  for (int output = thread; output < P::Bis112; output += P::Threads) {
    const int tensor_index = output % P::K2;
    int first, second;
    decode_upper_pair(output / P::K2, P::K1, first, second);
    float packed[5];
    float left[3];
    float right[3];
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      packed[component] = probes[3 * P::K1 + component * P::K2 + tensor_index];
    }
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      left[component] = probes[component * P::K1 + first];
      right[component] = probes[component * P::K1 + second];
    }
    float product[3];
    matrix_vector(packed_to_stf(packed), right, product);
    const float value =
        -kInvSqrtFive *
        (left[0] * product[0] + left[1] * product[1] + left[2] * product[2]);
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputBis112 + output,
                     (first == second ? 1.0f : kSqrtTwo) * value);
  }

  if (thread < P::Bis222) {
    constexpr int entries[4][3] = {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    constexpr float scales[4] = {1.0f, kSqrtThree, kSqrtThree, 1.0f};
    Matrix3 matrices[3];
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      float packed[5];
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        packed[component] =
            probes[3 * P::K1 + component * P::K2 + entries[thread][axis]];
      }
      matrices[axis] = packed_to_stf(packed);
    }
    const float value =
        kBis222Scale *
        matrix_trace(matrix_product(matrix_product(matrices[0], matrices[1]),
                                    matrices[2]));
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputBis222 + thread,
                     scales[thread] * value);
  }

  if constexpr (Lmax >= 3) {
    // The terms of one output are dealt to the lanes of the warp, and one
    // reduction per output completes it.
    for (int record = 0; record < args.coupling_count; ++record) {
      const int* meta = args.coupling_meta + record * 8;
      const int degree_1 = meta[0];
      const int degree_2 = meta[1];
      const int degree_3 = meta[2];
      const int nonzero_begin = meta[3];
      const int nonzero_count = meta[4];
      const int probe_begin = meta[5];
      const int probe_count = meta[6];
      const int coordinate = meta[7];
      for (int output = 0; output < probe_count; ++output) {
        const int selection = __ldg(args.coupling_entry + probe_begin + output);
        const int index_1 = selection & 0xFF;
        const int index_2 = (selection >> 8) & 0xFF;
        const int index_3 = (selection >> 16) & 0xFF;
        float value = 0.0f;
        for (int term = thread; term < nonzero_count; term += P::Threads) {
          const int components =
              __ldg(args.coupling_entry + nonzero_begin + term);
          const float weight =
              __ldg(args.coupling_value + nonzero_begin + term);
          const float first = probe_value<Channels, Lmax>(
              probes, moments, degree_1, components & 0xFF, index_1);
          const float second = probe_value<Channels, Lmax>(
              probes, moments, degree_2, (components >> 8) & 0xFF, index_2);
          const float third = probe_value<Channels, Lmax>(
              probes, moments, degree_3, (components >> 16) & 0xFF, index_3);
          value = fmaf(weight * first * second, third, value);
        }
        value = warp_sum(value);
        if (thread == 0) {
          store_descriptor(
              args.descriptor, args.output_mean, args.output_inv_std, node,
              P::OutputWidth, coordinate + output,
              value * __ldg(args.coupling_value + probe_begin + output));
        }
      }
    }
  }

  for (int output = thread; output < P::Quartic; output += P::Threads) {
    const int vector_index = output % P::K1;
    const int tensor_index = output / P::K1;
    float packed[5];
    float value[3];
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      packed[component] = probes[3 * P::K1 + component * P::K2 + tensor_index];
    }
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      value[component] = probes[component * P::K1 + vector_index];
    }
    float product[3];
    matrix_vector(packed_to_stf(packed), value, product);
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputQuartic + output,
                     product[0] * product[0] + product[1] * product[1] +
                         product[2] * product[2]);
  }

  // === Step 3. Emit the spin invariants of even spin order ===
  if constexpr (HasSpin) {
    for (int pair = thread; pair < P::SpinGramVector; pair += P::Threads) {
      int row, column;
      decode_upper_pair(pair, P::SpinVectorWidth, row, column);
      float value = 0.0f;
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        value =
            fmaf(spin_vector_value<Channels, Lmax>(moments, component, row),
                 spin_vector_value<Channels, Lmax>(moments, component, column),
                 value);
      }
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputSpinGramVector + pair,
                       (row == column ? 1.0f : kSqrtTwo) * value);
    }
    if (thread < P::SpinGramTensor) {
      int row, column;
      decode_spin_tensor_pair(thread, row, column);
      float value = 0.0f;
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        value =
            fmaf(spin_tensor_value<Channels, Lmax>(moments, component, row),
                 spin_tensor_value<Channels, Lmax>(moments, component, column),
                 value);
      }
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputSpinGramTensor + thread,
                       (row == column ? 1.0f : kSqrtTwo) * value);
    }
    // Cross Gram against the unaligned geometric degree-two moments. Both
    // factors have even spin order, and with a unit direction the on-site row
    // evaluates to the single-ion anisotropy sum over neighbours.
    for (int output = thread; output < P::SpinCross; output += P::Threads) {
      const int probe = output / P::C2;
      const int channel = output % P::C2;
      float value = 0.0f;
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        value =
            fmaf(spin_tensor_value<Channels, Lmax>(moments, component, probe),
                 moments[P::TensorOffset + component * P::C2 + channel], value);
      }
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputSpinCross + output,
                       value);
    }
    for (int channel = thread; channel < P::Cs; channel += P::Threads) {
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputSpinMagnitude + channel,
                       moments[P::SpinMagnitude + channel]);
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth,
                       P::OutputSpinCoordination + channel,
                       moments[P::SpinCoordination + channel]);
    }
  }
}

// === Node readout backward ===

// The four symmetric 222 outputs are
//   k * {tr(Q0^3), sqrt(3) tr(Q0^2 Q1), sqrt(3) tr(Q0 Q1^2), tr(Q1^3)}.
// Their gradients need only Q0^2, Q1^2, and the symmetrized Q0 Q1 product, so
// evaluating this closed form inside the node backward avoids a second probe
// projection and the associated global gradient checkpoint.
template <int Channels, int Lmax, bool HasSpin>
__device__ __forceinline__ void add_bis222_probe_gradient(
    int lane,
    long node,
    const Arguments& args,
    const float* __restrict__ probes,
    float (&d_tensor)[5]) {
  using P = Profile<Channels, Lmax, HasSpin>;
  float packed_0[5];
  float packed_1[5];
#pragma unroll
  for (int component = 0; component < 5; ++component) {
    packed_0[component] = probes[component * P::K2 + 0];
    packed_1[component] = probes[component * P::K2 + 1];
  }
  const Matrix3 matrix_0 = packed_to_stf(packed_0);
  const Matrix3 matrix_1 = packed_to_stf(packed_1);
  const float gradient_000 =
      kBis222Scale * load_output_gradient(args.descriptor_gradient,
                                          args.output_inv_std, node,
                                          P::OutputWidth, P::OutputBis222 + 0);
  const float gradient_001 =
      kBis222Scale * kSqrtThree *
      load_output_gradient(args.descriptor_gradient, args.output_inv_std, node,
                           P::OutputWidth, P::OutputBis222 + 1);
  const float gradient_011 =
      kBis222Scale * kSqrtThree *
      load_output_gradient(args.descriptor_gradient, args.output_inv_std, node,
                           P::OutputWidth, P::OutputBis222 + 2);
  const float gradient_111 =
      kBis222Scale * load_output_gradient(args.descriptor_gradient,
                                          args.output_inv_std, node,
                                          P::OutputWidth, P::OutputBis222 + 3);

  Matrix3 matrix_gradient{};
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      float product_00 = 0.0f;
      float product_11 = 0.0f;
      float product_01 = 0.0f;
      float product_10 = 0.0f;
#pragma unroll
      for (int inner = 0; inner < 3; ++inner) {
        product_00 = fmaf(matrix_0.value[row][inner],
                          matrix_0.value[inner][column], product_00);
        product_11 = fmaf(matrix_1.value[row][inner],
                          matrix_1.value[inner][column], product_11);
        product_01 = fmaf(matrix_0.value[row][inner],
                          matrix_1.value[inner][column], product_01);
        product_10 = fmaf(matrix_1.value[row][inner],
                          matrix_0.value[inner][column], product_10);
      }
      matrix_gradient.value[row][column] =
          lane == 0 ? 3.0f * gradient_000 * product_00 +
                          gradient_001 * (product_01 + product_10) +
                          gradient_011 * product_11
                    : gradient_001 * product_00 +
                          gradient_011 * (product_01 + product_10) +
                          3.0f * gradient_111 * product_11;
    }
  }
  float packed_gradient[5];
  matrix_gradient_to_packed(matrix_gradient, packed_gradient);
#pragma unroll
  for (int component = 0; component < 5; ++component) {
    d_tensor[component] += packed_gradient[component];
  }
}

// Index of one coupling operand among the targets of the coupling VJP: the
// probe layout for degrees one and two, followed by the single-channel high
// degrees in the order of the moment layout.
template <int Channels, int Lmax, bool HasSpin>
__device__ __forceinline__ int coupling_target(int degree,
                                               int component,
                                               int rank_index) {
  using P = Profile<Channels, Lmax, HasSpin>;
  if (degree == 1) {
    return component * P::K1 + rank_index;
  }
  if (degree == 2) {
    return 3 * P::K1 + component * P::K2 + rank_index;
  }
  return P::ProbeWidth + (degree == 3 ? 0 : P::High3) + component;
}

// VJP of every sparse coupling of one node, evaluated by the lanes of its node
// group together. Each (output, term) pair of a record contributes to the
// three operands it reads, and the pairs are dealt to the lanes in a fixed
// order. Every lane accumulates into a private shared-memory row indexed by
// `coupling_target`, and the rows are folded in lane order into `total`, so
// the result is deterministic while no lane walks more than its share of the
// pairs. Equal degrees occupy several operand slots of one record and receive
// one contribution per slot.
template <int Channels, int Lmax, bool HasSpin, int NodeLanes>
__device__ __forceinline__ void coupling_gradient(
    int lane,
    long node,
    const Arguments& args,
    const float* __restrict__ probes,
    const float* __restrict__ moments,
    float* __restrict__ rows,
    float* __restrict__ total) {
  using P = Profile<Channels, Lmax, HasSpin, NodeLanes>;
  constexpr int Width = P::NodeWidth;
  constexpr int Targets = P::ProbeWidth + P::HighCount;
  float* row = rows + lane * Targets;
#pragma unroll
  for (int target = 0; target < Targets; ++target) {
    row[target] = 0.0f;
  }
  for (int record = 0; record < args.coupling_count; ++record) {
    const int* meta = args.coupling_meta + record * 8;
    const int degrees[3] = {meta[0], meta[1], meta[2]};
    const int nonzero_begin = meta[3];
    const int nonzero_count = meta[4];
    const int probe_begin = meta[5];
    const int probe_count = meta[6];
    const int coordinate = meta[7];
    const int pairs = probe_count * nonzero_count;
    if (pairs == 0) {
      continue;
    }
    int output = lane / nonzero_count;
    int term = lane - output * nonzero_count;
    for (int pair = lane; pair < pairs; pair += Width) {
      const int selection = __ldg(args.coupling_entry + probe_begin + output);
      const int components = __ldg(args.coupling_entry + nonzero_begin + term);
      const int indices[3] = {selection & 0xFF, (selection >> 8) & 0xFF,
                              (selection >> 16) & 0xFF};
      const int component[3] = {components & 0xFF, (components >> 8) & 0xFF,
                                (components >> 16) & 0xFF};
      const float weight =
          __ldg(args.coupling_value + probe_begin + output) *
          __ldg(args.coupling_value + nonzero_begin + term) *
          load_output_gradient(args.descriptor_gradient, args.output_inv_std,
                               node, P::OutputWidth, coordinate + output);
      float operand[3];
#pragma unroll
      for (int slot = 0; slot < 3; ++slot) {
        operand[slot] = probe_value<Channels, Lmax>(
            probes, moments, degrees[slot], component[slot], indices[slot]);
      }
#pragma unroll
      for (int slot = 0; slot < 3; ++slot) {
        row[coupling_target<Channels, Lmax, HasSpin>(
            degrees[slot], component[slot], indices[slot])] +=
            weight * operand[(slot + 1) % 3] * operand[(slot + 2) % 3];
      }
      term += Width;
      while (term >= nonzero_count) {
        term -= nonzero_count;
        ++output;
      }
    }
  }
  __syncthreads();
  for (int target = lane; target < Targets; target += Width) {
    float sum = 0.0f;
    for (int source = 0; source < Width; ++source) {
      sum += rows[source * Targets + target];
    }
    total[target] = sum;
  }
  __syncthreads();
}

// Four independent lane groups share one warp. An incomplete final block
// aliases inactive groups to the last valid node so every thread reaches each
// block-wide barrier; stores from those groups are suppressed.
template <int Channels, int Lmax, bool HasSpin, int NodeLanes>
__global__ __launch_bounds__(
    Profile<Channels, Lmax, HasSpin, NodeLanes>::Threads,
    2) void node_backward_kernel(Arguments args) {
  using P = Profile<Channels, Lmax, HasSpin, NodeLanes>;
  const int thread = threadIdx.x;
  const int group = thread / P::NodeWidth;
  const int lane = thread & (P::NodeWidth - 1);
  const long candidate = static_cast<long>(blockIdx.x) * P::NodeGroups + group;
  const bool active = candidate < args.node_count;
  const long node = active ? candidate : args.node_count - 1;
  const long state_offset = node * P::StateWidth;
  if constexpr (Lmax >= 3) {
    // The stages below read the descriptor cotangent of the node many times,
    // mostly from loops whose loads feed the arithmetic of the same
    // iteration, so the first touch of each of its lines would otherwise wait
    // for the one before it. Degrees three and four hold a shared-memory
    // footprint that leaves few warps per multiprocessor to hide those waits,
    // so the node's lines are requested up front. The output scale and the
    // readout matrices are shared by every node and stay cached after the
    // first warp has read them.
    prefetch_lines<CacheLevel::L1>(
        args.descriptor_gradient + node * P::OutputWidth, P::OutputWidth, lane,
        P::NodeWidth);
  }

  __shared__ float moments_storage[P::NodeGroups * P::MomentWidth];
  float* moments = moments_storage + group * P::MomentWidth;
  stage_shared<P::MomentWidth, P::NodeWidth>(moments, args.state + state_offset,
                                             lane);
  const float scalar_norm = __ldg(args.state + state_offset + P::MomentWidth);
  const float angular_norm =
      __ldg(args.state + state_offset + P::MomentWidth + 1);
  __syncthreads();

  __shared__ float aligned_storage[P::NodeGroups * P::AlignedWidth];
  float* aligned = aligned_storage + group * P::AlignedWidth;
  for (int channel = lane; channel < P::C1; channel += P::NodeWidth) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C1; ++input) {
        value = fmaf(moments[P::VectorOffset + component * P::C1 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 0,
                                                    input, channel),
                     value);
      }
      aligned[component * P::C1 + channel] = value;
    }
  }
  if (lane < P::C2) {
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C2; ++input) {
        value = fmaf(moments[P::TensorOffset + component * P::C2 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 2,
                                                    input, lane),
                     value);
      }
      aligned[3 * P::C1 + component * P::C2 + lane] = value;
    }
  }
  __syncthreads();

  __shared__ float probes_storage[P::NodeGroups * P::ProbeWidth];
  float* probes = probes_storage + group * P::ProbeWidth;
  if (lane < P::K1) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C1; ++input) {
        value = fmaf(aligned[component * P::C1 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 4,
                                                    input, lane),
                     value);
      }
      probes[component * P::K1 + lane] = value;
    }
  }
  if (lane < P::K2) {
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C2; ++input) {
        value = fmaf(aligned[3 * P::C1 + component * P::C2 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 6,
                                                    input, lane),
                     value);
      }
      probes[3 * P::K1 + component * P::K2 + lane] = value;
    }
  }
  __syncthreads();

  // Sparse-coupling VJP of the node, one private accumulation row per lane
  // and the folded total per node group.
  constexpr int CouplingTargets = P::ProbeWidth + P::HighCount;
  __shared__ float
      coupling_rows[Lmax >= 3 ? P::NodeGroups * P::NodeWidth * CouplingTargets
                              : 1];
  __shared__ float
      coupling_storage[Lmax >= 3 ? P::NodeGroups * CouplingTargets : 1];
  const float* coupling = coupling_storage + group * CouplingTargets;
  if constexpr (Lmax >= 3) {
    coupling_gradient<Channels, Lmax, HasSpin, NodeLanes>(
        lane, node, args, probes, moments,
        coupling_rows + group * P::NodeWidth * CouplingTargets,
        coupling_storage + group * CouplingTargets);
  }

  __shared__ float d_probes_storage[P::NodeGroups * P::ProbeWidth];
  float* d_probes = d_probes_storage + group * P::ProbeWidth;
  if (lane < P::K1) {
    float d_vector[3] = {};
    for (int output = 0; output < P::Bis112; ++output) {
      const int tensor_index = output % P::K2;
      int first, second;
      decode_upper_pair(output / P::K2, P::K1, first, second);
      if (lane != first && lane != second) {
        continue;
      }
      float packed[5];
      float other[3];
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        packed[component] =
            probes[3 * P::K1 + component * P::K2 + tensor_index];
      }
      const int other_index = lane == first ? second : first;
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        other[component] = probes[component * P::K1 + other_index];
      }
      const float gradient =
          -kInvSqrtFive * (first == second ? 2.0f : kSqrtTwo) *
          load_output_gradient(args.descriptor_gradient, args.output_inv_std,
                               node, P::OutputWidth, P::OutputBis112 + output);
      float product[3];
      matrix_vector(packed_to_stf(packed), other, product);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        d_vector[component] =
            fmaf(gradient, product[component], d_vector[component]);
      }
    }
    for (int tensor_index = 0; tensor_index < P::K2; ++tensor_index) {
      const float gradient = load_output_gradient(
          args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
          P::OutputQuartic + tensor_index * P::K1 + lane);
      float packed[5];
      float value[3];
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        packed[component] =
            probes[3 * P::K1 + component * P::K2 + tensor_index];
      }
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        value[component] = probes[component * P::K1 + lane];
      }
      const Matrix3 matrix = packed_to_stf(packed);
      float first[3];
      float second[3];
      matrix_vector(matrix, value, first);
      matrix_vector(matrix, first, second);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        d_vector[component] =
            fmaf(2.0f * gradient, second[component], d_vector[component]);
      }
    }
    if constexpr (Lmax >= 3) {
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        d_vector[component] += coupling[component * P::K1 + lane];
      }
    }
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      d_probes[component * P::K1 + lane] = d_vector[component];
    }
  }

  if (lane < P::K2) {
    float d_tensor[5] = {};
    add_bis222_probe_gradient<Channels, Lmax, HasSpin>(
        lane, node, args, probes + 3 * P::K1, d_tensor);
    for (int output = lane; output < P::Bis112; output += P::K2) {
      int first, second;
      decode_upper_pair(output / P::K2, P::K1, first, second);
      float left[3];
      float right[3];
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        left[component] = probes[component * P::K1 + first];
        right[component] = probes[component * P::K1 + second];
      }
      const float gradient =
          -kInvSqrtFive * (first == second ? 1.0f : kSqrtTwo) *
          load_output_gradient(args.descriptor_gradient, args.output_inv_std,
                               node, P::OutputWidth, P::OutputBis112 + output);
      Matrix3 matrix{};
#pragma unroll
      for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int column = 0; column < 3; ++column) {
          matrix.value[row][column] = gradient * left[row] * right[column];
        }
      }
      float packed_gradient[5];
      matrix_gradient_to_packed(matrix, packed_gradient);
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        d_tensor[component] += packed_gradient[component];
      }
    }
    for (int vector_index = 0; vector_index < P::K1; ++vector_index) {
      const float gradient = load_output_gradient(
          args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
          P::OutputQuartic + lane * P::K1 + vector_index);
      float packed[5];
      float value[3];
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        packed[component] = probes[3 * P::K1 + component * P::K2 + lane];
      }
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        value[component] = probes[component * P::K1 + vector_index];
      }
      float product[3];
      matrix_vector(packed_to_stf(packed), value, product);
      Matrix3 matrix{};
#pragma unroll
      for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int column = 0; column < 3; ++column) {
          matrix.value[row][column] =
              2.0f * gradient * product[row] * value[column];
        }
      }
      float packed_gradient[5];
      matrix_gradient_to_packed(matrix, packed_gradient);
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        d_tensor[component] += packed_gradient[component];
      }
    }
    if constexpr (Lmax >= 3) {
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        d_tensor[component] += coupling[3 * P::K1 + component * P::K2 + lane];
      }
    }
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      d_probes[3 * P::K1 + component * P::K2 + lane] = d_tensor[component];
    }
  }
  __syncthreads();

  __shared__ float d_aligned_storage[P::NodeGroups * P::AlignedWidth];
  float* d_aligned = d_aligned_storage + group * P::AlignedWidth;
  for (int channel = lane; channel < P::C1; channel += P::NodeWidth) {
    float gradient[3] = {};
    for (int other = 0; other < P::C1; ++other) {
      const float upstream =
          (channel == other ? 2.0f : kSqrtTwo) *
          load_output_gradient(
              args.descriptor_gradient, args.output_inv_std, node,
              P::OutputWidth,
              P::OutputGram1 + gram_pair_position(channel, other, P::C1));
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        gradient[component] = fmaf(upstream, aligned[component * P::C1 + other],
                                   gradient[component]);
      }
    }
    for (int probe = 0; probe < P::K1; ++probe) {
      const float weight = readout_weight<Channels, Lmax>(args.readout_matrices,
                                                          5, probe, channel);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        gradient[component] = fmaf(weight, d_probes[component * P::K1 + probe],
                                   gradient[component]);
      }
    }
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      d_aligned[component * P::C1 + channel] = gradient[component];
    }
  }
  if (lane < P::C2) {
    float gradient[5] = {};
    for (int other = 0; other < P::C2; ++other) {
      const float upstream =
          (lane == other ? 2.0f : kSqrtTwo) *
          load_output_gradient(
              args.descriptor_gradient, args.output_inv_std, node,
              P::OutputWidth,
              P::OutputGram2 + gram_pair_position(lane, other, P::C2));
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        gradient[component] =
            fmaf(upstream, aligned[3 * P::C1 + component * P::C2 + other],
                 gradient[component]);
      }
    }
    for (int probe = 0; probe < P::K2; ++probe) {
      const float weight =
          readout_weight<Channels, Lmax>(args.readout_matrices, 7, probe, lane);
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        gradient[component] =
            fmaf(weight, d_probes[3 * P::K1 + component * P::K2 + probe],
                 gradient[component]);
      }
    }
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      d_aligned[3 * P::C1 + component * P::C2 + lane] = gradient[component];
    }
  }
  __syncthreads();

  __shared__ float d_moments_storage[P::NodeGroups * P::MomentWidth];
  float* d_moments = d_moments_storage + group * P::MomentWidth;
  for (int channel = lane; channel < Channels; channel += P::NodeWidth) {
    d_moments[P::ScalarOffset + channel] =
        load_output_gradient(args.descriptor_gradient, args.output_inv_std,
                             node, P::OutputWidth, P::OutputScalar + channel);
  }
  for (int channel = lane; channel < P::C1; channel += P::NodeWidth) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int output = 0; output < P::C1; ++output) {
        value = fmaf(d_aligned[component * P::C1 + output],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 1,
                                                    output, channel),
                     value);
      }
      d_moments[P::VectorOffset + component * P::C1 + channel] = value;
    }
  }
  if (lane < P::C2) {
    // The spin cross Gram contracts the spin quadrupoles against the
    // unaligned degree-two moments, so its cotangent joins this channel here
    // rather than passing through the alignment transpose above.
    float cross[HasSpin ? 2 : 1];
    if constexpr (HasSpin) {
#pragma unroll
      for (int probe = 0; probe < 2; ++probe) {
        cross[probe] = load_output_gradient(
            args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
            P::OutputSpinCross + probe * P::C2 + lane);
      }
    }
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int output = 0; output < P::C2; ++output) {
        value = fmaf(d_aligned[3 * P::C1 + component * P::C2 + output],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 3,
                                                    output, lane),
                     value);
      }
      if constexpr (HasSpin) {
#pragma unroll
        for (int probe = 0; probe < 2; ++probe) {
          value =
              fmaf(cross[probe],
                   spin_tensor_value<Channels, Lmax>(moments, component, probe),
                   value);
        }
      }
      d_moments[P::TensorOffset + component * P::C2 + lane] = value;
    }
  }
  if constexpr (Lmax >= 3) {
    if (lane < Lmax - 2) {
      const int degree = 3 + lane;
      const int offset = P::HighOffset + (lane == 0 ? 0 : P::High3);
      const int count = 2 * degree + 1;
      const float gram =
          2.0f * load_output_gradient(args.descriptor_gradient,
                                      args.output_inv_std, node, P::OutputWidth,
                                      P::OutputGram3 + lane);
      const float* high_coupling =
          coupling + P::ProbeWidth + (lane == 0 ? 0 : P::High3);
      for (int component = 0; component < count; ++component) {
        d_moments[offset + component] =
            fmaf(gram, moments[offset + component], high_coupling[component]);
      }
    }
  }

  // === Spin readout VJP and on-site magnetic gradient ===
  if constexpr (HasSpin) {
    float onsite_vector[3] = {0.0f, 0.0f, 0.0f};
    float onsite_tensor[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int channel = lane; channel < P::SpinVectorWidth;
         channel += P::NodeWidth) {
      float gradient[3] = {0.0f, 0.0f, 0.0f};
      for (int other = 0; other < P::SpinVectorWidth; ++other) {
        const float upstream =
            (channel == other ? 2.0f : kSqrtTwo) *
            load_output_gradient(
                args.descriptor_gradient, args.output_inv_std, node,
                P::OutputWidth,
                P::OutputSpinGramVector +
                    gram_pair_position(channel, other, P::SpinVectorWidth));
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          gradient[component] =
              fmaf(upstream,
                   spin_vector_value<Channels, Lmax>(moments, component, other),
                   gradient[component]);
        }
      }
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        if (channel == 0) {
          // The on-site family carries no normalizer, so its cotangent leaves
          // through the magnetic gradient below and must not reach the edge
          // scan or the mass VJP.
          onsite_vector[component] = gradient[component];
          d_moments[P::SpinOnsiteVector + component] = 0.0f;
        } else {
          d_moments[spin_vector_offset<Channels, Lmax>(component, channel)] =
              gradient[component];
        }
      }
    }
    if (lane < 2) {
      float gradient[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      for (int other = 0; other < 2; ++other) {
        // The on-site self-term is not emitted, so it contributes nothing.
        if (lane == 0 && other == 0) {
          continue;
        }
        const float upstream =
            (lane == other ? 2.0f : kSqrtTwo) *
            load_output_gradient(
                args.descriptor_gradient, args.output_inv_std, node,
                P::OutputWidth,
                P::OutputSpinGramTensor + (lane == 1 && other == 1 ? 1 : 0));
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          gradient[component] =
              fmaf(upstream,
                   spin_tensor_value<Channels, Lmax>(moments, component, other),
                   gradient[component]);
        }
      }
      for (int channel = 0; channel < P::C2; ++channel) {
        const float upstream = load_output_gradient(
            args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
            P::OutputSpinCross + lane * P::C2 + channel);
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          gradient[component] = fmaf(
              upstream, moments[P::TensorOffset + component * P::C2 + channel],
              gradient[component]);
        }
      }
      if (lane == 0) {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          onsite_tensor[component] = gradient[component];
          d_moments[P::SpinOnsiteTensor + component] = 0.0f;
        }
      } else {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          d_moments[P::SpinTensor + component] = gradient[component];
        }
      }
    }
    for (int channel = lane; channel < P::Cs; channel += P::NodeWidth) {
      d_moments[P::SpinMagnitude + channel] = load_output_gradient(
          args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
          P::OutputSpinMagnitude + channel);
      d_moments[P::SpinCoordination + channel] = load_output_gradient(
          args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
          P::OutputSpinCoordination + channel);
    }
    // Lane zero owns both on-site cotangents, so the magnetic gradient of the
    // centre closes here without a shuffle.
    if (active && lane == 0) {
      const int center_type =
          static_cast<int>(args.atype[args.node_begin + node]);
      const SpinTypeWeights weights =
          load_spin_type(args.spin_type, center_type);
      float center_spin[3];
      load_conditioned_spin(args.spin, weights.scale, args.node_begin + node,
                            center_spin);
      float quadrupole[3];
      spin_quadrupole_vjp(onsite_tensor, 1.0f, center_spin, quadrupole);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        args.spin_gradient[node * 3 + component] =
            weights.scale * fmaf(weights.quadrupole, quadrupole[component],
                                 weights.vector * onsite_vector[component]);
      }
    }
  }
  __syncthreads();

  // === Normalizer VJPs ===
  // The scalar and non-scalar blocks carry independent envelope masses, so
  // each contributes its own smooth normalizer cotangent.
  float scalar_dot = 0.0f;
  float angular_dot = 0.0f;
  for (int coordinate = lane; coordinate < P::MomentWidth;
       coordinate += P::NodeWidth) {
    const float product = d_moments[coordinate] * moments[coordinate];
    if (coordinate < Channels) {
      scalar_dot += product;
    } else {
      angular_dot += product;
    }
  }
  const unsigned node_mask = subwarp_mask<P::NodeWidth>(group * P::NodeWidth);
  scalar_dot = subwarp_sum<P::NodeWidth>(scalar_dot, node_mask);
  angular_dot = subwarp_sum<P::NodeWidth>(angular_dot, node_mask);
  // Each mass reaches the output twice: through the moments it normalizes,
  // whose cotangent carries the factor -n^2/2, and through its own divisor
  // sqrt(mass + floor), whose derivative is n/2.
  const float scalar_mass_gradient =
      0.5f * scalar_norm *
      (load_output_gradient(args.descriptor_gradient, args.output_inv_std, node,
                            P::OutputWidth, P::OutputDivisor + 0) -
       scalar_dot * scalar_norm);
  const float angular_mass_gradient =
      0.5f * angular_norm *
      (load_output_gradient(args.descriptor_gradient, args.output_inv_std, node,
                            P::OutputWidth, P::OutputDivisor + 1) -
       angular_dot * angular_norm);
  for (int coordinate = lane; coordinate < P::MomentWidth;
       coordinate += P::NodeWidth) {
    const float value = d_moments[coordinate] *
                        (coordinate < Channels ? scalar_norm : angular_norm);
    if (active) {
      args.moment_gradient[node * P::StateWidth + coordinate] = value;
    }
  }
  if (active && lane == 0) {
    args.moment_gradient[node * P::StateWidth + P::MomentWidth] =
        scalar_mass_gradient;
    args.moment_gradient[node * P::StateWidth + P::MomentWidth + 1] =
        angular_mass_gradient;
  }
}

// === Edge recomputation backward ===

/// Zero the stored cotangents of an edge that contributes nothing. A backward
/// that assembles the force itself may skip either store.
template <bool HasSpin>
__device__ __forceinline__ void clear_edge_cotangents(const Arguments& args,
                                                      long edge) {
  if (args.edge_gradient != nullptr) {
    args.edge_gradient[edge * 3 + 0] = 0.0f;
    args.edge_gradient[edge * 3 + 1] = 0.0f;
    args.edge_gradient[edge * 3 + 2] = 0.0f;
  }
  if constexpr (HasSpin) {
    if (args.edge_spin_gradient != nullptr) {
      args.edge_spin_gradient[edge * 3 + 0] = 0.0f;
      args.edge_spin_gradient[edge * 3 + 1] = 0.0f;
      args.edge_spin_gradient[edge * 3 + 2] = 0.0f;
    }
  }
}

template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          bool Bridged,
          bool Canonical,
          typename index_t,
          int Warps,
          int Split,
          Scan Fold>
__global__ __launch_bounds__(
    Warps * Split * kWarpSize,
    (slots_types(Fold)
         ? slot_resident_warps<HasModes, windowed<Warps>() ? Warps : 0>()
         : edge_resident_blocks<Channels, Lmax, HasModes, HasSpin, true>()) /
        (Warps * Split)) void edge_backward_kernel(Arguments args) {
  using P = Profile<Channels, Lmax, HasSpin>;
  enter_edge_scan<Channels, Lmax, HasModes, HasSpin, Canonical, true, Fold>(
      args);
  constexpr bool SlotScan = slots_types(Fold);
  constexpr int EdgeWidth = P::BackwardEdgeWidth;
  constexpr int Groups = kWarpSize / EdgeWidth;
  constexpr int ChannelTiles = Channels / EdgeWidth;
  constexpr int AngularTiles = (P::C1 + EdgeWidth - 1) / EdgeWidth;
  constexpr int TensorTiles = (P::C2 + EdgeWidth - 1) / EdgeWidth;
  constexpr int HighTiles = (P::HighCount + EdgeWidth - 1) / EdgeWidth;
  static_assert(
      P::Basis == 0 || (P::C1 % EdgeWidth == 0 && P::Basis % EdgeWidth == 0),
      "the low-rank columns tile the lanes of an edge group");

  static_assert(Warps == 1 || Split == 1,
                "a block either holds several rows or shares one");
  const int thread = threadIdx.x % kWarpSize;
  // A block holds `Warps` rows, one warp each, or one row shared by `Split`
  // warps; `part` is the warp's share of a shared row.
  const int warp = Warps > 1 ? threadIdx.x / kWarpSize : 0;
  const int part = Split > 1 ? threadIdx.x / kWarpSize : 0;
  constexpr bool Windowed = windowed<Warps>();
  constexpr int WalkWarps = Windowed ? Warps : 1;
  const long node = static_cast<long>(blockIdx.x) * Warps + warp;
  // A warp past the last node of the final block owns no row: it reads node
  // zero for its constants and scans an empty row, so that it still reaches
  // every barrier of its block.
  const bool active = node < args.node_count;
  const long row_node = active ? node : 0;
  const int base_channel = thread & (EdgeWidth - 1);
  const int group = thread / EdgeWidth;
  const int leader = group * EdgeWidth;
  const unsigned mask = subwarp_mask<EdgeWidth>(leader);
  const auto* edge_index = static_cast<const index_t*>(args.edge_index);
  const auto* destination_order =
      static_cast<const index_t*>(args.destination_order);
  const int center_type =
      static_cast<int>(args.atype[args.node_begin + row_node]);
  // The centre half of every pair length scale is a node constant, like the
  // centre type it follows, and so is the length from which the window of no
  // pair of the row needs resolving.
  const float center_radius =
      Bridged ? __ldg(args.contact_radius + center_type) : 0.0f;
  const float bound = Bridged ? window_bound(args, center_radius) : 0.0f;
  const long begin = args.destination_row_ptr[row_node];
  const long end = active ? args.destination_row_ptr[row_node + 1] : begin;
  const int radial_modes = HasModes ? args.radial_modes : 0;
  const long gradient_offset = row_node * P::StateWidth;
  const float scalar_mass_gradient =
      __ldg(args.moment_gradient + gradient_offset + P::MomentWidth);
  const float angular_mass_gradient =
      __ldg(args.moment_gradient + gradient_offset + P::MomentWidth + 1);
  // Only a bridged specialization carries the analytical pair potential, and
  // its descriptor-only backward still runs without one. Half of every pair
  // energy belongs to the destination, so the half and the node's energy
  // cotangent fold into one weight of the radial slope.
  const bool has_pair = Bridged && args.pair_table != nullptr;
  const float pair_weight =
      has_pair ? 0.5f * static_cast<float>(args.pair_seed[row_node]) : 0.0f;
  float pair_energy = 0.0f;

  // The scalar cotangent is one vector of width C0 that every edge rereads.
  // Holding it in registers costs one entry per channel tile and spills the
  // widest profiles; shared memory serves it as a conflict-free broadcast
  // because concurrent edge groups address identical channels. The warps of
  // a shared row stage it together, each its own share of the channels.
  __shared__ float scalar_gradient_storage[Warps][Channels];
  float* const scalar_gradient = scalar_gradient_storage[warp];
  for (int channel = part * kWarpSize + thread; channel < Channels;
       channel += Split * kWarpSize) {
    scalar_gradient[channel] = __ldg(args.moment_gradient + gradient_offset +
                                     P::ScalarOffset + channel);
  }
  // The reduced spin cotangent is a node constant that every edge rereads and
  // that every lane of a group needs in full, so it is staged in shared
  // memory for the same reason as the scalar cotangent above.
  __shared__ float spin_gradient_storage[Warps][HasSpin ? P::SpinEdgeWidth : 1];
  float* const spin_gradient = spin_gradient_storage[warp];
  if constexpr (HasSpin) {
    for (int coordinate = part * kWarpSize + thread;
         coordinate < P::SpinEdgeWidth; coordinate += Split * kWarpSize) {
      spin_gradient[coordinate] = __ldg(args.moment_gradient + gradient_offset +
                                        P::SpinOffset + coordinate);
    }
  }
  float d_vector[AngularTiles][3] = {};
  float d_tensor[TensorTiles][5] = {};
  float d_high[HighTiles > 0 ? HighTiles : 1] = {};
#pragma unroll
  for (int tile = 0; tile < AngularTiles; ++tile) {
    const int channel = base_channel + tile * EdgeWidth;
    if (channel < P::C1) {
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        d_vector[tile][component] =
            __ldg(args.moment_gradient + gradient_offset + P::VectorOffset +
                  component * P::C1 + channel);
      }
    }
  }
#pragma unroll
  for (int tile = 0; tile < TensorTiles; ++tile) {
    const int channel = base_channel + tile * EdgeWidth;
    if (channel < P::C2) {
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        d_tensor[tile][component] =
            __ldg(args.moment_gradient + gradient_offset + P::TensorOffset +
                  component * P::C2 + channel);
      }
    }
  }
  if constexpr (Lmax >= 3) {
#pragma unroll
    for (int component = 0; component < P::HighCount; ++component) {
      if (component % EdgeWidth == base_channel) {
        d_high[component / EdgeWidth] = __ldg(
            args.moment_gradient + gradient_offset + P::HighOffset + component);
      }
    }
  }

  // Explicitly aligned: the mode residual reads these rows as float4, and
  // a preceding shared array of arbitrary length would otherwise leave the
  // block on a four-byte boundary.
  __shared__ __align__(
      16) float mode_cache[Warps * Split][HasModes ? Groups * kModeStride : 1];
  __shared__ __align__(16) float
      mode_derivative_cache[Warps * Split][HasModes ? Groups * kModeStride : 1];
  const int mode_offset = HasModes ? group * kModeStride : 0;
  float* modes = mode_cache[warp + part] + mode_offset;
  float* mode_derivatives = mode_derivative_cache[warp + part] + mode_offset;
  // A co-scheduled block of a slotted low-rank scan stages the transpose of
  // its expansion matrix for the projections of the cotangents of its rows
  // onto the basis functions of the slots. The blocks of the single-type
  // scans, which also serve the systems that scan every edge pairwise and
  // never project, and the smaller blocks keep the shared memory to the L1
  // cache that their table reads use.
  constexpr bool StageBasis = slots_basis(Fold) && windowed<Warps>();
  __shared__ float
      transposed_basis[StageBasis ? (P::C0 - P::C1) * (P::Basis + 1) : 1];
  if constexpr (StageBasis) {
    stage_transposed_basis<P>(args.basis_matrix, transposed_basis);
    __syncthreads();
  } else {
    // The warps of a shared row read the node cotangents once all of them
    // have staged their shares.
    sync_row<Split>();
  }

  // === Folded cotangents of a single-type system ===
  // With the pair fixed, the coefficients move from the edges onto the node
  // cotangents: the FiLM scale multiplies the channel cotangents, and each
  // extra feature receives the coefficient-weighted sums of the channel
  // cotangents, held by the lane that owns the feature in `extra`.
  constexpr bool Foldable = foldable<Channels, HasSpin, Canonical>();
  constexpr int ExtraFeatures = (HasModes ? kFoldedModes : 0) + 1;
  constexpr int ExtraTiles = (ExtraFeatures + EdgeWidth - 1) / EdgeWidth;
  float extra[ExtraTiles][9] = {};
  int fold_type = -1;
  if constexpr (Foldable && !SlotScan) {
    fold_type = single_type(args, radial_modes);
  }
  const bool folded = fold_type >= 0;
  // A slotted scan reads the cotangents of the mode profiles and of the basis
  // functions of the slot of the source type of every edge.
  static_assert(
      !SlotScan || slottable<Channels, HasModes, HasSpin, Canonical>(),
      "a slotted scan serves a profile that slots");
  constexpr bool SlotModes = SlotScan && HasModes;
  constexpr bool SlotBasis = slots_basis(Fold);
  constexpr int Slots = type_slots(Fold);
  // A spin profile keeps the slot cotangents of its modes in shared memory,
  // one row per mode of a slot, which the lanes of the modes of an edge group
  // read at the slot of every edge: the spin families leave no registers for
  // them. A row holds the nine cotangents of the harmonic components padded
  // to twelve floats, so that two `float4` loads and one float read it. A
  // spin-free profile keeps them in registers, in the slot tiles of
  // `slot_extra`.
  constexpr bool SharedSlots = SlotModes && HasSpin;
  constexpr int CotangentStride = 12;
  constexpr int SlotTiles =
      SlotModes && !SharedSlots ? slot_tiles<EdgeWidth, Slots>() : 1;
  [[maybe_unused]] constexpr int SlotLanes = EdgeWidth / kFoldedModes;
  static_assert(
      !SlotModes || (EdgeWidth % kFoldedModes == 0 && Slots % SlotLanes == 0),
      "the modes of the slots tile the lanes of an edge group");
  [[maybe_unused]] float slot_extra[SlotTiles][9] = {};
  __shared__ __align__(16) float slot_cotangent_storage
      [Warps][SharedSlots ? Slots * kFoldedModes * CotangentStride : 1];
  [[maybe_unused]] float* const slot_cotangent = slot_cotangent_storage[warp];
  TypeSlots<Slots> slots;
  if constexpr (SlotScan) {
    slots = load_type_slots<Slots>(args);
  }
  // Cotangents of the basis functions of a low-rank single-type scan, which
  // replace those of the channels they expand.
  __shared__ float basis_gradient_storage[Warps]
                                         [Fold == Scan::LowRank ? P::Basis : 1];
  float* const basis_gradient = basis_gradient_storage[warp];
  // Cotangents of the basis functions of each slot of a slotted low-rank
  // scan, in rows of `Basis + 1` floats: the basis functions, then the
  // cotangent of the FiLM shift of the channels from `C1` on of the slot. A
  // lane holds `BasisTiles` basis functions.
  constexpr int BasisTiles = SlotBasis ? P::Basis / EdgeWidth : 1;
  constexpr int BasisStride = P::Basis + 1;
  __shared__ float slot_basis_storage[Warps]
                                     [SlotBasis ? Slots * BasisStride : 1];
  float* const slot_basis_gradient = slot_basis_storage[warp];
  float2 leading_film = make_float2(0.0f, 0.0f);
  const float* leading_mixing = nullptr;
  if (folded) {
    const long pair =
        static_cast<long>(center_type) * args.type_count + fold_type;
    const float2* film_row =
        reinterpret_cast<const float2*>(args.pair_film + pair * Channels * 2) +
        base_channel;
    const float* mixing_row =
        HasModes
            ? args.pair_mixing + (pair * Channels + base_channel) * radial_modes
            : nullptr;
    leading_film = __ldg(reinterpret_cast<const float2*>(args.pair_film) +
                         pair * Channels);
    leading_mixing =
        HasModes ? args.pair_mixing + pair * Channels * radial_modes : nullptr;
#pragma unroll
    for (int feature = 0; feature < ExtraFeatures; ++feature) {
      if (feature <= radial_modes) {
        float sums[9] = {};
#pragma unroll
        for (int tile = 0; tile < ChannelTiles; ++tile) {
          const int channel = base_channel + tile * EdgeWidth;
          const float weight =
              feature == radial_modes
                  ? __ldg(film_row + tile * EdgeWidth).y
                  : __ldg(mixing_row + tile * EdgeWidth * radial_modes +
                          feature);
          sums[0] = fmaf(weight, scalar_gradient[channel], sums[0]);
          if (channel < P::C1) {
#pragma unroll
            for (int component = 0; component < 3; ++component) {
              sums[1 + component] =
                  fmaf(weight, d_vector[tile][component], sums[1 + component]);
            }
          }
          if (channel < P::C2) {
#pragma unroll
            for (int component = 0; component < 5; ++component) {
              sums[4 + component] =
                  fmaf(weight, d_tensor[tile][component], sums[4 + component]);
            }
          }
        }
#pragma unroll
        for (int component = 0; component < 9; ++component) {
          sums[component] = subwarp_allsum<EdgeWidth>(sums[component], mask);
          if (feature % EdgeWidth == base_channel) {
            extra[feature / EdgeWidth][component] = sums[component];
          }
        }
      }
    }
    // Every warp of the row has read the staged scalar cotangent before the
    // first one scales it in place.
    sync_row<Split>();
#pragma unroll
    for (int tile = 0; tile < ChannelTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      const float scale = __ldg(film_row + tile * EdgeWidth).x;
      if (group == 0 && part == 0) {
        scalar_gradient[channel] *= scale;
      }
      if (channel < P::C1) {
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          d_vector[tile][component] *= scale;
        }
      }
      if (channel < P::C2) {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          d_tensor[tile][component] *= scale;
        }
      }
    }
    sync_row<Split>();
    if constexpr (Fold == Scan::LowRank) {
      if (part == 0) {
        project_basis<P>(scalar_gradient, args.basis_matrix, basis_gradient,
                         thread);
      }
      sync_row<Split>();
    }
  }
  if constexpr (SlotModes) {
    // Each mode of a slot receives the sums of the channel cotangents
    // weighted by the mode mixing of the pair of the slot, held by the lane
    // that owns the mode of the slot in `slot_extra`.
#pragma unroll
    for (int slot = 0; slot < Slots; ++slot) {
      if (slot < slots.count) {
        const long pair = static_cast<long>(center_type) * args.type_count +
                          slot_type(slots, slot);
        const float* mixing_row =
            args.pair_mixing + (pair * Channels + base_channel) * radial_modes;
#pragma unroll
        for (int mode = 0; mode < kFoldedModes; ++mode) {
          if (mode < radial_modes) {
            float sums[9] = {};
#pragma unroll
            for (int tile = 0; tile < ChannelTiles; ++tile) {
              const int channel = base_channel + tile * EdgeWidth;
              const float weight =
                  __ldg(mixing_row + tile * EdgeWidth * radial_modes + mode);
              sums[0] = fmaf(weight, scalar_gradient[channel], sums[0]);
              if (channel < P::C1) {
#pragma unroll
                for (int component = 0; component < 3; ++component) {
                  sums[1 + component] = fmaf(weight, d_vector[tile][component],
                                             sums[1 + component]);
                }
              }
              if (channel < P::C2) {
#pragma unroll
                for (int component = 0; component < 5; ++component) {
                  sums[4 + component] = fmaf(weight, d_tensor[tile][component],
                                             sums[4 + component]);
                }
              }
            }
#pragma unroll
            for (int component = 0; component < 9; ++component) {
              sums[component] =
                  subwarp_allsum<EdgeWidth>(sums[component], mask);
              if constexpr (SharedSlots) {
                // Every warp of a shared row forms the same sums; the first
                // one publishes them.
                if (part == 0 && thread == component) {
                  slot_cotangent[(slot * kFoldedModes + mode) *
                                     CotangentStride +
                                 component] = sums[component];
                }
              } else if (base_channel == slot_owner<EdgeWidth>(slot, mode)) {
                slot_extra[slot / SlotLanes][component] = sums[component];
              }
            }
          }
        }
      }
    }
    if constexpr (SharedSlots) {
      sync_row<Split>();
    }
  }
  if constexpr (SlotBasis) {
    // The cotangent of basis function `k` of a slot is the sum, over the
    // channels from `C1` on in increasing order, of the channel cotangent
    // times the FiLM scale of the pair of the slot times expansion
    // coefficient `k` of the channel; that of the FiLM shift of the slot is
    // the sum of the channel cotangents times the shifts. Lane `k` of the
    // first warp of the row forms the cotangents of basis function `k` of
    // every slot, and lane zero those of the shifts, for every warp of the
    // row.
    static_assert(P::Basis == kWarpSize,
                  "every lane forms the cotangents of one basis function");
    if (part == 0) {
      constexpr int Expanded = P::C0 - P::C1;
      const float2* film_rows[Slots];
#pragma unroll
      for (int slot = 0; slot < Slots; ++slot) {
        const int type = slot_type(slots, slot < slots.count ? slot : 0);
        film_rows[slot] =
            reinterpret_cast<const float2*>(
                args.pair_film +
                (static_cast<long>(center_type) * args.type_count + type) *
                    Channels * 2) +
            P::C1;
      }
      float basis_sums[Slots] = {};
      float shift_sums[Slots] = {};
      for (int first = 0; first < Expanded; first += 4) {
        const float4 quad =
            StageBasis ? staged_basis_quad<P>(transposed_basis, thread, first)
                       : basis_quad<P>(args.basis_matrix, thread, first);
        const float coefficients[4] = {quad.x, quad.y, quad.z, quad.w};
#pragma unroll
        for (int offset = 0; offset < 4; ++offset) {
          const int channel = first + offset;
          const float gradient = scalar_gradient[P::C1 + channel];
#pragma unroll
          for (int slot = 0; slot < Slots; ++slot) {
            if (slot < slots.count) {
              const float2 film = __ldg(film_rows[slot] + channel);
              basis_sums[slot] = fmaf(coefficients[offset], gradient * film.x,
                                      basis_sums[slot]);
              shift_sums[slot] = fmaf(gradient, film.y, shift_sums[slot]);
            }
          }
        }
      }
#pragma unroll
      for (int slot = 0; slot < Slots; ++slot) {
        slot_basis_gradient[slot * BasisStride + thread] = basis_sums[slot];
        if (thread == 0) {
          slot_basis_gradient[slot * BasisStride + P::Basis] = shift_sums[slot];
        }
      }
    }
    sync_row<Split>();
  }

  // Fused assembly state. Each lane of an edge group owns the force and
  // virial components congruent to its channel modulo the group width, so the
  // edge totals are distributed rather than serialized on the leading lane.
  constexpr int ForceSlots = (3 + EdgeWidth - 1) / EdgeWidth;
  constexpr int VirialSlots = (9 + EdgeWidth - 1) / EdgeWidth;
  const bool assemble = args.force != nullptr;
  float destination_force[ForceSlots] = {};
  float node_virial[VirialSlots] = {};

  // Offsets of the five spin families inside the staged cotangent row.
  constexpr int SpinMagnitudeSlot = P::SpinMagnitude - P::SpinOffset;
  constexpr int SpinCoordinationSlot = P::SpinCoordination - P::SpinOffset;
  constexpr int SpinVectorSlot = P::SpinVector - P::SpinOffset;
  constexpr int SpinBondSlot = P::SpinBond - P::SpinOffset;
  constexpr int SpinTensorSlot = P::SpinTensor - P::SpinOffset;

  const auto scan_edge = [&](const long position, auto scan) {
    constexpr Scan Kind = decltype(scan)::value;
    constexpr bool Folded = Kind == Scan::Direct || Kind == Scan::LowRank;
    constexpr int FoldColumns = fold_columns<P, Kind>();
    constexpr int FoldTiles = FoldColumns / EdgeWidth;
    // The tiles of a low-rank scan from ``C1 / EdgeWidth`` on hold basis
    // functions, whose cotangents replace those of the channels.
    constexpr int DirectTiles =
        Kind == Scan::LowRank ? P::C1 / EdgeWidth : FoldTiles;
    // Channel tiles whose FiLM amplitude the edges carry.
    constexpr int EdgeTiles =
        slots_basis(Kind) ? P::C1 / EdgeWidth : ChannelTiles;
    const long edge = edge_at_position<Canonical>(position, destination_order);
    if (args.edge_mask != nullptr && !args.edge_mask[edge]) {
      if (thread == leader) {
        clear_edge_cotangents<HasSpin>(args, edge);
      }
      return;
    }
    const long source = static_cast<long>(edge_index[edge]);
    const int source_type = static_cast<int>(args.atype[source]);
    if constexpr (!Canonical) {
      if (center_type >= args.type_count - 1 ||
          source_type >= args.type_count - 1) {
        if (thread == leader) {
          clear_edge_cotangents<HasSpin>(args, edge);
        }
        return;
      }
    }
    const EdgeGeometry geometry = load_geometry<Bridged>(
        edge, source, source_type, args.rcut, args.f_inner, args.f_outer,
        args.inverse_f_width, center_radius, bound, args.contact_radius,
        args.eps, args.edge_vec);
    const TableLocation location = locate_table<Canonical>(
        geometry.seen, args.table_stride, args.table_max, args.interval_count);
    // The pair term depends on the geometry alone, so it starts here and runs
    // alongside the channel scan; its slope joins the radial cotangent at the
    // tail, which keeps the exponentials off the dependency chain of the scan.
    float pair_slope = 0.0f;
    if (has_pair) {
      const long pair_row = static_cast<long>(center_type) * args.type_count +
                            geometry.source_type;
      const PairTerm term =
          pair_term<EdgeWidth>(args.pair_table + pair_row * 8, base_channel,
                               geometry.radius, geometry.inverse_radius);
      pair_energy = fmaf(0.5f, term.energy, pair_energy);
      pair_slope = term.slope;
    }
    const TableRow row =
        reads_low_rank(Kind)
            ? table_row(args.basis_table, location, FoldColumns + radial_modes)
            : table_row(args.table, location, args.table_width);
    const float coordinate = location.coordinate;
    const bool clamped = location.clamped;
    if constexpr (HasModes) {
      if constexpr (Kind == Scan::Pairwise) {
        __syncwarp(mask);
        for (int mode = base_channel; mode < radial_modes; mode += EdgeWidth) {
          const float2 value = evaluate_table_with_derivative(
              row, Channels + mode, coordinate, clamped);
          modes[mode] = value.x;
          mode_derivatives[mode] = value.y;
        }
        __syncwarp(mask);
      }
    }

    float basis[9];
    fill_angular_basis(geometry, basis);
    const float envelope = geometry.envelope;
    const long pair =
        static_cast<long>(center_type) * args.type_count + geometry.source_type;

    // The high-degree cotangent enters the leading channel only, so its
    // angular contraction is reduced onto the lane that owns that channel.
    [[maybe_unused]] float high_angular = 0.0f;
    if constexpr (Lmax >= 3) {
#pragma unroll
      for (int component = 0; component < P::HighCount; ++component) {
        if (component % EdgeWidth == base_channel) {
          high_angular =
              fmaf(d_high[component / EdgeWidth],
                   high_basis_value(geometry, component), high_angular);
        }
      }
      high_angular = subwarp_sum<EdgeWidth>(high_angular, mask);
    }

    // The spin payload of this edge, hoisted like the forward one: every
    // channel multiplies the same conditioned neighbour moment and the same
    // projection of that moment onto the bond.
    [[maybe_unused]] float neighbor_spin[3] = {0.0f, 0.0f, 0.0f};
    [[maybe_unused]] float neighbor_magnitude = 0.0f;
    [[maybe_unused]] float neighbor_gate = 0.0f;
    [[maybe_unused]] float neighbor_scale = 0.0f;
    [[maybe_unused]] float bond_alignment = 0.0f;
    [[maybe_unused]] float spin_cotangent[3] = {0.0f, 0.0f, 0.0f};
    [[maybe_unused]] const float2* spin_row = nullptr;
    if constexpr (HasSpin) {
      const SpinTypeWeights weights =
          load_spin_type(args.spin_type, geometry.source_type);
      load_conditioned_spin(args.spin, weights.scale, geometry.source,
                            neighbor_spin);
      neighbor_magnitude = neighbor_spin[0] * neighbor_spin[0] +
                           neighbor_spin[1] * neighbor_spin[1] +
                           neighbor_spin[2] * neighbor_spin[2];
      neighbor_gate = weights.gate;
      neighbor_scale = weights.scale;
      bond_alignment = neighbor_spin[0] * basis[1] +
                       neighbor_spin[1] * basis[2] +
                       neighbor_spin[2] * basis[3];
      spin_row =
          reinterpret_cast<const float2*>(args.spin_pair + pair * P::Cs * 2) +
          base_channel;
    }

    float radial_gradient = 0.0f;
    float envelope_gradient = 0.0f;
    float d_basis[9] = {};
    // Direction cotangent of every family that evaluates its own angular chain
    // rule: the single-channel high degrees, which fill it after the channel
    // loop, and the bond-projected spin channels, which fill it inside. Both
    // reach the coordinate gradient through the one transverse projection in
    // `basis_vjp`, so no family is reduced separately.
    float direction_gradient[3] = {0.0f, 0.0f, 0.0f};
    // Angular payload of the leading channel, held by the leader of the group,
    // and the slope of that channel that the high degrees read: of its bare
    // radial feature in a folded scan, and in the other scans of its FiLM
    // amplitude before the mode mixing, whose value `leading_value` holds.
    [[maybe_unused]] float leading_payload = 0.0f;
    [[maybe_unused]] float leading_value = 0.0f;
    [[maybe_unused]] float leading_slope = 0.0f;
    if constexpr (Folded) {
      // The pair coefficients live in the folded cotangents, so a channel
      // contributes through its bare radial feature, and every extra feature
      // through its own value on the lane that owns it. The angular cotangent
      // accumulates without the squared envelope, which multiplies it once.
      const float squared = envelope * envelope;
      float leading = 0.0f;
#pragma unroll
      for (int tile = 0; tile < FoldTiles; ++tile) {
        const int channel = base_channel + tile * EdgeWidth;
        const float2 radial =
            evaluate_table_with_derivative(row, channel, coordinate, clamped);
        float angular = 0.0f;
        if (channel < P::C1) {
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            angular =
                fmaf(d_vector[tile][component], basis[1 + component], angular);
          }
        }
        if (channel < P::C2) {
#pragma unroll
          for (int component = 0; component < 5; ++component) {
            angular =
                fmaf(d_tensor[tile][component], basis[4 + component], angular);
          }
        }
        const float scaled = envelope * angular;
        const float gradient = tile < DirectTiles
                                   ? scalar_gradient[channel]
                                   : basis_gradient[channel - P::C1];
        radial_gradient =
            fmaf(envelope * (gradient + scaled), radial.y, radial_gradient);
        envelope_gradient =
            fmaf(fmaf(2.0f, scaled, gradient), radial.x, envelope_gradient);
        if (channel < P::C1) {
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            d_basis[1 + component] = fmaf(d_vector[tile][component], radial.x,
                                          d_basis[1 + component]);
          }
        }
        if (channel < P::C2) {
#pragma unroll
          for (int component = 0; component < 5; ++component) {
            d_basis[4 + component] = fmaf(d_tensor[tile][component], radial.x,
                                          d_basis[4 + component]);
          }
        }
        if (tile == 0) {
          leading = radial.x;
          leading_slope = radial.y;
        }
      }
      float feature_value[ExtraTiles] = {};
      float feature_slope[ExtraTiles] = {};
#pragma unroll
      for (int tile = 0; tile < ExtraTiles; ++tile) {
        const int feature = base_channel + tile * EdgeWidth;
        if (tile * EdgeWidth <= radial_modes) {
          const float2 value =
              feature < radial_modes
                  ? evaluate_table_with_derivative(row, FoldColumns + feature,
                                                   coordinate, clamped)
                  : make_float2(feature == radial_modes ? 1.0f : 0.0f, 0.0f);
          feature_value[tile] = value.x;
          feature_slope[tile] = value.y;
          float angular = 0.0f;
#pragma unroll
          for (int component = 1; component < 9; ++component) {
            angular = fmaf(extra[tile][component], basis[component], angular);
          }
          const float scaled = envelope * angular;
          radial_gradient = fmaf(envelope * (extra[tile][0] + scaled), value.y,
                                 radial_gradient);
          envelope_gradient = fmaf(fmaf(2.0f, scaled, extra[tile][0]), value.x,
                                   envelope_gradient);
#pragma unroll
          for (int component = 1; component < 9; ++component) {
            d_basis[component] =
                fmaf(extra[tile][component], value.x, d_basis[component]);
          }
        }
      }
      if constexpr (Lmax >= 3) {
        // The high-degree cotangent reaches the leading channel through its
        // full amplitude, which the leader reassembles from the bare feature,
        // its coefficients and the mode profiles of the group.
        float film_value = fmaf(leading_film.x, leading, leading_film.y);
        float film_slope = leading_film.x * leading_slope;
        if constexpr (HasModes) {
#pragma unroll
          for (int mode = 0; mode < kFoldedModes; ++mode) {
            if (mode < radial_modes) {
              const int owner = leader + mode % EdgeWidth;
              const float weight = __ldg(leading_mixing + mode);
              film_value = fmaf(
                  weight,
                  __shfl_sync(mask, feature_value[mode / EdgeWidth], owner),
                  film_value);
              film_slope = fmaf(
                  weight,
                  __shfl_sync(mask, feature_slope[mode / EdgeWidth], owner),
                  film_slope);
            }
          }
        }
        if (thread == leader) {
          radial_gradient =
              fmaf(squared * high_angular, film_slope, radial_gradient);
          envelope_gradient = fmaf(2.0f * envelope * high_angular, film_value,
                                   envelope_gradient);
        }
        leading_payload = film_value * squared;
      }
#pragma unroll
      for (int component = 1; component < 9; ++component) {
        d_basis[component] *= squared;
      }
    } else {
      const float2* film_row = reinterpret_cast<const float2*>(
                                   args.pair_film + pair * Channels * 2) +
                               base_channel;
      [[maybe_unused]] const float* mixing_row =
          HasModes ? args.pair_mixing +
                         (pair * Channels + base_channel) * radial_modes
                   : nullptr;
#pragma unroll
      for (int tile = 0; tile < EdgeTiles; ++tile) {
        const int channel = base_channel + tile * EdgeWidth;
        const float2 radial =
            evaluate_table_with_derivative(row, channel, coordinate, clamped);
        const float2 film = __ldg(film_row + tile * EdgeWidth);
        // ``film_value`` is the pre-envelope FiLM amplitude phi; the reduced
        // payload is phi * chi for degree zero and phi * chi^2 above it.
        float film_value = fmaf(film.x, radial.x, film.y);
        float film_derivative = film.x * radial.y;
        if constexpr (HasModes && Kind == Scan::Pairwise) {
          accumulate_modes_with_derivative(
              mixing_row + tile * (EdgeWidth * radial_modes), modes,
              mode_derivatives, radial_modes, film_value, film_derivative);
        }
        float angular = 0.0f;
        if (channel < P::C1) {
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            angular =
                fmaf(d_vector[tile][component], basis[1 + component], angular);
          }
        }
        if (channel < P::C2) {
#pragma unroll
          for (int component = 0; component < 5; ++component) {
            angular =
                fmaf(d_tensor[tile][component], basis[4 + component], angular);
          }
        }
        if constexpr (Lmax >= 3 && Kind == Scan::Pairwise) {
          if (channel == 0) {
            angular += high_angular;
          }
        }
        // d/dphi   = chi * (p + chi * a)
        // d/dchi   = phi * (p + 2 * chi * a)
        const float scaled = envelope * angular;
        const float film_gradient = scalar_gradient[channel] + scaled;
        radial_gradient =
            fmaf(envelope * film_gradient, film_derivative, radial_gradient);
        envelope_gradient =
            fmaf(film_gradient + scaled, film_value, envelope_gradient);
        const float angular_payload = film_value * envelope * envelope;
        if (channel < P::C1) {
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            d_basis[1 + component] =
                fmaf(d_vector[tile][component], angular_payload,
                     d_basis[1 + component]);
          }
        }
        if (channel < P::C2) {
#pragma unroll
          for (int component = 0; component < 5; ++component) {
            d_basis[4 + component] =
                fmaf(d_tensor[tile][component], angular_payload,
                     d_basis[4 + component]);
          }
        }
        if constexpr (HasSpin) {
          if (channel < P::Cs) {
            const float2 spin_film = __ldg(spin_row + tile * EdgeWidth);
            const float profile = fmaf(spin_film.x, radial.x, spin_film.y);
            const float envelope_squared = envelope * envelope;
            const float weight = profile * envelope_squared;
            const float magnitude_gradient =
                spin_gradient[SpinMagnitudeSlot + channel];
            // The two degree-one cotangents are read once and used by the
            // amplitude, the magnetic and the angular accumulations below.
            float vector_gradient[3];
            float bond_gradient[3];
#pragma unroll
            for (int component = 0; component < 3; ++component) {
              vector_gradient[component] =
                  spin_gradient[SpinVectorSlot + component * P::Cs + channel];
              bond_gradient[component] =
                  spin_gradient[SpinBondSlot + component * P::Cs + channel];
            }
            // Component of the bond cotangent along the bond. The
            // bond-projected family is `P = w (s . u) u`, so this one dot
            // product carries all three of its VJPs: `dP/dw` contracts to
            // `(s . u) (dP . u)`, `dP/ds` to `w (dP . u) u`, and the direction
            // term below reuses it once more.
            float bond_projection = 0.0f;
#pragma unroll
            for (int component = 0; component < 3; ++component) {
              bond_projection = fmaf(bond_gradient[component],
                                     basis[1 + component], bond_projection);
            }
            float sigma = fmaf(
                magnitude_gradient, neighbor_magnitude,
                spin_gradient[SpinCoordinationSlot + channel] * neighbor_gate);
            sigma = fmaf(bond_alignment, bond_projection, sigma);
#pragma unroll
            for (int component = 0; component < 3; ++component) {
              sigma = fmaf(vector_gradient[component], neighbor_spin[component],
                           sigma);
            }
            // Magnetic cotangent of the source moment. The coordination family
            // reads the type gate rather than the moment, so it contributes
            // nothing here.
            const float linear = 2.0f * weight * magnitude_gradient;
            const float projected = weight * bond_projection;
#pragma unroll
            for (int component = 0; component < 3; ++component) {
              spin_cotangent[component] =
                  fmaf(linear, neighbor_spin[component],
                       fmaf(projected, basis[1 + component],
                            fmaf(weight, vector_gradient[component],
                                 spin_cotangent[component])));
            }
            // Angular cotangent of the bond-projected family. Differentiating
            // `(s . u) u` with respect to `u` at fixed `s` gives the rank-one
            // sum `s (dP . u) + (s . u) dP`, which the transverse projection in
            // `basis_vjp` turns into the coordinate gradient. Every other spin
            // family is independent of the direction and contributes nothing.
            const float aligned_weight = weight * bond_alignment;
#pragma unroll
            for (int component = 0; component < 3; ++component) {
              direction_gradient[component] =
                  fmaf(projected, neighbor_spin[component],
                       fmaf(aligned_weight, bond_gradient[component],
                            direction_gradient[component]));
            }
            if (channel == 0) {
              // The single neighbour quadrupole rides the leading spin channel.
              // Both of its contributions are formed here rather than hoisted,
              // so its five cotangent components never stay live across the
              // channel loop.
#pragma unroll
              for (int component = 0; component < 5; ++component) {
                sigma = fmaf(spin_gradient[SpinTensorSlot + component],
                             spin_quadrupole_component(
                                 neighbor_spin[0], neighbor_spin[1],
                                 neighbor_spin[2], component),
                             sigma);
              }
              float quadrupole[3];
              spin_quadrupole_vjp(spin_gradient + SpinTensorSlot, weight,
                                  neighbor_spin, quadrupole);
#pragma unroll
              for (int component = 0; component < 3; ++component) {
                spin_cotangent[component] += quadrupole[component];
              }
            }
            // w = chi^2 (gamma g + beta): the distance enters through the table
            // and through the envelope, and `sigma` is the cotangent of that
            // shared amplitude over all five families.
            radial_gradient = fmaf(envelope_squared * sigma,
                                   spin_film.x * radial.y, radial_gradient);
            envelope_gradient =
                fmaf(2.0f * envelope * profile, sigma, envelope_gradient);
          }
        }
        if constexpr (Lmax >= 3) {
          if (channel == 0) {
            leading_payload = angular_payload;
            leading_value = film_value;
            leading_slope = film_derivative;
          }
        }
      }
      if constexpr (slots_types(Kind)) {
        const int slot = type_slot(slots, geometry.source_type);
        [[maybe_unused]] const float squared = envelope * envelope;
        [[maybe_unused]] float2 profile = make_float2(0.0f, 0.0f);
        if constexpr (HasModes) {
          // The mode of the lane contributes through its bare profile, on the
          // lanes that own the slot of the source type, as an extra feature
          // of a single-type scan does.
          const int mode = base_channel % kFoldedModes;
          if (mode < radial_modes) {
            profile = evaluate_table_with_derivative(row, FoldColumns + mode,
                                                     coordinate, clamped);
          }
          // The contribution of mode `mode` of the slot of the edge, from its
          // nine cotangents `cotangent`.
          const auto add_mode = [&](const float (&cotangent)[9]) {
            float angular = 0.0f;
#pragma unroll
            for (int component = 1; component < 9; ++component) {
              angular = fmaf(cotangent[component], basis[component], angular);
            }
            const float scaled = envelope * angular;
            radial_gradient = fmaf(envelope * (cotangent[0] + scaled),
                                   profile.y, radial_gradient);
            envelope_gradient = fmaf(fmaf(2.0f, scaled, cotangent[0]),
                                     profile.x, envelope_gradient);
            const float weight = profile.x * squared;
#pragma unroll
            for (int component = 1; component < 9; ++component) {
              d_basis[component] =
                  fmaf(cotangent[component], weight, d_basis[component]);
            }
          };
          if constexpr (SharedSlots) {
            // The lanes of the modes of the descriptor read the row of their
            // mode of the slot of the edge; the rows of the modes past
            // `radial_modes` hold no cotangents. A slotted launch carries at
            // most `kFoldedModes` modes, so lane `l < radial_modes` is the
            // lane of mode `l`.
            if (base_channel < radial_modes) {
              const float* const cotangent_row =
                  slot_cotangent +
                  (slot * kFoldedModes + mode) * CotangentStride;
              const float4 low =
                  reinterpret_cast<const float4*>(cotangent_row)[0];
              const float4 high =
                  reinterpret_cast<const float4*>(cotangent_row)[1];
              const float cotangent[9] = {low.x,  low.y,  low.z,
                                          low.w,  high.x, high.y,
                                          high.z, high.w, cotangent_row[8]};
              add_mode(cotangent);
            }
          } else {
#pragma unroll
            for (int tile = 0; tile < SlotTiles; ++tile) {
              if (owned_slot<EdgeWidth>(base_channel, tile) == slot) {
                add_mode(slot_extra[tile]);
              }
            }
          }
        }
        if constexpr (slots_basis(Kind)) {
          // The basis functions of the lane contribute through their bare
          // values with the cotangents of the slot of the source type, and
          // the FiLM shift of the slot through the envelope on the leader.
          const float* const basis_cotangent =
              slot_basis_gradient + slot * BasisStride;
#pragma unroll
          for (int tile = 0; tile < BasisTiles; ++tile) {
            const int function = base_channel + tile * EdgeWidth;
            const float2 radial = evaluate_table_with_derivative(
                row, P::C1 + function, coordinate, clamped);
            const float gradient = basis_cotangent[function];
            radial_gradient =
                fmaf(envelope * gradient, radial.y, radial_gradient);
            envelope_gradient = fmaf(gradient, radial.x, envelope_gradient);
          }
          if (thread == leader) {
            envelope_gradient += basis_cotangent[P::Basis];
          }
        }
        if constexpr (Lmax >= 3) {
          // The high-degree cotangent reaches the leading channel through its
          // full amplitude, which the leader completes with the mode mixing
          // of the pair; lane `leader + r` of the group holds the profile of
          // mode `r`.
          float value = leading_value;
          float slope = leading_slope;
          if constexpr (HasModes) {
            const float* const pair_modes =
                args.pair_mixing + pair * Channels * radial_modes;
#pragma unroll
            for (int mixed = 0; mixed < kFoldedModes; ++mixed) {
              if (mixed < radial_modes) {
                const float weight = __ldg(pair_modes + mixed);
                value =
                    fmaf(weight, __shfl_sync(mask, profile.x, leader + mixed),
                         value);
                slope =
                    fmaf(weight, __shfl_sync(mask, profile.y, leader + mixed),
                         slope);
              }
            }
          }
          if (thread == leader) {
            radial_gradient =
                fmaf(squared * high_angular, slope, radial_gradient);
            envelope_gradient =
                fmaf(2.0f * envelope * high_angular, value, envelope_gradient);
          }
          leading_payload = value * squared;
        }
      }
    }

    if constexpr (Lmax >= 3) {
      const float amplitude = __shfl_sync(mask, leading_payload, leader);
#pragma unroll
      for (int component = 0; component < P::HighCount; ++component) {
        if (component % EdgeWidth == base_channel) {
          high_basis_gradient(geometry, component,
                              d_high[component / EdgeWidth] * amplitude,
                              direction_gradient);
        }
      }
    }

    // Both envelope masses contribute once per edge, so their cotangent joins
    // the single lane that also owns the leading channel.
    if (thread == leader) {
      const float squared = envelope * envelope;
      envelope_gradient =
          fmaf(4.0f * squared * envelope, angular_mass_gradient,
               fmaf(2.0f * envelope, scalar_mass_gradient, envelope_gradient));
    }
    // The table terms above differentiate the length the table reads; the
    // pair potential acts on the true length and joins after the chain rule.
    radial_gradient =
        radial_chain<Bridged>(geometry, args, center_radius, bound,
                              radial_gradient, envelope_gradient);
    if constexpr (Bridged) {
      radial_gradient = fmaf(pair_weight, pair_slope, radial_gradient);
    }

    // The basis VJP is linear in the radial and angular cotangents, so
    // applying it per lane reduces three Cartesian components instead of the
    // full set of angular components across the edge group.
    float output[3];
    basis_vjp(geometry, d_basis, direction_gradient, radial_gradient, output);
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      output[component] = subwarp_allsum<EdgeWidth>(output[component], mask);
    }
    if (thread == leader && args.edge_gradient != nullptr) {
      args.edge_gradient[edge * 3 + 0] = output[0];
      args.edge_gradient[edge * 3 + 1] = output[1];
      args.edge_gradient[edge * 3 + 2] = output[2];
    }
    if (assemble) {
      // The source half of the force belongs to a node this warp does not
      // own; the destination half and the virial accumulate in registers.
      // Lane c of a group adds component c, so the source atomics of an edge
      // share one instruction and one sector once the group spans the three
      // components.
      const float* vector = args.edge_vec + edge * 3;
#pragma unroll
      for (int first = 0; first < 3; first += EdgeWidth) {
        const int component = first + base_channel;
        if (component < 3) {
          const float value = vector_component(output, component);
          unsigned long long* const source_words =
              args.force + geometry.source * kForceWords;
          check_force_range(value, source_words);
          atomicAdd(source_words + component, fixed_point(-value));
          destination_force[first / EdgeWidth] += value;
        }
      }
#pragma unroll
      for (int component = 0; component < 9; ++component) {
        if (component % EdgeWidth == base_channel) {
          node_virial[component / EdgeWidth] =
              fmaf(-output[component / 3], vector[component % 3],
                   node_virial[component / EdgeWidth]);
        }
      }
    }
    if constexpr (HasSpin) {
      // The magnetic cotangent belongs to the source node, which this
      // destination-major scan does not own. It is either emitted per edge
      // and reduced onto sources by the shared edge assembly, or added onto
      // its source by the fused assembly. Every contribution above
      // differentiates the conditioned moment, so the source scale is the
      // single remaining chain factor to the raw input moment.
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        spin_cotangent[component] =
            neighbor_scale *
            subwarp_allsum<EdgeWidth>(spin_cotangent[component], mask);
      }
      if (thread == leader && args.edge_spin_gradient != nullptr) {
        args.edge_spin_gradient[edge * 3 + 0] = spin_cotangent[0];
        args.edge_spin_gradient[edge * 3 + 1] = spin_cotangent[1];
        args.edge_spin_gradient[edge * 3 + 2] = spin_cotangent[2];
      }
      if (assemble) {
#pragma unroll
        for (int first = 0; first < 3; first += EdgeWidth) {
          const int component = first + base_channel;
          if (component < 3) {
            const float value = vector_component(spin_cotangent, component);
            check_force_range(value,
                              args.force + geometry.source * kForceWords);
            atomicAdd(
                args.spin_force + geometry.source * kForceWords + component,
                fixed_point(value));
          }
        }
      }
    }
  };
  // The first warp of a shared row requests the lines of the row for every
  // warp of the row.
  if constexpr (Canonical) {
    if (part == 0) {
      prefetch_row_bonds(args, begin, end, thread);
      if constexpr (SlotScan) {
        prefetch_row_sources(args, edge_index, begin, end, thread);
      }
    }
  }
  const auto pairwise_edge = [&](const long position) {
    scan_edge(position, ScanTag<Scan::Pairwise>{});
  };
  if constexpr (SlotScan) {
    walk_row<WalkWarps, Groups, Split>(
        begin, end, group, thread, part, args.edge_vec, args.table_max,
        [&](const long position) { scan_edge(position, ScanTag<Fold>{}); });
  } else if constexpr (Foldable) {
    if (folded) {
      walk_row<WalkWarps, Groups, Split>(
          begin, end, group, thread, part, args.edge_vec, args.table_max,
          [&](const long position) { scan_edge(position, ScanTag<Fold>{}); });
    } else {
      walk_row<WalkWarps, Groups, Split>(begin, end, group, thread, part,
                                         args.edge_vec, args.table_max,
                                         pairwise_edge);
    }
  } else {
    walk_row<WalkWarps, Groups, Split>(begin, end, group, thread, part,
                                       args.edge_vec, args.table_max,
                                       pairwise_edge);
  }
  if (!active) {
    return;
  }
  // The node totals of the warps that share the row, which `accumulators`
  // visits in its order.
  constexpr int Accumulators = ForceSlots + VirialSlots + (Bridged ? 1 : 0);
  __shared__ float split_buffer[split_buffer_floats<Split>()];
  const auto accumulators = [&](auto&& visit) {
#pragma unroll
    for (int slot = 0; slot < ForceSlots; ++slot) {
      visit(destination_force[slot]);
    }
#pragma unroll
    for (int slot = 0; slot < VirialSlots; ++slot) {
      visit(node_virial[slot]);
    }
    if constexpr (Bridged) {
      visit(pair_energy);
    }
  };
  if (!merge_split_row<Split, Accumulators>(part, thread, split_buffer,
                                            accumulators)) {
    return;
  }
  if (assemble) {
    // Every group holds partial totals of its own edges in the same slots,
    // so one merge over the groups completes the node.
#pragma unroll
    for (int slot = 0; slot < ForceSlots; ++slot) {
      destination_force[slot] =
          reduce_channel_groups<EdgeWidth>(destination_force[slot]);
    }
#pragma unroll
    for (int slot = 0; slot < VirialSlots; ++slot) {
      node_virial[slot] = reduce_channel_groups<EdgeWidth>(node_virial[slot]);
    }
    if (thread < EdgeWidth) {
      const long absolute = args.node_begin + node;
#pragma unroll
      for (int first = 0; first < 3; first += EdgeWidth) {
        const int component = first + base_channel;
        if (component < 3) {
          const float value = destination_force[first / EdgeWidth];
          check_force_range(value, args.force + absolute * kForceWords);
          atomicAdd(args.force + absolute * kForceWords + component,
                    fixed_point(value));
        }
      }
#pragma unroll
      for (int component = 0; component < 9; ++component) {
        if (component % EdgeWidth == base_channel) {
          args.node_virial[node * 9 + component] =
              node_virial[component / EdgeWidth];
        }
      }
    }
  }
  if (has_pair) {
    // The lanes hold disjoint terms of disjoint edge groups, so one reduction
    // over the warp is the pair energy of the node.
    const float total = warp_sum(pair_energy);
    if (thread == 0) {
      args.pair_energy[node] += static_cast<double>(total);
    }
  }
}

template <bool Canonical, typename index_t>
__global__ void zero_padding_kernel(long node_count,
                                    long edge_count,
                                    const index_t* destination_order,
                                    const long* destination_row_ptr,
                                    float* edge_gradient,
                                    float* edge_spin_gradient) {
  const long valid_edge_count = destination_row_ptr[node_count];
  for (long position = valid_edge_count + blockIdx.x * blockDim.x + threadIdx.x;
       position < edge_count;
       position += static_cast<long>(blockDim.x) * gridDim.x) {
    const long edge = edge_at_position<Canonical>(position, destination_order);
    edge_gradient[edge * 3 + 0] = 0.0f;
    edge_gradient[edge * 3 + 1] = 0.0f;
    edge_gradient[edge * 3 + 2] = 0.0f;
    if (edge_spin_gradient != nullptr) {
      edge_spin_gradient[edge * 3 + 0] = 0.0f;
      edge_spin_gradient[edge * 3 + 1] = 0.0f;
      edge_spin_gradient[edge * 3 + 2] = 0.0f;
    }
  }
}

// === Launch dispatch ===

/// Launch the edge scans `Family::kernel<Rows, 1, Fold>` of blocks of `Rows`
/// independent rows, or, for a canonical system of 32-bit indices whose rows
/// `row_split` shares among `Split > 1` warps, the scans
/// `Family::kernel<1, Split, Fold>`.
template <class Family, int Rows, Scan Fold, bool Canonical, typename index_t>
void launch_row_scans(const Arguments& args, cudaStream_t stream) {
  if constexpr (Canonical && std::is_same_v<index_t, std::uint32_t>) {
    switch (row_split<Family::template kernel<Rows, 1, Fold>, Rows>(
        args.system_node_count)) {
      case 8:
        launch_edge_scan<1, 8>(Family::template kernel<1, 8, Fold>, args,
                               stream);
        return;
      case 4:
        launch_edge_scan<1, 4>(Family::template kernel<1, 4, Fold>, args,
                               stream);
        return;
      case 2:
        launch_edge_scan<1, 2>(Family::template kernel<1, 2, Fold>, args,
                               stream);
        return;
      default:
        break;
    }
  }
  launch_edge_scan<Rows, 1>(Family::template kernel<Rows, 1, Fold>, args,
                            stream);
}

/// Launch the co-scheduled blocks `Family::kernel<Warps, 1, Fold>` when the
/// system fills the device with them, and return whether it did; a profile
/// that does not co-schedule (`Warps == 1`) never does.
template <class Family, int Warps, Scan Fold>
bool launch_coscheduled_scans(const Arguments& args, cudaStream_t stream) {
  if constexpr (Warps > 1) {
    if (fills_coscheduled<Warps>(args.system_node_count)) {
      launch_edge_scan<Warps, 1>(Family::template kernel<Warps, 1, Fold>, args,
                                 stream);
      return true;
    }
  }
  return false;
}

/// Launch the edge scans of one direction, whose kernels are the
/// `Family::kernel<Warps, Split, Fold>` of the single-type scan `Fold` of the
/// descriptor: co-scheduled blocks of `Warps` rows when the system fills the
/// device with them, rows that a small canonical system shares among several
/// warps, and blocks of independent rows otherwise, one for a spin-free
/// profile and `kSlotRows` for a spin profile. Every launch also launches the
/// slotted kernels of the profile (`Family::launch_slots<Fold>`): the number
/// of node types of the system, which the device knows, selects the kernel
/// that scans the rows, and the blocks of the others retire.
template <class Family,
          bool Backward,
          int Channels,
          int Lmax,
          bool HasSpin,
          bool Canonical,
          typename index_t>
void launch_edge_scans(const Arguments& args, cudaStream_t stream) {
  dispatch_fold<Channels, Lmax, HasSpin, Canonical>(args, [&](auto fold) {
    constexpr Scan Fold = decltype(fold)::value;
    constexpr int Warps =
        coscheduled_warps<Channels, Lmax, HasSpin, Canonical, Backward>();
    if (!launch_coscheduled_scans<Family, Warps, Fold>(args, stream)) {
      launch_row_scans<Family, HasSpin ? kSlotRows : 1, Fold, Canonical,
                       index_t>(args, stream);
    }
    Family::template launch_slots<Fold>(args, stream);
  });
}

/// The slotted scan of `Slots` slots of a descriptor whose single-type scan
/// is `Fold`: the one that slots the basis functions of its low-rank table,
/// or the one that reads the radial table.
template <Scan Fold, int Slots>
constexpr Scan slotted_scan() {
  if constexpr (Slots > 4) {
    return Fold == Scan::LowRank ? Scan::SlottedLowRank8 : Scan::Slotted8;
  } else {
    return Fold == Scan::LowRank ? Scan::SlottedLowRank4 : Scan::Slotted4;
  }
}

/// Launch the slotted kernels `Family::kernel<Warps, Split, Slotted>` of a
/// launch that carries the bitmap of the node types, the launches whose
/// systems they may serve: co-scheduled blocks of `slot_coscheduled_warps`
/// rows when the system fills the device with them, rows shared among the
/// warps of a block when the system leaves most of the resident warps of the
/// slotted kernel idle (`row_split`), and blocks of `kSlotRows` independent
/// rows otherwise.
template <class Family,
          bool Backward,
          int Channels,
          int Lmax,
          bool HasSpin,
          bool Canonical,
          typename index_t,
          Scan Slotted>
void launch_slot_scans(const Arguments& args, cudaStream_t stream) {
  if (args.type_present == nullptr) {
    return;
  }
  constexpr int Warps = slot_coscheduled_warps<Channels, Lmax, HasSpin,
                                               Backward, type_slots(Slotted)>();
  if (!launch_coscheduled_scans<Family, Warps, Slotted>(args, stream)) {
    launch_row_scans<Family, kSlotRows, Slotted, Canonical, index_t>(args,
                                                                     stream);
  }
}

template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          bool Bridged,
          bool Canonical,
          typename index_t>
struct ForwardLauncher {
  template <int Warps, int Split, Scan Fold>
  static constexpr auto kernel = forward_kernel<Channels,
                                                Lmax,
                                                HasModes,
                                                HasSpin,
                                                Bridged,
                                                Canonical,
                                                index_t,
                                                Warps,
                                                Split,
                                                Fold>;

  /// Launch the slotted forward scans of a descriptor whose single-type
  /// scan is `Fold` (see `runs_slotted`).
  template <Scan Fold>
  static void launch_slots(const Arguments& args, cudaStream_t stream) {
    if constexpr (runs_slotted<Channels, HasModes, HasSpin, Canonical, false,
                               Fold, 4>()) {
      launch_slot_scans<ForwardLauncher, false, Channels, Lmax, HasSpin,
                        Canonical, index_t, slotted_scan<Fold, 4>()>(args,
                                                                     stream);
    }
    if constexpr (runs_slotted<Channels, HasModes, HasSpin, Canonical, false,
                               Fold, 8>()) {
      launch_slot_scans<ForwardLauncher, false, Channels, Lmax, HasSpin,
                        Canonical, index_t, slotted_scan<Fold, 8>()>(args,
                                                                     stream);
    }
  }

  static void run(const Arguments& args, cudaStream_t stream) {
    using P = Profile<Channels, Lmax, HasSpin>;
    launch_edge_scans<ForwardLauncher, false, Channels, Lmax, HasSpin,
                      Canonical, index_t>(args, stream);
    forward_readout_kernel<Channels, Lmax, HasSpin>
        <<<static_cast<int>(args.node_count), P::Threads, 0, stream>>>(args);
  }
};

template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          bool Bridged,
          bool Canonical,
          typename index_t>
struct BackwardLauncher {
  // Pick the node-group width from what the running device grants the two
  // compiled kernels rather than from its architecture: the six shared arrays
  // of the node backward scale with the group count and the scalar width, so
  // the same source is shared-memory bound on a part with a 100 KB budget per
  // multiprocessor and register bound on one with more than twice that.
  //
  // Widening the group halves that footprint and so buys resident warps, but
  // every lane of a group reloads its node's shared state, so it also doubles
  // the redundant load work. Measured across the scalar widths, the trade pays
  // only where the narrow variant is starved of warps outright: at a quarter of
  // the device's warp capacity the wide variant wins by 2.1 % of the whole
  // backward, at a third the two are within noise, and above that the narrow
  // variant wins by up to 3.6 %. The threshold is therefore stated against the
  // device's own warp capacity, and the query is made once per configuration
  // and device.
  static bool prefer_wide_nodes() {
    static DeviceMemo preferences;
    return preferences.get([](int device) {
      using Narrow = Profile<Channels, Lmax, HasSpin, kNodeLanesNarrow>;
      using Wide = Profile<Channels, Lmax, HasSpin, kNodeLanesWide>;
      int narrow_blocks = 0;
      int wide_blocks = 0;
      check_query(
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &narrow_blocks,
              node_backward_kernel<Channels, Lmax, HasSpin, kNodeLanesNarrow>,
              Narrow::Threads, 0),
          "the occupancy query of the node backward");
      check_query(
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &wide_blocks,
              node_backward_kernel<Channels, Lmax, HasSpin, kNodeLanesWide>,
              Wide::Threads, 0),
          "the occupancy query of the node backward");
      int threads_per_sm = 0;
      check_query(
          cudaDeviceGetAttribute(
              &threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device),
          "the thread capacity of a multiprocessor");
      const int warps_per_sm = threads_per_sm / kWarpSize;
      const int narrow_warps = narrow_blocks * (Narrow::Threads / kWarpSize);
      return static_cast<long>(wide_blocks > narrow_blocks &&
                               narrow_warps * 4 < warps_per_sm);
    }) != 0;
  }

  template <int Warps, int Split, Scan Fold>
  static constexpr auto kernel = edge_backward_kernel<Channels,
                                                      Lmax,
                                                      HasModes,
                                                      HasSpin,
                                                      Bridged,
                                                      Canonical,
                                                      index_t,
                                                      Warps,
                                                      Split,
                                                      Fold>;

  /// Launch the slotted edge backwards of a descriptor whose single-type
  /// scan is `Fold` (see `runs_slotted`).
  template <Scan Fold>
  static void launch_slots(const Arguments& args, cudaStream_t stream) {
    if constexpr (runs_slotted<Channels, HasModes, HasSpin, Canonical, true,
                               Fold, 4>()) {
      launch_slot_scans<BackwardLauncher, true, Channels, Lmax, HasSpin,
                        Canonical, index_t, slotted_scan<Fold, 4>()>(args,
                                                                     stream);
    }
    if constexpr (runs_slotted<Channels, HasModes, HasSpin, Canonical, true,
                               Fold, 8>()) {
      launch_slot_scans<BackwardLauncher, true, Channels, Lmax, HasSpin,
                        Canonical, index_t, slotted_scan<Fold, 8>()>(args,
                                                                     stream);
    }
  }

  template <int NodeLanes>
  static void launch_node_backward(const Arguments& args, cudaStream_t stream) {
    using NodeProfile = Profile<Channels, Lmax, HasSpin, NodeLanes>;
    const int node_blocks =
        static_cast<int>((args.node_count + NodeProfile::NodeGroups - 1) /
                         NodeProfile::NodeGroups);
    node_backward_kernel<Channels, Lmax, HasSpin, NodeLanes>
        <<<node_blocks, NodeProfile::Threads, 0, stream>>>(args);
  }

  static void run(const Arguments& args, cudaStream_t stream) {
    if (prefer_wide_nodes()) {
      launch_node_backward<kNodeLanesWide>(args, stream);
    } else {
      launch_node_backward<kNodeLanesNarrow>(args, stream);
    }
    launch_edge_scans<BackwardLauncher, true, Channels, Lmax, HasSpin,
                      Canonical, index_t>(args, stream);
    // The reserved edge slots beyond the physical count are only known on the
    // device, so the grid is sized from the storage bound and the surplus
    // blocks retire immediately.
    if (args.clear_padding && args.edge_gradient != nullptr) {
      constexpr int kPaddingThreads = 128;
      constexpr long kPaddingBlockLimit = 1024;
      const long padding_blocks = min(
          kPaddingBlockLimit,
          max(1L, (args.edge_count + kPaddingThreads - 1) / kPaddingThreads));
      zero_padding_kernel<Canonical, index_t>
          <<<static_cast<int>(padding_blocks), kPaddingThreads, 0, stream>>>(
              args.node_count, args.edge_count,
              static_cast<const index_t*>(args.destination_order),
              args.destination_row_ptr, args.edge_gradient,
              HasSpin ? args.edge_spin_gradient : nullptr);
    }
  }
};

// A signed and an unsigned 32-bit index share their representation over the
// non-negative range that node and edge identifiers occupy, so the topology
// collapses to two element widths.
template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          bool Bridged,
          template <int, int, bool, bool, bool, bool, typename> class L>
void dispatch_topology(const Arguments& args, cudaStream_t stream) {
  const bool wide = args.index_kind == IndexKind::Bits64;
  if (args.canonical) {
    if (wide) {
      L<Channels, Lmax, HasModes, HasSpin, Bridged, true, long>::run(args,
                                                                     stream);
    } else {
      L<Channels, Lmax, HasModes, HasSpin, Bridged, true, std::uint32_t>::run(
          args, stream);
    }
  } else {
    if (wide) {
      L<Channels, Lmax, HasModes, HasSpin, Bridged, false, long>::run(args,
                                                                      stream);
    } else {
      L<Channels, Lmax, HasModes, HasSpin, Bridged, false, std::uint32_t>::run(
          args, stream);
    }
  }
}

// Zone bridging is a compile-time specialization for the same reason as the
// native spin: a plain descriptor must not carry the switch arithmetic of the
// edge envelope, nor the accumulator and the branch of the analytical pair
// potential, of a bridged one.
template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          template <int, int, bool, bool, bool, bool, typename> class L>
void dispatch_bridging(const Arguments& args, cudaStream_t stream) {
  if (args.bridged) {
    dispatch_topology<Channels, Lmax, HasModes, HasSpin, true, L>(args, stream);
  } else {
    dispatch_topology<Channels, Lmax, HasModes, HasSpin, false, L>(args,
                                                                   stream);
  }
}

// Native spin is a compile-time specialization for the same reason as the
// mode residual: a spinless descriptor must not carry the spin accumulators,
// the staged spin cotangent, or the wider moment state of one that has them.
// The neighbour spin width is derived from the degree profile, so presence is
// the whole choice and the instantiation matrix only doubles.
template <int Channels,
          int Lmax,
          bool HasModes,
          template <int, int, bool, bool, bool, bool, typename> class L>
void dispatch_spin(const Arguments& args, cudaStream_t stream) {
  if (args.has_spin) {
    dispatch_bridging<Channels, Lmax, HasModes, true, L>(args, stream);
  } else {
    dispatch_bridging<Channels, Lmax, HasModes, false, L>(args, stream);
  }
}

// The mode residual is a compile-time specialization for the same reason as
// the angular degree: a descriptor without radial modes must not carry the
// vector temporaries and the shared profile cache of one that has them.
template <int Channels,
          int Lmax,
          template <int, int, bool, bool, bool, bool, typename> class L>
void dispatch_modes(const Arguments& args, cudaStream_t stream) {
  if (args.radial_modes > 0) {
    dispatch_spin<Channels, Lmax, true, L>(args, stream);
  } else {
    dispatch_spin<Channels, Lmax, false, L>(args, stream);
  }
}

template <int Channels, int Lmax>
void launch_forward(const Arguments& arguments, cudaStream_t stream) {
  dispatch_modes<Channels, Lmax, ForwardLauncher>(arguments, stream);
}

template <int Channels, int Lmax>
void launch_backward(const Arguments& arguments, cudaStream_t stream) {
  dispatch_modes<Channels, Lmax, BackwardLauncher>(arguments, stream);
}

}  // namespace deepmd_dpa4c
