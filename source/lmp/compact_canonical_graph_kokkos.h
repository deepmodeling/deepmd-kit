// SPDX-License-Identifier: LGPL-3.0-or-later
// The compact canonical graph builder is available when the LAMMPS Kokkos
// package is enabled.
#ifdef LMP_KOKKOS

#ifndef LMP_COMPACT_CANONICAL_GRAPH_KOKKOS_H
#define LMP_COMPACT_CANONICAL_GRAPH_KOKKOS_H

#include <Kokkos_NestedSort.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "kokkos_type.h"
#include "neigh_list_kokkos.h"
#include "neighbor.h"
#include "pointers.h"

namespace LAMMPS_NS {

#if defined(KOKKOS_ENABLE_CUDA)
/// Arrangement of the 32 * K elements a warp holds K per lane in registers:
/// element ``k * 32 + lane`` in register ``k`` of ``lane`` (register-major),
/// or element ``lane * K + k`` (lane-major).
enum class CompactCanonicalLayout { kRegisterMajor, kLaneMajor };

// Bitonic sort of the 32 * K keys a warp holds K per lane, in the element
// order of ``Layout``. Partners whose elements differ only in the bits of the
// register index live in other registers of the same lane; the others live in
// other lanes and are exchanged by shuffle. In the register-major layout the
// five low bits of an element are those of its lane, so every stage of a
// stride below the warp width shuffles; in the lane-major layout they are the
// high bits, so only the stages of strides of at least ``K`` do, fifteen in
// all, and the network runs without shared memory or barriers in either.
// Every loop bound is a compile-time constant, which keeps the register array
// statically indexed.
template <int K, CompactCanonicalLayout Layout, typename Key>
__device__ __forceinline__ void compact_canonical_warp_sort(Key (&key)[K],
                                                            const int lane) {
  constexpr bool lane_major = Layout == CompactCanonicalLayout::kLaneMajor;
  const auto element = [lane](const int k) {
    return lane_major ? lane * K + k : k * 32 + lane;
  };
#pragma unroll
  for (int size = 2; size <= 32 * K; size <<= 1) {
#pragma unroll
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      if (lane_major ? stride < K : stride >= 32) {
        const int register_stride = lane_major ? stride : stride >> 5;
#pragma unroll
        for (int k = 0; k < K; ++k) {
          if ((k & register_stride) == 0) {
            const int partner = k | register_stride;
            const bool ascending = (element(k) & size) == 0;
            const Key low = min(key[k], key[partner]);
            const Key high = max(key[k], key[partner]);
            key[k] = ascending ? low : high;
            key[partner] = ascending ? high : low;
          }
        }
      } else {
        const int lane_stride = lane_major ? stride / K : stride;
        const bool lower = (lane & lane_stride) == 0;
#pragma unroll
        for (int k = 0; k < K; ++k) {
          const Key other = __shfl_xor_sync(0xffffffffu, key[k], lane_stride);
          const bool ascending = (element(k) & size) == 0;
          key[k] =
              (lower == ascending) ? min(key[k], other) : max(key[k], other);
        }
      }
    }
  }
}

#endif

/// Bits of the candidate slot in the sort keys of the single-pass fill, which
/// serves candidate lists of up to ``1 << kCompactCanonicalSlotBits`` entries.
constexpr int kCompactCanonicalSlotBits = 9;

#if defined(KOKKOS_ENABLE_CUDA)
// Read the packed record of one atom: its fixed-point coordinates and, in the
// fourth slot, the model node it resolves to as a candidate. A record is 16
// bytes and aligned to them, so it arrives in one 128-bit load.
template <class Packed>
__device__ __forceinline__ int4 compact_canonical_record(const Packed& packed,
                                                         const int atom) {
  return __ldg(reinterpret_cast<const int4*>(&packed(atom, 0)));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Copy one word from global into shared memory without passing it through a
// register, and fetch the whole 128-byte line that holds it into L2. A line
// of the neighbor list holds one slot of 32 consecutive centers, which four
// consecutive blocks of the fill stage at about the same time, so each line
// is read from DRAM once and in one piece. The prefetch size takes PTX ISA
// 7.4, CUDA 11.4; an earlier toolkit copies the word alone. The copies of a
// thread complete at compact_canonical_await_copies.
__device__ __forceinline__ void compact_canonical_copy_async(
    int* destination, const int* source) {
  const unsigned address =
      static_cast<unsigned>(__cvta_generic_to_shared(destination));
#if CUDART_VERSION >= 11040
  asm volatile(
      "cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n" ::"r"(address),
      "l"(source)
      : "memory");
#else
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(address),
               "l"(source)
               : "memory");
#endif
}

__device__ __forceinline__ void compact_canonical_await_copies() {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
}
#endif

// One component of the displacement between two fixed-point coordinates, in
// fixed-point steps. A coordinate is held modulo 2^32, so the unsigned
// difference wraps to the exact displacement whenever that displacement is
// shorter than 2^31 steps, which the step of the fill guarantees for every
// candidate.
__device__ __forceinline__ float compact_canonical_displacement(const int qj,
                                                                const int qi) {
  return static_cast<float>(
      static_cast<int>(static_cast<unsigned>(qj) - static_cast<unsigned>(qi)));
}

// Exclusive prefix of the published totals of every block before ``block``,
// by the decoupled look-back of a single-pass prefix scan. The warp reads the
// status words of 32 predecessors at once, nearest first, and sums the totals
// down to the nearest inclusive prefix; a window that still holds an
// unpublished word before that prefix is read again.
__device__ __forceinline__ unsigned long long compact_canonical_look_back(
    const int block,
    const int lane,
    const unsigned long long stamp,
    unsigned long long* status) {
  constexpr unsigned kFull = 0xffffffffu;
  constexpr unsigned long long kInclusive = 2ull << 32;
  unsigned long long prefix = 0;
  for (int base = block - 1; base >= 0;) {
    const int predecessor = base - lane;
    unsigned long long value = stamp | kInclusive;
    if (predecessor >= 0) {
      value =
          *(static_cast<volatile unsigned long long*>(status) + predecessor);
    }
    const bool current = (value & ~((1ull << 34) - 1)) == stamp;
    const unsigned long long state = current ? (value & (3ull << 32)) : 0ull;
    const unsigned inclusive = __ballot_sync(kFull, state == kInclusive);
    const unsigned pending = __ballot_sync(kFull, state == 0ull);
    const int stop = inclusive ? __ffs(inclusive) - 1 : 31;
    const unsigned needed = stop == 31 ? kFull : ((2u << stop) - 1u);
    if (pending & needed) {
      continue;
    }
    unsigned long long part =
        lane <= stop ? static_cast<unsigned long long>(value & kFull) : 0ull;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      part += __shfl_xor_sync(kFull, part, offset);
    }
    prefix += part;
    if (inclusive) {
      break;
    }
    base -= 32;
  }
  return prefix;
}

// The place of one block in the decoupled look-back of the single-pass build:
// its index, the stamp of the build, the status words of every block, the
// total of the block, and the shared words through which the warp that
// resolves the block offset claims the look-back and hands the offset to the
// other warps. The total and the inclusive prefix of a block may reach its
// status word in either order, since different warps may store them: the
// total is exact and the inclusive prefix is the sum of every total up to the
// block, so a later block that reads either derives the same offset, and one
// that reads the total merely looks further back. A sum that saturates the
// inclusive value yields an offset at least as large as that value either
// way, which the host rejects.
struct CompactCanonicalScan {
  int block;
  unsigned long long stamp;
  unsigned long long* status;
  unsigned total;
  int* claim;
  volatile unsigned long long* offset;
  volatile int* ready;

  // Look back for the block offset, publish the inclusive prefix of the block
  // and hand the offset over, unless another warp of the block has claimed
  // the look-back before; the first warp to arrive runs it.
  __device__ __forceinline__ void resolve_once(const int lane) const {
    constexpr unsigned kFull = 0xffffffffu;
    constexpr unsigned long long kInclusive = 2ull << 32;
    const int claimed = lane == 0 ? atomicExch(claim, 1) : 0;
    if (__shfl_sync(kFull, claimed, 0) != 0) {
      return;
    }
    const unsigned long long prefix =
        compact_canonical_look_back(block, lane, stamp, status);
    if (lane == 0) {
      if (block != 0) {
        // The inclusive value saturates at the largest 32-bit value, so a
        // graph of that many edges or more carries a prefix of at least that
        // value through every later block into the end of the last row,
        // which the host reads back and rejects.
        const unsigned long long inclusive = prefix + total;
        const unsigned long long saturated =
            inclusive < kFull ? inclusive
                              : static_cast<unsigned long long>(kFull);
        *(static_cast<volatile unsigned long long*>(status) + block) =
            stamp | kInclusive | saturated;
      }
      *offset = prefix;
      __threadfence_block();
      *ready = 1;
    }
  }

  // The block offset, once ``resolve_once`` has handed it over. A device of
  // compute capability below 7.0 has no sleep instruction and polls the
  // volatile flag without pausing.
  __device__ __forceinline__ std::int64_t await() const {
    while (*ready == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      __nanosleep(32);
#endif
    }
    __threadfence_block();
    return static_cast<std::int64_t>(*offset);
  }
};

// Order, place and emit one destination row of the single-pass build from the
// keys of its candidates, ``K`` registers per lane, of which ``count`` are
// survivors. The survivors are sorted on ``KS`` registers per lane, a power of
// two: when that is fewer than ``K``, they are first compacted through the
// warp's ``staging`` row of shared memory, and otherwise the keys stay in
// their registers and the registers past ``K`` hold rejected keys, which sort
// behind every survivor. Either way the multiset of keys, and therefore the
// sorted row, is unchanged, while the sorting network shrinks with the square
// of its depth. The network sorts in the lane-major layout, so that only its
// fifteen stages across lanes shuffle, and the sorted row returns to the
// register-major layout through ``slot_atom``, which holds the neighbor-list
// entry of every candidate slot of the row until the source atoms of the
// survivors take their place, at their positions in the row. The emitting
// stores of a warp then cover consecutive positions.
//
// The row offset is the block offset plus ``row_offset`` within the block.
// The first warp of the block to have sorted and laid out its row looks back
// for the block offset, by which time the blocks before it have had the time
// of the sort to publish their totals, and the other warps await it.
template <int KS,
          int K,
          class Tags,
          class Packed,
          class RowPointers,
          class Sources,
          class Vectors,
          class Counts>
__device__ __forceinline__ void compact_canonical_finish_row(
    const std::uint32_t (&candidate_key)[K],
    const unsigned count,
    const unsigned row_offset,
    const CompactCanonicalScan& scan,
    const int lane,
    std::uint32_t* staging,
    int* slot_atom,
    const int center,
    const int mi,
    const int4 position_i,
    const int inum,
    const int tail_begin,
    const int tail_end,
    const std::int64_t capacity,
    std::int64_t* fill_total,
    const float vector_scale,
    const Tags& tag,
    const Packed& packed,
    const RowPointers& destination_row_ptr,
    const Sources& source,
    const Vectors& edge_vec,
    const bool count_sources,
    const Counts& source_counts) {
  constexpr unsigned kFull = 0xffffffffu;
  constexpr std::uint32_t kRejected = ~std::uint32_t{0};
  constexpr int kSlot = kCompactCanonicalSlotBits;
  constexpr std::uint32_t kSlotMask = (1u << kSlot) - 1u;
  constexpr auto kLaneMajor = CompactCanonicalLayout::kLaneMajor;

  // === Gather the survivors onto the sorting registers ===
  // The network sorts any arrangement of its input, so the survivors enter it
  // in the order they are gathered.
  static_assert((KS & (KS - 1)) == 0 && KS >= 4,
                "the sorting network spans 32 * KS keys, KS a power of two "
                "of at least four");
  std::uint32_t key[KS];
  if constexpr (KS >= K) {
#pragma unroll
    for (int k = 0; k < KS; ++k) {
      key[k] = k < K ? candidate_key[k] : kRejected;
    }
  } else {
    const unsigned below = (1u << lane) - 1u;
    unsigned placed = 0;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const bool survivor = candidate_key[k] != kRejected;
      const unsigned ballot = __ballot_sync(kFull, survivor);
      if (survivor) {
        staging[placed + __popc(ballot & below)] = candidate_key[k];
      }
      placed += __popc(ballot);
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < KS; ++k) {
      const unsigned position = k * 32 + lane;
      key[k] = position < count ? staging[position] : kRejected;
    }
  }
  compact_canonical_warp_sort<KS, kLaneMajor>(key, lane);

  // === Order survivors of equal quantized length by source tag ===
  // The predecessor of element ``lane * KS + k`` is the element before it in
  // the same lane, or the last element of the previous lane.
  bool tie = false;
  const std::uint32_t last_above = __shfl_up_sync(kFull, key[KS - 1], 1);
#pragma unroll
  for (int k = 0; k < KS; ++k) {
    const std::uint32_t previous =
        k > 0 ? key[k > 0 ? k - 1 : 0] : (lane == 0 ? kRejected : last_above);
    tie = tie || (key[k] != kRejected && previous != kRejected &&
                  (key[k] >> kSlot) == (previous >> kSlot));
  }
  if (__any_sync(kFull, tie)) {
    unsigned long long wide[KS];
#pragma unroll
    for (int k = 0; k < KS; ++k) {
      wide[k] = ~0ull;
      if (key[k] != kRejected) {
        const int jj = static_cast<int>(key[k] & kSlotMask);
        const unsigned long long rank =
            static_cast<std::uint32_t>(tag(slot_atom[jj] & NEIGHMASK));
        wide[k] =
            (static_cast<unsigned long long>(key[k] >> kSlot) << (32 + kSlot)) |
            (rank << kSlot) | static_cast<unsigned long long>(jj);
      }
    }
    compact_canonical_warp_sort<KS, kLaneMajor>(wide, lane);
#pragma unroll
    for (int k = 0; k < KS; ++k) {
      key[k] = static_cast<std::uint32_t>(((wide[k] >> (32 + kSlot)) << kSlot) |
                                          (wide[k] & kSlotMask));
    }
  }

  // === Lay the source atoms out in the register-major order of the row ===
  // The survivors precede the rejected keys, so position ``p`` of the row
  // holds a survivor when ``p < count``. Every lane looks up the atoms of its
  // positions before any is stored over the entries, and writes them four at
  // a time: a quad that starts below ``count`` ends below ``32 * K``, inside
  // the row.
  int atom[KS];
#pragma unroll
  for (int k = 0; k < KS; ++k) {
    atom[k] =
        key[k] != kRejected ? slot_atom[key[k] & kSlotMask] & NEIGHMASK : 0;
  }
  __syncwarp();
#pragma unroll
  for (int quad = 0; quad < KS; quad += 4) {
    if (lane * KS + quad < count) {
      *reinterpret_cast<int4*>(&slot_atom[lane * KS + quad]) =
          make_int4(atom[quad], atom[quad + 1], atom[quad + 2], atom[quad + 3]);
    }
  }
  __syncwarp();
#pragma unroll
  for (int k = 0; k < KS; ++k) {
    const unsigned position = k * 32 + lane;
    atom[k] = position < count ? slot_atom[position] : -1;
  }

  // === Await the block offset and publish the row ===
  scan.resolve_once(lane);
  const std::int64_t prefix = scan.await() + row_offset;
  if (mi >= 0 && lane == 0) {
    destination_row_ptr(mi) = prefix;
  }
  if (center == inum - 1) {
    const std::int64_t total = prefix + count;
    for (int m = tail_begin + lane; m <= tail_end; m += 32) {
      destination_row_ptr(m) = total;
    }
    // Rows the previous build filled past this build's end become guards.
    const std::int64_t end = total < capacity ? total : capacity;
    const std::int64_t used = *fill_total;
    for (std::int64_t position = end + lane; position < used; position += 32) {
      source(position) = 0u;
      edge_vec(3 * position + 0) = 0.0f;
      edge_vec(3 * position + 1) = 0.0f;
      edge_vec(3 * position + 2) = 0.0f;
    }
    __syncwarp();
    if (lane == 0) {
      *fill_total = end;
    }
  }

  // === Emit the row in key order ===
  // The records of four survivors are requested before the first of them is
  // used, as in the key pass.
  constexpr int kBatch = 4;
  static_assert(KS % kBatch == 0, "the slots split into whole batches");
#pragma unroll
  for (int first = 0; first < KS; first += kBatch) {
    int4 record[kBatch];
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      record[u] = atom[first + u] >= 0
                      ? compact_canonical_record(packed, atom[first + u])
                      : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      const std::int64_t position = prefix + (first + u) * 32 + lane;
      if (atom[first + u] >= 0 && position < capacity) {
        source(position) = static_cast<std::uint32_t>(record[u].w);
        edge_vec(3 * position + 0) =
            compact_canonical_displacement(record[u].x, position_i.x) *
            vector_scale;
        edge_vec(3 * position + 1) =
            compact_canonical_displacement(record[u].y, position_i.y) *
            vector_scale;
        edge_vec(3 * position + 2) =
            compact_canonical_displacement(record[u].z, position_i.z) *
            vector_scale;
        if (count_sources) {
          Kokkos::atomic_fetch_add(&source_counts(record[u].w),
                                   std::uint32_t{1});
        }
      }
    }
  }
}
#endif

// Fill the destination rows of ``Warps`` consecutive centers in one block of
// the single-pass build, one warp per center.
//
// The rows are laid out in model-node order, which is the order of the
// centers, so the row offset of a center is the number of edges of every
// center before it. A block takes one ticket, which orders it after every
// block that started before it, and fills the rows of the ``Warps`` centers
// that the ticket names. Consecutive centers are spatial neighbors, so the
// warps of a block gather largely the same neighbor coordinates and serve one
// another from L1.
//
// The neighbor list stores the candidates of one slot for consecutive centers
// next to each other, so the block reads the candidate entries of all its
// centers together, eight centers of a slot per 32-byte sector, into shared
// memory, where the warps read their own rows from then on. Each warp keys
// its candidates; the block sums the survivors of its centers, publishes the
// sum, and derives its offset by looking back over the published sums of the
// blocks before it -- the decoupled look-back of a single-pass prefix scan --
// once the first of its warps has sorted its row, while the others sort
// theirs. A block publishes its sum as soon as it is known and its inclusive
// prefix once its own look-back closes, so a window of 32 predecessors read
// at once usually ends at an inclusive prefix. A block can wait only on
// tickets taken before its own, which belong to running or finished blocks,
// so the scan cannot deadlock.
//
// A status word holds the build epoch above a two-bit state above a 32-bit
// value, so a word left by an earlier build reads as not yet published and the
// array never needs clearing. Edges beyond ``capacity`` are not written; the
// caller compares the total with the capacity and repeats the build with
// larger arrays.
//
// The keys of up to ``32 * K`` candidates are formed in the registers of the
// warp: the squared bond length quantized to the 23 bits above the nine bits
// of the candidate slot, which resolves lengths far finer than any thermal
// displacement. A rejected candidate carries the largest key and sorts behind
// every survivor, so the sorted position of a survivor is its slot in the row;
// the survivors are sorted on the fewest registers that hold them. The order
// of the candidates in the neighbor list varies from build to build, and so
// does the order of the local atoms, so survivors of equal quantized length
// are ordered by the tag of their source atom: a row whose sorted keys show
// such a tie is sorted again on 64-bit keys that insert the low 32 bits of the
// tag between the length and the slot. Such ties arise in symmetric
// configurations, such as a perfect lattice, and almost never in a thermal
// one. A row holds every tag once when the box is thicker than twice the
// cutoff along every periodic direction, as the folded node set of a single
// rank requires, and the row order, and with it the summation order of every
// reduction over the row, is then a function of the geometry and the atom
// tags alone, which makes the forces reproducible from run to run. In a
// thinner box, two images of one atom in a row tie on the tag as well and
// keep the order of their slots.
//
// The warps also refresh the artifact's node types, which a center writes
// for its own node and the halo nodes of the extended set take in a stride
// over the centers, and the last center keeps the edge storage past the
// physical end of the graph cleared: it zeroes the rows that the previous
// build used and this one does not, up to the watermark ``fill_total``.
//
// Only a CUDA device pass compiles the body. The selection is made inside
// this function rather than around its call because nvcc orders the captures
// of an extended lambda by their first use in its body, so a body that
// differs between the host and the device pass would name a different kernel
// on each side.
template <int K,
          int Warps,
          class Neighbors,
          class Indices,
          class ModelMap,
          class Tags,
          class Packed,
          class TypeMap,
          class RowPointers,
          class Sources,
          class Vectors,
          class Counts>
KOKKOS_INLINE_FUNCTION void compact_canonical_fill_row(
    const int rank,
    unsigned long long* ticket_counter,
    const unsigned long long ticket_base,
    unsigned long long* status,
    const unsigned epoch,
    const int inum,
    const int tail_begin,
    const int tail_end,
    const std::int64_t capacity,
    std::int64_t* fill_total,
    const float cutsq,
    const float key_scale,
    const float vector_scale,
    const Indices& numneigh,
    const Neighbors& neighbors,
    const ModelMap& loc2model,
    const ModelMap& model2loc,
    const Tags& tag,
    const Packed& packed,
    const Indices& type,
    const TypeMap& type_map,
    const RowPointers& model_type,
    const RowPointers& destination_row_ptr,
    const Sources& source,
    const Vectors& edge_vec,
    const bool count_sources,
    const Counts& source_counts) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)
  constexpr unsigned kFull = 0xffffffffu;
  constexpr std::uint32_t kRejected = ~std::uint32_t{0};
  // Highest length level of a survivor, one below the levels the key holds.
  constexpr std::uint32_t kTopLevel =
      (std::uint32_t{1} << (32 - kCompactCanonicalSlotBits)) - 2u;
  constexpr unsigned long long kAggregate = 1ull << 32;
  constexpr unsigned long long kInclusive = 2ull << 32;
  // Candidate slots of one row; the row stride adds four, which spreads the
  // stores of a warp, eight centers by four slots, over all 32 banks.
  constexpr int kSlots = 32 * K;
  constexpr int kSlotStride = kSlots + 4;
  const unsigned long long stamp = static_cast<unsigned long long>(epoch) << 34;

  // === Take the ticket that orders this block ===
  const int lane = rank & 31;
  const int warp = rank >> 5;
  __shared__ unsigned long long block_ticket;
  __shared__ unsigned center_count[Warps];
  __shared__ unsigned long long block_offset;
  __shared__ int claim;
  __shared__ int ready;
  __shared__ alignas(16) int slot_atom[Warps][kSlotStride];
  __shared__ std::uint32_t staging[Warps][256];
  if (rank == 0) {
    block_ticket = atomicAdd(ticket_counter, 1ull);
    claim = 0;
    ready = 0;
  }
  __syncthreads();
  const int block = static_cast<int>(block_ticket - ticket_base);
  const int center = block * Warps + warp;
  const bool active = center < inum;
  const int i = active ? center : 0;

  // === Stage the candidate entries of every center of the block ===
  // Thread ``rank`` stages slots ``rank / Warps + 32 s`` of center
  // ``rank % Warps``, so a warp reads one sector per slot, holding the slot
  // for all centers, and lane ``warp`` of every warp stages the warp's own
  // center, whose model node and candidate count it hands the other lanes.
  // The model nodes, the candidate counts and the center types are requested
  // together. The entries are copied as the list holds them: every reader
  // masks off the special-bond bits, and the key pass passes over the slots
  // past the candidates of its center. Every copy of a thread is in flight
  // before the first completes: on a device of compute capability 8.0 or
  // higher as asynchronous copies, which hold no registers, and otherwise as
  // loads that are all issued before the first is stored.
  static_assert(32 % Warps == 0, "a warp covers whole slots of the block");
  const int c = rank % Warps;
  const int atom_c = block * Warps + c;
  const int model_c = atom_c < inum ? loc2model(atom_c) : -1;
  const int listed_c = atom_c < inum ? numneigh(atom_c) : 0;
  const int type_i = active ? type(i) : 1;
  const int candidates_c = model_c >= 0 ? listed_c : 0;
  {
    const int first_slot = rank / Warps;
#if __CUDA_ARCH__ >= 800
#pragma unroll
    for (int s = 0; s < K; ++s) {
      const int jj = first_slot + 32 * s;
      if (jj < candidates_c) {
        compact_canonical_copy_async(&slot_atom[c][jj], &neighbors(atom_c, jj));
      }
    }
#else
    int entry[K];
#pragma unroll
    for (int s = 0; s < K; ++s) {
      const int jj = first_slot + 32 * s;
      entry[s] = jj < candidates_c ? neighbors(atom_c, jj) : 0;
    }
#pragma unroll
    for (int s = 0; s < K; ++s) {
      const int jj = first_slot + 32 * s;
      if (jj < candidates_c) {
        slot_atom[c][jj] = entry[s];
      }
    }
#endif
  }
  const int mi = __shfl_sync(kFull, model_c, warp);
  const int candidates = __shfl_sync(kFull, candidates_c, warp);

  // === Refresh the node types before the staged entries are awaited ===
  if (mi >= 0 && lane == 0) {
    model_type(mi) = type_map(type_i - 1);
  }
  if (active) {
    for (std::int64_t m = tail_begin + std::int64_t{32} * center + lane;
         m < tail_end; m += std::int64_t{32} * inum) {
      model_type(m) = type_map(type(model2loc(m)) - 1);
    }
  }
#if __CUDA_ARCH__ >= 800
  compact_canonical_await_copies();
#endif
  __syncthreads();
  const int* const row_atom = slot_atom[warp];
  const int4 position_i = compact_canonical_record(packed, i);

  // === Key every candidate and count the survivors ===
  // The records of four slots are requested before the first of them is
  // used, so the gathers of a lane overlap.
  constexpr int kBatch = 4;
  static_assert(K % kBatch == 0, "the slots split into whole batches");
  std::uint32_t key[K];
  unsigned count = 0;
#pragma unroll
  for (int first_slot = 0; first_slot < K; first_slot += kBatch) {
    int4 record[kBatch];
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      const int slot = (first_slot + u) * 32 + lane;
      record[u] = slot < candidates ? compact_canonical_record(
                                          packed, row_atom[slot] & NEIGHMASK)
                                    : make_int4(0, 0, 0, -1);
    }
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      const int k = first_slot + u;
      const float dx =
          compact_canonical_displacement(record[u].x, position_i.x);
      const float dy =
          compact_canonical_displacement(record[u].y, position_i.y);
      const float dz =
          compact_canonical_displacement(record[u].z, position_i.z);
      const float length2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
      // The level of a survivor stays below the top one, whatever the
      // rounding of the scale, so that its key never equals the rejection
      // value.
      const std::uint32_t level =
          min(static_cast<std::uint32_t>(length2 * key_scale), kTopLevel);
      key[k] = record[u].w >= 0 && length2 < cutsq
                   ? (level << kCompactCanonicalSlotBits) |
                         static_cast<std::uint32_t>(k * 32 + lane)
                   : kRejected;
      count += __popc(__ballot_sync(kFull, key[k] != kRejected));
    }
  }
  if (lane == 0) {
    center_count[warp] = count;
  }
  __syncthreads();
  unsigned row_offset = 0;
  unsigned block_total = 0;
#pragma unroll
  for (int w = 0; w < Warps; ++w) {
    const unsigned value = center_count[w];
    row_offset += w < warp ? value : 0u;
    block_total += value;
  }

  // === Publish the block total ===
  if (warp == 0 && lane == 0) {
    *(static_cast<volatile unsigned long long*>(status) + block) =
        stamp | (block == 0 ? kInclusive : kAggregate) | block_total;
  }
  if (!active) {
    return;
  }

  // === Place, order and emit the row on the fewest registers that hold it ===
  // Every block has the center of its first warp, so some warp of the block
  // resolves the block offset.
  const CompactCanonicalScan scan{block,  stamp,         status, block_total,
                                  &claim, &block_offset, &ready};
#define DEEPMD_COMPACT_CANONICAL_FINISH(KS)                               \
  compact_canonical_finish_row<KS, K>(                                    \
      key, count, row_offset, scan, lane, staging[warp], slot_atom[warp], \
      center, mi, position_i, inum, tail_begin, tail_end, capacity,       \
      fill_total, vector_scale, tag, packed, destination_row_ptr, source, \
      edge_vec, count_sources, source_counts)
  if (count <= 128u) {
    DEEPMD_COMPACT_CANONICAL_FINISH(4);
  } else if (K > 8 && count <= 256u) {
    DEEPMD_COMPACT_CANONICAL_FINISH(8);
  } else {
    DEEPMD_COMPACT_CANONICAL_FINISH((K > 8 ? 16 : 8));
  }
#undef DEEPMD_COMPACT_CANONICAL_FINISH
#endif
}

#if defined(KOKKOS_ENABLE_CUDA)
// Sort the ``count`` distinct values of a row in place, ascending, with the
// warp holding them ``K`` per lane in registers; the slots past the row hold
// the largest value and sort behind it.
template <int K>
__device__ __forceinline__ void compact_canonical_sort_in_registers(
    std::uint32_t* row, const int count, const int lane) {
  std::uint32_t key[K];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int index = k * 32 + lane;
    key[k] = index < count ? row[index] : ~std::uint32_t{0};
  }
  compact_canonical_warp_sort<K, CompactCanonicalLayout::kRegisterMajor>(key,
                                                                         lane);
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int index = k * 32 + lane;
    if (index < count) {
      row[index] = key[k];
    }
  }
}

// Sort a row too long for the registers of a warp in place, ascending, by
// odd-even transposition: ``count`` rounds of disjoint neighbour exchanges,
// which the lanes of the warp share.
__device__ inline void compact_canonical_sort_in_memory(std::uint32_t* row,
                                                        const long count,
                                                        const int lane) {
  for (long round = 0; round < count; ++round) {
    for (long index = (round & 1) + 2 * lane; index + 1 < count; index += 64) {
      const std::uint32_t low = row[index];
      const std::uint32_t high = row[index + 1];
      if (low > high) {
        row[index] = high;
        row[index + 1] = low;
      }
    }
    __syncwarp();
  }
}
#endif

// Sort the rows of ``Warps`` consecutive nodes of the source CSR, one warp per
// row, in the block of league rank ``league``. A row of up to 512 positions,
// which holds every row of a node whose candidates the single-pass fill
// serves, is sorted in the registers of its warp; a longer one in memory.
//
// Only a CUDA device pass compiles the body, for the reason given at
// compact_canonical_fill_row.
template <int Warps, class RowPointers, class Order>
KOKKOS_INLINE_FUNCTION void compact_canonical_sort_source_rows(
    const int rank,
    const int league,
    const int node_count,
    const RowPointers& source_row_ptr,
    const Order& source_order) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)
  const int lane = rank & 31;
  const int node = league * Warps + (rank >> 5);
  if (node >= node_count) {
    return;
  }
  const std::int64_t begin = source_row_ptr(node);
  const std::int64_t count = source_row_ptr(node + 1) - begin;
  std::uint32_t* row = source_order.data() + begin;
  if (count <= 1) {
    return;
  }
  if (count <= 128) {
    compact_canonical_sort_in_registers<4>(row, static_cast<int>(count), lane);
  } else if (count <= 256) {
    compact_canonical_sort_in_registers<8>(row, static_cast<int>(count), lane);
  } else if (count <= 512) {
    compact_canonical_sort_in_registers<16>(row, static_cast<int>(count), lane);
  } else {
    compact_canonical_sort_in_memory(row, count, lane);
  }
#endif
}

// Accelerator stream of an execution space, which a device-resident model
// evaluation is enqueued on so that it is ordered after the graph build and
// before the force scatter without a host synchronization. Spaces without a
// stream of their own yield null, which asks the model to synchronize.
template <class ExecutionSpace>
void* compact_canonical_stream(const ExecutionSpace&) {
  return nullptr;
}
#if defined(KOKKOS_ENABLE_CUDA)
inline void* compact_canonical_stream(const Kokkos::Cuda& space) {
  return space.cuda_stream();
}
#endif

// Device-resident model node set and compact canonical neighbor graph.
//
// A ``.pt2`` artifact frozen with the compact canonical lower consumes a
// destination-major topology over a contiguous range of model nodes: uint32
// source indices, float32 edge vectors and an int64 row pointer per node. The
// source-major CSR -- a second row pointer and the permutation that reorders
// the edge list into source order -- is built only on request, for an
// artifact whose force assembly groups edges by source or for an evaluation
// that wants the source-attributed per-atom virial. This component derives
// the graph from the Kokkos device neighbor list and leaves it on the device,
// so a pair style hands the model raw device pointers and never stages the
// graph through the host.
//
// The node set is the model's view of the atoms. Atom types that map to no
// model element (NULL in ``pair_coeff``) are virtual: they carry no node, so
// the surviving atoms are compacted into a contiguous node index range, which
// degenerates to the identity when no type maps to NULL. A single rank folds
// every ghost onto the local atom that owns it and works on the minimum-image
// node set, which requires the box to be thicker than twice the cutoff along
// every periodic direction. Domain decomposition instead gives every real
// ghost a node of its own -- the extended node set -- and leaves the pair
// style to fold the ghost outputs onto their owners by reverse communication.
//
// build() is one function, and public, because CUDA forbids extended device
// lambdas inside non-public members: its stages that launch device lambdas
// cannot be split into private helpers.
template <class DeviceType>
class CompactCanonicalGraphKokkos : protected Pointers {
 public:
  typedef ArrayTypes<DeviceType> AT;

  CompactCanonicalGraphKokkos(class LAMMPS* lmp)
      : Pointers(lmp),
        nloc_model(0),
        nnode_model(0),
        has_null_types(false),
        storage_count(0),
        execution_space(ExecutionSpaceFromDevice<DeviceType>::space),
        extended_nodes(false),
        edge_capacity(0) {}

  // Cache the LAMMPS type (1-based) -> model type map on the device and select
  // the node-set mode; ``extended`` gives every real ghost a node of its own.
  // The map comes from the pair style's coeff(), so this runs once the styles
  // know their model, before the first graph is built.
  void setup(const std::vector<int>& type_idx_map, bool extended) {
    host_type_map = type_idx_map;
    extended_nodes = extended;
    has_null_types = false;
    const int ntypes = static_cast<int>(host_type_map.size());
    d_type_map =
        Kokkos::View<int*, DeviceType>("compact_canonical:type_map", ntypes);
    auto h_type_map = Kokkos::create_mirror_view(d_type_map);
    for (int t = 0; t < ntypes; ++t) {
      if (host_type_map[t] < 0) {
        has_null_types = true;  // some LAMMPS type is a virtual (NULL) atom
      }
      h_type_map(t) = host_type_map[t];
    }
    Kokkos::deep_copy(d_type_map, h_type_map);
    // The pair styles construct the builder before LAMMPS confirms that
    // KOKKOS is enabled, so its views are allocated here, once; the fill
    // watermark starts at zero, as fresh storage is.
    if (!d_fill_total.is_allocated()) {
      d_fill_total = Kokkos::View<std::int64_t, DeviceType>(
          "compact_canonical:fill_total");
      h_edge_count = Kokkos::View<std::int64_t, Kokkos::SharedHostPinnedSpace>(
          "compact_canonical:edge_count");
    }
  }

  // Rebuild the atom -> node compaction, which only moves when the neighbor
  // list is rebuilt or the atom count outgrows the maps. A pair style that
  // builds its own graph over the same node set calls this directly; build()
  // calls it for the compact canonical graph.
  void refresh_nodes() {
    const int nlocal = atom->nlocal;
    const int nall = atom->nlocal + atom->nghost;

    if (neighbor->ago != 0 && (int)k_loc2model.extent(0) >= nall) {
      return;
    }
    if ((int)k_candidate_to_model.extent(0) < nall) {
      k_candidate_to_model =
          DAT::tdual_int_1d("compact_canonical:candidate_to_model", nall);
    }
    if ((int)k_loc2model.extent(0) < nall) {
      k_loc2model = DAT::tdual_int_1d("compact_canonical:loc2model", nall);
      k_model2loc = DAT::tdual_int_1d("compact_canonical:model2loc", nall);
    }
    atomKK->sync(Host, TAG_MASK | TYPE_MASK);
    auto h_loc2model = k_loc2model.view_host();
    auto h_model2loc = k_model2loc.view_host();
    const int* lmp_type = atom->type;
    int m = 0;
    for (int i = 0; i < nlocal; ++i) {
      if (host_type_map[lmp_type[i] - 1] >= 0) {
        h_loc2model(i) = m;
        h_model2loc(m) = i;
        ++m;
      } else {
        h_loc2model(i) = -1;
      }
    }
    nloc_model = m;
    for (int j = nlocal; j < nall; ++j) {
      if (extended_nodes && host_type_map[lmp_type[j] - 1] >= 0) {
        h_loc2model(j) = m;
        h_model2loc(m) = j;
        ++m;
      } else {
        h_loc2model(j) = -1;
      }
    }
    nnode_model = m;

    // Resolve each candidate atom to its model node once, on the host. In the
    // folded representation a ghost contributes to the node of the local atom
    // that owns it, so the resolution is a composition of the ownership map
    // with the model map; the extended representation gives ghosts their own
    // nodes and the composition degenerates to the model map. Collapsing it
    // here leaves the device traversal, which visits every candidate of every
    // center, with a single gather.
    auto h_candidate_to_model = k_candidate_to_model.view_host();
    if (extended_nodes) {
      for (int j = 0; j < nall; ++j) {
        h_candidate_to_model(j) = h_loc2model(j);
      }
    } else {
      for (int j = 0; j < nall; ++j) {
        const int owner = (j < nlocal) ? j : atom->map(atom->tag[j]);
        h_candidate_to_model(j) = owner < 0 ? -1 : h_loc2model(owner);
      }
    }
    k_candidate_to_model.template modify<LMPHostType>();
    k_candidate_to_model.template sync<DeviceType>();
    d_candidate_to_model = k_candidate_to_model.template view<DeviceType>();
    k_loc2model.template modify<LMPHostType>();
    k_loc2model.template sync<DeviceType>();
    d_loc2model = k_loc2model.template view<DeviceType>();
    k_model2loc.template modify<LMPHostType>();
    k_model2loc.template sync<DeviceType>();
    d_model2loc = k_model2loc.template view<DeviceType>();
  }

  // Build the compact canonical graph of the current configuration from the
  // Kokkos full neighbor list, with the source-major CSR when ``source_csr``
  // asks for it. Bond vectors are the center-to-neighbor displacements divided
  // by ``dist_unit_cvt_factor``, the same conversion the pair style applies to
  // coordinates it hands the model.
  void build(class NeighList* list,
             double cutoff,
             double dist_unit_cvt_factor,
             bool source_csr) {
    refresh_nodes();

    auto* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
    const int inum = k_list->inum;
    auto d_numneigh = k_list->d_numneigh;
    auto d_neighbors = k_list->d_neighbors;
    auto d_ilist = k_list->d_ilist;
    auto loc2model = d_loc2model;
    auto candidate_to_model = d_candidate_to_model;
    auto model2loc = d_model2loc;
    const double cutsq = cutoff * cutoff;
    const double inv_dist = 1.0 / dist_unit_cvt_factor;
    const int node_count_int = nnode_model;
    const std::size_t node_count = static_cast<std::size_t>(node_count_int);
    const int nall = atom->nlocal + atom->nghost;

    // === Node types in the artifact's index layout ===
    // The single-pass fill writes them alongside the rows; every other
    // configuration maps them here.
    atomKK->sync(execution_space, TYPE_MASK);
    auto type = atomKK->k_type.template view<DeviceType>();
    auto type_map = d_type_map;
    if ((int)d_model_type.extent(0) < node_count_int) {
      d_model_type = Kokkos::View<std::int64_t*, DeviceType>(
          "compact_canonical:model_type", nall);
    }
    auto model_type = d_model_type;

    atomKK->sync(execution_space, X_MASK | TAG_MASK);
    auto x = atomKK->k_x.template view<DeviceType>();
    auto tag = atomKK->k_tag.template view<DeviceType>();

    if (d_destination_row_ptr.extent(0) < node_count + 1) {
      d_destination_row_ptr = Kokkos::View<std::int64_t*, DeviceType>(
          "compact_canonical:destination_row_ptr", node_count + 1);
    }
    if (source_csr && d_source_row_ptr.extent(0) < node_count + 1) {
      d_source_counts = Kokkos::View<std::uint32_t*, DeviceType>(
          "compact_canonical:source_counts", node_count);
      d_source_row_ptr = Kokkos::View<std::int64_t*, DeviceType>(
          "compact_canonical:source_row_ptr", node_count + 1);
      d_source_cursor = Kokkos::View<std::uint32_t*, DeviceType>(
          "compact_canonical:source_cursor", node_count);
    }
    if (source_csr) {
      Kokkos::deep_copy(typename DeviceType::execution_space(), d_source_counts,
                        std::uint32_t{0});
    }
    if (node_count_int == 0) {
      storage_count = 0;
      return;
    }
    auto destination_row_ptr = d_destination_row_ptr;
    auto source_counts = d_source_counts;

    // === Destination-major fill, one warp per center, by bond length ===
    // Every row is emitted in increasing bond length. A compressed descriptor
    // reads one radial-table interval per edge, and rows of one length order
    // make the concurrently resident warps of a multiprocessor, each scanning
    // its own center, address the same intervals at the same time, so the
    // table is served from L1 instead of being streamed from L2 once per edge.
    // In a box thicker than twice the cutoff, where a row holds every tag
    // once, the order follows from the geometry and the atom tags alone, which
    // also fixes the summation order of every destination reduction
    // independently of the neighbor list (see compact_canonical_fill_row).
    //
    // The team fill forms one 64-bit sort key per survivor: the squared length
    // quantized to 16 bits, then the tag of the source atom, then the
    // candidate slot. The keys are sorted and the survivors written at
    // consecutive slots, so the dominant store stream stays coalesced; the
    // geometry of a survivor is recomputed from its slot, whose coordinates
    // the key pass has just brought into cache. The single-pass fill keys a
    // survivor by its squared length quantized to 23 bits and its slot, and
    // orders survivors of equal quantized length by tag only in the rows that
    // hold such ties. The two orders differ.
    //
    // On a CUDA device whose candidate lists fit 512 entries, the warp of a
    // center sorts in registers and the row offsets come from the decoupled
    // look-back of compact_canonical_fill_row, so counting, offsets and the
    // fill are one pass. Every other configuration counts the rows first,
    // scans the counts, and fills from team scratch.
    constexpr int neighbor_lanes = 32;  // lanes cooperating on one center
    // Consecutive centers filled by one block of the single-pass fill: eight,
    // so that one 32-byte sector of the neighbor list holds a candidate slot
    // of every center of the block.
    constexpr int fill_warps = 8;
    static_assert(16 * neighbor_lanes <= (1 << kCompactCanonicalSlotBits),
                  "the single-pass sort keys must address every candidate");
    const int key_capacity = static_cast<int>(d_neighbors.extent(1));
    if (key_capacity >= (1 << 16)) {
      error->one(FLERR,
                 "Compact canonical graph supports at most 65535 neighbor "
                 "candidates per atom");
    }
    using team_policy = Kokkos::TeamPolicy<DeviceType>;
    // The single-pass fill runs one warp per center and eight consecutive
    // centers per team. A team that keys up to twelve candidates per lane
    // stages them in about 21 KB of shared memory, so four teams, 32 warps,
    // stay resident per multiprocessor and hide the latency of the staging
    // and the gathers; one that keys sixteen takes 25 KB, and three teams, 24
    // warps, fit. Either bound holds the registers of the whole kernel at the
    // budget of its common path rather than at that of the rarely taken
    // tie-break sort.
    using fill_policy = Kokkos::TeamPolicy<
        DeviceType,
        Kokkos::LaunchBounds<fill_warps * neighbor_lanes, 32 / fill_warps>>;
    using wide_fill_policy = Kokkos::TeamPolicy<
        DeviceType,
        Kokkos::LaunchBounds<fill_warps * neighbor_lanes, 24 / fill_warps>>;
    using member_type = typename team_policy::member_type;
    // Squared lengths quantize to the key bits above the slot: 23 bits in the
    // single-pass fill and 16 bits in the 64-bit keys of the team fill.
    const double fill_key_levels =
        static_cast<double>((1u << (32 - kCompactCanonicalSlotBits)) - 1u);
    const double key_scale = 65535.0 / cutsq;
    // The candidate lists move only with the neighbor list: their longest
    // entry selects the fill, and the candidates of the model centers bound
    // the edge count until the next rebuild.
    if (neighbor->ago == 0 || longest_candidates < 0) {
      int longest = 0;
      std::int64_t total = 0;
      Kokkos::parallel_reduce(
          "compact_canonical:candidates",
          Kokkos::RangePolicy<DeviceType>(0, inum),
          KOKKOS_LAMBDA(const int ii, int& value, std::int64_t& sum) {
            const int i = d_ilist(ii);
            const int jnum = d_numneigh(i);
            value = jnum > value ? jnum : value;
            sum += loc2model(i) >= 0 ? jnum : 0;
          },
          Kokkos::Max<int>(longest), Kokkos::Sum<std::int64_t>(total));
      longest_candidates = longest;
      candidate_total = total;
    }
    std::int64_t edge_count = 0;
    // The single-pass fill lays the rows out in the order of the centers and
    // writes the row pointer and the type of a local node from its own center,
    // so it serves a list whose centers are all the local atoms in index
    // order, `ilist(ii) == ii`, as the Kokkos neighbor lists give them, and
    // takes center `ii` for local atom `ii`; a list that leaves some out
    // (`neigh_modify include`) takes the counted fill.
    bool single_pass = false;
#if defined(KOKKOS_ENABLE_CUDA)
    single_pass =
        std::is_same_v<typename DeviceType::execution_space, Kokkos::Cuda> &&
        inum > 0 && inum == atom->nlocal &&
        longest_candidates <= 16 * neighbor_lanes;
#endif
    if (single_pass) {
      // === Pack every atom the fill can visit into one aligned record ===
      // The fill works on fixed-point coordinates, the low 32 bits of
      // round(x / step) for a step of 2^-position_bits length units, the
      // finest power of two for which 2^31 steps span four neighbor cutoffs.
      // The wrapped difference of two coordinates is then the exact
      // fixed-point displacement of any pair closer than four neighbor
      // cutoffs, twice the farthest a candidate drifts from its center while
      // its neighbor list is valid, so every bond vector and squared length of
      // the fill is formed in single precision from integers. A bond vector
      // carries, on top of its own single-precision rounding, the rounding of
      // its two coordinates to the step: at most one step, 1.5e-8 length
      // units at a neighbor cutoff of 8.
      const int position_bits =
          std::ilogb(std::ldexp(1.0, 29) / neighbor->cutneighmax);
      const double position_scale = std::ldexp(1.0, position_bits);
      const double step = std::ldexp(1.0, -position_bits);
      if ((int)d_packed.extent(0) < nall) {
        d_packed = Kokkos::View<int* [4], Kokkos::LayoutRight, DeviceType>(
            "compact_canonical:packed", nall);
      }
      auto packed = d_packed;
      Kokkos::parallel_for(
          "compact_canonical:pack", Kokkos::RangePolicy<DeviceType>(0, nall),
          KOKKOS_LAMBDA(const int a) {
            for (int d = 0; d < 3; ++d) {
              packed(a, d) = static_cast<int>(
                  static_cast<std::uint32_t>(static_cast<std::int64_t>(
                      Kokkos::rint(x(a, d) * position_scale))));
            }
            packed(a, 3) = candidate_to_model(a);
          });
      // The cutoff and the key scale in squared fixed-point steps. The scale
      // derives from the rounded cutoff, which keeps every key of a squared
      // length below that cutoff within the key bits.
      const float fill_cutsq =
          static_cast<float>(cutsq * position_scale * position_scale);
      const float fill_key_scale =
          static_cast<float>(fill_key_levels / fill_cutsq);
      const float vector_scale = static_cast<float>(step * inv_dist);
      const int teams = (inum + fill_warps - 1) / fill_warps;
      if ((int)d_fill_status.extent(0) < teams) {
        d_fill_status = Kokkos::View<unsigned long long*, DeviceType>(
            "compact_canonical:fill_status", teams);
      }
      if (d_fill_ticket.extent(0) == 0) {
        d_fill_ticket = Kokkos::View<unsigned long long*, DeviceType>(
            "compact_canonical:fill_ticket", 1);
        ticket_base = 0;
      }
      const int tail_begin = nloc_model;
      // Edges are a subset of the candidates, so storage sized by the
      // candidate total cannot overflow before the next rebuild and the edge
      // count can stay on the device. Systems whose candidate storage exceeds
      // the bound read the count back instead, which their step amortizes.
      const bool bounded =
          !source_csr &&
          static_cast<std::uint64_t>(candidate_total) * edge_storage_bytes <=
              bounded_storage_bytes();
      if (bounded) {
        reserve_edges(candidate_total);
      }
      for (;;) {
        // The epoch distinguishes the status words of this build from those
        // of every earlier one, and the ticket counter runs on across builds.
        fill_epoch = (fill_epoch + 1) & ((1u << 30) - 1);
        const unsigned epoch = fill_epoch;
        const unsigned long long base = ticket_base;
        // A block takes one ticket, so the tickets of one build number its
        // blocks.
        ticket_base += static_cast<unsigned long long>(teams);
        unsigned long long* ticket_counter = d_fill_ticket.data();
        unsigned long long* status = d_fill_status.data();
        std::int64_t* fill_total = d_fill_total.data();
        const std::int64_t capacity = static_cast<std::int64_t>(edge_capacity);
        auto source = d_source;
        auto edge_vec = d_edge_vec;
        if (longest_candidates <= 8 * neighbor_lanes) {
          Kokkos::parallel_for(
              "compact_canonical:fill",
              fill_policy(teams, fill_warps * neighbor_lanes),
              KOKKOS_LAMBDA(const typename fill_policy::member_type& team) {
                compact_canonical_fill_row<8, fill_warps>(
                    team.team_rank(), ticket_counter, base, status, epoch, inum,
                    tail_begin, node_count_int, capacity, fill_total,
                    fill_cutsq, fill_key_scale, vector_scale, d_numneigh,
                    d_neighbors, loc2model, model2loc, tag, packed, type,
                    type_map, model_type, destination_row_ptr, source, edge_vec,
                    source_csr, source_counts);
              });
        } else if (longest_candidates <= 12 * neighbor_lanes) {
          Kokkos::parallel_for(
              "compact_canonical:fill",
              fill_policy(teams, fill_warps * neighbor_lanes),
              KOKKOS_LAMBDA(const typename fill_policy::member_type& team) {
                compact_canonical_fill_row<12, fill_warps>(
                    team.team_rank(), ticket_counter, base, status, epoch, inum,
                    tail_begin, node_count_int, capacity, fill_total,
                    fill_cutsq, fill_key_scale, vector_scale, d_numneigh,
                    d_neighbors, loc2model, model2loc, tag, packed, type,
                    type_map, model_type, destination_row_ptr, source, edge_vec,
                    source_csr, source_counts);
              });
        } else {
          Kokkos::parallel_for(
              "compact_canonical:fill",
              wide_fill_policy(teams, fill_warps * neighbor_lanes),
              KOKKOS_LAMBDA(
                  const typename wide_fill_policy::member_type& team) {
                compact_canonical_fill_row<16, fill_warps>(
                    team.team_rank(), ticket_counter, base, status, epoch, inum,
                    tail_begin, node_count_int, capacity, fill_total,
                    fill_cutsq, fill_key_scale, vector_scale, d_numneigh,
                    d_neighbors, loc2model, model2loc, tag, packed, type,
                    type_map, model_type, destination_row_ptr, source, edge_vec,
                    source_csr, source_counts);
              });
        }
        if (bounded) {
          break;
        }
        edge_count = read_edge_count(node_count_int);
        // The look-back saturates at the largest 32-bit value, so a total of
        // that value or more is a graph beyond the edge-index range.
        if (static_cast<std::uint64_t>(edge_count) >=
            std::numeric_limits<std::uint32_t>::max()) {
          error->one(FLERR,
                     "Compact canonical graph exceeds the uint32 edge-index "
                     "range");
        }
        if (edge_count <= capacity) {
          break;
        }
        // Edges past the capacity were dropped and the source histogram is
        // incomplete; the build repeats with arrays of the measured size.
        reserve_edges(edge_count);
        if (source_csr) {
          Kokkos::deep_copy(typename DeviceType::execution_space(),
                            d_source_counts, std::uint32_t{0});
        }
      }
    } else {
      Kokkos::parallel_for(
          "compact_canonical:node_type",
          Kokkos::RangePolicy<DeviceType>(0, node_count_int),
          KOKKOS_LAMBDA(const int m) {
            model_type(m) = type_map(type(model2loc(m)) - 1);
          });

      // === Per-node edge count, then its prefix sum ===
      Kokkos::deep_copy(typename DeviceType::execution_space(),
                        d_destination_row_ptr, std::int64_t{0});
      Kokkos::parallel_for(
          "compact_canonical:count", Kokkos::RangePolicy<DeviceType>(0, inum),
          KOKKOS_LAMBDA(const int ii) {
            const int i = d_ilist(ii);
            const int mi = loc2model(i);
            if (mi < 0) {
              return;
            }
            const double xi = x(i, 0);
            const double yi = x(i, 1);
            const double zi = x(i, 2);
            const int jnum = d_numneigh(i);
            std::int64_t count = 0;
            for (int jj = 0; jj < jnum; ++jj) {
              const int j = d_neighbors(i, jj) & NEIGHMASK;
              const int mj = candidate_to_model(j);
              if (mj < 0) {
                continue;
              }
              const double dx = x(j, 0) - xi;
              const double dy = x(j, 1) - yi;
              const double dz = x(j, 2) - zi;
              if (dx * dx + dy * dy + dz * dz < cutsq) {
                ++count;
              }
            }
            destination_row_ptr(mi) = count;
          });
      Kokkos::parallel_scan(
          "compact_canonical:destination_scan",
          Kokkos::RangePolicy<DeviceType>(0, node_count_int),
          KOKKOS_LAMBDA(const int node, std::int64_t& update,
                        const bool final) {
            const std::int64_t count = destination_row_ptr(node);
            if (final) {
              destination_row_ptr(node) = update;
            }
            update += count;
            if (final && node == node_count_int - 1) {
              destination_row_ptr(node_count_int) = update;
            }
          });
      edge_count = read_edge_count(node_count_int);
      reserve_edges(edge_count);

      // === Fill from team scratch ===
      auto source = d_source;
      auto edge_vec = d_edge_vec;
      using key_scratch =
          Kokkos::View<unsigned long long*,
                       typename DeviceType::scratch_memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      const int scratch_bytes = key_scratch::shmem_size(neighbor_lanes) +
                                key_scratch::shmem_size(key_capacity);
      // The keys of a row stay in the on-chip scratch of level 0 while the
      // candidate lists fit it, and in the device memory of level 1 beyond.
      team_policy fill_teams(inum, neighbor_lanes);
      const int scratch_level =
          scratch_bytes <= fill_teams.scratch_size_max(0) ? 0 : 1;
      Kokkos::parallel_for(
          "compact_canonical:fill",
          fill_teams.set_scratch_size(scratch_level,
                                      Kokkos::PerTeam(scratch_bytes)),
          KOKKOS_LAMBDA(const member_type& team) {
            const int i = d_ilist(team.league_rank());
            const int mi = loc2model(i);
            if (mi < 0) {
              return;
            }
            key_scratch lane_key(team.team_scratch(scratch_level),
                                 neighbor_lanes);
            key_scratch keys(team.team_scratch(scratch_level), key_capacity);

            const double xi = x(i, 0);
            const double yi = x(i, 1);
            const double zi = x(i, 2);
            const int jnum = d_numneigh(i);
            const int lane = team.team_rank();
            const std::int64_t edge = destination_row_ptr(mi);
            constexpr unsigned long long kRejected = ~0ull;

            // === Collect the sort keys of the survivors ===
            int count = 0;
            for (int base = 0; base < jnum; base += neighbor_lanes) {
              const int jj = base + lane;
              unsigned long long key = kRejected;
              if (jj < jnum) {
                const int j = d_neighbors(i, jj) & NEIGHMASK;
                if (candidate_to_model(j) >= 0) {
                  const double dx = x(j, 0) - xi;
                  const double dy = x(j, 1) - yi;
                  const double dz = x(j, 2) - zi;
                  const double length2 = dx * dx + dy * dy + dz * dz;
                  if (length2 < cutsq) {
                    key = (static_cast<unsigned long long>(length2 * key_scale)
                           << 48) |
                          (static_cast<unsigned long long>(
                               static_cast<std::uint32_t>(tag(j)))
                           << 16) |
                          static_cast<unsigned long long>(jj);
                  }
                }
              }
              lane_key(lane) = key;
              team.team_barrier();
              int kept = 0;
              Kokkos::parallel_scan(
                  Kokkos::TeamThreadRange(team, neighbor_lanes),
                  [&](const int slot, int& offset, const bool final) {
                    const unsigned long long value = lane_key(slot);
                    if (final && value != kRejected) {
                      keys(count + offset) = value;
                    }
                    offset += value != kRejected ? 1 : 0;
                  },
                  kept);
              count += kept;
              team.team_barrier();
            }
            Kokkos::Experimental::sort_team(
                team, Kokkos::subview(keys, Kokkos::pair<int, int>(0, count)));

            // === Emit the row in key order ===
            for (int slot = lane; slot < count; slot += neighbor_lanes) {
              const int jj = static_cast<int>(keys(slot) & 0xffffu);
              const int j = d_neighbors(i, jj) & NEIGHMASK;
              const int mj = candidate_to_model(j);
              const std::int64_t position = edge + slot;
              source(position) = static_cast<std::uint32_t>(mj);
              edge_vec(3 * position + 0) =
                  static_cast<float>((x(j, 0) - xi) * inv_dist);
              edge_vec(3 * position + 1) =
                  static_cast<float>((x(j, 1) - yi) * inv_dist);
              edge_vec(3 * position + 2) =
                  static_cast<float>((x(j, 2) - zi) * inv_dist);
              if (source_csr) {
                Kokkos::atomic_fetch_add(&source_counts(mj), std::uint32_t{1});
              }
            }
          });

      // === Guard edges past the physical end of the graph ===
      // The single-pass fill clears the rows it leaves behind itself; the
      // counted fill reserved the storage before it filled the rows.
      const auto storage = static_cast<std::int64_t>(edge_capacity);
      if (storage > edge_count) {
        Kokkos::parallel_for(
            "compact_canonical:guards",
            Kokkos::RangePolicy<DeviceType, Kokkos::IndexType<std::int64_t>>(
                edge_count, storage),
            KOKKOS_LAMBDA(const std::int64_t edge) {
              source(edge) = std::uint32_t{0};
              edge_vec(3 * edge + 0) = 0.0f;
              edge_vec(3 * edge + 1) = 0.0f;
              edge_vec(3 * edge + 2) = 0.0f;
            });
      }
      Kokkos::deep_copy(d_fill_total, edge_count);
    }
    reserve_edges(edge_count);
    storage_count = static_cast<std::int64_t>(edge_capacity);
    if (!source_csr) {
      return;
    }

    // === Source-major CSR and the permutation into source order ===
    if (d_source_order.extent(0) < edge_capacity) {
      d_source_order = Kokkos::View<std::uint32_t*, DeviceType>(
          "compact_canonical:source_order", edge_capacity);
    }
    auto source_row_ptr = d_source_row_ptr;
    Kokkos::parallel_scan(
        "compact_canonical:source_scan",
        Kokkos::RangePolicy<DeviceType>(0, node_count_int),
        KOKKOS_LAMBDA(const int node, std::int64_t& update, const bool final) {
          const std::int64_t count =
              static_cast<std::int64_t>(source_counts(node));
          if (final) {
            source_row_ptr(node) = update;
          }
          update += count;
          if (final && node == node_count_int - 1) {
            source_row_ptr(node_count_int) = update;
          }
        });
    auto source_cursor = d_source_cursor;
    Kokkos::parallel_for(
        "compact_canonical:source_cursor",
        Kokkos::RangePolicy<DeviceType>(0, node_count_int),
        KOKKOS_LAMBDA(const int node) {
          source_cursor(node) =
              static_cast<std::uint32_t>(source_row_ptr(node));
        });
    auto source = d_source;
    auto source_order = d_source_order;
    Kokkos::parallel_for(
        "compact_canonical:source_scatter",
        Kokkos::RangePolicy<DeviceType, Kokkos::IndexType<std::int64_t>>(
            0, edge_count),
        KOKKOS_LAMBDA(const std::int64_t edge) {
          const auto position = Kokkos::atomic_fetch_add(
              &source_cursor(source(edge)), std::uint32_t{1});
          source_order(position) = static_cast<std::uint32_t>(edge);
        });
    // The cursor places the edges of a source row in the order their threads
    // arrive, which varies from run to run, so every row is then sorted by
    // edge position. The source order is a function of the destination-major
    // graph alone, and the reductions that walk it -- the force assembly of
    // DPA1 and the source-attributed per-atom virial -- sum in the same order
    // from run to run. A row holds distinct positions, so the sorted row is
    // the same whichever sort produces it. On a CUDA device whose candidate
    // lists fit 512 entries, a warp sorts each row, in its registers up to
    // that length and in memory beyond it, which no row of such a list
    // reaches in practice; any other configuration sorts each row with a
    // team sort.
    bool warp_sort = false;
#if defined(KOKKOS_ENABLE_CUDA)
    warp_sort =
        std::is_same_v<typename DeviceType::execution_space, Kokkos::Cuda> &&
        longest_candidates <= 16 * neighbor_lanes;
#endif
    if (warp_sort) {
      constexpr int kSortWarps = 8;
      using sort_policy =
          Kokkos::TeamPolicy<DeviceType, Kokkos::LaunchBounds<kSortWarps * 32>>;
      Kokkos::parallel_for(
          "compact_canonical:source_order_sort",
          sort_policy((node_count_int + kSortWarps - 1) / kSortWarps,
                      kSortWarps * 32),
          KOKKOS_LAMBDA(const typename sort_policy::member_type& team) {
            compact_canonical_sort_source_rows<kSortWarps>(
                team.team_rank(), team.league_rank(), node_count_int,
                source_row_ptr, source_order);
          });
    } else {
      using sort_policy = Kokkos::TeamPolicy<DeviceType>;
      constexpr int sort_lanes = 32;
      Kokkos::parallel_for(
          "compact_canonical:source_order_sort",
          sort_policy(node_count_int, sort_lanes),
          KOKKOS_LAMBDA(const typename sort_policy::member_type& team) {
            const int node = team.league_rank();
            const auto row = Kokkos::make_pair(source_row_ptr(node),
                                               source_row_ptr(node + 1));
            Kokkos::Experimental::sort_team(team,
                                            Kokkos::subview(source_order, row));
          });
    }
    if (storage_count > edge_count) {
      Kokkos::parallel_for(
          "compact_canonical:source_order_guards",
          Kokkos::RangePolicy<DeviceType, Kokkos::IndexType<std::int64_t>>(
              edge_count, storage_count),
          KOKKOS_LAMBDA(const std::int64_t edge) {
            source_order(edge) = static_cast<std::uint32_t>(edge);
          });
    }
  }

  // === Node set, valid after refresh_nodes() ===
  int nloc_model;   // real local model nodes; the energy is summed over these
  int nnode_model;  // total model nodes (== nloc_model folded; + ghost
                    // extended)
  bool has_null_types;  // some LAMMPS type maps to no model element
  // LAMMPS type (1-based) -> model type, resident on the device.
  Kokkos::View<int*, DeviceType> d_type_map;
  DAT::tdual_int_1d k_model2loc;      // (nall) model node index -> atom index
  typename AT::t_int_1d d_loc2model;  // (nall) atom -> model node index, or -1
  typename AT::t_int_1d d_model2loc;
  // (nall) candidate atom -> model node index, or -1, with a ghost already
  // folded onto its owner when the node set is not extended.
  typename AT::t_int_1d d_candidate_to_model;

  // === Compact canonical artifact, valid after build() ===
  // The edge arrays are handed over at their allocated extent,
  // ``storage_count`` rows, rather than at the edge count: the edge axis keeps
  // its length while the count fluctuates from step to step, so an unchanged
  // evaluation can be replayed, and it always meets the lower bound of two
  // that the traced program declares. Rows past the physical end of the graph
  // are guards that carry a zero bond vector on node zero.
  std::int64_t storage_count;
  Kokkos::View<std::int64_t*, DeviceType> d_model_type;  // (nnode_model)
  Kokkos::View<std::uint32_t*, DeviceType> d_source;     // (storage_count)
  Kokkos::View<float*, DeviceType> d_edge_vec;           // (3 * storage_count)
  Kokkos::View<std::int64_t*, DeviceType> d_destination_row_ptr;
  // Source-major CSR, current only after a build() that asked for it.
  Kokkos::View<std::int64_t*, DeviceType> d_source_row_ptr;
  Kokkos::View<std::uint32_t*, DeviceType> d_source_order;

 private:
  // Shortest edge axis the traced program accepts.
  static constexpr std::int64_t min_storage_edges = 2;
  // Bytes of edge storage per edge: one source index and one bond vector.
  static constexpr std::uint64_t edge_storage_bytes =
      sizeof(std::uint32_t) + 3 * sizeof(float);
  // Largest edge storage sized by the candidate total rather than by the
  // edge count read back after the fill: 1/64 of the device memory. The
  // readback holds the host until the fill ends, so the device idles while
  // the host launches the model, about 25 us per step on compute capability
  // 12.0, which is 1% of the step of a system of 150,000 atoms of the
  // smallest grade at a neighbor skin of 1 Å; the candidate storage of such a
  // system takes 0.7 GB, 0.3 GB more than storage sized by its edges.
  std::uint64_t bounded_storage_bytes() const {
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<typename DeviceType::execution_space,
                                 Kokkos::Cuda>) {
      return typename DeviceType::execution_space()
                 .cuda_device_prop()
                 .totalGlobalMem /
             64;
    }
#endif
    return 0;
  }

  // Read the physical edge count of the graph just filled. The copy goes
  // through pinned memory on the execution space instance, so the host waits
  // for the graph build alone instead of fencing the whole device.
  std::int64_t read_edge_count(const int node_count) {
    const typename DeviceType::execution_space instance;
    Kokkos::deep_copy(instance, h_edge_count,
                      Kokkos::subview(d_destination_row_ptr, node_count));
    instance.fence("compact_canonical:edge_count");
    return h_edge_count();
  }

  // Grow the edge arrays to hold ``edge_count`` edges; storage never shrinks.
  void reserve_edges(const std::int64_t edge_count) {
    const std::int64_t storage =
        std::max<std::int64_t>(edge_count, min_storage_edges);
    if (static_cast<std::uint64_t>(storage) >
        std::numeric_limits<std::uint32_t>::max()) {
      error->one(FLERR,
                 "Compact canonical graph exceeds the uint32 edge-index range");
    }
    const std::size_t required = static_cast<std::size_t>(storage);
    if (edge_capacity >= required) {
      return;
    }
    // A reserve of 2% covers the thermal fluctuations of the cutoff counts,
    // so that the storage is not reallocated from step to step, without
    // retaining several GiB of unused storage at billion-edge scale. The
    // reserve stops at the uint32 range that the model addresses edges in,
    // which ``required`` already lies within.
    const std::size_t slack = required / 50 + 64;
    edge_capacity = std::min<std::size_t>(
        required + slack, std::numeric_limits<std::uint32_t>::max());
    d_source = Kokkos::View<std::uint32_t*, DeviceType>(
        "compact_canonical:source", edge_capacity);
    d_edge_vec = Kokkos::View<float*, DeviceType>("compact_canonical:edge_vec",
                                                  edge_capacity * 3);
    // Fresh storage is zero, so no row of it needs clearing.
    Kokkos::deep_copy(d_fill_total, std::int64_t{0});
  }

  ExecutionSpace execution_space;
  Kokkos::View<std::int64_t, Kokkos::SharedHostPinnedSpace> h_edge_count;
  std::vector<int> host_type_map;
  bool extended_nodes;
  std::size_t edge_capacity;
  // Longest candidate list of the current neighbor list, or -1 before the
  // first build, and the candidates of all model centers.
  int longest_candidates = -1;
  std::int64_t candidate_total = 0;
  // State of the single-pass fill: one status word per block of centers, in
  // an array sized by the center count, the ticket counter, the tickets
  // issued by earlier builds, and the build epoch.
  Kokkos::View<unsigned long long*, DeviceType> d_fill_status;
  Kokkos::View<unsigned long long*, DeviceType> d_fill_ticket;
  // Fixed-point coordinates and candidate model node of every atom, one
  // 16-byte record per atom, which the single-pass fill gathers in place of
  // the two arrays.
  Kokkos::View<int* [4], Kokkos::LayoutRight, DeviceType> d_packed;
  // Rows of the edge storage the last build wrote, by either fill; storage
  // past it is zero.
  Kokkos::View<std::int64_t, DeviceType> d_fill_total;
  unsigned long long ticket_base = 0;
  unsigned fill_epoch = 0;

  DAT::tdual_int_1d k_loc2model;
  DAT::tdual_int_1d k_candidate_to_model;
  // Per-node counts are bounded by the LAMMPS neighbor limit. CSR offsets
  // remain int64 so the compact graph retains its full global edge range.
  Kokkos::View<std::uint32_t*, DeviceType> d_source_counts;
  Kokkos::View<std::uint32_t*, DeviceType> d_source_cursor;
};

}  // namespace LAMMPS_NS

#endif

#endif  // LMP_KOKKOS
