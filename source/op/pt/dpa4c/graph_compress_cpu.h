// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Shared declarations of the compressed DPA4C CPU kernels.
//
// The CPU path keeps the arithmetic of the CUDA path exactly, and changes the
// two things a cache hierarchy cares about that a warp scheduler does not:
//
//   * the radial table is re-laid out coefficient-major, so evaluating the
//     spline over a block of channels is a chain of contiguous vector fused
//     multiply-adds rather than a per-channel gather of six scalars;
//   * a node and all of its edges belong to one thread, so the destination
//     reduction accumulates in registers and the operator contains no atomic.
//
// The re-layout lives here rather than in the compression artifact because
// one artifact has to serve both devices, and the transposed copy is built
// once per model load.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace deepmd_dpa4c_cpu {

/// Allocator placing a buffer on a cache-line boundary.
///
/// The prepared tables are read exclusively through vector loads whose base
/// is a whole number of blocks from the buffer start, so aligning the start
/// keeps every one of them within a single cache line.
template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
  using value_type = T;

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  AlignedAllocator() = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}

  T* allocate(std::size_t count) {
    const std::size_t bytes =
        ((count * sizeof(T) + Alignment - 1) / Alignment) * Alignment;
#if defined(_MSC_VER)
    void* memory = _aligned_malloc(bytes, Alignment);
#else
    void* memory = std::aligned_alloc(Alignment, bytes);
#endif
    if (memory == nullptr) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(memory);
  }

  void deallocate(T* pointer, std::size_t) {
#if defined(_MSC_VER)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
  }

  template <typename U>
  bool operator==(const AlignedAllocator<U, Alignment>&) const {
    return true;
  }

  template <typename U>
  bool operator!=(const AlignedAllocator<U, Alignment>&) const {
    return false;
  }
};

using AlignedFloats = std::vector<float, AlignedAllocator<float>>;

/// Largest angular degree the kernels compile.
constexpr int kMaxDegree = 4;

/// Every runtime width the kernels derive from the operator inputs.
///
/// The CUDA kernels specialize the scalar width at compile time because it
/// fixes their thread mapping. The CPU kernels vectorize along the channel
/// axis with a runtime trip count, so only the angular degree and the
/// presence of radial modes change the compiled body.
struct Layout {
  int channels;                         ///< Scalar degree-zero width \f$C_0\f$.
  int modes;                            ///< Shared radial mode count \f$R\f$.
  int lmax;                             ///< Maximum angular degree.
  int degree_channels[kMaxDegree + 1];  ///< Channel width of each degree.
  int ranks[kMaxDegree];  ///< Probe rank of degrees one through `lmax`.
  int type_count;         ///< Type-table height \f$T+1\f$.
  int table_width;        ///< Tabulated width \f$C_0+R\f$.
  int spline_count;       ///< Number of spline intervals.

  int moment_width;     ///< Flat moment width \f$S\f$.
  int output_width;     ///< Invariant descriptor width.
  int gram_base;        ///< First Gram coordinate.
  int bispectrum_base;  ///< First bispectrum coordinate.
  int quartic_base;     ///< First projected-quartic coordinate.
  int divisor_base;     ///< Coordinate of the scalar divisor.
  int type_base;        ///< First centre-type coordinate.
  int closed_222_base;  ///< Coordinate of the symmetric 222 block.

  int block;            ///< Vector width in float lanes.
  int channel_blocks;   ///< Blocks covering the scalar channels.
  int padded_channels;  ///< `channel_blocks * block`.
  int spline_stride;    ///< Floats per prepared spline interval.

  /// Offset of degree `l` inside the flat moment vector.
  int degree_offset(int degree) const {
    int offset = 0;
    for (int lower = 0; lower < degree; ++lower) {
      offset += (2 * lower + 1) * degree_channels[lower];
    }
    return offset;
  }
};

/// Derive every width from the operator inputs.
///
/// \param channels Scalar degree-zero width.
/// \param modes Shared radial mode count.
/// \param lmax Maximum angular degree.
/// \param type_count Type-table height.
/// \param spline_count Number of spline intervals.
/// \param block Vector width in float lanes.
/// \return The complete layout.
Layout make_layout(int channels,
                   int modes,
                   int lmax,
                   int type_count,
                   int spline_count,
                   int block);

/// Radial table, ordered FiLM and mode caches in the layout the kernels read.
///
/// The spline interval is stored as `channel_blocks` groups of six
/// coefficient vectors followed by the mode coefficients, so one interval is
/// a single contiguous stream and one channel block is six aligned vector
/// loads. The FiLM scale and shift planes are separated and padded to the
/// block width for the same reason.
struct PreparedTables {
  AlignedFloats spline;  ///< `(spline_count, spline_stride)`.
  AlignedFloats film;    ///< `(type_count^2, 2, padded_channels)`.
  AlignedFloats mixing;  ///< `(type_count^2, modes, padded_channels)`.
};

/// Build the prepared tables from the compression artifacts.
///
/// \param table Spline coefficients with shape `(spline_count, 6 * width)`,
///              quartet block followed by pair block.
/// \param pair_film Ordered scale and shift with shape `(P, C_0, 2)`.
/// \param pair_mixing Ordered mode mixing with shape `(P, C_0, R)`, or null.
/// \param layout Derived widths.
/// \return Tables in the kernel layout.
PreparedTables prepare_tables(const float* table,
                              const float* pair_film,
                              const float* pair_mixing,
                              const Layout& layout);

/// Immutable inputs and outputs of one descriptor evaluation.
///
/// The graph form addresses an edge through `destination_order` and honours
/// `edge_mask`; the canonical form addresses it directly and carries neither.
struct Arguments {
  const float* edge_vec;             ///< `(E, 3)` in the model precision.
  const int64_t* source;             ///< `(E,)` source node of each edge.
  const int64_t* destination_order;  ///< `(E,)` or null for canonical.
  const bool* edge_mask;             ///< `(E,)` or null for canonical.
  const int64_t* row_ptr;            ///< `(N + 1,)` destination CSR.
  const int64_t* atype;              ///< `(N,)` node types.

  const PreparedTables* tables;   ///< Prepared radial and ordered caches.
  const float* type_embedding;    ///< `(T + 1, C_0)` centre type table.
  const float* readout;           ///< `(8, C_1, C_1)` packed projections.
  const int32_t* coupling_meta;   ///< `(M, 8)` sparse coupling records.
  const int32_t* coupling_entry;  ///< Packed components and coordinates.
  const float* coupling_value;    ///< Gaunt values and probe scales.
  const float* output_mean;       ///< `(D,)` calibration shift.
  const float* output_inv_std;    ///< `(D,)` calibration scale.

  float* descriptor;  ///< `(N, D)` output.
  float* state;       ///< `(N, S + 2)` saved state.

  const float* descriptor_gradient;  ///< `(N, D)` cotangent, backward only.
  float* edge_gradient;              ///< `(E, 3)` output, backward only.

  int64_t node_count;
  int64_t edge_count;
  int coupling_count;

  float table_stride;
  float table_max;
  float rcut;
  float eps;
  float degree_floor;
};

/// Evaluate the descriptor over one contiguous node range.
using ScanFunction = void (*)(const Arguments&,
                              const Layout&,
                              int64_t,
                              int64_t);

/// Entry points of one compiled instruction-set level.
struct Kernels {
  ScanFunction forward;
  ScanFunction backward;
};

namespace scalar {
Kernels kernels(int lmax, bool has_modes);
}  // namespace scalar

namespace avx2 {
Kernels kernels(int lmax, bool has_modes);
}  // namespace avx2

namespace avx512 {
Kernels kernels(int lmax, bool has_modes);
}  // namespace avx512

}  // namespace deepmd_dpa4c_cpu
