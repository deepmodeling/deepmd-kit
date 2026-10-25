// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "commonPT.h"

#ifdef DEEPMD_TORCH_HAS_CUDA
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

namespace deepmd {
namespace ptexpt {

/**
 * @brief Build the all-pairs topology a long-range module of the graph lower
 * reads.
 *
 * Inference-path twin of Python ``build_pair_graph`` in
 * ``deepmd/dpmodel/utils/neighbor_graph/pair_graph.py`` for a single frame:
 * every unordered pair of nodes once, ``[src, dst]`` with ``src < dst``, its
 * vector ``r_src - r_dst``, and two masked guard pairs that keep the pair axis
 * at the lower bound of the exported lower. The model-level pair exclusion is
 * applied to the mask, as it is to the neighbor graph.
 *
 * @param coord Node positions, shape ``(N, 3)``, float64, on the model device.
 * @param atype Node types, shape ``(N,)``, int64.
 * @param type_mask_table Device keep table of the pair exclusion, or an
 * undefined tensor when the model excludes no pair.
 * @param ntypes Number of atom types.
 * @param vec_dtype Precision of the pair vectors, that of the edge vectors of
 * the artifact.
 * @return ``{pair_index (2, P + 2) int64, pair_vec (P + 2, 3), pair_mask
 * (P + 2,) bool}``.
 */
inline std::vector<torch::Tensor> buildAllPairsTopology(
    const torch::Tensor& coord,
    const torch::Tensor& atype,
    const torch::Tensor& type_mask_table,
    const int ntypes,
    const torch::ScalarType vec_dtype) {
  const std::int64_t node_count = coord.size(0);
  const auto index_options =
      torch::TensorOptions().dtype(torch::kInt64).device(coord.device());
  const auto mask_options =
      torch::TensorOptions().dtype(torch::kBool).device(coord.device());
  const torch::Tensor index =
      torch::triu_indices(node_count, node_count, 1, index_options);
  const torch::Tensor vec = (coord.index_select(0, index.select(0, 0)) -
                             coord.index_select(0, index.select(0, 1)))
                                .to(vec_dtype);
  const torch::Tensor mask =
      applyPairExclusion(index, torch::ones({index.size(1)}, mask_options),
                         atype, type_mask_table, ntypes);
  return {torch::cat({index, torch::zeros({2, 2}, index_options)}, 1),
          torch::cat({vec, torch::zeros({2, 3}, vec.options())}, 0),
          torch::cat({mask, torch::zeros({2}, mask_options)})};
}

/**
 * @brief Build the all-pairs topology of a periodic frame, which holds no
 * pair: the reciprocal-space topology describes the interactions.
 *
 * Counterpart of Python ``empty_pair_graph`` for the exported lower: two
 * masked guard pairs, the lower bound of its pair axis.
 */
inline std::vector<torch::Tensor> buildGuardPairTopology(
    const torch::Device& device, const torch::ScalarType vec_dtype) {
  const auto options = torch::TensorOptions().device(device);
  return {torch::zeros({2, 2}, options.dtype(torch::kInt64)),
          torch::zeros({2, 3}, options.dtype(vec_dtype)),
          torch::zeros({2}, options.dtype(torch::kBool))};
}

/**
 * @brief The smallest even integer of at least ``n`` without a prime factor
 * above 7.
 */
inline std::int64_t evenFftSize(std::int64_t n) {
  std::int64_t m = std::max<std::int64_t>(2, n + n % 2);
  while (true) {
    std::int64_t rest = m;
    for (const std::int64_t p : {2, 3, 5, 7}) {
      while (rest % p == 0) {
        rest /= p;
      }
    }
    if (rest == 1) {
      return m;
    }
    m += 2;
  }
}

/**
 * @brief The size of the mesh of a periodic frame along its lattice vectors.
 *
 * Twin of Python ``mesh_dims_of_cells`` and of the long-range operator for a
 * single frame: along every lattice vector ``a_d`` the mesh holds the
 * smallest even number of points without a prime factor above 7 of at least
 * ``ceil(|a_d| / spacing)``. The squared length sums the squares of the
 * components, each rounded on its own as Python forms it; the explicit fma of
 * a square with zero is that rounded square, which the compiler does not
 * fuse into the sum, so that the three size the mesh identically whatever
 * the floating-point contraction of the build.
 *
 * @param box The row-major cell, the rows its lattice vectors, nine values in
 * the length unit of the model.
 * @param spacing The largest mesh spacing, in the length unit of the model.
 * @throw deepmd::deepmd_exception If the mesh exceeds 2^20 points along a
 * lattice vector, the limit of the operator.
 */
inline std::array<std::int64_t, 3> meshDims(const std::vector<double>& box,
                                            const double spacing) {
  std::array<std::int64_t, 3> dims = {};
  for (int d = 0; d < 3; ++d) {
    const double* v = box.data() + 3 * d;
    const double length =
        std::sqrt(std::fma(v[0], v[0], 0.0) + std::fma(v[1], v[1], 0.0) +
                  std::fma(v[2], v[2], 0.0));
    const double points = std::ceil(length / spacing);
    if (!(points < double(1 << 20))) {
      throw deepmd::deepmd_exception(
          "the mesh of the cell exceeds 2^20 points along a lattice vector; "
          "a lattice vector of " +
          std::to_string(length) + " at a mesh spacing of " +
          std::to_string(spacing) + " needs " + std::to_string(points) + ".");
    }
    dims[d] = evenFftSize(static_cast<std::int64_t>(points));
  }
  return dims;
}

/**
 * @brief Build the mesh topology a periodic long-range module of the graph
 * lower reads.
 *
 * Twin of Python ``build_mesh_grid`` in
 * ``deepmd/dpmodel/utils/neighbor_graph/reciprocal_graph.py`` for a single
 * frame, on the mesh of ``meshDims``. Without a cell, the frame is
 * non-periodic: the cell is the identity, the mesh minimal and the frame
 * masked.
 *
 * @param coord Node positions, shape ``(N, 3)``, on the model device.
 * @param box The row-major cell, the rows its lattice vectors, nine values in
 * the length unit of the model, or empty for a non-periodic frame.
 * @param spacing The largest mesh spacing, in the length unit of the model.
 * @param dtype Precision of the geometry, that of the edge vectors.
 * @return ``{box (1, 3, 3), volume (1,), coord (N, 3), strain (1, 3, 3),
 * mask (1,) bool, mode_1 (n1,), mode_2 (n2,), mode_3 (n3,)}``.
 */
inline std::vector<torch::Tensor> buildMeshTopology(
    const torch::Tensor& coord,
    const std::vector<double>& box,
    const double spacing,
    const torch::ScalarType dtype) {
  const auto options =
      torch::TensorOptions().dtype(dtype).device(coord.device());
  const bool periodic = !box.empty();
  std::array<double, 9> cell = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::array<std::int64_t, 3> dims = {2, 2, 2};
  if (periodic) {
    for (int k = 0; k < 9; ++k) {
      cell[k] = box[k];
    }
    dims = meshDims(box, spacing);
  }
  const double volume =
      std::abs(cell[0] * (cell[4] * cell[8] - cell[5] * cell[7]) -
               cell[1] * (cell[3] * cell[8] - cell[5] * cell[6]) +
               cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]));
  // The copy detaches the tensor from the local array, which a conversion
  // to float64 on the host would alias.
  const auto cell_tensor =
      torch::from_blob(cell.data(), {1, 3, 3}, torch::kFloat64)
          .to(options, /*non_blocking=*/false, /*copy=*/true);
  std::vector<torch::Tensor> fields = {
      cell_tensor, torch::full({1}, volume, options), coord.to(dtype),
      torch::zeros({1, 3, 3}, options),
      torch::full({1}, periodic, options.dtype(torch::kBool))};
  for (const std::int64_t n : dims) {
    // The integer frequencies of an axis of n points, in FFT order.
    const auto m = torch::arange(n, options);
    fields.push_back(torch::where(m > n / 2, m - n, m));
  }
  return fields;
}

/**
 * @brief The cell input of the device-resident evaluations of a model with a
 * long-range module, in pinned host memory.
 *
 * The compact lower reads the cell on the device while the stream runs its
 * evaluation, so that a CUDA graph capturing the evaluation reads the cell
 * of every replay from the same memory. One event orders those reads: an
 * evaluation on another stream than the previous one first waits on the
 * device for the previous one, so that the event of the last evaluation
 * follows every earlier read, and a cell other than the one held waits for
 * that event before the memory is written.
 */
class PinnedCell {
 public:
  PinnedCell() = default;
  PinnedCell(const PinnedCell&) = delete;
  PinnedCell& operator=(const PinnedCell&) = delete;
  // The destructor reports no error.
  ~PinnedCell() {
#ifdef DEEPMD_TORCH_HAS_CUDA
    if (read_ != nullptr) {
      int current = 0;
      cudaGetDevice(&current);
      cudaSetDevice(device_);
      cudaEventDestroy(read_);
      cudaSetDevice(current);
    }
#endif
  }

  /**
   * @brief Hold ``box`` for the next evaluation, which is enqueued on the
   * current stream of ``device``.
   *
   * @param box The row-major cell, nine values, or empty for a non-periodic
   * frame, whose cell is zero.
   * @param device Index of the CUDA device of the evaluation.
   * @return The cell input, a pinned float64 CPU tensor of shape ``(1, 9)``.
   */
  const torch::Tensor& hold(const std::vector<double>& box, const int device) {
    if (!cell_.defined()) {
      cell_ = torch::zeros(
          {1, 9},
          torch::TensorOptions().dtype(torch::kFloat64).pinned_memory(true));
    }
#ifdef DEEPMD_TORCH_HAS_CUDA
    device_ = static_cast<c10::DeviceIndex>(device);
    stream_ = c10::cuda::getCurrentCUDAStream(device_).stream();
    if (read_ != nullptr && stream_ != read_stream_) {
      C10_CUDA_CHECK(cudaStreamWaitEvent(stream_, read_, 0));
    }
#else
    (void)device;
#endif
    std::array<double, 9> values = {};
    std::copy(box.begin(), box.end(), values.begin());
    double* const cell = cell_.data_ptr<double>();
    if (std::memcmp(cell, values.data(), sizeof(values)) != 0) {
#ifdef DEEPMD_TORCH_HAS_CUDA
      if (read_ != nullptr) {
        C10_CUDA_CHECK(cudaEventSynchronize(read_));
      }
#endif
      std::memcpy(cell, values.data(), sizeof(values));
    }
    return cell_;
  }

  /**
   * @brief Mark the evaluation of the cell last held, once enqueued, as the
   * last that reads the cell.
   */
  void record() {
#ifdef DEEPMD_TORCH_HAS_CUDA
    const c10::cuda::CUDAGuard device_guard(device_);
    if (read_ == nullptr) {
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&read_, cudaEventDisableTiming));
    }
    C10_CUDA_CHECK(cudaEventRecord(read_, stream_));
    read_stream_ = stream_;
#endif
  }

 private:
  torch::Tensor cell_;
#ifdef DEEPMD_TORCH_HAS_CUDA
  c10::DeviceIndex device_ = 0;
  // The stream of the evaluation of the cell last held.
  cudaStream_t stream_ = nullptr;
  // The last evaluation that reads the cell, and its stream.
  cudaEvent_t read_ = nullptr;
  cudaStream_t read_stream_ = nullptr;
#endif
};

}  // namespace ptexpt
}  // namespace deepmd
