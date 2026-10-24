// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <torch/torch.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "commonPT.h"

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
 * @brief Build the mesh topology a periodic long-range module of the graph
 * lower reads.
 *
 * Twin of Python ``build_mesh_grid`` in
 * ``deepmd/dpmodel/utils/neighbor_graph/reciprocal_graph.py`` for a single
 * frame: along every lattice vector ``a_d`` the mesh holds the smallest even
 * number of points without a prime factor above 7 of at least
 * ``ceil(|a_d| / spacing)``, with the length formed term by term as Python
 * forms it, so that both size the mesh identically. Without a cell, the frame
 * is non-periodic: the cell is the identity, the mesh minimal and the frame
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
    for (int d = 0; d < 3; ++d) {
      const double* v = cell.data() + 3 * d;
      const double length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      dims[d] =
          evenFftSize(static_cast<std::int64_t>(std::ceil(length / spacing)));
    }
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

}  // namespace ptexpt
}  // namespace deepmd
