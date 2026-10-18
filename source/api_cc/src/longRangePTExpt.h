// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <torch/torch.h>

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

}  // namespace ptexpt
}  // namespace deepmd
