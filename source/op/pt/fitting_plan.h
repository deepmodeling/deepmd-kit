// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Layer geometry of one energy fitting network.
//
// The layout of the saved pre-activations is part of the operator contract
// rather than of either device's kernel, so it lives in a header both can
// include: a CUDA translation unit cannot be reached from the CPU half, and
// the reverse would drag the CUDA runtime into a CPU-only build.

#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <vector>

/// Prefix sums of the hidden widths, which address the saved buffer.
struct FittingLayerPlan {
  std::vector<long> offset;  //!< Prefix sum of the hidden widths.
  long width_max;            //!< Widest hidden layer.
  int n_layer;

  /// Floats of saved state per node.
  long saved_width() const { return offset[n_layer]; }
};

/// Derive the layer geometry from the weight list.
inline FittingLayerPlan fitting_layer_plan(
    const std::vector<torch::Tensor>& ws) {
  FittingLayerPlan plan{std::vector<long>(ws.size() + 1, 0), 0,
                        static_cast<int>(ws.size())};
  for (size_t layer = 0; layer < ws.size(); ++layer) {
    plan.offset[layer + 1] = plan.offset[layer] + ws[layer].size(1);
    plan.width_max =
        std::max(plan.width_max, static_cast<long>(ws[layer].size(1)));
  }
  return plan;
}
