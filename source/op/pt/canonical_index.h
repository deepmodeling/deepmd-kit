// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <torch/types.h>

#include <cstdint>

namespace deepmd {

/**
 * Restore integer words from an exported canonical index buffer without a copy.
 *
 * AOTInductor fixes input dtypes, so the deployment boundary carries integer
 * words as bytes with a final axis of length four or eight. Its symbolic extent
 * selects the integer width at runtime; the descriptor kernels receive ordinary
 * typed tensors. Eager calls may supply those typed tensors directly.
 */
inline torch::Tensor canonical_index_view(const torch::Tensor& buffer) {
  if (buffer.scalar_type() != torch::kUInt8) {
    return buffer;
  }
  TORCH_CHECK(
      buffer.dim() >= 2 && buffer.is_contiguous(),
      "canonical index bytes must be contiguous with a final word axis");
  const auto bytes = buffer.size(-1);
  TORCH_CHECK(bytes == 4 || bytes == 8,
              "canonical index word width must be 4 or 8 bytes, got ", bytes);
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(buffer.data_ptr()) % bytes == 0,
              "canonical index bytes must be aligned to their word width");
  const auto dtype = bytes == 8 ? torch::kInt64 : torch::kUInt32;
  return buffer.view(dtype).squeeze(-1);
}

}  // namespace deepmd
