// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Torch bindings of the compressed DPA4C descriptor on the CPU.
//
// This translation unit owns the width derivation, the one-time table
// re-layout, the instruction-set selection, the thread partition, and the
// operator registration. The arithmetic lives in the per-level kernels.

#include "graph_compress_cpu.h"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#include "../cpu/dispatch.h"
#include "../cpu/partition.h"

namespace deepmd_dpa4c_cpu {

Layout make_layout(int channels,
                   int modes,
                   int lmax,
                   int type_count,
                   int spline_count,
                   int block) {
  Layout layout{};
  layout.channels = channels;
  layout.modes = modes;
  layout.lmax = lmax;
  layout.type_count = type_count;
  layout.table_width = channels + modes;
  layout.spline_count = spline_count;

  // `channels` is a power of two, so the geometric mean of the scalar width
  // and the floor is an exact shift. Mirrors `derive_degree_channels`.
  int exponent = 0;
  while ((1 << (exponent + 1)) <= channels) {
    ++exponent;
  }
  const int degree_one = std::max(4, 1 << ((exponent + 1) / 2));
  layout.degree_channels[0] = channels;
  layout.degree_channels[1] = degree_one;
  layout.degree_channels[2] = std::max(4, degree_one >> 1);
  for (int degree = 3; degree <= lmax; ++degree) {
    layout.degree_channels[degree] = 1;
  }
  layout.ranks[0] = layout.degree_channels[2];
  layout.ranks[1] = 2;
  for (int degree = 3; degree <= lmax; ++degree) {
    layout.ranks[degree - 1] = 1;
  }

  layout.moment_width = 0;
  for (int degree = 0; degree <= lmax; ++degree) {
    layout.moment_width += (2 * degree + 1) * layout.degree_channels[degree];
  }

  int gram_total = 0;
  for (int degree = 1; degree <= lmax; ++degree) {
    const int width = layout.degree_channels[degree];
    gram_total += width * (width + 1) / 2;
  }
  layout.gram_base = channels;
  layout.bispectrum_base = layout.gram_base + gram_total;

  // Enumerate the O(3)-even degree triples in the order the layout builder
  // uses, so the closed-form 222 block lands on the coordinate the artifact
  // reserved for it.
  int offset = 0;
  int closed_222 = 0;
  int bispectrum_total = 0;
  for (int first = 1; first <= lmax; ++first) {
    for (int second = first; second <= lmax; ++second) {
      for (int third = second; third <= lmax; ++third) {
        if (third > first + second || (first + second + third) % 2 != 0) {
          continue;
        }
        const int rank_one = layout.ranks[first - 1];
        const int rank_two = layout.ranks[second - 1];
        const int rank_three = layout.ranks[third - 1];
        int count = 0;
        if (first == third) {
          count = rank_one * (rank_one + 1) * (rank_one + 2) / 6;
        } else if (first == second) {
          count = rank_one * (rank_one + 1) / 2 * rank_three;
        } else if (second == third) {
          count = rank_one * (rank_two * (rank_two + 1) / 2);
        } else {
          count = rank_one * rank_two * rank_three;
        }
        if (first == 2 && second == 2 && third == 2) {
          closed_222 = layout.bispectrum_base + offset;
        }
        offset += count;
        bispectrum_total += count;
      }
    }
  }
  layout.closed_222_base = closed_222;
  layout.quartic_base = layout.bispectrum_base + bispectrum_total;
  layout.divisor_base = layout.quartic_base + layout.ranks[0] * layout.ranks[1];
  layout.type_base = layout.divisor_base + 2;
  layout.output_width = layout.type_base + channels;

  layout.block = block;
  layout.channel_blocks = (channels + block - 1) / block;
  layout.padded_channels = layout.channel_blocks * block;
  // The mode coefficients follow the channel blocks inside one interval. The
  // stride is rounded to a cache line so that every interval, not only the
  // first, starts aligned.
  constexpr int kLine = 16;
  layout.spline_stride =
      (layout.channel_blocks * 6 * block + 6 * modes + kLine - 1) / kLine *
      kLine;
  return layout;
}

PreparedTables prepare_tables(const float* table,
                              const float* pair_film,
                              const float* pair_mixing,
                              const Layout& layout) {
  PreparedTables prepared;
  const int width = layout.table_width;
  const int block = layout.block;
  const int padded = layout.padded_channels;
  const int64_t pairs =
      static_cast<int64_t>(layout.type_count) * layout.type_count;

  prepared.spline.assign(
      static_cast<size_t>(layout.spline_count) * layout.spline_stride, 0.0f);
  for (int64_t interval = 0; interval < layout.spline_count; ++interval) {
    const float* source = table + interval * 6 * width;
    float* target = prepared.spline.data() + interval * layout.spline_stride;
    for (int channel = 0; channel < layout.channels; ++channel) {
      const int group = channel / block;
      const int lane = channel % block;
      float* out = target + group * 6 * block + lane;
      for (int order = 0; order < 4; ++order) {
        out[order * block] = source[4 * channel + order];
      }
      out[4 * block] = source[4 * width + 2 * channel];
      out[5 * block] = source[4 * width + 2 * channel + 1];
    }
    float* modes = target + layout.channel_blocks * 6 * block;
    for (int mode = 0; mode < layout.modes; ++mode) {
      const int channel = layout.channels + mode;
      for (int order = 0; order < 4; ++order) {
        modes[6 * mode + order] = source[4 * channel + order];
      }
      modes[6 * mode + 4] = source[4 * width + 2 * channel];
      modes[6 * mode + 5] = source[4 * width + 2 * channel + 1];
    }
  }

  prepared.film.assign(static_cast<size_t>(pairs) * 2 * padded, 0.0f);
  for (int64_t pair = 0; pair < pairs; ++pair) {
    const float* source = pair_film + pair * layout.channels * 2;
    float* scale = prepared.film.data() + pair * 2 * padded;
    float* shift = scale + padded;
    for (int channel = 0; channel < layout.channels; ++channel) {
      scale[channel] = source[2 * channel];
      shift[channel] = source[2 * channel + 1];
    }
  }

  if (layout.modes > 0) {
    prepared.mixing.assign(static_cast<size_t>(pairs) * layout.modes * padded,
                           0.0f);
    for (int64_t pair = 0; pair < pairs; ++pair) {
      const float* source = pair_mixing + pair * layout.channels * layout.modes;
      float* target = prepared.mixing.data() + pair * layout.modes * padded;
      for (int channel = 0; channel < layout.channels; ++channel) {
        for (int mode = 0; mode < layout.modes; ++mode) {
          target[mode * padded + channel] =
              source[channel * layout.modes + mode];
        }
      }
    }
  }
  return prepared;
}

namespace {

using deepmd_cpu::Isa;

/// Vector width the selected instruction set operates on.
int isa_block() {
  switch (deepmd_cpu::host_isa()) {
    case Isa::kAvx512:
      return 16;
    case Isa::kAvx2:
      return 8;
    default:
      return 4;
  }
}

/// Resolve the kernels of the running CPU.
Kernels resolve_kernels(int lmax, bool has_modes) {
  switch (deepmd_cpu::host_isa()) {
    case Isa::kAvx512:
      return avx512::kernels(lmax, has_modes);
    case Isa::kAvx2:
      return avx2::kernels(lmax, has_modes);
    default:
      return scalar::kernels(lmax, has_modes);
  }
}

/// Process-local cache of the re-laid-out tables.
///
/// The artifacts are immutable buffers of a compressed snapshot, so one
/// entry serves every step of a molecular-dynamics run. The entry holds a
/// strong reference to the source storage, which both keeps the identifying
/// pointer from being recycled under a new tensor and makes the lifetime
/// explicit.
class TableCache {
 public:
  const PreparedTables& get(const torch::Tensor& table,
                            const torch::Tensor& pair_film,
                            const torch::Tensor& pair_mixing,
                            const Layout& layout) {
    const void* key = table.const_data_ptr();
    std::lock_guard<std::mutex> guard(mutex_);
    for (const Entry& entry : entries_) {
      if (entry.key == key && entry.block == layout.block) {
        return *entry.tables;
      }
    }
    Entry entry;
    entry.key = key;
    entry.block = layout.block;
    entry.retained = {table, pair_film, pair_mixing};
    entry.tables = std::make_shared<PreparedTables>(prepare_tables(
        table.const_data_ptr<float>(), pair_film.const_data_ptr<float>(),
        layout.modes > 0 ? pair_mixing.const_data_ptr<float>() : nullptr,
        layout));
    // A process serves one model at a time in production and a handful in a
    // test session; the bound keeps a long-lived session from retaining every
    // table it has ever seen.
    if (entries_.size() >= kCapacity) {
      entries_.erase(entries_.begin());
    }
    entries_.push_back(std::move(entry));
    return *entries_.back().tables;
  }

 private:
  static constexpr size_t kCapacity = 8;

  struct Entry {
    const void* key;
    int block;
    std::vector<torch::Tensor> retained;
    std::shared_ptr<PreparedTables> tables;
  };

  std::mutex mutex_;
  std::vector<Entry> entries_;
};

TableCache& table_cache() {
  static TableCache cache;
  return cache;
}

/// Bundle the operator inputs shared by the forward and the backward.
struct Inputs {
  torch::Tensor edge_vec;
  torch::Tensor source;
  torch::Tensor destination_order;
  torch::Tensor edge_mask;
  torch::Tensor row_ptr;
  torch::Tensor atype;
  Layout layout;
  const PreparedTables* tables;
};

/// Validate and normalize the graph-form inputs.
Inputs build_inputs(const torch::Tensor& edge_vec,
                    const torch::Tensor& edge_index,
                    const torch::Tensor& edge_mask,
                    const torch::Tensor& destination_order,
                    const torch::Tensor& destination_row_ptr,
                    const torch::Tensor& atype,
                    const torch::Tensor& table,
                    const torch::Tensor& pair_film,
                    const torch::Tensor& pair_mixing,
                    const torch::Tensor& type_embedding,
                    bool canonical,
                    int64_t lmax) {
  TORCH_CHECK(edge_vec.device().is_cpu(),
              "dpa4c_graph_compress: the CPU kernel needs CPU tensors");
  TORCH_CHECK(edge_vec.dim() == 2 && edge_vec.size(1) == 3,
              "dpa4c_graph_compress: edge_vec must have shape (E, 3)");
  const int channels = static_cast<int>(type_embedding.size(1));
  const int modes =
      pair_mixing.numel() == 0 ? 0 : static_cast<int>(pair_mixing.size(2));
  const int type_count = static_cast<int>(type_embedding.size(0));
  const int spline_count = static_cast<int>(table.size(0));
  Inputs inputs;
  inputs.layout = make_layout(channels, modes, static_cast<int>(lmax),
                              type_count, spline_count, isa_block());
  TORCH_CHECK(table.size(1) == 6 * inputs.layout.table_width,
              "dpa4c_graph_compress: the radial table width does not match "
              "the channel and mode counts");
  inputs.edge_vec = edge_vec.to(torch::kFloat32).contiguous();
  inputs.source = edge_index.select(0, 0).to(torch::kLong).contiguous();
  inputs.atype = atype.to(torch::kLong).contiguous();
  inputs.row_ptr = destination_row_ptr.to(torch::kLong).contiguous();
  if (!canonical) {
    inputs.destination_order = destination_order.to(torch::kLong).contiguous();
    inputs.edge_mask = edge_mask.to(torch::kBool).contiguous();
  }
  inputs.tables = &table_cache().get(table.contiguous(), pair_film.contiguous(),
                                     pair_mixing.contiguous(), inputs.layout);
  return inputs;
}

/// Fill the device-neutral argument block.
Arguments build_arguments(const Inputs& inputs,
                          const torch::Tensor& type_embedding,
                          const torch::Tensor& readout_matrices,
                          const torch::Tensor& coupling_meta,
                          const torch::Tensor& coupling_entry,
                          const torch::Tensor& coupling_value,
                          const torch::Tensor& output_mean,
                          const torch::Tensor& output_inv_std,
                          double table_stride,
                          double table_max,
                          double rcut,
                          double eps,
                          double degree_floor) {
  Arguments arguments{};
  arguments.edge_vec = inputs.edge_vec.const_data_ptr<float>();
  arguments.source = inputs.source.const_data_ptr<int64_t>();
  arguments.destination_order =
      inputs.destination_order.defined()
          ? inputs.destination_order.const_data_ptr<int64_t>()
          : nullptr;
  arguments.edge_mask = inputs.edge_mask.defined()
                            ? inputs.edge_mask.const_data_ptr<bool>()
                            : nullptr;
  arguments.row_ptr = inputs.row_ptr.const_data_ptr<int64_t>();
  arguments.atype = inputs.atype.const_data_ptr<int64_t>();
  arguments.tables = inputs.tables;
  arguments.type_embedding = type_embedding.const_data_ptr<float>();
  arguments.readout = readout_matrices.const_data_ptr<float>();
  arguments.coupling_meta = coupling_meta.numel() == 0
                                ? nullptr
                                : coupling_meta.const_data_ptr<int32_t>();
  arguments.coupling_entry = coupling_entry.numel() == 0
                                 ? nullptr
                                 : coupling_entry.const_data_ptr<int32_t>();
  arguments.coupling_value = coupling_value.numel() == 0
                                 ? nullptr
                                 : coupling_value.const_data_ptr<float>();
  arguments.output_mean = output_mean.const_data_ptr<float>();
  arguments.output_inv_std = output_inv_std.const_data_ptr<float>();
  arguments.node_count = inputs.atype.size(0);
  arguments.edge_count = inputs.edge_vec.size(0);
  arguments.coupling_count = static_cast<int>(coupling_meta.numel() / 8);
  arguments.table_stride = static_cast<float>(table_stride);
  arguments.table_max = static_cast<float>(table_max);
  arguments.rcut = static_cast<float>(rcut);
  arguments.eps = static_cast<float>(eps);
  arguments.degree_floor = static_cast<float>(degree_floor);
  return arguments;
}

/// Run one scan over the whole node axis with a balanced edge partition.
void run_scan(ScanFunction scan,
              const Arguments& arguments,
              const Layout& layout) {
  const int threads = std::max(1, at::get_num_threads());
  const std::vector<deepmd_cpu::NodeRange> ranges = deepmd_cpu::balanced_ranges(
      arguments.row_ptr, arguments.node_count, threads);
  at::parallel_for(0, static_cast<int64_t>(ranges.size()), 1,
                   [&](int64_t begin, int64_t end) {
                     for (int64_t part = begin; part < end; ++part) {
                       scan(arguments, layout, ranges[part].begin,
                            ranges[part].end);
                     }
                   });
}

std::tuple<torch::Tensor, torch::Tensor> forward(
    torch::Tensor edge_vec,
    torch::Tensor edge_index,
    torch::Tensor edge_mask,
    torch::Tensor destination_order,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor table,
    torch::Tensor pair_film,
    torch::Tensor pair_mixing,
    torch::Tensor type_embedding,
    torch::Tensor readout_matrices,
    torch::Tensor coupling_meta,
    torch::Tensor coupling_entry,
    torch::Tensor coupling_value,
    torch::Tensor output_mean,
    torch::Tensor output_inv_std,
    torch::Tensor spin,
    torch::Tensor spin_pair,
    torch::Tensor spin_type,
    bool canonical,
    int64_t lmax,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor) {
  TORCH_CHECK(spin.dim() != 2,
              "dpa4c_graph_compress: the CPU kernel has no native spin "
              "branch; evaluate a spin-conditioned descriptor eagerly");
  const Inputs inputs = build_inputs(
      edge_vec, edge_index, edge_mask, destination_order, destination_row_ptr,
      atype, table, pair_film, pair_mixing, type_embedding, canonical, lmax);
  const Layout& layout = inputs.layout;
  auto options = edge_vec.options().dtype(torch::kFloat32);
  auto descriptor =
      torch::empty({inputs.atype.size(0), layout.output_width}, options);
  auto state =
      torch::empty({inputs.atype.size(0), layout.moment_width + 2}, options);
  Arguments arguments =
      build_arguments(inputs, type_embedding.contiguous(),
                      readout_matrices.contiguous(), coupling_meta.contiguous(),
                      coupling_entry.contiguous(), coupling_value.contiguous(),
                      output_mean.contiguous(), output_inv_std.contiguous(),
                      table_stride, table_max, rcut, eps, degree_floor);
  arguments.descriptor = descriptor.data_ptr<float>();
  arguments.state = state.data_ptr<float>();
  run_scan(resolve_kernels(static_cast<int>(lmax), layout.modes > 0).forward,
           arguments, layout);
  return {descriptor, state};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(
    torch::Tensor descriptor_gradient,
    torch::Tensor state,
    torch::Tensor edge_vec,
    torch::Tensor edge_index,
    torch::Tensor edge_mask,
    torch::Tensor destination_order,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor table,
    torch::Tensor pair_film,
    torch::Tensor pair_mixing,
    torch::Tensor type_embedding,
    torch::Tensor readout_matrices,
    torch::Tensor coupling_meta,
    torch::Tensor coupling_entry,
    torch::Tensor coupling_value,
    torch::Tensor output_mean,
    torch::Tensor output_inv_std,
    torch::Tensor spin,
    torch::Tensor spin_pair,
    torch::Tensor spin_type,
    bool canonical,
    int64_t lmax,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor) {
  TORCH_CHECK(spin.dim() != 2,
              "dpa4c_graph_compress_backward: the CPU kernel has no native "
              "spin branch");
  const Inputs inputs = build_inputs(
      edge_vec, edge_index, edge_mask, destination_order, destination_row_ptr,
      atype, table, pair_film, pair_mixing, type_embedding, canonical, lmax);
  const Layout& layout = inputs.layout;
  auto options = edge_vec.options().dtype(torch::kFloat32);
  auto edge_gradient = torch::empty({inputs.edge_vec.size(0), 3}, options);
  auto absent = torch::empty({0}, options);
  Arguments arguments =
      build_arguments(inputs, type_embedding.contiguous(),
                      readout_matrices.contiguous(), coupling_meta.contiguous(),
                      coupling_entry.contiguous(), coupling_value.contiguous(),
                      output_mean.contiguous(), output_inv_std.contiguous(),
                      table_stride, table_max, rcut, eps, degree_floor);
  auto contiguous_gradient =
      descriptor_gradient.to(torch::kFloat32).contiguous();
  auto contiguous_state = state.to(torch::kFloat32).contiguous();
  arguments.descriptor_gradient = contiguous_gradient.const_data_ptr<float>();
  arguments.state =
      const_cast<float*>(contiguous_state.const_data_ptr<float>());
  arguments.edge_gradient = edge_gradient.data_ptr<float>();

  // A masked edge sorts past the last destination row, so no row reaches it
  // and the scan never writes its slot. Both topology forms keep those slots
  // in the suffix of the destination permutation, which is the identity for
  // a canonical payload, so one pass over that suffix clears exactly the
  // uncovered set.
  const int64_t covered = inputs.row_ptr[inputs.atype.size(0)].item<int64_t>();
  const int64_t stored = edge_gradient.size(0);
  if (covered < stored) {
    float* gradient = edge_gradient.data_ptr<float>();
    if (arguments.destination_order == nullptr) {
      std::memset(gradient + 3 * covered, 0,
                  sizeof(float) * 3 * (stored - covered));
    } else {
      const int64_t* order = arguments.destination_order;
      at::parallel_for(covered, stored, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t entry = begin; entry < end; ++entry) {
          float* slot = gradient + 3 * order[entry];
          slot[0] = 0.0f;
          slot[1] = 0.0f;
          slot[2] = 0.0f;
        }
      });
    }
  }
  run_scan(resolve_kernels(static_cast<int>(lmax), layout.modes > 0).backward,
           arguments, layout);
  return {edge_gradient.to(edge_vec.scalar_type()), absent, absent.clone()};
}

}  // namespace
}  // namespace deepmd_dpa4c_cpu

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.impl("dpa4c_graph_compress", torch::kCPU, &deepmd_dpa4c_cpu::forward);
  library.impl("dpa4c_graph_compress_backward", torch::kCPU,
               &deepmd_dpa4c_cpu::backward);
}
