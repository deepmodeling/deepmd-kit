// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Force and virial assembly of an edge graph on the CPU.
//
// The two CSR views make every node's incidence lists contiguous, so a node
// reduces both of them into registers and writes its force and virial once:
//
//   force[node]       = sum(dst=node) g_e - sum(src=node) g_e
//   atom_virial[node] = sum(src=node) -g_e (x) edge_vec
//
// The aten lowering this replaces expresses the same reduction as three
// scatters into the node axis. On a GPU those serialize on colliding edges;
// on a CPU they become a locked read-modify-write per component, which at a
// hundred and fifty neighbours per atom is the single most expensive
// operation of a compressed step. Owning the node removes the contention
// entirely.
//
// Per-frame virials accumulate in double whatever the stored precision, and
// the partial sums follow the thread partition rather than arrival order, so
// the result is reproducible for a fixed thread count.

#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include "group.h"
#include "partition.h"

namespace {

/// Row pointer of the frame partition of the node axis.
std::vector<int64_t> frame_row_pointer(const torch::Tensor& n_node_per_frame) {
  const auto counts = n_node_per_frame.to(torch::kCPU).contiguous();
  const int64_t frames = counts.numel();
  std::vector<int64_t> offsets(frames + 1, 0);
  const int64_t* data = counts.const_data_ptr<int64_t>();
  for (int64_t frame = 0; frame < frames; ++frame) {
    offsets[frame + 1] = offsets[frame] + data[frame];
  }
  return offsets;
}

/// Reduce one node range's incidence lists.
template <typename scalar_t, typename index_t, bool HasSpin>
void assemble_range(int64_t node_begin,
                    int64_t node_end,
                    const scalar_t* __restrict__ edge_gradient,
                    const scalar_t* __restrict__ edge_vec,
                    const bool* __restrict__ edge_mask,
                    const index_t* __restrict__ destination_order,
                    const int64_t* __restrict__ destination_row_ptr,
                    const index_t* __restrict__ source_order,
                    const int64_t* __restrict__ source_row_ptr,
                    const scalar_t* __restrict__ edge_spin_gradient,
                    scalar_t* __restrict__ force,
                    scalar_t* __restrict__ node_virial,
                    scalar_t* __restrict__ magnetic_force) {
  for (int64_t node = node_begin; node < node_end; ++node) {
    scalar_t incoming[3] = {0, 0, 0};
    scalar_t outgoing[3] = {0, 0, 0};
    scalar_t magnetic[3] = {0, 0, 0};
    scalar_t virial[9] = {};

    for (int64_t position = destination_row_ptr[node];
         position < destination_row_ptr[node + 1]; ++position) {
      const int64_t edge =
          destination_order ? static_cast<int64_t>(destination_order[position])
                            : position;
      if (edge_mask && !edge_mask[edge]) {
        continue;
      }
      incoming[0] += edge_gradient[edge * 3 + 0];
      incoming[1] += edge_gradient[edge * 3 + 1];
      incoming[2] += edge_gradient[edge * 3 + 2];
    }

    for (int64_t position = source_row_ptr[node];
         position < source_row_ptr[node + 1]; ++position) {
      const int64_t edge = static_cast<int64_t>(source_order[position]);
      if (edge_mask && !edge_mask[edge]) {
        continue;
      }
      const scalar_t* gradient = edge_gradient + edge * 3;
      const scalar_t* vector = edge_vec + edge * 3;
      outgoing[0] += gradient[0];
      outgoing[1] += gradient[1];
      outgoing[2] += gradient[2];
      if (HasSpin) {
        magnetic[0] += edge_spin_gradient[edge * 3 + 0];
        magnetic[1] += edge_spin_gradient[edge * 3 + 1];
        magnetic[2] += edge_spin_gradient[edge * 3 + 2];
      }
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          virial[row * 3 + column] -= gradient[row] * vector[column];
        }
      }
    }

    for (int component = 0; component < 3; ++component) {
      force[node * 3 + component] = incoming[component] - outgoing[component];
    }
    for (int component = 0; component < 9; ++component) {
      node_virial[node * 9 + component] = virial[component];
    }
    if (HasSpin) {
      for (int component = 0; component < 3; ++component) {
        magnetic_force[node * 3 + component] = magnetic[component];
      }
    }
  }
}

/// Reduce per-node values into per-frame sums in double precision.
template <typename scalar_t, int kComponents>
void reduce_frames(const std::vector<int64_t>& frame_row_ptr,
                   const scalar_t* __restrict__ node_values,
                   scalar_t* __restrict__ frame_values) {
  const int64_t frames = static_cast<int64_t>(frame_row_ptr.size()) - 1;
  at::parallel_for(0, frames, 1, [&](int64_t begin, int64_t end) {
    for (int64_t frame = begin; frame < end; ++frame) {
      double totals[kComponents] = {};
      for (int64_t node = frame_row_ptr[frame]; node < frame_row_ptr[frame + 1];
           ++node) {
        for (int component = 0; component < kComponents; ++component) {
          totals[component] +=
              static_cast<double>(node_values[node * kComponents + component]);
        }
      }
      for (int component = 0; component < kComponents; ++component) {
        frame_values[frame * kComponents + component] =
            static_cast<scalar_t>(totals[component]);
      }
    }
  });
}

/// Reduce the per-frame virial when a single frame owns the whole node axis.
///
/// One frame is the molecular-dynamics case, where the frame loop above would
/// leave the reduction to one thread. Splitting the node axis across threads
/// and merging their doubles keeps the accumulation exact to double and the
/// order fixed by the partition.
template <typename scalar_t, int kComponents>
void reduce_single_frame(int64_t node_count,
                         const scalar_t* __restrict__ node_values,
                         scalar_t* __restrict__ frame_values) {
  const int threads = std::max(1, at::get_num_threads());
  std::vector<double> partial(static_cast<size_t>(threads) * kComponents, 0.0);
  at::parallel_for(0, threads, 1, [&](int64_t begin, int64_t end) {
    for (int64_t part = begin; part < end; ++part) {
      const int64_t first = node_count * part / threads;
      const int64_t last = node_count * (part + 1) / threads;
      double* totals = partial.data() + part * kComponents;
      for (int64_t node = first; node < last; ++node) {
        for (int component = 0; component < kComponents; ++component) {
          totals[component] +=
              static_cast<double>(node_values[node * kComponents + component]);
        }
      }
    }
  });
  for (int component = 0; component < kComponents; ++component) {
    double total = 0.0;
    for (int part = 0; part < threads; ++part) {
      total += partial[static_cast<size_t>(part) * kComponents + component];
    }
    frame_values[component] = static_cast<scalar_t>(total);
  }
}

/// Assemble force, node virial, per-frame virial and magnetic force.
template <typename scalar_t, typename index_t>
void assemble(int64_t node_count,
              const torch::Tensor& edge_gradient,
              const torch::Tensor& edge_vec,
              const torch::Tensor& edge_mask,
              const torch::Tensor& destination_order,
              const torch::Tensor& destination_row_ptr,
              const torch::Tensor& source_order,
              const torch::Tensor& source_row_ptr,
              const torch::Tensor& edge_spin_gradient,
              bool has_spin,
              torch::Tensor& force,
              torch::Tensor& node_virial,
              torch::Tensor& magnetic_force) {
  const int64_t* destination_pointer =
      destination_row_ptr.const_data_ptr<int64_t>();
  const int64_t* source_pointer = source_row_ptr.const_data_ptr<int64_t>();
  const index_t* destination_index =
      destination_order.numel() == 0
          ? nullptr
          : destination_order.const_data_ptr<index_t>();
  const bool* mask =
      edge_mask.numel() == 0 ? nullptr : edge_mask.const_data_ptr<bool>();
  const scalar_t* spin_gradient =
      has_spin ? edge_spin_gradient.const_data_ptr<scalar_t>() : nullptr;
  scalar_t* magnetic = has_spin ? magnetic_force.data_ptr<scalar_t>() : nullptr;

  const int threads = std::max(1, at::get_num_threads());
  const std::vector<deepmd_cpu::NodeRange> ranges =
      deepmd_cpu::balanced_ranges(source_pointer, node_count, threads);
  at::parallel_for(
      0, static_cast<int64_t>(ranges.size()), 1,
      [&](int64_t begin, int64_t end) {
        for (int64_t part = begin; part < end; ++part) {
          const auto& range = ranges[part];
          if (has_spin) {
            assemble_range<scalar_t, index_t, true>(
                range.begin, range.end,
                edge_gradient.const_data_ptr<scalar_t>(),
                edge_vec.const_data_ptr<scalar_t>(), mask, destination_index,
                destination_pointer, source_order.const_data_ptr<index_t>(),
                source_pointer, spin_gradient, force.data_ptr<scalar_t>(),
                node_virial.data_ptr<scalar_t>(), magnetic);
          } else {
            assemble_range<scalar_t, index_t, false>(
                range.begin, range.end,
                edge_gradient.const_data_ptr<scalar_t>(),
                edge_vec.const_data_ptr<scalar_t>(), mask, destination_index,
                destination_pointer, source_order.const_data_ptr<index_t>(),
                source_pointer, spin_gradient, force.data_ptr<scalar_t>(),
                node_virial.data_ptr<scalar_t>(), magnetic);
          }
        }
      });
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
assemble_entry(int64_t node_count,
               const torch::Tensor& edge_gradient,
               const torch::Tensor& edge_vec,
               const torch::Tensor& edge_mask,
               const torch::Tensor& destination_order,
               const torch::Tensor& destination_row_ptr,
               const torch::Tensor& source_order,
               const torch::Tensor& source_row_ptr,
               const torch::Tensor& n_node_per_frame,
               const torch::Tensor& edge_spin_gradient,
               bool want_atom_virial) {
  const int64_t frame_count = n_node_per_frame.size(0);
  auto options = edge_gradient.options();
  auto force = torch::empty({node_count, 3}, options);
  auto atom_virial =
      torch::empty({want_atom_virial ? node_count : 0, 3, 3}, options);
  auto node_virial = want_atom_virial
                         ? atom_virial
                         : torch::empty({node_count, 3, 3}, options);
  auto virial = torch::zeros({frame_count, 3, 3}, options);
  const bool has_spin = edge_spin_gradient.dim() == 2;
  auto magnetic_force = has_spin ? torch::empty({node_count, 3}, options)
                                 : torch::empty({0}, options);
  if (node_count == 0 || frame_count == 0) {
    return {force, atom_virial, virial, magnetic_force};
  }

  AT_DISPATCH_FLOATING_TYPES(
      edge_gradient.scalar_type(), "edge_force_virial_cpu", [&] {
        switch (source_order.scalar_type()) {
          case torch::kInt32:
            assemble<scalar_t, int32_t>(
                node_count, edge_gradient, edge_vec, edge_mask,
                destination_order, destination_row_ptr, source_order,
                source_row_ptr, edge_spin_gradient, has_spin, force,
                node_virial, magnetic_force);
            break;
          case torch::kUInt32:
            assemble<scalar_t, uint32_t>(
                node_count, edge_gradient, edge_vec, edge_mask,
                destination_order, destination_row_ptr, source_order,
                source_row_ptr, edge_spin_gradient, has_spin, force,
                node_virial, magnetic_force);
            break;
          default:
            assemble<scalar_t, int64_t>(
                node_count, edge_gradient, edge_vec, edge_mask,
                destination_order, destination_row_ptr, source_order,
                source_row_ptr, edge_spin_gradient, has_spin, force,
                node_virial, magnetic_force);
            break;
        }
        if (frame_count == 1) {
          reduce_single_frame<scalar_t, 9>(
              node_count, node_virial.const_data_ptr<scalar_t>(),
              virial.data_ptr<scalar_t>());
        } else {
          reduce_frames<scalar_t, 9>(frame_row_pointer(n_node_per_frame),
                                     node_virial.const_data_ptr<scalar_t>(),
                                     virial.data_ptr<scalar_t>());
        }
      });
  return {force, atom_virial, virial, magnetic_force};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
edge_force_virial(torch::Tensor edge_gradient,
                  torch::Tensor edge_vec,
                  torch::Tensor edge_index,
                  torch::Tensor edge_mask,
                  torch::Tensor destination_order,
                  torch::Tensor destination_row_ptr,
                  torch::Tensor source_order,
                  torch::Tensor source_row_ptr,
                  torch::Tensor n_node_per_frame,
                  torch::Tensor edge_spin_gradient,
                  c10::SymInt node_capacity,
                  bool want_atom_virial) {
  TORCH_CHECK(edge_gradient.device().is_cpu(),
              "edge_force_virial: the CPU kernel needs CPU tensors");
  (void)edge_index;
  return assemble_entry(
      node_capacity.expect_int(), edge_gradient.contiguous(),
      edge_vec.contiguous(), edge_mask.contiguous(),
      destination_order.contiguous(), destination_row_ptr.contiguous(),
      source_order.contiguous(), source_row_ptr.contiguous(), n_node_per_frame,
      edge_spin_gradient.contiguous(), want_atom_virial);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
canonical_edge_force_virial(torch::Tensor edge_gradient,
                            torch::Tensor edge_vec,
                            torch::Tensor destination_row_ptr,
                            torch::Tensor source_row_ptr,
                            torch::Tensor source_order,
                            torch::Tensor n_node_per_frame,
                            torch::Tensor edge_spin_gradient,
                            c10::SymInt node_capacity,
                            bool want_atom_virial) {
  TORCH_CHECK(edge_gradient.device().is_cpu(),
              "canonical_edge_force_virial: the CPU kernel needs CPU tensors");
  const auto empty_index = torch::empty({0}, source_order.options());
  const auto empty_mask =
      torch::empty({0}, edge_gradient.options().dtype(torch::kBool));
  return assemble_entry(node_capacity.expect_int(), edge_gradient.contiguous(),
                        edge_vec.contiguous(), empty_mask, empty_index,
                        destination_row_ptr.contiguous(),
                        source_order.contiguous(), source_row_ptr.contiguous(),
                        n_node_per_frame, edge_spin_gradient.contiguous(),
                        want_atom_virial);
}

torch::Tensor frame_scalar_sum(torch::Tensor node_scalar,
                               torch::Tensor n_node_per_frame) {
  TORCH_CHECK(node_scalar.device().is_cpu(),
              "frame_scalar_sum: the CPU kernel needs CPU tensors");
  TORCH_CHECK(node_scalar.dim() == 2 && node_scalar.size(1) == 1,
              "frame_scalar_sum: node_scalar must have shape (N, 1)");
  auto contiguous = node_scalar.contiguous();
  const int64_t frames = n_node_per_frame.size(0);
  auto total = torch::zeros({frames, 1}, contiguous.options());
  if (frames == 0) {
    return total;
  }
  AT_DISPATCH_FLOATING_TYPES(
      contiguous.scalar_type(), "frame_scalar_sum_cpu", [&] {
        if (frames == 1) {
          reduce_single_frame<scalar_t, 1>(
              contiguous.size(0), contiguous.const_data_ptr<scalar_t>(),
              total.data_ptr<scalar_t>());
        } else {
          reduce_frames<scalar_t, 1>(frame_row_pointer(n_node_per_frame),
                                     contiguous.const_data_ptr<scalar_t>(),
                                     total.data_ptr<scalar_t>());
        }
      });
  return total;
}

/**
 * @brief Build both compressed-sparse-row views of a destination-major graph.
 *
 * The caller guarantees that the physical edges form a destination-grouped
 * prefix, which is what every producer of this ABI emits, so the destination
 * permutation is the identity and its row pointers are a histogram of the
 * destination column. Only the source view needs a grouping pass.
 *
 * @param edge_index Endpoints with shape ``(2, E)`` in ``[source,
 *   destination]`` order, the physical edges forming the prefix.
 * @param node_count_symbol Number of nodes.
 * @param valid_edge_count_symbol Length of the physical prefix.
 *
 * @return ``(destination_order, destination_row_ptr, source_order,
 *   source_row_ptr)``. Masked slots form the suffix of each permutation,
 *   outside every row.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_graph_csr(torch::Tensor edge_index,
                c10::SymInt node_count_symbol,
                c10::SymInt valid_edge_count_symbol) {
  const std::int64_t node_count = node_count_symbol.expect_int();
  const std::int64_t valid_edge_count = valid_edge_count_symbol.expect_int();
  const std::int64_t edge_count = edge_index.size(1);
  TORCH_CHECK(edge_index.device().is_cpu(),
              "build_graph_csr: edge_index must be on CPU");
  TORCH_CHECK(edge_index.scalar_type() == torch::kInt64,
              "build_graph_csr: edge_index must be int64");
  TORCH_CHECK(node_count > 0, "build_graph_csr: node_count must be positive");
  TORCH_CHECK(valid_edge_count >= 0 && valid_edge_count <= edge_count,
              "build_graph_csr: valid_edge_count must lie in [0, E]");
  const auto contiguous = edge_index.contiguous();
  const auto* source = contiguous.const_data_ptr<std::int64_t>();
  const auto* destination = source + edge_count;

  const auto index_options = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor destination_order = torch::arange(edge_count, index_options);
  torch::Tensor destination_row_ptr =
      torch::empty({node_count + 1}, index_options);
  torch::Tensor source_row_ptr = torch::empty({node_count + 1}, index_options);
  torch::Tensor source_order = torch::empty({edge_count}, index_options);
  // The destination column ascends by precondition, so its offsets are where
  // each node first appears. Searching for them is parallel over nodes and
  // touches log(E) elements each, against a histogram's serial pass over the
  // whole edge axis.
  auto* destination_row = destination_row_ptr.data_ptr<std::int64_t>();
  at::parallel_for(
      0, node_count + 1, 64, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t node = begin; node < end; ++node) {
          destination_row[node] =
              std::lower_bound(destination, destination + valid_edge_count,
                               node) -
              destination;
        }
      });
  auto* order = source_order.data_ptr<std::int64_t>();
  deepmd::group_by_node(source, valid_edge_count, node_count,
                        source_row_ptr.data_ptr<std::int64_t>(), order);
  for (std::int64_t slot = valid_edge_count; slot < edge_count; ++slot) {
    order[slot] = slot;
  }
  return {destination_order, destination_row_ptr, source_order, source_row_ptr};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.impl("edge_force_virial", torch::kCPU, &edge_force_virial);
  library.impl("canonical_edge_force_virial", torch::kCPU,
               &canonical_edge_force_virial);
  library.impl("frame_scalar_sum", torch::kCPU, &frame_scalar_sum);
  library.impl("build_graph_csr", torch::kCPU, &build_graph_csr);
}
