// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Force and virial assembly for an edge graph with two CSR views.
//
// Both CSR views store permutations into the original edge payload. One warp
// owns one node and reduces both incidence lists:
//
//   force[node]       = sum(dst=node) g_e - sum(src=node) g_e
//   atom_virial[node] = sum(src=node) -g_e (x) edge_vec
//
// Every node therefore writes its force and atom virial exactly once; the hot
// path contains no global floating-point atomics. Per-frame virials are reduced
// from the per-node values through two FP64 stages and cast to the model
// precision only at the output boundary.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / 32;
constexpr int kMaximumVirialPartials = 1024;

bool valid_spin_cotangent(const torch::Tensor& edge_gradient,
                          const torch::Tensor& edge_spin_gradient) {
  if (edge_spin_gradient.dim() == 1) {
    return edge_spin_gradient.numel() == 0;
  }
  return edge_spin_gradient.dim() == 2 && edge_spin_gradient.is_cuda() &&
         edge_spin_gradient.is_contiguous() &&
         edge_spin_gradient.device() == edge_gradient.device() &&
         edge_spin_gradient.sizes() == edge_gradient.sizes() &&
         edge_spin_gradient.scalar_type() == edge_gradient.scalar_type();
}

#define FORCE_CHECK_LAUNCH(name)                                              \
  do {                                                                        \
    const cudaError_t error = cudaGetLastError();                             \
    TORCH_CHECK(error == cudaSuccess, name, ": ", cudaGetErrorString(error)); \
  } while (0)

__global__ void build_source_order_kernel(long valid_edge_count,
                                          long edge_count,
                                          const long* __restrict__ source,
                                          long* __restrict__ cursor,
                                          long* __restrict__ source_order) {
  for (long edge = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
       edge < edge_count; edge += static_cast<long>(blockDim.x) * gridDim.x) {
    if (edge < valid_edge_count) {
      const long position = atomicAdd(
          reinterpret_cast<unsigned long long*>(cursor + source[edge]), 1ULL);
      source_order[position] = edge;
    } else {
      source_order[edge] = edge;
    }
  }
}

template <typename scalar_t, typename index_t, bool HasSpin>
__global__ void edge_force_virial_kernel(
    long node_count,
    const scalar_t* __restrict__ edge_gradient,
    const scalar_t* __restrict__ edge_vec,
    const bool* __restrict__ edge_mask,
    const index_t* __restrict__ destination_order,
    const long* __restrict__ destination_row_ptr,
    const index_t* __restrict__ source_order,
    const long* __restrict__ source_row_ptr,
    const scalar_t* __restrict__ edge_spin_gradient,
    scalar_t* __restrict__ force,
    scalar_t* __restrict__ node_virial,
    scalar_t* __restrict__ magnetic_force) {
  constexpr unsigned kWarpMask = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const long node = static_cast<long>(blockIdx.x) * kWarpsPerBlock + warp;

  scalar_t destination_x = 0;
  scalar_t destination_y = 0;
  scalar_t destination_z = 0;
  scalar_t source_x = 0;
  scalar_t source_y = 0;
  scalar_t source_z = 0;
  // The magnetic cotangent of an edge belongs to the node that sources it, the
  // same grouping the force reduction already walks, so it rides the source
  // loop and the warp fold rather than a second pass over the edge axis. The
  // geometric instantiation carries neither the accumulators nor the fold.
  scalar_t spin_x = 0;
  scalar_t spin_y = 0;
  scalar_t spin_z = 0;
  scalar_t virial[9] = {};

  if (node < node_count) {
    const long destination_begin = destination_row_ptr[node];
    const long destination_end = destination_row_ptr[node + 1];
    for (long position = destination_begin + lane; position < destination_end;
         position += 32) {
      const long edge = destination_order
                            ? static_cast<long>(destination_order[position])
                            : position;
      if (edge_mask && !edge_mask[edge]) {
        continue;
      }
      destination_x += edge_gradient[edge * 3 + 0];
      destination_y += edge_gradient[edge * 3 + 1];
      destination_z += edge_gradient[edge * 3 + 2];
    }

    const long source_begin = source_row_ptr[node];
    const long source_end = source_row_ptr[node + 1];
    for (long position = source_begin + lane; position < source_end;
         position += 32) {
      const long edge = static_cast<long>(source_order[position]);
      if (edge_mask && !edge_mask[edge]) {
        continue;
      }
      const scalar_t gx = edge_gradient[edge * 3 + 0];
      const scalar_t gy = edge_gradient[edge * 3 + 1];
      const scalar_t gz = edge_gradient[edge * 3 + 2];
      const scalar_t x = edge_vec[edge * 3 + 0];
      const scalar_t y = edge_vec[edge * 3 + 1];
      const scalar_t z = edge_vec[edge * 3 + 2];
      source_x += gx;
      source_y += gy;
      source_z += gz;
      if constexpr (HasSpin) {
        spin_x += edge_spin_gradient[edge * 3 + 0];
        spin_y += edge_spin_gradient[edge * 3 + 1];
        spin_z += edge_spin_gradient[edge * 3 + 2];
      }
      virial[0] = fma(-gx, x, virial[0]);
      virial[1] = fma(-gx, y, virial[1]);
      virial[2] = fma(-gx, z, virial[2]);
      virial[3] = fma(-gy, x, virial[3]);
      virial[4] = fma(-gy, y, virial[4]);
      virial[5] = fma(-gy, z, virial[5]);
      virial[6] = fma(-gz, x, virial[6]);
      virial[7] = fma(-gz, y, virial[7]);
      virial[8] = fma(-gz, z, virial[8]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    destination_x += __shfl_down_sync(kWarpMask, destination_x, offset);
    destination_y += __shfl_down_sync(kWarpMask, destination_y, offset);
    destination_z += __shfl_down_sync(kWarpMask, destination_z, offset);
    source_x += __shfl_down_sync(kWarpMask, source_x, offset);
    source_y += __shfl_down_sync(kWarpMask, source_y, offset);
    source_z += __shfl_down_sync(kWarpMask, source_z, offset);
    if constexpr (HasSpin) {
      spin_x += __shfl_down_sync(kWarpMask, spin_x, offset);
      spin_y += __shfl_down_sync(kWarpMask, spin_y, offset);
      spin_z += __shfl_down_sync(kWarpMask, spin_z, offset);
    }
#pragma unroll
    for (int component = 0; component < 9; ++component) {
      virial[component] +=
          __shfl_down_sync(kWarpMask, virial[component], offset);
    }
  }

  if (lane == 0 && node < node_count) {
    force[node * 3 + 0] = destination_x - source_x;
    force[node * 3 + 1] = destination_y - source_y;
    force[node * 3 + 2] = destination_z - source_z;
    scalar_t* output = node_virial + node * 9;
#pragma unroll
    for (int component = 0; component < 9; ++component) {
      output[component] = virial[component];
    }
    if constexpr (HasSpin) {
      magnetic_force[node * 3 + 0] = spin_x;
      magnetic_force[node * 3 + 1] = spin_y;
      magnetic_force[node * 3 + 2] = spin_z;
    }
  }
}

// Partial pass of a segment sum over the node axis. Each frame owns a
// contiguous span of that axis, so a block can reduce a strided slice of one
// (frame, component) pair without any atomic, and the reduction order follows
// the launch geometry alone and is therefore reproducible. Accumulation is
// fp64 whatever the stored type.
template <typename scalar_t, int kComponents>
__global__ void reduce_frame_segment_kernel(
    long frame_count,
    int partial_count,
    const long* __restrict__ frame_row_ptr,
    const scalar_t* __restrict__ node_values,
    double* __restrict__ partial) {
  __shared__ double values[kThreads];
  const long task_count =
      static_cast<long>(frame_count) * kComponents * partial_count;
  for (long task = blockIdx.x; task < task_count; task += gridDim.x) {
    const int partial_index = task % partial_count;
    const long output = task / partial_count;
    const long frame = output / kComponents;
    const int component = output % kComponents;
    const long begin = frame_row_ptr[frame];
    const long end = frame_row_ptr[frame + 1];
    double sum = 0.0;
    for (long node = begin + partial_index * static_cast<long>(blockDim.x) +
                     threadIdx.x;
         node < end; node += static_cast<long>(partial_count) * blockDim.x) {
      sum += static_cast<double>(node_values[node * kComponents + component]);
    }
    values[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        values[threadIdx.x] += values[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      partial[output * partial_count + partial_index] = values[0];
    }
    __syncthreads();
  }
}

// Final pass of a segment sum: fold the per-block partials of each output.
template <typename scalar_t>
__global__ void finalize_frame_segment_kernel(
    long output_count,
    int partial_count,
    const double* __restrict__ partial,
    scalar_t* __restrict__ out) {
  __shared__ double values[kThreads];
  for (long output = blockIdx.x; output < output_count; output += gridDim.x) {
    double sum = 0.0;
    for (int index = threadIdx.x; index < partial_count; index += blockDim.x) {
      sum += partial[output * partial_count + index];
    }
    values[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        values[threadIdx.x] += values[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      out[output] = static_cast<scalar_t>(values[0]);
    }
    __syncthreads();
  }
}

// Exclusive prefix of the per-frame node counts, giving each frame's span of
// the node axis as ``[row_ptr[f], row_ptr[f + 1])``.
torch::Tensor frame_row_pointer(const torch::Tensor& n_node_per_frame) {
  return torch::cat({torch::zeros({1}, n_node_per_frame.options()),
                     torch::cumsum(n_node_per_frame, 0)})
      .to(torch::kInt64)
      .contiguous();
}

// Slices per frame that the partial pass reduces in parallel. Bounding it
// keeps the partial buffer small for a single large frame, which is the
// molecular dynamics case.
int frame_segment_partials(long node_count, long frame_count) {
  const long average_node_count = (node_count + frame_count - 1) / frame_count;
  return static_cast<int>(
      std::min((average_node_count + kThreads - 1) / kThreads,
               static_cast<long>(kMaximumVirialPartials)));
}

// Sum ``node_values``, laid out as (node, component), over the node segment of
// each frame. Returns a flat (frame * kComponents) buffer.
template <typename scalar_t, int kComponents>
void launch_frame_segment_sum(long node_count,
                              long frame_count,
                              const torch::Tensor& frame_row_ptr,
                              const scalar_t* node_values,
                              torch::Tensor& partial,
                              scalar_t* out,
                              cudaStream_t stream) {
  const int partial_count = static_cast<int>(partial.size(1));
  const long output_count = frame_count * kComponents;

  const int partial_blocks = std::min(output_count * partial_count, 65535L);
  reduce_frame_segment_kernel<scalar_t, kComponents>
      <<<partial_blocks, kThreads, 0, stream>>>(
          frame_count, partial_count, frame_row_ptr.data_ptr<long>(),
          node_values, partial.data_ptr<double>());
  FORCE_CHECK_LAUNCH("frame segment sum partial reduction");

  const int final_blocks = std::min(output_count, 65535L);
  finalize_frame_segment_kernel<scalar_t>
      <<<final_blocks, kThreads, 0, stream>>>(output_count, partial_count,
                                              partial.data_ptr<double>(), out);
  FORCE_CHECK_LAUNCH("frame segment sum final reduction");
}

template <typename scalar_t, typename index_t>
void launch_force_virial(long node_count,
                         long frame_count,
                         const torch::Tensor& edge_gradient,
                         const torch::Tensor& edge_vec,
                         const torch::Tensor& edge_mask,
                         const torch::Tensor& destination_order,
                         const torch::Tensor& destination_row_ptr,
                         const torch::Tensor& source_order,
                         const torch::Tensor& source_row_ptr,
                         const torch::Tensor& frame_row_ptr,
                         const torch::Tensor& edge_spin_gradient,
                         bool has_spin,
                         torch::Tensor& force,
                         torch::Tensor& node_virial,
                         torch::Tensor& magnetic_force,
                         torch::Tensor& virial_partial,
                         torch::Tensor& virial,
                         cudaStream_t stream) {
  const int node_blocks =
      static_cast<int>((node_count + kWarpsPerBlock - 1) / kWarpsPerBlock);
  auto assemble = [&](auto spin_tag) {
    edge_force_virial_kernel<scalar_t, index_t, decltype(spin_tag)::value>
        <<<node_blocks, kThreads, 0, stream>>>(
            node_count, edge_gradient.data_ptr<scalar_t>(),
            edge_vec.data_ptr<scalar_t>(),
            edge_mask.numel() ? edge_mask.data_ptr<bool>() : nullptr,
            destination_order.numel() ? destination_order.data_ptr<index_t>()
                                      : nullptr,
            destination_row_ptr.data_ptr<long>(),
            source_order.data_ptr<index_t>(), source_row_ptr.data_ptr<long>(),
            has_spin ? edge_spin_gradient.data_ptr<scalar_t>() : nullptr,
            force.data_ptr<scalar_t>(), node_virial.data_ptr<scalar_t>(),
            has_spin ? magnetic_force.data_ptr<scalar_t>() : nullptr);
  };
  if (has_spin) {
    assemble(std::true_type{});
  } else {
    assemble(std::false_type{});
  }
  FORCE_CHECK_LAUNCH("edge_force_virial node reduction");

  launch_frame_segment_sum<scalar_t, 9>(
      node_count, frame_count, frame_row_ptr, node_virial.data_ptr<scalar_t>(),
      virial_partial, virial.data_ptr<scalar_t>(), stream);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
assemble_force_virial(long node_count,
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
  const long frame_count = n_node_per_frame.size(0);
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

  auto frame_row_ptr = frame_row_pointer(n_node_per_frame);
  const int partial_count = frame_segment_partials(node_count, frame_count);
  auto virial_partial = torch::empty({frame_count * 9, partial_count},
                                     options.dtype(torch::kFloat64));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
      edge_gradient.scalar_type(), "edge_force_virial", [&] {
        if (source_order.scalar_type() == torch::kInt32) {
          launch_force_virial<scalar_t, int>(
              node_count, frame_count, edge_gradient, edge_vec, edge_mask,
              destination_order, destination_row_ptr, source_order,
              source_row_ptr, frame_row_ptr, edge_spin_gradient, has_spin,
              force, node_virial, magnetic_force, virial_partial, virial,
              stream);
        } else if (source_order.scalar_type() == torch::kUInt32) {
          launch_force_virial<scalar_t, std::uint32_t>(
              node_count, frame_count, edge_gradient, edge_vec, edge_mask,
              destination_order, destination_row_ptr, source_order,
              source_row_ptr, frame_row_ptr, edge_spin_gradient, has_spin,
              force, node_virial, magnetic_force, virial_partial, virial,
              stream);
        } else {
          launch_force_virial<scalar_t, long>(
              node_count, frame_count, edge_gradient, edge_vec, edge_mask,
              destination_order, destination_row_ptr, source_order,
              source_row_ptr, frame_row_ptr, edge_spin_gradient, has_spin,
              force, node_virial, magnetic_force, virial_partial, virial,
              stream);
        }
      });
  return {force, atom_virial, virial, magnetic_force};
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_graph_csr(torch::Tensor edge_index,
                c10::SymInt node_count_symbol,
                c10::SymInt valid_edge_count_symbol) {
  const long node_count = node_count_symbol.expect_int();
  const long valid_edge_count = valid_edge_count_symbol.expect_int();
  const long edge_count = edge_index.size(1);
  TORCH_CHECK(edge_index.is_cuda() && edge_index.is_contiguous(),
              "build_graph_csr: edge_index must be a contiguous CUDA tensor");
  TORCH_CHECK(edge_index.scalar_type() == torch::kInt64,
              "build_graph_csr: edge_index must be int64");
  TORCH_CHECK(node_count > 0, "build_graph_csr: node_count must be positive");
  TORCH_CHECK(valid_edge_count >= 0 && valid_edge_count <= edge_count,
              "build_graph_csr: valid_edge_count must lie in [0, E]");

  const auto valid_source =
      edge_index.select(0, 0).slice(0, 0, valid_edge_count);
  const auto valid_destination =
      edge_index.select(0, 1).slice(0, 0, valid_edge_count);
  const auto source_counts = torch::bincount(valid_source, {}, node_count);
  const auto destination_counts =
      torch::bincount(valid_destination, {}, node_count);
  const auto zero = torch::zeros({1}, source_counts.options());
  auto source_row_ptr = torch::cat({zero, torch::cumsum(source_counts, 0)})
                            .to(torch::kInt64)
                            .contiguous();
  auto destination_row_ptr =
      torch::cat({zero, torch::cumsum(destination_counts, 0)})
          .to(torch::kInt64)
          .contiguous();
  auto destination_order = torch::arange(edge_count, edge_index.options());
  auto source_order = torch::empty({edge_count}, edge_index.options());
  auto cursor = source_row_ptr.slice(0, 0, node_count).clone();

  const int blocks =
      std::min(static_cast<int>((edge_count + kThreads - 1) / kThreads), 65535);
  if (blocks > 0) {
    const auto stream = at::cuda::getCurrentCUDAStream();
    build_source_order_kernel<<<blocks, kThreads, 0, stream>>>(
        valid_edge_count, edge_count, edge_index.select(0, 0).data_ptr<long>(),
        cursor.data_ptr<long>(), source_order.data_ptr<long>());
    FORCE_CHECK_LAUNCH("build_graph_csr source order");
  }
  return {destination_order, destination_row_ptr, source_order, source_row_ptr};
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
  const long node_count = node_capacity.expect_int();
  TORCH_CHECK(edge_gradient.is_cuda() && edge_vec.is_cuda() &&
                  edge_mask.is_cuda() && destination_order.is_cuda() &&
                  destination_row_ptr.is_cuda() && source_order.is_cuda() &&
                  source_row_ptr.is_cuda() && n_node_per_frame.is_cuda(),
              "edge_force_virial: edge and CSR tensors must be CUDA tensors");
  TORCH_CHECK(edge_gradient.is_contiguous() && edge_vec.is_contiguous() &&
                  edge_mask.is_contiguous(),
              "edge_force_virial: edge tensors must be contiguous");
  TORCH_CHECK(edge_gradient.scalar_type() == edge_vec.scalar_type(),
              "edge_force_virial: gradient and edge vector dtypes must match");
  TORCH_CHECK(edge_mask.scalar_type() == torch::kBool,
              "edge_force_virial: edge_mask must be bool");
  TORCH_CHECK(destination_row_ptr.scalar_type() == torch::kInt64 &&
                  source_row_ptr.scalar_type() == torch::kInt64,
              "edge_force_virial: CSR row pointers must be int64");
  TORCH_CHECK(destination_order.is_contiguous() &&
                  destination_row_ptr.is_contiguous() &&
                  source_order.is_contiguous() &&
                  source_row_ptr.is_contiguous(),
              "edge_force_virial: CSR tensors must be contiguous");
  TORCH_CHECK(
      (source_order.scalar_type() == torch::kInt32 ||
       source_order.scalar_type() == torch::kUInt32 ||
       source_order.scalar_type() == torch::kInt64) &&
          destination_order.scalar_type() == source_order.scalar_type(),
      "edge_force_virial: destination_order and source_order must have the "
      "same int32, uint32, or int64 dtype");
  TORCH_CHECK(valid_spin_cotangent(edge_gradient, edge_spin_gradient),
              "edge_force_virial: edge_spin_gradient must match the gradient "
              "in device, layout, shape and dtype, or be a rank-one empty "
              "sentinel");
  return assemble_force_virial(node_count, edge_gradient, edge_vec, edge_mask,
                               destination_order, destination_row_ptr,
                               source_order, source_row_ptr, n_node_per_frame,
                               edge_spin_gradient, want_atom_virial);
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
  const long node_count = node_capacity.expect_int();
  TORCH_CHECK(edge_gradient.is_cuda() && edge_vec.is_cuda() &&
                  destination_row_ptr.is_cuda() && source_row_ptr.is_cuda() &&
                  source_order.is_cuda() && n_node_per_frame.is_cuda(),
              "canonical_edge_force_virial: inputs must be CUDA tensors");
  TORCH_CHECK(edge_gradient.is_contiguous() && edge_vec.is_contiguous() &&
                  destination_row_ptr.is_contiguous() &&
                  source_row_ptr.is_contiguous() &&
                  source_order.is_contiguous(),
              "canonical_edge_force_virial: inputs must be contiguous");
  TORCH_CHECK(edge_gradient.sizes() == edge_vec.sizes() &&
                  edge_gradient.scalar_type() == edge_vec.scalar_type(),
              "canonical_edge_force_virial: gradient and edge vector must "
              "share shape and dtype");
  TORCH_CHECK(destination_row_ptr.scalar_type() == torch::kInt64 &&
                  source_row_ptr.scalar_type() == torch::kInt64,
              "canonical_edge_force_virial: row pointers must be int64");
  TORCH_CHECK(source_order.scalar_type() == torch::kInt32 ||
                  source_order.scalar_type() == torch::kUInt32 ||
                  source_order.scalar_type() == torch::kInt64,
              "canonical_edge_force_virial: source_order must be int32, "
              "uint32, or int64");
  TORCH_CHECK(destination_row_ptr.numel() == node_count + 1 &&
                  source_row_ptr.numel() == node_count + 1,
              "canonical_edge_force_virial: row pointers must have N + 1 "
              "entries");
  TORCH_CHECK(
      valid_spin_cotangent(edge_gradient, edge_spin_gradient),
      "canonical_edge_force_virial: edge_spin_gradient must match the "
      "gradient in device, layout, shape and dtype, or be a rank-one empty "
      "sentinel");

  auto edge_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto destination_order = torch::empty({0}, source_order.options());
  return assemble_force_virial(node_count, edge_gradient, edge_vec, edge_mask,
                               destination_order, destination_row_ptr,
                               source_order, source_row_ptr, n_node_per_frame,
                               edge_spin_gradient, want_atom_virial);
}

// Per-frame total of a scalar carried on the node axis, the energy being the
// only such quantity. Scattering it with an index add would serialize one
// atomic per node on a single address whenever the batch holds one frame,
// which is the molecular dynamics case; the segment reduction that already
// serves the virial has no atomic and no node-length index to materialize.
//
// The frames cover ``[0, sum(n_node_per_frame))`` of the node axis; nodes past
// that are padding and contribute to no frame. The caller owns that
// invariant, since checking it would force a device synchronization.
torch::Tensor frame_scalar_sum(torch::Tensor node_scalar,
                               torch::Tensor n_node_per_frame) {
  const long node_count = node_scalar.size(0);
  const long frame_count = n_node_per_frame.size(0);
  TORCH_CHECK(node_scalar.is_cuda() && n_node_per_frame.is_cuda(),
              "frame_scalar_sum: inputs must be CUDA tensors");
  TORCH_CHECK(node_scalar.device() == n_node_per_frame.device(),
              "frame_scalar_sum: inputs must share one device");
  TORCH_CHECK(node_scalar.is_contiguous(),
              "frame_scalar_sum: node_scalar must be contiguous");
  TORCH_CHECK(node_scalar.dim() == 2 && node_scalar.size(1) == 1,
              "frame_scalar_sum: node_scalar must have shape (N, 1)");
  TORCH_CHECK(n_node_per_frame.dim() == 1,
              "frame_scalar_sum: n_node_per_frame must be one-dimensional");

  auto out = torch::zeros({frame_count, 1}, node_scalar.options());
  if (node_count == 0 || frame_count == 0) {
    return out;
  }
  auto frame_row_ptr = frame_row_pointer(n_node_per_frame);
  const int partial_count = frame_segment_partials(node_count, frame_count);
  auto partial = torch::empty({frame_count, partial_count},
                              node_scalar.options().dtype(torch::kFloat64));
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(node_scalar.scalar_type(), "frame_scalar_sum",
                             [&] {
                               launch_frame_segment_sum<scalar_t, 1>(
                                   node_count, frame_count, frame_row_ptr,
                                   node_scalar.data_ptr<scalar_t>(), partial,
                                   out.data_ptr<scalar_t>(), stream);
                             });
  return out;
}

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.impl("build_graph_csr", torch::kCUDA, &build_graph_csr);
  library.impl("edge_force_virial", torch::kCUDA, &edge_force_virial);
  library.impl("canonical_edge_force_virial", torch::kCUDA,
               &canonical_edge_force_virial);
  library.impl("frame_scalar_sum", torch::kCUDA, &frame_scalar_sum);
}
