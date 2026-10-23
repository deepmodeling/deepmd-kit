// SPDX-License-Identifier: LGPL-3.0-or-later
//
// CUDA kernels of the latent-charge long range of periodic frames, by the
// smooth particle-mesh Ewald method (schema and formulas in ops.cc, the
// evaluation in deepmd/dpmodel/long_range/reciprocal.py).
//
// Every frame owns a mesh of n1 x n2 x n3 points spanning its cell. The
// charges spread to the mesh with cardinal B-splines of order p along the
// fractional coordinates s = r B^-1, the lattice vectors being the rows of the
// cell B: the scaled coordinate u_d = n_d s_d reaches the p points
// floor(u_d) - p + 1, ..., floor(u_d) with the weights M_p(u_d - m). A real
// FFT gives the half spectrum Q(m) of the charge mesh, and the influence
// function
//
//   G(m) = prefactor * K^(k_m) / V * prod_d |D_d(m_d)|^-2,  k_m = 2 pi B^-1 m,
//
// with the Fourier transform K^ of the kernel and the assignment factor
// D_d(m) = sum_l M_p(l + 1) exp(2 pi i m l / n_d), turns it into the
// potential mesh, the unnormalized inverse transform of G Q. The potential at
// a node is its interpolation by the B-splines of the node, less the potential
// prefactor * K(0) q of the charge at itself, and the force at fixed charges
// the negative gradient of that interpolation times the charges. The zero mode
// and the modes at the Nyquist frequency of an axis, which have no unique wave
// number, carry no weight.
//
// The virial of a frame is that of the strain of its cell, under which the
// fractional coordinates stay put while k_m and the volume change:
//
//   W = sum_m e_m (1 + 2 dln K^ / d(k^2) k_m (x) k_m),  e_m = G |Q|^2 / 2,
//
// the sum over the full spectrum, every interior plane of the half spectrum
// counting twice. Its per-node share is F_i (x) r_i, the force index first,
// plus the rest of W shared evenly over the charged nodes.
//
// The charges spread in 64-bit fixed point, so that the mesh, and every
// output, does not depend on the order of the atomic additions. The sums over
// the modes and over the nodes of a frame add per-block float64 sums in a
// fixed order. The meshes and the transforms are float64; the stencils, which
// would otherwise be bound by the float64 throughput of the device, weigh and
// read them in float32, the precision of the outputs, each warp adding the
// sums of its lanes in float64.
//
// The nodes whose outputs a call evaluates, its owners, are either every node
// or, across the domains of a decomposed system, the charged nodes of one MPI
// rank, whose frame then holds the charged nodes of every rank. Every rank
// spreads every charge and transforms the whole mesh, and interpolates only at
// its owners; the per-frame sums of the owners are added over the domains.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "charge_sum.h"
#include "frame_sums.cuh"

namespace {

constexpr int kThreads = 256;
constexpr int kMaxChannels = 8;
constexpr int kMaxOrder = 12;

enum KernelKind : int { kLes = 0, kSog = 1 };

// Resolution of the fixed-point charge mesh: 2^-40 e, far below the float32
// precision of the raw charges, for a sum of at most 2^23 e at a point.
constexpr double kFixedScale = 1099511627776.0;  // 2^40
constexpr float kFixedScaleFloat = 1099511627776.f;

constexpr double kPi = 3.14159265358979323846;

// The cell of one frame and its mesh.
struct FrameMesh {
  // B^-1, row-major: s_d = sum_j r_j inverse[3 j + d].
  double inverse[9];
  double volume;
  int n[3];
  // The first point of the meshes of the frame, channel-major, in the mesh
  // buffer of the call.
  int64_t offset;
};

// The squared lengths of the lattice vectors are formed term by term, as the
// Python front end forms them, so that both round the mesh size identically.
int64_t even_fft_size(int64_t n) {
  int64_t m = std::max<int64_t>(2, n + n % 2);
  while (true) {
    int64_t rest = m;
    for (const int64_t p : {2, 3, 5, 7}) {
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

FrameMesh frame_mesh(const double* cell, double spacing) {
  FrameMesh mesh{};
  const double a = cell[0], b = cell[1], c = cell[2];
  const double d = cell[3], e = cell[4], f = cell[5];
  const double g = cell[6], h = cell[7], i = cell[8];
  const double det =
      a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  // A volume that is a vanishing fraction of the product of the lengths of
  // the lattice vectors, or not finite, spans no lattice.
  const double lengths = std::sqrt(a * a + b * b + c * c) *
                         std::sqrt(d * d + e * e + f * f) *
                         std::sqrt(g * g + h * h + i * i);
  TORCH_CHECK(std::abs(det) > 1e-6 * lengths,
              "latent_charge_sum: a periodic cell is singular");
  const double adjugate[9] = {e * i - f * h, c * h - b * i, b * f - c * e,
                              f * g - d * i, a * i - c * g, c * d - a * f,
                              d * h - e * g, b * g - a * h, a * e - b * d};
  for (int k = 0; k < 9; ++k) {
    mesh.inverse[k] = adjugate[k] / det;
  }
  mesh.volume = std::abs(det);
  for (int axis = 0; axis < 3; ++axis) {
    const double* v = cell + 3 * axis;
    const double length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    const double points = std::ceil(length / spacing);
    TORCH_CHECK(points < 1 << 20,
                "latent_charge_sum: the mesh of a cell exceeds 2^20 points "
                "along a lattice vector");
    mesh.n[axis] =
        static_cast<int>(even_fft_size(static_cast<int64_t>(points)));
  }
  return mesh;
}

__host__ __device__ int64_t mesh_points(const FrameMesh& mesh) {
  return int64_t{mesh.n[0]} * mesh.n[1] * mesh.n[2];
}

// Nodes per block of the kernels that spread and interpolate, one warp each.
constexpr int kStencilWarps = 4;
constexpr int kWarp = 32;

// The stencil of one node along one axis of its mesh: the index of the
// first of its p points on the axis, which the others follow cyclically, and
// their B-spline weights and slopes, evaluated in float64 and stored in
// float32, the precision of the outputs: rounding a recursion of float32 steps
// instead would bias the sum of the weights.
template <int kOrder>
struct AxisStencil {
  int start;
  float value[kOrder];
  float slope[kOrder];
};

// The index of the point ``start + k`` of an axis of ``n`` points, for
// ``start`` < n and k < p, which exceeds 2 n only when a stencil wider than
// the axis wraps more than once.
__device__ __forceinline__ int wrap_index(int index, int n) {
  if (index < n) {
    return index;
  }
  return index - n < n ? index - n : index % n;
}

// The stencil of a node along axis ``d``. The scaled coordinate u = n_d s_d
// of the fractional coordinate s_d reaches the points floor(u) - p + 1 + k,
// k = 0, ..., p - 1, with the weights value[k] = M_p(t + p - 1 - k), t the
// fractional part of u, and slope[k] their derivative with respect to u. The
// recursion M_n(t + j) = ((t + j) M_{n-1}(t + j) + (n - t - j) M_{n-1}(t + j -
// 1)) / (n - 1) builds a_j = M_n(t + j) from M_2; the slope follows from
// M_p'(x) = M_{p-1}(x) - M_{p-1}(x - 1).
template <int kOrder>
__device__ void axis_stencil(const double* coord,
                             const FrameMesh& mesh,
                             int d,
                             AxisStencil<kOrder>& stencil) {
  double s = coord[0] * mesh.inverse[d] + coord[1] * mesh.inverse[3 + d] +
             coord[2] * mesh.inverse[6 + d];
  s -= floor(s);
  const double u = s * mesh.n[d];
  const double base = floor(u);
  const double t = u - base;
  double a[kOrder];
  a[0] = t;
  a[1] = 1.0 - t;
#pragma unroll
  for (int n = 3; n <= kOrder; ++n) {
    if (n == kOrder) {
      // The weights of order p - 1 give the slopes.
#pragma unroll
      for (int j = 0; j < kOrder; ++j) {
        const double high = j < kOrder - 1 ? a[j] : 0.0;
        const double low = j > 0 ? a[j - 1] : 0.0;
        stencil.slope[kOrder - 1 - j] = static_cast<float>(high - low);
      }
    }
    const double scale = 1.0 / (n - 1);
    a[n - 1] = scale * (1.0 - t) * a[n - 2];
#pragma unroll
    for (int j = n - 2; j > 0; --j) {
      a[j] = scale * ((t + j) * a[j] + (n - t - j) * a[j - 1]);
    }
    a[0] = scale * t * a[0];
  }
  if constexpr (kOrder == 2) {
    stencil.slope[0] = -1.f;
    stencil.slope[1] = 1.f;
  }
  const int first = static_cast<int>(base) - kOrder + 1;
  stencil.start = ((first % mesh.n[d]) + mesh.n[d]) % mesh.n[d];
#pragma unroll
  for (int k = 0; k < kOrder; ++k) {
    stencil.value[k] = static_cast<float>(a[kOrder - 1 - k]);
  }
}

// Writes the stencils of the charged nodes along the three axes of their
// meshes, one thread per node and axis, which the kernels that spread and
// interpolate read: every warp instruction in float64 occupies the float64
// pipe of the device whatever the number of its active lanes, so the
// recursion runs once per node and axis with every lane busy.
template <int kOrder>
__global__ void __launch_bounds__(kThreads)
    node_stencils(const double* __restrict__ coord,
                  const bool* __restrict__ node_mask,
                  const int64_t* __restrict__ frame_ptr,
                  const FrameMesh* __restrict__ frames,
                  int frame_count,
                  int64_t node_count,
                  AxisStencil<kOrder>* __restrict__ stencils) {
  for (int64_t index = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       index < 3 * node_count; index += int64_t{gridDim.x} * blockDim.x) {
    const int64_t node = index / 3;
    const int f = frame_of(frame_ptr, frame_count, node);
    if (f >= 0 && node_mask[node]) {
      axis_stencil<kOrder>(coord + 3 * node, frames[f],
                           static_cast<int>(index - 3 * node), stencils[index]);
    }
  }
}

// The point ``t`` of the p^3 points of a stencil: its offset on the flat mesh
// of a frame and its indices (k1, k2, k3) along the axes. The index of the
// last axis runs fastest, so that the lanes of a warp touch contiguous runs
// of the mesh.
template <int kOrder>
struct StencilPoint {
  int64_t offset;
  int k1;
  int k2;
  int k3;

  __device__ StencilPoint(int t,
                          const FrameMesh& mesh,
                          const AxisStencil<kOrder>* axes)
      : k1(t / (kOrder * kOrder)), k2((t / kOrder) % kOrder), k3(t % kOrder) {
    offset = (int64_t{wrap_index(axes[0].start + k1, mesh.n[0])} * mesh.n[1] +
              wrap_index(axes[1].start + k2, mesh.n[1])) *
                 mesh.n[2] +
             wrap_index(axes[2].start + k3, mesh.n[2]);
  }
};

// Writes the charges of every node, shifted in the constrained channel and
// zero on the nodes that carry none, and spreads them to the meshes of their
// frames, one warp per node, the lanes taking the points of its stencil in
// turn.
template <int kOrder>
__global__ void __launch_bounds__(kStencilWarps* kWarp)
    spread_charges(const float* __restrict__ charge,
                   const bool* __restrict__ node_mask,
                   const int64_t* __restrict__ frame_ptr,
                   const FrameMesh* __restrict__ frames,
                   const AxisStencil<kOrder>* __restrict__ stencils,
                   const double* __restrict__ frame_stat,
                   int frame_count,
                   int64_t node_count,
                   int channels,
                   int channel,
                   float* __restrict__ node_charge,
                   unsigned long long* __restrict__ mesh) {
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  for (int64_t node = int64_t{blockIdx.x} * kStencilWarps + warp;
       node < node_count; node += int64_t{gridDim.x} * kStencilWarps) {
    const int f = frame_of(frame_ptr, frame_count, node);
    const bool charged = f >= 0 && node_mask[node];
    const double shift =
        charged && channel >= 0 ? frame_stat[f * kStatWidth + kStatShift] : 0.0;
    float q[kMaxChannels];
#pragma unroll
    for (int c = 0; c < kMaxChannels; ++c) {
      const double raw =
          charged && c < channels ? charge[node * channels + c] : 0.0;
      q[c] = static_cast<float>(c == channel ? raw - shift : raw);
    }
    if (lane < channels) {
      node_charge[node * channels + lane] = charged ? q[lane] : 0.f;
    }
    if (!charged) {
      continue;
    }
    const FrameMesh grid = frames[f];
    const AxisStencil<kOrder>* axes = stencils + 3 * node;
    const int64_t points = mesh_points(grid);
    unsigned long long* frame_mesh = mesh + grid.offset;
    for (int t = lane; t < kOrder * kOrder * kOrder; t += kWarp) {
      const StencilPoint<kOrder> point(t, grid, axes);
      const float w = axes[0].value[point.k1] * axes[1].value[point.k2] *
                      axes[2].value[point.k3] * kFixedScaleFloat;
#pragma unroll
      for (int c = 0; c < kMaxChannels; ++c) {
        if (c < channels) {
          atomicAdd(frame_mesh + c * points + point.offset,
                    static_cast<unsigned long long>(__float2ll_rn(q[c] * w)));
        }
      }
    }
  }
}

// Converts the fixed-point mesh to float64 in place.
__global__ void fixed_to_double(int64_t count, unsigned long long* mesh) {
  for (int64_t index = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       index < count; index += int64_t{gridDim.x} * blockDim.x) {
    const long long fixed = static_cast<long long>(mesh[index]);
    reinterpret_cast<double*>(mesh)[index] =
        static_cast<double>(fixed) / kFixedScale;
  }
}

// The inverse squared assignment factor |D_d(m)|^-2 along every axis of the
// half spectrum of one frame: n1 entries, then n2, then n3 / 2 + 1.
__global__ void assignment_factor(FrameMesh mesh,
                                  int order,
                                  const double* __restrict__ integer_weight,
                                  double* __restrict__ factor) {
  const int half = mesh.n[2] / 2 + 1;
  const int total = mesh.n[0] + mesh.n[1] + half;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += gridDim.x * blockDim.x) {
    int axis = 0;
    int m = index;
    if (m >= mesh.n[0]) {
      m -= mesh.n[0];
      axis = 1;
      if (m >= mesh.n[1]) {
        m -= mesh.n[1];
        axis = 2;
      }
    }
    const double theta = 2.0 * kPi * m / mesh.n[axis];
    double re = 0.0;
    double im = 0.0;
    for (int l = 0; l < order - 1; ++l) {
      double s;
      double c;
      sincos(theta * l, &s, &c);
      re += integer_weight[l] * c;
      im += integer_weight[l] * s;
    }
    factor[index] = 1.0 / (re * re + im * im);
  }
}

// The transform of the kernel and its logarithmic derivative with respect to
// k^2. ``coefficient`` holds alpha, or the amplitudes and then the rates
// c_g = log2(e) / (2 sigma_g^2) of the Gaussians.
__device__ void kernel_transform(int kernel,
                                 const double* __restrict__ coefficient,
                                 int gaussians,
                                 double k_sq,
                                 double& value,
                                 double& log_slope) {
  if (kernel == kLes) {
    const double alpha = coefficient[0];
    const double inv_four_alpha_sq = 0.25 / (alpha * alpha);
    value = 4.0 * kPi * exp(-k_sq * inv_four_alpha_sq) / k_sq;
    log_slope = -inv_four_alpha_sq - 1.0 / k_sq;
    return;
  }
  // (2 pi)^{3/2}
  constexpr double kGaussianNorm = 15.749609945722419;
  value = 0.0;
  double slope = 0.0;
  for (int g = 0; g < gaussians; ++g) {
    // sigma^2 = log2(e) / (2 c) = 1 / (2 c ln 2)
    const double width_sq = 0.72134752044448170 / coefficient[gaussians + g];
    const double term = coefficient[g] * kGaussianNorm * width_sq *
                        sqrt(width_sq) * exp(-0.5 * width_sq * k_sq);
    value += term;
    slope -= 0.5 * width_sq * term;
  }
  log_slope = value != 0.0 ? slope / value : 0.0;
}

// The influence function of one frame on its half spectrum: multiplies every
// mode of the spectra of the channels by G in place and sums, per block, the
// energy of the modes and their virial, xx yy zz xy xz yz.
constexpr int kModeSums = 7;

__global__ void __launch_bounds__(kThreads)
    apply_influence(FrameMesh mesh,
                    int channels,
                    int kernel,
                    const double* __restrict__ coefficient,
                    int gaussians,
                    double prefactor,
                    const double* __restrict__ factor,
                    double2* __restrict__ spectrum,
                    double* __restrict__ block_sum) {
  const int n1 = mesh.n[0];
  const int n2 = mesh.n[1];
  const int n3 = mesh.n[2];
  const int half = n3 / 2 + 1;
  const int64_t modes = int64_t{n1} * n2 * half;
  double sum[kModeSums] = {};
  for (int64_t index = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       index < modes; index += int64_t{gridDim.x} * blockDim.x) {
    const int i3 = static_cast<int>(index % half);
    const int i2 = static_cast<int>((index / half) % n2);
    const int i1 = static_cast<int>(index / (int64_t{half} * n2));
    const int m1 = i1 <= n1 / 2 ? i1 : i1 - n1;
    const int m2 = i2 <= n2 / 2 ? i2 : i2 - n2;
    const int m3 = i3;
    const bool weightless = (m1 == 0 && m2 == 0 && m3 == 0) || 2 * m1 == n1 ||
                            2 * m2 == n2 || 2 * m3 == n3;
    double influence = 0.0;
    double k[3] = {0.0, 0.0, 0.0};
    double log_slope = 0.0;
    if (!weightless) {
      for (int a = 0; a < 3; ++a) {
        k[a] = 2.0 * kPi *
               (mesh.inverse[3 * a] * m1 + mesh.inverse[3 * a + 1] * m2 +
                mesh.inverse[3 * a + 2] * m3);
      }
      const double k_sq = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
      double transform;
      kernel_transform(kernel, coefficient, gaussians, k_sq, transform,
                       log_slope);
      influence = prefactor * transform / mesh.volume * factor[i1] *
                  factor[n1 + i2] * factor[n1 + n2 + i3];
    }
    // An interior plane of the half spectrum stands for its mirror as well.
    const double multiplicity = m3 == 0 ? 1.0 : 2.0;
    double energy = 0.0;
    for (int c = 0; c < channels; ++c) {
      double2& value = spectrum[c * modes + index];
      energy += value.x * value.x + value.y * value.y;
      value.x *= influence;
      value.y *= influence;
    }
    energy *= 0.5 * multiplicity * influence;
    const double twice = 2.0 * log_slope * energy;
    sum[0] += energy;
    sum[1] += energy + twice * k[0] * k[0];
    sum[2] += energy + twice * k[1] * k[1];
    sum[3] += energy + twice * k[2] * k[2];
    sum[4] += twice * k[0] * k[1];
    sum[5] += twice * k[0] * k[2];
    sum[6] += twice * k[1] * k[2];
  }
  __shared__ double partial[kModeSums][kThreads];
  for (int s = 0; s < kModeSums; ++s) {
    partial[s][threadIdx.x] = sum[s];
  }
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      for (int s = 0; s < kModeSums; ++s) {
        partial[s][threadIdx.x] += partial[s][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < kModeSums) {
    block_sum[blockIdx.x * kModeSums + threadIdx.x] = partial[threadIdx.x][0];
  }
}

// Adds the block sums of apply_influence into the virial of the frame,
// row-major: every thread sums a fixed strided subset of the blocks, and the
// block adds the thread sums in a fixed order.
__global__ void __launch_bounds__(kThreads)
    mode_virial(const double* __restrict__ block_sum,
                int block_count,
                double* __restrict__ virial) {
  double value[kModeSums] = {};
  for (int block = threadIdx.x; block < block_count; block += kThreads) {
#pragma unroll
    for (int s = 0; s < kModeSums; ++s) {
      value[s] += block_sum[block * kModeSums + s];
    }
  }
  __shared__ double partial[kModeSums][kThreads];
#pragma unroll
  for (int s = 0; s < kModeSums; ++s) {
    partial[s][threadIdx.x] = value[s];
  }
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int s = 0; s < kModeSums; ++s) {
        partial[s][threadIdx.x] += partial[s][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  // Components 1..6 are xx yy zz xy xz yz; the energy is not stored.
  constexpr int kEntry[kModeSums][2] = {{-1, -1}, {0, 0}, {4, 4}, {8, 8},
                                        {1, 3},   {2, 6}, {5, 7}};
  if (threadIdx.x > 0 && threadIdx.x < kModeSums) {
    virial[kEntry[threadIdx.x][0]] = partial[threadIdx.x][0];
    virial[kEntry[threadIdx.x][1]] = partial[threadIdx.x][0];
  }
}

// The contact value K(0) of the kernel.
__device__ double contact_value(int kernel,
                                const double* __restrict__ coefficient,
                                int gaussians) {
  if (kernel == kLes) {
    return 1.1283791670955126 * coefficient[0];
  }
  double value = 0.0;
  for (int g = 0; g < gaussians; ++g) {
    value += coefficient[g];
  }
  return value;
}

// Adds ``value`` over the lanes of a warp in a fixed butterfly order; every
// lane receives the total.
__device__ __forceinline__ double warp_sum(double value) {
#pragma unroll
  for (int stride = kWarp / 2; stride > 0; stride /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, stride);
  }
  return value;
}

// Interpolates the potential mesh at every owner node, one warp per node:
// its potential, less that of its charge at itself, its force at fixed
// charges and the share F (x) r of its virial. Owners that carry no charge get
// zeros. The lanes take the points of the stencil in turn and the warp adds
// their sums.
template <int kOrder>
__global__ void __launch_bounds__(kStencilWarps* kWarp)
    interpolate_owners(const double* __restrict__ coord,
                       const float* __restrict__ node_charge,
                       const bool* __restrict__ node_mask,
                       const int64_t* __restrict__ frame_ptr,
                       const FrameMesh* __restrict__ frames,
                       const AxisStencil<kOrder>* __restrict__ stencils,
                       const float* __restrict__ potential_mesh,
                       const double* __restrict__ coefficient,
                       int frame_count,
                       int owner_begin,
                       int owner_count,
                       int channels,
                       int kernel,
                       int gaussians,
                       double prefactor,
                       double* __restrict__ potential,
                       double* __restrict__ force,
                       double* __restrict__ moment,
                       double* __restrict__ atom_energy) {
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const double self = prefactor * contact_value(kernel, coefficient, gaussians);
  for (int owner = blockIdx.x * kStencilWarps + warp; owner < owner_count;
       owner += gridDim.x * kStencilWarps) {
    const int64_t node = int64_t{owner_begin} + owner;
    const int f = frame_of(frame_ptr, frame_count, node);
    const bool charged = f >= 0 && node_mask[node];
    const float* q = node_charge + node * channels;
    double phi[kMaxChannels] = {};
    double f_node[3] = {0.0, 0.0, 0.0};
    if (charged) {
      const FrameMesh grid = frames[f];
      const AxisStencil<kOrder>* axes = stencils + 3 * node;
      const int64_t points = mesh_points(grid);
      const float* frame_mesh = potential_mesh + grid.offset;
      // Every lane sums its points in float32, the warp the lane sums in
      // float64. ``gradient`` is the derivative of sum_c q_c phi_c with
      // respect to u_d.
      float lane_phi[kMaxChannels] = {};
      float lane_gradient[3] = {0.f, 0.f, 0.f};
      for (int t = lane; t < kOrder * kOrder * kOrder; t += kWarp) {
        const StencilPoint<kOrder> point(t, grid, axes);
        const float v1 = axes[0].value[point.k1];
        const float v2 = axes[1].value[point.k2];
        const float v3 = axes[2].value[point.k3];
        float charge_potential = 0.f;
#pragma unroll
        for (int c = 0; c < kMaxChannels; ++c) {
          if (c < channels) {
            const float value = frame_mesh[c * points + point.offset];
            lane_phi[c] += v1 * v2 * v3 * value;
            charge_potential += q[c] * value;
          }
        }
        lane_gradient[0] +=
            axes[0].slope[point.k1] * v2 * v3 * charge_potential;
        lane_gradient[1] +=
            v1 * axes[1].slope[point.k2] * v3 * charge_potential;
        lane_gradient[2] +=
            v1 * v2 * axes[2].slope[point.k3] * charge_potential;
      }
#pragma unroll
      for (int c = 0; c < kMaxChannels; ++c) {
        if (c < channels) {
          phi[c] = warp_sum(lane_phi[c]) - self * q[c];
        }
      }
      double gradient[3];
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        gradient[d] = warp_sum(lane_gradient[d]) * grid.n[d];
      }
      // d E / d r_j = sum_d d E / d u_d n_d (B^-1)_jd
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        f_node[j] = -(gradient[0] * grid.inverse[3 * j] +
                      gradient[1] * grid.inverse[3 * j + 1] +
                      gradient[2] * grid.inverse[3 * j + 2]);
      }
    }
    if (lane == 0) {
      double energy = 0.0;
      for (int c = 0; c < channels; ++c) {
        potential[int64_t{owner} * channels + c] = phi[c];
        energy += q[c] * phi[c];
      }
      atom_energy[owner] = 0.5 * energy;
      const double* r = coord + 3 * node;
      for (int a = 0; a < 3; ++a) {
        force[int64_t{owner} * 3 + a] = f_node[a];
        for (int b = 0; b < 3; ++b) {
          moment[int64_t{owner} * 9 + 3 * a + b] = f_node[a] * r[b];
        }
      }
    }
  }
}

// Per-frame sums over one slice of the owners: the potential of the charged
// owners in the constrained channel, the number of charged owners and the
// nine components of ``moment``.
constexpr int kOwnerSums = 11;

__global__ void __launch_bounds__(kFrameThreads)
    owner_slices(const double* __restrict__ potential,
                 const double* __restrict__ moment,
                 const bool* __restrict__ owner_mask,
                 const int64_t* __restrict__ owner_frame_ptr,
                 int channels,
                 int channel,
                 double* __restrict__ slice_sum) {
  const NodeRange slice = slice_of_frame(owner_frame_ptr);
  double value[kOwnerSums] = {};
  for (int64_t owner = slice.begin + threadIdx.x; owner < slice.end;
       owner += kFrameThreads) {
    if (!owner_mask[owner]) {
      continue;
    }
    if (channel >= 0) {
      value[0] += potential[owner * channels + channel];
    }
    value[1] += 1.0;
    for (int k = 0; k < 9; ++k) {
      value[2 + k] += moment[owner * 9 + k];
    }
  }
  write_slice_sum<kOwnerSums>(value, slice_sum);
}

// Adds the slice sums of owner_slices of every frame in slice order.
__global__ void owner_totals(const double* __restrict__ slice_sum,
                             int slice_count,
                             double* __restrict__ total) {
  if (threadIdx.x >= kOwnerSums) {
    return;
  }
  double value = 0.0;
  for (int slice = 0; slice < slice_count; ++slice) {
    value +=
        slice_sum[(static_cast<int64_t>(blockIdx.x) * slice_count + slice) *
                      kOwnerSums +
                  threadIdx.x];
  }
  total[blockIdx.x * kOwnerSums + threadIdx.x] = value;
}

// Writes the outputs of every owner: the charge gradient w (phi - [c = ch]
// mean), the force, and the virial F (x) r plus the share of the rest of the
// virial of the frame over its charged nodes, then the owner sums of that
// virial per frame.
__global__ void finish_owners(const double* __restrict__ potential,
                              const double* __restrict__ force,
                              const double* __restrict__ moment,
                              const bool* __restrict__ owner_mask,
                              const int64_t* __restrict__ owner_frame_ptr,
                              const double* __restrict__ frame_virial,
                              const double* __restrict__ owner_total,
                              const double* __restrict__ frame_stat,
                              int frame_count,
                              int owner_count,
                              int channels,
                              int channel,
                              float* __restrict__ charge_gradient,
                              float* __restrict__ force_out,
                              float* __restrict__ atom_virial) {
  for (int owner = blockIdx.x * blockDim.x + threadIdx.x; owner < owner_count;
       owner += gridDim.x * blockDim.x) {
    const int f = frame_of(owner_frame_ptr, frame_count, owner);
    const bool charged = f >= 0 && owner_mask[owner];
    // The charged nodes of the frame, of every domain.
    const double count =
        charged ? frame_stat[f * kStatWidth + kStatCount] : 1.0;
    const double mean =
        charged && channel >= 0 ? owner_total[f * kOwnerSums] / count : 0.0;
    for (int c = 0; c < channels; ++c) {
      const double phi =
          charged ? potential[int64_t{owner} * channels + c] : 0.0;
      charge_gradient[int64_t{owner} * channels + c] =
          static_cast<float>(c == channel ? phi - mean : phi);
    }
    for (int a = 0; a < 3; ++a) {
      force_out[int64_t{owner} * 3 + a] =
          static_cast<float>(force[int64_t{owner} * 3 + a]);
    }
    for (int k = 0; k < 9; ++k) {
      const double rest = charged ? (frame_virial[f * 9 + k] -
                                     owner_total[f * kOwnerSums + 2 + k]) /
                                        count
                                  : 0.0;
      atom_virial[int64_t{owner} * 9 + k] =
          static_cast<float>(moment[int64_t{owner} * 9 + k] + rest);
    }
  }
}

// Sums, over one slice of the owners of one frame, their virial.
__global__ void __launch_bounds__(kFrameThreads)
    virial_slices(const float* __restrict__ atom_virial,
                  const int64_t* __restrict__ owner_frame_ptr,
                  double* __restrict__ slice_sum) {
  const NodeRange slice = slice_of_frame(owner_frame_ptr);
  double value[9] = {};
  for (int64_t owner = slice.begin + threadIdx.x; owner < slice.end;
       owner += kFrameThreads) {
    for (int k = 0; k < 9; ++k) {
      value[k] += atom_virial[owner * 9 + k];
    }
  }
  write_slice_sum<9>(value, slice_sum);
}

__global__ void virial_total(const double* __restrict__ slice_sum,
                             int slice_count,
                             float* __restrict__ virial) {
  if (threadIdx.x >= 9) {
    return;
  }
  double value = 0.0;
  for (int slice = 0; slice < slice_count; ++slice) {
    value +=
        slice_sum[(static_cast<int64_t>(blockIdx.x) * slice_count + slice) * 9 +
                  threadIdx.x];
  }
  virial[blockIdx.x * 9 + threadIdx.x] = static_cast<float>(value);
}

// The weights M_p(1), ..., M_p(p - 1) of the assignment factor.
std::vector<double> integer_bspline_weights(int order) {
  std::vector<double> weight(order - 1);
  for (int l = 0; l < order - 1; ++l) {
    const double x = l + 1.0;
    double value = 0.0;
    double binomial = 1.0;
    for (int j = 0; j <= order; ++j) {
      if (j > 0) {
        binomial = binomial * (order - j + 1) / j;
      }
      const double base = std::max(x - j, 0.0);
      value += (j % 2 == 0 ? 1.0 : -1.0) * binomial * std::pow(base, order - 1);
    }
    double factorial = 1.0;
    for (int k = 2; k < order; ++k) {
      factorial *= k;
    }
    weight[l] = value / factorial;
  }
  return weight;
}

int blocks_for(int64_t work) {
  return static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>((work + kThreads - 1) / kThreads, 65535)));
}

// Blocks of a stencil kernel over ``nodes`` nodes, one warp each.
int stencil_blocks_for(int64_t nodes) {
  return static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>((nodes + kStencilWarps - 1) / kStencilWarps,
                           int64_t{1} << 20)));
}

// Calls ``launch`` with the B-spline order as a compile-time constant, for
// which the stencil kernels unroll their loops.
template <typename Launch>
void dispatch_order(int64_t order, const Launch& launch) {
  switch (order) {
    case 2:
      return launch(std::integral_constant<int, 2>{});
    case 3:
      return launch(std::integral_constant<int, 3>{});
    case 4:
      return launch(std::integral_constant<int, 4>{});
    case 5:
      return launch(std::integral_constant<int, 5>{});
    case 6:
      return launch(std::integral_constant<int, 6>{});
    case 7:
      return launch(std::integral_constant<int, 7>{});
    case 8:
      return launch(std::integral_constant<int, 8>{});
    case 9:
      return launch(std::integral_constant<int, 9>{});
    case 10:
      return launch(std::integral_constant<int, 10>{});
    case 11:
      return launch(std::integral_constant<int, 11>{});
    case 12:
      return launch(std::integral_constant<int, 12>{});
    default:
      TORCH_CHECK(false, "latent_charge_sum: unsupported B-spline order ",
                  order);
  }
}

}  // namespace

namespace deepmd {
namespace long_range {

ChargeSumOutputs mesh_sum(const torch::Tensor& coord,
                          const torch::Tensor& charge,
                          const torch::Tensor& node_mask,
                          const torch::Tensor& n_node_per_frame,
                          const torch::Tensor& total_charge,
                          const torch::Tensor& box,
                          const torch::Tensor& coefficient,
                          int64_t kernel,
                          int64_t channel,
                          double prefactor,
                          int64_t order,
                          double spacing,
                          int owner_begin,
                          int owner_count,
                          const torch::Tensor* communicator) {
  TORCH_CHECK(order >= 2 && order <= kMaxOrder,
              "latent_charge_sum: the B-spline order must lie in [2, ",
              kMaxOrder, "], got ", order);
  TORCH_CHECK(std::isfinite(spacing) && spacing > 0.0,
              "latent_charge_sum: the mesh spacing must be positive, got ",
              spacing);
  const int64_t node_count = coord.size(0);
  TORCH_CHECK(node_count < std::numeric_limits<int>::max(),
              "latent_charge_sum: too many nodes");
  const int frame_count = static_cast<int>(n_node_per_frame.size(0));
  const int channels = static_cast<int>(charge.size(1));
  const int gaussians =
      kernel == kSog ? static_cast<int>(coefficient.numel() / 2) : 0;
  const auto double_options = coord.options();
  const auto float_options = charge.options();
  if (node_count == 0 || frame_count == 0) {
    return {torch::zeros({owner_count, 1}, double_options),
            torch::zeros({owner_count, channels}, float_options),
            torch::zeros({owner_count, 3}, float_options),
            torch::zeros({owner_count, 9}, float_options),
            torch::zeros({frame_count, 9}, float_options)};
  }
  const bool every_node = owner_begin == 0 && owner_count == node_count;
  TORCH_CHECK(every_node || frame_count == 1,
              "latent_charge_sum: owners other than every node lie in one "
              "frame");
  const auto stream = at::cuda::getCurrentCUDAStream();

  // === Step 1. The meshes of the frames, sized on the host ===
  std::vector<FrameMesh> frames(frame_count);
  const double* cells = box.data_ptr<double>();
  int64_t mesh_total = 0;
  for (int f = 0; f < frame_count; ++f) {
    frames[f] = frame_mesh(cells + 9 * f, spacing);
    frames[f].offset = mesh_total;
    mesh_total += channels * mesh_points(frames[f]);
  }
  const auto byte_options = torch::TensorOptions().dtype(torch::kUInt8);
  auto host_frames =
      torch::empty({static_cast<int64_t>(sizeof(FrameMesh)) * frame_count},
                   byte_options.pinned_memory(true));
  std::memcpy(host_frames.data_ptr(), frames.data(),
              sizeof(FrameMesh) * frame_count);
  const auto device_frames =
      host_frames.to(coord.device(), /*non_blocking=*/true);
  const auto* frames_ptr =
      reinterpret_cast<const FrameMesh*>(device_frames.data_ptr());
  const std::vector<double> weight_host = integer_bspline_weights(order);
  auto host_weight = torch::from_blob(const_cast<double*>(weight_host.data()),
                                      {order - 1}, torch::kFloat64)
                         .pin_memory();
  const auto integer_weight =
      host_weight.to(coord.device(), /*non_blocking=*/true);

  // === Step 2. The constraint and the charged nodes of every frame ===
  const auto counts = n_node_per_frame.to(torch::kInt64).contiguous();
  auto frame_ptr = torch::empty({frame_count + 1}, counts.options());
  frame_offsets<<<1, kOffsetThreads, 0, stream>>>(
      counts.data_ptr<int64_t>(), frame_count, frame_ptr.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t average_nodes = (node_count + frame_count - 1) / frame_count;
  const auto slices_for = [](int64_t nodes) {
    return static_cast<int>(std::min<int64_t>(
        kMaxSlices,
        std::max<int64_t>(1, (nodes + kSliceNodes - 1) / kSliceNodes)));
  };
  const int slice_count = slices_for(average_nodes);
  auto slice_charge = torch::empty(
      {static_cast<int64_t>(frame_count) * slice_count * 2}, double_options);
  auto frame_stat = torch::empty({frame_count * kStatWidth}, double_options);
  charge_slices<<<dim3(frame_count, slice_count), kFrameThreads, 0, stream>>>(
      charge.data_ptr<float>(), node_mask.data_ptr<bool>(),
      frame_ptr.data_ptr<int64_t>(), channels,
      static_cast<int>(std::max<int64_t>(channel, 0)),
      slice_charge.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  charge_shift<<<(frame_count + kFrameThreads - 1) / kFrameThreads,
                 kFrameThreads, 0, stream>>>(
      slice_charge.data_ptr<double>(), slice_count, frame_count,
      total_charge.data_ptr<double>(), frame_stat.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // === Step 3. The stencils of the nodes and the charge meshes ===
  auto mesh = torch::zeros({mesh_total}, double_options.dtype(torch::kInt64));
  auto node_charge = torch::empty({node_count, channels}, float_options);
  torch::Tensor stencils;
  dispatch_order(order, [&](auto order_constant) {
    constexpr int kOrder = decltype(order_constant)::value;
    stencils = torch::empty(
        {3 * node_count * static_cast<int64_t>(sizeof(AxisStencil<kOrder>))},
        byte_options.device(coord.device()));
    auto* node_stencil =
        reinterpret_cast<AxisStencil<kOrder>*>(stencils.data_ptr());
    node_stencils<kOrder><<<blocks_for(3 * node_count), kThreads, 0, stream>>>(
        coord.data_ptr<double>(), node_mask.data_ptr<bool>(),
        frame_ptr.data_ptr<int64_t>(), frames_ptr, frame_count, node_count,
        node_stencil);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    spread_charges<kOrder>
        <<<stencil_blocks_for(node_count), kStencilWarps * kWarp, 0, stream>>>(
            charge.data_ptr<float>(), node_mask.data_ptr<bool>(),
            frame_ptr.data_ptr<int64_t>(), frames_ptr, node_stencil,
            frame_stat.data_ptr<double>(), frame_count, node_count, channels,
            static_cast<int>(channel), node_charge.data_ptr<float>(),
            reinterpret_cast<unsigned long long*>(mesh.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  fixed_to_double<<<blocks_for(mesh_total), kThreads, 0, stream>>>(
      mesh_total,
      reinterpret_cast<unsigned long long*>(mesh.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const auto charge_mesh = mesh.view(torch::kFloat64);

  // === Step 4. Influence function and potential meshes, frame by frame ===
  auto potential_mesh = torch::empty({mesh_total}, double_options);
  auto frame_virial = torch::zeros({frame_count, 9}, double_options);
  for (int f = 0; f < frame_count; ++f) {
    const FrameMesh& grid = frames[f];
    const int64_t points = mesh_points(grid);
    const std::vector<int64_t> shape = {channels, grid.n[0], grid.n[1],
                                        grid.n[2]};
    auto spectrum = at::fft_rfftn(
        charge_mesh.narrow(0, grid.offset, channels * points).view(shape),
        c10::nullopt, std::vector<int64_t>{1, 2, 3});
    const int half = grid.n[2] / 2 + 1;
    auto factor = torch::empty({grid.n[0] + grid.n[1] + half}, double_options);
    assignment_factor<<<blocks_for(factor.numel()), kThreads, 0, stream>>>(
        grid, static_cast<int>(order), integer_weight.data_ptr<double>(),
        factor.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t modes = int64_t{grid.n[0]} * grid.n[1] * half;
    // A few blocks per multiprocessor fill the device; every thread then
    // walks several modes, which keeps the block sums few.
    const int mode_blocks = std::min(
        blocks_for(modes),
        4 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
    auto block_sum =
        torch::empty({int64_t{mode_blocks} * kModeSums}, double_options);
    apply_influence<<<mode_blocks, kThreads, 0, stream>>>(
        grid, channels, static_cast<int>(kernel),
        coefficient.data_ptr<double>(), gaussians, prefactor,
        factor.data_ptr<double>(),
        reinterpret_cast<double2*>(spectrum.data_ptr<c10::complex<double>>()),
        block_sum.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    mode_virial<<<1, kThreads, 0, stream>>>(block_sum.data_ptr<double>(),
                                            mode_blocks,
                                            frame_virial[f].data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto frame_potential =
        potential_mesh.narrow(0, grid.offset, channels * points).view(shape);
    at::fft_irfftn_out(frame_potential, spectrum,
                       std::vector<int64_t>{grid.n[0], grid.n[1], grid.n[2]},
                       std::vector<int64_t>{1, 2, 3}, "forward");
  }

  // === Step 5. Potential, force and virial of the owners ===
  // The stencils read the potential meshes in float32, the precision of the
  // outputs.
  const auto potential_mesh_float = potential_mesh.to(torch::kFloat32);
  auto potential = torch::empty({owner_count, channels}, double_options);
  auto force = torch::empty({owner_count, 3}, double_options);
  auto moment = torch::empty({owner_count, 9}, double_options);
  auto atom_energy = torch::empty({owner_count, 1}, double_options);
  if (owner_count > 0) {
    dispatch_order(order, [&](auto order_constant) {
      constexpr int kOrder = decltype(order_constant)::value;
      interpolate_owners<kOrder><<<stencil_blocks_for(owner_count),
                                   kStencilWarps * kWarp, 0, stream>>>(
          coord.data_ptr<double>(), node_charge.data_ptr<float>(),
          node_mask.data_ptr<bool>(), frame_ptr.data_ptr<int64_t>(), frames_ptr,
          reinterpret_cast<const AxisStencil<kOrder>*>(stencils.data_ptr()),
          potential_mesh_float.data_ptr<float>(),
          coefficient.data_ptr<double>(), frame_count, owner_begin, owner_count,
          channels, static_cast<int>(kernel), gaussians, prefactor,
          potential.data_ptr<double>(), force.data_ptr<double>(),
          moment.data_ptr<double>(), atom_energy.data_ptr<double>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // The frames of the owners, on the axis of the owners: the frames
  // themselves when every node is an owner, else the one frame they share.
  torch::Tensor owner_frame_ptr = frame_ptr;
  if (!every_node) {
    owner_frame_ptr =
        torch::full({2}, int64_t{owner_count}, frame_ptr.options());
    owner_frame_ptr.select(0, 0).zero_();
  }
  const bool* owner_mask = node_mask.data_ptr<bool>() + owner_begin;
  const int owner_slice_count = slices_for(
      (int64_t{owner_count} + frame_count - 1) / std::max(frame_count, 1));
  auto owner_slice = torch::empty(
      {int64_t{frame_count} * owner_slice_count * kOwnerSums}, double_options);
  owner_slices<<<dim3(frame_count, owner_slice_count), kFrameThreads, 0,
                 stream>>>(
      potential.data_ptr<double>(), moment.data_ptr<double>(), owner_mask,
      owner_frame_ptr.data_ptr<int64_t>(), channels, static_cast<int>(channel),
      owner_slice.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto owner_total =
      torch::empty({int64_t{frame_count} * kOwnerSums}, double_options);
  owner_totals<<<frame_count, 32, 0, stream>>>(owner_slice.data_ptr<double>(),
                                               owner_slice_count,
                                               owner_total.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (communicator != nullptr) {
    // The mean potential of the constraint and the virial share run over the
    // charged nodes of every domain.
    owner_total.copy_(sum_over_domains(*communicator, owner_total));
  }
  auto charge_gradient = torch::empty({owner_count, channels}, float_options);
  auto force_out = torch::empty({owner_count, 3}, float_options);
  auto atom_virial = torch::empty({owner_count, 9}, float_options);
  if (owner_count > 0) {
    finish_owners<<<blocks_for(owner_count), kThreads, 0, stream>>>(
        potential.data_ptr<double>(), force.data_ptr<double>(),
        moment.data_ptr<double>(), owner_mask,
        owner_frame_ptr.data_ptr<int64_t>(), frame_virial.data_ptr<double>(),
        owner_total.data_ptr<double>(), frame_stat.data_ptr<double>(),
        frame_count, owner_count, channels, static_cast<int>(channel),
        charge_gradient.data_ptr<float>(), force_out.data_ptr<float>(),
        atom_virial.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  auto virial_slice = torch::empty(
      {int64_t{frame_count} * owner_slice_count * 9}, double_options);
  virial_slices<<<dim3(frame_count, owner_slice_count), kFrameThreads, 0,
                  stream>>>(atom_virial.data_ptr<float>(),
                            owner_frame_ptr.data_ptr<int64_t>(),
                            virial_slice.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto virial = torch::empty({frame_count, 9}, float_options);
  virial_total<<<frame_count, 32, 0, stream>>>(virial_slice.data_ptr<double>(),
                                               owner_slice_count,
                                               virial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {atom_energy, charge_gradient, force_out, atom_virial, virial};
}

}  // namespace long_range
}  // namespace deepmd
