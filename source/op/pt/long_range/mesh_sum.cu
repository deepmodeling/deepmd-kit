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
// the modes and over the nodes of a frame add float64 sums in a fixed order.
// The meshes, the transforms and the influence function are float32, against
// a float64 throughput of a small fraction of the float32 one on many
// devices; the B-spline weights are evaluated in float64, whose rounding to
// float32 leaves their sum unbiased. The forces then carry the rounding of
// float32, and the energy of a frame that of the reciprocal sum including the
// interaction of every charge with itself, which the self term cancels: a few
// 1e-7 of the self energy.
//
// The nodes whose outputs a call evaluates, its owners, are either every node
// or, across the domains of a decomposed system, the charged nodes of one MPI
// rank, whose frame then holds the charged nodes of every rank. Every rank
// spreads every charge and transforms the whole mesh, and interpolates only at
// its owners; the per-frame sums of the owners are added over the domains.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include "charge_sum.h"
#include "frame_sums.cuh"

namespace {

constexpr int kThreads = 256;
constexpr int kMaxGaussians = 32;
constexpr int kMaxOrder = 12;

enum KernelKind : int { kLes = 0, kSog = 1 };

// Resolution of the fixed-point charge mesh: 2^-40 e, far below the float32
// precision of the raw charges, for a sum of at most 2^23 e at a point.
constexpr float kFixedScaleFloat = 1099511627776.f;  // 2^40

constexpr double kPi = 3.14159265358979323846;

// The cell of one frame and its mesh. The host sizes the mesh, which fixes
// the buffers and the launches of a call, and the device derives the
// geometry from the cell and the prefactor of the call, so that a CUDA graph
// capturing a call evaluates the cell of every replay on the meshes of the
// capture.
struct FrameMesh {
  // B^-1, row-major: s_d = sum_j r_j inverse[3 j + d].
  double inverse[9];
  // 2 pi B^-1 in float32, row-major like inverse: the wave vector of the mode
  // m is k_a = sum_d reciprocal[3 a + d] m_d.
  float reciprocal[9];
  // The factor prefactor / V of the influence function in float32, V the
  // volume of the cell.
  float influence_scale;
  // -n_d (B^-1)_jd, row-major in (j, d): the force at fixed charges from the
  // derivatives of the energy with respect to the scaled coordinates u_d.
  float gradient_to_force[9];
  int n[3];
  // The first point of the meshes of the frame, channel-major, in the mesh
  // buffer of the call.
  int64_t offset;
};

// The smallest even number of at least ``n`` without a prime factor above 7.
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

// The mesh of the cell ``cell`` at a spacing of at most ``spacing``, sized
// along every lattice vector; store_frame_mesh adds the geometry of the cell
// on the device.
FrameMesh frame_mesh_size(const double* cell, double spacing) {
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
  // The squared length of a lattice vector sums the squares of its
  // components, each rounded on its own as the Python front end and the C++
  // interface form them: the explicit fma of a square with zero is that
  // rounded square, which the compiler does not fuse into the sum, so that
  // all three size the mesh identically.
  for (int axis = 0; axis < 3; ++axis) {
    const double* v = cell + 3 * axis;
    const double length =
        std::sqrt(std::fma(v[0], v[0], 0.0) + std::fma(v[1], v[1], 0.0) +
                  std::fma(v[2], v[2], 0.0));
    const double points = std::ceil(length / spacing);
    TORCH_CHECK(points < 1 << 20,
                "latent_charge_sum: the mesh of a cell exceeds 2^20 points "
                "along a lattice vector");
    mesh.n[axis] =
        static_cast<int>(even_fft_size(static_cast<int64_t>(points)));
  }
  TORCH_CHECK(int64_t{mesh.n[0]} * mesh.n[1] * mesh.n[2] <
                  std::numeric_limits<int>::max(),
              "latent_charge_sum: the mesh of a cell exceeds 2^31 points");
  return mesh;
}

__host__ __device__ int64_t mesh_points(const FrameMesh& mesh) {
  return int64_t{mesh.n[0]} * mesh.n[1] * mesh.n[2];
}

// Nodes per block of the kernels that spread and interpolate, one warp each.
constexpr int kStencilWarps = 4;
constexpr int kWarp = 32;

// The stencil of one node along one axis of its mesh: for each of its p
// points on the axis, its offset on the flat mesh of a frame along that axis,
// the index of the point times the stride of the axis, and its B-spline
// weight and slope, evaluated in float64 and stored in float32, the precision
// of the outputs: rounding a recursion of float32 steps instead would bias
// the sum of the weights.
template <int kOrder>
struct AxisStencil {
  int offset[kOrder];
  float value[kOrder];
  float slope[kOrder];
};

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
  const int n = mesh.n[d];
  const int stride = d == 0 ? mesh.n[1] * mesh.n[2] : d == 1 ? mesh.n[2] : 1;
  // A stencil wider than the axis wraps more than once.
  const int first = ((static_cast<int>(base) - kOrder + 1) % n + n) % n;
#pragma unroll
  for (int k = 0; k < kOrder; ++k) {
    stencil.value[k] = static_cast<float>(a[kOrder - 1 - k]);
    stencil.offset[k] = (first + k) % n * stride;
  }
}

// Writes, one thread per node and axis, the stencils of the charged nodes
// along the three axes of their meshes, and, on the thread of the first axis,
// the charges of every node, shifted in the constrained channel and zero on
// the nodes that carry none. Every float64 warp instruction occupies the
// float64 pipe of the device whatever the number of its active lanes, so the
// float64 work of a node runs here with every lane busy rather than in the
// warps of the kernels that spread and interpolate.
template <int kOrder>
__global__ void __launch_bounds__(kThreads)
    node_stencils(const double* __restrict__ coord,
                  const float* __restrict__ charge,
                  const bool* __restrict__ node_mask,
                  const int64_t* __restrict__ frame_ptr,
                  const FrameMesh* __restrict__ frames,
                  const double* __restrict__ frame_stat,
                  int frame_count,
                  int64_t node_count,
                  int channels,
                  int channel,
                  AxisStencil<kOrder>* __restrict__ stencils,
                  float* __restrict__ node_charge) {
  for (int64_t index = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       index < 3 * node_count; index += int64_t{gridDim.x} * blockDim.x) {
    const int64_t node = index / 3;
    const int axis = static_cast<int>(index - 3 * node);
    const int f = frame_of(frame_ptr, frame_count, node);
    const bool charged = f >= 0 && node_mask[node];
    if (charged) {
      axis_stencil<kOrder>(coord + 3 * node, frames[f], axis, stencils[index]);
    }
    if (axis == 0) {
      const double shift = charged && channel >= 0
                               ? frame_stat[f * kStatWidth + kStatShift]
                               : 0.0;
      for (int c = 0; c < channels; ++c) {
        const double raw = charged ? charge[node * channels + c] : 0.0;
        node_charge[node * channels + c] =
            static_cast<float>(c == channel ? raw - shift : raw);
      }
    }
  }
}

// Rounds ``x`` to the nearest integer, ties to even, as __float2ll_rn does,
// in integer operations: the conversion instruction occupies the float64 pipe.
// A finite x is a 24-bit significand times 2^exponent; a negative exponent
// shifts the significand right with rounding. Magnitudes of 2^63 and beyond,
// infinities and NaN saturate.
__device__ __forceinline__ long long float_to_fixed(float x) {
  const unsigned bits = __float_as_uint(x);
  const int biased = static_cast<int>((bits >> 23) & 0xffu);
  // |x| < 1/2, zero and subnormal numbers included, rounds to zero.
  if (biased < 126) {
    return 0;
  }
  const int exponent = biased - 150;
  if (exponent >= 40) {
    return bits >> 31 ? std::numeric_limits<long long>::min()
                      : std::numeric_limits<long long>::max();
  }
  const long long significand = (bits & 0x7fffffu) | 0x800000u;
  long long magnitude;
  if (exponent >= 0) {
    magnitude = significand << exponent;
  } else {
    const int shift = -exponent;
    const long long kept = significand >> shift;
    const long long rest = significand & ((1ll << shift) - 1);
    const long long half = 1ll << (shift - 1);
    magnitude = kept + (rest > half || (rest == half && (kept & 1)) ? 1 : 0);
  }
  return bits >> 31 ? -magnitude : magnitude;
}

// The assignment of the p^3 points of a stencil to the lanes of a warp: the
// lanes take the (k2, k3) pairs, k3 running fastest, so that a warp touches
// contiguous runs of the mesh, and every lane walks the k1 axis for its
// pairs; while the pairs are fewer than the lanes, groups of lanes split the
// k1 axis.
template <int kOrder>
struct LaneSlots {
  static constexpr int kPairs = kOrder * kOrder;
  static constexpr int kGroups = kPairs >= kWarp ? 1 : kWarp / kPairs;
  static constexpr int kSlots = kGroups * kPairs;
  static constexpr int kRounds = (kSlots + kWarp - 1) / kWarp;
};

// Copies the stencils of a node along its three axes to the shared memory of
// its warp.
template <int kOrder>
__device__ __forceinline__ void load_stencil(
    const AxisStencil<kOrder>* __restrict__ stencils,
    int64_t node,
    int lane,
    AxisStencil<kOrder>* shared) {
  constexpr int kWords = 3 * sizeof(AxisStencil<kOrder>) / sizeof(int);
  const int* source = reinterpret_cast<const int*>(stencils + 3 * node);
  int* target = reinterpret_cast<int*>(shared);
  for (int word = lane; word < kWords; word += kWarp) {
    target[word] = source[word];
  }
  __syncwarp();
}

// Spreads the charges of the charged nodes to the meshes of their frames, one
// warp per node, one pass per channel.
template <int kOrder>
__global__ void __launch_bounds__(kStencilWarps* kWarp)
    spread_charges(const bool* __restrict__ node_mask,
                   const int64_t* __restrict__ frame_ptr,
                   const FrameMesh* __restrict__ frames,
                   const AxisStencil<kOrder>* __restrict__ stencils,
                   const float* __restrict__ node_charge,
                   int frame_count,
                   int64_t node_count,
                   int channels,
                   unsigned long long* __restrict__ mesh) {
  using Slots = LaneSlots<kOrder>;
  __shared__ AxisStencil<kOrder> shared[kStencilWarps][3];
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const AxisStencil<kOrder>* axes = shared[warp];
  for (int64_t node = int64_t{blockIdx.x} * kStencilWarps + warp;
       node < node_count; node += int64_t{gridDim.x} * kStencilWarps) {
    const int f = frame_of(frame_ptr, frame_count, node);
    if (f < 0 || !node_mask[node]) {
      continue;
    }
    load_stencil<kOrder>(stencils, node, lane, shared[warp]);
    const FrameMesh& grid = frames[f];
    for (int c = 0; c < channels; ++c) {
      const float q = node_charge[node * channels + c] * kFixedScaleFloat;
      unsigned long long* channel_mesh =
          mesh + grid.offset + c * mesh_points(grid);
#pragma unroll
      for (int round = 0; round < Slots::kRounds; ++round) {
        const int slot = lane + kWarp * round;
        if (slot < Slots::kSlots) {
          const int group = slot / Slots::kPairs;
          const int pair = slot - group * Slots::kPairs;
          const int k2 = pair / kOrder;
          const int k3 = pair - k2 * kOrder;
          const float weight = q * axes[1].value[k2] * axes[2].value[k3];
          const int column = axes[1].offset[k2] + axes[2].offset[k3];
#pragma unroll
          for (int k1 = group; k1 < kOrder; k1 += Slots::kGroups) {
            atomicAdd(channel_mesh + axes[0].offset[k1] + column,
                      static_cast<unsigned long long>(
                          float_to_fixed(weight * axes[0].value[k1])));
          }
        }
      }
    }
    // The stencil of the next node of the warp overwrites this one.
    __syncwarp();
  }
}

// Stores the mesh of one frame, sized on the host and carried by value, with
// the geometry of its cell ``cell``. Thread k = 3 j + d of kGeometryThreads
// forms the entry (B^-1)_jd = C_dj / det B of the inverse of the cell B, C
// its cofactors, and the wave-vector and force factors of that entry; the
// first thread also forms the factor prefactor / V of the influence
// function.
constexpr int kGeometryThreads = 9;

__global__ void store_frame_mesh(FrameMesh mesh,
                                 const double* __restrict__ cell,
                                 double prefactor,
                                 FrameMesh* __restrict__ out) {
  const int k = threadIdx.x;
  const int j = k / 3;
  const int d = k % 3;
  const double det = cell[0] * (cell[4] * cell[8] - cell[5] * cell[7]) -
                     cell[1] * (cell[3] * cell[8] - cell[5] * cell[6]) +
                     cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
  const double* row1 = cell + 3 * ((d + 1) % 3);
  const double* row2 = cell + 3 * ((d + 2) % 3);
  const double inverse = (row1[(j + 1) % 3] * row2[(j + 2) % 3] -
                          row1[(j + 2) % 3] * row2[(j + 1) % 3]) /
                         det;
  out->inverse[k] = inverse;
  out->reciprocal[k] = static_cast<float>(2.0 * kPi * inverse);
  out->gradient_to_force[k] = static_cast<float>(-mesh.n[d] * inverse);
  if (k == 0) {
    out->influence_scale = static_cast<float>(prefactor / fabs(det));
    for (int axis = 0; axis < 3; ++axis) {
      out->n[axis] = mesh.n[axis];
    }
    out->offset = mesh.offset;
  }
}

// Converts the fixed-point mesh to the float32 mesh that the transforms read.
__global__ void fixed_to_float(int64_t count,
                               const unsigned long long* __restrict__ fixed,
                               float* __restrict__ mesh) {
  constexpr float kFixedInverse = 1.f / kFixedScaleFloat;
  for (int64_t index = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       index < count; index += int64_t{gridDim.x} * blockDim.x) {
    mesh[index] =
        __ll2float_rn(static_cast<long long>(fixed[index])) * kFixedInverse;
  }
}

// The weights M_p(1), ..., M_p(p - 1) of the assignment factor.
struct IntegerWeights {
  double value[kMaxOrder - 1];
};

// The inverse squared assignment factor |D_d(m)|^-2 along every axis of the
// half spectrum of a mesh: n1 entries, then n2, then n3 / 2 + 1. It depends
// only on the shape of the mesh and the order of the B-splines, so the plans
// of a shape hold it (create_mesh_plans).
__global__ void assignment_factor(FrameMesh mesh,
                                  int order,
                                  IntegerWeights integer_weight,
                                  float* __restrict__ factor) {
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
      re += integer_weight.value[l] * c;
      im += integer_weight.value[l] * s;
    }
    factor[index] = static_cast<float>(1.0 / (re * re + im * im));
  }
}

// The terms of the transform of the kernel, in float32, the precision of the
// transforms: 1 / (4 alpha^2) of the les kernel, or, for every Gaussian of the
// sog kernel, its weight a_g (2 pi)^(3/2) sigma_g^3 and half its squared
// width, from ``coefficient``, which holds alpha, or the amplitudes and then
// the rates c_g = log2(e) / (2 sigma_g^2).
struct KernelTerms {
  float inv_four_alpha_sq;
  float weight[kMaxGaussians];
  float half_width_sq[kMaxGaussians];
};

__device__ void load_kernel_terms(int kernel,
                                  const double* __restrict__ coefficient,
                                  int gaussians,
                                  KernelTerms& terms) {
  if (kernel == kLes) {
    if (threadIdx.x == 0) {
      const double alpha = coefficient[0];
      terms.inv_four_alpha_sq = static_cast<float>(0.25 / (alpha * alpha));
    }
    return;
  }
  // (2 pi)^{3/2}
  constexpr double kGaussianNorm = 15.749609945722419;
  for (int g = threadIdx.x; g < gaussians; g += blockDim.x) {
    // sigma^2 = log2(e) / (2 c) = 1 / (2 c ln 2)
    const double width_sq = 0.72134752044448170 / coefficient[gaussians + g];
    terms.weight[g] = static_cast<float>(coefficient[g] * kGaussianNorm *
                                         width_sq * sqrt(width_sq));
    terms.half_width_sq[g] = static_cast<float>(0.5 * width_sq);
  }
}

// The transform of the kernel at k^2 and its logarithmic derivative with
// respect to k^2.
__device__ __forceinline__ void kernel_transform(int kernel,
                                                 const KernelTerms& terms,
                                                 int gaussians,
                                                 float k_sq,
                                                 float& value,
                                                 float& log_slope) {
  if (kernel == kLes) {
    value = 4.f * static_cast<float>(kPi) *
            expf(-k_sq * terms.inv_four_alpha_sq) / k_sq;
    log_slope = -terms.inv_four_alpha_sq - 1.f / k_sq;
    return;
  }
  value = 0.f;
  float slope = 0.f;
  for (int g = 0; g < gaussians; ++g) {
    const float term = terms.weight[g] * expf(-terms.half_width_sq[g] * k_sq);
    value += term;
    slope -= terms.half_width_sq[g] * term;
  }
  log_slope = value != 0.f ? slope / value : 0.f;
}

// The influence function of one frame on its half spectrum: multiplies every
// mode of the spectra of the channels by G in place and sums, per block, the
// energy of the modes and their virial, xx yy zz xy xz yz. The modes are
// evaluated in float32, the precision of the transforms, summed in float32
// over the modes of a row that a lane takes, and in float64 over the rows.
constexpr int kModeSums = 7;

__global__ void __launch_bounds__(kThreads)
    apply_influence(const FrameMesh* __restrict__ frame,
                    int channels,
                    int kernel,
                    const double* __restrict__ coefficient,
                    int gaussians,
                    const float* __restrict__ factor,
                    float2* __restrict__ spectrum,
                    double* __restrict__ block_sum) {
  __shared__ KernelTerms terms;
  load_kernel_terms(kernel, coefficient, gaussians, terms);
  __syncthreads();
  const FrameMesh& mesh = *frame;
  const int n1 = mesh.n[0];
  const int n2 = mesh.n[1];
  const int n3 = mesh.n[2];
  const int half = n3 / 2 + 1;
  const int modes = n1 * n2 * half;
  // k_a = sum_d reciprocal[3 a + d] m_d = 2 pi sum_d (B^-1)_ad m_d
  float reciprocal[9];
#pragma unroll
  for (int k = 0; k < 9; ++k) {
    reciprocal[k] = mesh.reciprocal[k];
  }
  const float scale = mesh.influence_scale;
  // Every warp takes rows (i1, i2) of the half spectrum, its lanes the modes
  // of a row, which lie contiguously.
  const int lane = threadIdx.x % kWarp;
  const int warps = gridDim.x * (kThreads / kWarp);
  double sum[kModeSums] = {};
  for (int row = blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
       row < n1 * n2; row += warps) {
    float row_sum[kModeSums] = {};
    const int i1 = row / n2;
    const int i2 = row - i1 * n2;
    const int m1 = i1 <= n1 / 2 ? i1 : i1 - n1;
    const int m2 = i2 <= n2 / 2 ? i2 : i2 - n2;
    const bool row_weightless = 2 * m1 == n1 || 2 * m2 == n2;
    float k_row[3];
#pragma unroll
    for (int a = 0; a < 3; ++a) {
      k_row[a] = reciprocal[3 * a] * m1 + reciprocal[3 * a + 1] * m2;
    }
    const float row_factor = scale * factor[i1] * factor[n1 + i2];
    for (int i3 = lane; i3 < half; i3 += kWarp) {
      const int m3 = i3;
      const bool weightless =
          row_weightless || 2 * m3 == n3 || (m1 == 0 && m2 == 0 && m3 == 0);
      float influence = 0.f;
      float k[3] = {0.f, 0.f, 0.f};
      float log_slope = 0.f;
      if (!weightless) {
#pragma unroll
        for (int a = 0; a < 3; ++a) {
          k[a] = k_row[a] + reciprocal[3 * a + 2] * m3;
        }
        const float k_sq = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
        float transform;
        kernel_transform(kernel, terms, gaussians, k_sq, transform, log_slope);
        influence = row_factor * transform * factor[n1 + n2 + i3];
      }
      // An interior plane of the half spectrum stands for its mirror as well.
      const float multiplicity = m3 == 0 ? 1.f : 2.f;
      const int index = row * half + i3;
      float energy = 0.f;
      for (int c = 0; c < channels; ++c) {
        float2& value = spectrum[int64_t{c} * modes + index];
        energy += value.x * value.x + value.y * value.y;
        value.x *= influence;
        value.y *= influence;
      }
      energy *= 0.5f * multiplicity * influence;
      const float twice = 2.f * log_slope * energy;
      row_sum[0] += energy;
      row_sum[1] += energy + twice * k[0] * k[0];
      row_sum[2] += energy + twice * k[1] * k[1];
      row_sum[3] += energy + twice * k[2] * k[2];
      row_sum[4] += twice * k[0] * k[1];
      row_sum[5] += twice * k[0] * k[2];
      row_sum[6] += twice * k[1] * k[2];
    }
#pragma unroll
    for (int s = 0; s < kModeSums; ++s) {
      sum[s] += row_sum[s];
    }
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
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int stride = kWarp / 2; stride > 0; stride /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, stride);
  }
  return value;
}

// Interpolates the potential meshes at every owner node, one warp per node,
// one pass per channel: the potential phi^rec_c of the mesh, which still holds
// that of the charge at itself, and the force at fixed charges. Owners that
// carry no charge get zeros. Every lane sums its points in float32, the
// precision of the meshes, and the warp adds the lane sums.
template <int kOrder>
__global__ void __launch_bounds__(kStencilWarps* kWarp)
    interpolate_owners(const bool* __restrict__ node_mask,
                       const int64_t* __restrict__ frame_ptr,
                       const FrameMesh* __restrict__ frames,
                       const AxisStencil<kOrder>* __restrict__ stencils,
                       const float* __restrict__ node_charge,
                       const float* __restrict__ potential_mesh,
                       int frame_count,
                       int owner_begin,
                       int owner_count,
                       int channels,
                       float* __restrict__ potential,
                       float* __restrict__ force) {
  using Slots = LaneSlots<kOrder>;
  __shared__ AxisStencil<kOrder> shared[kStencilWarps][3];
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const AxisStencil<kOrder>* axes = shared[warp];
  for (int owner = blockIdx.x * kStencilWarps + warp; owner < owner_count;
       owner += gridDim.x * kStencilWarps) {
    const int64_t node = int64_t{owner_begin} + owner;
    const int f = frame_of(frame_ptr, frame_count, node);
    if (f < 0 || !node_mask[node]) {
      for (int c = lane; c < channels; c += kWarp) {
        potential[int64_t{owner} * channels + c] = 0.f;
      }
      if (lane < 3) {
        force[int64_t{owner} * 3 + lane] = 0.f;
      }
      continue;
    }
    load_stencil<kOrder>(stencils, node, lane, shared[warp]);
    const FrameMesh& grid = frames[f];
    // The derivative of sum_c q_c phi_c with respect to u_d.
    float gradient[3] = {0.f, 0.f, 0.f};
    for (int c = 0; c < channels; ++c) {
      const float* channel_mesh =
          potential_mesh + grid.offset + c * mesh_points(grid);
      float phi = 0.f;
      float slope[3] = {0.f, 0.f, 0.f};
#pragma unroll
      for (int round = 0; round < Slots::kRounds; ++round) {
        const int slot = lane + kWarp * round;
        if (slot < Slots::kSlots) {
          const int group = slot / Slots::kPairs;
          const int pair = slot - group * Slots::kPairs;
          const int k2 = pair / kOrder;
          const int k3 = pair - k2 * kOrder;
          const float v2 = axes[1].value[k2];
          const float v3 = axes[2].value[k3];
          const float v23 = v2 * v3;
          const float s2v3 = axes[1].slope[k2] * v3;
          const float v2s3 = v2 * axes[2].slope[k3];
          const int column = axes[1].offset[k2] + axes[2].offset[k3];
#pragma unroll
          for (int k1 = group; k1 < kOrder; k1 += Slots::kGroups) {
            const float value = channel_mesh[axes[0].offset[k1] + column];
            const float v1_value = axes[0].value[k1] * value;
            phi += v1_value * v23;
            slope[0] += axes[0].slope[k1] * value * v23;
            slope[1] += v1_value * s2v3;
            slope[2] += v1_value * v2s3;
          }
        }
      }
      phi = warp_sum(phi);
      const float q = node_charge[node * channels + c];
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        gradient[d] += q * warp_sum(slope[d]);
      }
      if (lane == 0) {
        potential[int64_t{owner} * channels + c] = phi;
      }
    }
    if (lane < 3) {
      // F_j = -sum_d dE/du_d n_d (B^-1)_jd
      force[int64_t{owner} * 3 + lane] =
          gradient[0] * grid.gradient_to_force[3 * lane] +
          gradient[1] * grid.gradient_to_force[3 * lane + 1] +
          gradient[2] * grid.gradient_to_force[3 * lane + 2];
    }
    // The stencil of the next node of the warp overwrites this one.
    __syncwarp();
  }
}

// The potential prefactor * K(0) of a unit charge at itself, which the
// potential mesh holds and the potential of a node excludes, for every thread
// of the block.
__device__ double self_potential(int kernel,
                                 const double* __restrict__ coefficient,
                                 int gaussians,
                                 double prefactor) {
  __shared__ double self;
  if (threadIdx.x == 0) {
    self = prefactor * contact_value(kernel, coefficient, gaussians);
  }
  __syncthreads();
  return self;
}

// Per-frame sums over one slice of the owners: the potential of the charged
// owners in the constrained channel, the number of charged owners and the
// nine components of sum_i F_i (x) r_i.
constexpr int kOwnerSums = 11;

__global__ void __launch_bounds__(kFrameThreads)
    owner_slices(const float* __restrict__ potential,
                 const float* __restrict__ force,
                 const double* __restrict__ owner_coord,
                 const float* __restrict__ owner_charge,
                 const bool* __restrict__ owner_mask,
                 const int64_t* __restrict__ owner_frame_ptr,
                 int channels,
                 int channel,
                 int kernel,
                 const double* __restrict__ coefficient,
                 int gaussians,
                 double prefactor,
                 double* __restrict__ slice_sum) {
  const double self = self_potential(kernel, coefficient, gaussians, prefactor);
  const NodeRange slice = slice_of_frame(owner_frame_ptr);
  double value[kOwnerSums] = {};
  for (int64_t owner = slice.begin + threadIdx.x; owner < slice.end;
       owner += kFrameThreads) {
    if (!owner_mask[owner]) {
      continue;
    }
    if (channel >= 0) {
      value[0] += double{potential[owner * channels + channel]} -
                  self * owner_charge[owner * channels + channel];
    }
    value[1] += 1.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        value[2 + 3 * a + b] +=
            double{force[owner * 3 + a]} * owner_coord[owner * 3 + b];
      }
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

// Writes the outputs of every owner: the atomic energy q . phi / 2 with phi =
// phi^rec - prefactor K(0) q, the charge gradient w (phi - [c = ch] mean), and
// the virial F (x) r plus the share of the rest of the virial W of the frame
// over its charged nodes. The first threads write the share of the virial of
// every frame held by the owners, sum_i F_i (x) r_i + n (W - sum_j F_j (x)
// r_j) / N over the n charged owners of the N charged nodes of the frame.
// ``owner_local`` holds the owner sums of this domain, ``owner_total`` those
// of every domain.
__global__ void finish_owners(const float* __restrict__ potential,
                              const float* __restrict__ force,
                              const double* __restrict__ owner_coord,
                              const float* __restrict__ owner_charge,
                              const bool* __restrict__ owner_mask,
                              const int64_t* __restrict__ owner_frame_ptr,
                              const double* __restrict__ frame_virial,
                              const double* __restrict__ owner_local,
                              const double* __restrict__ owner_total,
                              const double* __restrict__ frame_stat,
                              int frame_count,
                              int owner_count,
                              int channels,
                              int channel,
                              int kernel,
                              const double* __restrict__ coefficient,
                              int gaussians,
                              double prefactor,
                              double* __restrict__ atom_energy,
                              float* __restrict__ charge_gradient,
                              float* __restrict__ atom_virial,
                              float* __restrict__ virial) {
  const double self = self_potential(kernel, coefficient, gaussians, prefactor);
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int threads = gridDim.x * blockDim.x;
  for (int index = start; index < 9 * frame_count; index += threads) {
    const int f = index / 9;
    const int k = index - 9 * f;
    const double count = frame_stat[f * kStatWidth + kStatCount];
    const double* local = owner_local + f * kOwnerSums;
    const double rest =
        count > 0.0
            ? (frame_virial[index] - owner_total[f * kOwnerSums + 2 + k]) /
                  count
            : 0.0;
    virial[index] = static_cast<float>(local[2 + k] + local[1] * rest);
  }
  for (int owner = start; owner < owner_count; owner += threads) {
    const int f = frame_of(owner_frame_ptr, frame_count, owner);
    const bool charged = f >= 0 && owner_mask[owner];
    // The charged nodes of the frame, of every domain.
    const double count =
        charged ? frame_stat[f * kStatWidth + kStatCount] : 1.0;
    const double mean =
        charged && channel >= 0 ? owner_total[f * kOwnerSums] / count : 0.0;
    double energy = 0.0;
    for (int c = 0; c < channels; ++c) {
      const int64_t entry = int64_t{owner} * channels + c;
      const double q = owner_charge[entry];
      const double phi = charged ? potential[entry] - self * q : 0.0;
      energy += q * phi;
      charge_gradient[entry] =
          static_cast<float>(c == channel ? phi - mean : phi);
    }
    atom_energy[owner] = 0.5 * energy;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        const int k = 3 * a + b;
        const double moment = double{force[int64_t{owner} * 3 + a]} *
                              owner_coord[int64_t{owner} * 3 + b];
        const double rest = charged ? (frame_virial[f * 9 + k] -
                                       owner_total[f * kOwnerSums + 2 + k]) /
                                          count
                                    : 0.0;
        atom_virial[int64_t{owner} * 9 + k] = static_cast<float>(moment + rest);
      }
    }
  }
}

IntegerWeights integer_bspline_weights(int order) {
  IntegerWeights weight{};
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
    weight.value[l] = value / factorial;
  }
  return weight;
}

// The launch of a kernel of float64 work, one item per thread. The float64
// pipe of a multiprocessor serves every warp on it, so few items spread over
// as many multiprocessors as there are warps of them, in blocks of a few
// warps, and many items fill blocks of kThreads.
struct Float64Launch {
  int blocks;
  int threads;
};

Float64Launch float64_launch(int64_t work) {
  const int64_t multiprocessors =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t warps = (work + kWarp - 1) / kWarp;
  const int64_t warps_per_block = std::clamp<int64_t>(
      (warps + multiprocessors - 1) / multiprocessors, 1, kThreads / kWarp);
  const int threads = static_cast<int>(warps_per_block * kWarp);
  return {static_cast<int>(std::max<int64_t>(
              1, std::min<int64_t>((work + threads - 1) / threads, 65535))),
          threads};
}

// Blocks of the influence function of a mesh: a few per multiprocessor fill
// the device, and no more blocks than the rows of the half spectrum fill, a
// row to a warp, so that the block sums stay few. The count depends on the
// mesh and the device alone, so that the modes of a frame are summed in the
// same order whatever the other frames of its call.
int influence_blocks(const FrameMesh& mesh) {
  constexpr int64_t kWarps = kThreads / kWarp;
  const int64_t rows = int64_t{mesh.n[0]} * mesh.n[1];
  return static_cast<int>(std::clamp<int64_t>(
      (rows + kWarps - 1) / kWarps, 1,
      4 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount));
}

void check_cufft(cufftResult result, const char* operation) {
  TORCH_CHECK(result == CUFFT_SUCCESS, "latent_charge_sum: cuFFT ", operation,
              " failed with error ", static_cast<int>(result));
}

// The real-to-complex and complex-to-real transforms of the meshes of one
// shape, batched over the channels, and the inverse squared assignment
// factors of the shape at one B-spline order, which depend on nothing else
// and are formed once. The plans allocate no work area: every transform runs
// on a work area of the workspace of its call, set with the stream under
// ``mutex`` just before the transform is enqueued, so that concurrent calls
// never share one. A CUDA graph that captures a transform runs it on these
// plans at every replay, so it holds them while it lives: ``graphs`` counts
// the graphs that hold them.
struct MeshPlans {
  cufftHandle forward = 0;
  cufftHandle inverse = 0;
  size_t work_bytes = 0;
  // float32, n1 + n2 + n3 / 2 + 1 entries on the device (assignment_factor).
  torch::Tensor factor;
  c10::DeviceIndex device = 0;
  std::mutex mutex;
  std::atomic<int> graphs{0};

  MeshPlans() = default;
  MeshPlans(const MeshPlans&) = delete;
  MeshPlans& operator=(const MeshPlans&) = delete;
  // Plans leave the cache only past kPlanCacheSize shapes and once no graph
  // holds them; the transforms of eager calls may still be in flight, so the
  // device finishes them first. The destructor reports no error.
  ~MeshPlans() {
    int current = 0;
    cudaGetDevice(&current);
    cudaSetDevice(device);
    cudaDeviceSynchronize();
    if (forward != 0) {
      cufftDestroy(forward);
    }
    if (inverse != 0) {
      cufftDestroy(inverse);
    }
    cudaSetDevice(current);
  }
};

// The plans of the meshes of ``mesh`` with ``channels`` channels on the
// current device, with the assignment factors of B-splines of order
// ``order``, formed on ``stream``, which the creation waits for so that calls
// on every stream read them.
std::shared_ptr<MeshPlans> create_mesh_plans(const FrameMesh& mesh,
                                             int channels,
                                             int order,
                                             cudaStream_t stream) {
  auto plans = std::make_shared<MeshPlans>();
  plans->device = c10::cuda::current_device();
  long long shape[3] = {mesh.n[0], mesh.n[1], mesh.n[2]};
  const long long points = shape[0] * shape[1] * shape[2];
  const long long modes = shape[0] * shape[1] * (shape[2] / 2 + 1);
  size_t forward_bytes = 0;
  size_t inverse_bytes = 0;
  check_cufft(cufftCreate(&plans->forward), "plan creation");
  check_cufft(cufftSetAutoAllocation(plans->forward, 0), "plan setup");
  check_cufft(
      cufftMakePlanMany64(plans->forward, 3, shape, nullptr, 1, points, nullptr,
                          1, modes, CUFFT_R2C, channels, &forward_bytes),
      "plan creation");
  check_cufft(cufftCreate(&plans->inverse), "plan creation");
  check_cufft(cufftSetAutoAllocation(plans->inverse, 0), "plan setup");
  check_cufft(
      cufftMakePlanMany64(plans->inverse, 3, shape, nullptr, 1, modes, nullptr,
                          1, points, CUFFT_C2R, channels, &inverse_bytes),
      "plan creation");
  plans->work_bytes = std::max(forward_bytes, inverse_bytes);
  const int64_t factor_count =
      int64_t{mesh.n[0]} + mesh.n[1] + mesh.n[2] / 2 + 1;
  plans->factor =
      torch::empty({factor_count}, torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(torch::kCUDA, plans->device));
  const Float64Launch launch = float64_launch(factor_count);
  assignment_factor<<<launch.blocks, launch.threads, 0, stream>>>(
      mesh, order, integer_bspline_weights(order),
      plans->factor.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  return plans;
}

// Ends the hold of a graph on the plans it captured. CUDA runs it on a thread
// of its own, from which no CUDA call may be made, so the hold is a count
// that the cache reads rather than an owner that would release the plans.
void CUDART_CB release_graph_hold(void* plans) {
  static_cast<MeshPlans*>(plans)->graphs.fetch_sub(1);
}

// Makes ``graph``, which a capture is recording, hold ``plans`` until it and
// every graph instantiated from it are destroyed.
void hold_in_graph(MeshPlans& plans, cudaGraph_t graph) {
  cudaUserObject_t hold = nullptr;
  C10_CUDA_CHECK(cudaUserObjectCreate(&hold, &plans, release_graph_hold, 1,
                                      cudaUserObjectNoDestructorSync));
  plans.graphs.fetch_add(1);
  const cudaError_t moved =
      cudaGraphRetainUserObject(graph, hold, 1, cudaGraphUserObjectMove);
  if (moved != cudaSuccess) {
    // The reference stays with this call, whose release ends the hold.
    cudaUserObjectRelease(hold, 1);
  }
  C10_CUDA_CHECK(moved);
}

// Whether a CUDA graph captures the stream of a call, and that graph.
struct StreamCapture {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cudaGraph_t graph = nullptr;
};

StreamCapture stream_capture(cudaStream_t stream) {
  StreamCapture capture;
#if CUDART_VERSION >= 12000
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &capture.status, nullptr,
                                          &capture.graph));
#else
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(stream, &capture.status, nullptr,
                                             &capture.graph));
#endif
  return capture;
}

// The plans of the meshes of ``mesh`` with ``channels`` channels and
// B-splines of order ``order`` on the current device, kept across calls. A
// call under ``capture`` makes the graph of the capture hold them and needs
// them created before, by an eager call, since their creation waits for the
// stream. The other calls release the least recently used plans that no
// graph holds past kPlanCacheSize shapes, which a call under capture may not
// do, since a release waits for the device. The cache itself lives until the
// process ends and is never destroyed, since its plans would otherwise be
// released after the CUDA runtime shuts down.
constexpr size_t kPlanCacheSize = 16;

std::shared_ptr<MeshPlans> mesh_plans(const FrameMesh& mesh,
                                      int channels,
                                      int order,
                                      const StreamCapture& capture,
                                      cudaStream_t stream) {
  using Key = std::array<int, 6>;
  using Cache = std::list<std::pair<Key, std::shared_ptr<MeshPlans>>>;
  static std::mutex cache_mutex;
  static Cache& cache = *new Cache();
  const Key key = {c10::cuda::current_device(),
                   mesh.n[0],
                   mesh.n[1],
                   mesh.n[2],
                   channels,
                   order};
  // Plans that leave the cache are released after the lock, since their
  // release waits for the device.
  std::vector<std::shared_ptr<MeshPlans>> evicted;
  const std::lock_guard<std::mutex> lock(cache_mutex);
  const auto found =
      std::find_if(cache.begin(), cache.end(),
                   [&](const auto& entry) { return entry.first == key; });
  if (found != cache.end()) {
    cache.splice(cache.begin(), cache, found);
  } else {
    TORCH_CHECK(capture.status == cudaStreamCaptureStatusNone,
                "latent_charge_sum: a CUDA graph captures only meshes that an "
                "eager call has evaluated, and none has evaluated the mesh of ",
                mesh.n[0], " x ", mesh.n[1], " x ", mesh.n[2],
                " points, channel count ", channels, " and B-spline order ",
                order);
    cache.emplace_front(key, create_mesh_plans(mesh, channels, order, stream));
  }
  std::shared_ptr<MeshPlans> plans = cache.front().second;
  if (capture.status == cudaStreamCaptureStatusActive) {
    hold_in_graph(*plans, capture.graph);
  } else if (capture.status == cudaStreamCaptureStatusNone) {
    // From the least recently used back to, but short of, the front entry,
    // which holds the plans of this call.
    auto entry = cache.end();
    while (cache.size() > kPlanCacheSize && --entry != cache.begin()) {
      if (entry->second->graphs.load() == 0) {
        evicted.push_back(std::move(entry->second));
        entry = cache.erase(entry);
      }
    }
  }
  return plans;
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

// The value of ``evaluate`` at the B-spline order as a compile-time constant.
template <typename Evaluate>
auto dispatch_order_value(int64_t order, const Evaluate& evaluate) {
  decltype(evaluate(std::integral_constant<int, 2>{})) value{};
  dispatch_order(
      order, [&](auto order_constant) { value = evaluate(order_constant); });
  return value;
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
  // The device reads pinned cells in place, and a copy of pageable ones; a
  // CUDA graph that captures the call reads the cells again at every replay,
  // which only pinned memory allows.
  const StreamCapture capture = stream_capture(stream);
  const bool pinned = box.is_pinned();
  TORCH_CHECK(capture.status == cudaStreamCaptureStatusNone || pinned,
              "latent_charge_sum: a CUDA graph that captures periodic frames "
              "reads their cells at every replay, from box in pinned memory");

  // === Step 1. The meshes of the frames, sized on the host ===
  std::vector<FrameMesh> frames(frame_count);
  std::vector<std::shared_ptr<MeshPlans>> plans(frame_count);
  const double* cells = box.data_ptr<double>();
  int64_t mesh_total = 0;
  int64_t spectrum_size = 0;
  int mode_blocks = 1;
  size_t work_bytes = 0;
  for (int f = 0; f < frame_count; ++f) {
    frames[f] = frame_mesh_size(cells + 9 * f, spacing);
    frames[f].offset = mesh_total;
    mesh_total += channels * mesh_points(frames[f]);
    const int half = frames[f].n[2] / 2 + 1;
    spectrum_size = std::max(spectrum_size, int64_t{channels} * frames[f].n[0] *
                                                frames[f].n[1] * half);
    mode_blocks = std::max(mode_blocks, influence_blocks(frames[f]));
    plans[f] = mesh_plans(frames[f], channels, static_cast<int>(order), capture,
                          stream);
    work_bytes = std::max(work_bytes, plans[f]->work_bytes);
  }

  // === Step 2. The buffers of the call ===
  const int64_t average_nodes = (node_count + frame_count - 1) / frame_count;
  const auto slices_for = [](int64_t nodes) {
    return static_cast<int>(std::min<int64_t>(
        kMaxSlices,
        std::max<int64_t>(1, (nodes + kSliceNodes - 1) / kSliceNodes)));
  };
  const int slice_count = slices_for(average_nodes);
  const int owner_slice_count = slices_for(
      (int64_t{owner_count} + frame_count - 1) / std::max(frame_count, 1));
  const int64_t stencil_bytes = dispatch_order_value(order, [](auto constant) {
    return static_cast<int64_t>(sizeof(AxisStencil<decltype(constant)::value>));
  });
  Workspace workspace;
  const int64_t frames_at =
      workspace.reserve<uint8_t>(sizeof(FrameMesh) * frame_count);
  const int64_t cells_at =
      workspace.reserve<double>(pinned ? 0 : int64_t{frame_count} * 9);
  const int64_t frame_ptr_at = workspace.reserve<int64_t>(frame_count + 1);
  const int64_t slice_charge_at =
      workspace.reserve<double>(int64_t{frame_count} * slice_count * 2);
  const int64_t frame_stat_at =
      workspace.reserve<double>(int64_t{frame_count} * kStatWidth);
  const int64_t stencils_at =
      workspace.reserve<uint8_t>(3 * node_count * stencil_bytes);
  const int64_t node_charge_at =
      workspace.reserve<float>(node_count * channels);
  const int64_t fixed_mesh_at = workspace.reserve<int64_t>(mesh_total);
  // The potential mesh of a frame overwrites its charge mesh, which its
  // forward transform has consumed.
  const int64_t mesh_at = workspace.reserve<float>(mesh_total);
  const int64_t spectrum_at = workspace.reserve<float2>(spectrum_size);
  const int64_t work_at = workspace.reserve<uint8_t>(work_bytes);
  const int64_t block_sum_at =
      workspace.reserve<double>(int64_t{mode_blocks} * kModeSums);
  const int64_t frame_virial_at =
      workspace.reserve<double>(int64_t{frame_count} * 9);
  const int64_t potential_at =
      workspace.reserve<float>(int64_t{owner_count} * channels);
  const int64_t owner_slice_at = workspace.reserve<double>(
      int64_t{frame_count} * owner_slice_count * kOwnerSums);
  const int64_t owner_local_at =
      workspace.reserve<double>(int64_t{frame_count} * kOwnerSums);
  workspace.allocate(coord.device());
  auto* frames_ptr = workspace.at<FrameMesh>(frames_at);
  auto* frame_ptr = workspace.at<int64_t>(frame_ptr_at);
  auto* slice_charge = workspace.at<double>(slice_charge_at);
  auto* frame_stat = workspace.at<double>(frame_stat_at);
  auto* node_charge = workspace.at<float>(node_charge_at);
  auto* fixed_mesh = workspace.at<unsigned long long>(fixed_mesh_at);
  auto* charge_mesh = workspace.at<float>(mesh_at);
  float* potential_mesh = charge_mesh;
  auto* spectrum = workspace.at<float2>(spectrum_at);
  void* work = workspace.at<uint8_t>(work_at);
  auto* block_sum = workspace.at<double>(block_sum_at);
  auto* frame_virial = workspace.at<double>(frame_virial_at);
  auto* potential = workspace.at<float>(potential_at);
  auto* owner_slice = workspace.at<double>(owner_slice_at);
  auto* owner_local = workspace.at<double>(owner_local_at);
  const double* device_cells = nullptr;
  if (pinned) {
    void* mapped = nullptr;
    C10_CUDA_CHECK(
        cudaHostGetDevicePointer(&mapped, const_cast<double*>(cells), 0));
    device_cells = static_cast<const double*>(mapped);
  } else {
    // Pageable cells are staged in pinned memory, from which the copy runs
    // without waiting for the stream; the host allocator keeps the staging
    // memory until the copy has completed.
    auto staged =
        torch::empty({frame_count, 9}, box.options().pinned_memory(true));
    std::memcpy(staged.data_ptr<double>(), cells,
                sizeof(double) * 9 * frame_count);
    auto* copy = workspace.at<double>(cells_at);
    torch::from_blob(copy, {frame_count, 9}, coord.options())
        .copy_(staged, /*non_blocking=*/true);
    device_cells = copy;
  }
  for (int f = 0; f < frame_count; ++f) {
    store_frame_mesh<<<1, kGeometryThreads, 0, stream>>>(
        frames[f], device_cells + 9 * f, prefactor, frames_ptr + f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  C10_CUDA_CHECK(
      cudaMemsetAsync(fixed_mesh, 0, sizeof(int64_t) * mesh_total, stream));

  // === Step 3. The constraint, and the stencils and charges of the nodes ===
  const auto counts = n_node_per_frame.to(torch::kInt64).contiguous();
  frame_offsets<<<1, kOffsetThreads, 0, stream>>>(counts.data_ptr<int64_t>(),
                                                  frame_count, frame_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  charge_slices<<<dim3(frame_count, slice_count), kFrameThreads, 0, stream>>>(
      charge.data_ptr<float>(), node_mask.data_ptr<bool>(), frame_ptr, channels,
      static_cast<int>(std::max<int64_t>(channel, 0)), slice_charge);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  charge_shift<<<(frame_count + kFrameThreads - 1) / kFrameThreads,
                 kFrameThreads, 0, stream>>>(
      slice_charge, slice_count, frame_count, total_charge.data_ptr<double>(),
      frame_stat);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dispatch_order(order, [&](auto order_constant) {
    constexpr int kOrder = decltype(order_constant)::value;
    auto* node_stencil = workspace.at<AxisStencil<kOrder>>(stencils_at);
    const Float64Launch stencil_launch = float64_launch(3 * node_count);
    node_stencils<kOrder>
        <<<stencil_launch.blocks, stencil_launch.threads, 0, stream>>>(
            coord.data_ptr<double>(), charge.data_ptr<float>(),
            node_mask.data_ptr<bool>(), frame_ptr, frames_ptr, frame_stat,
            frame_count, node_count, channels, static_cast<int>(channel),
            node_stencil, node_charge);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // === Step 4. The charge meshes ===
    spread_charges<kOrder>
        <<<stencil_blocks_for(node_count), kStencilWarps * kWarp, 0, stream>>>(
            node_mask.data_ptr<bool>(), frame_ptr, frames_ptr, node_stencil,
            node_charge, frame_count, node_count, channels, fixed_mesh);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  fixed_to_float<<<blocks_for(mesh_total), kThreads, 0, stream>>>(
      mesh_total, fixed_mesh, charge_mesh);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // === Step 5. Influence function and potential meshes, frame by frame ===
  for (int f = 0; f < frame_count; ++f) {
    const FrameMesh& grid = frames[f];
    MeshPlans& plan = *plans[f];
    {
      const std::lock_guard<std::mutex> lock(plan.mutex);
      check_cufft(cufftSetStream(plan.forward, stream), "stream setup");
      check_cufft(cufftSetWorkArea(plan.forward, work), "work area setup");
      check_cufft(cufftExecR2C(plan.forward, charge_mesh + grid.offset,
                               reinterpret_cast<cufftComplex*>(spectrum)),
                  "transform");
    }
    const int blocks = influence_blocks(grid);
    apply_influence<<<blocks, kThreads, 0, stream>>>(
        frames_ptr + f, channels, static_cast<int>(kernel),
        coefficient.data_ptr<double>(), gaussians,
        plan.factor.data_ptr<float>(), spectrum, block_sum);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    mode_virial<<<1, kThreads, 0, stream>>>(block_sum, blocks,
                                            frame_virial + 9 * f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    {
      const std::lock_guard<std::mutex> lock(plan.mutex);
      check_cufft(cufftSetStream(plan.inverse, stream), "stream setup");
      check_cufft(cufftSetWorkArea(plan.inverse, work), "work area setup");
      check_cufft(
          cufftExecC2R(plan.inverse, reinterpret_cast<cufftComplex*>(spectrum),
                       potential_mesh + grid.offset),
          "transform");
    }
  }

  // === Step 6. Potential, force and virial of the owners ===
  auto force = torch::empty({owner_count, 3}, float_options);
  if (owner_count > 0) {
    dispatch_order(order, [&](auto order_constant) {
      constexpr int kOrder = decltype(order_constant)::value;
      interpolate_owners<kOrder><<<stencil_blocks_for(owner_count),
                                   kStencilWarps * kWarp, 0, stream>>>(
          node_mask.data_ptr<bool>(), frame_ptr, frames_ptr,
          workspace.at<AxisStencil<kOrder>>(stencils_at), node_charge,
          potential_mesh, frame_count, owner_begin, owner_count, channels,
          potential, force.data_ptr<float>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // The frames of the owners, on the axis of the owners: the frames
  // themselves when every node is an owner, else the one frame they share.
  torch::Tensor owner_frame_tensor;
  const int64_t* owner_frame_ptr = frame_ptr;
  if (!every_node) {
    owner_frame_tensor =
        torch::full({2}, int64_t{owner_count}, counts.options());
    owner_frame_tensor.select(0, 0).zero_();
    owner_frame_ptr = owner_frame_tensor.data_ptr<int64_t>();
  }
  const bool* owner_mask = node_mask.data_ptr<bool>() + owner_begin;
  const double* owner_coord = coord.data_ptr<double>() + 3 * owner_begin;
  const float* owner_charge = node_charge + int64_t{owner_begin} * channels;
  owner_slices<<<dim3(frame_count, owner_slice_count), kFrameThreads, 0,
                 stream>>>(potential, force.data_ptr<float>(), owner_coord,
                           owner_charge, owner_mask, owner_frame_ptr, channels,
                           static_cast<int>(channel), static_cast<int>(kernel),
                           coefficient.data_ptr<double>(), gaussians, prefactor,
                           owner_slice);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  owner_totals<<<frame_count, 32, 0, stream>>>(owner_slice, owner_slice_count,
                                               owner_local);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The mean potential of the constraint and the virial share run over the
  // charged nodes of every domain.
  torch::Tensor owner_sum;
  const double* owner_total = owner_local;
  if (communicator != nullptr) {
    owner_sum = sum_over_domains(
        *communicator, torch::from_blob(owner_local, {frame_count * kOwnerSums},
                                        double_options));
    owner_total = owner_sum.data_ptr<double>();
  }
  auto atom_energy = torch::empty({owner_count, 1}, double_options);
  auto charge_gradient = torch::empty({owner_count, channels}, float_options);
  auto atom_virial = torch::empty({owner_count, 9}, float_options);
  auto virial = torch::empty({frame_count, 9}, float_options);
  const Float64Launch owner_launch =
      float64_launch(std::max<int64_t>(owner_count, 9 * frame_count));
  finish_owners<<<owner_launch.blocks, owner_launch.threads, 0, stream>>>(
      potential, force.data_ptr<float>(), owner_coord, owner_charge, owner_mask,
      owner_frame_ptr, frame_virial, owner_local, owner_total, frame_stat,
      frame_count, owner_count, channels, static_cast<int>(channel),
      static_cast<int>(kernel), coefficient.data_ptr<double>(), gaussians,
      prefactor, atom_energy.data_ptr<double>(),
      charge_gradient.data_ptr<float>(), atom_virial.data_ptr<float>(),
      virial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {atom_energy, charge_gradient, force, atom_virial, virial};
}

}  // namespace long_range
}  // namespace deepmd
