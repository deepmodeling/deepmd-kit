// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Runtime resource selection for the compressed DPA1 CUDA kernels.

#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace deepmd::dpa1_compress_tuning {

enum class ResourcePolicy : int {
  kBalanced = 0,
  kOccupancy = 1,
};

enum class KernelDirection : int {
  kForward = 0,
  kBackward = 1,
};

struct LaunchConfig {
  ResourcePolicy resource;
  int threads;
};

struct TuningKey {
  int device;
  int direction;
  int width;
  int axis;
  int canonical;
  int index_bytes;
  int model_flags;
  int model_stride;
  int type_class;
  int size_class;
  int degree_class;

  bool operator==(const TuningKey& other) const {
    return device == other.device && direction == other.direction &&
           width == other.width && axis == other.axis &&
           canonical == other.canonical && index_bytes == other.index_bytes &&
           model_flags == other.model_flags &&
           model_stride == other.model_stride &&
           type_class == other.type_class && size_class == other.size_class &&
           degree_class == other.degree_class;
  }
};

struct TuningKeyHash {
  std::size_t operator()(const TuningKey& key) const {
    std::size_t value = 0;
    const std::array<int, 11> fields = {
        key.device,     key.direction,   key.width,       key.axis,
        key.canonical,  key.index_bytes, key.model_flags, key.model_stride,
        key.type_class, key.size_class,  key.degree_class};
    for (const int field : fields) {
      value ^=
          std::hash<int>{}(field) + 0x9e3779b9 + (value << 6) + (value >> 2);
    }
    return value;
  }
};

inline std::mutex& tuning_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

inline std::unordered_map<TuningKey, LaunchConfig, TuningKeyHash>&
tuning_cache() {
  static std::unordered_map<TuningKey, LaunchConfig, TuningKeyHash> cache;
  return cache;
}

inline const cudaDeviceProp& device_properties(int device) {
  constexpr int kMaximumCachedDevices = 64;
  TORCH_CHECK(device >= 0 && device < kMaximumCachedDevices,
              "dpa1_graph_compress: unsupported CUDA device index ", device);
  static std::array<std::once_flag, kMaximumCachedDevices> initialized;
  static std::array<cudaDeviceProp, kMaximumCachedDevices> properties;
  std::call_once(initialized[device], [device] {
    TORCH_CHECK(
        cudaGetDeviceProperties(&properties[device], device) == cudaSuccess,
        "dpa1_graph_compress: cannot query CUDA device properties");
  });
  return properties[device];
}

inline LaunchConfig architecture_fallback(const cudaDeviceProp& properties) {
  if (properties.major >= 9) {
    if (properties.multiProcessorCount <= 80) {
      return {ResourcePolicy::kOccupancy, 256};
    }
    return {ResourcePolicy::kBalanced, 256};
  }
  if (properties.major == 8) {
    return {ResourcePolicy::kBalanced, 256};
  }
  if (properties.major == 7) {
    return {ResourcePolicy::kBalanced, 128};
  }
  return {ResourcePolicy::kBalanced, 256};
}

inline int workload_size_class(long node_count, int multiprocessor_count) {
  const long nodes_per_sm = node_count / std::max(multiprocessor_count, 1);
  if (nodes_per_sm < 8) {
    return 0;
  }
  if (nodes_per_sm < 64) {
    return 1;
  }
  return 2;
}

inline int workload_degree_class(long node_count, long edge_count) {
  const long degree = edge_count / std::max(node_count, 1L);
  if (degree < 32) {
    return 0;
  }
  if (degree < 128) {
    return 1;
  }
  return 2;
}

inline int type_count_class(int type_count) {
  if (type_count <= 4) {
    return 0;
  }
  if (type_count <= 16) {
    return 1;
  }
  return 2;
}

template <typename LaunchFunction>
LaunchConfig select_launch_config(const TuningKey& key,
                                  const cudaDeviceProp& properties,
                                  long node_count,
                                  cudaStream_t stream,
                                  const LaunchFunction& launch) {
  const LaunchConfig fallback = architecture_fallback(properties);

  std::lock_guard<std::mutex> lock(tuning_cache_mutex());
  auto& cache = tuning_cache();
  const auto cached = cache.find(key);
  if (cached != cache.end()) {
    return cached->second;
  }

  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  const cudaError_t capture_error =
      cudaStreamIsCapturing(stream, &capture_status);
  if (capture_error != cudaSuccess ||
      capture_status != cudaStreamCaptureStatusNone) {
    return fallback;
  }
  if (node_count < 128) {
    cache.emplace(key, fallback);
    return fallback;
  }

  const long sample_node_count = std::min(
      node_count,
      std::max(4096L, static_cast<long>(properties.multiProcessorCount) * 64));
  constexpr std::array<LaunchConfig, 4> kCandidates = {{
      {ResourcePolicy::kBalanced, 128},
      {ResourcePolicy::kBalanced, 256},
      {ResourcePolicy::kOccupancy, 128},
      {ResourcePolicy::kOccupancy, 256},
  }};
  constexpr int kRepetitions = 2;

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (cudaEventCreate(&start) != cudaSuccess ||
      cudaEventCreate(&stop) != cudaSuccess) {
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
    cache.emplace(key, fallback);
    return fallback;
  }

  LaunchConfig best = fallback;
  float best_milliseconds = std::numeric_limits<float>::infinity();
  for (const LaunchConfig& candidate : kCandidates) {
    launch(candidate, sample_node_count);
    if (cudaPeekAtLastError() != cudaSuccess ||
        cudaEventRecord(start, stream) != cudaSuccess) {
      continue;
    }
    for (int repetition = 0; repetition < kRepetitions; ++repetition) {
      launch(candidate, sample_node_count);
    }
    if (cudaPeekAtLastError() != cudaSuccess ||
        cudaEventRecord(stop, stream) != cudaSuccess ||
        cudaEventSynchronize(stop) != cudaSuccess) {
      continue;
    }
    float milliseconds = 0.0f;
    if (cudaEventElapsedTime(&milliseconds, start, stop) == cudaSuccess) {
      milliseconds /= kRepetitions;
      if (milliseconds < best_milliseconds) {
        best_milliseconds = milliseconds;
        best = candidate;
      }
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cache.emplace(key, best);
  return best;
}

}  // namespace deepmd::dpa1_compress_tuning
