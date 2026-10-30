// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Ridge point of the current device, which the operators that trade memory
// traffic for float32 SIMT arithmetic compare with their arithmetic intensity.

#pragma once

#include <cuda_runtime.h>

#include <map>
#include <mutex>

/// Ridge point of the current device: peak fp32 FMA throughput over DRAM
/// bandwidth, in FLOP per byte.
///
/// A fused kernel that keeps its intermediate tensors on chip saves memory
/// traffic at the cost of arithmetic efficiency, so whether it pays scales
/// with this ratio: the routing gate of the fused SO(2) convolution normalizes
/// its measured threshold by the ridge of the card it was calibrated on, and
/// the fused fitting network compares the ridge with the arithmetic intensity
/// of the per-layer kernels it replaces. Both decisions thereby follow the
/// actual part rather than an architecture name. Attribute queries cover every
/// supported toolkit; the FMA-per-SM width is 128 lanes on every consumer and
/// data-center part since Ampere except the A100 die (64), which the
/// compute-capability pair identifies. Each device answers once.
inline double fp32_ridge() {
  static std::mutex mutex;
  static std::map<int, double> ridges;
  int dev = 0;
  cudaGetDevice(&dev);
  const std::lock_guard<std::mutex> lock(mutex);
  const auto known = ridges.find(dev);
  if (known != ridges.end()) {
    return known->second;
  }
  int sm_count = 0, clock_khz = 0, mem_clock_khz = 0, bus_width = 0;
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
  cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
  cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  const int lanes = (major < 8 || (major == 8 && minor == 0)) ? 64 : 128;
  const double flops = 2.0 * lanes * sm_count * (clock_khz * 1e3);
  const double bytes = (mem_clock_khz * 1e3) * (bus_width / 8.0) * 2.0;
  const double ridge = bytes > 0.0 ? flops / bytes : 0.0;
  ridges.emplace(dev, ridge);
  return ridge;
}
