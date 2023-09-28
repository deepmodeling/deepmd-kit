// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <assert.h>
#include <hip/hip_runtime.h>
#include <stdio.h>

#include <string>
#include <vector>
// #include<rocprim/rocprim.hpp>
// #include <hipcub/hipcub.hpp>
#include "errors.h"

#define GPU_MAX_NBOR_SIZE 4096

#define gpuGetLastError hipGetLastError
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuMemcpy hipMemcpy
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemset hipMemset

#define DPErrcheck(res) \
  { DPAssert((res), __FILE__, __LINE__); }
inline void DPAssert(hipError_t code,
                     const char *file,
                     int line,
                     bool abort = true) {
  if (code != hipSuccess) {
    std::string error_msg = "HIP runtime library throws an error: " +
                            std::string(hipGetErrorString(code)) +
                            ", in file " + std::string(file) + ": " +
                            std::to_string(line);
    if (abort) {
      throw deepmd::deepmd_exception(error_msg);
    } else {
      fprintf(stderr, "%s\n", error_msg.c_str());
    }
  }
}

#define nborErrcheck(res) \
  { nborAssert((res), __FILE__, __LINE__); }
inline void nborAssert(hipError_t code,
                       const char *file,
                       int line,
                       bool abort = true) {
  if (code != hipSuccess) {
    std::string error_msg = "DeePMD-kit: Illegal nbor list sorting: ";
    try {
      DPAssert(code, file, line, true);
    } catch (deepmd::deepmd_exception &e) {
      error_msg += e.what();
      if (abort) {
        throw deepmd::deepmd_exception(error_msg);
      } else {
        fprintf(stderr, "%s\n", error_msg.c_str());
      }
    }
  }
}

namespace deepmd {
inline void DPGetDeviceCount(int &gpu_num) { hipGetDeviceCount(&gpu_num); }

inline hipError_t DPSetDevice(int rank) { return hipSetDevice(rank); }

template <typename FPTYPE>
void memcpy_host_to_device(FPTYPE *device, std::vector<FPTYPE> &host) {
  DPErrcheck(hipMemcpy(device, &host[0], sizeof(FPTYPE) * host.size(),
                       hipMemcpyHostToDevice));
}

template <typename FPTYPE>
void memcpy_host_to_device(FPTYPE *device, const FPTYPE *host, const int size) {
  DPErrcheck(
      hipMemcpy(device, host, sizeof(FPTYPE) * size, hipMemcpyHostToDevice));
}

template <typename FPTYPE>
void memcpy_device_to_host(const FPTYPE *device, std::vector<FPTYPE> &host) {
  DPErrcheck(hipMemcpy(&host[0], device, sizeof(FPTYPE) * host.size(),
                       hipMemcpyDeviceToHost));
}
template <typename FPTYPE>
void memcpy_device_to_host(const FPTYPE *device, FPTYPE *host, const int size) {
  DPErrcheck(
      hipMemcpy(host, device, sizeof(FPTYPE) * size, hipMemcpyDeviceToHost));
}

template <typename FPTYPE>
void malloc_device_memory(FPTYPE *&device, std::vector<FPTYPE> &host) {
  DPErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
}

template <typename FPTYPE>
void malloc_device_memory(FPTYPE *&device, const int size) {
  DPErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * size));
}

template <typename FPTYPE>
void malloc_device_memory_sync(FPTYPE *&device, std::vector<FPTYPE> &host) {
  DPErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
  memcpy_host_to_device(device, host);
}
template <typename FPTYPE>
void malloc_device_memory_sync(FPTYPE *&device,
                               const FPTYPE *host,
                               const int size) {
  DPErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * size));
  memcpy_host_to_device(device, host, size);
}

template <typename FPTYPE>
void delete_device_memory(FPTYPE *&device) {
  if (device != NULL) {
    DPErrcheck(hipFree(device));
  }
}

template <typename FPTYPE>
void memset_device_memory(FPTYPE *device, const int var, const int size) {
  DPErrcheck(hipMemset(device, var, sizeof(FPTYPE) * size));
}
}  // namespace deepmd
