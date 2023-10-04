// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <string>
#include <vector>

#include "errors.h"

#define gpuGetLastError cudaGetLastError
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemset cudaMemset

#define GPU_MAX_NBOR_SIZE 4096
#define DPErrcheck(res) \
  { DPAssert((res), __FILE__, __LINE__); }
inline void DPAssert(cudaError_t code,
                     const char *file,
                     int line,
                     bool abort = true) {
  if (code != cudaSuccess) {
    std::string error_msg = "CUDA Runtime library throws an error: " +
                            std::string(cudaGetErrorString(code)) +
                            ", in file " + std::string(file) + ": " +
                            std::to_string(line);
    if (code == 2) {
      // out of memory
      error_msg +=
          "\nYour memory is not enough, thus an error has been raised "
          "above. You need to take the following actions:\n"
          "1. Check if the network size of the model is too large.\n"
          "2. Check if the batch size of training or testing is too large. "
          "You can set the training batch size to `auto`.\n"
          "3. Check if the number of atoms is too large.\n"
          "4. Check if another program is using the same GPU by execuating "
          "`nvidia-smi`. "
          "The usage of GPUs is controlled by `CUDA_VISIBLE_DEVICES` "
          "environment variable.";
      if (abort) {
        throw deepmd::deepmd_exception_oom(error_msg);
      }
    }
    if (abort) {
      throw deepmd::deepmd_exception(error_msg);
    } else {
      fprintf(stderr, "%s\n", error_msg.c_str());
    }
  }
}

#define nborErrcheck(res) \
  { nborAssert((res), __FILE__, __LINE__); }
inline void nborAssert(cudaError_t code,
                       const char *file,
                       int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    std::string error_msg = "DeePMD-kit: Illegal nbor list sorting: ";
    try {
      DPAssert(code, file, line, true);
    } catch (deepmd::deepmd_exception_oom &e) {
      error_msg += e.what();
      if (abort) {
        throw deepmd::deepmd_exception_oom(error_msg);
      } else {
        fprintf(stderr, "%s\n", error_msg.c_str());
      }
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN) } while (assumed != old);
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

namespace deepmd {

inline void DPGetDeviceCount(int &gpu_num) { cudaGetDeviceCount(&gpu_num); }

inline cudaError_t DPSetDevice(int rank) { return cudaSetDevice(rank); }

template <typename FPTYPE>
void memcpy_host_to_device(FPTYPE *device, const std::vector<FPTYPE> &host) {
  DPErrcheck(cudaMemcpy(device, &host[0], sizeof(FPTYPE) * host.size(),
                        cudaMemcpyHostToDevice));
}

template <typename FPTYPE>
void memcpy_host_to_device(FPTYPE *device, const FPTYPE *host, const int size) {
  DPErrcheck(
      cudaMemcpy(device, host, sizeof(FPTYPE) * size, cudaMemcpyHostToDevice));
}

template <typename FPTYPE>
void memcpy_device_to_host(const FPTYPE *device, std::vector<FPTYPE> &host) {
  DPErrcheck(cudaMemcpy(&host[0], device, sizeof(FPTYPE) * host.size(),
                        cudaMemcpyDeviceToHost));
}

template <typename FPTYPE>
void memcpy_device_to_host(const FPTYPE *device, FPTYPE *host, const int size) {
  DPErrcheck(
      cudaMemcpy(host, device, sizeof(FPTYPE) * size, cudaMemcpyDeviceToHost));
}

template <typename FPTYPE>
void malloc_device_memory(FPTYPE *&device, const std::vector<FPTYPE> &host) {
  DPErrcheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
}

template <typename FPTYPE>
void malloc_device_memory(FPTYPE *&device, const int size) {
  DPErrcheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * size));
}

template <typename FPTYPE>
void malloc_device_memory_sync(FPTYPE *&device,
                               const std::vector<FPTYPE> &host) {
  DPErrcheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
  memcpy_host_to_device(device, host);
}

template <typename FPTYPE>
void malloc_device_memory_sync(FPTYPE *&device,
                               const FPTYPE *host,
                               const int size) {
  DPErrcheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * size));
  memcpy_host_to_device(device, host, size);
}

template <typename FPTYPE>
void delete_device_memory(FPTYPE *&device) {
  if (device != NULL) {
    DPErrcheck(cudaFree(device));
  }
}

template <typename FPTYPE>
void memset_device_memory(FPTYPE *device, const int var, const int size) {
  DPErrcheck(cudaMemset(device, var, sizeof(FPTYPE) * size));
}
}  // end of namespace deepmd
