#pragma once
#include <vector>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "errors.h"

#define GPU_MAX_NBOR_SIZE 4096
#define DPErrcheck(res) {DPAssert((res), __FILE__, __LINE__);}
inline void DPAssert(cudaError_t code, const char *file, int line, bool abort=true) 
{
  if (code != cudaSuccess) {
    fprintf(stderr,"cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (code == 2) {
      // out of memory
      fprintf(stderr, "Your memory is not enough, thus an error has been raised " \
        "above. You need to take the following actions:\n" \
        "1. Check if the network size of the model is too large.\n" \
        "2. Check if the batch size of training or testing is too large. " \
        "You can set the training batch size to `auto`.\n" \
        "3. Check if the number of atoms is too large.\n" \
        "4. Check if another program is using the same GPU by execuating `nvidia-smi`. " \
        "The usage of GPUs is controlled by `CUDA_VISIBLE_DEVICES` " \
        "environment variable.\n");
      if (abort) throw deepmd::deepmd_exception_oom("CUDA Assert");
    }
    if (abort) throw deepmd::deepmd_exception("CUDA Assert");
  }
}

#define nborErrcheck(res) {nborAssert((res), __FILE__, __LINE__);}
inline void nborAssert(cudaError_t code, const char *file, int line, bool abort=true) 
{
    if (code != cudaSuccess) {
        fprintf(stderr,"cuda assert: %s %s %d\n", "DeePMD-kit:\tillegal nbor list sorting", file, line);
        if (code == 2) {
          // out of memory
          fprintf(stderr, "Your memory is not enough, thus an error has been raised " \
            "above. You need to take the following actions:\n" \
            "1. Check if the network size of the model is too large.\n" \
            "2. Check if the batch size of training or testing is too large. " \
            "You can set the training batch size to `auto`.\n" \
            "3. Check if the number of atoms is too large.\n" \
            "4. Check if another program is using the same GPU by execuating `nvidia-smi`. " \
            "The usage of GPUs is controlled by `CUDA_VISIBLE_DEVICES` " \
            "environment variable.\n");
            if (abort) throw deepmd::deepmd_exception_oom("CUDA Assert");
        }
        if (abort) throw deepmd::deepmd_exception("CUDA Assert");
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(
    double* address, 
    double val) 
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
          __double_as_longlong(val + __longlong_as_double(assumed)));
  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

namespace deepmd {
  
inline void DPGetDeviceCount(int &gpu_num) { cudaGetDeviceCount(&gpu_num) ;}

inline cudaError_t DPSetDevice(int rank) { return  cudaSetDevice(rank); }

template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    const std::vector<FPTYPE> &host) 
{
  DPErrcheck(cudaMemcpy(device, &host[0], sizeof(FPTYPE) * host.size(), cudaMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    const FPTYPE * host,
    const int size) 
{
  DPErrcheck(cudaMemcpy(device, host, sizeof(FPTYPE) * size, cudaMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_device_to_host(
    const FPTYPE * device, 
    std::vector<FPTYPE> &host) 
{
  DPErrcheck(cudaMemcpy(&host[0], device, sizeof(FPTYPE) * host.size(), cudaMemcpyDeviceToHost));  
}

template <typename FPTYPE>
void memcpy_device_to_host(
    const FPTYPE * device, 
    FPTYPE * host,
    const int size) 
{
  DPErrcheck(cudaMemcpy(host, device, sizeof(FPTYPE) * size, cudaMemcpyDeviceToHost));  
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    const std::vector<FPTYPE> &host) 
{
  DPErrcheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    const int size) 
{
  DPErrcheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * size));
}

template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    const std::vector<FPTYPE> &host) 
{
  DPErrcheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
  memcpy_host_to_device(device, host);
}

template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    const FPTYPE * host,
    const int size)
{
  DPErrcheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * size));
  memcpy_host_to_device(device, host, size);
}

template <typename FPTYPE>
void delete_device_memory(
    FPTYPE * &device) 
{
  if (device != NULL) {
    DPErrcheck(cudaFree(device));
  }
}

template <typename FPTYPE>
void memset_device_memory(
    FPTYPE * device, 
    const FPTYPE var,
    const int size) 
{
  DPErrcheck(cudaMemset(device, var, sizeof(FPTYPE) * size));  
}
} // end of namespace deepmd