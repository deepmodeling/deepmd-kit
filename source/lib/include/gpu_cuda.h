#pragma once
#include <vector>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define GPU_MAX_NBOR_SIZE 4096
#define cudaErrcheck(res) {cudaAssert((res), __FILE__, __LINE__);}
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
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
template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    const std::vector<FPTYPE> &host) 
{
  cudaErrcheck(cudaMemcpy(device, &host[0], sizeof(FPTYPE) * host.size(), cudaMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    const FPTYPE * host,
    const int size) 
{
  cudaErrcheck(cudaMemcpy(device, host, sizeof(FPTYPE) * size, cudaMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_device_to_host(
    const FPTYPE * device, 
    std::vector<FPTYPE> &host) 
{
  cudaErrcheck(cudaMemcpy(&host[0], device, sizeof(FPTYPE) * host.size(), cudaMemcpyDeviceToHost));  
}

template <typename FPTYPE>
void memcpy_device_to_host(
    const FPTYPE * device, 
    FPTYPE * host,
    const int size) 
{
  cudaErrcheck(cudaMemcpy(host, device, sizeof(FPTYPE) * size, cudaMemcpyDeviceToHost));  
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    const std::vector<FPTYPE> &host) 
{
  cudaErrcheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    const int size) 
{
  cudaErrcheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * size));
}

template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    const std::vector<FPTYPE> &host) 
{
  cudaErrcheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
  memcpy_host_to_device(device, host);
}

template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    const FPTYPE * host,
    const int size)
{
  cudaErrcheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * size));
  memcpy_host_to_device(device, host, size);
}

template <typename FPTYPE>
void delete_device_memory(
    FPTYPE * &device) 
{
  if (device != NULL) {
    cudaErrcheck(cudaFree(device));
  }
}

template <typename FPTYPE>
void memset_device_memory(
    FPTYPE * device, 
    const FPTYPE var,
    const int size) 
{
  cudaErrcheck(cudaMemset(device, var, sizeof(FPTYPE) * size));  
}
} // end of namespace deepmd