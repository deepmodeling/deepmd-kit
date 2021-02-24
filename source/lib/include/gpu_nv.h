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

template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    std::vector<FPTYPE> &host) 
{
  cudaErrcheck(cudaMemcpy(device, &host[0], sizeof(FPTYPE) * host.size(), cudaMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_device_to_host(
    FPTYPE * device, 
    std::vector<FPTYPE> &host) 
{
  cudaErrcheck(cudaMemcpy(&host[0], device, sizeof(FPTYPE) * host.size(), cudaMemcpyDeviceToHost));  
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    std::vector<FPTYPE> &host) 
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
    std::vector<FPTYPE> &host) 
{
  cudaErrcheck(cudaMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
  memcpy_host_to_device(device, host);
}

template <typename FPTYPE>
void delete_device_memory(
    FPTYPE * device) 
{
  cudaErrcheck(cudaFree(device));
}