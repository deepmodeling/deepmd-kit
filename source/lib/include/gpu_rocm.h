#pragma once 
#include <vector>
#include <stdio.h>
#include <assert.h>
#include<hip/hip_runtime.h>
//#include<rocprim/rocprim.hpp>
//#include <hipcub/hipcub.hpp>

#define GPU_MAX_NBOR_SIZE 4096

#define hipErrcheck(res) { hipAssert((res), __FILE__, __LINE__); }
inline void hipAssert(hipError_t code, const char *file, int line, bool abort=true) {
    if (code != hipSuccess) {
        fprintf(stderr,"hip assert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define nborErrcheck(res) {nborAssert((res), __FILE__, __LINE__);}
inline void nborAssert(hipError_t code, const char *file, int line, bool abort=true) {
    if (code != hipSuccess) {
        fprintf(stderr,"hip assert: %s %s %d\n", "DeePMD-kit:\tillegal nbor list sorting", file, line);
        if (abort) exit(code);
    }
}

namespace deepmd {
template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    std::vector<FPTYPE> &host) 
{
  hipErrcheck(hipMemcpy(device, &host[0], sizeof(FPTYPE) * host.size(), hipMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    const FPTYPE * host,
    const int size) 
{
  hipErrcheck(hipMemcpy(device, host, sizeof(FPTYPE) * size, hipMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_device_to_host(
    FPTYPE * device, 
    std::vector<FPTYPE> &host) 
{
  hipErrcheck(hipMemcpy(&host[0], device, sizeof(FPTYPE) * host.size(), hipMemcpyDeviceToHost));  
}
template <typename FPTYPE>
void memcpy_device_to_host(
    const FPTYPE * device, 
    FPTYPE * host,
    const int size) 
{
  hipErrcheck(hipMemcpy(host, device, sizeof(FPTYPE) * size, hipMemcpyDeviceToHost));  
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    std::vector<FPTYPE> &host) 
{
  hipErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    const int size) 
{
  hipErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * size));
}

template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    std::vector<FPTYPE> &host) 
{
  hipErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
  memcpy_host_to_device(device, host);
}
template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    const FPTYPE * host,
    const int size)
{
  hipErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * size));
  memcpy_host_to_device(device, host, size);
}

template <typename FPTYPE>
void delete_device_memory(
    FPTYPE * &device) 
{
  if (device != NULL) {
    hipErrcheck(hipFree(device));
  }
}

template <typename FPTYPE>
void memset_device_memory(
  FPTYPE * device,
  const FPTYPE var,
  const int size)
  {
    hipErrcheck(hipMemset(device,var,sizeof(FPTYPE)*size));
  }
}



