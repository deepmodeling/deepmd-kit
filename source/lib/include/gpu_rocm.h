#pragma once 
#include <vector>
#include <stdio.h>
#include <assert.h>
#include<hip/hip_runtime.h>
//#include<rocprim/rocprim.hpp>
//#include <hipcub/hipcub.hpp>
#include "errors.h"

#define GPU_MAX_NBOR_SIZE 4096

#define DPErrcheck(res) { DPAssert((res), __FILE__, __LINE__); }
inline void DPAssert(hipError_t code, const char *file, int line, bool abort=true) {
    if (code != hipSuccess) {
        fprintf(stderr,"hip assert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) throw deepmd::deepmd_exception("CUDA Assert");
    }
}

#define nborErrcheck(res) {nborAssert((res), __FILE__, __LINE__);}
inline void nborAssert(hipError_t code, const char *file, int line, bool abort=true) {
    if (code != hipSuccess) {
        fprintf(stderr,"hip assert: %s %s %d\n", "DeePMD-kit:\tillegal nbor list sorting", file, line);
        if (abort) throw deepmd::deepmd_exception("CUDA Assert");
    }
}


namespace deepmd {
inline void DPGetDeviceCount(int &gpu_num) { hipGetDeviceCount(&gpu_num) ;}

inline hipError_t DPSetDevice(int rank) { return  hipSetDevice(rank); }

template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    std::vector<FPTYPE> &host) 
{
  DPErrcheck(hipMemcpy(device, &host[0], sizeof(FPTYPE) * host.size(), hipMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    const FPTYPE * host,
    const int size) 
{
  DPErrcheck(hipMemcpy(device, host, sizeof(FPTYPE) * size, hipMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_device_to_host(
    FPTYPE * device, 
    std::vector<FPTYPE> &host) 
{
  DPErrcheck(hipMemcpy(&host[0], device, sizeof(FPTYPE) * host.size(), hipMemcpyDeviceToHost));  
}
template <typename FPTYPE>
void memcpy_device_to_host(
    const FPTYPE * device, 
    FPTYPE * host,
    const int size) 
{
  DPErrcheck(hipMemcpy(host, device, sizeof(FPTYPE) * size, hipMemcpyDeviceToHost));  
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    std::vector<FPTYPE> &host) 
{
  DPErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    const int size) 
{
  DPErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * size));
}

template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    std::vector<FPTYPE> &host) 
{
  DPErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
  memcpy_host_to_device(device, host);
}
template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    const FPTYPE * host,
    const int size)
{
  DPErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * size));
  memcpy_host_to_device(device, host, size);
}

template <typename FPTYPE>
void delete_device_memory(
    FPTYPE * &device) 
{
  if (device != NULL) {
    DPErrcheck(hipFree(device));
  }
}

template <typename FPTYPE>
void memset_device_memory(
  FPTYPE * device,
  const FPTYPE var,
  const int size)
  {
    DPErrcheck(hipMemset(device,var,sizeof(FPTYPE)*size));
  }
}



