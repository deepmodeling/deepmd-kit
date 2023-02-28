/*
  dynamically load CUDA runtime library
*/
#include <dlfcn.h>
#include <fcntl.h>

#include <iostream>
#include <string>

#include "cuda_runtime_api.h"

// wraps cuda runtime with dso loader

namespace {
void *GetDsoHandle() {
  static auto handle = []() -> void * {
#if defined(__gnu_linux__)
    std::string libname = "libcudart.so";
#elif defined(__APPLE__)
    std::string libname = "libcudart.dylib";
#elif defined(_WIN32)
    std::string libname = "cudart.dll";
#endif
#if defined(_WIN32)
    void *dso_handle = LoadLibrary(libname.c_str());
#else
    void *dso_handle = dlopen(libname.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (!dso_handle) {
      std::cerr << "DeePMD-kit: Cannot find " << libname << std::endl;
      return nullptr;
    }
    std::cerr << "DeePMD-kit: Successfully load " << libname << std::endl;
    return dso_handle;
  }();
  return handle;
}

template <typename T>
T LoadSymbol(const char *symbol_name) {
  void *symbol = nullptr;
  void *handle = GetDsoHandle();
  if (handle) {
    symbol = dlsym(handle, symbol_name);
  }
  return reinterpret_cast<T>(symbol);
}

// the following is copied from TensorFlow
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

cudaError_t GetSymbolNotFoundError() {
  return cudaErrorSharedObjectSymbolNotFound;
}
}  // namespace

#define __dv(v)
#define __CUDA_DEPRECATED
// CUDART_VERSION is defined in cuda_runtime_api.h
#if CUDART_VERSION < 10000
#include "cuda_runtime_9_0.inc"
#elif CUDART_VERSION < 10010
#include "cuda_runtime_10_0.inc"
#elif CUDART_VERSION < 10020
#include "cuda_runtime_10_1.inc"
#elif CUDART_VERSION < 11000
#include "cuda_runtime_10_2.inc"
#elif CUDART_VERSION < 11020
#include "cuda_runtime_11_0.inc"
#elif CUDART_VERSION < 11080
#include "cuda_runtime_11_2.inc"
#elif CUDART_VERSION < 12000
#include "cuda_runtime_11_8.inc"
#else
#include "cuda_runtime_12_0.inc"
#endif
#undef __dv
#undef __CUDA_DEPRECATED

extern "C" {

// Following are private symbols in libcudart that got inserted by nvcc.
extern void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle,
                                             const char *hostFun,
                                             char *deviceFun,
                                             const char *deviceName,
                                             int thread_limit,
                                             uint3 *tid,
                                             uint3 *bid,
                                             dim3 *bDim,
                                             dim3 *gDim,
                                             int *wSize) {
  using FuncPtr = void(CUDARTAPI *)(void **fatCubinHandle, const char *hostFun,
                                    char *deviceFun, const char *deviceName,
                                    int thread_limit, uint3 *tid, uint3 *bid,
                                    dim3 *bDim, dim3 *gDim, int *wSize);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaRegisterFunction");
  if (!func_ptr) return;
  func_ptr(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
           bid, bDim, gDim, wSize);
}

extern void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle) {
  using FuncPtr = void(CUDARTAPI *)(void **fatCubinHandle);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaUnregisterFatBinary");
  if (!func_ptr) return;
  func_ptr(fatCubinHandle);
}

extern void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle,
                                        char *hostVar,
                                        char *deviceAddress,
                                        const char *deviceName,
                                        int ext,
                                        size_t size,
                                        int constant,
                                        int global) {
  using FuncPtr = void(CUDARTAPI *)(
      void **fatCubinHandle, char *hostVar, char *deviceAddress,
      const char *deviceName, int ext, size_t size, int constant, int global);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaRegisterVar");
  if (!func_ptr) return;
  func_ptr(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size,
           constant, global);
}

extern void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
  using FuncPtr = void **(CUDARTAPI *)(void *fatCubin);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaRegisterFatBinary");
  if (!func_ptr) return nullptr;
  return (void **)func_ptr(fatCubin);
}

extern cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim,
                                                        dim3 *blockDim,
                                                        size_t *sharedMem,
                                                        void *stream) {
  using FuncPtr = cudaError_t(CUDARTAPI *)(dim3 * gridDim, dim3 * blockDim,
                                           size_t * sharedMem, void *stream);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaPopCallConfiguration");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(gridDim, blockDim, sharedMem, stream);
}

extern __host__ __device__ unsigned CUDARTAPI __cudaPushCallConfiguration(
    dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void *stream = 0) {
  using FuncPtr = unsigned(CUDARTAPI *)(dim3 gridDim, dim3 blockDim,
                                        size_t sharedMem, void *stream);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaPushCallConfiguration");
  if (!func_ptr) return 0;
  return func_ptr(gridDim, blockDim, sharedMem, stream);
}

extern char CUDARTAPI __cudaInitModule(void **fatCubinHandle) {
  using FuncPtr = char(CUDARTAPI *)(void **fatCubinHandle);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaInitModule");
  if (!func_ptr) return 0;
  return func_ptr(fatCubinHandle);
}

#if CUDART_VERSION >= 10010
extern void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  using FuncPtr = void(CUDARTAPI *)(void **fatCubinHandle);
  static auto func_ptr = LoadSymbol<FuncPtr>("__cudaRegisterFatBinaryEnd");
  if (!func_ptr) return;
  func_ptr(fatCubinHandle);
}
#endif
}
