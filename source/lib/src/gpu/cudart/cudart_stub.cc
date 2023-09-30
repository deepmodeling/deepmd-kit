// SPDX-License-Identifier: LGPL-3.0-or-later
/*
  dynamically load CUDA runtime library
*/
#include <dlfcn.h>
#include <fcntl.h>

#include <iostream>
#include <string>

#include "cuda_runtime_api.h"

extern "C" {

static cudaError_t DP_CudartGetSymbolNotFoundError() {
  return cudaErrorSharedObjectSymbolNotFound;
}

void *DP_cudart_dlopen(char *libname) {
  static auto handle = [](std::string libname) -> void * {
#if defined(_WIN32)
    void *dso_handle = LoadLibrary(libname.c_str());
#else
    void *dso_handle = dlopen(libname.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (!dso_handle) {
      std::cerr << "DeePMD-kit: Cannot find " << libname << std::endl;
      // fake a handle to avoid crash
      return reinterpret_cast<void *>(0x1);
    }
    std::cerr << "DeePMD-kit: Successfully load " << libname << std::endl;
    return dso_handle;
  }(std::string(libname));
  return handle;
}

void *DP_cudart_dlsym(void *handle, const char *sym_name) {
  // check if the fake handle, if so, return a function that
  // returns cudaErrorSharedObjectSymbolNotFound
  if (!handle || reinterpret_cast<uintptr_t>(handle) == 0x1) {
    return reinterpret_cast<void *>(&DP_CudartGetSymbolNotFoundError);
  }
  void *symbol = dlsym(handle, sym_name);
  if (!symbol) {
    return reinterpret_cast<void *>(&DP_CudartGetSymbolNotFoundError);
  }
  return symbol;
};
}
