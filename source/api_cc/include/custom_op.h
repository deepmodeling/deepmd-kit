#include <vector>
#include <string>
#include <iostream>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#ifdef HIGH_PREC
typedef double VALUETYPE;
typedef double ENERGYTYPE;
#else 
typedef float  VALUETYPE;
typedef double ENERGYTYPE;
#endif

typedef double boxtensor_t ;
typedef double compute_t;

// functions used in custom ops
struct DeviceFunctor {
  void operator()(
      std::string& device, 
      const CPUDevice& d) 
  {
    device = "CPU";
  }
#if GOOGLE_CUDA
  void operator()(
      std::string& device, 
      const GPUDevice& d) 
  {
    device = "GPU";
  }
#endif // GOOGLE_CUDA
};