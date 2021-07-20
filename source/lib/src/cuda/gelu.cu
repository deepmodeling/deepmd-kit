#include "gelu.h"
#include "device.h"

template <typename FPTYPE>
__global__ void gelu(
    FPTYPE * out, 
    const FPTYPE * xx, 
    int const size) 
{
  int const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  out[idx] = xx[idx] * 0.5 * (1.0 + tanh(SQRT_2_PI * (xx[idx] + 0.044715 * xx[idx] * xx[idx] *xx[idx])));
}

template <typename FPTYPE>
__global__ void gelu_grad(
    FPTYPE * out, 
    const FPTYPE * xx, 
    const FPTYPE * dy, 
    int const size) 
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  // out[idx] = xx[idx] * 0.5 * (1.0 + tanh(SQRT_2_PI * (xx[idx] + 0.044715 * xx[idx] * xx[idx] *xx[idx])));
  const FPTYPE var = tanh(SQRT_2_PI * (xx[idx] + 0.044715 * xx[idx] * xx[idx] *xx[idx]));
  out[idx] = dy[idx] * (0.5 * SQRT_2_PI * xx[idx] * (1 - var * var) * (0.134145 * xx[idx] * xx[idx] + 1) + 0.5 * var + 0.5);
}

template <typename FPTYPE>
__global__ void gelu_grad_grad(
    FPTYPE * out, 
    const FPTYPE * xx, 
    const FPTYPE * dy, 
    const FPTYPE * dy_2,
    int const size) 
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  // out[idx] = xx[idx] * 0.5 * (1.0 + tanh(SQRT_2_PI * (xx[idx] + 0.044715 * xx[idx] * xx[idx] *xx[idx])));
  const FPTYPE var1 = tanh(SQRT_2_PI * (xx[idx] + 0.044715 * xx[idx] * xx[idx] *xx[idx]));
  const FPTYPE var2 = SQRT_2_PI * (1 - var1 * var1) * (0.134145 * xx[idx] * xx[idx] + 1);
  out[idx] = dy[idx] * dy_2[idx] * (0.134145 * SQRT_2_PI * xx[idx] * xx[idx] * (1 - var1 * var1) - SQRT_2_PI * xx[idx] * var2 * (0.134145 * xx[idx] * xx[idx] + 1) * var1 + var2);
}

namespace deepmd {
template<typename FPTYPE>
void gelu_gpu_cuda(
    FPTYPE * out, 
    const FPTYPE * xx, 
    const int size)
{
  const int THREAD_ITEMS = 1024;
  const int BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

  gelu<<<BLOCK_NUMS, THREAD_ITEMS>>>(out, xx, size);
}

template<typename FPTYPE>
void gelu_grad_gpu_cuda(
    FPTYPE * out, 
    const FPTYPE * xx,
    const FPTYPE * dy, 
    const int size)
{
  const int THREAD_ITEMS = 1024;
  const int BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

  gelu_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(out, xx, dy, size);
}

template<typename FPTYPE>
void gelu_grad_grad_gpu_cuda(
    FPTYPE * out,
    const FPTYPE * xx,
    const FPTYPE * dy, 
    const FPTYPE * dy_2,
    const int size)
{
  const int THREAD_ITEMS = 1024;
  const int BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;
  
  gelu_grad_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(out, xx, dy, dy_2, size);
}

template void gelu_gpu_cuda<float>(float * out, const float * x, const int size);
template void gelu_gpu_cuda<double>(double * out, const double * x, const int size);
template void gelu_grad_gpu_cuda<float>(float * out, const float * x, const float * dy, const int size);
template void gelu_grad_gpu_cuda<double>(double * out, const double * x, const double * dy, const int size);
template void gelu_grad_grad_gpu_cuda<float>(float * out, const float * x, const float * dy, const float * dy_2, const int size);
template void gelu_grad_grad_gpu_cuda<double>(double * out, const double * x, const double * dy, const double * dy_2, const int size);
}