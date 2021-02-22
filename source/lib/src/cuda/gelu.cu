#include "DeviceFunctor.h"
#include "gpu_nv.h"

template <typename FPTYPE>
__global__ void gelu(const FPTYPE * in, FPTYPE * out, int const size) {
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {return;}

    out[idx] = in[idx] * 0.5 * (1.0 + tanh(SQRT_2_PI * (in[idx] + 0.044715 * in[idx] * in[idx] *in[idx])));
}

template <typename FPTYPE>
__global__ void gelu_grad(const FPTYPE * dy, const FPTYPE * in, FPTYPE * out, int const size) {
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {return;}

    // out[idx] = in[idx] * 0.5 * (1.0 + tanh(SQRT_2_PI * (in[idx] + 0.044715 * in[idx] * in[idx] *in[idx])));
    FPTYPE const var1 = tanh(SQRT_2_PI * (in[idx] + 0.044715 * in[idx] * in[idx] *in[idx]));
    out[idx] = dy[idx] * (0.5 * SQRT_2_PI * in[idx] * (1 - var1 * var1) * (0.134145 * in[idx] * in[idx] + 1) + 0.5 * var1 + 0.5);
}

template <typename FPTYPE>
__global__ void gelu_grad_grad(const FPTYPE * dy, const FPTYPE * dy_, const FPTYPE * in, FPTYPE * out, int const size) {
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {return;}

    // out[idx] = in[idx] * 0.5 * (1.0 + tanh(SQRT_2_PI * (in[idx] + 0.044715 * in[idx] * in[idx] *in[idx])));
    FPTYPE const var1 = tanh(SQRT_2_PI * (in[idx] + 0.044715 * in[idx] * in[idx] *in[idx]));
    FPTYPE const var2 = SQRT_2_PI * (1 - var1 * var1) * (0.134145 * in[idx] * in[idx] + 1);
    
	out[idx] = dy[idx] * dy_[idx] * (0.134145 * SQRT_2_PI * in[idx] * in[idx] * (1 - var1 * var1) - SQRT_2_PI * in[idx] * var2 * (0.134145 * in[idx] * in[idx] + 1) * var1 + var2);
}

void GeluLauncher(const float * in, float * out, int const size) {
    int const THREAD_ITEMS = 1024;
    int const BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

    gelu<<<BLOCK_NUMS, THREAD_ITEMS>>>(in, out, size);
}

void GeluLauncher(const double * in, double * out, int const size) {
    int const THREAD_ITEMS = 1024;
    int const BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

    gelu<<<BLOCK_NUMS, THREAD_ITEMS>>>(in, out, size);
}

void GeluGradLauncher(const float * dy, const float * in, float * out, int const size) {
    int const THREAD_ITEMS = 1024;
    int const BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

    gelu_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(dy, in, out, size);
}

void GeluGradLauncher(const double * dy, const double * in, double * out, int const size) {
    int const THREAD_ITEMS = 1024;
    int const BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

    gelu_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(dy, in, out, size);
}

void GeluGradGradLauncher(const float * dy, const float * dy_, const float * in, float * out, int const size) {
    int const THREAD_ITEMS = 1024;
    int const BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

    gelu_grad_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(dy, dy_, in, out, size);
}

void GeluGradGradLauncher(const double * dy, const double * dy_, const double * in, double * out, int const size) {
    int const THREAD_ITEMS = 1024;
    int const BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

    gelu_grad_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(dy, dy_, in, out, size);
}

template <typename FPTYPE>
void GeluGPUExecuteFunctor<FPTYPE>::operator()(const FPTYPE * in, FPTYPE * out, int const size) {
    int const THREAD_ITEMS = 1024;
    int const BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

    gelu<<<BLOCK_NUMS, THREAD_ITEMS>>>(in, out, size);
}

template <typename FPTYPE>
void GeluGradGPUExecuteFunctor<FPTYPE>::operator()(const FPTYPE * dy, const FPTYPE * in, FPTYPE * out, int const size) {
    int const THREAD_ITEMS = 1024;
    int const BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

    gelu_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(dy, in, out, size);
}
 
template <typename FPTYPE>
void GeluGradGradGPUExecuteFunctor<FPTYPE>::operator()(const FPTYPE * dy, const FPTYPE * dy_, const FPTYPE * in, FPTYPE * out, int const size) {
    int const THREAD_ITEMS = 1024;
    int const BLOCK_NUMS = (size + THREAD_ITEMS - 1) / THREAD_ITEMS;

    gelu_grad_grad<<<BLOCK_NUMS, THREAD_ITEMS>>>(dy, dy_, in, out, size);
}

template struct GeluGPUExecuteFunctor<float>;
template struct GeluGPUExecuteFunctor<double>;
template struct GeluGradGPUExecuteFunctor<float>;
template struct GeluGradGPUExecuteFunctor<double>;
template struct GeluGradGradGPUExecuteFunctor<float>;
template struct GeluGradGradGPUExecuteFunctor<double>;