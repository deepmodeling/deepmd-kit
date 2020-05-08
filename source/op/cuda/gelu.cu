#include <cuda_runtime.h>
#include <stdio.h>

#define SQRT_2_PI 0.7978845608028654 

template <typename T>
__global__ void gelu(const T * in, T * out, int const size) {
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {return;}

    out[idx] = in[idx] * 0.5 * (1.0 + tanh(SQRT_2_PI * (in[idx] + 0.044715 * in[idx] * in[idx] *in[idx])));
}

template <typename T>
__global__ void gelu_grad(const T * dy, const T * in, T * out, int const size) {
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {return;}

    // out[idx] = in[idx] * 0.5 * (1.0 + tanh(SQRT_2_PI * (in[idx] + 0.044715 * in[idx] * in[idx] *in[idx])));
    T const var1 = tanh(SQRT_2_PI * (in[idx] + 0.044715 * in[idx] * in[idx] *in[idx]));
    out[idx] = dy[idx] * (0.5 * SQRT_2_PI * in[idx] * (1 - var1 * var1) * (0.134145 * in[idx] * in[idx] + 1) + 0.5 * var1 + 0.5);
}

template <typename T>
__global__ void gelu_grad_grad(const T * dy, const T * dy_, const T * in, T * out, int const size) {
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {return;}

    // out[idx] = in[idx] * 0.5 * (1.0 + tanh(SQRT_2_PI * (in[idx] + 0.044715 * in[idx] * in[idx] *in[idx])));
    T const var1 = tanh(SQRT_2_PI * (in[idx] + 0.044715 * in[idx] * in[idx] *in[idx]));
    T const var2 = SQRT_2_PI * (1 - var1 * var1) * (0.134145 * in[idx] * in[idx] + 1);
    
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
