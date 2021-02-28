#pragma once

template<typename FPTYPE>
void gelu_cpu(
    FPTYPE * out, 
    const FPTYPE * x, 
    const int size);

template<typename FPTYPE>
void gelu_grad_cpu(
    FPTYPE * out, 
    const FPTYPE * x,
    const FPTYPE * dy, 
    const int size);

template<typename FPTYPE>
void gelu_grad_grad_cpu(
    FPTYPE * out,
    const FPTYPE * x,
    const FPTYPE * dy, 
    const FPTYPE * dy_2,
    const int size);

#if GOOGLE_CUDA
template<typename FPTYPE>
void gelu_gpu_cuda(
    FPTYPE * out, 
    const FPTYPE * x, 
    const int size);

template<typename FPTYPE>
void gelu_grad_gpu_cuda(
    FPTYPE * out, 
    const FPTYPE * x,
    const FPTYPE * dy, 
    const int size);

template<typename FPTYPE>
void gelu_grad_grad_gpu_cuda(
    FPTYPE * out,
    const FPTYPE * x,
    const FPTYPE * dy, 
    const FPTYPE * dy_2,
    const int size);
#endif // GOOGLE_CUDA
