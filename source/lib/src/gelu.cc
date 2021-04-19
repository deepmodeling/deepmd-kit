#include "gelu.h"
#include "math.h"
#include "device.h"

template<typename FPTYPE>
void deepmd::gelu_cpu(
    FPTYPE * out, 
    const FPTYPE * xx, 
    const int size)
{
  for (int ii = 0; ii < size; ii++) {
    out[ii] = xx[ii] * 0.5 * (1.0 + tanh(SQRT_2_PI * (xx[ii] + 0.044715 * xx[ii] * xx[ii] *xx[ii])));
  }
}

template<typename FPTYPE>
void deepmd::gelu_grad_cpu(
    FPTYPE * out, 
    const FPTYPE * xx,
    const FPTYPE * dy, 
    const int size)
{
  for (int ii = 0; ii < size; ii++) {
    const FPTYPE var = tanh(SQRT_2_PI * (xx[ii] + 0.044715 * xx[ii] * xx[ii] * xx[ii]));
    out[ii] = dy[ii] * (0.5 * SQRT_2_PI * xx[ii] * (1 - var * var) * (0.134145 * xx[ii] * xx[ii] + 1) + 0.5 * var + 0.5);
  }
}

template<typename FPTYPE>
void deepmd::gelu_grad_grad_cpu(
    FPTYPE * out,
    const FPTYPE * xx,
    const FPTYPE * dy, 
    const FPTYPE * dy_2,
    const int size) 
{
  for (int ii = 0; ii < size; ii++) {
    const FPTYPE var1 = tanh(SQRT_2_PI * (xx[ii] + 0.044715 * xx[ii] * xx[ii] *xx[ii]));
    const FPTYPE var2 = SQRT_2_PI * (1 - var1 * var1) * (0.134145 * xx[ii] * xx[ii] + 1);
	out[ii] = dy[ii] * dy_2[ii] * (0.134145 * SQRT_2_PI * xx[ii] * xx[ii] * (1 - var1 * var1) - SQRT_2_PI * xx[ii] * var2 * (0.134145 * xx[ii] * xx[ii] + 1) * var1 + var2);
  }
}

template void deepmd::gelu_cpu<float>(float * out, const float * x, const int size);
template void deepmd::gelu_cpu<double>(double * out, const double * x, const int size);
template void deepmd::gelu_grad_cpu<float>(float * out, const float * x, const float * dy, const int size);
template void deepmd::gelu_grad_cpu<double>(double * out, const double * x, const double * dy, const int size);
template void deepmd::gelu_grad_grad_cpu<float>(float * out, const float * x, const float * dy, const float * dy_2, const int size);
template void deepmd::gelu_grad_grad_cpu<double>(double * out, const double * x, const double * dy, const double * dy_2, const int size);
