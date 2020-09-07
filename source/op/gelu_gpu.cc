#include <algorithm>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define SQRT_2_PI 0.7978845608028654 
REGISTER_OP("Gelu")
    .Attr("T: {float, double}")
    .Input("x: T")
    .Output("output: T");

REGISTER_OP("GeluGrad")
    .Attr("T: {float, double}")
    .Input("dy: T")
    .Input("x: T")
    .Output("output: T");

REGISTER_OP("GeluGradGrad")
    .Attr("T: {float, double}")
    .Input("dy: T")
    .Input("dy_: T")
    .Input("x: T")
    .Output("output: T");

#if GOOGLE_CUDA
// maybe instead use cudnn activation forward 
void GeluGPULauncher(const float * in, float * out, int const size);
void GeluGPULauncher(const double * in, double * out, int const size);

void GeluGradGPULauncher(const float * dy, const float * in, float * out, int const size);
void GeluGradGPULauncher(const double * dy, const double * in, double * out, int const size);

void GeluGradGradGPULauncher(const float * dy, const float * dy_, const float * in, float * out, int const size);
void GeluGradGradGPULauncher(const double * dy, const double * dy_, const double * in, double * out, int const size);
#endif // GOOGLE_CUDA

template<typename T>
void GeluCPULauncher(const T * in, T * out, int const size) {
    for (int ii = 0; ii < size; ii++) {
        out[ii] = in[ii] * 0.5 * (1.0 + tanh(SQRT_2_PI * (in[ii] + 0.044715 * in[ii] * in[ii] *in[ii])));
    }
}

template<typename T>
void GeluGradCPULauncher(const T * dy, const T * in, T * out, int const size) {
    for (int ii = 0; ii < size; ii++) {
        T const var1 = tanh(SQRT_2_PI * (in[ii] + 0.044715 * in[ii] * in[ii] *in[ii]));
        out[ii] = dy[ii] * (0.5 * SQRT_2_PI * in[ii] * (1 - var1 * var1) * (0.134145 * in[ii] * in[ii] + 1) + 0.5 * var1 + 0.5);
    }
}

template <typename T>
void GeluGradGradCPULauncher(const T * dy, const T * dy_, const T * in, T * out, int const size) {
    for (int ii = 0; ii < size; ii++) {
        T const var1 = tanh(SQRT_2_PI * (in[ii] + 0.044715 * in[ii] * in[ii] *in[ii]));
        T const var2 = SQRT_2_PI * (1 - var1 * var1) * (0.134145 * in[ii] * in[ii] + 1);
	    out[ii] = dy[ii] * dy_[ii] * (0.134145 * SQRT_2_PI * in[ii] * in[ii] * (1 - var1 * var1) - SQRT_2_PI * in[ii] * var2 * (0.134145 * in[ii] * in[ii] + 1) * var1 + var2);
    }
}

template <typename Device, typename T>
struct GeluFunctor {
    void operator()(const CPUDevice& d, const T * in, T * out, int const size) {
		GeluCPULauncher(in, out, size);
	}
    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const T * in, T * out, int const size) {
        GeluGPULauncher(in, out, size);
    }
    #endif
};

template <typename Device, typename T>
struct GeluGradFunctor {
    void operator()(const CPUDevice& d, const T * dy, const T * in, T * out, int const size) {
        GeluGradCPULauncher(dy, in, out, size);
    }
    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const T * dy, const T * in, T * out, int const size) {
        GeluGradGPULauncher(dy, in, out, size);
    }
    #endif
};

template <typename Device, typename T>
struct GeluGradGradFunctor {
    void operator()(const CPUDevice& d, const T * dy, const T * dy_, const T * in, T * out, int const size) {
        GeluGradGradCPULauncher(dy, dy_, in, out, size);
    }
    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const T * dy, const T * dy_, const T * in, T * out, int const size) {
        GeluGradGradGPULauncher(dy, dy_, in, out, size);
    }
    #endif
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class GeluOp : public OpKernel {
  public :
    explicit GeluOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& x = context->input(0);
        Tensor * output = NULL;
        int context_output_index = 0;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
					    x.shape(),
					    &output));
		
		GeluFunctor<Device, T>()(
			context->eigen_device<Device>(),
			x.flat<T>().data(),
			output->flat<T>().data(),
			static_cast<int>(output->NumElements())
		);
    }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class GeluGradOp : public OpKernel {
  public :
    explicit GeluGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& dy = context->input(0);
        const Tensor& x  = context->input(1);
        
        Tensor * output = NULL;
        int context_output_index = 0;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
					    x.shape(),
					    &output));
		
		GeluGradFunctor<Device, T>()(
            context->eigen_device<Device>(),
            dy.flat<T>().data(),
            x.flat<T>().data(),
            output->flat<T>().data(),
            static_cast<int>(output->NumElements())
        );
    }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class GeluGradGradOp : public OpKernel {
  public :
    explicit GeluGradGradOp(OpKernelConstruction* context) : OpKernel(context) {}
	
    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& dy = context->input(0);
        const Tensor& dy_ = context->input(1);
        const Tensor& x  = context->input(2);
	
		Tensor * output = NULL;
		int context_output_index = 0;	
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
					    x.shape(),
					    &output));
		
		GeluGradGradFunctor<Device, T>()(
            context->eigen_device<Device>(),
            dy.flat<T>().data(),
            dy_.flat<T>().data(),
            x.flat<T>().data(),
            output->flat<T>().data(),
            static_cast<int>(output->NumElements())
        );
    }
};

#define REGISTER_CPU(T)                                                 \
/* Declare explicit instantiations in kernel_example.cu.cc. */          \
REGISTER_KERNEL_BUILDER(                                                \
    Name("Gelu").Device(DEVICE_CPU).TypeConstraint<T>("T"),             \
    GeluOp<CPUDevice, T>);                                              \
/* Declare explicit instantiations in kernel_example.cu.cc. */          \
REGISTER_KERNEL_BUILDER(                                                \
    Name("GeluGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),         \
    GeluGradOp<CPUDevice, T>);                                          \
/* Declare explicit instantiations in kernel_example.cu.cc. */          \
REGISTER_KERNEL_BUILDER(                                                \
    Name("GeluGradGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
    GeluGradGradOp<CPUDevice, T>);                                      
REGISTER_CPU(float);
REGISTER_CPU(double);

#if GOOGLE_CUDA
#define REGISTER_GPU(T)                                                 \
    /* Declare explicit instantiations in kernel_example.cu.cc. */          \
    REGISTER_KERNEL_BUILDER(                                                \
        Name("Gelu").Device(DEVICE_GPU).TypeConstraint<T>("T"),             \
        GeluOp<GPUDevice, T>);                                              \
    /* Declare explicit instantiations in kernel_example.cu.cc. */          \
    REGISTER_KERNEL_BUILDER(                                                \
        Name("GeluGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"),         \
        GeluGradOp<GPUDevice, T>);                                          \
    /* Declare explicit instantiations in kernel_example.cu.cc. */          \
    REGISTER_KERNEL_BUILDER(                                                \
        Name("GeluGradGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
        GeluGradGradOp<GPUDevice, T>);                                      
    REGISTER_GPU(float);
    REGISTER_GPU(double);
#endif // GOOGLE_CUDA