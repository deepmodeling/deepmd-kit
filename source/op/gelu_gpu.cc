#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

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

// maybe instead use cudnn activation forward 
void GeluLauncher(const float * in, float * out, int const size);
void GeluLauncher(const double * in, double * out, int const size);

void GeluGradLauncher(const float * dy, const float * in, float * out, int const size);
void GeluGradLauncher(const double * dy, const double * in, double * out, int const size);

void GeluGradGradLauncher(const float * dy, const float * dy_, const float * in, float * out, int const size);
void GeluGradGradLauncher(const double * dy, const double * dy_, const double * in, double * out, int const size);

template <typename Device, typename T>
struct GeluFunctor {
    void operator()(const Device& d, const T * in, T * out, int const size) {
		GeluLauncher(in, out, size);
	}
};

template <typename Device, typename T>
struct GeluGradFunctor {
    void operator()(const Device& d, const T * dy, const T * in, T * out, int const size) {
        GeluGradLauncher(dy, in, out, size);
    }
};

template <typename Device, typename T>
struct GeluGradGradFunctor {
    void operator()(const Device& d, const T * dy, const T * dy_, const T * in, T * out, int const size) {
        GeluGradGradLauncher(dy, dy_, in, out, size);
    }
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
        // GeluLauncher(x.flat<T>().data(), output->flat<T>().data(), static_cast<int>(output->NumElements()));
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
        // GeluGradLauncher(dy.flat<T>().data(), x.flat<T>().data(), output->flat<T>().data(), static_cast<int>(output->NumElements()));
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
        // GeluGradGradLauncher(dy.flat<T>().data(), x.flat<T>().data(), output->flat<T>().data(), static_cast<int>(output->NumElements()));
    }
};

#define REGISTER_GPU(T)                                                     \
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
