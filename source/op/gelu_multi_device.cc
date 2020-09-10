#include "common.h"
#include "CustomeOperation.h"

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

template <typename T>
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

template <typename T>
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

template <typename T>
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
		
		GeluFunctor<T>()(
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
		
		GeluGradFunctor<T>()(
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
		
		GeluGradGradFunctor<T>()(
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
REGISTER_KERNEL_BUILDER(                                                \
    Name("Gelu").Device(DEVICE_CPU).TypeConstraint<T>("T"),             \
    GeluOp<CPUDevice, T>);                                              \
REGISTER_KERNEL_BUILDER(                                                \
    Name("GeluGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),         \
    GeluGradOp<CPUDevice, T>);                                          \
REGISTER_KERNEL_BUILDER(                                                \
    Name("GeluGradGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
    GeluGradGradOp<CPUDevice, T>);                                      
REGISTER_CPU(float);
REGISTER_CPU(double);

#if GOOGLE_CUDA
#define REGISTER_GPU(T)                                                 \
REGISTER_KERNEL_BUILDER(                                                \
    Name("Gelu").Device(DEVICE_GPU).TypeConstraint<T>("T"),             \
    GeluOp<GPUDevice, T>);                                              \
REGISTER_KERNEL_BUILDER(                                                \
    Name("GeluGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"),         \
    GeluGradOp<GPUDevice, T>);                                          \
REGISTER_KERNEL_BUILDER(                                                \
    Name("GeluGradGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
    GeluGradGradOp<GPUDevice, T>);                                      
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif // GOOGLE_CUDA