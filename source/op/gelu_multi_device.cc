#include "custom_op.h"
#include "gelu.h"

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

// OpKernel definition.
// template parameter <FPTYPE> is the datatype of the tensors.
template <typename Device, typename FPTYPE>
class GeluOp : public OpKernel {
 public :
  explicit GeluOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& x_tensor = context->input(0);
    Tensor * output_tensor = NULL;
    int context_output_index = 0;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++, 
			  x_tensor.shape(),
			  &output_tensor));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );
    // flat the tensors
    FPTYPE * out = output_tensor->flat<FPTYPE>().data();
    const FPTYPE * x = x_tensor.flat<FPTYPE>().data();
    const int size = static_cast<int>(output_tensor->NumElements());

    if (device == "GPU") {
      #if GOOGLE_CUDA
      deepmd::gelu_gpu_cuda(
          out, 
          x, size);
      #endif // GOOGLE_CUDA
    }
    else if (device == "CPU") {
      deepmd::gelu_cpu(
          out, 
          x, size);
    }
  }
 private :
  std::string device;
};

// OpKernel definition.
// template parameter <FPTYPE> is the datatype of the tensors.
template <typename Device, typename FPTYPE>
class GeluGradOp : public OpKernel {
 public :
  explicit GeluGradOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& dy_tensor = context->input(0);
    const Tensor& x_tensor  = context->input(1);
    Tensor * output_tensor = NULL;
    int context_output_index = 0;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++, 
        x_tensor.shape(),
        &output_tensor));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );
    // flat the tensors
    FPTYPE * out = output_tensor->flat<FPTYPE>().data();
    const FPTYPE * x = x_tensor.flat<FPTYPE>().data();
    const FPTYPE * dy = dy_tensor.flat<FPTYPE>().data();
    const int size = static_cast<int>(output_tensor->NumElements());

    if (device == "GPU") {
      #if GOOGLE_CUDA
      deepmd::gelu_grad_gpu_cuda(
          out, 
          x, dy, size);
      #endif // GOOGLE_CUDA
    }
    else if (device == "CPU") {
      deepmd::gelu_grad_cpu(
          out, 
          x, dy, size);
    }
  }
 private :
  std::string device;
};

// OpKernel definition.
// template parameter <FPTYPE> is the datatype of the tensors.
template <typename Device, typename FPTYPE>
class GeluGradGradOp : public OpKernel {
 public :
  explicit GeluGradGradOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& dy_tensor = context->input(0);
    const Tensor& dy_2_tensor = context->input(1);
    const Tensor& x_tensor  = context->input(2);
		Tensor * output_tensor = NULL;
		int context_output_index = 0;	
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++, 
        x_tensor.shape(),
        &output_tensor));
    // flat the tensors
    FPTYPE * out = output_tensor->flat<FPTYPE>().data();
    const FPTYPE * x = x_tensor.flat<FPTYPE>().data();
    const FPTYPE * dy = dy_tensor.flat<FPTYPE>().data();
    const FPTYPE * dy_2 = dy_2_tensor.flat<FPTYPE>().data();
    const int size = static_cast<int>(output_tensor->NumElements());

    if (device == "GPU") {
      #if GOOGLE_CUDA
      deepmd::gelu_grad_grad_gpu_cuda(
          out, 
          x, dy, dy_2, size);
      #endif // GOOGLE_CUDA
    }
    else if (device == "CPU") {
      deepmd::gelu_grad_grad_cpu(
          out, 
          x, dy, dy_2, size);
    }
  }
 private :
  std::string device;
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
