#include "custom_op.h"
#include "tabulate.h"

REGISTER_OP("TabulateFusion")
    .Attr("T: {float, double}")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Attr("last_layer_size: int")
    .Output("descriptor: T");

REGISTER_OP("TabulateFusionGrad")
    .Attr("T: {float, double}")
    .Input("table: T")
    .Input("table_info: T")
    .Input("em_x: T")
    .Input("em: T")
    .Input("dy: T")        
    .Input("descriptor: T")         
    .Output("dy_dem_x: T")
    .Output("dy_dem: T");

template<typename Device, typename FPTYPE>
class TabulateFusionOp : public OpKernel {
 public:
  explicit TabulateFusionOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("last_layer_size", &last_layer_size));
  }
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor	= context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_x_tensor	= context->input(context_input_index++);
    const Tensor& em_tensor	= context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES (context, (table_tensor.shape().dims() == 2),   errors::InvalidArgument ("Dim of table should be 2"));
    OP_REQUIRES (context, (em_x_tensor.shape().dims() == 2),    errors::InvalidArgument ("Dim of input should be 2"));
    OP_REQUIRES (context, (em_tensor.shape().dims() == 3),      errors::InvalidArgument ("Dim of input should be 3"));
    TensorShape descriptor_shape;
    descriptor_shape.AddDim (em_tensor.shape().dim_size(0));
    descriptor_shape.AddDim (4); // TODO: be careful here;
    descriptor_shape.AddDim (last_layer_size);
    int context_output_index = 0;
    Tensor* descriptor_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++,
	  		descriptor_shape,
	  		&descriptor_tensor));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );
    // flat the tensors
    FPTYPE * descriptor = descriptor_tensor->flat<FPTYPE>().data();
    const FPTYPE * table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE * table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE * em_x = em_x_tensor.flat<FPTYPE>().data();
    const FPTYPE * em = em_tensor.flat<FPTYPE>().data();
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei = em_tensor.shape().dim_size(1);

    if (device == "GPU") {
      #if GOOGLE_CUDA
      deepmd::tabulate_fusion_gpu_cuda(    
          descriptor,
          table, table_info, em_x, em, nloc, nnei, last_layer_size);
      #endif // GOOGLE_CUDA
    }
    else if (device == "CPU") {
      deepmd::tabulate_fusion_cpu(    
          descriptor,
          table, table_info, em_x, em, nloc, nnei, last_layer_size);
    }
  }
private:
    int last_layer_size;
    std::string device;
};

template<typename Device, typename FPTYPE>
class TabulateFusionGradOp : public OpKernel {
 public:
  explicit TabulateFusionGradOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& table_tensor	= context->input(context_input_index++);
    const Tensor& table_info_tensor = context->input(context_input_index++);
    const Tensor& em_x_tensor	= context->input(context_input_index++);
    const Tensor& em_tensor	= context->input(context_input_index++);
    const Tensor& dy_tensor	= context->input(context_input_index++);
    const Tensor& descriptor_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES (context, (dy_tensor.shape().dims() == 3), errors::InvalidArgument ("Dim of table should be 3"));
    int context_output_index = 0;
    Tensor* dy_dem_x_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++,
	  		em_x_tensor.shape(),
        &dy_dem_x_tensor));
    Tensor* dy_dem_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        context_output_index++,
	  		em_tensor.shape(),
	  		&dy_dem_tensor));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );

    // flat the tensors
    FPTYPE * dy_dem_x = dy_dem_x_tensor->flat<FPTYPE>().data();
    FPTYPE * dy_dem = dy_dem_tensor->flat<FPTYPE>().data();
    const FPTYPE * descriptor = descriptor_tensor.flat<FPTYPE>().data();
    const FPTYPE * table = table_tensor.flat<FPTYPE>().data();
    const FPTYPE * table_info = table_info_tensor.flat<FPTYPE>().data();
    const FPTYPE * em_x = em_x_tensor.flat<FPTYPE>().data();
    const FPTYPE * em = em_tensor.flat<FPTYPE>().data();
    const FPTYPE * dy = dy_tensor.flat<FPTYPE>().data();
    const int nloc = em_tensor.shape().dim_size(0);
    const int nnei = em_tensor.shape().dim_size(1);
    const int last_layer_size = descriptor_tensor.shape().dim_size(2);

    if (device == "GPU") {
      #if GOOGLE_CUDA
      deepmd::tabulate_fusion_grad_gpu_cuda(    
          dy_dem_x, dy_dem,
          table, table_info, em_x, em, dy, nloc, nnei, last_layer_size);
      #endif // GOOGLE_CUDA
    }
    else if (device == "CPU") {
      deepmd::tabulate_fusion_grad_cpu(    
          dy_dem_x, dy_dem,
          table, table_info, em_x, em, dy, nloc, nnei, last_layer_size);
    }
  }
private:
    std::string device;
};

#define REGISTER_CPU(T)                                                                             \
REGISTER_KERNEL_BUILDER(                                                                            \
    Name("TabulateFusion").Device(DEVICE_CPU).TypeConstraint<T>("T").HostMemory("table_info"),      \
    TabulateFusionOp<CPUDevice, T>);                                                                \
REGISTER_KERNEL_BUILDER(                                                                            \
    Name("TabulateFusionGrad").Device(DEVICE_CPU).TypeConstraint<T>("T").HostMemory("table_info"),  \
    TabulateFusionGradOp<CPUDevice, T>);                                                                
REGISTER_CPU(float);
REGISTER_CPU(double);

#if  GOOGLE_CUDA
#define REGISTER_GPU(T)                                                                             \
REGISTER_KERNEL_BUILDER(                                                                            \
    Name("TabulateFusion").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("table_info"),      \
    TabulateFusionOp<GPUDevice, T>);                                                                \
REGISTER_KERNEL_BUILDER(                                                                            \
    Name("TabulateFusionGrad").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("table_info"),  \
    TabulateFusionGradOp<GPUDevice, T>);                                                                
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA
