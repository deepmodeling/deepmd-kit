#include "common.h"
#include "CustomeOperation.h"

REGISTER_OP("TabulateFusion")
    .Attr("T: {float, double}")
    .Input("table: T")
    .Input("table_info: T")
    .Input("input: T")
    .Input("ff: T")
    .Attr("last_layer_size: int")
    .Output("output: T");

REGISTER_OP("TabulateFusionGrad")
    .Attr("T: {float, double}")
    .Input("table: T")
    .Input("table_info: T")
    .Input("input: T")
    .Input("ff: T")
    .Input("dy: T")        
    .Input("output: T")         
    .Output("dy_dx: T")
    .Output("dy_df: T");

template <typename FPTYPE>
struct TabulateFusionFunctor {
    void operator()(const CPUDevice& d, const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const int nloc, const int nnei, const int last_layer_size, FPTYPE * out) {
        TabulateFusionCPULauncher(table, table_info, in, ff, nloc, nnei, last_layer_size, out);
    }
    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const int nloc, const int nnei, const int last_layer_size, FPTYPE * out) {
        //Currently, Do nothing at all! 
        TabulateFusionGPULauncher(table, table_info, in, ff, nloc, nnei, last_layer_size, out);
    }
    #endif // GOOGLE_CUDA 
};

template <typename FPTYPE>
struct TabulateFusionGradFunctor {
    void operator()(const CPUDevice& d, const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const FPTYPE * dy, const int nloc, const int nnei, const int last_layer_size, FPTYPE * dy_dx, FPTYPE * dy_df) {
        TabulateFusionGradCPULauncher(table, table_info, in, ff, dy, nloc, nnei, last_layer_size, dy_dx, dy_df);
    }
    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const FPTYPE * dy, const int nloc, const int nnei, const int last_layer_size, FPTYPE * dy_dx, FPTYPE * dy_df) {
        //Currently, Do nothing at all! 
        TabulateFusionGradGPULauncher(table, table_info, in, ff, dy, nloc, nnei, last_layer_size, dy_dx, dy_df);
    }
    #endif // GOOGLE_CUDA 
};

template <typename FPTYPE>
struct TabulateCheckerFunctor {
    void operator()(const CPUDevice& d, const FPTYPE * table_info, const FPTYPE * in, int * out, const int nloc, const int nnei) {
        TabulateCheckerCPULauncher(table_info, in, out, nloc, nnei);
    }
    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const FPTYPE * table_info, const FPTYPE * in, int * out, const int nloc, const int nnei) {
        //Currently, Do nothing at all! 
        TabulateCheckerGPULauncher(table_info, in, out, nloc, nnei);
    }
    #endif // GOOGLE_CUDA 
};

template<typename Device, typename FPTYPE>
class TabulateFusionOp : public OpKernel {
  public:
    explicit TabulateFusionOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("last_layer_size", &last_layer_size));
        counter = -1;
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        int context_input_index = 0;
        const Tensor& table	= context->input(context_input_index++);
        const Tensor& table_info = context->input(context_input_index++);
        const Tensor& input	= context->input(context_input_index++);
        const Tensor& ff	= context->input(context_input_index++);

        // set size of the sample
        OP_REQUIRES (context, (table.shape().dims() == 2),	    errors::InvalidArgument ("Dim of table should be 2"));
        OP_REQUIRES (context, (input.shape().dims() == 2),		errors::InvalidArgument ("Dim of input should be 2"));
        OP_REQUIRES (context, (ff.shape().dims() == 3),		    errors::InvalidArgument ("Dim of input should be 3"));

        TensorShape output_shape;
        output_shape.AddDim (ff.shape().dim_size(0));
        output_shape.AddDim (4);
        output_shape.AddDim (last_layer_size);

        int context_output_index = 0;
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
	    					     output_shape,
	    					     &output));

        // counter++;
        // if ((int)table_info.flat<FPTYPE>().data()[5] != -1 && counter % (int)table_info.flat<FPTYPE>().data()[5] == 0) {
        //     Tensor int_temp;
        //     TensorShape int_shape;
        //     int_shape.AddDim(2 * ff.shape().dim_size(0));
        //     OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, int_shape, &int_temp));
        //     TabulateCheckerFunctor<FPTYPE>()(
        //         context->eigen_device<Device>(),
        //         table_info.flat<FPTYPE>().data(),
        //         input.flat<FPTYPE>().data(),
        //         int_temp.flat<int>().data(),
        //         ff.shape().dim_size(0),
        //         ff.shape().dim_size(1)
        //     );
        // }

        TabulateFusionFunctor<FPTYPE>()(
            context->eigen_device<Device>(),            // define actually graph execution device
            table.flat<FPTYPE>().data(),
            table_info.flat<FPTYPE>().data(),
            input.flat<FPTYPE>().data(),
            ff.flat<FPTYPE>().data(),
            ff.shape().dim_size(0),
            ff.shape().dim_size(1),
            last_layer_size,
            output->flat<FPTYPE>().data()
        );
    }
private:
    int counter;
    int last_layer_size;
};

template<typename Device, typename FPTYPE>
class TabulateFusionGradOp : public OpKernel {
 public:
    explicit TabulateFusionGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // std::cout << "I'm here" << std::endl;
        // Grab the input tensor
        int context_input_index = 0;
        const Tensor& table	= context->input(context_input_index++);
        const Tensor& table_info = context->input(context_input_index++);
        const Tensor& input	= context->input(context_input_index++);
        const Tensor& ff	= context->input(context_input_index++);
        const Tensor& dy	= context->input(context_input_index++);
        const Tensor& output = context->input(context_input_index++);

        // set size of the sample
        OP_REQUIRES (context, (dy.shape().dims() == 3),	    errors::InvalidArgument ("Dim of table should be 1"));

        int context_output_index = 0;
        Tensor* dy_dx = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
	    					     input.shape(),
	    					     &dy_dx));
        Tensor* dy_df = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
	    					     ff.shape(),
	    					     &dy_df));

        TabulateFusionGradFunctor<FPTYPE>()(
            context->eigen_device<Device>(),            // define actually graph execution device
            table.flat<FPTYPE>().data(),
            table_info.flat<FPTYPE>().data(),
            input.flat<FPTYPE>().data(),
            ff.flat<FPTYPE>().data(),
            dy.flat<FPTYPE>().data(),
            ff.shape().dim_size(0),
            ff.shape().dim_size(1),
            output.shape().dim_size(2),
            dy_dx->flat<FPTYPE>().data(),
            dy_df->flat<FPTYPE>().data()
        );
    }
private:
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
