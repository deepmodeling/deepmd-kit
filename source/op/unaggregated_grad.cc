#include "custom_op.h"
#include "ComputeDescriptor.h"
#include "neighbor_list.h"

REGISTER_OP("UnaggregatedDyDxS")
    .Attr("T: {float, double}") 
    .Input("y: T")                
    .Input("w: T")              
    .Output("dy_dx: T");

REGISTER_OP("UnaggregatedDyDx")
    .Attr("T: {float, double}")
    .Input("z: T")           
    .Input("w: T")     
    .Input("dy_dx: T")     
    .Output("dz_dx: T");

REGISTER_OP("UnaggregatedDy2DxS")
    .Attr("T: {float, double}") 
    .Input("y: T")                
    .Input("dy: T")                
    .Input("w: T")              
    .Output("dy2_dx: T");

REGISTER_OP("UnaggregatedDy2Dx")
    .Attr("T: {float, double}")
    .Input("z: T")           
    .Input("w: T")     
    .Input("dz_dx: T")     
    .Input("dy_dx: T")     
    .Input("dy2_dx: T")     
    .Output("dz2_dx: T");

template <typename FPTYPE>
struct UnaggregatedDyDxSFunctor {
    void operator()(const CPUDevice& d, const FPTYPE * y, const FPTYPE * w, const int length, const int width, FPTYPE * dy_dx) {
        #pragma omp parallel for
        for (int ii = 0; ii < length; ii++) {
            for (int jj = 0; jj < width; jj++) {
                dy_dx[ii * width + jj] = (1 - y[ii * width + jj] * y[ii * width + jj]) * w[jj];
            }
        }
    }

    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const FPTYPE * y, const FPTYPE * w, const int length, const int width, FPTYPE * dy_dx) {
        //Currently, Do nothing at all! 
        return;
    }
    #endif // GOOGLE_CUDA 
};

// calculate the gradient for all variables!
template <typename FPTYPE>
struct UnaggregatedDyDxFunctor {
    void operator()(const CPUDevice& d, const FPTYPE * z, const FPTYPE * w, const FPTYPE * dy_dx, const int length, const int width, const int size, FPTYPE * dz_dx) {
        #pragma omp parallel for
        for (int kk = 0; kk < length; kk++) {
            for (int ii = 0; ii < width; ii++) {
                //FPTYPE dz_drou = 1 - (z[kk * width + ii] - y[kk * size + ii % size]) * (z[kk * width + ii] - y[kk * size + ii % size]);
                FPTYPE dz_drou = 1 - z[kk * width + ii] * z[kk * width + ii];
                FPTYPE accumulator = 0.0;
                for (int jj = 0; jj < size; jj++) {
                    accumulator += w[jj * width + ii] * dy_dx[kk * size + jj];
                }
                dz_drou *= accumulator;
                dz_drou += dy_dx[kk * size + ii % size];
                dz_dx[kk * width + ii] = dz_drou;
            }
        }
    }

    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const FPTYPE * z, const FPTYPE * w, const FPTYPE * dy_dx, const int length, const int width, const int size, FPTYPE * dz_dx) {
        //Currently, Do nothing at all! 
        return;
    }
    #endif // GOOGLE_CUDA
};

template <typename FPTYPE>
struct UnaggregatedDy2DxSFunctor {
    void operator()(const CPUDevice& d, const FPTYPE * y, const FPTYPE * dy, const FPTYPE * w, const int length, const int width, FPTYPE * dy2_dx) {
        #pragma omp parallel for
        for (int ii = 0; ii < length; ii++) {
            for (int jj = 0; jj < width; jj++) {
                dy2_dx[ii * width + jj] = -2 * w[jj] * y[ii * width + jj] * dy[ii * width + jj];
            }
        }
    }

    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const FPTYPE * y, const FPTYPE * w, const int length, const int width, FPTYPE * dy_dx) {
        //Currently, Do nothing at all! 
        return;
    }
    #endif // GOOGLE_CUDA 
};

// calculate the gradient for all variables!
template <typename FPTYPE>
struct UnaggregatedDy2DxFunctor {
    void operator()(const CPUDevice& d, const FPTYPE * z, const FPTYPE * w, const FPTYPE * dz_dx, const FPTYPE * dy_dx, const FPTYPE * dy2_dx, const int length, const int width, const int size, FPTYPE * dz2_dx) {
        #pragma omp parallel for
        for (int kk = 0; kk < length; kk++) {
            for (int ii = 0; ii < width; ii++) {
                //FPTYPE dz_drou = 1 - (z[kk * width + ii] - y[kk * size + ii % size]) * (z[kk * width + ii] - y[kk * size + ii % size]);
                FPTYPE dz_drou = 1 - z[kk * width + ii] * z[kk * width + ii];
                FPTYPE accumulator = 0.0;
                for (int jj = 0; jj < size; jj++) {
                    accumulator += w[jj * width + ii] * dy2_dx[kk * size + jj];
                }
                dz_drou *= accumulator;
                accumulator = 0.0;
                for (int jj = 0; jj < size; jj++) {
                    accumulator += w[jj * width + ii] * dy_dx[kk * size + jj];
                }
                dz_drou -= 2 * z[kk * width + ii] * (dz_dx[kk * width + ii] - dy_dx[kk * size + ii % size]) * accumulator;
                dz_drou += dy2_dx[kk * size + ii % size];
                dz2_dx[kk * width + ii] = dz_drou;
            }
        }
    }

    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const FPTYPE * z, const FPTYPE * w, const FPTYPE * dz_dx, const FPTYPE * dy_dx, const FPTYPE * dy2_dx, const int length, const int width, const int size, FPTYPE * dz2_dx) {
        //Currently, Do nothing at all! 
        return;
    }
    #endif // GOOGLE_CUDA
};

template<typename Device, typename FPTYPE>
class UnaggregatedDyDxSOp : public OpKernel {
 public:
    explicit UnaggregatedDyDxSOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        int context_input_index = 0;
        const Tensor& y	= context->input(context_input_index++);
        const Tensor& w	= context->input(context_input_index++);

        // set size of the sample
        OP_REQUIRES (context, (y.shape().dims() == 2),	    errors::InvalidArgument ("Dim of table should be 1"));
        OP_REQUIRES (context, (w.shape().dims() == 2),		errors::InvalidArgument ("Dim of input should be 2"));

        int context_output_index = 0;
        Tensor* dy_dx = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
	    					     y.shape(),
	    					     &dy_dx));

        UnaggregatedDyDxSFunctor<FPTYPE>()(
            context->eigen_device<Device>(),            // define actually graph execution device
            y.flat<FPTYPE>().data(),
            w.flat<FPTYPE>().data(),
            y.shape().dim_size(0),
            y.shape().dim_size(1),
            dy_dx->flat<FPTYPE>().data()
        );
    }
private:
};

template<typename Device, typename FPTYPE>
class UnaggregatedDy2DxSOp : public OpKernel {
 public:
    explicit UnaggregatedDy2DxSOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        int context_input_index = 0;
        const Tensor& y	    = context->input(context_input_index++);
        const Tensor& dy	= context->input(context_input_index++);
        const Tensor& w	    = context->input(context_input_index++);

        // set size of the sample
        OP_REQUIRES (context, (y.shape().dims()  == 2),	    errors::InvalidArgument ("Dim of input should be 2"));
        OP_REQUIRES (context, (dy.shape().dims() == 2),	    errors::InvalidArgument ("Dim of input should be 2"));
        OP_REQUIRES (context, (w.shape().dims()  == 2),		errors::InvalidArgument ("Dim of input should be 2"));
    
        int context_output_index = 0;
        Tensor* dy2_dx = NULL; 
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
	    					     y.shape(),
	    					     &dy2_dx));

        UnaggregatedDy2DxSFunctor<FPTYPE>()(
            context->eigen_device<Device>(),            // define actually graph execution device
            y.flat<FPTYPE>().data(),
            dy.flat<FPTYPE>().data(),
            w.flat<FPTYPE>().data(),
            y.shape().dim_size(0),
            y.shape().dim_size(1),
            dy2_dx->flat<FPTYPE>().data()
        );
    }
private:
};

template<typename Device, typename FPTYPE>
class UnaggregatedDyDxOp : public OpKernel {
 public:
    explicit UnaggregatedDyDxOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        int context_input_index = 0;
        const Tensor& z	= context->input(context_input_index++);
        const Tensor& w	= context->input(context_input_index++);
        const Tensor& dy_dx	= context->input(context_input_index++);

        // set size of the sample
        OP_REQUIRES (context, (z.shape().dims() == 2),	        errors::InvalidArgument ("Dim of table should be 1"));
        OP_REQUIRES (context, (w.shape().dims() == 2),		    errors::InvalidArgument ("Dim of input should be 2"));
        OP_REQUIRES (context, (dy_dx.shape().dims() == 2),		errors::InvalidArgument ("Dim of input should be 2"));

        int context_output_index = 0;
        Tensor* dz_dx = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
	    					     z.shape(),
	    					     &dz_dx));

        UnaggregatedDyDxFunctor<FPTYPE>()(
            context->eigen_device<Device>(),            // define actually graph execution device
            z.flat<FPTYPE>().data(),
            w.flat<FPTYPE>().data(),
            dy_dx.flat<FPTYPE>().data(),
            z.shape().dim_size(0),
            z.shape().dim_size(1),
            w.shape().dim_size(0),
            dz_dx->flat<FPTYPE>().data()
        );
    }
private:
};

template<typename Device, typename FPTYPE>
class UnaggregatedDy2DxOp : public OpKernel {
 public:
    explicit UnaggregatedDy2DxOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        int context_input_index = 0;
        const Tensor& z	= context->input(context_input_index++);
        const Tensor& w	= context->input(context_input_index++);
        const Tensor& dz_dx	= context->input(context_input_index++);
        const Tensor& dy_dx	= context->input(context_input_index++);
        const Tensor& dy2_dx = context->input(context_input_index++);

        // set size of the sample
        OP_REQUIRES (context, (z.shape().dims() == 2),	        errors::InvalidArgument ("Dim of input should be 2"));
        OP_REQUIRES (context, (w.shape().dims() == 2),		    errors::InvalidArgument ("Dim of input should be 2"));
        OP_REQUIRES (context, (dz_dx.shape().dims() == 2),		errors::InvalidArgument ("Dim of input should be 2"));
        OP_REQUIRES (context, (dy_dx.shape().dims() == 2),		errors::InvalidArgument ("Dim of input should be 2"));
        OP_REQUIRES (context, (dy2_dx.shape().dims() == 2),		errors::InvalidArgument ("Dim of input should be 2"));

        int context_output_index = 0;
        Tensor* dz2_dx = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
	    					     z.shape(),
	    					     &dz2_dx));

        UnaggregatedDy2DxFunctor<FPTYPE>()(
            context->eigen_device<Device>(),            // define actually graph execution device
            z.flat<FPTYPE>().data(),
            w.flat<FPTYPE>().data(),
            dz_dx.flat<FPTYPE>().data(),
            dy_dx.flat<FPTYPE>().data(),
            dy2_dx.flat<FPTYPE>().data(),
            z.shape().dim_size(0),
            z.shape().dim_size(1),
            w.shape().dim_size(0),
            dz2_dx->flat<FPTYPE>().data()
        );
    }
private:
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                                 \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("UnaggregatedDyDxS").Device(DEVICE_CPU).TypeConstraint<T>("T"),                \
    UnaggregatedDyDxSOp<CPUDevice, T>);                                                 \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("UnaggregatedDyDx").Device(DEVICE_CPU).TypeConstraint<T>("T"),                 \
    UnaggregatedDyDxOp<CPUDevice, T>);                                                  \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("UnaggregatedDy2DxS").Device(DEVICE_CPU).TypeConstraint<T>("T"),               \
    UnaggregatedDy2DxSOp<CPUDevice, T>);                                                \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("UnaggregatedDy2Dx").Device(DEVICE_CPU).TypeConstraint<T>("T"),                \
    UnaggregatedDy2DxOp<CPUDevice, T>);             
REGISTER_CPU(float);
REGISTER_CPU(double);
// Not required in the current situation
// // Register the GPU kernels.
// #if GOOGLE_CUDA
// #define REGISTER_GPU(T)                                                                 \
// REGISTER_KERNEL_BUILDER(                                                                \
//     Name("UnaggregatedDyDxS").Device(DEVICE_GPU).TypeConstraint<T>("T"),                \
//     UnaggregatedDyDxSOp<GPUDevice, T>);                                                 \
// REGISTER_KERNEL_BUILDER(                                                                \
//     Name("UnaggregatedDyDx").Device(DEVICE_GPU).TypeConstraint<T>("T"),                 \
//     UnaggregatedDyDxOp<GPUDevice, T>);                         
// REGISTER_GPU(float);
// REGISTER_GPU(double);
// #endif  // GOOGLE_CUDA
