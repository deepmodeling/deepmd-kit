#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

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

#if GOOGLE_CUDA
void TabulateFusionLauncher(const float * table, const float * table_info, const float * in, const float * ff, const int nloc, const int nnei, const int last_layer_size, float * out);
void TabulateFusionLauncher(const double * table, const double * table_info, const double * in, const double * ff, const int nloc, const int nnei, const int last_layer_size, double * out);
void TabulateFusionGradLauncher(const float * table, const float * table_info, const float * in, const float * ff, const float * dy, const int nloc, const int nnei, const int last_layer_size, float * dy_dx, float * dy_df);
void TabulateFusionGradLauncher(const double * table, const double * table_info, const double * in, const double * ff, const double * dy, const int nloc, const int nnei, const int last_layer_size, double * dy_dx, double * dy_df);
void TabulateCheckerLauncher(const float * table_info, const float * in, int * out, const int nloc, const int nnei);
void TabulateCheckerLauncher(const double * table_info, const double * in, int * out, const int nloc, const int nnei);
#endif

template <typename FPTYPE>
inline FPTYPE dot(FPTYPE a[4], FPTYPE b[4]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; 
}

/*
    This inline function was designed to get the table info and bias value for current input xx!
    lower:      indicate the lower boundary of the first table;
    upper:      indicate the upper boundary of the first table as well as the lower boundary of the second table;
    max:        indicate the upper boundary of the second table;
    stride0:    indicate the stride of the first table;
    stride1:    indicate the stride of the second table;
    xx:         indicate the inputs value;
    table_idx:  indicate the location of table info of input value xx;
*/
template <typename FPTYPE>
inline void locate_xx(const FPTYPE& lower, const FPTYPE& upper,  const FPTYPE& max, const FPTYPE& stride0, const FPTYPE& stride1, FPTYPE& xx, int& table_idx) {
    if (xx < lower) {
        table_idx = 0;
        xx = 0;
    }
    else if (xx < upper) {
        table_idx = (int)((xx - lower) / stride0);
        xx -= (table_idx * stride0 + lower);
    }
    else if (xx < max) {
        int first_stride = int((upper - lower) / stride0);
        table_idx = first_stride + (int)((xx - upper) / stride1);
        xx -= ((table_idx - first_stride) * stride1 + upper);
    }
    else {
        table_idx = int((upper - lower) / stride0) + (int)((max - upper) / stride1) - 1;
        xx = 0;
    }
}

template <typename FPTYPE>
struct TabulateFusionFunctor {
    void operator()(const CPUDevice& d, const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const int nloc, const int nnei, const int last_layer_size, FPTYPE * out) {
        //Currently, Do nothing at all! 
        // std::cout << "I'm in tabulate @CPU!" << std::endl;
        memset(out, 0.0, sizeof(FPTYPE) * nloc * 4 * last_layer_size);
        FPTYPE const lower   = table_info[0];
        FPTYPE const upper   = table_info[1];
        FPTYPE const _max    = table_info[2];
        FPTYPE const stride0 = table_info[3];
        FPTYPE const stride1 = table_info[4];
        // for every atom, execute a small gemm~
        // FPTYPE * res = new FPTYPE[4 * last_layer_size];
        #pragma omp parallel for
        for (int ii = 0; ii < nloc; ii++) {
            FPTYPE ll[4] = {0};
            FPTYPE ago = in[ii * nnei + nnei - 1];
            bool unloop = false; 
            for (int jj = 0; jj < nnei; jj++) { 
                ll[0] = ff[ii * nnei * 4 + jj * 4 + 0];
                ll[1] = ff[ii * nnei * 4 + jj * 4 + 1];
                ll[2] = ff[ii * nnei * 4 + jj * 4 + 2];
                ll[3] = ff[ii * nnei * 4 + jj * 4 + 3];
                FPTYPE xx = in[ii * nnei + jj]; 
                if (ago == xx) {
                    unloop = true;
                }
                int table_idx = 0;
                locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
                for (int kk = 0; kk < last_layer_size; kk++) {
                    // 1.094 timesteps/s                                       
                    FPTYPE a0  = table[table_idx * last_layer_size * 6 + 6 * kk + 0]; 
                    FPTYPE a1  = table[table_idx * last_layer_size * 6 + 6 * kk + 1]; 
                    FPTYPE a2  = table[table_idx * last_layer_size * 6 + 6 * kk + 2]; 
                    FPTYPE a3  = table[table_idx * last_layer_size * 6 + 6 * kk + 3];
                    FPTYPE a4  = table[table_idx * last_layer_size * 6 + 6 * kk + 4];
                    FPTYPE a5  = table[table_idx * last_layer_size * 6 + 6 * kk + 5];
                    // FPTYPE var = a0 + a1 * xx + a2 * xx * xx + a3 * xx * xx * xx + a4 * xx * xx * xx * xx + a5 * xx * xx * xx * xx * xx;
                    FPTYPE var = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;
                    if (unloop) {
                        out[ii * last_layer_size * 4 + 0 * last_layer_size + kk] += (nnei - jj) * var * ll[0];
                        out[ii * last_layer_size * 4 + 1 * last_layer_size + kk] += (nnei - jj) * var * ll[1];
                        out[ii * last_layer_size * 4 + 2 * last_layer_size + kk] += (nnei - jj) * var * ll[2];
                        out[ii * last_layer_size * 4 + 3 * last_layer_size + kk] += (nnei - jj) * var * ll[3];
                    }
                    else {
                        out[ii * last_layer_size * 4 + 0 * last_layer_size + kk] += var * ll[0];
                        out[ii * last_layer_size * 4 + 1 * last_layer_size + kk] += var * ll[1];
                        out[ii * last_layer_size * 4 + 2 * last_layer_size + kk] += var * ll[2];
                        out[ii * last_layer_size * 4 + 3 * last_layer_size + kk] += var * ll[3];
                    }
                }
                if (unloop) break;
            }
        }
    }
    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const int nloc, const int nnei, const int last_layer_size, FPTYPE * out) {
        //Currently, Do nothing at all! 
        TabulateFusionLauncher(table, table_info, in, ff, nloc, nnei, last_layer_size, out);
    }
    #endif // GOOGLE_CUDA 
};

template <typename FPTYPE>
struct TabulateFusionGradFunctor {
    void operator()(const CPUDevice& d, const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const FPTYPE * dy, const int nloc, const int nnei, const int last_layer_size, FPTYPE * dy_dx, FPTYPE * dy_df) {
        // std::cout << "I'm in tabulate gradient @CPU!" << std::endl;
        memset(dy_dx, 0.0, sizeof(FPTYPE) * nloc * nnei);
        memset(dy_df, 0.0, sizeof(FPTYPE) * nloc * nnei * 4);
        FPTYPE const lower   = table_info[0];
        FPTYPE const upper   = table_info[1];
        FPTYPE const _max    = table_info[2];
        FPTYPE const stride0 = table_info[3];
        FPTYPE const stride1 = table_info[4];
        // for every atom, execute a small gemm~
        // FPTYPE * res = new FPTYPE[4 * last_layer_size];
        #pragma omp parallel for
        for (int ii = 0; ii < nloc; ii++) {
            FPTYPE ll[4];
            FPTYPE rr[4];
            FPTYPE ago = in[ii * nnei + nnei - 1];
            bool unloop = false;
            for (int jj = 0; jj < nnei; jj++) {
                // construct the dy/dx
                ll[0] = ff[ii * nnei * 4 + jj * 4 + 0];
                ll[1] = ff[ii * nnei * 4 + jj * 4 + 1];
                ll[2] = ff[ii * nnei * 4 + jj * 4 + 2];
                ll[3] = ff[ii * nnei * 4 + jj * 4 + 3];
                FPTYPE xx = in[ii * nnei + jj]; 
                if (ago == xx) {
                    unloop = true;
                }
                int table_idx = 0;
                locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
                FPTYPE grad = 0.0;
                for (int kk = 0; kk < last_layer_size; kk++) {
                    rr[0] = dy[ii * last_layer_size * 4 + 0 * last_layer_size + kk];
                    rr[1] = dy[ii * last_layer_size * 4 + 1 * last_layer_size + kk];
                    rr[2] = dy[ii * last_layer_size * 4 + 2 * last_layer_size + kk];
                    rr[3] = dy[ii * last_layer_size * 4 + 3 * last_layer_size + kk];
                    // 1.094 timesteps/s
                    FPTYPE a0  = table[table_idx * last_layer_size * 6 + 6 * kk + 0]; 
                    FPTYPE a1  = table[table_idx * last_layer_size * 6 + 6 * kk + 1]; 
                    FPTYPE a2  = table[table_idx * last_layer_size * 6 + 6 * kk + 2]; 
                    FPTYPE a3  = table[table_idx * last_layer_size * 6 + 6 * kk + 3];
                    FPTYPE a4  = table[table_idx * last_layer_size * 6 + 6 * kk + 4];
                    FPTYPE a5  = table[table_idx * last_layer_size * 6 + 6 * kk + 5];
                    // FPTYPE res = a0 + a1 * xx + a2 * xx * xx + a3 * xx * xx * xx + a4 * xx * xx * xx * xx + a5 * xx * xx * xx * xx * xx;
                    FPTYPE res = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;

                    if (unloop) {
                        // grad += (a1 + 2 * a2 * xx + 3 * a3 * xx * xx + 4 * a4 * xx * xx * xx + 5 * a5 * xx * xx * xx * xx) * dot(ll, rr) * (nnei - jj);
                        grad += (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * xx) * xx) * xx) * xx) * dot(ll, rr) * (nnei - jj);
                        dy_df[ii * nnei * 4 + jj * 4 + 0] += res * rr[0] * (nnei - jj);
                        dy_df[ii * nnei * 4 + jj * 4 + 1] += res * rr[1] * (nnei - jj);
                        dy_df[ii * nnei * 4 + jj * 4 + 2] += res * rr[2] * (nnei - jj);
                        dy_df[ii * nnei * 4 + jj * 4 + 3] += res * rr[3] * (nnei - jj);
                    }
                    else {
                        // grad += (a1 + 2 * a2 * xx + 3 * a3 * xx * xx + 4 * a4 * xx * xx * xx + 5 * a5 * xx * xx * xx * xx) * dot(ll, rr);
                        grad += (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * xx) * xx) * xx) * xx) * dot(ll, rr);
                        dy_df[ii * nnei * 4 + jj * 4 + 0] += res * rr[0];
                        dy_df[ii * nnei * 4 + jj * 4 + 1] += res * rr[1];
                        dy_df[ii * nnei * 4 + jj * 4 + 2] += res * rr[2];
                        dy_df[ii * nnei * 4 + jj * 4 + 3] += res * rr[3];
                    }
                }
                dy_dx[ii * nnei + jj] = grad;
                if (unloop) break;
            }
        }
    }
    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const FPTYPE * table, const FPTYPE * table_info, const FPTYPE * in, const FPTYPE * ff, const FPTYPE * dy, const int nloc, const int nnei, const int last_layer_size, FPTYPE * dy_dx, FPTYPE * dy_df) {
        //Currently, Do nothing at all! 
        TabulateFusionGradLauncher(table, table_info, in, ff, dy, nloc, nnei, last_layer_size, dy_dx, dy_df);
    }
    #endif // GOOGLE_CUDA 
};

template <typename FPTYPE>
struct TabulateCheckerFunctor {
    void operator()(const CPUDevice& d, const FPTYPE * table_info, const FPTYPE * in, int * out, const int nloc, const int nnei) {
        FPTYPE const lower   = table_info[0];
        FPTYPE const upper   = table_info[1];
        FPTYPE const _max    = table_info[2];
        FPTYPE const stride0 = table_info[3];
        FPTYPE const stride1 = table_info[4];
        // for every atom, execute a small gemm~
        // FPTYPE * res = new FPTYPE[4 * last_layer_size];
        int Csub = 0;    // summation of second table approximate;
        int Dsub = 0;    // summation of the endpoint approximate;
        for (int ii = 0; ii < nloc; ii++) {
            for (int jj = 0; jj < nnei; jj++) {
                FPTYPE xx = in[ii * nnei + jj];
                if (xx < lower || xx > _max) {
                    Csub += 1;
                }
                else if (xx >= upper && xx <= _max) {
                    Dsub += 1;
                }
            }
        }
        if(Csub > 0) {
            std::cout << "# DEEPMD: warning! some values [" << Csub << "/" << nloc * nnei << "] overflow the range of the table, using the endpoint approximate processing.." << std::endl;
        }
        if(Dsub > 0) {
            std::cout << "# DEEPMD: warning! some values [" << Dsub << "/" << nloc * nnei << "] overflow the range of the table, using second table approximate processing.." << std::endl;
        }
    }
    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const FPTYPE * table_info, const FPTYPE * in, int * out, const int nloc, const int nnei) {
        //Currently, Do nothing at all! 
        TabulateCheckerLauncher(table_info, in, out, nloc, nnei);
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
