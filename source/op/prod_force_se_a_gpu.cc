#include "common.h"
#include "CustomeOperation.h"

#ifdef HIGH_PREC
REGISTER_OP("ProdForceSeA")
    .Input("net_deriv: double")
    .Input("in_deriv: double")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Attr("n_a_sel: int")
    .Attr("n_r_sel: int")
    .Output("force: double");
#else
REGISTER_OP("ProdForceSeA")
    .Input("net_deriv: float")
    .Input("in_deriv: float")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Attr("n_a_sel: int")
    .Attr("n_r_sel: int")
    .Output("force: float");
#endif

template <typename Device, typename T>
struct ProdForceSeAFunctor {
    void operator()(const CPUDevice& d, T * force, const T * net_deriv, const T * in_deriv, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const int n_a_sel, const int n_a_shift) {
        ProdForceSeACPULauncher(force, net_deriv, in_deriv, nlist, nloc, nall, nnei, ndescrpt, n_a_sel, n_a_shift);
    }
    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, T * force, const T * net_deriv, const T * in_deriv, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const int n_a_sel, const int n_a_shift) {
        ProdForceSeAGPULauncher(force, net_deriv, in_deriv, nlist, nloc, nall, nnei, ndescrpt, n_a_sel, n_a_shift);
    }
    #endif // GOOGLE_CUDA
};

template<typename Device>
class ProdForceSeAOp : public OpKernel {
public:
    explicit ProdForceSeAOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("n_a_sel", &n_a_sel));
        OP_REQUIRES_OK(context, context->GetAttr("n_r_sel", &n_r_sel));
        n_a_shift = n_a_sel * 4;
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        int context_input_index = 0;
        const Tensor& net_deriv_tensor	= context->input(context_input_index++);
        const Tensor& in_deriv_tensor	= context->input(context_input_index++);
        const Tensor& nlist_tensor		= context->input(context_input_index++);
        const Tensor& natoms_tensor		= context->input(context_input_index++);

        // set size of the sample
        OP_REQUIRES (context, (net_deriv_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of net deriv should be 2"));
        OP_REQUIRES (context, (in_deriv_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of input deriv should be 2"));
        OP_REQUIRES (context, (nlist_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of nlist should be 2"));
        OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of natoms should be 1"));

        OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),	errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
        const int * natoms = natoms_tensor.flat<int>().data();
        int nloc = natoms[0];
        int nall = natoms[1];
        int nframes = net_deriv_tensor.shape().dim_size(0);
        int ndescrpt = net_deriv_tensor.shape().dim_size(1) / nloc;
        int nnei = nlist_tensor.shape().dim_size(1) / nloc;

        // check the sizes
        OP_REQUIRES (context, (nframes == in_deriv_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
        OP_REQUIRES (context, (nframes == nlist_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));

        OP_REQUIRES (context, (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1)), errors::InvalidArgument ("number of descriptors should match"));
        OP_REQUIRES (context, (nnei == n_a_sel + n_r_sel),				errors::InvalidArgument ("number of neighbors should match"));
        OP_REQUIRES (context, (0 == n_r_sel),					errors::InvalidArgument ("Rotational free only support all-angular information"));

        // Create an output tensor
        TensorShape force_shape ;
        force_shape.AddDim (nframes);
        force_shape.AddDim (3 * nall);
        Tensor* force_tensor = NULL;
        int context_output_index = 0;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
	    					     force_shape, &force_tensor));

        // flat the tensors
        auto net_deriv = net_deriv_tensor.flat<VALUETYPE>();
        auto in_deriv = in_deriv_tensor.flat<VALUETYPE>();
        auto nlist = nlist_tensor.flat<int>();
        auto force = force_tensor->flat<VALUETYPE>();

        assert (nframes == force_shape.dim_size(0));
        assert (nframes == net_deriv_tensor.shape().dim_size(0));
        assert (nframes == in_deriv_tensor.shape().dim_size(0));
        assert (nframes == nlist_tensor.shape().dim_size(0));
        assert (nall * 3 == force_shape.dim_size(1));
        assert (nloc * ndescrpt == net_deriv_tensor.shape().dim_size(1));
        assert (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1));
        assert (nloc * nnei == nlist_tensor.shape().dim_size(1));
        assert (nnei * 4 == ndescrpt);	    

        ProdForceSeAFunctor<Device, VALUETYPE>()(
            context->eigen_device<Device>(),
            force_tensor->flat<VALUETYPE>().data(),
            net_deriv_tensor.flat<VALUETYPE>().data(),
            in_deriv_tensor.flat<VALUETYPE>().data(),
            nlist_tensor.flat<int>().data(),
            nloc,
            nall, 
            nnei,
            ndescrpt,
            n_a_sel,
            n_a_shift
        );
    }
private:
    int n_r_sel, n_a_sel, n_a_shift;
};

// Register the CPU kernels.
#define REGISTER_CPU()                                            \
REGISTER_KERNEL_BUILDER(                                          \
    Name("ProdForceSeA").Device(DEVICE_CPU),                      \
    ProdForceSeAOp<CPUDevice>);
REGISTER_CPU();

// Register the GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_GPU()                                            \
REGISTER_KERNEL_BUILDER(                                          \
    Name("ProdForceSeA").Device(DEVICE_GPU).HostMemory("natoms"), \
    ProdForceSeAOp<GPUDevice>);
REGISTER_GPU();
#endif  // GOOGLE_CUDA