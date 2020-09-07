#include "common.h"
#include "CustomeOperation.h"

#ifdef HIGH_PREC
REGISTER_OP("ProdVirialSeA")
    .Input("net_deriv: double")
    .Input("in_deriv: double")
    .Input("rij: double")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Attr("n_a_sel: int")
    .Attr("n_r_sel: int")
    .Output("virial: double")
    .Output("atom_virial: double");
#else
REGISTER_OP("ProdVirialSeA")
    .Input("net_deriv: float")
    .Input("in_deriv: float")
    .Input("rij: float")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Attr("n_a_sel: int")
    .Attr("n_r_sel: int")
    .Output("virial: float")
    .Output("atom_virial: float");
#endif

template<typename Device, typename T>
struct ProdVirialSeAFunctor {
    void operator()(const CPUDevice& d, T * virial, T * atom_virial, const T * net_deriv, const T * in_deriv, const T * rij, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const int n_a_sel, const int n_a_shift) {
        ProdVirialSeACPULauncher(virial, atom_virial, net_deriv, in_deriv, rij, nlist, nloc, nall, nnei, ndescrpt, n_a_sel, n_a_shift);
    }
    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, T * virial, T * atom_virial, const T * net_deriv, const T * in_deriv, const T * rij, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const int n_a_sel, const int n_a_shift) {
        ProdVirialSeAGPULauncher(virial, atom_virial, net_deriv, in_deriv, rij, nlist, nloc, nall, nnei, ndescrpt, n_a_sel, n_a_shift);
    }
    #endif // GOOGLE_CUDA
};

template<typename Device>
class ProdVirialSeAOp : public OpKernel {
 public:
    explicit ProdVirialSeAOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("n_a_sel", &n_a_sel));
        OP_REQUIRES_OK(context, context->GetAttr("n_r_sel", &n_r_sel));
        n_a_shift = n_a_sel * 4;
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        int context_input_index = 0;
        const Tensor& net_deriv_tensor	= context->input(context_input_index++);
        const Tensor& in_deriv_tensor	= context->input(context_input_index++);
        const Tensor& rij_tensor		= context->input(context_input_index++);
        const Tensor& nlist_tensor		= context->input(context_input_index++);
        const Tensor& natoms_tensor		= context->input(context_input_index++);

        // set size of the sample
        OP_REQUIRES (context, (net_deriv_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of net deriv should be 2"));
        OP_REQUIRES (context, (in_deriv_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of input deriv should be 2"));
        OP_REQUIRES (context, (rij_tensor.shape().dims() == 2),		    errors::InvalidArgument ("Dim of rij should be 2"));
        OP_REQUIRES (context, (nlist_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of nlist should be 2"));
        OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of natoms should be 1"));

        OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),	errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
        const int * natoms = natoms_tensor.flat<int>().data();
        int nloc = natoms[0];
        int nall = natoms[1];
        int nnei = nlist_tensor.shape().dim_size(1) / nloc;
        int nframes = net_deriv_tensor.shape().dim_size(0);
        int ndescrpt = net_deriv_tensor.shape().dim_size(1) / nloc;

        // check the sizes
        OP_REQUIRES (context, (nframes == in_deriv_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
        OP_REQUIRES (context, (nframes == rij_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of samples should match"));
        OP_REQUIRES (context, (nframes == nlist_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));

        OP_REQUIRES (context, (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1)), errors::InvalidArgument ("number of descriptors should match"));
        OP_REQUIRES (context, (nloc * nnei * 3 == rij_tensor.shape().dim_size(1)),	errors::InvalidArgument ("dim of rij should be nnei * 3"));
        OP_REQUIRES (context, (nnei == n_a_sel + n_r_sel),				errors::InvalidArgument ("number of neighbors should match"));

        // Create an output tensor
        TensorShape virial_shape ;
        virial_shape.AddDim (nframes);
        virial_shape.AddDim (9);
        Tensor* virial_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, virial_shape, &virial_tensor));
        TensorShape atom_virial_shape;
        atom_virial_shape.AddDim (nframes);
        atom_virial_shape.AddDim (9 * nall);
        Tensor* atom_virial_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, atom_virial_shape, &atom_virial_tensor));

        // flat the tensors
        auto net_deriv = net_deriv_tensor.flat<VALUETYPE>();
        auto in_deriv = in_deriv_tensor.flat<VALUETYPE>();
        auto rij = rij_tensor.flat<VALUETYPE>();
        auto nlist = nlist_tensor.flat<int>();
        auto virial = virial_tensor->flat<VALUETYPE>();
        auto atom_virial = atom_virial_tensor->flat<VALUETYPE>();
        
        ProdVirialSeAFunctor<Device, VALUETYPE>()(
            context->eigen_device<Device>(),
            virial_tensor->flat<VALUETYPE>().data(), 
            atom_virial_tensor->flat<VALUETYPE>().data(),
            net_deriv_tensor.flat<VALUETYPE>().data(),
            in_deriv_tensor.flat<VALUETYPE>().data(),
            rij_tensor.flat<VALUETYPE>().data(),
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
#define REGISTER_CPU()                                             \
REGISTER_KERNEL_BUILDER(                                           \
    Name("ProdVirialSeA").Device(DEVICE_CPU),                      \
    ProdVirialSeAOp<CPUDevice>);
REGISTER_CPU();

// Register the GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_GPU()                                             \
REGISTER_KERNEL_BUILDER(                                           \
    Name("ProdVirialSeA").Device(DEVICE_GPU).HostMemory("natoms"), \
    ProdVirialSeAOp<GPUDevice>);
REGISTER_GPU();
#endif  // GOOGLE_CUDA