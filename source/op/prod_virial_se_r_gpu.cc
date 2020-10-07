#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>
#include <cuda_runtime.h>

#ifdef HIGH_PREC
    typedef double VALUETYPE;
#else
    typedef float  VALUETYPE;
#endif

#ifdef HIGH_PREC
REGISTER_OP("ProdVirialSeR")
    .Input("net_deriv: double")
    .Input("in_deriv: double")
    .Input("rij: double")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Output("virial: double")
    .Output("atom_virial: double");
#else
REGISTER_OP("ProdVirialSeR")
    .Input("net_deriv: float")
    .Input("in_deriv: float")
    .Input("rij: float")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Output("virial: float")
    .Output("atom_virial: float");
#endif

using namespace tensorflow;

#define cudaErrcheck(res) { cudaAssert((res), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void ProdVirialSeRLauncher(VALUETYPE * virial, 
                        VALUETYPE * atom_virial,
                        const VALUETYPE * net_deriv,
                        const VALUETYPE * in_deriv,
                        const VALUETYPE * rij,
                        const int * nlist,
                        const int nloc,
                        const int nall,
                        const int nnei,
                        const int ndescrpt,
                        const int n_a_sel,
                        const int n_a_shift);

class ProdVirialSeROp : public OpKernel {
 public:
    explicit ProdVirialSeROp(OpKernelConstruction* context) : OpKernel(context) {
        // std::cout << "I'm in prod_virial_se_r_gpu.cc" << std::endl;
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
        cudaErrcheck(cudaDeviceSynchronize());
        OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),	errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
        int * natoms = new int[natoms_tensor.shape().dim_size(0)];
        cudaErrcheck(cudaMemcpy(natoms, natoms_tensor.flat<int>().data(), sizeof(int) * natoms_tensor.shape().dim_size(0), cudaMemcpyDeviceToHost));
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
        OP_REQUIRES (context, (nloc * nnei * 3 == rij_tensor.shape().dim_size(1)),	        errors::InvalidArgument ("dim of rij should be nnei * 3"));

        // Create an output tensor
        TensorShape virial_shape;
        virial_shape.AddDim (nframes);
        virial_shape.AddDim (9);
        Tensor* virial_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, virial_shape, &virial_tensor));
        TensorShape atom_virial_shape ;
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

        for (int II = 0; II < nframes; II++) {
            ProdVirialSeRLauncher(virial_tensor->flat<VALUETYPE>().data() + II * 9, 
                                atom_virial_tensor->flat<VALUETYPE>().data() + II * (nall * 9),
                                net_deriv_tensor.flat<VALUETYPE>().data() + II * (nloc * ndescrpt),
                                in_deriv_tensor.flat<VALUETYPE>().data() + II * (nloc * ndescrpt * 3),
                                rij_tensor.flat<VALUETYPE>().data() + II * (nloc * nnei * 3),
                                nlist_tensor.flat<int>().data() + II * (nloc * nnei),
                                nloc,
                                nall,
                                nnei,
                                ndescrpt,
                                n_a_sel,
                                n_a_shift
            );
        }
        delete[] natoms;
    }
private:
    int n_r_sel, n_a_sel, n_a_shift;
};

REGISTER_KERNEL_BUILDER(Name("ProdVirialSeR").Device(DEVICE_GPU), ProdVirialSeROp);