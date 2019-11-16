#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cuda_runtime.h>
#include <iostream>

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

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else
typedef float  VALUETYPE;
#endif

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

void ProdForceSeALauncher(VALUETYPE * force, 
                        const VALUETYPE * net_deriv,
                        const VALUETYPE * in_deriv,
                        const int * nlist,
                        const int nloc,
                        const int nall,
                        const int ndescrpt,
                        const int nnei,
                        const int n_a_sel,
                        const int n_a_shift);

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
        int * natoms = new int[natoms_tensor.shape().dim_size(0)];
        cudaErrcheck(cudaMemcpy(natoms, natoms_tensor.flat<int>().data(), sizeof(int) * natoms_tensor.shape().dim_size(0), cudaMemcpyDeviceToHost));
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

        for (int II = 0; II < nframes; II++) {
            ProdForceSeALauncher(force_tensor->flat<VALUETYPE>().data() + II * (nall * 3),
                                net_deriv_tensor.flat<VALUETYPE>().data() + II * (nloc * ndescrpt),
                                in_deriv_tensor.flat<VALUETYPE>().data() + II * (nloc * ndescrpt * 3),
                                nlist_tensor.flat<int>().data() + II * (nloc * nnei),
                                nloc,
                                nall, 
                                ndescrpt,
                                nnei,
                                n_a_sel,
                                n_a_shift
            );
        }
        delete[] natoms;
    }
private:
    int n_r_sel, n_a_sel, n_a_shift;
};

REGISTER_KERNEL_BUILDER(Name("ProdForceSeA").Device(DEVICE_GPU), ProdForceSeAOp);