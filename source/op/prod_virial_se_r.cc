#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

using namespace tensorflow;
using namespace std;

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
.Output("atom_virial: double")
;
#else
REGISTER_OP("ProdVirialSeR")
.Input("net_deriv: float")
.Input("in_deriv: float")
.Input("rij: float")
.Input("nlist: int32")
.Input("natoms: int32")
.Output("virial: float")
.Output("atom_virial: float")
;
#endif

using namespace tensorflow;

class ProdVirialSeROp : public OpKernel {
 public:
  explicit ProdVirialSeROp(OpKernelConstruction* context) : OpKernel(context) {
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
    OP_REQUIRES (context, (rij_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of rij should be 2"));
    OP_REQUIRES (context, (nlist_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of natoms should be 1"));

    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),	errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    auto natoms	= natoms_tensor	.flat<int>();

    int nframes = net_deriv_tensor.shape().dim_size(0);
    int nloc = natoms(0);
    int nall = natoms(1);
    int ndescrpt = net_deriv_tensor.shape().dim_size(1) / nloc;
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;

    // check the sizes
    OP_REQUIRES (context, (nframes == in_deriv_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nframes == rij_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nframes == nlist_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));

    OP_REQUIRES (context, (nloc * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1)), errors::InvalidArgument ("number of descriptors should match"));
    OP_REQUIRES (context, (nloc * nnei * 3 == rij_tensor.shape().dim_size(1)),	errors::InvalidArgument ("dim of rij should be nnei * 3"));

    // Create an output tensor
    TensorShape virial_shape ;
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

    // loop over samples
#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk){
      int net_iter	= kk * nloc * ndescrpt;
      int in_iter	= kk * nloc * ndescrpt * 3;
      int rij_iter	= kk * nloc * nnei * 3;
      int nlist_iter	= kk * nloc * nnei;
      int virial_iter	= kk * 9;
      int atom_virial_iter	= kk * nall * 9;

      for (int ii = 0; ii < 9; ++ ii){
	virial (virial_iter + ii) = 0.;
      }
      for (int ii = 0; ii < 9 * nall; ++ ii){
	atom_virial (atom_virial_iter + ii) = 0.;
      }

      // compute virial of a frame
      for (int ii = 0; ii < nloc; ++ii){
	int i_idx = ii;

	// deriv wrt neighbors
	for (int jj = 0; jj < nnei; ++jj){
	  int j_idx = nlist (nlist_iter + i_idx * nnei + jj);
	  if (j_idx < 0) continue;
	  VALUETYPE pref = -1.0 * net_deriv (net_iter + i_idx * ndescrpt + jj);
	  for (int dd0 = 0; dd0 < 3; ++dd0){
	    for (int dd1 = 0; dd1 < 3; ++dd1){
	      VALUETYPE tmp_v = pref * rij (rij_iter + i_idx * nnei * 3 + jj * 3 + dd0) *  in_deriv (in_iter + i_idx * ndescrpt * 3 + jj * 3 + dd1);
	      virial (virial_iter + dd0 * 3 + dd1) -= tmp_v;
	      atom_virial (atom_virial_iter + j_idx * 9 + dd0 * 3 + dd1) -= tmp_v;
	    }
	  }
	}
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ProdVirialSeR").Device(DEVICE_CPU), ProdVirialSeROp);



