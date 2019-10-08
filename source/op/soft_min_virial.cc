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
REGISTER_OP("SoftMinVirial")
.Input("du: double")
.Input("sw_deriv: double")
.Input("rij: double")
.Input("nlist: int32")
.Input("natoms: int32")
.Attr("n_a_sel: int")
.Attr("n_r_sel: int")
.Output("virial: double")
.Output("atom_virial: double")
;
#else
REGISTER_OP("SoftMinVirial")
.Input("du: float")
.Input("sw_deriv: float")
.Input("rij: float")
.Input("nlist: int32")
.Input("natoms: int32")
.Attr("n_a_sel: int")
.Attr("n_r_sel: int")
.Output("virial: float")
.Output("atom_virial: float")
;
#endif

using namespace tensorflow;

class SoftMinVirialOp : public OpKernel {
 public:
  explicit SoftMinVirialOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_a_sel", &n_a_sel));
    OP_REQUIRES_OK(context, context->GetAttr("n_r_sel", &n_r_sel));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& du_tensor		= context->input(context_input_index++);
    const Tensor& sw_deriv_tensor	= context->input(context_input_index++);
    const Tensor& rij_tensor		= context->input(context_input_index++);
    const Tensor& nlist_tensor		= context->input(context_input_index++);
    const Tensor& natoms_tensor		= context->input(context_input_index++);

    // set size of the sample
    OP_REQUIRES (context, (du_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of net deriv should be 2"));
    OP_REQUIRES (context, (sw_deriv_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of input deriv should be 2"));
    OP_REQUIRES (context, (rij_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of rij should be 2"));
    OP_REQUIRES (context, (nlist_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of natoms should be 1"));

    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),	errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    auto natoms	= natoms_tensor	.flat<int>();

    int nframes = du_tensor.shape().dim_size(0);
    int nloc = natoms(0);
    int nall = natoms(1);
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;

    // check the sizes
    OP_REQUIRES (context, (nframes == sw_deriv_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nframes == rij_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nframes == nlist_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));

    OP_REQUIRES (context, (nloc == du_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of du should match"));
    OP_REQUIRES (context, (nloc * nnei * 3 == sw_deriv_tensor.shape().dim_size(1)), errors::InvalidArgument ("number of sw_deriv should match"));
    OP_REQUIRES (context, (nloc * nnei * 3 == rij_tensor.shape().dim_size(1)),	errors::InvalidArgument ("dim of rij should be nnei * 3"));
    OP_REQUIRES (context, (nnei == n_a_sel + n_r_sel),				errors::InvalidArgument ("number of neighbors should match"));

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
    auto du = du_tensor.matrix<VALUETYPE>();
    auto sw_deriv = sw_deriv_tensor.matrix<VALUETYPE>();
    auto rij = rij_tensor.matrix<VALUETYPE>();
    auto nlist = nlist_tensor.matrix<int>();
    auto virial = virial_tensor->matrix<VALUETYPE>();
    auto atom_virial = atom_virial_tensor->matrix<VALUETYPE>();

    // loop over samples
#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk){

      for (int ii = 0; ii < 9; ++ ii){
	virial (kk, ii) = 0.;
      }
      for (int ii = 0; ii < 9 * nall; ++ ii){
	atom_virial (kk, ii) = 0.;
      }

      // compute virial of a frame
      for (int ii = 0; ii < nloc; ++ii){
	int i_idx = ii;
	// loop over neighbors
	for (int jj = 0; jj < nnei; ++jj){	  
	  int j_idx = nlist (kk, i_idx * nnei + jj);
	  if (j_idx < 0) continue;
	  int rij_idx_shift = (ii * nnei + jj) * 3;
	  for (int dd0 = 0; dd0 < 3; ++dd0){
	    for (int dd1 = 0; dd1 < 3; ++dd1){
	      VALUETYPE tmp_v = du(kk, i_idx) * sw_deriv(kk, rij_idx_shift + dd0) * rij(kk, rij_idx_shift + dd1);
	      virial(kk, dd0 * 3 + dd1) -= tmp_v;		  
	      atom_virial(kk, j_idx * 9 + dd0 * 3 + dd1) -= tmp_v;
	    }
	  }
	}
      }      
    }
  }
private:
  int n_r_sel, n_a_sel;
};

REGISTER_KERNEL_BUILDER(Name("SoftMinVirial").Device(DEVICE_CPU), SoftMinVirialOp);



