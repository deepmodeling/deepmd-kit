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
REGISTER_OP("SoftMinForceGrad")
.Input("grad: double")
.Input("du: double")
.Input("sw_deriv: double")
.Input("nlist: int32")
.Input("natoms: int32")
.Attr("n_a_sel: int")
.Attr("n_r_sel: int")
.Output("grad_net: double");
#else
REGISTER_OP("SoftMinForceGrad")
.Input("grad: float")
.Input("du: float")
.Input("sw_deriv: float")
.Input("nlist: int32")
.Input("natoms: int32")
.Attr("n_a_sel: int")
.Attr("n_r_sel: int")
.Output("grad_net: float");
#endif

class SoftMinForceGradOp : public OpKernel 
{
public:
  explicit SoftMinForceGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_a_sel", &n_a_sel));    
    OP_REQUIRES_OK(context, context->GetAttr("n_r_sel", &n_r_sel));    
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& grad_tensor		= context->input(context_input_index++);
    const Tensor& du_tensor		= context->input(context_input_index++);
    const Tensor& sw_deriv_tensor	= context->input(context_input_index++);
    const Tensor& nlist_tensor		= context->input(context_input_index++);
    const Tensor& natoms_tensor		= context->input(context_input_index++);

    // set size of the sample
    TensorShape grad_shape		= grad_tensor.shape();
    TensorShape du_shape		= du_tensor.shape();
    TensorShape sw_deriv_shape		= sw_deriv_tensor.shape();
    TensorShape nlist_shape		= nlist_tensor.shape();

    OP_REQUIRES (context, (grad_shape.dims() == 2),	errors::InvalidArgument ("Dim of grad should be 2"));
    OP_REQUIRES (context, (du_shape.dims() == 2),	errors::InvalidArgument ("Dim of du should be 2"));
    OP_REQUIRES (context, (sw_deriv_shape.dims() == 2), errors::InvalidArgument ("Dim of sw deriv should be 2"));
    OP_REQUIRES (context, (nlist_shape.dims() == 2),	errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of natoms should be 1"));

    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),	errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    auto natoms	= natoms_tensor	.flat<int>();

    int nframes = du_tensor.shape().dim_size(0);
    int nloc = natoms(0);
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;

    // check the sizes
    OP_REQUIRES (context, (nframes == grad_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES (context, (nframes == sw_deriv_shape.dim_size(0)),	errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES (context, (nframes == nlist_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    
    OP_REQUIRES (context, (nloc * 3 == grad_shape.dim_size(1)),		errors::InvalidArgument ("input grad shape should be 3 x natoms"));
    OP_REQUIRES (context, (nloc * nnei * 3 == sw_deriv_shape.dim_size(1)),errors::InvalidArgument ("number of sw deriv should match"));
    OP_REQUIRES (context, (nnei == n_a_sel + n_r_sel),			errors::InvalidArgument ("number of neighbors should match"));

    // Create an output tensor
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (nframes);
    grad_net_shape.AddDim (nloc);

    // allocate the output tensor
    Tensor* grad_net_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape, &grad_net_tensor));
    
    // flat the tensors
    auto grad		= grad_tensor		.matrix<VALUETYPE>();
    auto du		= du_tensor		.matrix<VALUETYPE>();
    auto sw_deriv	= sw_deriv_tensor	.matrix<VALUETYPE>();
    auto nlist		= nlist_tensor		.matrix<int>();
    auto grad_net	= grad_net_tensor	->matrix<VALUETYPE>();

    // loop over frames
#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk){
      // reset the frame to 0
      for (int ii = 0; ii < nloc; ++ii){
	grad_net (kk, ii) = 0;
      }      

      // compute grad of one frame
      for (int ii = 0; ii < nloc; ++ii){
	int i_idx = ii;
	// deriv wrt center atom	
	for (int jj = 0; jj < nnei; ++jj){
	  int j_idx = nlist (kk, i_idx * nnei + jj);	  
	  if (j_idx < 0) continue;
	  int rij_idx_shift = (ii * nnei + jj) * 3;
	  grad_net(kk, i_idx) += grad(kk, i_idx * 3 + 0) * sw_deriv(kk, rij_idx_shift + 0);
	  grad_net(kk, i_idx) += grad(kk, i_idx * 3 + 1) * sw_deriv(kk, rij_idx_shift + 1);
	  grad_net(kk, i_idx) += grad(kk, i_idx * 3 + 2) * sw_deriv(kk, rij_idx_shift + 2);
	  grad_net(kk, i_idx) -= grad(kk, j_idx * 3 + 0) * sw_deriv(kk, rij_idx_shift + 0);
	  grad_net(kk, i_idx) -= grad(kk, j_idx * 3 + 1) * sw_deriv(kk, rij_idx_shift + 1);
	  grad_net(kk, i_idx) -= grad(kk, j_idx * 3 + 2) * sw_deriv(kk, rij_idx_shift + 2);
	}
      }
    }
  }
private:
  int n_r_sel, n_a_sel;
};

REGISTER_KERNEL_BUILDER(Name("SoftMinForceGrad").Device(DEVICE_CPU), SoftMinForceGradOp);
