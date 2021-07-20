#include "custom_op.h"

REGISTER_OP("ProdForceGrad")
.Attr("T: {float, double}")
.Input("grad: T")
.Input("net_deriv: T")
.Input("in_deriv: T")
.Input("nlist: int32")
.Input("axis: int32")
.Input("natoms: int32")
.Attr("n_a_sel: int")
.Attr("n_r_sel: int")
.Output("grad_net: T");

using CPUDevice = Eigen::ThreadPoolDevice;

template<typename Device, typename FPTYPE>
class ProdForceGradOp : public OpKernel 
{
public:
  explicit ProdForceGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_a_sel", &n_a_sel));    
    OP_REQUIRES_OK(context, context->GetAttr("n_r_sel", &n_r_sel));    
    n_a_shift = n_a_sel * 4;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& grad_tensor		= context->input(0);
    const Tensor& net_deriv_tensor	= context->input(1);
    const Tensor& in_deriv_tensor	= context->input(2);
    const Tensor& nlist_tensor		= context->input(3);
    const Tensor& axis_tensor		= context->input(4);
    const Tensor& natoms_tensor		= context->input(5);

    // set size of the sample
    TensorShape grad_shape		= grad_tensor.shape();
    TensorShape net_deriv_shape		= net_deriv_tensor.shape();
    TensorShape in_deriv_shape		= in_deriv_tensor.shape();
    TensorShape nlist_shape		= nlist_tensor.shape();
    TensorShape axis_shape		= axis_tensor.shape();

    OP_REQUIRES (context, (grad_shape.dims() == 2),	errors::InvalidArgument ("Dim of grad should be 2"));
    OP_REQUIRES (context, (net_deriv_shape.dims() == 2),errors::InvalidArgument ("Dim of net deriv should be 2"));
    OP_REQUIRES (context, (in_deriv_shape.dims() == 2), errors::InvalidArgument ("Dim of input deriv should be 2"));
    OP_REQUIRES (context, (nlist_shape.dims() == 2),	errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (axis_shape.dims() == 2),	errors::InvalidArgument ("Dim of axis should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of natoms should be 1"));

    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),	errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    auto natoms	= natoms_tensor	.flat<int>();

    int nframes = net_deriv_tensor.shape().dim_size(0);
    int nloc = natoms(0);
    int ndescrpt = net_deriv_tensor.shape().dim_size(1) / nloc;
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;

    // check the sizes
    OP_REQUIRES (context, (nframes == grad_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES (context, (nframes == in_deriv_shape.dim_size(0)),	errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES (context, (nframes == nlist_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES (context, (nframes == axis_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    
    OP_REQUIRES (context, (nloc * 3 == grad_shape.dim_size(1)),		errors::InvalidArgument ("input grad shape should be 3 x natoms"));
    OP_REQUIRES (context, (nloc * ndescrpt * 12 == in_deriv_shape.dim_size(1)),errors::InvalidArgument ("number of descriptors should match"));
    OP_REQUIRES (context, (nnei == n_a_sel + n_r_sel),			errors::InvalidArgument ("number of neighbors should match"));
    OP_REQUIRES (context, (nloc * 4 == axis_shape.dim_size(1)),		errors::InvalidArgument ("number of axis type+id should be 2+2"));

    // Create an output tensor
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (nframes);
    grad_net_shape.AddDim (nloc * ndescrpt);

    // allocate the output tensor
    Tensor* grad_net_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape, &grad_net_tensor));
    
    // flat the tensors
    auto grad		= grad_tensor		.flat<FPTYPE>();
    auto net_deriv	= net_deriv_tensor	.flat<FPTYPE>();
    auto in_deriv	= in_deriv_tensor	.flat<FPTYPE>();
    auto nlist		= nlist_tensor		.flat<int>();
    auto axis		= axis_tensor		.flat<int>();
    auto grad_net	= grad_net_tensor	->flat<FPTYPE>();

    // loop over frames
#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk){

      int grad_iter	= kk * nloc * 3;
      int net_iter	= kk * nloc * ndescrpt;
      int in_iter	= kk * nloc * ndescrpt * 12;
      int nlist_iter	= kk * nloc * nnei;
      int axis_iter	= kk * nloc * 4;
      int grad_net_iter	= kk * nloc * ndescrpt;

      // reset the frame to 0
      for (int ii = 0; ii < nloc; ++ii){
	for (int aa = 0; aa < ndescrpt; ++aa){
	  grad_net (grad_net_iter + ii * ndescrpt + aa) = 0;
	}
      }      

      // compute grad of one frame
      for (int ii = 0; ii < nloc; ++ii){
	int i_idx = ii;
	
	// deriv wrt center atom
	for (int aa = 0; aa < ndescrpt; ++aa){
	  for (int dd = 0; dd < 3; ++dd){
	    grad_net (grad_net_iter + i_idx * ndescrpt + aa) -= grad (grad_iter + i_idx * 3 + dd) * in_deriv (in_iter + i_idx * ndescrpt * 12 + aa * 12 + dd);
	  }
	}

	// set axes
	int axis0_type = axis (axis_iter + i_idx * 4 + 0);
	int axis1_type = axis (axis_iter + i_idx * 4 + 2);
	int axis_0  = axis (axis_iter + i_idx * 4 + 1);
	int axis_1  = axis (axis_iter + i_idx * 4 + 3);
	if (axis0_type == 1) axis_0 += n_a_sel;
	if (axis1_type == 1) axis_1 += n_a_sel;

	// loop over neighbors
	for (int jj = 0; jj < nnei; ++jj){
	  int j_idx = nlist (nlist_iter + i_idx * nnei + jj);	  
	  if (j_idx > nloc) j_idx = j_idx % nloc;
	  if (j_idx < 0) continue;
	  if (jj == axis_0) {
	    for (int aa = 0; aa < ndescrpt; ++aa){
	      for (int dd = 0; dd < 3; ++dd){
		grad_net (grad_net_iter + i_idx * ndescrpt + aa) -= grad (grad_iter + j_idx * 3 + dd) * in_deriv (in_iter + i_idx * ndescrpt * 12 + aa * 12 + 3 + dd);
	      }
	    }
	  }
	  else if (jj == axis_1) {
	    for (int aa = 0; aa < ndescrpt; ++aa){
	      for (int dd = 0; dd < 3; ++dd){
		grad_net (grad_net_iter + i_idx * ndescrpt + aa) -= grad (grad_iter + j_idx * 3 + dd) * in_deriv (in_iter + i_idx * ndescrpt * 12 + aa * 12 + 6 + dd);
	      }
	    }
	  }
	  else {
	    int aa_start, aa_end;
	    make_descript_range (aa_start, aa_end, jj);
	    for (int aa = aa_start; aa < aa_end; ++aa){
	      for (int dd = 0; dd < 3; ++dd){
		grad_net (grad_net_iter + i_idx * ndescrpt + aa) -= grad (grad_iter + j_idx * 3 + dd) * in_deriv (in_iter + i_idx * ndescrpt * 12 + aa * 12 + 9 + dd);
	      }
	    }
	  }
	}
      }
    }
  }
private:
  int n_r_sel, n_a_sel, n_a_shift;
  inline void 
  make_descript_range (int & idx_start,
		       int & idx_end,
		       const int & nei_idx) {
    if (nei_idx < n_a_sel) {
      idx_start = nei_idx * 4;
      idx_end   = nei_idx * 4 + 4;
    }
    else {
      idx_start = n_a_shift + (nei_idx - n_a_sel);
      idx_end   = n_a_shift + (nei_idx - n_a_sel) + 1;
    }
  }
};

#define REGISTER_CPU(T)                                                                 \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("ProdForceGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),                       \
    ProdForceGradOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
