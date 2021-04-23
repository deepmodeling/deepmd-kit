#include "custom_op.h"
#include "soft_min_switch_virial_grad.h"

REGISTER_OP("SoftMinVirialGrad")
.Attr("T: {float, double}")
.Input("grad: T")
.Input("du: T")
.Input("sw_deriv: T")
.Input("rij: T")
.Input("nlist: int32")
.Input("natoms: int32")
.Attr("n_a_sel: int")
.Attr("n_r_sel: int")
.Output("grad_net: T");

using CPUDevice = Eigen::ThreadPoolDevice;

template<typename Device, typename FPTYPE>
class SoftMinVirialGradOp : public OpKernel 
{
public:
  explicit SoftMinVirialGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_a_sel", &n_a_sel));    
    OP_REQUIRES_OK(context, context->GetAttr("n_r_sel", &n_r_sel));    
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& grad_tensor		= context->input(context_input_index++);
    const Tensor& du_tensor		= context->input(context_input_index++);
    const Tensor& sw_deriv_tensor	= context->input(context_input_index++);
    const Tensor& rij_tensor		= context->input(context_input_index++);
    const Tensor& nlist_tensor		= context->input(context_input_index++);
    const Tensor& natoms_tensor		= context->input(context_input_index++);

    // set size of the sample
    TensorShape grad_shape		= grad_tensor.shape();
    TensorShape du_shape		= du_tensor.shape();
    TensorShape sw_deriv_shape		= sw_deriv_tensor.shape();
    TensorShape rij_shape		= rij_tensor.shape();
    TensorShape nlist_shape		= nlist_tensor.shape();

    OP_REQUIRES (context, (grad_shape.dims() == 2),	errors::InvalidArgument ("Dim of grad should be 2"));
    OP_REQUIRES (context, (du_shape.dims() == 2),errors::InvalidArgument ("Dim of net deriv should be 2"));
    OP_REQUIRES (context, (sw_deriv_shape.dims() == 2), errors::InvalidArgument ("Dim of input deriv should be 2"));
    OP_REQUIRES (context, (rij_shape.dims() == 2),	errors::InvalidArgument ("Dim of rij should be 2"));
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
    OP_REQUIRES (context, (nframes == rij_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES (context, (nframes == nlist_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    
    OP_REQUIRES (context, (9 == grad_shape.dim_size(1)),		errors::InvalidArgument ("input grad shape should be 3 x natoms"));
    OP_REQUIRES (context, (nloc == du_tensor.shape().dim_size(1)),	errors::InvalidArgument ("number of du should match"));
    OP_REQUIRES (context, (nloc * nnei * 3 == sw_deriv_shape.dim_size(1)),errors::InvalidArgument ("number of descriptors should match"));
    OP_REQUIRES (context, (nloc * nnei * 3 == rij_shape.dim_size(1)),	errors::InvalidArgument ("dim of rij should be  nnei * 3"));
    OP_REQUIRES (context, (nnei == n_a_sel + n_r_sel),			errors::InvalidArgument ("number of neighbors should match"));

    // Create an output tensor
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (nframes);
    grad_net_shape.AddDim (nloc);

    // allocate the output tensor
    Tensor* grad_net_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape, &grad_net_tensor));
    
    // flat the tensors
    auto grad		= grad_tensor		.matrix<FPTYPE>();
    auto du		= du_tensor		.matrix<FPTYPE>();
    auto sw_deriv	= sw_deriv_tensor	.matrix<FPTYPE>();
    auto rij		= rij_tensor		.matrix<FPTYPE>();
    auto nlist		= nlist_tensor		.matrix<int>();
    auto grad_net	= grad_net_tensor	->matrix<FPTYPE>();

    // loop over frames
#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk){
      deepmd::soft_min_switch_virial_grad_cpu(
	  &grad_net(kk, 0),
	  &grad(kk, 0),
	  &sw_deriv(kk, 0),
	  &rij(kk, 0),
	  &nlist(kk, 0),
	  nloc,
	  nnei);
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

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                                   \
REGISTER_KERNEL_BUILDER(                                                                  \
    Name("SoftMinVirialGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),                      \
    SoftMinVirialGradOp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);
