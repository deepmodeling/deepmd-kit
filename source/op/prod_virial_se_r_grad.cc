#include "custom_op.h"
#include "prod_virial_grad.h"

REGISTER_OP("ProdVirialSeRGrad")
.Attr("T: {float, double}")
.Input("grad: T")
.Input("net_deriv: T")
.Input("in_deriv: T")
.Input("rij: T")
.Input("nlist: int32")
.Input("natoms: int32")
.Output("grad_net: T");

using CPUDevice = Eigen::ThreadPoolDevice;

template<typename Device, typename FPTYPE>
class ProdVirialSeRGradOp : public OpKernel 
{
public:
  explicit ProdVirialSeRGradOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& grad_tensor		= context->input(context_input_index++);
    const Tensor& net_deriv_tensor	= context->input(context_input_index++);
    const Tensor& in_deriv_tensor	= context->input(context_input_index++);
    const Tensor& rij_tensor		= context->input(context_input_index++);
    const Tensor& nlist_tensor		= context->input(context_input_index++);
    const Tensor& natoms_tensor		= context->input(context_input_index++);

    // set size of the sample
    TensorShape grad_shape		= grad_tensor.shape();
    TensorShape net_deriv_shape		= net_deriv_tensor.shape();
    TensorShape in_deriv_shape		= in_deriv_tensor.shape();
    TensorShape rij_shape		= rij_tensor.shape();
    TensorShape nlist_shape		= nlist_tensor.shape();

    OP_REQUIRES (context, (grad_shape.dims() == 2),	errors::InvalidArgument ("Dim of grad should be 2"));
    OP_REQUIRES (context, (net_deriv_shape.dims() == 2),errors::InvalidArgument ("Dim of net deriv should be 2"));
    OP_REQUIRES (context, (in_deriv_shape.dims() == 2), errors::InvalidArgument ("Dim of input deriv should be 2"));
    OP_REQUIRES (context, (rij_shape.dims() == 2),	errors::InvalidArgument ("Dim of rij should be 2"));
    OP_REQUIRES (context, (nlist_shape.dims() == 2),	errors::InvalidArgument ("Dim of nlist should be 2"));
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
    OP_REQUIRES (context, (nframes == rij_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES (context, (nframes == nlist_shape.dim_size(0)),		errors::InvalidArgument ("number of frames should match"));
    
    OP_REQUIRES (context, (9 == grad_shape.dim_size(1)),		errors::InvalidArgument ("input grad shape should be 3 x natoms"));
    OP_REQUIRES (context, (nloc * ndescrpt * 3 == in_deriv_shape.dim_size(1)),errors::InvalidArgument ("number of descriptors should match"));
    OP_REQUIRES (context, (nloc * nnei * 3 == rij_shape.dim_size(1)),	errors::InvalidArgument ("dim of rij should be  nnei * 3"));

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
    auto rij		= rij_tensor		.flat<FPTYPE>();
    auto nlist		= nlist_tensor		.flat<int>();
    auto grad_net	= grad_net_tensor	->flat<FPTYPE>();

    // loop over frames
#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk){

      int grad_iter	= kk * 9;
      int in_iter	= kk * nloc * ndescrpt * 3;
      int rij_iter	= kk * nloc * nnei * 3;
      int nlist_iter	= kk * nloc * nnei;
      int grad_net_iter	= kk * nloc * ndescrpt;

      deepmd::prod_virial_grad_r_cpu(
	  &grad_net(grad_net_iter),
	  &grad(grad_iter),
	  &in_deriv(in_iter),
	  &rij(rij_iter),
	  &nlist(nlist_iter),
	  nloc,
	  nnei);
    }
  }
};

// Register the GPU kernels.
#define REGISTER_CPU(T)                                                                   \
REGISTER_KERNEL_BUILDER(                                                                  \
    Name("ProdVirialSeRGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),                      \
    ProdVirialSeRGradOp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);
