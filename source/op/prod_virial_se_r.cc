#include "custom_op.h"
#include "prod_virial.h"

REGISTER_OP("ProdVirialSeR")
.Attr("T: {float, double}")
.Input("net_deriv: T")
.Input("in_deriv: T")
.Input("rij: T")
.Input("nlist: int32")
.Input("natoms: int32")
.Output("virial: T")
.Output("atom_virial: T");

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

template<typename Device, typename FPTYPE>
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
    auto net_deriv = net_deriv_tensor.flat<FPTYPE>();
    auto in_deriv = in_deriv_tensor.flat<FPTYPE>();
    auto rij = rij_tensor.flat<FPTYPE>();
    auto nlist = nlist_tensor.flat<int>();
    auto virial = virial_tensor->flat<FPTYPE>();
    auto atom_virial = atom_virial_tensor->flat<FPTYPE>();

    // loop over samples
#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk){
      int net_iter	= kk * nloc * ndescrpt;
      int in_iter	= kk * nloc * ndescrpt * 3;
      int rij_iter	= kk * nloc * nnei * 3;
      int nlist_iter	= kk * nloc * nnei;
      int virial_iter	= kk * 9;
      int atom_virial_iter	= kk * nall * 9;

      deepmd::prod_virial_r_cpu(
	  &virial(virial_iter),
	  &atom_virial(atom_virial_iter),
	  &net_deriv(net_iter),
	  &in_deriv(in_iter),
	  &rij(rij_iter),
	  &nlist(nlist_iter),
	  nloc,
	  nall,
	  nnei);
    }
  }
};

#define REGISTER_CPU(T)                                                                 \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("ProdVirialSeR").Device(DEVICE_CPU).TypeConstraint<T>("T"),                       \
    ProdVirialSeROp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);



