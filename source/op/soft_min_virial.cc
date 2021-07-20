#include "custom_op.h"
#include "soft_min_switch_virial.h"

REGISTER_OP("SoftMinVirial")
.Attr("T: {float, double}")
.Input("du: T")
.Input("sw_deriv: T")
.Input("rij: T")
.Input("nlist: int32")
.Input("natoms: int32")
.Attr("n_a_sel: int")
.Attr("n_r_sel: int")
.Output("virial: T")
.Output("atom_virial: T");

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

template<typename Device, typename FPTYPE>
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
    auto du = du_tensor.matrix<FPTYPE>();
    auto sw_deriv = sw_deriv_tensor.matrix<FPTYPE>();
    auto rij = rij_tensor.matrix<FPTYPE>();
    auto nlist = nlist_tensor.matrix<int>();
    auto virial = virial_tensor->matrix<FPTYPE>();
    auto atom_virial = atom_virial_tensor->matrix<FPTYPE>();

    // loop over samples
#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk){
      deepmd::soft_min_switch_virial_cpu(
	  &virial(kk,0),
	  &atom_virial(kk,0),
	  &du(kk,0),
	  &sw_deriv(kk,0),
	  &rij(kk,0),
	  &nlist(kk,0),
	  nloc,
	  nall,
	  nnei);
    }
  }
private:
  int n_r_sel, n_a_sel;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                                   \
REGISTER_KERNEL_BUILDER(                                                                  \
    Name("SoftMinVirial").Device(DEVICE_CPU).TypeConstraint<T>("T"),                      \
    SoftMinVirialOp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);


