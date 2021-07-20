#include "custom_op.h"
#include "map_aparam.h"

REGISTER_OP("MapAparam")
.Attr("T: {float, double}")
.Input("aparam: T")
.Input("nlist: int32")
.Input("natoms: int32")
.Attr("n_a_sel: int")
.Attr("n_r_sel: int")
.Output("output: T");

template <typename Device, typename FPTYPE>
class MapAparamOp : public OpKernel {
 public:
  explicit MapAparamOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_a_sel", &n_a_sel));
    OP_REQUIRES_OK(context, context->GetAttr("n_r_sel", &n_r_sel));
    n_a_shift = n_a_sel * 4;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& aparam_tensor		= context->input(context_input_index++);
    const Tensor& nlist_tensor		= context->input(context_input_index++);
    const Tensor& natoms_tensor		= context->input(context_input_index++);

    // set size of the sample
    OP_REQUIRES (context, (aparam_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of aparam should be 2"));
    OP_REQUIRES (context, (nlist_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of natoms should be 1"));

    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),	errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    auto natoms	= natoms_tensor	.flat<int>();

    int nframes = aparam_tensor.shape().dim_size(0);
    int nloc = natoms(0);
    int nall = natoms(1);
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;
    int numb_aparam = aparam_tensor.shape().dim_size(1) / nall;

    // check the sizes
    OP_REQUIRES (context, (nframes == nlist_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nnei == n_a_sel + n_r_sel),				errors::InvalidArgument ("number of neighbors should match"));

    // Create an output tensor
    TensorShape output_shape ;
    output_shape.AddDim (nframes);
    output_shape.AddDim (nloc * nnei * numb_aparam);
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    
    // flat the tensors
    auto aparam = aparam_tensor.flat<FPTYPE>();
    auto nlist = nlist_tensor.flat<int>();
    auto output = output_tensor->flat<FPTYPE>();

    // loop over samples
#pragma omp parallel for 
    for (int kk = 0; kk < nframes; ++kk){
      int output_iter	= kk * nloc * nnei * numb_aparam;
      int aparam_iter	= kk * nall * numb_aparam;
      int nlist_iter	= kk * nloc * nnei;
      deepmd::map_aparam_cpu(
	  &output(output_iter),
	  &aparam(aparam_iter),
	  &nlist(nlist_iter),
	  nloc,
	  nnei,
	  numb_aparam);
    }
  }
private:
  int n_r_sel, n_a_sel, n_a_shift;
};

#define REGISTER_CPU(T)                                                                 \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("MapAparam").Device(DEVICE_CPU).TypeConstraint<T>("T"),                        \
    MapAparamOp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);



