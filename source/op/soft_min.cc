#include "custom_op.h"
#include "ComputeDescriptor.h"
#include "soft_min_switch.h"

REGISTER_OP("SoftMinSwitch")
.Attr("T: {float, double}")
.Input("type: int32")
.Input("rij: T")
.Input("nlist: int32")
.Input("natoms: int32")
.Attr("sel_a: list(int)")
.Attr("sel_r: list(int)")
.Attr("alpha: float")
.Attr("rmin: float")
.Attr("rmax: float")
.Output("sw_value: T")
.Output("sw_deriv: T");

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

template<typename Device, typename FPTYPE>
class SoftMinSwitchOp : public OpKernel {
 public:
  explicit SoftMinSwitchOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("sel_a", &sel_a));
    OP_REQUIRES_OK(context, context->GetAttr("sel_r", &sel_r));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha));
    OP_REQUIRES_OK(context, context->GetAttr("rmin", &rmin));
    OP_REQUIRES_OK(context, context->GetAttr("rmax", &rmax));
    cum_sum (sec_a, sel_a);
    cum_sum (sec_r, sel_r);
    nnei_a = sec_a.back();
    nnei_r = sec_r.back();
    nnei = nnei_a + nnei_r;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int tmp_idx = 0;
    const Tensor& type_tensor	= context->input(tmp_idx++);
    const Tensor& rij_tensor	= context->input(tmp_idx++);
    const Tensor& nlist_tensor	= context->input(tmp_idx++);
    const Tensor& natoms_tensor	= context->input(tmp_idx++);

    // set size of the sample
    OP_REQUIRES (context, (type_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of type should be 2"));
    OP_REQUIRES (context, (rij_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of rij should be 2"));
    OP_REQUIRES (context, (nlist_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of natoms should be 1"));

    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),	errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    auto natoms	= natoms_tensor	.flat<int>();

    int nframes = type_tensor.shape().dim_size(0);
    int nloc = natoms(0);
    int nall = natoms(1);
    int ntypes = natoms_tensor.shape().dim_size(0) - 2;
    assert(sel_a.size() == ntypes);
    assert(sel_r.size() == ntypes);

    // check the sizes
    OP_REQUIRES (context, (nframes == type_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nframes == rij_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nframes == nlist_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nall == type_tensor.shape().dim_size(1)),		errors::InvalidArgument ("shape of type should be nall"));
    OP_REQUIRES (context, (3 * nnei * nloc == rij_tensor.shape().dim_size(1)),	errors::InvalidArgument ("shape of rij should be 3 * nloc * nnei"));
    OP_REQUIRES (context, (nnei * nloc == nlist_tensor.shape().dim_size(1)),	errors::InvalidArgument ("shape of nlist should be nloc * nnei"));

    // Create an output tensor
    TensorShape sw_value_shape ;
    sw_value_shape.AddDim (nframes);
    sw_value_shape.AddDim (nloc);
    TensorShape sw_deriv_shape ;
    sw_deriv_shape.AddDim (nframes);
    sw_deriv_shape.AddDim (3 * nnei * nloc);
    Tensor* sw_value_tensor = NULL;
    Tensor* sw_deriv_tensor = NULL;
    tmp_idx = 0;
    OP_REQUIRES_OK(context, context->allocate_output(tmp_idx++, sw_value_shape, &sw_value_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(tmp_idx++, sw_deriv_shape, &sw_deriv_tensor ));
    
    // flat the tensors
    auto type	= type_tensor	.matrix<int>();
    auto rij	= rij_tensor	.matrix<FPTYPE>();
    auto nlist	= nlist_tensor	.matrix<int>();
    auto sw_value = sw_value_tensor	->matrix<FPTYPE>();
    auto sw_deriv = sw_deriv_tensor	->matrix<FPTYPE>();

    // loop over samples
#pragma omp parallel for 
    for (int kk = 0; kk < nframes; ++kk){
      deepmd::soft_min_switch_cpu<FPTYPE>(
	  &sw_value(kk, 0),
	  &sw_deriv(kk, 0),
	  &rij(kk, 0),
	  &nlist(kk, 0),
	  nloc,
	  nnei,
	  alpha,
	  rmin,
	  rmax);
    }
  }
private:
  std::vector<int32> sel_r;
  std::vector<int32> sel_a;
  std::vector<int> sec_a;
  std::vector<int> sec_r;
  float alpha, rmin, rmax;
  int nnei, nnei_a, nnei_r;
  void
  cum_sum (std::vector<int> & sec,
	   const std::vector<int32> & n_sel) const {
    sec.resize (n_sel.size() + 1);
    sec[0] = 0;
    for (int ii = 1; ii < sec.size(); ++ii){
      sec[ii] = sec[ii-1] + n_sel[ii-1];
    }
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                                   \
REGISTER_KERNEL_BUILDER(                                                                  \
    Name("SoftMinSwitch").Device(DEVICE_CPU).TypeConstraint<T>("T"),                      \
    SoftMinSwitchOp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);


