#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

using namespace tensorflow;
// using namespace std;

REGISTER_OP("SoftMinForce")
.Attr("T: {float, double}")
.Input("du: T")
.Input("sw_deriv: T")
.Input("nlist: int32")
.Input("natoms: int32")
.Attr("n_a_sel: int")
.Attr("n_r_sel: int")
.Output("force: T");

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

template<typename Device, typename FPTYPE>
class SoftMinForceOp : public OpKernel {
 public:
  explicit SoftMinForceOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_a_sel", &n_a_sel));
    OP_REQUIRES_OK(context, context->GetAttr("n_r_sel", &n_r_sel));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& du_tensor		= context->input(0);
    const Tensor& sw_deriv_tensor	= context->input(1);
    const Tensor& nlist_tensor		= context->input(2);
    const Tensor& natoms_tensor		= context->input(3);

    // set size of the sample
    OP_REQUIRES (context, (du_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of du should be 2"));
    OP_REQUIRES (context, (sw_deriv_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of switch deriv should be 2"));
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
    OP_REQUIRES (context, (nframes == nlist_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));

    OP_REQUIRES (context, (nloc == du_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of du should match"));
    OP_REQUIRES (context, (nloc * nnei * 3 == sw_deriv_tensor.shape().dim_size(1)), errors::InvalidArgument ("number of switch deriv should match"));
    OP_REQUIRES (context, (nnei == n_a_sel + n_r_sel),				errors::InvalidArgument ("number of neighbors should match"));

    // Create an output tensor
    TensorShape force_shape ;
    force_shape.AddDim (nframes);
    force_shape.AddDim (3 * nall);
    Tensor* force_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, force_shape, &force_tensor));
    
    // flat the tensors
    auto du = du_tensor.matrix<FPTYPE>();
    auto sw_deriv = sw_deriv_tensor.matrix<FPTYPE>();
    auto nlist = nlist_tensor.matrix<int>();
    auto force = force_tensor->matrix<FPTYPE>();

    // loop over samples
#pragma omp parallel for 
    for (int kk = 0; kk < nframes; ++kk){
      // set zeros
      for (int ii = 0; ii < nall; ++ii){
	int i_idx = ii;
	force (kk, i_idx * 3 + 0) = 0;
	force (kk, i_idx * 3 + 1) = 0;
	force (kk, i_idx * 3 + 2) = 0;
      }
      // compute force of a frame
      for (int ii = 0; ii < nloc; ++ii){
	int i_idx = ii;	
	for (int jj = 0; jj < nnei; ++jj){	  
	  int j_idx = nlist (kk, i_idx * nnei + jj);
	  if (j_idx < 0) continue;
	  int rij_idx_shift = (ii * nnei + jj) * 3;
	  force(kk, i_idx * 3 + 0) += du(kk, i_idx) * sw_deriv(kk, rij_idx_shift + 0);
	  force(kk, i_idx * 3 + 1) += du(kk, i_idx) * sw_deriv(kk, rij_idx_shift + 1);
	  force(kk, i_idx * 3 + 2) += du(kk, i_idx) * sw_deriv(kk, rij_idx_shift + 2);
	  force(kk, j_idx * 3 + 0) -= du(kk, i_idx) * sw_deriv(kk, rij_idx_shift + 0);
	  force(kk, j_idx * 3 + 1) -= du(kk, i_idx) * sw_deriv(kk, rij_idx_shift + 1);
	  force(kk, j_idx * 3 + 2) -= du(kk, i_idx) * sw_deriv(kk, rij_idx_shift + 2);
	  // std::cout << "soft_min_force " << i_idx << " " << j_idx << " " 
	  //      << du(kk, i_idx) << " " 
	  //      << du(kk, i_idx) * sw_deriv(kk, rij_idx_shift + 0)
	  //      << std::endl;
	}
      }
    }
  }
private:
  int n_r_sel, n_a_sel;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                                   \
REGISTER_KERNEL_BUILDER(                                                                  \
    Name("SoftMinForce").Device(DEVICE_CPU).TypeConstraint<T>("T"),                      \
    SoftMinForceOp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);
