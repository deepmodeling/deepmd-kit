#include "custom_op.h"

REGISTER_OP("ProdVirial")
.Attr("T: {float, double}")
.Input("net_deriv: T")
.Input("in_deriv: T")
.Input("rij: T")
.Input("nlist: int32")
.Input("axis: int32")
.Input("natoms: int32")
.Attr("n_a_sel: int")
.Attr("n_r_sel: int")
.Output("virial: T")
.Output("atom_virial: T");


using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

template<typename Device, typename FPTYPE>
class ProdVirialOp : public OpKernel {
 public:
  explicit ProdVirialOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_a_sel", &n_a_sel));
    OP_REQUIRES_OK(context, context->GetAttr("n_r_sel", &n_r_sel));
    n_a_shift = n_a_sel * 4;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& net_deriv_tensor	= context->input(0);
    const Tensor& in_deriv_tensor	= context->input(1);
    const Tensor& rij_tensor		= context->input(2);
    const Tensor& nlist_tensor		= context->input(3);
    const Tensor& axis_tensor		= context->input(4);
    const Tensor& natoms_tensor		= context->input(5);

    // set size of the sample
    OP_REQUIRES (context, (net_deriv_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of net deriv should be 2"));
    OP_REQUIRES (context, (in_deriv_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of input deriv should be 2"));
    OP_REQUIRES (context, (rij_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of rij should be 2"));
    OP_REQUIRES (context, (nlist_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (axis_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of axis should be 2"));
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
    OP_REQUIRES (context, (nframes == axis_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of samples should match"));

    OP_REQUIRES (context, (nloc * ndescrpt * 12 == in_deriv_tensor.shape().dim_size(1)), errors::InvalidArgument ("number of descriptors should match"));
    OP_REQUIRES (context, (nloc * nnei * 3 == rij_tensor.shape().dim_size(1)),	errors::InvalidArgument ("dim of rij should be nnei * 3"));
    OP_REQUIRES (context, (nnei == n_a_sel + n_r_sel),				errors::InvalidArgument ("number of neighbors should match"));
    OP_REQUIRES (context, (nloc * 4 == axis_tensor.shape().dim_size(1)),	errors::InvalidArgument ("number of axis type+id should be 2+2"));

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
    auto axis = axis_tensor.flat<int>();
    auto virial = virial_tensor->flat<FPTYPE>();
    auto atom_virial = atom_virial_tensor->flat<FPTYPE>();

    // loop over samples
#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk){
      int net_iter	= kk * nloc * ndescrpt;
      int in_iter	= kk * nloc * ndescrpt * 12;
      int rij_iter	= kk * nloc * nnei * 3;
      int nlist_iter	= kk * nloc * nnei;
      int axis_iter	= kk * nloc * 4;
      int virial_iter	= kk * 9;
      int atom_virial_iter	= kk * nall * 9;

      for (int ii = 0; ii < 9; ++ ii){
	virial (virial_iter + ii) = 0.;
      }
      for (int ii = 0; ii < 9 * nall; ++ ii){
	atom_virial (atom_virial_iter + ii) = 0.;
      }

      // compute virial of a frame
      for (int ii = 0; ii < nloc; ++ii){
	int i_idx = ii;
	
	// set axes
	int axis0_type = axis (axis_iter + i_idx * 4 + 0);
	int axis1_type = axis (axis_iter + i_idx * 4 + 2);
	int axis_0  = axis (axis_iter + i_idx * 4 + 1);
	int axis_1  = axis (axis_iter + i_idx * 4 + 3);
	if (axis0_type == 1) axis_0 += n_a_sel;
	if (axis1_type == 1) axis_1 += n_a_sel;

	// deriv wrt neighbors
	for (int jj = 0; jj < nnei; ++jj){
	  int j_idx = nlist (nlist_iter + i_idx * nnei + jj);
	  if (j_idx < 0) continue;
	  if (jj == axis_0) {
	    for (int aa = 0; aa < ndescrpt; ++aa){
	      FPTYPE pref = -1.0 * net_deriv (net_iter + i_idx * ndescrpt + aa);
	      for (int dd0 = 0; dd0 < 3; ++dd0){
		for (int dd1 = 0; dd1 < 3; ++dd1){
		  FPTYPE tmp_v = pref * rij (rij_iter + i_idx * nnei * 3 + jj * 3 + dd1) *  in_deriv (in_iter + i_idx * ndescrpt * 12 + aa * 12 + 3 + dd0);
		  virial (virial_iter + dd0 * 3 + dd1) += tmp_v;
		  atom_virial (atom_virial_iter + j_idx * 9 + dd0 * 3 + dd1) += tmp_v;
		}
	      }
	    }
	  }
	  else if (jj == axis_1) {
	    for (int aa = 0; aa < ndescrpt; ++aa){
	      FPTYPE pref = -1.0 * net_deriv (net_iter + i_idx * ndescrpt + aa);
	      for (int dd0 = 0; dd0 < 3; ++dd0){
		for (int dd1 = 0; dd1 < 3; ++dd1){
		  FPTYPE tmp_v = pref * rij (rij_iter + i_idx * nnei * 3 + jj * 3 + dd1) *  in_deriv (in_iter + i_idx * ndescrpt * 12 + aa * 12 + 6 + dd0);
		  virial (virial_iter + dd0 * 3 + dd1) += tmp_v;
		  atom_virial (atom_virial_iter + j_idx * 9 + dd0 * 3 + dd1) += tmp_v;
		}
	      }
	    }
	  }
	  else {
	    int aa_start, aa_end;
	    make_descript_range (aa_start, aa_end, jj);
	    for (int aa = aa_start; aa < aa_end; ++aa) {
	      FPTYPE pref = -1.0 * net_deriv (net_iter + i_idx * ndescrpt + aa);
	      for (int dd0 = 0; dd0 < 3; ++dd0){
		for (int dd1 = 0; dd1 < 3; ++dd1){
		  FPTYPE tmp_v = pref * rij (rij_iter + i_idx * nnei * 3 + jj * 3 + dd1) *  in_deriv (in_iter + i_idx * ndescrpt * 12 + aa * 12 + 9 + dd0);
		  virial (virial_iter + dd0 * 3 + dd1) += tmp_v;
		  atom_virial (atom_virial_iter + j_idx * 9 + dd0 * 3 + dd1) += tmp_v;
		}
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
    Name("ProdVirial").Device(DEVICE_CPU).TypeConstraint<T>("T"),                       \
    ProdVirialOp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);


