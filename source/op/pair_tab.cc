#include "custom_op.h"
#include "pair_tab.h"

REGISTER_OP("PairTab")
.Attr("T: {float, double}")
.Input("table_info: double")
.Input("table_data: double")
.Input("type: int32")
.Input("rij: T")
.Input("nlist: int32")
.Input("natoms: int32")
.Input("scale: T")
.Attr("sel_a: list(int)")
.Attr("sel_r: list(int)")
.Output("atom_energy: T")
.Output("force: T")
.Output("atom_virial: T");

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

template<typename Device, typename FPTYPE>
class PairTabOp : public OpKernel {
 public:
  explicit PairTabOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("sel_a", &sel_a));
    OP_REQUIRES_OK(context, context->GetAttr("sel_r", &sel_r));
    cum_sum (sec_a, sel_a);
    cum_sum (sec_r, sel_r);
    nnei_a = sec_a.back();
    nnei_r = sec_r.back();
    nnei = nnei_a + nnei_r;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int tmp_idx = 0;
    const Tensor& table_info_tensor	= context->input(tmp_idx++);
    const Tensor& table_data_tensor	= context->input(tmp_idx++);
    const Tensor& type_tensor	= context->input(tmp_idx++);
    const Tensor& rij_tensor	= context->input(tmp_idx++);
    const Tensor& nlist_tensor	= context->input(tmp_idx++);
    const Tensor& natoms_tensor	= context->input(tmp_idx++);
    const Tensor& scale_tensor	= context->input(tmp_idx++);

    // set size of the sample
    OP_REQUIRES (context, (table_info_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of table_info should be 1"));
    OP_REQUIRES (context, (table_data_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of table_data should be 1"));
    OP_REQUIRES (context, (type_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of type should be 2"));
    OP_REQUIRES (context, (rij_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of rij should be 2"));
    OP_REQUIRES (context, (nlist_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of natoms should be 1"));
    OP_REQUIRES (context, (scale_tensor.shape().dims() == 2),		errors::InvalidArgument ("Dim of scale should be 2"));

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
    OP_REQUIRES (context, (nloc == scale_tensor.shape().dim_size(1)),		errors::InvalidArgument ("shape of scale should be nloc"));

    // Create an output tensor
    TensorShape energy_shape ;
    energy_shape.AddDim (nframes);
    energy_shape.AddDim (nloc);
    TensorShape force_shape ;
    force_shape.AddDim (nframes);
    force_shape.AddDim (3 * nall);
    TensorShape virial_shape ;
    virial_shape.AddDim (nframes);
    virial_shape.AddDim (9 * nall);
    Tensor* energy_tensor = NULL;
    Tensor* force_tensor = NULL;
    Tensor* virial_tensor = NULL;
    tmp_idx = 0;
    OP_REQUIRES_OK(context, context->allocate_output(tmp_idx++, energy_shape, &energy_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(tmp_idx++, force_shape,  &force_tensor ));
    OP_REQUIRES_OK(context, context->allocate_output(tmp_idx++, virial_shape, &virial_tensor));
    
    // flat the tensors
    auto table_info = table_info_tensor.flat<FPTYPE>();
    auto table_data = table_data_tensor.flat<FPTYPE>();
    auto type	= type_tensor	.matrix<int>();
    auto rij	= rij_tensor	.matrix<FPTYPE>();
    auto nlist	= nlist_tensor	.matrix<int>();
    auto scale  = scale_tensor	.matrix<FPTYPE>();
    auto energy = energy_tensor	->matrix<FPTYPE>();
    auto force	= force_tensor	->matrix<FPTYPE>();
    auto virial = virial_tensor	->matrix<FPTYPE>();

    OP_REQUIRES (context, (ntypes == int(table_info(3)+0.1)),	errors::InvalidArgument ("ntypes provided in table does not match deeppot"));
    int nspline = table_info(2)+0.1;
    int tab_stride = 4 * nspline;
    assert(ntypes * ntypes * tab_stride == table_data_tensor.shape().dim_size(0));
    std::vector<double > d_table_info(4);
    std::vector<double > d_table_data(ntypes * ntypes * tab_stride);
    for (unsigned ii = 0; ii < d_table_info.size(); ++ii){
      d_table_info[ii] = table_info(ii);
    }
    for (unsigned ii = 0; ii < d_table_data.size(); ++ii){
      d_table_data[ii] = table_data(ii);
    }
    const double * p_table_info = &(d_table_info[0]);
    const double * p_table_data = &(d_table_data[0]);

    std::vector<int > t_sel_a(sel_a.size()), t_sel_r(sel_r.size());
    for (int ii = 0; ii < sel_a.size(); ++ii){
      t_sel_a[ii] = sel_a[ii];
    }
    for (int ii = 0; ii < sel_r.size(); ++ii){
      t_sel_r[ii] = sel_r[ii];
    }
    // loop over samples
#pragma omp parallel for 
    for (int kk = 0; kk < nframes; ++kk){
      deepmd::pair_tab_cpu<FPTYPE>(
	  &energy(kk,0),
	  &force(kk,0),
	  &virial(kk,0),
	  p_table_info,
	  p_table_data,
	  &rij(kk,0),
	  &scale(kk,0),
	  &type(kk,0),
	  &nlist(kk,0),
	  &natoms(0),
	  t_sel_a,
	  t_sel_r);
    }
  }
private:
  std::vector<int32> sel_r;
  std::vector<int32> sel_a;
  std::vector<int> sec_a;
  std::vector<int> sec_r;
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
    Name("PairTab").Device(DEVICE_CPU).TypeConstraint<T>("T"),                      \
    PairTabOp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);

