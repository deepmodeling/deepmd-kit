#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

using namespace tensorflow;
using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else
typedef float  VALUETYPE;
#endif

#ifdef HIGH_PREC
REGISTER_OP("TabInter")
.Input("table_info: double")
.Input("table_data: double")
.Input("type: int32")
.Input("rij: double")
.Input("nlist: int32")
.Input("natoms: int32")
.Input("scale: double")
.Attr("sel_a: list(int)")
.Attr("sel_r: list(int)")
.Output("atom_energy: double")
.Output("force: double")
.Output("atom_virial: double");
#else
REGISTER_OP("TabInter")
.Input("table_info: double")
.Input("table_data: double")
.Input("type: int32")
.Input("rij: float")
.Input("nlist: int32")
.Input("natoms: int32")
.Input("scale: float")
.Attr("sel_a: list(int)")
.Attr("sel_r: list(int)")
.Output("atom_energy: float")
.Output("force: float")
.Output("atom_virial: float");
#endif

using namespace tensorflow;

inline 
void tabulated_inter (double & ener, 
		      double & fscale, 
		      const double * table_info,
		      const double * table_data,
		      const double * dr)
{
  // info size: 3
  const double & rmin = table_info[0];
  const double & hh = table_info[1];
  const double hi = 1./hh;
  const unsigned nspline = unsigned(table_info[2] + 0.1);
  const unsigned ndata = nspline * 4;

  double r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
  double rr = sqrt(r2);
  double uu = (rr - rmin) * hi;
  // cout << rr << " " << rmin << " " << hh << " " << uu << endl;
  if (uu < 0) {
    cerr << "coord go beyond table lower boundary" << endl;
    exit(1);
  }
  int idx = uu;
  if (idx >= nspline) {
    fscale = ener = 0;
    return;
  }
  uu -= idx;
  assert(idx >= 0);
  assert(uu >= 0 && uu < 1);

  const double & a3 = table_data[4 * idx + 0];
  const double & a2 = table_data[4 * idx + 1];
  const double & a1 = table_data[4 * idx + 2];
  const double & a0 = table_data[4 * idx + 3];
  
  double etmp = (a3 * uu + a2) * uu + a1;
  ener = etmp * uu + a0;
  fscale = (2. * a3 * uu + a2) * uu + etmp;
  fscale *= -hi;
}

class TabInterOp : public OpKernel {
 public:
  explicit TabInterOp(OpKernelConstruction* context) : OpKernel(context) {
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
    auto table_info = table_info_tensor.flat<VALUETYPE>();
    auto table_data = table_data_tensor.flat<VALUETYPE>();
    auto type	= type_tensor	.matrix<int>();
    auto rij	= rij_tensor	.matrix<VALUETYPE>();
    auto nlist	= nlist_tensor	.matrix<int>();
    auto scale  = scale_tensor	.matrix<VALUETYPE>();
    auto energy = energy_tensor	->matrix<VALUETYPE>();
    auto force	= force_tensor	->matrix<VALUETYPE>();
    auto virial = virial_tensor	->matrix<VALUETYPE>();

    OP_REQUIRES (context, (ntypes == int(table_info(3)+0.1)),	errors::InvalidArgument ("ntypes provided in table does not match deeppot"));
    int nspline = table_info(2)+0.1;
    int tab_stride = 4 * nspline;
    assert(ntypes * ntypes * tab_stride == table_data_tensor.shape().dim_size(0));
    vector<double > d_table_info(4);
    vector<double > d_table_data(ntypes * ntypes * tab_stride);
    for (unsigned ii = 0; ii < d_table_info.size(); ++ii){
      d_table_info[ii] = table_info(ii);
    }
    for (unsigned ii = 0; ii < d_table_data.size(); ++ii){
      d_table_data[ii] = table_data(ii);
    }
    const double * p_table_info = &(d_table_info[0]);
    const double * p_table_data = &(d_table_data[0]);

    // loop over samples
#pragma omp parallel for 
    for (int kk = 0; kk < nframes; ++kk){
      // fill results with 0
      for (int ii = 0; ii < nloc; ++ii){
	int i_idx = ii;
	energy(kk, i_idx) = 0;
      }
      for (int ii = 0; ii < nall; ++ii){
	int i_idx = ii;
	force(kk, i_idx * 3 + 0) = 0;
	force(kk, i_idx * 3 + 1) = 0;
	force(kk, i_idx * 3 + 2) = 0;
	for (int dd = 0; dd < 9; ++dd) {
	  virial(kk, i_idx * 9 + dd) = 0;
	}
      }
      // compute force of a frame
      int i_idx = 0;
      for (int tt = 0; tt < ntypes; ++tt) {
	for (int ii = 0; ii < natoms(2+tt); ++ii){
	  int i_type = type(kk, i_idx);
	  VALUETYPE i_scale = scale(kk, i_idx);
	  assert(i_type == tt) ;
	  int jiter = 0;
	  // a neighbor
	  for (int ss = 0; ss < sel_a.size(); ++ss){
	    int j_type = ss;
	    const double * cur_table_data = 
		p_table_data + (i_type * ntypes + j_type) * tab_stride;
	    for (int jj = 0; jj < sel_a[ss]; ++jj){
	      int j_idx = nlist(kk, i_idx * nnei + jiter);
	      if (j_idx < 0){
		jiter++;
		continue;
	      }
	      assert(j_type == type(kk, j_idx));
	      double dr[3];
	      for (int dd = 0; dd < 3; ++dd){
		dr[dd] = rij(kk, (i_idx * nnei + jiter) * 3 + dd);
	      }
	      double r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
	      double ri = 1./sqrt(r2);
	      double ener, fscale;
	      tabulated_inter(ener,
			      fscale, 
			      p_table_info, 
			      cur_table_data, 
			      dr);
	      // printf("tabforce  %d %d  r: %12.8f  ener: %12.8f %12.8f %8.5f  fj: %8.5f %8.5f %8.5f  dr: %9.6f %9.6f %9.6f\n", 
	      // 	     i_idx, j_idx, 
	      // 	     1/ri,
	      // 	     ener, fscale, i_scale,
	      // 	     -fscale * dr[00] * ri * 0.5 * i_scale,  -fscale * dr[01] * ri * 0.5 * i_scale,  -fscale * dr[02] * ri * 0.5 * i_scale,
	      // 	     dr[0], dr[1], dr[2]
	      // 	  );
	      energy(kk, i_idx) += 0.5 * ener;
	      for (int dd = 0; dd < 3; ++dd) {
		force(kk, i_idx * 3 + dd) -= fscale * dr[dd] * ri * 0.5 * i_scale;
		force(kk, j_idx * 3 + dd) += fscale * dr[dd] * ri * 0.5 * i_scale;
	      }
	      for (int dd0 = 0; dd0 < 3; ++dd0) {
		for (int dd1 = 0; dd1 < 3; ++dd1) {
		  virial(kk, i_idx * 9 + dd0 * 3 + dd1) 
		      += 0.5 * fscale * dr[dd0] * dr[dd1] * ri * 0.5 * i_scale;
		  virial(kk, j_idx * 9 + dd0 * 3 + dd1) 
		      += 0.5 * fscale * dr[dd0] * dr[dd1] * ri * 0.5 * i_scale;
		}
	      }
	      jiter++;
	    }
	  }
	  // r neighbor
	  for (int ss = 0; ss < sel_r.size(); ++ss){
	    int j_type = ss;
	    const double * cur_table_data = 
		p_table_data + (i_type * ntypes + j_type) * tab_stride;
	    for (int jj = 0; jj < sel_r[ss]; ++jj){
	      int j_idx = nlist(kk, i_idx * nnei + jiter);
	      if (j_idx < 0){
		jiter ++;
		continue;
	      }
	      assert(j_type == type(kk, j_idx));
	      double dr[3];
	      for (int dd = 0; dd < 3; ++dd){
		dr[dd] = rij(kk, (i_idx * nnei + jiter) * 3 + dd);
	      }
	      double r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
	      double ri = 1./sqrt(r2);
	      double ener, fscale;
	      tabulated_inter(ener,
			      fscale, 
			      p_table_info, 
			      cur_table_data, 
			      dr);
	      // printf("tabforce  %d %d  %8.5f  %12.8f %12.8f %8.5f  fj: %8.5f %8.5f %8.5f\n", 
	      // 	     i_idx, j_idx, 
	      // 	     1/ri, 
	      // 	     ener, fscale, i_scale,
	      // 	     -fscale * dr[00] * ri * 0.5 * i_scale,  -fscale * dr[01] * ri * 0.5 * i_scale,  -fscale * dr[02] * ri * 0.5 * i_scale);
	      energy(kk, i_idx) += 0.5 * ener;
	      for (int dd = 0; dd < 3; ++dd) {
		force(kk, i_idx * 3 + dd) -= fscale * dr[dd] * ri * 0.5 * i_scale;
		force(kk, j_idx * 3 + dd) += fscale * dr[dd] * ri * 0.5 * i_scale;
	      }
	      for (int dd0 = 0; dd0 < 3; ++dd0) {
		for (int dd1 = 0; dd1 < 3; ++dd1) {
		  virial(kk, j_idx * 9 + dd0 * 3 + dd1) 
		      += fscale * dr[dd0] * dr[dd1] * ri * 0.5 * i_scale;
		}
	      }
	      jiter++;
	    }
	  }
	  i_idx ++;
	}
      }
    }
  }
private:
  vector<int32> sel_r;
  vector<int32> sel_a;
  vector<int> sec_a;
  vector<int> sec_r;
  int nnei, nnei_a, nnei_r;
  void
  cum_sum (vector<int> & sec,
	   const vector<int32> & n_sel) const {
    sec.resize (n_sel.size() + 1);
    sec[0] = 0;
    for (int ii = 1; ii < sec.size(); ++ii){
      sec[ii] = sec[ii-1] + n_sel[ii-1];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("TabInter").Device(DEVICE_CPU), TabInterOp);



