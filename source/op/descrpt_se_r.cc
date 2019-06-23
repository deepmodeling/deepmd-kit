#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

#include "ComputeDescriptor.h"
#include "NeighborList.h"

typedef double boxtensor_t ;
typedef double compute_t;

using namespace tensorflow;
using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE ;
#else 
typedef float  VALUETYPE ;
#endif

REGISTER_OP("DescrptSeR")
#ifdef HIGH_PREC
.Input("coord: double")
.Input("type: int32")
.Input("natoms: int32")
.Input("box: double")
.Input("mesh: int32")
.Input("davg: double")
.Input("dstd: double")
.Attr("rcut: float")
.Attr("rcut_smth: float")
.Attr("sel: list(int)")
.Output("descrpt: double")
.Output("descrpt_deriv: double")
.Output("rij: double")
.Output("nlist: int32");
#else
.Input("coord: float")
.Input("type: int32")
.Input("natoms: int32")
.Input("box: float")
.Input("mesh: int32")
.Input("davg: float")
.Input("dstd: float")
.Attr("rcut: float")
.Attr("rcut_smth: float")
.Attr("sel: list(int)")
.Output("descrpt: float")
.Output("descrpt_deriv: float")
.Output("rij: float")
.Output("nlist: int32");
#endif

class DescrptSeROp : public OpKernel {
public:
  explicit DescrptSeROp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("rcut", &rcut));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_smth", &rcut_smth));
    OP_REQUIRES_OK(context, context->GetAttr("sel", &sel));
    cum_sum (sec, sel);
    sel_null.resize(3, 0);
    cum_sum (sec_null, sel_null);
    ndescrpt = sec.back() * 1;
    nnei = sec.back();
    fill_nei_a = true;
    count_nei_idx_overflow = 0;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& coord_tensor	= context->input(context_input_index++);
    const Tensor& type_tensor	= context->input(context_input_index++);
    const Tensor& natoms_tensor	= context->input(context_input_index++);
    const Tensor& box_tensor	= context->input(context_input_index++);
    const Tensor& mesh_tensor	= context->input(context_input_index++);
    const Tensor& avg_tensor	= context->input(context_input_index++);
    const Tensor& std_tensor	= context->input(context_input_index++);

    // set size of the sample
    OP_REQUIRES (context, (coord_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of coord should be 2"));
    OP_REQUIRES (context, (type_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of type should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of natoms should be 1"));
    OP_REQUIRES (context, (box_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of box should be 2"));
    OP_REQUIRES (context, (mesh_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of mesh should be 1"));
    OP_REQUIRES (context, (avg_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of avg should be 2"));
    OP_REQUIRES (context, (std_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of std should be 2"));
    OP_REQUIRES (context, (fill_nei_a),				errors::InvalidArgument ("Rotational free descriptor only support the case rcut_a < 0"));

    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),		errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    auto natoms	= natoms_tensor	.flat<int>();
    int nloc = natoms(0);
    int nall = natoms(1);
    int ntypes = natoms_tensor.shape().dim_size(0) - 2;
    int nsamples = coord_tensor.shape().dim_size(0);

    // check the sizes
    OP_REQUIRES (context, (nsamples == type_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nsamples == box_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (ntypes == avg_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of avg should be ntype"));
    OP_REQUIRES (context, (ntypes == std_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of std should be ntype"));

    OP_REQUIRES (context, (nall * 3 == coord_tensor.shape().dim_size(1)),	errors::InvalidArgument ("number of atoms should match"));
    OP_REQUIRES (context, (nall == type_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of atoms should match"));
    OP_REQUIRES (context, (9 == box_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of box should be 9"));
    OP_REQUIRES (context, (ndescrpt == avg_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of avg should be ndescrpt"));
    OP_REQUIRES (context, (ndescrpt == std_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of std should be ndescrpt"));

    int nei_mode = 0;
    if (mesh_tensor.shape().dim_size(0) == 16) {
      nei_mode = 3;
    }
    else if (mesh_tensor.shape().dim_size(0) == 12) {
      nei_mode = 2;
    }
    else if (mesh_tensor.shape().dim_size(0) == 6) {
      assert (nloc == nall);
      nei_mode = 1;
    }

    // Create an output tensor
    TensorShape descrpt_shape ;
    descrpt_shape.AddDim (nsamples);
    descrpt_shape.AddDim (nloc * ndescrpt);
    TensorShape descrpt_deriv_shape ;
    descrpt_deriv_shape.AddDim (nsamples);
    descrpt_deriv_shape.AddDim (nloc * ndescrpt * 3);
    TensorShape rij_shape ;
    rij_shape.AddDim (nsamples);
    rij_shape.AddDim (nloc * nnei * 3);
    TensorShape nlist_shape ;
    nlist_shape.AddDim (nsamples);
    nlist_shape.AddDim (nloc * nnei);

    int context_output_index = 0;
    Tensor* descrpt_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
						     descrpt_shape, 
						     &descrpt_tensor));
    Tensor* descrpt_deriv_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
						     descrpt_deriv_shape, 
						     &descrpt_deriv_tensor));
    Tensor* rij_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
						     rij_shape,
						     &rij_tensor));
    Tensor* nlist_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
						     nlist_shape,
						     &nlist_tensor));
    
    auto coord	= coord_tensor	.matrix<VALUETYPE>();
    auto type	= type_tensor	.matrix<int>();
    auto box	= box_tensor	.matrix<VALUETYPE>();
    auto mesh	= mesh_tensor	.flat<int>();
    auto avg	= avg_tensor	.matrix<VALUETYPE>();
    auto std	= std_tensor	.matrix<VALUETYPE>();
    auto descrpt	= descrpt_tensor	->matrix<VALUETYPE>();
    auto descrpt_deriv	= descrpt_deriv_tensor	->matrix<VALUETYPE>();
    auto rij		= rij_tensor		->matrix<VALUETYPE>();
    auto nlist		= nlist_tensor		->matrix<int>();

    OP_REQUIRES (context, (ntypes == int(sel.size())),	errors::InvalidArgument ("number of types should match the length of sel array"));

    for (int kk = 0; kk < nsamples; ++kk){
      // set region
      boxtensor_t boxt [9] = {0};
      for (int dd = 0; dd < 9; ++dd) {
	boxt[dd] = box(kk, dd);
      }
      SimulationRegion<compute_t > region;
      region.reinitBox (boxt);

      // set & normalize coord
      vector<compute_t > d_coord3 (nall*3);
      for (int ii = 0; ii < nall; ++ii){
	for (int dd = 0; dd < 3; ++dd){
	  d_coord3[ii*3+dd] = coord(kk, ii*3+dd);
	}
	if (nei_mode <= 1){
	  compute_t inter[3];
	  region.phys2Inter (inter, &d_coord3[3*ii]);
	  for (int dd = 0; dd < 3; ++dd){
	    if      (inter[dd] < 0 ) inter[dd] += 1.;
	    else if (inter[dd] >= 1) inter[dd] -= 1.;
	  }
	  region.inter2Phys (&d_coord3[3*ii], inter);
	}
      }

      // set type
      vector<int > d_type (nall);
      for (int ii = 0; ii < nall; ++ii) d_type[ii] = type(kk, ii);
      
      // build nlist
      vector<vector<int > > d_nlist;
      vector<vector<int > > d_nlist_null;
      vector<int> nlist_map;
      bool b_nlist_map = false;
      if (nei_mode == 3) {	
	int * pilist, *pjrange, *pjlist;
	memcpy (&pilist, &mesh(4), sizeof(int *));
	memcpy (&pjrange, &mesh(8), sizeof(int *));
	memcpy (&pjlist, &mesh(12), sizeof(int *));
	int inum = mesh(1);
	assert (inum == nloc);
	d_nlist_null.resize (inum);
	d_nlist.resize (inum);
	for (unsigned ii = 0; ii < inum; ++ii){
	  d_nlist.reserve (pjrange[inum] / inum + 10);
	}
	for (unsigned ii = 0; ii < inum; ++ii){
	  int i_idx = pilist[ii];
	  for (unsigned jj = pjrange[ii]; jj < pjrange[ii+1]; ++jj){
	    int j_idx = pjlist[jj];
	    d_nlist[i_idx].push_back (j_idx);
	  }
	}
      }
      else if (nei_mode == 2) {
	vector<int > nat_stt = {mesh(1-1), mesh(2-1), mesh(3-1)};
	vector<int > nat_end = {mesh(4-1), mesh(5-1), mesh(6-1)};
	vector<int > ext_stt = {mesh(7-1), mesh(8-1), mesh(9-1)};
	vector<int > ext_end = {mesh(10-1), mesh(11-1), mesh(12-1)};
	vector<int > global_grid (3);
	for (int dd = 0; dd < 3; ++dd) global_grid[dd] = nat_end[dd] - nat_stt[dd];
	::build_nlist (d_nlist_null, d_nlist, d_coord3, nloc, -1, rcut, nat_stt, nat_end, ext_stt, ext_end, region, global_grid);
      }
      else if (nei_mode == 1) {
	vector<double > bk_d_coord3 = d_coord3;
	vector<int > bk_d_type = d_type;
	vector<int > ncell, ngcell;
	copy_coord(d_coord3, d_type, nlist_map, ncell, ngcell, bk_d_coord3, bk_d_type, rcut, region);	
	b_nlist_map = true;
	vector<int> nat_stt(3, 0);
	vector<int> ext_stt(3), ext_end(3);
	for (int dd = 0; dd < 3; ++dd){
	  ext_stt[dd] = -ngcell[dd];
	  ext_end[dd] = ncell[dd] + ngcell[dd];
	}
	::build_nlist (d_nlist_null, d_nlist, d_coord3, nloc, -1, rcut, nat_stt, ncell, ext_stt, ext_end, region, ncell);
      }
      else {
	build_nlist (d_nlist_null, d_nlist, -1, rcut, d_coord3, region);      
      }

      bool b_pbc = true;
      // if region is given extended, do not use pbc
      if (nei_mode >= 1) {
	b_pbc = false;
      }

      // loop over atoms, compute descriptors for each atom
#pragma omp parallel for 
      for (int ii = 0; ii < nloc; ++ii){
	vector<int> fmt_nlist_null;
	vector<int> fmt_nlist;
	int ret = -1;
	if (fill_nei_a){
	  if ((ret = format_nlist_fill_a (fmt_nlist, fmt_nlist_null, d_coord3, ntypes, d_type, region, b_pbc, ii, d_nlist_null[ii], d_nlist[ii], rcut, sec, sec_null)) != -1){
	    if (count_nei_idx_overflow == 0) {
	      cout << "WARNING: Radial neighbor list length of type " << ret << " is not enough" << endl;
	      flush(cout);
	      count_nei_idx_overflow ++;
	    }
	  }
	}
	// cout << ii << " " ;
	// for (int jj = 0 ; jj < fmt_nlist.size(); ++jj){
	//   cout << fmt_nlist[jj] << " " ;
	// }
	// cout << endl;

	vector<compute_t > d_descrpt;
	vector<compute_t > d_descrpt_deriv;
	vector<compute_t > d_rij;
	compute_descriptor_se_r (d_descrpt,
				  d_descrpt_deriv,
				  d_rij,
				  d_coord3,
				  ntypes, 
				  d_type,
				  region, 
				  b_pbc,
				  ii, 
				  fmt_nlist,
				  sec, 
				  rcut_smth, 
				  rcut);

	// check sizes
	assert (d_descrpt_deriv.size() == ndescrpt * 3);
	assert (d_rij.size() == nnei * 3);
	assert (int(fmt_nlist.size()) == nnei);
	// record outputs
	for (int jj = 0; jj < ndescrpt; ++jj) {
	  descrpt(kk, ii * ndescrpt + jj) = (d_descrpt[jj] - avg(d_type[ii], jj)) / std(d_type[ii], jj);
	}
	for (int jj = 0; jj < ndescrpt * 3; ++jj) {
	  descrpt_deriv(kk, ii * ndescrpt * 3 + jj) = d_descrpt_deriv[jj] / std(d_type[ii], jj/3);
	}
	for (int jj = 0; jj < nnei * 3; ++jj){
	  rij (kk, ii * nnei * 3 + jj) = d_rij[jj];
	}
	for (int jj = 0; jj < nnei; ++jj){
	  int record = fmt_nlist[jj];
	  if (b_nlist_map && record >= 0) {
	    record = nlist_map[record];
	  }
	  nlist (kk, ii * nnei + jj) = record;
	}
      }
    }
  }
private:
  float rcut;
  float rcut_smth;
  vector<int32> sel;
  vector<int32> sel_null;
  vector<int> sec;
  vector<int> sec_null;
  int ndescrpt;
  int nnei;
  bool fill_nei_a;
  int count_nei_idx_overflow;
  void 
  cum_sum (vector<int> & sec,
	   const vector<int32> & n_sel) const {
    sec.resize (n_sel.size() + 1);
    sec[0] = 0;
    for (int ii = 1; ii < sec.size(); ++ii){
      sec[ii] = sec[ii-1] + n_sel[ii-1];
    }
  }
  void 
  build_nlist (vector<vector<int > > & nlist0,
	       vector<vector<int > > & nlist1,
	       const compute_t & rc0_,
	       const compute_t & rc1_,
	       const vector<compute_t > & posi3,
	       const SimulationRegion<compute_t > & region) const {
    compute_t rc0 (rc0_);
    compute_t rc1 (rc1_);
    assert (rc0 <= rc1);
    compute_t rc02 = rc0 * rc0;
    // negative rc0 means not applying rc0
    if (rc0 < 0) rc02 = 0;
    compute_t rc12 = rc1 * rc1;

    unsigned natoms = posi3.size()/3;
    nlist0.clear();
    nlist1.clear();
    nlist0.resize(natoms);
    nlist1.resize(natoms);
    for (unsigned ii = 0; ii < natoms; ++ii){
      nlist0[ii].reserve (60);
      nlist1[ii].reserve (60);
    }
    for (unsigned ii = 0; ii < natoms; ++ii){
      for (unsigned jj = ii+1; jj < natoms; ++jj){
	compute_t diff[3];
	region.diffNearestNeighbor (posi3[jj*3+0], posi3[jj*3+1], posi3[jj*3+2],
				    posi3[ii*3+0], posi3[ii*3+1], posi3[ii*3+2],
				    diff[0], diff[1], diff[2]);
	compute_t r2 = MathUtilities::dot<compute_t> (diff, diff);
	if (r2 < rc02) {
	  nlist0[ii].push_back (jj);
	  nlist0[jj].push_back (ii);
	}
	else if (r2 < rc12) {
	  nlist1[ii].push_back (jj);
	  nlist1[jj].push_back (ii);
	}
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("DescrptSeR").Device(DEVICE_CPU), DescrptSeROp);

