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

#ifdef HIGH_PREC
REGISTER_OP("Descrpt")
.Input("coord: double")
.Input("type: int32")
.Input("natoms: int32")
.Input("box: double")
.Input("mesh: int32")
.Input("davg: double")
.Input("dstd: double")
.Attr("rcut_a: float")
.Attr("rcut_r: float")
.Attr("sel_a: list(int)")
.Attr("sel_r: list(int)")
.Attr("axis_rule: list(int)")
.Output("descrpt: double")
.Output("descrpt_deriv: double")
.Output("rij: double")
.Output("nlist: int32")
.Output("axis: int32")
.Output("rot_mat: double");
#else
REGISTER_OP("Descrpt")
.Input("coord: float")
.Input("type: int32")
.Input("natoms: int32")
.Input("box: float")
.Input("mesh: int32")
.Input("davg: float")
.Input("dstd: float")
.Attr("rcut_a: float")
.Attr("rcut_r: float")
.Attr("sel_a: list(int)")
.Attr("sel_r: list(int)")
.Attr("axis_rule: list(int)")
.Output("descrpt: float")
.Output("descrpt_deriv: float")
.Output("rij: float")
.Output("nlist: int32")
.Output("axis: int32")
.Output("rot_mat: float");
#endif

class DescrptOp : public OpKernel {
public:
  explicit DescrptOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("rcut_a", &rcut_a));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_r", &rcut_r));
    OP_REQUIRES_OK(context, context->GetAttr("sel_a", &sel_a));
    OP_REQUIRES_OK(context, context->GetAttr("sel_r", &sel_r));
    OP_REQUIRES_OK(context, context->GetAttr("axis_rule", &axis_rule));
    cum_sum (sec_a, sel_a);
    cum_sum (sec_r, sel_r);
    ndescrpt_a = sec_a.back() * 4;
    ndescrpt_r = sec_r.back() * 1;
    ndescrpt = ndescrpt_a + ndescrpt_r;
    nnei_a = sec_a.back();
    nnei_r = sec_r.back();
    nnei = nnei_a + nnei_r;
    fill_nei_a = (rcut_a < 0);
    count_nei_idx_overflow = 0;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& coord_tensor	= context->input(0);
    const Tensor& type_tensor	= context->input(1);
    const Tensor& natoms_tensor	= context->input(2);
    const Tensor& box_tensor	= context->input(3);
    const Tensor& mesh_tensor	= context->input(4);
    const Tensor& avg_tensor	= context->input(5);
    const Tensor& std_tensor	= context->input(6);

    // set size of the sample
    OP_REQUIRES (context, (coord_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of coord should be 2"));
    OP_REQUIRES (context, (type_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of type should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of natoms should be 1"));
    OP_REQUIRES (context, (box_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of box should be 2"));
    OP_REQUIRES (context, (mesh_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of mesh should be 1"));
    OP_REQUIRES (context, (avg_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of avg should be 2"));
    OP_REQUIRES (context, (std_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of std should be 2"));

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
    descrpt_deriv_shape.AddDim (nloc * ndescrpt * 12);
    TensorShape rij_shape ;
    rij_shape.AddDim (nsamples);
    rij_shape.AddDim (nloc * nnei * 3);
    TensorShape nlist_shape ;
    nlist_shape.AddDim (nsamples);
    nlist_shape.AddDim (nloc * nnei);
    TensorShape axis_shape ;
    axis_shape.AddDim (nsamples);
    axis_shape.AddDim (nloc * 4);
    TensorShape rot_mat_shape ;
    rot_mat_shape.AddDim (nsamples);
    rot_mat_shape.AddDim (nloc * 9);

    Tensor* descrpt_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, descrpt_shape, &descrpt_tensor));
    Tensor* descrpt_deriv_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, descrpt_deriv_shape, &descrpt_deriv_tensor));
    Tensor* rij_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, rij_shape, &rij_tensor));
    Tensor* nlist_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, nlist_shape, &nlist_tensor));
    Tensor* axis_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, axis_shape, &axis_tensor));
    Tensor* rot_mat_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, rot_mat_shape, &rot_mat_tensor));
    
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
    auto axis		= axis_tensor		->matrix<int>();
    auto rot_mat	= rot_mat_tensor		->matrix<VALUETYPE>();

    // // check the types
    // int max_type_v = 0;
    // for (int ii = 0; ii < natoms; ++ii){
    //   if (type(0, ii) > max_type_v) max_type_v = type(0, ii);
    // }
    // int ntypes = max_type_v + 1;
    OP_REQUIRES (context, (ntypes == int(sel_a.size())),	errors::InvalidArgument ("number of types should match the length of sel array"));
    OP_REQUIRES (context, (ntypes == int(sel_r.size())),	errors::InvalidArgument ("number of types should match the length of sel array"));

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
      vector<vector<int > > d_nlist_a;
      vector<vector<int > > d_nlist_r;
      vector<int> nlist_map;
      bool b_nlist_map = false;
      if (nei_mode == 3) {	
	int * pilist, *pjrange, *pjlist;
	memcpy (&pilist, &mesh(4), sizeof(int *));
	memcpy (&pjrange, &mesh(8), sizeof(int *));
	memcpy (&pjlist, &mesh(12), sizeof(int *));
	int inum = mesh(1);
	assert (inum == nloc);
	d_nlist_a.resize (inum);
	d_nlist_r.resize (inum);
	for (unsigned ii = 0; ii < inum; ++ii){
	  d_nlist_r.reserve (pjrange[inum] / inum + 10);
	}
	for (unsigned ii = 0; ii < inum; ++ii){
	  int i_idx = pilist[ii];
	  for (unsigned jj = pjrange[ii]; jj < pjrange[ii+1]; ++jj){
	    int j_idx = pjlist[jj];
	    d_nlist_r[i_idx].push_back (j_idx);
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
	::build_nlist (d_nlist_a, d_nlist_r, d_coord3, nloc, rcut_a, rcut_r, nat_stt, nat_end, ext_stt, ext_end, region, global_grid);
      }
      else if (nei_mode == 1) {
	vector<double > bk_d_coord3 = d_coord3;
	vector<int > bk_d_type = d_type;
	vector<int > ncell, ngcell;
	copy_coord(d_coord3, d_type, nlist_map, ncell, ngcell, bk_d_coord3, bk_d_type, rcut_r, region);	
	b_nlist_map = true;
	vector<int> nat_stt(3, 0);
	vector<int> ext_stt(3), ext_end(3);
	for (int dd = 0; dd < 3; ++dd){
	  ext_stt[dd] = -ngcell[dd];
	  ext_end[dd] = ncell[dd] + ngcell[dd];
	}
	::build_nlist (d_nlist_a, d_nlist_r, d_coord3, nloc, rcut_a, rcut_r, nat_stt, ncell, ext_stt, ext_end, region, ncell);
      }
      else {
	build_nlist (d_nlist_a, d_nlist_r, rcut_a, rcut_r, d_coord3, region);      
      }

      bool b_pbc = true;
      // if region is given extended, do not use pbc
      if (nei_mode >= 1) {
	b_pbc = false;
      }

      // loop over atoms, compute descriptors for each atom
#pragma omp parallel for 
      for (int ii = 0; ii < nloc; ++ii){
	vector<int> fmt_nlist_a;
	vector<int> fmt_nlist_r;
	int ret = -1;
	if (fill_nei_a){
	  if ((ret = format_nlist_fill_a (fmt_nlist_a, fmt_nlist_r, d_coord3, ntypes, d_type, region, b_pbc, ii, d_nlist_a[ii], d_nlist_r[ii], rcut_r, sec_a, sec_r)) != -1){
	    if (count_nei_idx_overflow == 0) {
	      cout << "WARNING: Radial neighbor list length of type " << ret << " is not enough" << endl;
	      flush(cout);
	      count_nei_idx_overflow ++;
	    }
	  }
	}

	// set axis
	vector<int> d_axis_type (2);
	vector<int> d_axis_idx  (2);
	make_axis (d_axis_type, d_axis_idx, d_type[ii], axis_rule, ii, fmt_nlist_a, fmt_nlist_r, d_coord3, region, b_pbc);
	// cout << ii  << " type " << d_type[ii] 
	//      << " axis 0: " << d_axis_type[0] << " " << d_axis_idx[0] 
	//      << " axis 1: " << d_axis_type[1] << " " << d_axis_idx[1] << endl;

	vector<compute_t > d_descrpt_a;
	vector<compute_t > d_descrpt_a_deriv;
	vector<compute_t > d_descrpt_r;
	vector<compute_t > d_descrpt_r_deriv;
	vector<compute_t > d_rij_a;
	vector<compute_t > d_rij_r;
	vector<compute_t > rot;
	compute_descriptor (d_descrpt_a,
			    d_descrpt_a_deriv,
			    d_descrpt_r,
			    d_descrpt_r_deriv,
			    d_rij_a,
			    d_rij_r,
			    rot,
			    d_coord3,
			    ntypes, 
			    d_type,
			    region, 
			    b_pbc,
			    ii, 
			    fmt_nlist_a,
			    fmt_nlist_r,
			    sec_a, 
			    sec_r, 
			    d_axis_type[0],
			    d_axis_idx [0],
			    d_axis_type[1],
			    d_axis_idx [1]);
	// check sizes
	assert (d_descrpt_a.size() == ndescrpt_a);
	assert (d_descrpt_r.size() == ndescrpt_r);
	assert (d_descrpt_a_deriv.size() == ndescrpt_a * 12);
	assert (d_descrpt_r_deriv.size() == ndescrpt_r * 12);
	assert (d_rij_a.size() == nnei_a * 3);
	assert (d_rij_r.size() == nnei_r * 3);
	assert (int(fmt_nlist_a.size()) == nnei_a);
	assert (int(fmt_nlist_r.size()) == nnei_r);
	// record outputs
	for (int jj = 0; jj < ndescrpt_a; ++jj) {
	  descrpt(kk, ii * ndescrpt + jj) = (d_descrpt_a[jj] - avg(d_type[ii], jj)) / std(d_type[ii], jj);
	}
	for (int jj = 0; jj < ndescrpt_r; ++jj) {
	  descrpt(kk, ii * ndescrpt + ndescrpt_a + jj) = (d_descrpt_r[jj] - avg(d_type[ii], ndescrpt_a + jj)) / std(d_type[ii], ndescrpt_a + jj);
	}
	for (int jj = 0; jj < ndescrpt_a * 12; ++jj) {
	  descrpt_deriv(kk, ii * ndescrpt * 12 + jj) = d_descrpt_a_deriv[jj] / std(d_type[ii], jj/12);
	}
	for (int jj = 0; jj < ndescrpt_r * 12; ++jj) {
	  descrpt_deriv(kk, ii * ndescrpt * 12 + ndescrpt_a * 12 + jj) = d_descrpt_r_deriv[jj] / std(d_type[ii], jj/12 + ndescrpt_a);
	}
	for (int jj = 0; jj < 9; ++jj){
	  rot_mat(kk, ii * 9 + jj) = rot[jj];
	}
	for (int jj = 0; jj < nnei_a * 3; ++jj){
	  rij (kk, ii * nnei * 3 + jj) = d_rij_a[jj];
	}
	for (int jj = 0; jj < nnei_r * 3; ++jj){
	  rij (kk, ii * nnei * 3 + nnei_a * 3 + jj) = d_rij_r[jj];
	}
	for (int jj = 0; jj < nnei_a; ++jj){
	  int record = fmt_nlist_a[jj];
	  if (b_nlist_map && record >= 0) {
	    record = nlist_map[record];
	  }
	  nlist (kk, ii * nnei + jj) = record;
	}
	for (int jj = 0; jj < nnei_r; ++jj){
	  int record = fmt_nlist_r[jj];
	  if (b_nlist_map && record >= 0) {
	    record = nlist_map[record];
	  }
	  nlist (kk, ii * nnei + nnei_a + jj) = record;
	}
	for (int jj = 0; jj < 2; ++jj){
	  axis (kk, ii * 4 + jj * 2 + 0) = d_axis_type[jj];
	  axis (kk, ii * 4 + jj * 2 + 1) = d_axis_idx [jj];
	}
      }
    }
  }
private:
  float rcut_a;
  float rcut_r;
  vector<int32> sel_r;
  vector<int32> sel_a;
  vector<int32> axis_rule;
  vector<int> sec_a;
  vector<int> sec_r;
  int ndescrpt, ndescrpt_a, ndescrpt_r;
  int nnei, nnei_a, nnei_r;
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
  void 
  make_axis (vector<int > & axis_type,
	     vector<int > & axis_idx,
	     const int & type,
	     const vector<int > & rule, 
	     const int ii,
	     const vector<int> & nlist_a,
	     const vector<int> & nlist_r,
	     const vector<compute_t> & coord3,
	     const SimulationRegion<compute_t > & region, 
	     const bool b_pbc) const {
    int backup_axis = -1;
    if (rule.size() == 0){
      make_axis_default (axis_type, axis_idx);
    }
    else{
      int ntypes = sel_a.size();
      // two axis, for each axis (a_or_r, type, id)
      assert(rule.size() == ntypes * 2 * 3);
      axis_type.resize(2);
      axis_idx .resize(2);
      vector<int>::const_iterator iter;
      iter = rule.begin() + type * 6;
      if (*(iter+1) >= 0) {
	make_one_axis (axis_type[0], axis_idx[0], iter);
      }
      else {
	make_one_axis (axis_type[0], axis_idx[0], iter, ii, nlist_a, nlist_r, coord3, region, b_pbc);
      }
      iter = rule.begin() + type * 6 + 3;
      if (*(iter+1) >= 0) {      
	make_one_axis (axis_type[1], axis_idx[1], iter);
      }
      else {
	make_one_axis (axis_type[1], axis_idx[1], iter, ii, nlist_a, nlist_r, coord3, region, b_pbc);
      }
      vector<int > backup_rule (3);
      copy (iter, iter+3, backup_rule.begin());
      backup_rule[2] ++;
      if (*(iter+1) >= 0) {      
	make_one_axis (axis_type[1], backup_axis, backup_rule.begin());
      }
      else {
	make_one_axis (axis_type[1], backup_axis, backup_rule.begin(), ii, nlist_a, nlist_r, coord3, region, b_pbc);
      }      
    }
    if (! check_axis (axis_type, axis_idx, ii, nlist_a, nlist_r, coord3, region, b_pbc)){
      if (backup_axis >= 0){
	axis_idx[1] = backup_axis;
      }
      else {
	axis_idx[1] ++;
	// cerr << "wrong backup axis, exit" << endl;
	// exit (1);
      }
    }
    for (int dd = 0; dd < 2; ++dd){
      if (axis_type[dd] == 0) {
	assert (nlist_a[axis_idx[dd]] >= 0);
      }
      else {
	assert (nlist_r[axis_idx[dd]] >= 0);
      }
    }
  }	     
  void
  make_one_axis (int & axis_type, 
		 int & axis_idx,
		 vector<int>::const_iterator info_i) const {
    axis_type = *info_i;
    if (axis_type == 0){
      axis_idx = sec_a[*(info_i+1)] + *(info_i+2);
    }
    else {
      axis_idx = sec_r[*(info_i+1)] + *(info_i+2);
    }
  }		 
  void
  make_one_axis (int & axis_type, 
		 int & axis_idx,
		 vector<int>::const_iterator info_i, 
		 const int id,
		 const vector<int> & nlist_a,
		 const vector<int> & nlist_r,
		 const vector<compute_t> & coord3,
		 const SimulationRegion<compute_t > & region, 
		 const bool b_pbc) const {
    axis_type = *info_i;
    if (axis_type == 0){
      vector<pair<compute_t, int> > sort_info;
      int excl_type = - (*(info_i+1) + 1);
      int ntypes = sel_a.size();
      for (unsigned ii = 0; ii < ntypes; ++ii){
	if (ii == excl_type) continue;
	compute_t diff[3];
	int list_idx, jd;
	// push axis candidates into sort_info
	for (int count = 0; count < 3; ++count){
	  list_idx = sec_a[ii] + count;
	  if (list_idx >= sec_a[ii+1]) continue;
	  jd = nlist_a[list_idx];
	  if (jd < 0) continue;
	  if (b_pbc){
	    region.diffNearestNeighbor (coord3[3*id+0], coord3[3*id+1], coord3[3*id+2],
					coord3[3*jd+0], coord3[3*jd+1], coord3[3*jd+2],
					diff[0], diff[1], diff[2]);
	  }
	  else {
	    for (int dd = 0; dd < 3; ++dd){
	      diff[dd] = coord3[3*id+dd] - coord3[3*jd+dd];
	    }
	  }
	  sort_info.push_back (pair<compute_t, int> 
			       (MathUtilities::dot<compute_t> (diff, diff), list_idx) );
	}
      }
      sort (sort_info.begin(), sort_info.end());
      assert (*(info_i+2) < sort_info.size());
      axis_idx = sort_info[*(info_i+2)].second;
    }
    else {
      vector<pair<compute_t, int> > sort_info;
      int excl_type = - *(info_i+1);
      int ntypes = sel_r.size();
      for (unsigned ii = 0; ii < ntypes; ++ii){
	if (ii == excl_type) continue;
	compute_t diff[3];
	int list_idx, jd;
	// push axis candidates for sort_info
	for (int count = 0; count < 3; ++count){
	  list_idx = sec_r[ii] + count;
	  if (list_idx >= sec_r[ii+1]) continue;
	  jd = nlist_r[list_idx];
	  if (jd < 0) continue;
	  if (b_pbc) {
	    region.diffNearestNeighbor (coord3[3*id+0], coord3[3*id+1], coord3[3*id+2],
					coord3[3*jd+0], coord3[3*jd+1], coord3[3*jd+2],
					diff[0], diff[1], diff[2]);
	  }
	  else {
	    for (int dd = 0; dd < 3; ++dd){
	      diff[dd] = coord3[3*id+dd] - coord3[3*jd+dd];
	    }
	  }
	  sort_info.push_back (pair<compute_t, int> 
			       (MathUtilities::dot<compute_t> (diff, diff), list_idx) );
	}
      }
      sort (sort_info.begin(), sort_info.end());
      assert (*(info_i+2) < sort_info.size());
      axis_idx = sort_info[*(info_i+2)].second;
    }
  }		 
  void 
  make_axis_default (vector<int > & axis_type,
		     vector<int > & axis_idx) const {
    axis_type.resize(2);
    axis_idx .resize(2);
    if (nnei_a > 1) {
      // use angular neighbors
      axis_type[0] = 0;
      axis_type[1] = 0;
    }
    else {
      // use radial neighbors
      axis_type[0] = 1;
      axis_type[1] = 1;
    }
    axis_idx[0] = 0;
    axis_idx[1] = 1;    
  }
  bool 
  check_axis (const vector<int > & axis_type,
	      const vector<int > & axis_idx,
	      const int id,
	      const vector<int> & nlist_a,
	      const vector<int> & nlist_r,
	      const vector<compute_t> & coord3,
	      const SimulationRegion<compute_t > & region, 
	      const bool b_pbc) const {
    compute_t diff[2][3];
    for (int ii = 0; ii < 2; ++ii){
      int jd = 0;
      if (axis_type[ii] == 0) {
	jd = nlist_a[axis_idx[ii]];
      }
      else {
	jd = nlist_r[axis_idx[ii]];
      }
      if (b_pbc){
	region.diffNearestNeighbor (&coord3[3*id], &coord3[3*jd], diff[ii]);
      }
      else {
	for (int dd = 0; dd < 3; ++dd){
	  diff[ii][dd] = coord3[3*id+dd] - coord3[3*jd+dd];
	}
      }
    }
    compute_t rij = MathUtilities::dot (diff[0], diff[1]);
    compute_t rii = MathUtilities::dot (diff[0], diff[0]);
    compute_t rjj = MathUtilities::dot (diff[1], diff[1]);
    if ( fabs (rij / sqrt(rii * rjj) + 1) < 1e-4  ) {
      return false;
    }
    else {
      return true;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Descrpt").Device(DEVICE_CPU), DescrptOp);

