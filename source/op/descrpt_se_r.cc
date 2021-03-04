#include "custom_op.h"
#include "ComputeDescriptor.h"
#include "neighbor_list.h"
#include "fmt_nlist.h"
#include "env_mat.h"

typedef double boxtensor_t ;
typedef double compute_t;

REGISTER_OP("DescrptSeR")
.Attr("T: {float, double}")
.Input("coord: T")
.Input("type: int32")
.Input("natoms: int32")
.Input("box: T")
.Input("mesh: int32")
.Input("davg: T")
.Input("dstd: T")
.Attr("rcut: float")
.Attr("rcut_smth: float")
.Attr("sel: list(int)")
.Output("descrpt: T")
.Output("descrpt_deriv: T")
.Output("rij: T")
.Output("nlist: int32");

template<typename Device, typename FPTYPE>
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
      // lammps neighbor list
      nei_mode = 3;
    }
    else if (mesh_tensor.shape().dim_size(0) == 12) {
      // user provided extended mesh
      nei_mode = 2;
    }
    else if (mesh_tensor.shape().dim_size(0) == 6) {
      // manual copied pbc
      assert (nloc == nall);
      nei_mode = 1;
    }
    else if (mesh_tensor.shape().dim_size(0) == 0) {
      // no pbc
      nei_mode = -1;
    }
    else {
      throw std::runtime_error("invalid mesh tensor");
    }
    bool b_pbc = true;
    // if region is given extended, do not use pbc
    if (nei_mode >= 1 || nei_mode == -1) {
      b_pbc = false;
    }
    bool b_norm_atom = false;
    if (nei_mode == 1){
      b_norm_atom = true;
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
    
    auto coord	= coord_tensor	.matrix<FPTYPE>();
    auto type	= type_tensor	.matrix<int>();
    auto box	= box_tensor	.matrix<FPTYPE>();
    auto mesh	= mesh_tensor	.flat<int>();
    auto avg	= avg_tensor	.matrix<FPTYPE>();
    auto std	= std_tensor	.matrix<FPTYPE>();
    auto descrpt	= descrpt_tensor	->matrix<FPTYPE>();
    auto descrpt_deriv	= descrpt_deriv_tensor	->matrix<FPTYPE>();
    auto rij		= rij_tensor		->matrix<FPTYPE>();
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
      std::vector<compute_t > d_coord3 (nall*3);
      for (int ii = 0; ii < nall; ++ii){
	for (int dd = 0; dd < 3; ++dd){
	  d_coord3[ii*3+dd] = coord(kk, ii*3+dd);
	}
	if (b_norm_atom){
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
      std::vector<int > d_type (nall);
      for (int ii = 0; ii < nall; ++ii) d_type[ii] = type(kk, ii);
      
      // build nlist
      std::vector<std::vector<int > > d_nlist;
      std::vector<std::vector<int > > d_nlist_null;
      std::vector<int> nlist_map;
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
	std::vector<int > nat_stt = {mesh(1-1), mesh(2-1), mesh(3-1)};
	std::vector<int > nat_end = {mesh(4-1), mesh(5-1), mesh(6-1)};
	std::vector<int > ext_stt = {mesh(7-1), mesh(8-1), mesh(9-1)};
	std::vector<int > ext_end = {mesh(10-1), mesh(11-1), mesh(12-1)};
	std::vector<int > global_grid (3);
	for (int dd = 0; dd < 3; ++dd) global_grid[dd] = nat_end[dd] - nat_stt[dd];
	::build_nlist (d_nlist_null, d_nlist, d_coord3, nloc, -1, rcut, nat_stt, nat_end, ext_stt, ext_end, region, global_grid);
      }
      else if (nei_mode == 1) {
	std::vector<compute_t > bk_d_coord3 = d_coord3;
	std::vector<int > bk_d_type = d_type;
	std::vector<int > ncell, ngcell;
	copy_coord(d_coord3, d_type, nlist_map, ncell, ngcell, bk_d_coord3, bk_d_type, rcut, region);	
	b_nlist_map = true;
	std::vector<int> nat_stt(3, 0);
	std::vector<int> ext_stt(3), ext_end(3);
	for (int dd = 0; dd < 3; ++dd){
	  ext_stt[dd] = -ngcell[dd];
	  ext_end[dd] = ncell[dd] + ngcell[dd];
	}
	::build_nlist (d_nlist_null, d_nlist, d_coord3, nloc, -1, rcut, nat_stt, ncell, ext_stt, ext_end, region, ncell);
      }
      else if (nei_mode == -1){
	::build_nlist (d_nlist_null, d_nlist, d_coord3, -1, rcut, NULL);
      }
      else {
	throw std::runtime_error("unknow neighbor mode");
      }

      // loop over atoms, compute descriptors for each atom
#pragma omp parallel for 
      for (int ii = 0; ii < nloc; ++ii){
	std::vector<int> fmt_nlist_null;
	std::vector<int> fmt_nlist;
	int ret = -1;
	if (fill_nei_a){
	  if ((ret = format_nlist_i_fill_a (fmt_nlist, fmt_nlist_null, d_coord3, ntypes, d_type, region, b_pbc, ii, d_nlist_null[ii], d_nlist[ii], rcut, sec, sec_null)) != -1){
	    if (count_nei_idx_overflow == 0) {
	      std::cout << "WARNING: Radial neighbor list length of type " << ret << " is not enough" << std::endl;
	      flush(std::cout);
	      count_nei_idx_overflow ++;
	    }
	  }
	}
	// std::cout << ii << " " ;
	// for (int jj = 0 ; jj < fmt_nlist.size(); ++jj){
	//   std::cout << fmt_nlist[jj] << " " ;
	// }
	// std::cout << std::endl;

	std::vector<compute_t > d_descrpt;
	std::vector<compute_t > d_descrpt_deriv;
	std::vector<compute_t > d_rij;
	env_mat_r (d_descrpt,
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
  std::vector<int32> sel;
  std::vector<int32> sel_null;
  std::vector<int> sec;
  std::vector<int> sec_null;
  int ndescrpt;
  int nnei;
  bool fill_nei_a;
  int count_nei_idx_overflow;
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

#define REGISTER_CPU(T)                                                                 \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("DescrptSeR").Device(DEVICE_CPU).TypeConstraint<T>("T"),                       \
    DescrptSeROp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);

