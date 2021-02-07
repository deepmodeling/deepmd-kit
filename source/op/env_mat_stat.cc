#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

#include "ComputeDescriptor.h"
#include "NeighborList.h"

typedef double boxtensor_t ;
typedef double compute_t;

using namespace tensorflow;
// using namespace std;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("EnvMatStat")
    .Attr("T: {float, double}")
    .Input("coord: T")          //atomic coordinates
    .Input("type: int32")       //atomic type
    .Input("natoms: int32")     //local atomic number; each type atomic number; daizheyingxiangqude atomic numbers
    .Input("box : T")
    .Input("mesh : int32")
    .Attr("rcut: float")      //no use
    .Attr("rcut_smth: float")
    .Attr("sel: list(int)")
    .Output("min_nbor_dist: T")
    .Output("max_nbor_size: int32");

template<typename Device, typename FPTYPE>
class EnvMatStatOp : public OpKernel {
public:
  explicit EnvMatStatOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("rcut", &rcut));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_smth", &rcut_smth));
    OP_REQUIRES_OK(context, context->GetAttr("sel", &sel));
    cum_sum (sec, sel);
    ndescrpt = sec.back() * 4;
    nnei = sec.back();
  }

  void Compute(OpKernelContext* context) override {
    counter++;
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& coord_tensor	= context->input(context_input_index++);
    const Tensor& type_tensor	= context->input(context_input_index++);
    const Tensor& natoms_tensor	= context->input(context_input_index++);
    const Tensor& box_tensor	= context->input(context_input_index++);
    const Tensor& mesh_tensor	= context->input(context_input_index++);

    // set size of the sample
    OP_REQUIRES (context, (coord_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of coord should be 2"));
    OP_REQUIRES (context, (type_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of type should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of natoms should be 1"));
    OP_REQUIRES (context, (box_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of box should be 2"));
    OP_REQUIRES (context, (mesh_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of mesh should be 1"));
    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),		errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    auto natoms	= natoms_tensor	.flat<int>();
    int nloc = natoms(0);
    int nall = natoms(1);
    int ntypes = natoms_tensor.shape().dim_size(0) - 2;
    int nsamples = coord_tensor.shape().dim_size(0);

    // check the sizes
    OP_REQUIRES (context, (nsamples == type_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nsamples == box_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nall * 3 == coord_tensor.shape().dim_size(1)),	errors::InvalidArgument ("number of atoms should match"));
    OP_REQUIRES (context, (nall == type_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of atoms should match"));
    OP_REQUIRES (context, (9 == box_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of box should be 9"));

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

    TensorShape min_nbor_dist_shape ;
    min_nbor_dist_shape.AddDim (nloc * nnei);
    TensorShape max_nbor_size_shape ;
    max_nbor_size_shape.AddDim (nloc);
    max_nbor_size_shape.AddDim (ntypes);

    int context_output_index = 0;
    Tensor* min_nbor_dist_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
						     min_nbor_dist_shape,
						     &min_nbor_dist_tensor));
    Tensor* max_nbor_size_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
						     max_nbor_size_shape,
						     &max_nbor_size_tensor));

    auto coord	= coord_tensor	.matrix<FPTYPE>();
    auto type	= type_tensor	.matrix<int>();
    auto box	= box_tensor	.matrix<FPTYPE>();
    auto mesh	= mesh_tensor	.flat<int>();
    auto min_nbor_dist	= min_nbor_dist_tensor ->flat<FPTYPE>();
    // find a potential bug here!
    auto max_nbor_size	= max_nbor_size_tensor ->flat<int>();
    
    for (int ii = 0; ii < static_cast<int>(min_nbor_dist_tensor->NumElements()); ii++) {
      min_nbor_dist(ii) = 10000.0;
    }
    for (int ii = 0; ii < static_cast<int>(max_nbor_size_tensor->NumElements()); ii++) {
      max_nbor_size(ii) = 0;
    }

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
      std::vector<std::vector<int > > d_nlist_a;
      std::vector<std::vector<int > > d_nlist_r;
      std::vector<int> nlist_map;
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
          // std::cout << "I'm in nei_mode 2" << std::endl;
	std::vector<int > nat_stt = {mesh(1-1), mesh(2-1), mesh(3-1)};
	std::vector<int > nat_end = {mesh(4-1), mesh(5-1), mesh(6-1)};
	std::vector<int > ext_stt = {mesh(7-1), mesh(8-1), mesh(9-1)};
	std::vector<int > ext_end = {mesh(10-1), mesh(11-1), mesh(12-1)};
	std::vector<int > global_grid (3);
	for (int dd = 0; dd < 3; ++dd) global_grid[dd] = nat_end[dd] - nat_stt[dd];
	::build_nlist (d_nlist_a, d_nlist_r, d_coord3, nloc, -1, rcut, nat_stt, nat_end, ext_stt, ext_end, region, global_grid);
      }
      else if (nei_mode == 1) {
          // std::cout << "I'm in nei_mode 1" << std::endl;
	std::vector<double > bk_d_coord3 = d_coord3;
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
	::build_nlist (d_nlist_a, d_nlist_r, d_coord3, nloc, -1, rcut, nat_stt, ncell, ext_stt, ext_end, region, ncell);
      }
      else if (nei_mode == -1){
	::build_nlist (d_nlist_a, d_nlist_r, d_coord3, -1, rcut, NULL);
      }
      else {
	throw std::runtime_error("unknow neighbor mode");
      }

  for (int ii = 0; ii < nloc; ii++) {
    for (int jj = 0; jj < d_nlist_r[ii].size(); jj++) {
        int type = d_type[d_nlist_r[ii][jj]];
        max_nbor_size(ii * ntypes + type) += 1;
    }
  }
      // loop over atoms, compute descriptors for each atom
#pragma omp parallel for 
      for (int ii = 0; ii < nloc; ++ii){
	std::vector<int> fmt_nlist_a;
	std::vector<int> fmt_nlist_r;
  std::vector<int> sec_r(sec.size(), 0);

	int ret = -1;
	if ((ret = format_nlist_fill_a (fmt_nlist_a, fmt_nlist_r, d_coord3, ntypes, d_type, region, b_pbc, ii, d_nlist_a[ii], d_nlist_r[ii], rcut, sec, sec_r)) != -1){
	  if (count_nei_idx_overflow == 0) {
	    std::cout << "WARNING: Radial neighbor list length of type " << ret << " is not enough" << std::endl;
	    flush(std::cout);
	    count_nei_idx_overflow ++;
	  }
	}

	std::vector<compute_t > d_rij_a;
	get_rij (d_rij_a,
				 d_coord3,
				 ntypes, 
				 d_type,
				 region, 
				 b_pbc,
				 ii, 
				 fmt_nlist_a,
				 sec, 
				 rcut_smth, 
				 rcut);

	// check sizes
	assert (d_rij_a.size() == nnei * 3);
	assert (int(fmt_nlist_a.size()) == nnei);
	for (int jj = 0; jj < nnei * 3; ++jj){
    if (jj % 3 == 0 && d_rij_a[jj] > 0) {
      min_nbor_dist(ii * nnei + jj / 3) = sqrt(d_rij_a[jj] * d_rij_a[jj] + d_rij_a[jj + 1] * d_rij_a[jj + 1] + d_rij_a[jj + 2] * d_rij_a[jj + 2]);
    }
	}
      }
    }
  }
private:
  int counter = -1;
  float rcut;
  float rcut_smth;
  std::vector<int32> sel;
  std::vector<int> sec;
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
    Name("EnvMatStat").Device(DEVICE_CPU).TypeConstraint<T>("T"),                      \
    EnvMatStatOp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);