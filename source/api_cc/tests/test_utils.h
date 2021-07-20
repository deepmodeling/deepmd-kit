#pragma once
#include <cmath>

inline void 
_fold_back(
    std::vector<double > &out,
    const std::vector<double > &in,
    const std::vector<int> &mapping,
    const int nloc,
    const int nall,
    const int ndim)
{
  out.resize(nloc*ndim);
  std::copy(in.begin(), in.begin() + nloc*ndim, out.begin());
  for(int ii = nloc; ii < nall; ++ii){
    int in_idx = ii;
    int out_idx = mapping[in_idx];
    for(int dd = 0; dd < ndim; ++dd){
      out[out_idx * ndim + dd] += in[in_idx * ndim + dd];
    }
  }
}

inline void
_build_nlist(
    std::vector<std::vector<int>> &nlist_data,
    std::vector<double > & coord_cpy,
    std::vector<int > & atype_cpy,
    std::vector<int > & mapping,
    const std::vector<double > & coord,
    const std::vector<int > & atype,
    const std::vector<double > & box,
    const float & rc)
{
  SimulationRegion<double > region;
  region.reinitBox(&box[0]);
  std::vector<int> ncell, ngcell;
  copy_coord(coord_cpy, atype_cpy, mapping, ncell, ngcell, coord, atype, rc, region);
  std::vector<int> nat_stt, ext_stt, ext_end;
  nat_stt.resize(3);
  ext_stt.resize(3);
  ext_end.resize(3);
  for (int dd = 0; dd < 3; ++dd){
    ext_stt[dd] = -ngcell[dd];
    ext_end[dd] = ncell[dd] + ngcell[dd];
  }
  int nloc = coord.size() / 3;
  int nall = coord_cpy.size() / 3;
  std::vector<std::vector<int>> nlist_r_cpy;
  build_nlist(nlist_data, nlist_r_cpy, coord_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);
}

template<typename VALUETYPE>
class EnergyModelTest
{
  double hh = 1e-5;
  double level = 1e-6;
public:
  virtual void compute (
      VALUETYPE & ener,
      std::vector<VALUETYPE> &	force,
      std::vector<VALUETYPE> &	virial,
      const std::vector<VALUETYPE> & coord,
      const std::vector<VALUETYPE> & box
      ) = 0;
  void test_f (
      const std::vector<VALUETYPE> & coord,
      const std::vector<VALUETYPE> & box) {
    int ndof = coord.size();
    VALUETYPE ener;
    std::vector<VALUETYPE> force, virial;
    compute(ener, force, virial, coord, box);
    for(int ii = 0; ii < ndof; ++ii){
      std::vector<VALUETYPE> coord0(coord), coord1(coord);
      VALUETYPE ener0, ener1;
      std::vector<VALUETYPE> forcet, virialt;
      coord0[ii] += hh;
      coord1[ii] -= hh;
      compute(ener0, forcet, virialt, coord0, box);
      compute(ener1, forcet, virialt, coord1, box);
      VALUETYPE num = - (ener0 - ener1) / (2.*hh);
      VALUETYPE ana = force[ii];
      EXPECT_LT(fabs(num - ana), level);
    }
  }  
  void test_v(
      const std::vector<VALUETYPE> & coord,
      const std::vector<VALUETYPE> & box) {
    std::vector<VALUETYPE> num_diff(9);
    VALUETYPE ener;
    std::vector<VALUETYPE> force, virial;
    compute(ener, force, virial, coord, box);
    deepmd::Region<VALUETYPE> region;
    init_region_cpu(region, &box[0]);
    for(int ii = 0; ii < 9; ++ii){
      std::vector<VALUETYPE> box0(box), box1(box);
      box0[ii] += hh;
      box1[ii] -= hh;
      deepmd::Region<VALUETYPE> region0, region1;
      init_region_cpu(region0, &box0[0]);
      init_region_cpu(region1, &box1[0]);
      std::vector<VALUETYPE> coord0(coord), coord1(coord);
      int natoms = coord.size() / 3;
      for(int ii = 0; ii < natoms; ++ii){
	VALUETYPE pi[3];
	convert_to_inter_cpu(pi, region, &coord[ii*3]);
	convert_to_phys_cpu(&coord0[ii*3], region0, pi);
      }
      for(int ii = 0; ii < natoms; ++ii){
	VALUETYPE pi[3];
	convert_to_inter_cpu(pi, region, &coord[ii*3]);
	convert_to_phys_cpu(&coord1[ii*3], region1, pi);
      }
      VALUETYPE ener0, ener1;
      std::vector<VALUETYPE> forcet, virialt;
      compute(ener0, forcet, virialt, coord0, box0);
      compute(ener1, forcet, virialt, coord1, box1);
      num_diff[ii] = - (ener0 - ener1) / (2.*hh);
    }
    std::vector<VALUETYPE> num_virial(9, 0);
    for(int dd0 = 0; dd0 < 3; ++dd0){
      for(int dd1 = 0; dd1 < 3; ++dd1){
	for(int dd = 0; dd < 3; ++dd){
	  num_virial[dd0*3+dd1] += num_diff[dd*3+dd0] * box[dd*3+dd1];
	  // num_virial[dd0*3+dd1] += num_diff[dd0*3+dd] * box[dd1*3+dd];
	}
      }
    }
    for(int ii = 0; ii < 9; ++ii){
      EXPECT_LT(fabs(num_virial[ii] - virial[ii]), level);
    }
  }
};


