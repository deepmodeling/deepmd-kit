// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cmath>

#include "SimulationRegion.h"
#include "gtest/gtest.h"
#include "neighbor_list.h"
#include "region.h"

#define EPSILON (std::is_same<VALUETYPE, double>::value ? 1e-10 : 1e-4)

typedef testing::Types<double, float> ValueTypes;

template <typename VALUETYPE>
inline void _fold_back(typename std::vector<VALUETYPE>::iterator out,
                       const typename std::vector<VALUETYPE>::const_iterator in,
                       const std::vector<int> &mapping,
                       const int nloc,
                       const int nall,
                       const int ndim,
                       const int nframes = 1) {
  // out.resize(nloc*ndim);
  for (int kk = 0; kk < nframes; ++kk) {
    std::copy(in + kk * nall * ndim, in + kk * nall * ndim + nloc * ndim,
              out + kk * nloc * ndim);
    for (int ii = nloc; ii < nall; ++ii) {
      int in_idx = ii;
      int out_idx = mapping[in_idx];
      for (int dd = 0; dd < ndim; ++dd) {
        *(out + kk * nloc * ndim + out_idx * ndim + dd) +=
            *(in + kk * nall * ndim + in_idx * ndim + dd);
      }
    }
  }
}

template <typename VALUETYPE>
inline void _fold_back(std::vector<VALUETYPE> &out,
                       const std::vector<VALUETYPE> &in,
                       const std::vector<int> &mapping,
                       const int nloc,
                       const int nall,
                       const int ndim,
                       const int nframes = 1) {
  out.resize(static_cast<size_t>(nframes) * nloc * ndim);
  _fold_back<VALUETYPE>(out.begin(), in.begin(), mapping, nloc, nall, ndim,
                        nframes);
}

template <typename VALUETYPE>
inline void _build_nlist(std::vector<std::vector<int>> &nlist_data,
                         std::vector<VALUETYPE> &coord_cpy,
                         std::vector<int> &atype_cpy,
                         std::vector<int> &mapping,
                         const std::vector<VALUETYPE> &coord,
                         const std::vector<int> &atype,
                         const std::vector<VALUETYPE> &box,
                         const float &rc) {
  // convert VALUETYPE to double, it looks like copy_coord only accepts double
  std::vector<double> coord_cpy_;
  std::vector<double> coord_(coord.begin(), coord.end());
  std::vector<double> box_(box.begin(), box.end());

  SimulationRegion<double> region;
  region.reinitBox(&box_[0]);
  std::vector<int> ncell, ngcell;
  copy_coord(coord_cpy_, atype_cpy, mapping, ncell, ngcell, coord_, atype, rc,
             region);
  std::vector<int> nat_stt, ext_stt, ext_end;
  nat_stt.resize(3);
  ext_stt.resize(3);
  ext_end.resize(3);
  for (int dd = 0; dd < 3; ++dd) {
    ext_stt[dd] = -ngcell[dd];
    ext_end[dd] = ncell[dd] + ngcell[dd];
  }
  int nloc = coord_.size() / 3;
  int nall = coord_cpy_.size() / 3;
  std::vector<std::vector<int>> nlist_r_cpy;
  build_nlist(nlist_data, nlist_r_cpy, coord_cpy_, nloc, rc, rc, nat_stt, ncell,
              ext_stt, ext_end, region, ncell);

  // convert double to VALUETYPE
  coord_cpy.assign(coord_cpy_.begin(), coord_cpy_.end());
}

template <typename VALUETYPE>
class EnergyModelTest {
  double hh = std::is_same<VALUETYPE, double>::value ? 1e-5 : 1e-2;
  double level =
      std::is_same<VALUETYPE, double>::value ? 1e-6 : 1e-2;  // expected?
 public:
  virtual void compute(double &ener,
                       std::vector<VALUETYPE> &force,
                       std::vector<VALUETYPE> &virial,
                       const std::vector<VALUETYPE> &coord,
                       const std::vector<VALUETYPE> &box) = 0;
  void test_f(const std::vector<VALUETYPE> &coord,
              const std::vector<VALUETYPE> &box) {
    int ndof = coord.size();
    double ener;
    std::vector<VALUETYPE> force, virial;
    compute(ener, force, virial, coord, box);
    for (int ii = 0; ii < ndof; ++ii) {
      std::vector<VALUETYPE> coord0(coord), coord1(coord);
      double ener0, ener1;
      std::vector<VALUETYPE> forcet, virialt;
      coord0[ii] += hh;
      coord1[ii] -= hh;
      compute(ener0, forcet, virialt, coord0, box);
      compute(ener1, forcet, virialt, coord1, box);
      VALUETYPE num = -(ener0 - ener1) / (2. * hh);
      VALUETYPE ana = force[ii];
      EXPECT_LT(fabs(num - ana), level);
    }
  }
  void test_v(const std::vector<VALUETYPE> &coord,
              const std::vector<VALUETYPE> &box) {
    std::vector<VALUETYPE> num_diff(9);
    double ener;
    std::vector<VALUETYPE> force, virial;
    compute(ener, force, virial, coord, box);
    deepmd::Region<VALUETYPE> region;
    init_region_cpu(region, &box[0]);
    for (int ii = 0; ii < 9; ++ii) {
      std::vector<VALUETYPE> box0(box), box1(box);
      box0[ii] += hh;
      box1[ii] -= hh;
      deepmd::Region<VALUETYPE> region0, region1;
      init_region_cpu(region0, &box0[0]);
      init_region_cpu(region1, &box1[0]);
      std::vector<VALUETYPE> coord0(coord), coord1(coord);
      int natoms = coord.size() / 3;
      for (int ii = 0; ii < natoms; ++ii) {
        VALUETYPE pi[3];
        convert_to_inter_cpu(pi, region, &coord[ii * 3]);
        convert_to_phys_cpu(&coord0[ii * 3], region0, pi);
      }
      for (int ii = 0; ii < natoms; ++ii) {
        VALUETYPE pi[3];
        convert_to_inter_cpu(pi, region, &coord[ii * 3]);
        convert_to_phys_cpu(&coord1[ii * 3], region1, pi);
      }
      double ener0, ener1;
      std::vector<VALUETYPE> forcet, virialt;
      compute(ener0, forcet, virialt, coord0, box0);
      compute(ener1, forcet, virialt, coord1, box1);
      num_diff[ii] = -(ener0 - ener1) / (2. * hh);
    }
    std::vector<VALUETYPE> num_virial(9, 0);
    for (int dd0 = 0; dd0 < 3; ++dd0) {
      for (int dd1 = 0; dd1 < 3; ++dd1) {
        for (int dd = 0; dd < 3; ++dd) {
          num_virial[dd0 * 3 + dd1] +=
              num_diff[dd * 3 + dd0] * box[dd * 3 + dd1];
          // num_virial[dd0*3+dd1] += num_diff[dd0*3+dd] * box[dd1*3+dd];
        }
      }
    }
    for (int ii = 0; ii < 9; ++ii) {
      EXPECT_LT(fabs(num_virial[ii] - virial[ii]), level);
    }
  }
};
