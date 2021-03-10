#pragma once

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
