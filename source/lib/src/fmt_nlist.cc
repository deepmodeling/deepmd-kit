#include <vector>
#include <cassert>
#include <algorithm>
#include "fmt_nlist.h"
#include "SimulationRegion.h"
#include <iostream>

using namespace deepmd;

struct NeighborInfo 
{
  int type;
  double dist;
  int index;
  NeighborInfo () 
      : type (0), dist(0), index(0) 
      {
      }
  NeighborInfo (int tt, double dd, int ii) 
      : type (tt), dist(dd), index(ii) 
      {
      }
  bool operator < (const NeighborInfo & b) const 
      {
	return (type < b.type || 
		(type == b.type && 
		 (dist < b.dist || 
		  (dist == b.dist && index < b.index) ) ) );
      }
};

int format_nlist_i_fill_a (
    std::vector<int > &			fmt_nei_idx_a,
    std::vector<int > &			fmt_nei_idx_r,
    const std::vector<double > &	posi,
    const int &				ntypes,
    const std::vector<int > &		type,
    const SimulationRegion<double> &	region,
    const bool &			b_pbc,
    const int &				i_idx,
    const std::vector<int > &		nei_idx_a, 
    const std::vector<int > &		nei_idx_r, 
    const double &			rcut,
    const std::vector<int > &		sec_a, 
    const std::vector<int > &		sec_r)
{
#ifdef DEBUG
  assert (sec_a.size() == ntypes + 1);
  assert (sec_r.size() == ntypes + 1);
#endif
  
  fmt_nei_idx_a.resize (sec_a.back());
  fmt_nei_idx_r.resize (sec_r.back());
  fill (fmt_nei_idx_a.begin(), fmt_nei_idx_a.end(), -1);
  fill (fmt_nei_idx_r.begin(), fmt_nei_idx_r.end(), -1);  
  
  // gether all neighbors
  std::vector<int > nei_idx (nei_idx_a);
  nei_idx.insert (nei_idx.end(), nei_idx_r.begin(), nei_idx_r.end());
  assert (nei_idx.size() == nei_idx_a.size() + nei_idx_r.size());
  // allocate the information for all neighbors
  std::vector<NeighborInfo > sel_nei ;
  sel_nei.reserve (nei_idx_a.size() + nei_idx_r.size());
  for (unsigned kk = 0; kk < nei_idx.size(); ++kk){
    double diff[3];
    const int & j_idx = nei_idx[kk];
    if (b_pbc){
      region.diffNearestNeighbor (posi[j_idx*3+0], posi[j_idx*3+1], posi[j_idx*3+2], 
				  posi[i_idx*3+0], posi[i_idx*3+1], posi[i_idx*3+2], 
				  diff[0], diff[1], diff[2]);
    }
    else {
      for (int dd = 0; dd < 3; ++dd) diff[dd] = posi[j_idx*3+dd] - posi[i_idx*3+dd];
    }
    double rr = sqrt(deepmd::dot3(diff, diff));    
    if (rr <= rcut) {
      sel_nei.push_back(NeighborInfo (type[j_idx], rr, j_idx));
    }
  }
  sort (sel_nei.begin(), sel_nei.end());  
  
  std::vector<int > nei_iter = sec_a;
  int overflowed = -1;
  for (unsigned kk = 0; kk < sel_nei.size(); ++kk){
    const int & nei_type = sel_nei[kk].type;
    if (nei_iter[nei_type] >= sec_a[nei_type+1]) {
      int r_idx_iter = (nei_iter[nei_type] ++) - sec_a[nei_type+1] + sec_r[nei_type];
      if (r_idx_iter >= sec_r[nei_type+1]) {
	// return nei_type;
	overflowed = nei_type;
      }
      else {
	fmt_nei_idx_r[r_idx_iter] = sel_nei[kk].index;
      }
    }
    else {
      fmt_nei_idx_a[nei_iter[nei_type] ++] = sel_nei[kk].index;
    }
  }
  return overflowed;
}


template<typename FPTYPE> 
int format_nlist_i_cpu (
    std::vector<int > &		fmt_nei_idx_a,
    const std::vector<FPTYPE > &posi,
    const std::vector<int > &   type,
    const int &			i_idx,
    const std::vector<int > &   nei_idx_a, 
    const float &		rcut,
    const std::vector<int > &   sec_a)
{
    fmt_nei_idx_a.resize (sec_a.back());
    fill (fmt_nei_idx_a.begin(), fmt_nei_idx_a.end(), -1);
  
    // gether all neighbors
    std::vector<int > nei_idx (nei_idx_a);
    // allocate the information for all neighbors
    std::vector<NeighborInfo > sel_nei;
    sel_nei.reserve (nei_idx_a.size());
    for (unsigned kk = 0; kk < nei_idx.size(); ++kk) {
        FPTYPE diff[3];
        const int & j_idx = nei_idx[kk];
        for (int dd = 0; dd < 3; ++dd) {
            diff[dd] = posi[j_idx * 3 + dd] - posi[i_idx * 3 + dd];
        }
        FPTYPE rr = sqrt(deepmd::dot3(diff, diff));    
        if (rr <= rcut) {
            sel_nei.push_back(NeighborInfo(type[j_idx], rr, j_idx));
        }
    }
    sort(sel_nei.begin(), sel_nei.end());  
  
    std::vector<int > nei_iter = sec_a;
    int overflowed = -1;
    for (unsigned kk = 0; kk < sel_nei.size(); ++kk) {
        const int & nei_type = sel_nei[kk].type;
        if (nei_iter[nei_type] < sec_a[nei_type+1]) {
            fmt_nei_idx_a[nei_iter[nei_type] ++] = sel_nei[kk].index;
        }
	else{
	  overflowed = nei_type;
	}
    }
    return overflowed;
}

template<typename FPTYPE> 
void 
deepmd::
format_nlist_cpu (
    int * nlist,
    const InputNlist & in_nlist,
    const FPTYPE * coord, 
    const int * type, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const std::vector<int> sec)
{
  std::vector<FPTYPE> posi_(nall * 3);
  std::vector<int> type_(nall);
  std::copy(coord, coord + nall * 3, posi_.begin());
  std::copy(type, type + nall, type_.begin());
  std::vector<int> ilist, fmt_ilist;
  int nnei = sec.back();
  
  for(int ii = 0; ii < in_nlist.inum; ++ii){
    int i_idx = in_nlist.ilist[ii];
    int i_num = in_nlist.numneigh[ii];
    ilist.resize(i_num);
    std::copy(in_nlist.firstneigh[ii], in_nlist.firstneigh[ii] + i_num, ilist.begin());
    format_nlist_i_cpu(
	fmt_ilist,
	posi_,
	type_,
	i_idx,
	ilist,
	rcut, 
	sec);	
    int * cur_nlist = nlist + i_idx * nnei;
    if(fmt_ilist.size() != nnei){
      std::cerr << "FATAL: formatted nlist of i have length " 
		<< fmt_ilist.size()
		<< " which does not match " 
		<< nnei	<< std::endl;
      exit(1);
    }
    std::copy(fmt_ilist.begin(), fmt_ilist.end(), cur_nlist);
  }
}

template
int format_nlist_i_cpu<double> (
    std::vector<int > &		fmt_nei_idx_a,
    const std::vector<double > &posi,
    const std::vector<int > &   type,
    const int &			i_idx,
    const std::vector<int > &   nei_idx_a, 
    const float &		rcut,
    const std::vector<int > &   sec_a);


template
int format_nlist_i_cpu<float> (
    std::vector<int > &		fmt_nei_idx_a,
    const std::vector<float > &	posi,
    const std::vector<int > &   type,
    const int &			i_idx,
    const std::vector<int > &   nei_idx_a, 
    const float &		rcut,
    const std::vector<int > &   sec_a);

template
void 
deepmd::
format_nlist_cpu<double> (
    int * nlist,
    const deepmd::InputNlist & in_nlist,
    const double * coord, 
    const int * type, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const std::vector<int> sec);


template
void 
deepmd::
format_nlist_cpu<float> (
    int * nlist,
    const deepmd::InputNlist & in_nlist,
    const float * coord, 
    const int * type, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const std::vector<int> sec);


