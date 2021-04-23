#pragma once

#include <algorithm>
#include <iterator>
#include <cassert>
#include <vector>

#include "region.h"
#include "utilities.h"
#include "SimulationRegion.h"

namespace deepmd{

// format of the input neighbor list
struct InputNlist
{
  int inum;
  int * ilist;
  int * numneigh;
  int ** firstneigh;
  InputNlist () 
      : inum(0), ilist(NULL), numneigh(NULL), firstneigh(NULL)
      {};
  InputNlist (
      int inum_, 
      int * ilist_,
      int * numneigh_, 
      int ** firstneigh_
      ) 
      : inum(inum_), ilist(ilist_), numneigh(numneigh_), firstneigh(firstneigh_)
      {};
  ~InputNlist(){};
};

void convert_nlist(
    InputNlist & to_nlist,
    std::vector<std::vector<int> > & from_nlist
    );

int max_numneigh(
    const InputNlist & to_nlist
    );

// build neighbor list.
// outputs
//	nlist, max_list_size
//	max_list_size is the maximal size of jlist.
// inputs
//	c_cpy, nloc, nall, mem_size, rcut, region
//	mem_size is the size of allocated memory for jlist.
// returns
//	0: succssful
//	1: the memory is not large enough to hold all neighbors.
//	   i.e. max_list_size > mem_nall
template <typename FPTYPE>
int
build_nlist_cpu(
    InputNlist & nlist,
    int * max_list_size,
    const FPTYPE * c_cpy,
    const int & nloc, 
    const int & nall, 
    const int & mem_size,
    const float & rcut);

#if GOOGLE_CUDA
void convert_nlist_gpu_cuda(
    InputNlist & gpu_nlist,
    InputNlist & cpu_nlist,
    int* & gpu_memory,
    const int & max_nbor_size);

void free_nlist_gpu_cuda(
    InputNlist & gpu_nlist);

// build neighbor list.
// outputs
//	nlist, max_list_size
//	max_list_size is the maximal size of jlist.
// inputs
//	c_cpy, nloc, nall, mem_size, rcut, region
//	mem_size is the size of allocated memory for jlist.
// returns
//	0: succssful
//	1: the memory is not large enough to hold all neighbors.
//	   i.e. max_list_size > mem_nall
template <typename FPTYPE>
int
build_nlist_gpu(
    InputNlist & nlist,
    int * max_list_size,
    int * nlist_data,
    const FPTYPE * c_cpy, 
    const int & nloc, 
    const int & nall, 
    const int & mem_size,
    const float & rcut);

void use_nlist_map(
    int * nlist, 
    const int * nlist_map, 
    const int nloc, 
    const int nnei);
	
#endif // GOOGLE_CUDA

} // namespace deepmd


////////////////////////////////////////////////////////
// legacy code
////////////////////////////////////////////////////////

// build nlist by an extended grid
void
build_nlist (std::vector<std::vector<int > > &	nlist0,
	     std::vector<std::vector<int > > &	nlist1,
	     const std::vector<double > &	coord,
	     const int &			nloc,
	     const double &			rc0,
	     const double &			rc1,
	     const std::vector<int > &		nat_stt_,
	     const std::vector<int > &		nat_end_,
	     const std::vector<int > &		ext_stt_,
	     const std::vector<int > &		ext_end_,
	     const SimulationRegion<double> &	region,
	     const std::vector<int > &		global_grid);

// build nlist by a grid for a periodic region
void
build_nlist (std::vector<std::vector<int > > &	nlist0,
	     std::vector<std::vector<int > > &	nlist1,
	     const std::vector<double > &	coord,
	     const double &			rc0,
	     const double &			rc1,
	     const std::vector<int > &		grid,
	     const SimulationRegion<double> &	region);

// build nlist by a grid for a periodic region, atoms selected by sel0 and sel1
void
build_nlist (std::vector<std::vector<int > > &	nlist0,
	     std::vector<std::vector<int > > &	nlist1,
	     const std::vector<double > &	coord,
	     const std::vector<int> &		sel0,
	     const std::vector<int> &		sel1,
	     const double &			rc0,
	     const double &			rc1,
	     const std::vector<int > &		grid,
	     const SimulationRegion<double> &	region);

// brute force (all-to-all distance computation) neighbor list building
// if region is NULL, open boundary is assumed,
// otherwise, periodic boundary condition is defined by region
void
build_nlist (std::vector<std::vector<int > > & nlist0,
	     std::vector<std::vector<int > > & nlist1,
	     const std::vector<double > &	coord,
	     const double &			rc0_,
	     const double &			rc1_,
	     const SimulationRegion<double > * region = NULL);

// copy periodic images for the system
void 
copy_coord (std::vector<double > &		out_c, 
	    std::vector<int > &			out_t, 
	    std::vector<int > &			mapping,
	    std::vector<int> &			ncell,
	    std::vector<int> &			ngcell,
	    const std::vector<double > &	in_c,
	    const std::vector<int > &		in_t,
	    const double &			rc,
	    const SimulationRegion<double > &	region);
