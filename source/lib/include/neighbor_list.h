#pragma once

#include <algorithm>
#include <iterator>
#include <cassert>
#include <vector>

#include "region.h"
#include "utilities.h"
#include "SimulationRegion.h"

namespace deepmd{

// construct InputNlist with the input LAMMPS nbor list info.
//
// members
//  inum, numneigh, firstneigh are provided by LAMMPS.
//  inum is the number of core region atoms.
//  ilist is the atoms index array within the core region.
//  numneigh is the array which indicates each core region atom's neighbor atom number.
//  firstneigh is two-dimensional array which store each core region atom's neighbor index. 
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

// construct the InputNlist with a two-dimensional vector.
// inputs
//  from_nlist
//  from_nlist is a two-dimensional vector which stores the neighbor information of the core region atoms.
// outputs
//  to_nlist
//  to_nlist is the InputNlist struct which stores the neighbor information of the core region atoms.
void convert_nlist(
    InputNlist & to_nlist,
    std::vector<std::vector<int> > & from_nlist
    );

// compute the max number of neighbors within the core region atoms
// outputs
//  to_nlist is the InputNlist struct which stores the neighbor information of the core region atoms.
// returns
//  integer: max number of neighbors
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// convert the a host memory InputNlist to a device memory InputNlist.
// inputs
//  cpu_nlist, gpu_memory, max_nbor_size
//  cpu_nlist is the host memory InputNlist struct which stores the neighbor information of the core region atoms.
//  gpu_memory is the device array which stores the elements of gpu_nlist.
//  max_nbor_size is the max neighbor size.
// outputs
//  gpu_nlist
//  gpu_nlist is the device memory InputNlist struct which stores the neighbor information of the core region atoms.
void convert_nlist_gpu_device(
    InputNlist & gpu_nlist,
    InputNlist & cpu_nlist,
    int* & gpu_memory,
    const int & max_nbor_size);

// Reclaim the allocated device memory of struct InputNlist
// inputs
//  gpu_nlist is the device memory InputNlist struct which stores the neighbor information of the core region atoms.
void free_nlist_gpu_device(
    InputNlist & gpu_nlist);

void use_nlist_map(
    int * nlist, 
    const int * nlist_map, 
    const int nloc, 
    const int nnei);

#endif //GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
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

#endif // GOOGLE_CUDA


#if TENSORFLOW_USE_ROCM
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
build_nlist_gpu_rocm(
    InputNlist & nlist,
    int * max_list_size,
    int * nlist_data,
    const FPTYPE * c_cpy, 
    const int & nloc, 
    const int & nall, 
    const int & mem_size,
    const float & rcut);
	
#endif // TENSORFLOW_USE_ROCM

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
