#pragma once

#include <algorithm>
#include <iterator>
#include <cassert>
#include <vector>

#include "utilities.h"
#include "SimulationRegion.h"

// format of the input neighbor list
struct InputNlist
{
  int inum;
  int * ilist;
  int * numneigh;
  int ** firstneigh;
  InputNlist (
      int inum_, 
      int * ilist_,
      int * numneigh_, 
      int ** firstneigh_
      ) 
      : inum(inum_), ilist(ilist_), numneigh(numneigh_), firstneigh(firstneigh_)
      {}
};

void convert_nlist(
    InputNlist & to_nlist,
    std::vector<std::vector<int> > & from_nlist
    );


// template <typename FPTYPE>
// void 
// build_nlist_nopbc(
//     InputNlist & nlist,
//     const FPTYPE * coord,
//     const int & nloc, 
//     const float & rcut);

// template <typename FPTYPE>
// void 
// build_nlist_pbc(
//     InputNlist & nlist,
//     const FPTYPE * coord,
//     const Region<FPTYPE> & region,
//     const int & nloc, 
//     const float & rcut);

// template <typename FPTYPE>
// void 
// build_nlist_pbc(
//     InputNlist & nlist,
//     const FPTYPE * coord,
//     const Region<FPTYPE> & region,
//     const int & nloc, 
//     const float & rcut);


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
