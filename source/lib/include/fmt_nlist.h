#pragma once

#include <vector>
#include "neighbor_list.h"

namespace deepmd{

template <typename FPTYPE>
void format_nlist_cpu(
    int * nlist,
    const InputNlist & in_nlist,
    const FPTYPE * coord, 
    const int * type, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const std::vector<int> sec);

}


////////////////////////////////////////////////////////
// legacy code
////////////////////////////////////////////////////////

#include "SimulationRegion.h"

// return:	-1	OK
//		> 0	the type of unsuccessful neighbor list
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
    const std::vector<int > &		sec_r);


template<typename FPTYPE> 
int format_nlist_i_cpu (
    std::vector<int > &			fmt_nei_idx_a,
    const std::vector<FPTYPE > &	posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		nei_idx_a, 
    const float &			rcut,
    const std::vector<int > &		sec_a);



