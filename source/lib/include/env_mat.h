#pragma once

#include <vector>

namespace deepmd{

template<typename FPTYPE> 
void env_mat_a_cpu (
    std::vector<FPTYPE > &	        descrpt_a,
    std::vector<FPTYPE > &	        descrpt_a_deriv,
    std::vector<FPTYPE > &	        rij_a,
    const std::vector<FPTYPE > &	posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec, 
    const float &			rmin,
    const float &			rmax) ;

template<typename FPTYPE> 
void env_mat_r_cpu (
    std::vector<FPTYPE > &	        descrpt_a,
    std::vector<FPTYPE > &	        descrpt_a_deriv,
    std::vector<FPTYPE > &	        rij_a,
    const std::vector<FPTYPE > &	posi,
    const std::vector<int > &		type,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist_a,
    const std::vector<int > &		sec_a, 
    const float &			rmin,
    const float &			rmax);

}

////////////////////////////////////////////////////////
// legacy code
////////////////////////////////////////////////////////

#include "SimulationRegion.h"

void env_mat_a (
    std::vector<double > &		descrpt_a,
    std::vector<double > &		descrpt_a_deriv,
    std::vector<double > &		rij_a,
    const std::vector<double > &	posi,
    const int &				ntypes,
    const std::vector<int > &		type,
    const SimulationRegion<double> &	region,
    const bool &			b_pbc,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec, 
    const double &			rmin,
    const double &			rmax);

void env_mat_r (
    std::vector<double > &		descrpt_r,
    std::vector<double > &		descrpt_r_deriv,
    std::vector<double > &		rij_r,
    const std::vector<double > &	posi,
    const int &				ntypes,
    const std::vector<int > &		type,
    const SimulationRegion<double> &	region,
    const bool &			b_pbc,
    const int &				i_idx,
    const std::vector<int > &		fmt_nlist,
    const std::vector<int > &		sec,
    const double &			rmin, 
    const double &			rmax);

