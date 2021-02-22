#pragma once
#include <vector>
#include "device_common.h"

template<typename FPTYPE>
void prod_env_mat_a_cpu(
    FPTYPE * em, 
    FPTYPE * em_deriv, 
    FPTYPE * rij, 
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const int * ilist, 
    const int * jrange, 
    const int * jlist,
    const int max_nbor_size,
    const FPTYPE * avg, 
    const FPTYPE * std, 
    const int nloc, 
    const int nall, 
    const int ntypes, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

#if GOOGLE_CUDA
template<typename FPTYPE> 
void prod_env_mat_a_gpu_nv(    
    FPTYPE * em, 
    FPTYPE * em_deriv, 
    FPTYPE * rij, 
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const int * ilist, 
    const int * jrange, 
    const int * jlist,
    int * array_int, 
    unsigned long long * array_longlong,
    const int max_nbor_size,
    const FPTYPE * avg, 
    const FPTYPE * std, 
    const int nloc, 
    const int nall, 
    const int ntypes, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

void env_mat_nbor_update(
    bool &init,
    int * &ilist,
    int * &jrange,
    int * &jlist,
    int &ilist_size,
    int &jrange_size,
    int &jlist_size,
    int &max_nbor_size,
    const int * mesh, 
    const int size);
#endif

