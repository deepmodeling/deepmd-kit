#pragma once
#include <vector>
#include "device.h"
#include "neighbor_list.h"

namespace deepmd{

template<typename FPTYPE>
void prod_env_mat_a_cpu(
    FPTYPE * em, 
    FPTYPE * em_deriv, 
    FPTYPE * rij, 
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const InputNlist & inlist,
    const int max_nbor_size,
    const FPTYPE * avg, 
    const FPTYPE * std, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

template<typename FPTYPE>
void prod_env_mat_r_cpu(
    FPTYPE * em, 
    FPTYPE * em_deriv, 
    FPTYPE * rij, 
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const InputNlist & inlist,
    const int max_nbor_size,
    const FPTYPE * avg, 
    const FPTYPE * std, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

#if GOOGLE_CUDA
template<typename FPTYPE> 
void prod_env_mat_a_gpu_cuda(    
    FPTYPE * em, 
    FPTYPE * em_deriv, 
    FPTYPE * rij, 
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const InputNlist & gpu_inlist,
    int * array_int, 
    unsigned long long * array_longlong,
    const int max_nbor_size,
    const FPTYPE * avg, 
    const FPTYPE * std, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

template<typename FPTYPE> 
void prod_env_mat_r_gpu_cuda(    
    FPTYPE * em, 
    FPTYPE * em_deriv, 
    FPTYPE * rij, 
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const InputNlist & gpu_inlist,
    int * array_int, 
    unsigned long long * array_longlong,
    const int max_nbor_size,
    const FPTYPE * avg, 
    const FPTYPE * std, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

void env_mat_nbor_update(
    InputNlist &inlist,
    InputNlist &gpu_inlist,
    int &max_nbor_size,
    int* &nbor_list_dev,
    const int * mesh, 
    const int size);
#endif // GOOGLE_CUDA

}

