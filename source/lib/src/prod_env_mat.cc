#include <cassert>
#include <iostream>
#include <string.h>
#include "prod_env_mat.h"
#include "fmt_nlist.h"
#include "env_mat.h"

using namespace deepmd;

template<typename FPTYPE>
void
deepmd::
prod_env_mat_a_cpu(
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
    const std::vector<int> sec) 
{
  const int nnei = sec.back();
  const int nem = nnei * 4;

  // set & normalize coord
  std::vector<FPTYPE> d_coord3(nall * 3);
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      d_coord3[ii * 3 + dd] = coord[ii * 3 + dd];
    }
  }

  // set type
  std::vector<int> d_type (nall);
  for (int ii = 0; ii < nall; ++ii) {
    d_type[ii] = type[ii];
  }
    
  // build nlist
  std::vector<std::vector<int > > d_nlist_a(nloc);

  assert(nloc == inlist.inum);
  for (unsigned ii = 0; ii < nloc; ++ii) {
    d_nlist_a[ii].reserve(max_nbor_size);
  }
  for (unsigned ii = 0; ii < nloc; ++ii) {
    int i_idx = inlist.ilist[ii];
    for(unsigned jj = 0; jj < inlist.numneigh[ii]; ++jj){
      int j_idx = inlist.firstneigh[ii][jj];
      d_nlist_a[i_idx].push_back (j_idx);
    }
  }
    
#pragma omp parallel for 
  for (int ii = 0; ii < nloc; ++ii) {
    std::vector<int> fmt_nlist_a;
    int ret = format_nlist_i_cpu(fmt_nlist_a, d_coord3, d_type, ii, d_nlist_a[ii], rcut, sec);
    std::vector<FPTYPE> d_em_a;
    std::vector<FPTYPE> d_em_a_deriv;
    std::vector<FPTYPE> d_em_r;
    std::vector<FPTYPE> d_em_r_deriv;
    std::vector<FPTYPE> d_rij_a;
    env_mat_a_cpu (d_em_a, d_em_a_deriv, d_rij_a, d_coord3, d_type, ii, fmt_nlist_a, sec, rcut_smth, rcut);

    // check sizes
    assert (d_em_a.size() == nem);
    assert (d_em_a_deriv.size() == nem * 3);
    assert (d_rij_a.size() == nnei * 3);
    assert (fmt_nlist_a.size() == nnei);
    // record outputs
    for (int jj = 0; jj < nem; ++jj) {
      em[ii * nem + jj] = (d_em_a[jj] - avg[d_type[ii] * nem + jj]) / std[d_type[ii] * nem + jj];
    }
    for (int jj = 0; jj < nem * 3; ++jj) {
      em_deriv[ii * nem * 3 + jj] = d_em_a_deriv[jj] / std[d_type[ii] * nem + jj / 3];
    }
    for (int jj = 0; jj < nnei * 3; ++jj) {
      rij[ii * nnei * 3 + jj] = d_rij_a[jj];
    }
    for (int jj = 0; jj < nnei; ++jj) {
      nlist[ii * nnei + jj] = fmt_nlist_a[jj];
    }
  }
}

template<typename FPTYPE>
void 
deepmd::
prod_env_mat_r_cpu(
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
    const std::vector<int> sec) 
{
  const int nnei = sec.back();
  const int nem = nnei * 1;

  // set & normalize coord
  std::vector<FPTYPE> d_coord3(nall * 3);
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      d_coord3[ii * 3 + dd] = coord[ii * 3 + dd];
    }
  }

  // set type
  std::vector<int> d_type (nall);
  for (int ii = 0; ii < nall; ++ii) {
    d_type[ii] = type[ii];
  }

  // build nlist
  std::vector<std::vector<int > > d_nlist_a(nloc);

  assert(nloc == inlist.inum);
  for (unsigned ii = 0; ii < nloc; ++ii) {
    d_nlist_a[ii].reserve(max_nbor_size);
  }
  for (unsigned ii = 0; ii < nloc; ++ii) {
    int i_idx = inlist.ilist[ii];
    for(unsigned jj = 0; jj < inlist.numneigh[ii]; ++jj){
      int j_idx = inlist.firstneigh[ii][jj];
      d_nlist_a[i_idx].push_back (j_idx);
    }
  }
    
#pragma omp parallel for 
  for (int ii = 0; ii < nloc; ++ii) {
    std::vector<int> fmt_nlist_a;
    int ret = format_nlist_i_cpu(fmt_nlist_a, d_coord3, d_type, ii, d_nlist_a[ii], rcut, sec);
    std::vector<FPTYPE> d_em_a;
    std::vector<FPTYPE> d_em_a_deriv;
    std::vector<FPTYPE> d_em_r;
    std::vector<FPTYPE> d_em_r_deriv;
    std::vector<FPTYPE> d_rij_a;
    env_mat_r_cpu (d_em_a, d_em_a_deriv, d_rij_a, d_coord3, d_type, ii, fmt_nlist_a, sec, rcut_smth, rcut);

    // check sizes
    assert (d_em_a.size() == nem);
    assert (d_em_a_deriv.size() == nem * 3);
    assert (d_rij_a.size() == nnei * 3);
    assert (fmt_nlist_a.size() == nnei);
    // record outputs
    for (int jj = 0; jj < nem; ++jj) {
      em[ii * nem + jj] = (d_em_a[jj] - avg[d_type[ii] * nem + jj]) / std[d_type[ii] * nem + jj];
    }
    for (int jj = 0; jj < nem * 3; ++jj) {
      em_deriv[ii * nem * 3 + jj] = d_em_a_deriv[jj] / std[d_type[ii] * nem + jj / 3];
    }
    for (int jj = 0; jj < nnei * 3; ++jj) {
      rij[ii * nnei * 3 + jj] = d_rij_a[jj];
    }
    for (int jj = 0; jj < nnei; ++jj) {
      nlist[ii * nnei + jj] = fmt_nlist_a[jj];
    }
  }
}


template
void 
deepmd::
prod_env_mat_a_cpu<double>(
    double * em, 
    double * em_deriv, 
    double * rij, 
    int * nlist, 
    const double * coord, 
    const int * type, 
    const InputNlist & inlist,
    const int max_nbor_size,
    const double * avg, 
    const double * std, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

template
void
deepmd::
prod_env_mat_a_cpu<float>(
    float * em, 
    float * em_deriv, 
    float * rij, 
    int * nlist, 
    const float * coord, 
    const int * type, 
    const InputNlist & inlist,
    const int max_nbor_size,
    const float * avg, 
    const float * std, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

template
void
deepmd::
prod_env_mat_r_cpu<double>(
    double * em, 
    double * em_deriv, 
    double * rij, 
    int * nlist, 
    const double * coord, 
    const int * type, 
    const InputNlist & inlist,
    const int max_nbor_size,
    const double * avg, 
    const double * std, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

template
void 
deepmd::
prod_env_mat_r_cpu<float>(
    float * em, 
    float * em_deriv, 
    float * rij, 
    int * nlist, 
    const float * coord, 
    const int * type, 
    const InputNlist & inlist,
    const int max_nbor_size,
    const float * avg, 
    const float * std, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

#if GOOGLE_CUDA
void deepmd::env_mat_nbor_update(
    InputNlist &inlist,
    InputNlist &gpu_inlist,
    int &max_nbor_size,
    int* &nbor_list_dev,
    const int * mesh, 
    const int size)
{
  int *mesh_host = new int[size];
  cudaErrcheck(cudaMemcpy(mesh_host, mesh, sizeof(int) * size, cudaMemcpyDeviceToHost));
  memcpy(&inlist.ilist, 4 + mesh_host, sizeof(int *));
	memcpy(&inlist.numneigh, 8 + mesh_host, sizeof(int *));
	memcpy(&inlist.firstneigh, 12 + mesh_host, sizeof(int **));
  const int ago = mesh_host[0];
  if (ago == 0) {
    const int inum = inlist.inum;
    if (gpu_inlist.inum < inum) {
      delete_device_memory(gpu_inlist.ilist);
      delete_device_memory(gpu_inlist.numneigh);
      delete_device_memory(gpu_inlist.firstneigh);
      malloc_device_memory(gpu_inlist.ilist, inum);
      malloc_device_memory(gpu_inlist.numneigh, inum);
      malloc_device_memory(gpu_inlist.firstneigh, inum);
    }
    memcpy_host_to_device(gpu_inlist.ilist, inlist.ilist, inum);
    memcpy_host_to_device(gpu_inlist.numneigh, inlist.numneigh, inum);
    int _max_nbor_size = max_numneigh(inlist);
    if (_max_nbor_size <= 1024) {
      _max_nbor_size = 1024;
    }
    else if (_max_nbor_size <= 2048) {
      _max_nbor_size = 2048;
    }
    else {
      _max_nbor_size = 4096;
    }
    if ( nbor_list_dev == NULL 
      || _max_nbor_size > max_nbor_size 
      || inum > gpu_inlist.inum) 
    {
      delete_device_memory(nbor_list_dev);
      malloc_device_memory(nbor_list_dev, inum * _max_nbor_size);
    }
    // update info
    gpu_inlist.inum = inum;
    max_nbor_size = _max_nbor_size;

    // copy nbor list from host to the device
    std::vector<int> nbor_list_host(inum * max_nbor_size, 0);
    int ** _firstneigh = (int**)malloc(sizeof(int*) * inum);
    for (int ii = 0; ii < inum; ii++) {
      _firstneigh[ii] = nbor_list_dev + ii * max_nbor_size;
      for (int jj = 0; jj < inlist.numneigh[ii]; jj++) {
        nbor_list_host[ii * max_nbor_size + jj] = inlist.firstneigh[ii][jj];
      }
    }
    memcpy_host_to_device(nbor_list_dev, &nbor_list_host[0], inum * max_nbor_size);
    memcpy_host_to_device(gpu_inlist.firstneigh, _firstneigh, inum);
    free(_firstneigh);
  }
  delete [] mesh_host;
}
#endif // GOOGLE_CUDA
