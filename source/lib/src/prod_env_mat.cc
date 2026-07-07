// SPDX-License-Identifier: LGPL-3.0-or-later
#include "prod_env_mat.h"

#include <string.h>

#include <cassert>
#include <iostream>

#include "env_mat.h"
#include "fmt_nlist.h"

using namespace deepmd;

template <typename FPTYPE>
void deepmd::prod_env_mat_a_cpu(FPTYPE* em,
                                FPTYPE* em_deriv,
                                FPTYPE* rij,
                                int* nlist,
                                const FPTYPE* coord,
                                const int* type,
                                const InputNlist& inlist,
                                const int max_nbor_size,
                                const FPTYPE* avg,
                                const FPTYPE* std,
                                const int nloc,
                                const int nall,
                                const int nframes,
                                const float rcut,
                                const float rcut_smth,
                                const std::vector<int> sec,
                                const int* f_type) {
  if (f_type == NULL) {
    f_type = type;
  }
  const int nnei = sec.back();
  const int nem = nnei * 4;
  assert(nframes * nloc == inlist.inum);

  std::vector<std::vector<FPTYPE> > frame_coords(nframes);
  std::vector<std::vector<int> > frame_f_types(nframes);
  std::vector<std::vector<std::vector<int> > > frame_nlists(nframes);

#pragma omp parallel
  {
#pragma omp for
    for (int ff = 0; ff < nframes; ++ff) {
      const FPTYPE* frame_coord = coord + static_cast<size_t>(ff) * nall * 3;
      const int* frame_f_type = f_type + static_cast<size_t>(ff) * nall;
      const int_64 row_offset = static_cast<int_64>(ff) * nloc;

      frame_coords[ff].resize(static_cast<size_t>(nall) * 3);
      for (int ii = 0; ii < nall; ++ii) {
        for (int dd = 0; dd < 3; ++dd) {
          frame_coords[ff][ii * 3 + dd] = frame_coord[ii * 3 + dd];
        }
      }

      frame_f_types[ff].resize(nall);
      for (int ii = 0; ii < nall; ++ii) {
        frame_f_types[ff][ii] = frame_f_type[ii];
      }

      frame_nlists[ff].resize(nloc);
      for (int ii = 0; ii < nloc; ++ii) {
        frame_nlists[ff][ii].reserve(max_nbor_size);
      }
      for (int ii = 0; ii < nloc; ++ii) {
        const int_64 row = row_offset + ii;
        const int i_idx = inlist.ilist[row];
        for (int jj = 0; jj < inlist.numneigh[row]; ++jj) {
          const int j_idx = inlist.firstneigh[row][jj];
          frame_nlists[ff][i_idx].push_back(j_idx);
        }
      }
    }

#pragma omp for
    for (int_64 row = 0; row < static_cast<int_64>(nframes) * nloc; ++row) {
      const int ff = row / nloc;
      const int ii = row % nloc;
      const int_64 row_offset = static_cast<int_64>(ff) * nloc;
      const int* frame_type = type + static_cast<size_t>(ff) * nall;
      FPTYPE* frame_em = em + static_cast<size_t>(row_offset) * nem;
      FPTYPE* frame_em_deriv =
          em_deriv + static_cast<size_t>(row_offset) * nem * 3;
      FPTYPE* frame_rij = rij + static_cast<size_t>(row_offset) * nnei * 3;
      int* frame_nlist = nlist + static_cast<size_t>(row_offset) * nnei;
      const std::vector<FPTYPE>& d_coord3 = frame_coords[ff];
      const std::vector<int>& d_f_type = frame_f_types[ff];
      std::vector<int> fmt_nlist_a;
      format_nlist_i_cpu(fmt_nlist_a, d_coord3, d_f_type, ii,
                         frame_nlists[ff][ii], rcut, sec);
      std::vector<FPTYPE> d_em_a;
      std::vector<FPTYPE> d_em_a_deriv;
      std::vector<FPTYPE> d_em_r;
      std::vector<FPTYPE> d_em_r_deriv;
      std::vector<FPTYPE> d_rij_a;
      env_mat_a_cpu(d_em_a, d_em_a_deriv, d_rij_a, d_coord3, d_f_type, ii,
                    fmt_nlist_a, sec, rcut_smth, rcut);

      // check sizes
      assert(d_em_a.size() == nem);
      assert(d_em_a_deriv.size() == nem * 3);
      assert(d_rij_a.size() == nnei * 3);
      assert(fmt_nlist_a.size() == nnei);
      // record outputs
      for (int jj = 0; jj < nem; ++jj) {
        if (frame_type[ii] >= 0) {
          frame_em[ii * nem + jj] =
              (d_em_a[jj] - avg[frame_type[ii] * nem + jj]) /
              std[frame_type[ii] * nem + jj];
        } else {
          frame_em[ii * nem + jj] = 0;
        }
      }
      for (int jj = 0; jj < nem * 3; ++jj) {
        if (frame_type[ii] >= 0) {
          frame_em_deriv[ii * nem * 3 + jj] =
              d_em_a_deriv[jj] / std[frame_type[ii] * nem + jj / 3];
        } else {
          frame_em_deriv[ii * nem * 3 + jj] = 0;
        }
      }
      for (int jj = 0; jj < nnei * 3; ++jj) {
        frame_rij[ii * nnei * 3 + jj] = d_rij_a[jj];
      }
      for (int jj = 0; jj < nnei; ++jj) {
        frame_nlist[ii * nnei + jj] = fmt_nlist_a[jj];
      }
    }
  }
}

template <typename FPTYPE>
void deepmd::prod_env_mat_r_cpu(FPTYPE* em,
                                FPTYPE* em_deriv,
                                FPTYPE* rij,
                                int* nlist,
                                const FPTYPE* coord,
                                const int* type,
                                const InputNlist& inlist,
                                const int max_nbor_size,
                                const FPTYPE* avg,
                                const FPTYPE* std,
                                const int nloc,
                                const int nall,
                                const int nframes,
                                const float rcut,
                                const float rcut_smth,
                                const std::vector<int> sec) {
  const int nnei = sec.back();
  const int nem = nnei * 1;
  assert(nframes * nloc == inlist.inum);

  std::vector<std::vector<FPTYPE> > frame_coords(nframes);
  std::vector<std::vector<int> > frame_types(nframes);
  std::vector<std::vector<std::vector<int> > > frame_nlists(nframes);

#pragma omp parallel
  {
#pragma omp for
    for (int ff = 0; ff < nframes; ++ff) {
      const FPTYPE* frame_coord = coord + static_cast<size_t>(ff) * nall * 3;
      const int* frame_type = type + static_cast<size_t>(ff) * nall;
      const int_64 row_offset = static_cast<int_64>(ff) * nloc;

      frame_coords[ff].resize(static_cast<size_t>(nall) * 3);
      for (int ii = 0; ii < nall; ++ii) {
        for (int dd = 0; dd < 3; ++dd) {
          frame_coords[ff][ii * 3 + dd] = frame_coord[ii * 3 + dd];
        }
      }

      frame_types[ff].resize(nall);
      for (int ii = 0; ii < nall; ++ii) {
        frame_types[ff][ii] = frame_type[ii];
      }

      frame_nlists[ff].resize(nloc);
      for (int ii = 0; ii < nloc; ++ii) {
        frame_nlists[ff][ii].reserve(max_nbor_size);
      }
      for (int ii = 0; ii < nloc; ++ii) {
        const int_64 row = row_offset + ii;
        const int i_idx = inlist.ilist[row];
        for (int jj = 0; jj < inlist.numneigh[row]; ++jj) {
          const int j_idx = inlist.firstneigh[row][jj];
          frame_nlists[ff][i_idx].push_back(j_idx);
        }
      }
    }

#pragma omp for
    for (int_64 row = 0; row < static_cast<int_64>(nframes) * nloc; ++row) {
      const int ff = row / nloc;
      const int ii = row % nloc;
      const int_64 row_offset = static_cast<int_64>(ff) * nloc;
      FPTYPE* frame_em = em + static_cast<size_t>(row_offset) * nem;
      FPTYPE* frame_em_deriv =
          em_deriv + static_cast<size_t>(row_offset) * nem * 3;
      FPTYPE* frame_rij = rij + static_cast<size_t>(row_offset) * nnei * 3;
      int* frame_nlist = nlist + static_cast<size_t>(row_offset) * nnei;
      const std::vector<FPTYPE>& d_coord3 = frame_coords[ff];
      const std::vector<int>& d_type = frame_types[ff];
      std::vector<int> fmt_nlist_a;
      format_nlist_i_cpu(fmt_nlist_a, d_coord3, d_type, ii,
                         frame_nlists[ff][ii], rcut, sec);
      std::vector<FPTYPE> d_em_a;
      std::vector<FPTYPE> d_em_a_deriv;
      std::vector<FPTYPE> d_em_r;
      std::vector<FPTYPE> d_em_r_deriv;
      std::vector<FPTYPE> d_rij_a;
      env_mat_r_cpu(d_em_a, d_em_a_deriv, d_rij_a, d_coord3, d_type, ii,
                    fmt_nlist_a, sec, rcut_smth, rcut);

      // check sizes
      assert(d_em_a.size() == nem);
      assert(d_em_a_deriv.size() == nem * 3);
      assert(d_rij_a.size() == nnei * 3);
      assert(fmt_nlist_a.size() == nnei);
      // record outputs
      for (int jj = 0; jj < nem; ++jj) {
        if (d_type[ii] >= 0) {
          frame_em[ii * nem + jj] = (d_em_a[jj] - avg[d_type[ii] * nem + jj]) /
                                    std[d_type[ii] * nem + jj];
        } else {
          frame_em[ii * nem + jj] = 0;
        }
      }
      for (int jj = 0; jj < nem * 3; ++jj) {
        if (d_type[ii] >= 0) {
          frame_em_deriv[ii * nem * 3 + jj] =
              d_em_a_deriv[jj] / std[d_type[ii] * nem + jj / 3];
        } else {
          frame_em_deriv[ii * nem * 3 + jj] = 0;
        }
      }
      for (int jj = 0; jj < nnei * 3; ++jj) {
        frame_rij[ii * nnei * 3 + jj] = d_rij_a[jj];
      }
      for (int jj = 0; jj < nnei; ++jj) {
        frame_nlist[ii * nnei + jj] = fmt_nlist_a[jj];
      }
    }
  }
}

template void deepmd::prod_env_mat_a_cpu<double>(double* em,
                                                 double* em_deriv,
                                                 double* rij,
                                                 int* nlist,
                                                 const double* coord,
                                                 const int* type,
                                                 const InputNlist& inlist,
                                                 const int max_nbor_size,
                                                 const double* avg,
                                                 const double* std,
                                                 const int nloc,
                                                 const int nall,
                                                 const int nframes,
                                                 const float rcut,
                                                 const float rcut_smth,
                                                 const std::vector<int> sec,
                                                 const int* f_type);

template void deepmd::prod_env_mat_a_cpu<float>(float* em,
                                                float* em_deriv,
                                                float* rij,
                                                int* nlist,
                                                const float* coord,
                                                const int* type,
                                                const InputNlist& inlist,
                                                const int max_nbor_size,
                                                const float* avg,
                                                const float* std,
                                                const int nloc,
                                                const int nall,
                                                const int nframes,
                                                const float rcut,
                                                const float rcut_smth,
                                                const std::vector<int> sec,
                                                const int* f_type);

template void deepmd::prod_env_mat_r_cpu<double>(double* em,
                                                 double* em_deriv,
                                                 double* rij,
                                                 int* nlist,
                                                 const double* coord,
                                                 const int* type,
                                                 const InputNlist& inlist,
                                                 const int max_nbor_size,
                                                 const double* avg,
                                                 const double* std,
                                                 const int nloc,
                                                 const int nall,
                                                 const int nframes,
                                                 const float rcut,
                                                 const float rcut_smth,
                                                 const std::vector<int> sec);

template void deepmd::prod_env_mat_r_cpu<float>(float* em,
                                                float* em_deriv,
                                                float* rij,
                                                int* nlist,
                                                const float* coord,
                                                const int* type,
                                                const InputNlist& inlist,
                                                const int max_nbor_size,
                                                const float* avg,
                                                const float* std,
                                                const int nloc,
                                                const int nall,
                                                const int nframes,
                                                const float rcut,
                                                const float rcut_smth,
                                                const std::vector<int> sec);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
void deepmd::env_mat_nbor_update(InputNlist& inlist,
                                 InputNlist& gpu_inlist,
                                 int& max_nbor_size,
                                 int*& nbor_list_dev,
                                 const int* mesh,
                                 const int size) {
  int* mesh_host = new int[size];
  memcpy_device_to_host(mesh, mesh_host, size);
  memcpy(&inlist.ilist, 4 + mesh_host, sizeof(int*));
  memcpy(&inlist.numneigh, 8 + mesh_host, sizeof(int*));
  memcpy(&inlist.firstneigh, 12 + mesh_host, sizeof(int**));
  const int ago = mesh_host[0];
  if (ago == 0 || gpu_inlist.inum < inlist.inum) {
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
    if (_max_nbor_size <= 256) {
      _max_nbor_size = 256;
    } else if (_max_nbor_size <= 512) {
      _max_nbor_size = 512;
    } else if (_max_nbor_size <= 1024) {
      _max_nbor_size = 1024;
    } else if (_max_nbor_size <= 2048) {
      _max_nbor_size = 2048;
    } else {
      _max_nbor_size = 4096;
    }
    if (nbor_list_dev == NULL || _max_nbor_size > max_nbor_size ||
        inum > gpu_inlist.inum) {
      delete_device_memory(nbor_list_dev);
      malloc_device_memory(nbor_list_dev, inum * _max_nbor_size);
    }
    // update info
    gpu_inlist.inum = inum;
    max_nbor_size = _max_nbor_size;

    // copy nbor list from host to the device
    std::vector<int> nbor_list_host(static_cast<size_t>(inum) * max_nbor_size,
                                    0);
    int** _firstneigh = (int**)malloc(sizeof(int*) * inum);
    for (int ii = 0; ii < inum; ii++) {
      _firstneigh[ii] = nbor_list_dev + ii * max_nbor_size;
      for (int jj = 0; jj < inlist.numneigh[ii]; jj++) {
        nbor_list_host[ii * max_nbor_size + jj] = inlist.firstneigh[ii][jj];
      }
    }
    memcpy_host_to_device(nbor_list_dev, &nbor_list_host[0],
                          static_cast<size_t>(inum) * max_nbor_size);
    memcpy_host_to_device(gpu_inlist.firstneigh, _firstneigh, inum);
    free(_firstneigh);
  }
  delete[] mesh_host;
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
