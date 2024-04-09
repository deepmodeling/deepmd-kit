#include "device.h"
#include "prod_force.h"

template <typename FPTYPE, int THREADS_PER_BLOCK>
__global__ void force_deriv_wrt_center_atom(FPTYPE* force,
                                            const FPTYPE* net_deriv,
                                            const FPTYPE* in_deriv,
                                            const int ndescrpt,
                                            const int nloc,
                                            const int nall) {
  __shared__ FPTYPE data[THREADS_PER_BLOCK * 3];
  int_64 bid = blockIdx.x;
  unsigned int tid = threadIdx.x;
  for (int ii = tid; ii < THREADS_PER_BLOCK * 3; ii += THREADS_PER_BLOCK) {
    data[ii] = (FPTYPE)0.;
  }
  for (int ii = tid; ii < ndescrpt; ii += THREADS_PER_BLOCK) {
    for (int jj = 0; jj < 3; jj++) {
      data[jj * THREADS_PER_BLOCK + tid] +=
          net_deriv[bid * ndescrpt + ii] *
          in_deriv[bid * ndescrpt * 3 + ii * 3 + jj];
    }
  }
  __syncthreads();
  // do reduction in shared memory
  for (int ii = THREADS_PER_BLOCK >> 1; ii > 0; ii >>= 1) {
    if (tid < ii) {
      for (int jj = 0; jj < 3; jj++) {
        data[jj * THREADS_PER_BLOCK + tid] +=
            data[jj * THREADS_PER_BLOCK + tid + ii];
      }
    }
    __syncthreads();
  }
  // write result for this block to global memory
  const int_64 kk = bid / nloc;  // frame index
  const int_64 ll = bid % nloc;  // atom index
  const int_64 i_idx_nall = kk * nall + ll;
  if (tid == 0) {
    force[i_idx_nall * 3 + 0] -= data[THREADS_PER_BLOCK * 0];
    force[i_idx_nall * 3 + 1] -= data[THREADS_PER_BLOCK * 1];
    force[i_idx_nall * 3 + 2] -= data[THREADS_PER_BLOCK * 2];
  }
}

template <typename FPTYPE,
          bool radial_only_ = false,
          int shared_memory_block_ = 32>
__global__ void force_deriv_wrt_neighbors(FPTYPE* force,
                                          const FPTYPE* net_deriv,
                                          const FPTYPE* in_deriv,
                                          const int* nlist,
                                          const int nframes,
                                          const int nloc,
                                          const int nall,
                                          const int nnei) {
  // limited to 2 billions atoms and 2 billions frames
  const int atom_id = blockIdx.x;
  const int frame_id = blockIdx.z * gridDim.y + blockIdx.y;

  if (frame_id >= nframes) {
    return;
  }

  const int ndescrpt = nnei * ((radial_only_) ? (1) : (4));

  // define various pointers for a specific frame.
  const FPTYPE* frame_net_deriv_ = &net_deriv[frame_id * nloc * ndescrpt];
  const FPTYPE* frame_in_deriv_ = &in_deriv[frame_id * nloc * ndescrpt * 3];
  const int* frame_neighbor_list_ = &nlist[frame_id * nnei * nloc];
  FPTYPE force_tmp[3] = {(FPTYPE)0., (FPTYPE)0., (FPTYPE)0.};

  for (int neighbor_id = threadIdx.x; neighbor_id < nnei;
       neighbor_id += blockDim.x) {
    // collect all terms $\partial E_j / \partial D_{ji} \nabla_R_j D_{ji}$
    // where the atom i is a neighbor of the atom j.
    //
    // Go through all neighbors of atom i, locate the position of
    // the atom i in the neighbor list of the atom j and retrieve all necessary
    // information.

    const int atom_j = frame_neighbor_list_[atom_id * nnei + neighbor_id];

    // The neighbors of a given atom are sorted by type and each resulting list
    // is separated from the other by a series of -1. More details about the
    // sorting can be found in https://doi.org/10.1016/j.cpc.2020.107624
    //
    // To illustrate this, take the neigbhors of a given atom of type a (in a
    // system with two atoms type a and b) deepmd stores the neighbors as
    //
    // [neighbors list of type a], -1, -1, -1, ...., [neighbor list of type b],
    // -1, -1, -1, .....

    if (atom_j < 0) {
      continue;
    }

    const int* nei_nei_list_ = &frame_neighbor_list_[atom_j * nnei];
    int atom_id_position = 0;

    // search the index of the atom i in the local neighbor list of atom j
    for (atom_id_position = 0; atom_id_position < nnei; atom_id_position++) {
      if (nei_nei_list_[atom_id_position] == atom_id) {
        break;
      }
    }

    const int64_t offset_j =
        (atom_j * nnei + atom_id_position) * ((radial_only_) ? (1) : (4));
    for (int idw = 0; idw < ((radial_only_) ? (1) : (4)); ++idw) {
      const FPTYPE cst1 = frame_net_deriv_[offset_j + idw];
      force_tmp[0] += cst1 * in_deriv[(offset_j + idw) * 3 + 0];
      force_tmp[1] += cst1 * in_deriv[(offset_j + idw) * 3 + 1];
      force_tmp[2] += cst1 * in_deriv[(offset_j + idw) * 3 + 2];
    }
  }

  __shared__ FPTYPE fx[shared_memory_block_];
  __shared__ FPTYPE fy[shared_memory_block_];
  __shared__ FPTYPE fz[shared_memory_block_];

  fx[threadIdx.x] = force_tmp[0];
  fy[threadIdx.x] = force_tmp[1];
  fz[threadIdx.x] = force_tmp[2];
  __syncthreads();

  // do the final reduction
  for (int tt = shared_memory_block_ / 2; tt > 0; tt >>= 1) {
    if (threadIdx.x < tt) {
      fx[threadIdx.x] += fx[threadIdx.x + tt];
      fy[threadIdx.x] += fy[threadIdx.x + tt];
      fz[threadIdx.x] += fz[threadIdx.x + tt];
    }
    __syncthreads();
  }

  /* Note the sign difference between the formula in the PRL paper and the code.
     it is due to \nabla_R_j D_{ji} = -\nabla_R_i D_{ji} */
  if (threadIdx.x == 0) {
    const int64_t offset = (frame_id * nall + atom_id) * 3;
    force[offset] += fx[0];
    force[offset + 1] += fy[0];
    force[offset + 2] += fz[0];
  }
}

template <typename FPTYPE, bool radial_only_ = true>
void prod_force_a_r_gpu(FPTYPE* force,
                        const FPTYPE* net_deriv,
                        const FPTYPE* in_deriv,
                        const int* nlist,
                        const int nloc,
                        const int nall,
                        const int nnei,
                        const int nframes) {
  DPErrcheck(gpuGetLastError());
  const int ndescrpt = nnei * 4;
  DPErrcheck(gpuMemset(force, 0, sizeof(FPTYPE) * nframes * nall * 3));

  force_deriv_wrt_center_atom<FPTYPE, TPB><<<nframes * nloc, TPB>>>(
      force, net_deriv, in_deriv, ndescrpt, nloc, nall);
  DPErrcheck(gpuGetLastError());

  const int sqrt_nframes = sqrt(nframes);
  dim3 block_grid(nloc, sqrt_nframes + 1, sqrt_nframes + 1);
  dim3 thread_grid(64, 1, 1);
  force_deriv_wrt_neighbors<FPTYPE, radial_only_, 64>
      <<<block_grid, thread_grid>>>(force, net_deriv, in_deriv, nlist, nframes,
                                    nloc, nall, nnei);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}
namespace deepmd {
template <typename FPTYPE>
void prod_force_a_gpu(FPTYPE* force,
                      const FPTYPE* net_deriv,
                      const FPTYPE* in_deriv,
                      const int* nlist,
                      const int nloc,
                      const int nall,
                      const int nnei,
                      const int nframes) {
  prod_force_a_r_gpu<FPTYPE, false>(force, net_deriv, in_deriv, nlist, nloc,
                                    nall, nnei, nframes);
}

template <typename FPTYPE>
void prod_force_r_gpu(FPTYPE* force,
                      const FPTYPE* net_deriv,
                      const FPTYPE* in_deriv,
                      const int* nlist,
                      const int nloc,
                      const int nall,
                      const int nnei,
                      const int nframes) {
  prod_force_a_r_gpu<FPTYPE, true>(force, net_deriv, in_deriv, nlist, nloc,
                                   nall, nnei, nframes);
}

template void prod_force_a_gpu<float>(float* force,
                                      const float* net_deriv,
                                      const float* in_deriv,
                                      const int* nlist,
                                      const int nloc,
                                      const int nall,
                                      const int nnei,
                                      const int nframes);
template void prod_force_a_gpu<double>(double* force,
                                       const double* net_deriv,
                                       const double* in_deriv,
                                       const int* nlist,
                                       const int nloc,
                                       const int nall,
                                       const int nnei,
                                       const int nframes);
template void prod_force_r_gpu<float>(float* force,
                                      const float* net_deriv,
                                      const float* in_deriv,
                                      const int* nlist,
                                      const int nloc,
                                      const int nall,
                                      const int nnei,
                                      const int nframes);
template void prod_force_r_gpu<double>(double* force,
                                       const double* net_deriv,
                                       const double* in_deriv,
                                       const int* nlist,
                                       const int nloc,
                                       const int nall,
                                       const int nnei,
                                       const int nframes);
}  // namespace deepmd
