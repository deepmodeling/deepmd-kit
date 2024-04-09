#include "device.h"
#include "prod_virial.h"

template <typename FPTYPE, int THREADS_PER_BLOCK>
__global__ void atom_virial_reduction(FPTYPE* virial,
                                      const FPTYPE* atom_virial,
                                      const int nall) {
  unsigned int bid = blockIdx.x;
  unsigned int tid = threadIdx.x;
  __shared__ FPTYPE data[THREADS_PER_BLOCK];
  data[tid] = (FPTYPE)0.;
  for (int ii = tid; ii < nall; ii += THREADS_PER_BLOCK) {
    data[tid] += atom_virial[ii * 9 + bid];
  }
  __syncthreads();
  // do reduction in shared memory
  for (int ii = THREADS_PER_BLOCK >> 1; ii > 0; ii >>= 1) {
    if (tid < ii) {
      data[tid] += data[tid + ii];
    }
    __syncthreads();
  }
  // write result for this block to global memory
  if (tid == 0) {
    virial[bid] = data[0];
  }
}

template <typename FPTYPE,
          bool radial_only_ = true,
          int shared_mem_block_size_ = 64>
__global__ void virial_deriv_wrt_neighbors(FPTYPE* virial,
                                           FPTYPE* atom_virial,
                                           const FPTYPE* net_deriv,
                                           const FPTYPE* in_deriv,
                                           const FPTYPE* rij,
                                           const int* nlist,
                                           const int nloc,
                                           const int nnei) {
  const int atom_i = blockIdx.x;
  const int frame_id = 0;
  const int ndescrpt = nnei * ((radial_only_) ? (1) : (4));

  // define various pointers to for a  specific frame.
  const FPTYPE* frame_net_deriv_ = &net_deriv[frame_id * nloc * ndescrpt];
  const FPTYPE* frame_in_deriv_ = &in_deriv[frame_id * nloc * ndescrpt * 3];
  const int* frame_neighbor_list_ = &nlist[frame_id * nnei * nloc];

  FPTYPE virial_tmp[9];
  memset(virial_tmp, 0, sizeof(FPTYPE) * 9);

  for (int neighbor_id = threadIdx.x; neighbor_id < nnei;
       neighbor_id += blockDim.x) {
    // Go through all neighbors of atom i, locate the position of the atom i in
    // the neighbor list of the atom j and retrieve all necessary information.

    const int atom_j = frame_neighbor_list_[atom_i * nnei + neighbor_id];

    if (atom_j < 0) {
      continue;
    }

    const int* nei_nei_list_ = &frame_neighbor_list_[atom_j * nnei];
    int atom_id_position = 0;
    /* search for the atom i position in the neighbors list of atom_j */
    for (atom_id_position = 0; atom_id_position < nnei; atom_id_position++) {
      if (nei_nei_list_[atom_id_position] == atom_i) {
        break;
      }
    }

    if (atom_id_position == nnei) {
      break;
    }

    /* take the rij from the atom i NOT the atom j as the coordinates system is
       locally dependent */
    const FPTYPE rij_[3] = {rij[(atom_j * nnei + atom_id_position) * 3],
                            rij[(atom_j * nnei + atom_id_position) * 3 + 1],
                            rij[(atom_j * nnei + atom_id_position) * 3 + 2]};
    const int64_t offset_j =
        (atom_j * nnei + atom_id_position) * ((radial_only_) ? (1) : (4));

    for (int idw = 0; idw < ((radial_only_) ? (1) : (4)); idw++) {
      const FPTYPE cnd = net_deriv[offset_j + idw];
      const FPTYPE in_der[3] = {in_deriv[(offset_j + idw) * 3 + 0],
                                in_deriv[(offset_j + idw) * 3 + 1],
                                in_deriv[(offset_j + idw) * 3 + 2]};

      virial_tmp[0] += cnd * rij_[0] * in_der[0];
      virial_tmp[1] += cnd * rij_[1] * in_der[0];
      virial_tmp[2] += cnd * rij_[2] * in_der[0];
      virial_tmp[3] += cnd * rij_[0] * in_der[1];
      virial_tmp[4] += cnd * rij_[1] * in_der[1];
      virial_tmp[5] += cnd * rij_[2] * in_der[1];
      virial_tmp[6] += cnd * rij_[0] * in_der[2];
      virial_tmp[7] += cnd * rij_[1] * in_der[2];
      virial_tmp[8] += cnd * rij_[2] * in_der[2];
    }
  }

  __syncthreads();
  __shared__ FPTYPE vab[shared_mem_block_size_];

  /* do the reduction over the thread block component by component */
  for (int i = 0; i < 9; i++) {
    vab[threadIdx.x] = virial_tmp[i];
    __syncthreads();

    for (int tt = blockDim.x / 2; tt > 0; tt >>= 1) {
      if (threadIdx.x < tt) {
        vab[threadIdx.x] += vab[threadIdx.x + tt];
      }
      __syncthreads();
    }
    virial_tmp[i] = vab[0];
  }

  if (threadIdx.x < 9) {
    atom_virial[9 * atom_i + threadIdx.x] = virial_tmp[threadIdx.x];
  }
}

template <typename FPTYPE, bool radial_only_ = true>
void prod_virial_a_r_gpu(FPTYPE* virial,
                         FPTYPE* atom_virial,
                         const FPTYPE* net_deriv,
                         const FPTYPE* in_deriv,
                         const FPTYPE* rij,
                         const int* nlist,
                         const int nloc,
                         const int nall,
                         const int nnei) {
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuMemset(virial, 0, sizeof(FPTYPE) * 9));
  DPErrcheck(gpuMemset(atom_virial, 0, sizeof(FPTYPE) * 9 * nall));

  dim3 block_grid = dim3(nloc, 1, 1);
  // to accomodate AMD warp size
  dim3 thread_grid = dim3(64, 1, 1);
  virial_deriv_wrt_neighbors<FPTYPE, radial_only_, 64>
      <<<block_grid, thread_grid>>>(virial, atom_virial, net_deriv, in_deriv,
                                    rij, nlist, nloc, nnei);
  DPErrcheck(gpuGetLastError());
  // reduction atom_virial to virial
  atom_virial_reduction<FPTYPE, TPB><<<9, TPB>>>(virial, atom_virial, nall);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}
namespace deepmd {
template <typename FPTYPE>
void prod_virial_a_gpu(FPTYPE* virial,
                       FPTYPE* atom_virial,
                       const FPTYPE* net_deriv,
                       const FPTYPE* in_deriv,
                       const FPTYPE* rij,
                       const int* nlist,
                       const int nloc,
                       const int nall,
                       const int nnei) {
  prod_virial_a_r_gpu<FPTYPE, true>(virial, atom_virial, net_deriv, in_deriv,
                                    rij, nlist, nloc, nall, nnei);
}

template <typename FPTYPE>
void prod_virial_r_gpu(FPTYPE* virial,
                       FPTYPE* atom_virial,
                       const FPTYPE* net_deriv,
                       const FPTYPE* in_deriv,
                       const FPTYPE* rij,
                       const int* nlist,
                       const int nloc,
                       const int nall,
                       const int nnei) {
  prod_virial_a_r_gpu<FPTYPE, false>(virial, atom_virial, net_deriv, in_deriv,
                                     rij, nlist, nloc, nall, nnei);
}

template void prod_virial_a_gpu<float>(float* virial,
                                       float* atom_virial,
                                       const float* net_deriv,
                                       const float* in_deriv,
                                       const float* rij,
                                       const int* nlist,
                                       const int nloc,
                                       const int nall,
                                       const int nnei);
template void prod_virial_a_gpu<double>(double* virial,
                                        double* atom_virial,
                                        const double* net_deriv,
                                        const double* in_deriv,
                                        const double* rij,
                                        const int* nlist,
                                        const int nloc,
                                        const int nall,
                                        const int nnei);
template void prod_virial_r_gpu<float>(float* virial,
                                       float* atom_virial,
                                       const float* net_deriv,
                                       const float* in_deriv,
                                       const float* rij,
                                       const int* nlist,
                                       const int nloc,
                                       const int nall,
                                       const int nnei);
template void prod_virial_r_gpu<double>(double* virial,
                                        double* atom_virial,
                                        const double* net_deriv,
                                        const double* in_deriv,
                                        const double* rij,
                                        const int* nlist,
                                        const int nloc,
                                        const int nall,
                                        const int nnei);
}  // namespace deepmd
