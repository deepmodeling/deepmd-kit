#include "device.h"
#include "prod_force_grad.h"

template <typename FPTYPE>
__device__ inline FPTYPE dev_dot(const FPTYPE* arr1, const FPTYPE* arr2) {
  return arr1[0] * arr2[0] + arr1[1] * arr2[1] + arr1[2] * arr2[2];
}

template <typename FPTYPE>
__global__ void force_grad_wrt_center_atom(FPTYPE* grad_net,
                                           const FPTYPE* grad,
                                           const FPTYPE* env_deriv,
                                           const int ndescrpt) {
  __shared__ FPTYPE grad_one[3];
  int_64 center_idx = blockIdx.x;
  unsigned int tid = threadIdx.x;
  if (tid < 3) {
    grad_one[tid] = grad[center_idx * 3 + tid];
  }
  __syncthreads();
  unsigned int descrpt_idx = blockIdx.y * blockDim.x + tid;
  if (descrpt_idx < ndescrpt) {
    grad_net[center_idx * ndescrpt + descrpt_idx] -= dev_dot(
        grad_one, env_deriv + center_idx * ndescrpt * 3 + descrpt_idx * 3);
  }
}

template <typename FPTYPE>
__global__ void force_grad_wrt_neighbors_a(FPTYPE* grad_net,
                                           const FPTYPE* grad,
                                           const FPTYPE* env_deriv,
                                           const int* nlist,
                                           const int nloc,
                                           const int nnei,
                                           const int nframes) {
  // idy -> nnei
  const int_64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y;
  const unsigned int idw = threadIdx.y;
  if (idx >= nframes * nloc) {
    return;
  }
  int j_idx = nlist[idx * nnei + idy];
  if (j_idx < 0) {
    return;
  }
  if (j_idx >= nloc) {
    j_idx = j_idx % nloc;
  }
  const int kk = idx / nloc;  // frame index
  grad_net[idx * nnei * 4 + idy * 4 + idw] +=
      dev_dot(grad + kk * nloc * 3 + j_idx * 3,
              env_deriv + idx * nnei * 4 * 3 + idy * 4 * 3 + idw * 3);
}

template <typename FPTYPE>
__global__ void force_grad_wrt_neighbors_r(FPTYPE* grad_net,
                                           const FPTYPE* grad,
                                           const FPTYPE* env_deriv,
                                           const int* nlist,
                                           const int nloc,
                                           const int nnei,
                                           const int nframes) {
  // idy -> nnei
  const int_64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y;
  if (idx >= nframes * nloc) {
    return;
  }
  int j_idx = nlist[idx * nnei + idy];
  if (j_idx < 0) {
    return;
  }
  if (j_idx >= nloc) {
    j_idx = j_idx % nloc;
  }
  const int kk = idx / nloc;  // frame index
  grad_net[idx * nnei + idy] += dev_dot(grad + kk * nloc * 3 + j_idx * 3,
                                        env_deriv + idx * nnei * 3 + idy * 3);
}

namespace deepmd {
template <typename FPTYPE>
void prod_force_grad_a_gpu(FPTYPE* grad_net,
                           const FPTYPE* grad,
                           const FPTYPE* env_deriv,
                           const int* nlist,
                           const int nloc,
                           const int nnei,
                           const int nframes) {
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int ndescrpt = nnei * 4;
  DPErrcheck(
      gpuMemset(grad_net, 0, sizeof(FPTYPE) * nframes * nloc * ndescrpt));
  const int nblock = (ndescrpt + TPB - 1) / TPB;
  dim3 block_grid(nframes * nloc, nblock);
  dim3 thread_grid(TPB, 1);
  force_grad_wrt_center_atom<<<block_grid, thread_grid>>>(grad_net, grad,
                                                          env_deriv, ndescrpt);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());

  const int LEN = 128;
  const int nblock_ = (nframes * nloc + LEN - 1) / LEN;
  dim3 block_grid_(nblock_, nnei);
  dim3 thread_grid_(LEN, 4);
  force_grad_wrt_neighbors_a<<<block_grid_, thread_grid_>>>(
      grad_net, grad, env_deriv, nlist, nloc, nnei, nframes);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template <typename FPTYPE>
void prod_force_grad_r_gpu(FPTYPE* grad_net,
                           const FPTYPE* grad,
                           const FPTYPE* env_deriv,
                           const int* nlist,
                           const int nloc,
                           const int nnei,
                           const int nframes) {
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
  const int ndescrpt = nnei * 1;
  DPErrcheck(
      gpuMemset(grad_net, 0, sizeof(FPTYPE) * nframes * nloc * ndescrpt));
  const int nblock = (ndescrpt + TPB - 1) / TPB;
  dim3 block_grid(nframes * nloc, nblock);
  dim3 thread_grid(TPB, 1);
  force_grad_wrt_center_atom<<<block_grid, thread_grid>>>(grad_net, grad,
                                                          env_deriv, ndescrpt);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());

  const int LEN = 128;
  const int nblock_ = (nframes * nloc + LEN - 1) / LEN;
  dim3 block_grid_(nblock_, nnei);
  dim3 thread_grid_(LEN, 1);
  force_grad_wrt_neighbors_r<<<block_grid_, thread_grid_>>>(
      grad_net, grad, env_deriv, nlist, nloc, nnei, nframes);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

template void prod_force_grad_a_gpu<float>(float* grad_net,
                                           const float* grad,
                                           const float* env_deriv,
                                           const int* nlist,
                                           const int nloc,
                                           const int nnei,
                                           const int nframes);
template void prod_force_grad_a_gpu<double>(double* grad_net,
                                            const double* grad,
                                            const double* env_deriv,
                                            const int* nlist,
                                            const int nloc,
                                            const int nnei,
                                            const int nframes);
template void prod_force_grad_r_gpu<float>(float* grad_net,
                                           const float* grad,
                                           const float* env_deriv,
                                           const int* nlist,
                                           const int nloc,
                                           const int nnei,
                                           const int nframes);
template void prod_force_grad_r_gpu<double>(double* grad_net,
                                            const double* grad,
                                            const double* env_deriv,
                                            const int* nlist,
                                            const int nloc,
                                            const int nnei,
                                            const int nframes);
}  // namespace deepmd
