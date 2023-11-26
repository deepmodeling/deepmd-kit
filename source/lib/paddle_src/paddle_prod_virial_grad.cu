#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_store.cuh>

#include "device.h"
#include "gpu_cuda.h"
#include "paddle/extension.h"
#include "prod_virial.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")
#define CHECK_INPUT_CPU(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")
#define CHECK_INPUT_DIM(x, value) \
  PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")
#define CHECK_INPUT_READY(x) \
  PD_CHECK(x.initialized(), #x " must be initialized before usage.")

template <typename FPTYPE>
__device__ inline FPTYPE dev_dot9(const FPTYPE* arr1, const FPTYPE* arr2) {
  FPTYPE result = (FPTYPE)0.0;
  for (int ii = 0; ii < 9; ii++) {
    result += arr1[ii] * arr2[ii];
  }
  return result;
}

template <typename FPTYPE>
__global__ void virial_grad_wrt_neighbors_a(FPTYPE* grad_net,
                                            const FPTYPE* grad,
                                            const FPTYPE* env_deriv,
                                            const FPTYPE* rij,
                                            const int* nlist,
                                            const int nloc,
                                            const int nnei) {
  // idy -> nnei
  const unsigned int tid = threadIdx.x;
  const int_64 idx = blockIdx.x * blockDim.x + tid;
  const unsigned int idy = blockIdx.y;
  const unsigned int idw = threadIdx.y;
  const int ndescrpt = nnei * 4;
  __shared__ FPTYPE grad_one[9];
  if (tid < 9) {
    grad_one[tid] = grad[tid];
  }
  __syncthreads();
  if (idx >= nloc) {
    return;
  }
  int j_idx = nlist[idx * nnei + idy];
  if (j_idx < 0) {
    return;
  }
  FPTYPE tmp[9];
  for (int dd0 = 0; dd0 < 3; ++dd0) {
    for (int dd1 = 0; dd1 < 3; ++dd1) {
      tmp[dd0 * 3 + dd1] =
          rij[idx * nnei * 3 + idy * 3 + dd1] *
          env_deriv[idx * ndescrpt * 3 + idy * 4 * 3 + idw * 3 + dd0];
    }
  }
  grad_net[idx * ndescrpt + idy * 4 + idw] -=
      (FPTYPE)-1.0 * dev_dot9(grad_one, tmp);
}

namespace deepmd {
template <typename FPTYPE>
void prod_virial_grad_a_gpu_cuda(FPTYPE* grad_net,
                                 const FPTYPE* grad,
                                 const FPTYPE* env_deriv,
                                 const FPTYPE* rij,
                                 const int* nlist,
                                 const int nloc,
                                 const int nnei) {
  const int ndescrpt = nnei * 4;
  DPErrcheck(cudaMemset(grad_net, 0, sizeof(FPTYPE) * nloc * ndescrpt));
  const int LEN = 128;
  const int nblock = (nloc + LEN - 1) / LEN;
  dim3 block_grid(nblock, nnei);
  dim3 thread_grid(LEN, 4);
  virial_grad_wrt_neighbors_a<<<block_grid, thread_grid>>>(
      grad_net, grad, env_deriv, rij, nlist, nloc, nnei);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template void prod_virial_grad_a_gpu_cuda<float>(float* grad_net,
                                                 const float* grad,
                                                 const float* env_deriv,
                                                 const float* rij,
                                                 const int* nlist,
                                                 const int nloc,
                                                 const int nnei);
template void prod_virial_grad_a_gpu_cuda<double>(double* grad_net,
                                                  const double* grad,
                                                  const double* env_deriv,
                                                  const double* rij,
                                                  const int* nlist,
                                                  const int nloc,
                                                  const int nnei);
}  // namespace deepmd

template <typename data_t>
void ProdForceSeAOpGPUBackwardKernel(int nloc,
                                     int nframes,
                                     int ndescrpt,
                                     int nnei,
                                     const data_t* virial_grad,
                                     const data_t* net_deriv,
                                     const data_t* in_deriv,
                                     const data_t* rij,
                                     const int* nlist,
                                     data_t* grad_net) {
  data_t* p_grad_net = grad_net;
  const data_t* p_grad = virial_grad;
  const data_t* p_in_deriv = in_deriv;
  const data_t* p_rij = rij;
  const int* p_nlist = nlist;
  for (int_64 kk = 0; kk < nframes; ++kk) {
    data_t* grad_net = p_grad_net + kk * nloc * ndescrpt;
    const data_t* virial_grad = p_grad + kk * 9;
    const data_t* in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
    const data_t* rij = p_rij + kk * nloc * nnei * 3;
    const int* nlist = p_nlist + kk * nloc * nnei;
    deepmd::prod_virial_grad_a_gpu_cuda(grad_net, virial_grad, in_deriv, rij,
                                        nlist, nloc, nnei);
  }
}

std::vector<paddle::Tensor> ProdVirialSeAOpCUDABackward(
    const paddle::Tensor& virial_grad_tensor,
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& rij_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel) {
  CHECK_INPUT_READY(virial_grad_tensor);
  CHECK_INPUT_READY(net_deriv_tensor);
  CHECK_INPUT_READY(in_deriv_tensor);
  CHECK_INPUT_READY(rij_tensor);
  CHECK_INPUT_READY(nlist_tensor);
  CHECK_INPUT_READY(natoms_tensor);

  auto grad_shape = virial_grad_tensor.shape();
  auto net_deriv_shape = net_deriv_tensor.shape();
  auto in_deriv_shape = in_deriv_tensor.shape();
  auto rij_shape = rij_tensor.shape();
  auto nlist_shape = nlist_tensor.shape();
  auto natoms_shape = natoms_tensor.shape();

  CHECK_INPUT_DIM(virial_grad_tensor, 2);
  CHECK_INPUT_DIM(net_deriv_tensor, 2);
  CHECK_INPUT_DIM(in_deriv_tensor, 2);
  CHECK_INPUT_DIM(rij_tensor, 2);
  CHECK_INPUT_DIM(nlist_tensor, 2);
  CHECK_INPUT_DIM(natoms_tensor, 1);

  PD_CHECK(natoms_shape[0] >= 3,
           "number of atoms should be larger than (or equal to) 3");

  CHECK_INPUT_CPU(natoms_tensor);
  const int* natoms = natoms_tensor.data<int>();
  int nframes = net_deriv_shape[0];
  int nloc = natoms[0];
  int ndescrpt = net_deriv_shape[1] / nloc;
  int nnei = nlist_shape[1] / nloc;

  PD_CHECK(nframes == grad_shape[0], "number of frames should match");
  PD_CHECK(nframes == in_deriv_shape[0], "number of samples should match");
  PD_CHECK(nframes == rij_shape[0], "number of frames should match");
  PD_CHECK(nframes == nlist_shape[0], "number of samples should match");
  PD_CHECK(9 == grad_shape[1], "input grad shape should be 3 x natoms");
  PD_CHECK(nloc * ndescrpt * 3 == in_deriv_shape[1],
           "number of descriptors should match");
  PD_CHECK(nloc * nnei * 3 == rij_shape[1], "dim of rij should be  nnei * 3");
  PD_CHECK(nnei == (n_a_sel + n_r_sel), "number of neighbors should match");

  std::vector<int64_t> grad_net_shape{nframes, nloc * ndescrpt};
  paddle::Tensor grad_net_tensor = paddle::empty(
      grad_net_shape, virial_grad_tensor.dtype(), virial_grad_tensor.place());

  PD_DISPATCH_FLOATING_TYPES(
      virial_grad_tensor.type(), "prod_force_se_a_cuda_backward_kernel", ([&] {
        ProdForceSeAOpGPUBackwardKernel<data_t>(
            nloc, nframes, ndescrpt, nnei, virial_grad_tensor.data<data_t>(),
            net_deriv_tensor.data<data_t>(), in_deriv_tensor.data<data_t>(),
            rij_tensor.data<data_t>(), nlist_tensor.data<int>(),
            grad_net_tensor.data<data_t>());
      }));
  return {grad_net_tensor};
}
