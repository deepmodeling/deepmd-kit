#include "device.h"
#include "gpu_cuda.h"
#include "paddle/extension.h"
#include "prod_force_grad.h"

#define CHECK_INPUT_DIM(x, value) \
  PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")
#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")
#define CHECK_INPUT_ON_CPU(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

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
                                           const int nnei) {
  // idy -> nnei
  const int_64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y;
  const unsigned int idw = threadIdx.y;
  if (idx >= nloc) {
    return;
  }
  int j_idx = nlist[idx * nnei + idy];
  if (j_idx < 0) {
    return;
  }
  if (j_idx >= nloc) j_idx = j_idx % nloc;
  grad_net[idx * nnei * 4 + idy * 4 + idw] += dev_dot(
      grad + j_idx * 3, env_deriv + idx * nnei * 4 * 3 + idy * 4 * 3 + idw * 3);
}

template <typename FPTYPE>
__global__ void force_grad_wrt_neighbors_r(FPTYPE* grad_net,
                                           const FPTYPE* grad,
                                           const FPTYPE* env_deriv,
                                           const int* nlist,
                                           const int nloc,
                                           const int nnei) {
  // idy -> nnei
  const int_64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y;
  if (idx >= nloc) {
    return;
  }
  int j_idx = nlist[idx * nnei + idy];
  if (j_idx < 0) {
    return;
  }
  if (j_idx >= nloc) j_idx = j_idx % nloc;
  grad_net[idx * nnei + idy] +=
      dev_dot(grad + j_idx * 3, env_deriv + idx * nnei * 3 + idy * 3);
}

namespace deepmd {
template <typename FPTYPE>
void prod_force_grad_a_gpu_cuda(FPTYPE* grad_net,
                                const FPTYPE* grad,
                                const FPTYPE* env_deriv,
                                const int* nlist,
                                const int nloc,
                                const int nnei) {
  const int ndescrpt = nnei * 4;
  DPErrcheck(cudaMemset(grad_net, 0, sizeof(FPTYPE) * nloc * ndescrpt));
  const int nblock = (ndescrpt + TPB - 1) / TPB;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(TPB, 1);
  force_grad_wrt_center_atom<<<block_grid, thread_grid>>>(grad_net, grad,
                                                          env_deriv, ndescrpt);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());

  const int LEN = 128;
  const int nblock_ = (nloc + LEN - 1) / LEN;
  dim3 block_grid_(nblock_, nnei);
  dim3 thread_grid_(LEN, 4);
  force_grad_wrt_neighbors_a<<<block_grid_, thread_grid_>>>(
      grad_net, grad, env_deriv, nlist, nloc, nnei);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template <typename FPTYPE>
void prod_force_grad_r_gpu_cuda(FPTYPE* grad_net,
                                const FPTYPE* grad,
                                const FPTYPE* env_deriv,
                                const int* nlist,
                                const int nloc,
                                const int nnei) {
  const int ndescrpt = nnei * 1;
  DPErrcheck(cudaMemset(grad_net, 0, sizeof(FPTYPE) * nloc * ndescrpt));
  const int nblock = (ndescrpt + TPB - 1) / TPB;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(TPB, 1);
  force_grad_wrt_center_atom<<<block_grid, thread_grid>>>(grad_net, grad,
                                                          env_deriv, ndescrpt);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());

  const int LEN = 128;
  const int nblock_ = (nloc + LEN - 1) / LEN;
  dim3 block_grid_(nblock_, nnei);
  dim3 thread_grid_(LEN, 1);
  force_grad_wrt_neighbors_r<<<block_grid_, thread_grid_>>>(
      grad_net, grad, env_deriv, nlist, nloc, nnei);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template void prod_force_grad_a_gpu_cuda<float>(float* grad_net,
                                                const float* grad,
                                                const float* env_deriv,
                                                const int* nlist,
                                                const int nloc,
                                                const int nnei);
template void prod_force_grad_a_gpu_cuda<double>(double* grad_net,
                                                 const double* grad,
                                                 const double* env_deriv,
                                                 const int* nlist,
                                                 const int nloc,
                                                 const int nnei);
template void prod_force_grad_r_gpu_cuda<float>(float* grad_net,
                                                const float* grad,
                                                const float* env_deriv,
                                                const int* nlist,
                                                const int nloc,
                                                const int nnei);
template void prod_force_grad_r_gpu_cuda<double>(double* grad_net,
                                                 const double* grad,
                                                 const double* env_deriv,
                                                 const int* nlist,
                                                 const int nloc,
                                                 const int nnei);
}  // namespace deepmd

template <typename data_t>
void ProdForceSeAOpCUDABackwardKernel(int nloc,
                                      int nframes,
                                      int ndescrpt,
                                      int nnei,
                                      const data_t* p_grad,
                                      const data_t* p_net_deriv,
                                      const data_t* p_in_deriv,
                                      const int* p_nlist,
                                      data_t* p_grad_net) {
  for (int_64 kk = 0; kk < nframes; ++kk) {
    data_t* grad_net = p_grad_net + kk * nloc * ndescrpt;
    const data_t* grad = p_grad + kk * nloc * 3;
    const data_t* in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
    const int* nlist = p_nlist + kk * nloc * nnei;
    deepmd::prod_force_grad_a_gpu_cuda(grad_net, grad, in_deriv, nlist, nloc,
                                       nnei);
  }
}

std::vector<paddle::Tensor> ProdForceSeAOpCUDABackward(
    const paddle::Tensor& force_grad_tensor,
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel) {
  auto grad_shape = force_grad_tensor.shape();
  auto net_deriv_shape = net_deriv_tensor.shape();
  auto in_deriv_shape = in_deriv_tensor.shape();
  auto nlist_shape = nlist_tensor.shape();
  auto natoms_shape = natoms_tensor.shape();

  CHECK_INPUT_DIM(force_grad_tensor, 2);
  CHECK_INPUT_DIM(net_deriv_tensor, 2);
  CHECK_INPUT_DIM(in_deriv_tensor, 2);
  CHECK_INPUT_DIM(nlist_tensor, 2);
  CHECK_INPUT_DIM(natoms_tensor, 1);

  PD_CHECK(natoms_shape[0] >= 3,
           "number of atoms should be larger than (or equal to) 3");

  CHECK_INPUT_ON_CPU(natoms_tensor);
  const int* natoms = natoms_tensor.data<int>();
  int nframes = net_deriv_shape[0];
  int nloc = natoms[0];
  int ndescrpt = net_deriv_shape[1] / nloc;
  int nnei = nlist_shape[1] / nloc;

  PD_CHECK(nframes == grad_shape[0], "number of frames should match");
  PD_CHECK(nframes == in_deriv_shape[0], "number of samples should match");
  PD_CHECK(nframes == nlist_shape[0], "number of samples should match");
  PD_CHECK(nloc * 3 == grad_shape[1], "input grad shape should be 3 x natoms");
  PD_CHECK(nloc * ndescrpt * 3 == in_deriv_shape[1],
           "number of descriptors should match");
  PD_CHECK(nnei == (n_a_sel + n_r_sel), "number of neighbors should match");

  std::vector<int64_t> grad_net_shape{nframes, nloc * ndescrpt};
  paddle::Tensor grad_net_tensor = paddle::empty(
      grad_net_shape, force_grad_tensor.dtype(), force_grad_tensor.place());

  PD_DISPATCH_FLOATING_TYPES(
      force_grad_tensor.type(), "prod_force_se_a_cuda_backward_kernel", ([&] {
        ProdForceSeAOpCUDABackwardKernel<data_t>(
            nloc, nframes, ndescrpt, nnei, force_grad_tensor.data<data_t>(),
            net_deriv_tensor.data<data_t>(), in_deriv_tensor.data<data_t>(),
            nlist_tensor.data<int>(), grad_net_tensor.data<data_t>());
      }));
  return {grad_net_tensor};
}

// std::vector<paddle::Tensor> ProdForceSeABackward(
//     const paddle::Tensor& force_grad_tensor,
//     const paddle::Tensor& net_deriv_tensor,
//     const paddle::Tensor& in_deriv_tensor,
//     const paddle::Tensor& nlist_tensor,
//     const paddle::Tensor& natoms_tensor,
//     int n_a_sel,
//     int n_r_sel) {
//   if (net_deriv_tensor.place() == paddle::GPUPlace()) {
//     return ProdForceSeAOpCUDABackward(force_grad_tensor, net_deriv_tensor,
//                                       in_deriv_tensor, nlist_tensor,
//                                       natoms_tensor, n_a_sel, n_r_sel);
//   }
//   else if (net_deriv_tensor.place() == paddle::CPUPlace()) {
//     return ProdForceSeAOpCPUBackward(force_grad_tensor, net_deriv_tensor,
//                                      in_deriv_tensor, nlist_tensor,
//                                      natoms_tensor, n_a_sel, n_r_sel);
//   } else {
//     PD_THROW("No Such kernel for ProdForceSeABackward.");
//   }
// }

// PD_BUILD_GRAD_OP(prod_force_se_a)
//     .Inputs({paddle::Grad("force"), "net_deriv", "in_deriv", "nlist",
//     "natoms"}) .Outputs({paddle::Grad("net_deriv")}) .Attrs({"n_a_sel: int",
//     "n_r_sel: int"}) .SetKernelFn(PD_KERNEL(ProdForceSeABackward));
