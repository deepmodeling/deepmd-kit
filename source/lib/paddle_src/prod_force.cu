#include "paddle/extension.h"

#include "device.h"
#include "prod_force.h"
#include "gpu_cuda.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")
#define CHECK_INPUT_DIM(x, value) PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")

template <typename FPTYPE, int THREADS_PER_BLOCK>
__global__ void force_deriv_wrt_center_atom(FPTYPE* force,
                                            const FPTYPE* net_deriv,
                                            const FPTYPE* in_deriv,
                                            const int ndescrpt) {
  __shared__ FPTYPE data[THREADS_PER_BLOCK * 3];
  int_64 bid = blockIdx.x;
  unsigned int tid = threadIdx.x;
  for (int ii = tid; ii < THREADS_PER_BLOCK * 3; ii += THREADS_PER_BLOCK) {
    data[ii] = 0.f;
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
  if (tid == 0) {
    force[bid * 3 + 0] -= data[THREADS_PER_BLOCK * 0];
    force[bid * 3 + 1] -= data[THREADS_PER_BLOCK * 1];
    force[bid * 3 + 2] -= data[THREADS_PER_BLOCK * 2];
  }
}

template <typename FPTYPE>
__global__ void force_deriv_wrt_neighbors_a(FPTYPE* force,
                                            const FPTYPE* net_deriv,
                                            const FPTYPE* in_deriv,
                                            const int* nlist,
                                            const int nloc,
                                            const int nnei) {
  // idy -> nnei
  const int_64 idx = blockIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.x + threadIdx.x;
  const unsigned int idz = threadIdx.y;
  const int ndescrpt = nnei * 4;
  if (idy >= nnei) {
    return;
  }
  // deriv wrt neighbors
  int j_idx = nlist[idx * nnei + idy];
  if (j_idx < 0) {
    return;
  }
  FPTYPE force_tmp = 0.f;
  for (int idw = 0; idw < 4; ++idw) {
    force_tmp += net_deriv[idx * ndescrpt + idy * 4 + idw] *
                 in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz];
  }
  atomicAdd(force + j_idx * 3 + idz, force_tmp);
}

template <typename FPTYPE>
__global__ void force_deriv_wrt_neighbors_r(FPTYPE* force,
                                            const FPTYPE* net_deriv,
                                            const FPTYPE* in_deriv,
                                            const int* nlist,
                                            const int nloc,
                                            const int nnei) {
  // idy -> nnei
  const int_64 idx = blockIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.x + threadIdx.x;
  const unsigned int idz = threadIdx.y;
  const int ndescrpt = nnei * 1;
  if (idy >= nnei) {
    return;
  }
  // deriv wrt neighbors
  int j_idx = nlist[idx * nnei + idy];
  if (j_idx < 0) {
    return;
  }
  atomicAdd(force + j_idx * 3 + idz,
            net_deriv[idx * ndescrpt + idy] *
                in_deriv[idx * ndescrpt * 3 + idy * 3 + idz]);
}

namespace deepmd {
template <typename FPTYPE>
void prod_force_a_gpu_cuda(FPTYPE* force,
                           const FPTYPE* net_deriv,
                           const FPTYPE* in_deriv,
                           const int* nlist,
                           const int nloc,
                           const int nall,
                           const int nnei) {
  const int ndescrpt = nnei * 4;
  DPErrcheck(cudaMemset(force, 0, sizeof(FPTYPE) * nall * 3));

  force_deriv_wrt_center_atom<FPTYPE, TPB>
      <<<nloc, TPB>>>(force, net_deriv, in_deriv, ndescrpt);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());

  const int LEN = 64;
  const int nblock = (nnei + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(LEN, 3);
  force_deriv_wrt_neighbors_a<<<block_grid, thread_grid>>>(
      force, net_deriv, in_deriv, nlist, nloc, nnei);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template <typename FPTYPE>
void prod_force_r_gpu_cuda(FPTYPE* force,
                           const FPTYPE* net_deriv,
                           const FPTYPE* in_deriv,
                           const int* nlist,
                           const int nloc,
                           const int nall,
                           const int nnei) {
  const int ndescrpt = nnei * 1;
  DPErrcheck(cudaMemset(force, 0, sizeof(FPTYPE) * nall * 3));

  force_deriv_wrt_center_atom<FPTYPE, TPB>
      <<<nloc, TPB>>>(force, net_deriv, in_deriv, ndescrpt);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());

  const int LEN = 64;
  const int nblock = (nnei + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(LEN, 3);
  force_deriv_wrt_neighbors_r<<<block_grid, thread_grid>>>(
      force, net_deriv, in_deriv, nlist, nloc, nnei);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template void prod_force_a_gpu_cuda<float>(float* force,
                                           const float* net_deriv,
                                           const float* in_deriv,
                                           const int* nlist,
                                           const int nloc,
                                           const int nall,
                                           const int nnei);
template void prod_force_a_gpu_cuda<double>(double* force,
                                            const double* net_deriv,
                                            const double* in_deriv,
                                            const int* nlist,
                                            const int nloc,
                                            const int nall,
                                            const int nnei);
template void prod_force_r_gpu_cuda<float>(float* force,
                                           const float* net_deriv,
                                           const float* in_deriv,
                                           const int* nlist,
                                           const int nloc,
                                           const int nall,
                                           const int nnei);
template void prod_force_r_gpu_cuda<double>(double* force,
                                            const double* net_deriv,
                                            const double* in_deriv,
                                            const int* nlist,
                                            const int nloc,
                                            const int nall,
                                            const int nnei);
}  // namespace deepmd


template <typename data_t>
void PdProdForceSeAOpForwardCUDAKernel(
  int nloc, int nall, int nframes, int ndescrpt, int nnei,
  data_t* p_force, const data_t* p_net_deriv, const data_t* p_in_deriv, const int* p_nlist
) {
  for(int kk = 0; kk < nframes; ++kk){
    data_t * force = p_force + kk * nall * 3;
    const data_t * net_deriv = p_net_deriv + kk * nloc * ndescrpt;
    const data_t * in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
    const int * nlist = p_nlist + kk * nloc * nnei;
    deepmd::prod_force_a_gpu_cuda(
        force,
        net_deriv, in_deriv, nlist, nloc, nall, nnei
    );
  }
}


std::vector<paddle::Tensor> PdProdForceSeAOpCUDAForward(
  const paddle::Tensor& net_deriv_tensor,
  const paddle::Tensor& in_deriv_tensor,
  const paddle::Tensor& nlist_tensor,
  const paddle::Tensor& natoms_tensor,
  int n_a_sel,
  int n_r_sel
) {
  CHECK_INPUT(net_deriv_tensor);
  CHECK_INPUT(in_deriv_tensor);
  CHECK_INPUT(nlist_tensor);
  // CHECK_INPUT(natoms_tensor);
  CHECK_INPUT_DIM(net_deriv_tensor, 2);
  CHECK_INPUT_DIM(in_deriv_tensor, 2);
  CHECK_INPUT_DIM(natoms_tensor, 1);

  PD_CHECK(natoms_tensor.shape()[0] >= 3, "number of atoms should be larger than (or equal to) 3");
  const int* natoms = natoms_tensor.data<int>();
  int nloc = natoms[0];
  int nall = natoms[1];
  int nframes = net_deriv_tensor.shape()[0];
  int ndescrpt = net_deriv_tensor.shape()[1] / nloc;
  int nnei = nlist_tensor.shape()[1] / nloc;

  PD_CHECK(nframes == in_deriv_tensor.shape()[0], "number of samples should match");
  PD_CHECK(nframes == nlist_tensor.shape()[0],"number of samples should match");
  PD_CHECK(nloc * ndescrpt * 3 == in_deriv_tensor.shape()[1], "number of descriptors should match");

  std::vector<int64_t> force_shape {nframes, 3 * nall};
  paddle::Tensor force_tensor = paddle::Tensor(paddle::PlaceType::kGPU, force_shape);

  assert (nframes == force_shape[0]);
  assert (nframes == net_deriv_tensor.shape()[0]);
  assert (nframes == in_deriv_tensor.shape()[0]);
  assert (nframes == nlist_tensor.shape()[0]);
  assert (nall * 3 == force_shape[1]);
  assert (nloc * ndescrpt == net_deriv_tensor.shape()[1]);
  assert (nloc * ndescrpt * 3 == in_deriv_tensor.shape()[1]);
  assert (nloc * nnei == nlist_tensor.shape()[1]);
  assert (nnei * 4 == ndescrpt);

  PD_DISPATCH_FLOATING_TYPES(
    net_deriv_tensor.type(), "pd_prod_force_se_a_cpu_forward_kernel", ([&] {
      PdProdForceSeAOpForwardCUDAKernel<data_t>(
          nloc, nall, nframes, ndescrpt, nnei,
          force_tensor.mutable_data<data_t>(), net_deriv_tensor.data<data_t>(),
          in_deriv_tensor.data<data_t>(), nlist_tensor.data<int>());
  }));

  return {force_tensor};
}


std::vector<paddle::Tensor> PdProdForceSeAForward(
  const paddle::Tensor& net_deriv_tensor,
  const paddle::Tensor& in_deriv_tensor,
  const paddle::Tensor& nlist_tensor,
  const paddle::Tensor& natoms_tensor,
  int n_a_sel,
  int n_r_sel
) {
    // if(net_deriv_tensor.place() == paddle::PlaceType::kCPU){
    //     return PdProdForceSeAOpCPUForward(net_deriv_tensor, in_deriv_tensor, nlist_tensor, natoms_tensor, n_a_sel, n_r_sel);
    // }else if(net_deriv_tensor.place() == paddle::PlaceType::kGPU){
    return PdProdForceSeAOpCUDAForward(net_deriv_tensor, in_deriv_tensor, nlist_tensor, natoms_tensor, n_a_sel, n_r_sel);
    // }else{
    //     PD_THROW("No Such kernel for PdFrodForceSeAForward!");
    // }
}

std::vector<std::vector<int64_t>> PdProdForceSeAInferShape(
  std::vector<int64_t> net_deriv_shape,
  std::vector<int64_t> in_deriv_shape,
  std::vector<int64_t> nlist_shape,
  std::vector<int64_t> natoms_shape,
  const int &n_a_sel,
  const int &n_r_sel
) {
  // int64_t nloc = /*natoms[0]*/ 192;
  int64_t nall = /*natoms[1]*/ 192;
  int64_t nframes = net_deriv_shape[0];
  std::vector<int64_t> force_shape = {nframes, 3 * nall};
  return {force_shape};
}

std::vector<paddle::DataType> PdProdForceSeAInferDtype(
  paddle::DataType net_deriv_dtype,
  paddle::DataType in_deriv_dtype,
  paddle::DataType nlist_dtype,
  paddle::DataType natoms_dtype
) {
  return {net_deriv_dtype};
}


PD_BUILD_OP(prod_force_se_a)
    .Inputs({"net_deriv", "in_deriv", "nlist", "natoms"})
    .Outputs({"force"})
    .Attrs({"n_a_sel: int", "n_r_sel: int"})
    .SetKernelFn(PD_KERNEL(PdProdForceSeAForward))
    .SetInferShapeFn(PD_INFER_SHAPE(PdProdForceSeAInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PdProdForceSeAInferDtype));
