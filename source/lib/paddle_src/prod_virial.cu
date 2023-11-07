#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_store.cuh>
#include "paddle/extension.h"

#include "device.h"
#include "prod_virial.h"
#include "gpu_cuda.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")
#define CHECK_INPUT_DIM(x, value) PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")
#define CHECK_INPUT_READY(x) PD_CHECK(x.IsInitialized(), #x " must be initialized before usage.")

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
  if (tid == 0) virial[bid] = data[0];
}

template <typename FPTYPE>
__global__ void virial_deriv_wrt_neighbors_a(FPTYPE* virial,
                                             FPTYPE* atom_virial,
                                             const FPTYPE* net_deriv,
                                             const FPTYPE* in_deriv,
                                             const FPTYPE* rij,
                                             const int* nlist,
                                             const int nloc,
                                             const int nnei) {
  // idx -> nloc
  // idy -> nnei
  // idz = dd0 * 3 + dd1
  // dd0 = idz / 3
  // dd1 = idz % 3
  const int_64 idx = blockIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.x + threadIdx.x;
  const unsigned int idz = threadIdx.y;
  const int ndescrpt = nnei * 4;
  if (idy >= nnei) {
    return;
  }
  int j_idx = nlist[idx * nnei + idy];
  if (j_idx < 0) {
    return;
  }
  // atomicAdd(
  //    virial + idz,
  //    net_deriv[idx * ndescrpt + idy * 4 + idw] * rij[idx * nnei * 3 + idy * 3
  //    + idz / 3] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz %
  //    3]);
  FPTYPE virial_tmp = (FPTYPE)0.;
  for (int idw = 0; idw < 4; ++idw) {
    virial_tmp += net_deriv[idx * ndescrpt + idy * 4 + idw] *
                  rij[idx * nnei * 3 + idy * 3 + idz % 3] *
                  in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz / 3];
  }
  atomicAdd(atom_virial + j_idx * 9 + idz, virial_tmp);
}

template <typename FPTYPE>
__global__ void virial_deriv_wrt_neighbors_r(FPTYPE* virial,
                                             FPTYPE* atom_virial,
                                             const FPTYPE* net_deriv,
                                             const FPTYPE* in_deriv,
                                             const FPTYPE* rij,
                                             const int* nlist,
                                             const int nloc,
                                             const int nnei) {
  // idx -> nloc
  // idy -> nnei
  // idz = dd0 * 3 + dd1
  // dd0 = idz / 3
  // dd1 = idz % 3
  const int_64 idx = blockIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.x + threadIdx.x;
  const unsigned int idz = threadIdx.y;
  const int ndescrpt = nnei * 1;

  if (idy >= nnei) {
    return;
  }
  int j_idx = nlist[idx * nnei + idy];
  if (j_idx < 0) {
    return;
  }
  // atomicAdd(
  //    virial + idz,
  //    net_deriv[idx * ndescrpt + idy * 4 + idw] * rij[idx * nnei * 3 + idy * 3
  //    + idz / 3] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz %
  //    3]);
  atomicAdd(atom_virial + j_idx * 9 + idz,
            net_deriv[idx * ndescrpt + idy] *
                rij[idx * nnei * 3 + idy * 3 + idz % 3] *
                in_deriv[idx * ndescrpt * 3 + idy * 3 + idz / 3]);
}

namespace deepmd {
template <typename FPTYPE>
void prod_virial_a_gpu_cuda(FPTYPE* virial,
                            FPTYPE* atom_virial,
                            const FPTYPE* net_deriv,
                            const FPTYPE* in_deriv,
                            const FPTYPE* rij,
                            const int* nlist,
                            const int nloc,
                            const int nall,
                            const int nnei) {
  DPErrcheck(cudaMemset(virial, 0, sizeof(FPTYPE) * 9));
  DPErrcheck(cudaMemset(atom_virial, 0, sizeof(FPTYPE) * 9 * nall));

  const int LEN = 16;
  int nblock = (nnei + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(LEN, 9);
  // compute virial of a frame
  virial_deriv_wrt_neighbors_a<<<block_grid, thread_grid>>>(
      virial, atom_virial, net_deriv, in_deriv, rij, nlist, nloc, nnei);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
  // reduction atom_virial to virial
  atom_virial_reduction<FPTYPE, TPB><<<9, TPB>>>(virial, atom_virial, nall);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template <typename FPTYPE>
void prod_virial_r_gpu_cuda(FPTYPE* virial,
                            FPTYPE* atom_virial,
                            const FPTYPE* net_deriv,
                            const FPTYPE* in_deriv,
                            const FPTYPE* rij,
                            const int* nlist,
                            const int nloc,
                            const int nall,
                            const int nnei) {
  DPErrcheck(cudaMemset(virial, 0, sizeof(FPTYPE) * 9));
  DPErrcheck(cudaMemset(atom_virial, 0, sizeof(FPTYPE) * 9 * nall));

  const int LEN = 16;
  int nblock = (nnei + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(LEN, 9);
  // compute virial of a frame
  virial_deriv_wrt_neighbors_r<<<block_grid, thread_grid>>>(
      virial, atom_virial, net_deriv, in_deriv, rij, nlist, nloc, nnei);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
  // reduction atom_virial to virial
  atom_virial_reduction<FPTYPE, TPB><<<9, TPB>>>(virial, atom_virial, nall);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}
}  // namespace deepmd

template <typename data_t>
void PdProdVirialSeAOpForwardCUDAKernel(
    int nloc, int nall, int ndescrpt, int nnei, int nframes,
    data_t* p_virial, data_t* p_atom_virial, const data_t* p_net_deriv,
    const data_t* p_in_deriv, const data_t* p_rij, const int* p_nlist){

    for(int kk = 0; kk < nframes; ++kk){
      data_t * virial = p_virial + kk * 9;
      data_t * atom_virial = p_atom_virial + kk * nall * 9;
      const data_t * net_deriv = p_net_deriv + kk * nloc * ndescrpt;
      const data_t * in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
      const data_t * rij = p_rij + kk * nloc * nnei * 3;
      const int * nlist = p_nlist + kk * nloc * nnei;
      deepmd::prod_virial_a_gpu_cuda(
          virial, atom_virial,
          net_deriv, in_deriv, rij, nlist, nloc, nall, nnei);
    }
}


std::vector<paddle::Tensor> PdProdVirialSeAOpCUDAForward(
const paddle::Tensor& net_deriv_tensor,
const paddle::Tensor& in_deriv_tensor,
const paddle::Tensor& rij_tensor,
const paddle::Tensor& nlist_tensor,
const paddle::Tensor& natoms_tensor,
int n_a_sel,
int n_r_sel
){
    CHECK_INPUT(net_deriv_tensor);
    CHECK_INPUT(in_deriv_tensor);
    CHECK_INPUT(rij_tensor);
    CHECK_INPUT(nlist_tensor);
    // CHECK_INPUT(natoms_tensor); // TODO: 暂时指定python端必须为cpu，gpu的copy_to会导致返回的指针数据不对

    CHECK_INPUT_DIM(net_deriv_tensor, 2);
    CHECK_INPUT_DIM(in_deriv_tensor, 2);
    CHECK_INPUT_DIM(rij_tensor, 2);
    CHECK_INPUT_DIM(nlist_tensor, 2);
    CHECK_INPUT_DIM(natoms_tensor, 1);

    PD_CHECK(natoms_tensor.shape()[0] >= 3, "number of atoms should be larger than (or equal to) 3");
    const int* natoms = natoms_tensor.data<int>();
    // printf("natoms_tensor.numel() = %d\n", natoms_tensor.numel());
    int nloc = natoms[0];
    // printf("nloc = %d\n", nloc);
    int nall = natoms[1];
    // printf("nall = %d\n", nall);
    int nnei = nlist_tensor.shape()[1] / nloc;
    int nframes = net_deriv_tensor.shape()[0];
    int ndescrpt = net_deriv_tensor.shape()[1] / nloc;
    PD_CHECK(nframes == in_deriv_tensor.shape()[0], "number of samples should match");
    PD_CHECK(nframes == rij_tensor.shape()[0], "number of samples should match");
    PD_CHECK(nframes == nlist_tensor.shape()[0],"number of samples should match");
    PD_CHECK(nloc * ndescrpt * 3 == in_deriv_tensor.shape()[1], "number of descriptors should match");
    PD_CHECK((nloc * nnei * 3) == rij_tensor.shape()[1], "dim of rij should be nnei * 3");

    std::vector<int64_t> virial_shape {nframes, 9};
    std::vector<int64_t> atom_virial_shape {nframes, 9 * nall};
    paddle::Tensor virial_tensor = paddle::Tensor(paddle::PlaceType::kGPU, virial_shape);
    paddle::Tensor atom_virial_tensor = paddle::Tensor(paddle::PlaceType::kGPU, atom_virial_shape);

    PD_DISPATCH_FLOATING_TYPES(
      net_deriv_tensor.type(), "pd_prod_virial_se_a_cpu_forward_kernel", ([&] {
        PdProdVirialSeAOpForwardCUDAKernel<data_t>(
            nloc, nall, ndescrpt, nnei, nframes,
            virial_tensor.mutable_data<data_t>(), atom_virial_tensor.mutable_data<data_t>(),
            net_deriv_tensor.data<data_t>(), in_deriv_tensor.data<data_t>(),
            rij_tensor.data<data_t>(), nlist_tensor.data<int>());
      }));

    return {virial_tensor, atom_virial_tensor};
}

std::vector<paddle::Tensor> PdProdVirialSeAForward(
  const paddle::Tensor& net_deriv_tensor,
  const paddle::Tensor& in_deriv_tensor,
  const paddle::Tensor& rij_tensor,
  const paddle::Tensor& nlist_tensor,
  const paddle::Tensor& natoms_tensor,
  int n_a_sel,
  int n_r_sel
) {
  if (net_deriv_tensor.is_gpu()) {
    // std::cout << natoms_tensor.dtype() << std::endl;
    // const int* natoms = natoms_tensor.data<int>();
    // printf("%d\n", natoms[0]);
    return PdProdVirialSeAOpCUDAForward(
      net_deriv_tensor,
      in_deriv_tensor,
      rij_tensor,
      nlist_tensor,
      natoms_tensor,
      n_a_sel,
      n_r_sel
    );
  } else {
    PD_THROW("Unsupported device type for forward function of custom relu operator.");
  }
}


/*以下是反向代码*/

// template <typename data_t>
// void PdProdForceSeAOpCPUBackwardKernel(
//     int nloc, int nframes, int ndescrpt, int nnei,
//     const data_t* grad, const data_t* net_deriv,
//     const data_t* in_deriv, const data_t* rij, const int* nlist,
//     data_t* grad_net){

// #pragma omp parallel for
//     for (int kk = 0; kk < nframes; ++kk){

//       int grad_iter	= kk * 9;
//       int in_iter	= kk * nloc * ndescrpt * 3;
//       int rij_iter	= kk * nloc * nnei * 3;
//       int nlist_iter	= kk * nloc * nnei;
//       int grad_net_iter	= kk * nloc * ndescrpt;
//       // int n_a_shift = n_a_sel * 4;

//       deepmd::prod_virial_grad_a_cpu(
//         &grad_net[grad_net_iter],
//         &grad[grad_iter],
//         &in_deriv[in_iter],
//         &rij[rij_iter],
//         &nlist[nlist_iter],
//         nloc,
//         nnei);
//     }
// }


// template <typename data_t>
// void PdProdForceSeAOpGPUBackwardKernel(
//   int nloc, int nframes, int ndescrpt, int nnei,
//   const data_t* virial_grad, const data_t* net_deriv,
//   const data_t* in_deriv, const data_t* rij, const int* nlist,
//   data_t* grad_net
// ) {
//     for (int_64 kk = 0; kk < nframes; ++kk) {
//       FPTYPE* grad_net = p_grad_net + kk * nloc * ndescrpt;
//       const FPTYPE* virial_grad = p_grad + kk * 9;
//       const FPTYPE* in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
//       const FPTYPE* rij = p_rij + kk * nloc * nnei * 3;
//       const int* nlist = p_nlist + kk * nloc * nnei;
//       if (device == "GPU") {
//         deepmd::prod_virial_grad_a_gpu_cuda(
//           grad_net, virial_grad, in_deriv, rij, nlist, nloc, nnei
//         );
//       }
//     }
// }


// std::vector<paddle::Tensor> PdProdVirialSeAOpCPUBackward(
// const paddle::Tensor& virial_grad_tensor,
// const paddle::Tensor& net_deriv_tensor,
// const paddle::Tensor& in_deriv_tensor,
// const paddle::Tensor& rij_tensor,
// const paddle::Tensor& nlist_tensor,
// const paddle::Tensor& natoms_tensor,
// int n_a_sel,
// int n_r_sel
// ){
//     // CHECK_INPUT_READY(virial_grad_tensor);
//     // CHECK_INPUT_READY(net_deriv_tensor);
//     // CHECK_INPUT_READY(in_deriv_tensor);
//     // CHECK_INPUT_READY(rij_tensor);
//     // CHECK_INPUT_READY(nlist_tensor);
//     // CHECK_INPUT_READY(natoms_tensor);

//     auto grad_shape = virial_grad_tensor.shape();
//     auto net_deriv_shape = net_deriv_tensor.shape();
//     auto in_deriv_shape = in_deriv_tensor.shape();
//     auto rij_shape = rij_tensor.shape();
//     auto nlist_shape = nlist_tensor.shape();
//     auto natoms_shape = natoms_tensor.shape();

//     CHECK_INPUT_DIM(virial_grad_tensor, 2);
//     CHECK_INPUT_DIM(net_deriv_tensor, 2);
//     CHECK_INPUT_DIM(in_deriv_tensor, 2);
//     CHECK_INPUT_DIM(rij_tensor, 2);
//     CHECK_INPUT_DIM(nlist_tensor, 2);
//     CHECK_INPUT_DIM(natoms_tensor, 1);

//     PD_CHECK(natoms_shape[0] >= 3, "number of atoms should be larger than (or equal to) 3");

//     const int* natoms = nullptr;
//     // if(natoms_tensor.place() != paddle::PlaceType::kCPU){
//     //     natoms = natoms_tensor.copy_to<int>(paddle::PlaceType::kCPU).data<int>();
//     // }else{
//     natoms = natoms_tensor.data<int>();
//     // }
//     int nframes = net_deriv_shape[0];
//     int nloc = natoms[0];
//     int ndescrpt = net_deriv_shape[1] / nloc;
//     int nnei = nlist_shape[1] / nloc;

//     PD_CHECK(nframes == grad_shape[0], "number of frames should match");
//     PD_CHECK(nframes == in_deriv_shape[0], "number of samples should match");
//     PD_CHECK(nframes == rij_shape[0], "number of frames should match");
//     PD_CHECK(nframes == nlist_shape[0],"number of samples should match");
//     PD_CHECK(9 == grad_shape[1], "input grad shape should be 3 x natoms");
//     PD_CHECK(nloc * ndescrpt * 3 == in_deriv_shape[1], "number of descriptors should match");
//     PD_CHECK(nloc * nnei * 3 == rij_shape[1], "dim of rij should be  nnei * 3");
//     PD_CHECK(nnei == (n_a_sel + n_r_sel), "number of neighbors should match");

//     std::vector<int64_t> grad_net_shape {nframes, nloc * ndescrpt};
//     // paddle::Tensor grad_net_tensor = paddle::Tensor(paddle::PlaceType::kCPU, grad_net_shape);
//     paddle::Tensor grad_net_tensor = paddle::empty(
//       grad_net_shape,
//       virial_grad_tensor.dtype(),
//       virial_grad_tensor.place()
//     );

//     if(virial_grad_tensor.place() == paddle::PlaceType::kCPU){
//         PD_DISPATCH_FLOATING_TYPES(
//         virial_grad_tensor.type(), "pd_prod_force_se_a_cpu_backward_kernel", ([&] {
//             PdProdForceSeAOpCPUBackwardKernel<data_t>(
//                 nloc, nframes, ndescrpt, nnei,
//                 virial_grad_tensor.data<data_t>(),
//                 net_deriv_tensor.data<data_t>(),
//                 in_deriv_tensor.data<data_t>(),
//                 rij_tensor.data<data_t>(), nlist_tensor.data<int>(),
//                 grad_net_tensor.mutable_data<data_t>());
//         }));
//     }else{
//         PD_DISPATCH_FLOATING_TYPES(
//         virial_grad_tensor.type(), "pd_prod_force_se_a_cpu_backward_kernel", ([&] {
//             PdProdForceSeAOpGPUBackwardKernel<data_t>(
//                 nloc, nframes, ndescrpt, nnei,
//                 virial_grad_tensor.data<data_t>(),
//                 net_deriv_tensor.data<data_t>(),
//                 in_deriv_tensor.data<data_t>(),
//                 rij_tensor.data<data_t>(),
//                 nlist_tensor.data<int>(),
//                 grad_net_tensor.mutable_data<data_t>());
//         }));
//     }
//     return {grad_net_tensor};
// }


// std::vector<paddle::Tensor> PdProdVirialSeABackward(
//   const paddle::Tensor& virial_grad_tensor,
//   const paddle::Tensor& net_deriv_tensor,
//   const paddle::Tensor& in_deriv_tensor,
//   const paddle::Tensor& rij_tensor,
//   const paddle::Tensor& nlist_tensor,
//   const paddle::Tensor& natoms_tensor,
//   int n_a_sel,
//   int n_r_sel
// ) {
//     return PdProdVirialSeAOpCPUBackward(
//         virial_grad_tensor, net_deriv_tensor, in_deriv_tensor,
//         rij_tensor,
//         nlist_tensor, natoms_tensor, n_a_sel, n_r_sel
//     );
// }

// std::vector<std::vector<int64_t>> PdProdVirialSeAInferShape(
//   // std::vector<int64_t> x_shape
//   std::vector<int64_t> net_deriv_shape,
//   std::vector<int64_t> in_deriv_shape,
//   std::vector<int64_t> rij_shape,
//   std::vector<int64_t> nlist_shape,
//   std::vector<int64_t> natoms_shape,
//   int n_a_sel,
//   int n_r_sel
// ) {
//   auto virial_shape {nframes, 9};
//   auto atom_virial_shape {nframes, 9 * nall};
//   return {virial_shape, atom_virial_shape};
// }

// std::vector<paddle::DataType> PdProdVirialSeAInferDtype(paddle::DataType x_dtype, ...) {
//   return {x_dtype, ...};
// }

std::vector<std::vector<int64_t>> PdProdVirialSeAInferShape(
  std::vector<int64_t> net_deriv_shape,
  std::vector<int64_t> in_deriv_shape,
  std::vector<int64_t> rij_shape,
  std::vector<int64_t> nlist_shape,
  std::vector<int64_t> natoms_shape,
  const int &n_a_sel,
  const int &n_r_sel
) {
  // int64_t nloc = /*natoms[0]*/ 192;
  int64_t nall = /*natoms[1]*/ 192;
  int64_t nframes = net_deriv_shape[0];

  std::vector<int64_t> virial_shape = {nframes, 9};
  std::vector<int64_t> atom_virial_shape = {nframes, 9 * nall};

  return {virial_shape, atom_virial_shape};
}

std::vector<paddle::DataType> PdProdVirialSeAInferDtype(
  paddle::DataType net_deriv_dtype,
  paddle::DataType in_deriv_dtype,
  paddle::DataType rij_dtype,
  paddle::DataType nlist_dtype,
  paddle::DataType natoms_dtype
) {
  return {net_deriv_dtype, net_deriv_dtype};
}


PD_BUILD_OP(prod_virial_se_a)
    .Inputs({"net_deriv", "in_deriv", "rij", "nlist", "natoms"})
    .Outputs({"virial", "atom_virial"})
    .Attrs({"n_a_sel: int", "n_r_sel: int"})
    .SetKernelFn(PD_KERNEL(PdProdVirialSeAForward))
    .SetInferShapeFn(PD_INFER_SHAPE(PdProdVirialSeAInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PdProdVirialSeAInferDtype));


// PD_BUILD_GRAD_OP(prod_virial_se_a)
//     .Inputs({paddle::Grad("virial"), "net_deriv", "in_deriv", "rij", "nlist", "natoms"})
//     .Outputs({paddle::Grad("net_deriv")})
//     .Attrs({"n_a_sel: int", "n_r_sel: int"})
//     .SetKernelFn(PD_KERNEL(PdProdVirialSeABackward));


