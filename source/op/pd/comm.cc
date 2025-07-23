// SPDX-License-Identifier: LGPL-3.0-or-later

#ifdef USE_MPI
#include <mpi.h>
#ifdef OMPI_MPI_H
#include <mpi-ext.h>
#endif
#endif
// #include <paddle/include/paddle_inference_api.h>
#include <cstdint>

#include "paddle/extension.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "device.h"
#endif

#ifdef USE_MPI
template <typename T>
static MPI_Datatype get_mpi_type();

template <>
MPI_Datatype get_mpi_type<float>() {
  return MPI_FLOAT;
}

template <>
MPI_Datatype get_mpi_type<double>() {
  return MPI_DOUBLE;
}
#endif

#ifdef USE_MPI
static void unpack_communicator(const paddle::Tensor& communicator_tensor,
                                MPI_Comm& mpi_comm) {
#ifdef OMPI_MPI_H
  const int64_t* communicator = communicator_tensor.data<int64_t>();
#else
  const int64_t* ptr = communicator_tensor.data<int64_t>();
  const int* communicator = reinterpret_cast<const int*>(ptr);
#endif
  mpi_comm = reinterpret_cast<MPI_Comm>(*communicator);
}
#endif

template <typename FPTYPE>
void Border_forward_t(const paddle::Tensor& sendlist_tensor,
                      const paddle::Tensor& sendproc_tensor,
                      const paddle::Tensor& recvproc_tensor,
                      const paddle::Tensor& sendnum_tensor,
                      const paddle::Tensor& recvnum_tensor,
                      paddle::Tensor& g1,
                      const paddle::Tensor& communicator_tensor,
                      const paddle::Tensor& nlocal_tensor,
                      const paddle::Tensor& nghost_tensor) {
  int** sendlist = reinterpret_cast<int**>((sendlist_tensor + 0).data<int>());
  const int* sendproc = sendproc_tensor.data<int>();
  const int* recvproc = recvproc_tensor.data<int>();
  const int* sendnum = sendnum_tensor.data<int>();
  const int* recvnum = recvnum_tensor.data<int>();
  int tensor_size = g1.dims()[1];
  int nswap = sendproc_tensor.dims()[0];

  int nlocal = *nlocal_tensor.data<int>();
  int nghost = *nghost_tensor.data<int>();
  int ntotal = nlocal + nghost;
  paddle::Tensor recv_g1_tensor = g1;

#ifdef USE_MPI
  int mpi_init = 0;
  MPI_Initialized(&mpi_init);
  int cuda_aware = 1;
  int me = 0;
  MPI_Comm world;
  int world_size = 0;
  if (mpi_init) {
    unpack_communicator(communicator_tensor, world);
    MPI_Comm_rank(world, &me);
    MPI_Comm_size(world, &world_size);
  }
  MPI_Datatype mpi_type = get_mpi_type<FPTYPE>();
  MPI_Request request;
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  if (world_size >= 1) {
    int version, subversion;
    MPI_Get_version(&version, &subversion);
    if (version >= 4) {
#ifdef NO_CUDA_AWARE
      cuda_aware = 0;
#else
      cuda_aware = MPIX_Query_cuda_support();
#endif
    } else {
      cuda_aware = 0;
    }
    if (cuda_aware == 0) {
      recv_g1_tensor = paddle::empty_like(g1, g1.dtype(), paddle::CPUPlace());
      std::cout << "[1]" << std::endl;
      recv_g1_tensor = g1.copy_to(recv_g1_tensor.place(), true);
      // recv_g1_tensor.copy_(g1);
    }
  }
#endif
#endif
  FPTYPE* recv_g1 = recv_g1_tensor.data<FPTYPE>() + nlocal * tensor_size;
  // auto int32_options = torch::TensorOptions().dtype(torch::kInt32);
  for (int iswap = 0; iswap < nswap; ++iswap) {
    int nrecv = recvnum[iswap];
    int nsend = sendnum[iswap];
    paddle::Tensor isendlist;
    paddle::Tensor send_g1_tensor;
    FPTYPE* send_g1;
    if (nsend != 0) {
      std::cout << "[2]" << std::endl;
      isendlist = paddle::from_blob(
          static_cast<void*>(sendlist[iswap]), {nsend}, paddle::DataType::INT32,
          phi::DataLayout::NCHW, recv_g1_tensor.place());
      send_g1_tensor = paddle::gather(recv_g1_tensor, isendlist, 0);
      send_g1 = send_g1_tensor.data<FPTYPE>();
    }
#ifdef USE_MPI
    if (sendproc[iswap] != me) {
      if (nrecv) {
        MPI_Irecv(recv_g1, nrecv * tensor_size, mpi_type, recvproc[iswap], 0,
                  world, &request);
      }
      if (nsend) {
        MPI_Send(send_g1, nsend * tensor_size, mpi_type, sendproc[iswap], 0,
                 world);
      }
      if (nrecv) {
        MPI_Wait(&request, MPI_STATUS_IGNORE);
      }
    } else {
#endif
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#ifdef USE_MPI
      if (cuda_aware == 0) {
        memcpy(recv_g1, send_g1,
               (unsigned long)nsend * tensor_size * sizeof(FPTYPE));
      } else {
        gpuMemcpy(recv_g1, send_g1,
                  (unsigned long)nsend * tensor_size * sizeof(FPTYPE),
                  gpuMemcpyDeviceToDevice);
      }
#else
      gpuMemcpy(recv_g1, send_g1,
                (unsigned long)nsend * tensor_size * sizeof(FPTYPE),
                gpuMemcpyDeviceToDevice);
#endif
#else
    memcpy(recv_g1, send_g1,
           (unsigned long)nsend * tensor_size * sizeof(FPTYPE));
#endif
#ifdef USE_MPI
    }
#endif
    recv_g1 += nrecv * tensor_size;
  }
#ifdef USE_MPI
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  if (cuda_aware == 0) {
    g1 = recv_g1_tensor;
  }
#endif
#endif
  // return {g1};
}

void Border_forward(const paddle::Tensor& sendlist_tensor,
                    const paddle::Tensor& sendproc_tensor,
                    const paddle::Tensor& recvproc_tensor,
                    const paddle::Tensor& sendnum_tensor,
                    const paddle::Tensor& recvnum_tensor,
                    paddle::Tensor& g1,
                    const paddle::Tensor& communicator_tensor,
                    const paddle::Tensor& nlocal_tensor,
                    const paddle::Tensor& nghost_tensor) {
  bool type_flag = (g1.dtype() == phi::DataType::FLOAT64) ? true : false;
  if (type_flag) {
    Border_forward_t<double>(sendlist_tensor, sendproc_tensor, recvproc_tensor,
                             sendnum_tensor, recvnum_tensor, g1,
                             communicator_tensor, nlocal_tensor, nghost_tensor);
  } else {
    Border_forward_t<float>(sendlist_tensor, sendproc_tensor, recvproc_tensor,
                            sendnum_tensor, recvnum_tensor, g1,
                            communicator_tensor, nlocal_tensor, nghost_tensor);
  }
}

template <typename FPTYPE>
std::vector<paddle::Tensor> Border_backward_t(
    const paddle::Tensor& sendlist_tensor,
    const paddle::Tensor& sendproc_tensor,
    const paddle::Tensor& recvproc_tensor,
    const paddle::Tensor& sendnum_tensor,
    const paddle::Tensor& recvnum_tensor,
    const paddle::Tensor& communicator_tensor,
    const paddle::Tensor& nlocal_tensor,
    const paddle::Tensor& nghost_tensor,
    paddle::Tensor& recv_g1_tensor_grad  // grad_output[0]
) {
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  gpuDeviceSynchronize();
#endif
  paddle::Tensor d_local_g1_tensor = (recv_g1_tensor_grad + 0).contiguous();
#ifdef USE_MPI
  int mpi_init = 0;
  MPI_Initialized(&mpi_init);
  int world_size = 0;
  int cuda_aware = 1;
  int me = 0;
  MPI_Comm world;
  if (mpi_init) {
    unpack_communicator(communicator_tensor, world);
    MPI_Comm_rank(world, &me);
    MPI_Comm_size(world, &world_size);
  }
  MPI_Datatype mpi_type = get_mpi_type<FPTYPE>();
  MPI_Request request;
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  if (world_size >= 1) {
    int version, subversion;
    MPI_Get_version(&version, &subversion);
    if (version >= 4) {
#ifdef NO_CUDA_AWARE
      cuda_aware = 0;
#else
      cuda_aware = MPIX_Query_cuda_support();
#endif
    } else {
      cuda_aware = 0;
    }
    if (cuda_aware == 0) {
      d_local_g1_tensor = paddle::empty_like(
          recv_g1_tensor_grad, recv_g1_tensor_grad.dtype(), paddle::CPUPlace());
      std::cout << "[3]" << std::endl;
      d_local_g1_tensor =
          recv_g1_tensor_grad.copy_to(d_local_g1_tensor.place(), true);
    }
  }
#endif
#endif
  int** recvlist = reinterpret_cast<int**>((sendlist_tensor + 0).data<int>());
  // swap send and recv here
  const int* recvproc = sendproc_tensor.data<int>();
  const int* sendproc = recvproc_tensor.data<int>();
  const int* recvnum = sendnum_tensor.data<int>();
  const int* sendnum = recvnum_tensor.data<int>();

  FPTYPE* local_g1 = d_local_g1_tensor.data<FPTYPE>();
  int tensor_size = d_local_g1_tensor.dims()[1];
  int nswap = sendproc_tensor.dims()[0];

  int nlocal = *nlocal_tensor.data<int>();
  int nghost = *nghost_tensor.data<int>();
  int ntotal = nlocal + nghost;
  paddle::Tensor send_g1_tensor;
  paddle::Tensor recv_g1_tensor;
  FPTYPE* recv_g1;
  FPTYPE* send_g1;
  if (nswap != 0) {
    send_g1_tensor = d_local_g1_tensor;
    int max_recvnum = *sendnum_tensor.max().data<int>();
    std::cout << "[4]" << std::endl;
    recv_g1_tensor =
        paddle::empty({max_recvnum, tensor_size}, d_local_g1_tensor.dtype(),
                      d_local_g1_tensor.place());
    recv_g1 = recv_g1_tensor.data<FPTYPE>();
    send_g1 = send_g1_tensor.data<FPTYPE>() + ntotal * tensor_size;
  }

  int end = ntotal;
  // auto int32_options = torch::TensorOptions().dtype(torch::kInt32);
  for (int iswap = nswap - 1; iswap >= 0; --iswap) {
    int nrecv = recvnum[iswap];
    int nsend = sendnum[iswap];

    paddle::Tensor irecvlist;
    if (nrecv) {
      std::cout << "[5]" << std::endl;
      irecvlist = paddle::from_blob(
          static_cast<void*>(recvlist[iswap]), {nrecv}, paddle::DataType::INT32,
          paddle::DataLayout::NCHW, d_local_g1_tensor.place());
    }
    if (nsend) {
      send_g1 -= nsend * tensor_size;
    }
#ifdef USE_MPI
    if (sendproc[iswap] != me) {
      if (nrecv) {
        MPI_Irecv(recv_g1, nrecv * tensor_size, mpi_type, recvproc[iswap], 0,
                  world, &request);
      }
      if (nsend) {
        MPI_Send(send_g1, nsend * tensor_size, mpi_type, sendproc[iswap], 0,
                 world);
      }
      if (nrecv) {
        MPI_Wait(&request, MPI_STATUS_IGNORE);
      }
    } else {
#endif
      if (nrecv) {
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#ifdef USE_MPI
        if (cuda_aware == 0) {
          memcpy(recv_g1, send_g1,
                 (unsigned long)nrecv * tensor_size * sizeof(FPTYPE));
        } else {
          gpuMemcpy(recv_g1, send_g1,
                    (unsigned long)nrecv * tensor_size * sizeof(FPTYPE),
                    gpuMemcpyDeviceToDevice);
        }
#else
        gpuMemcpy(recv_g1, send_g1,
                  (unsigned long)nrecv * tensor_size * sizeof(FPTYPE),
                  gpuMemcpyDeviceToDevice);
#endif
#else
      memcpy(recv_g1, send_g1,
             (unsigned long)nrecv * tensor_size * sizeof(FPTYPE));
#endif
      }
#ifdef USE_MPI
    }
#endif
    if (nrecv) {
      d_local_g1_tensor = paddle::experimental::index_add_(
          d_local_g1_tensor, irecvlist, recv_g1_tensor.slice(0, nrecv), 0);
    }
  }
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  gpuDeviceSynchronize();
#endif
#ifdef USE_MPI
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  if (cuda_aware == 0) {
    recv_g1_tensor_grad = d_local_g1_tensor;
    // d_local_g1_tensor.copy_to(recv_g1_tensor_grad.place(), true);
    // grad_output[0].copy_(d_local_g1_tensor);
  }
#endif
#endif
  return {paddle::Tensor(), paddle::Tensor(), paddle::Tensor(),
          paddle::Tensor(), paddle::Tensor(), recv_g1_tensor_grad,
          paddle::Tensor(), paddle::Tensor(), paddle::Tensor()};
}

std::vector<paddle::Tensor> Border_backward(
    const paddle::Tensor& sendlist_tensor,
    const paddle::Tensor& sendproc_tensor,
    const paddle::Tensor& recvproc_tensor,
    const paddle::Tensor& sendnum_tensor,
    const paddle::Tensor& recvnum_tensor,
    const paddle::Tensor& communicator_tensor,
    const paddle::Tensor& nlocal_tensor,
    const paddle::Tensor& nghost_tensor,
    paddle::Tensor& recv_g1_tensor_grad) {
  bool type_flag =
      (sendlist_tensor.dtype() == paddle::DataType::FLOAT64) ? true : false;
  if (type_flag) {
    return Border_backward_t<double>(
        sendlist_tensor, sendproc_tensor, recvproc_tensor, sendnum_tensor,
        recvnum_tensor, communicator_tensor, nlocal_tensor, nghost_tensor,
        recv_g1_tensor_grad);
  } else {
    return Border_backward_t<float>(
        sendlist_tensor, sendproc_tensor, recvproc_tensor, sendnum_tensor,
        recvnum_tensor, communicator_tensor, nlocal_tensor, nghost_tensor,
        recv_g1_tensor_grad);
  }
}

std::vector<std::vector<int64_t>> Border_forwardInferShape(
    std::vector<int64_t> sendlist_tensor_shape,
    std::vector<int64_t> sendproc_tensor_shape,
    std::vector<int64_t> recvproc_tensor_shape,
    std::vector<int64_t> sendnum_tensor_shape,
    std::vector<int64_t> recvnum_tensor_shape,
    std::vector<int64_t> g1_tensor_shape,
    std::vector<int64_t> communicator_tensor_shape,
    std::vector<int64_t> nlocal_tensor_shape,
    std::vector<int64_t> nghost_tenso_shape) {
  return {g1_tensor_shape};
}

std::vector<paddle::DataType> Border_forwardInferDtype(
    paddle::DataType sendlist_tensor_dtype,
    paddle::DataType sendproc_tensor_dtype,
    paddle::DataType recvproc_tensor_dtype,
    paddle::DataType sendnum_tensor_dtype,
    paddle::DataType recvnum_tensor_dtype,
    paddle::DataType g1_tensor_dtype,
    paddle::DataType communicator_tensor_dtype,
    paddle::DataType nlocal_tensor_dtype,
    paddle::DataType nghost_tenso_dtype) {
  return {g1_tensor_dtype};
}

std::vector<std::vector<int64_t>> Border_backwardInferShape(
    std::vector<int64_t> sendlist_shape,
    std::vector<int64_t> sendproc_shape,
    std::vector<int64_t> recvproc_shape,
    std::vector<int64_t> sendnum_shape,
    std::vector<int64_t> recvnum_shape,
    std::vector<int64_t> communicator_shape,
    std::vector<int64_t> nlocal_shape,
    std::vector<int64_t> nghost_shape,
    std::vector<int64_t> recv_g1_grad_shape) {
  return {recv_g1_grad_shape};
}

std::vector<paddle::DataType> Border_backwardInferDtype(
    paddle::DataType sendlist_dtype,
    paddle::DataType sendproc_dtype,
    paddle::DataType recvproc_dtype,
    paddle::DataType sendnum_dtype,
    paddle::DataType recvnum_dtype,
    paddle::DataType communicator_dtype,
    paddle::DataType nlocal_dtype,
    paddle::DataType nghost_dtype,
    paddle::DataType recv_g1_tens_dtype) {
  return {recv_g1_tens_dtype};
}

/**
 * @brief communicate the latest g1 info to other lmp proc
 * @param[out] recv_g1_tensor g1 after communication
 * @param[in]  sendlist_tensor list of atoms to send in each swap
 * @param[in]  sendproc_tensor proc to send to at each swap
 * @param[in]  recvproc_tensor proc to recv from at each swap
 * @param[in]  sendnum_tensor # of atoms to send in each swap
 * @param[in]  recvnum_tensor # of atoms to recv in each swap
 * @param[in]  g1_tensor tensor to store g1 info
 * @param[in]  communicator_tensor MPI_comm data in lmp
 * @param[in]  nlocal_tensor # of local atoms
 * @param[in]  nghost_tensor # of nghost atoms
 **/
PD_BUILD_OP(border_op)
    .Inputs({"sendlist_tensor", "sendproc_tensor", "recvproc_tensor",
             "sendnum_tensor", "recvnum_tensor", "g1_tensor",
             "communicator_tensor", "nlocal_tensor", "nghost_tensor"})
    .Outputs({"recv_g1_tensor"})
    .SetKernelFn(PD_KERNEL(Border_forward))
    .SetInplaceMap({{"g1_tensor", "recv_g1_tensor"}})
    .SetInferShapeFn(PD_INFER_SHAPE(Border_forwardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(Border_forwardInferDtype));

PD_BUILD_GRAD_OP(border_op)
    .Inputs({"sendlist_tensor", "sendproc_tensor", "recvproc_tensor",
             "sendnum_tensor", "recvnum_tensor", "g1_tensor",
             "communicator_tensor", "nlocal_tensor", "nghost_tensor",
             paddle::Grad("recv_g1_tensor")})
    .Outputs({paddle::Grad("g1_tensor")})
    .SetInplaceMap({{paddle::Grad("recv_g1_tensor"),
                     paddle::Grad("g1_tensor")}})
    .SetKernelFn(PD_KERNEL(Border_backward))
    .SetInferShapeFn(PD_INFER_SHAPE(Border_backwardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(Border_backwardInferDtype));
