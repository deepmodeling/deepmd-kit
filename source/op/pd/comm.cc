// SPDX-License-Identifier: LGPL-3.0-or-later

#ifdef USE_MPI
#include <mpi.h>
#ifdef OMPI_MPI_H
#include <mpi-ext.h>
#endif
#endif
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
  int64_t send_list_len = sendlist_tensor.numel();

  paddle::Tensor cpu_sendlist = paddle::empty(
      {send_list_len}, paddle::DataType::INT64, paddle::CPUPlace());
  cpu_sendlist.copy_(sendlist_tensor, paddle::CPUPlace(), true);
  int64_t* sendlist = cpu_sendlist.data<int64_t>();

  int nswap = sendproc_tensor.dims()[0];

  paddle::Tensor cpu_sendproc =
      paddle::empty({nswap}, paddle::DataType::INT32, paddle::CPUPlace());
  cpu_sendproc.copy_(sendproc_tensor, paddle::CPUPlace(), true);
  int* sendproc = cpu_sendproc.data<int>();

  paddle::Tensor cpu_recvproc =
      paddle::empty({nswap}, paddle::DataType::INT32, paddle::CPUPlace());
  cpu_recvproc.copy_(recvproc_tensor, paddle::CPUPlace(), true);
  int* recvproc = cpu_recvproc.data<int>();

  paddle::Tensor cpu_sendnum =
      paddle::empty({nswap}, paddle::DataType::INT32, paddle::CPUPlace());
  cpu_sendnum.copy_(sendnum_tensor, paddle::CPUPlace(), true);
  int* sendnum = cpu_sendnum.data<int>();

  paddle::Tensor cpu_recvnum =
      paddle::empty({nswap}, paddle::DataType::INT32, paddle::CPUPlace());
  cpu_recvnum.copy_(recvnum_tensor, paddle::CPUPlace(), true);
  int* recvnum = cpu_recvnum.data<int>();

  int tensor_size = g1.dims()[1];

  paddle::Tensor cpu_nlocal =
      paddle::empty({nswap}, paddle::DataType::INT32, paddle::CPUPlace());
  cpu_nlocal.copy_(nlocal_tensor, paddle::CPUPlace(), true);
  int nlocal = *(cpu_nlocal.data<int>());

  paddle::Tensor cpu_nghost =
      paddle::empty({nswap}, paddle::DataType::INT32, paddle::CPUPlace());
  cpu_nghost.copy_(nghost_tensor, paddle::CPUPlace(), true);
  int nghost = *(cpu_nghost.data<int>());

  int ntotal = nlocal + nghost;

  paddle::Tensor recv_g1_tensor = g1;

#ifdef USE_MPI
  // MPI initialization check
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
      recv_g1_tensor.copy_(g1, recv_g1_tensor.place(), true);
    }
  }
#endif

#endif  // USE_MPI
  FPTYPE* recv_g1 = recv_g1_tensor.data<FPTYPE>() + nlocal * tensor_size;

  for (int iswap = 0; iswap < nswap; ++iswap) {
    int nrecv = recvnum[iswap];
    int nsend = sendnum[iswap];
    paddle::Tensor isendlist;
    paddle::Tensor send_g1_tensor;
    FPTYPE* send_g1 = nullptr;

    if (nsend != 0) {
      std::intptr_t addr = static_cast<std::intptr_t>(sendlist[iswap]);
      int* isendlist_ptr = reinterpret_cast<int*>(addr);
      isendlist =
          paddle::from_blob(isendlist_ptr, {nsend}, paddle::DataType::INT32,
                            phi::DataLayout::NCHW, paddle::CPUPlace())
              .copy_to(recv_g1_tensor.place(), true);
      send_g1_tensor =
          paddle::experimental::index_select(recv_g1_tensor, isendlist, 0);
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
    g1.copy_(recv_g1_tensor, g1.place(), true);
  }
#endif
#endif
}

void Border_forward(const paddle::Tensor& sendlist_tensor,
                    const paddle::Tensor& sendproc_tensor,
                    const paddle::Tensor& recvproc_tensor,
                    const paddle::Tensor& sendnum_tensor,
                    const paddle::Tensor& recvnum_tensor,
                    paddle::Tensor& g1_tensor,
                    const paddle::Tensor& communicator_tensor,
                    const paddle::Tensor& nlocal_tensor,
                    const paddle::Tensor& nghost_tensor) {
  bool type_flag = (g1_tensor.dtype() == phi::DataType::FLOAT64) ? true : false;
  if (type_flag) {
    Border_forward_t<double>(sendlist_tensor, sendproc_tensor, recvproc_tensor,
                             sendnum_tensor, recvnum_tensor, g1_tensor,
                             communicator_tensor, nlocal_tensor, nghost_tensor);
  } else {
    Border_forward_t<float>(sendlist_tensor, sendproc_tensor, recvproc_tensor,
                            sendnum_tensor, recvnum_tensor, g1_tensor,
                            communicator_tensor, nlocal_tensor, nghost_tensor);
  }
}

template <typename FPTYPE>
void Border_backward_t(const paddle::Tensor& sendlist_tensor,
                       const paddle::Tensor& sendproc_tensor,
                       const paddle::Tensor& recvproc_tensor,
                       const paddle::Tensor& sendnum_tensor,
                       const paddle::Tensor& recvnum_tensor,
                       const paddle::Tensor& g1_tensor,
                       const paddle::Tensor& communicator_tensor,
                       const paddle::Tensor& nlocal_tensor,
                       const paddle::Tensor& nghost_tensor,
                       paddle::Tensor& recv_g1_tensor_grad) {
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  gpuDeviceSynchronize();
#endif
  paddle::Tensor d_local_g1_tensor =
      paddle::empty(recv_g1_tensor_grad.shape(), recv_g1_tensor_grad.dtype(),
                    recv_g1_tensor_grad.place());
  d_local_g1_tensor.copy_(recv_g1_tensor_grad.contiguous(),
                          d_local_g1_tensor.place(), true);

#ifdef USE_MPI
  int mpi_init = 0, world_size = 0, me = 0, cuda_aware = 1;
  MPI_Initialized(&mpi_init);

  MPI_Comm world;
  if (mpi_init) {
    unpack_communicator(communicator_tensor, world);
    MPI_Comm_rank(world, &me);
    MPI_Comm_size(world, &world_size);
  }

  auto mpi_type = get_mpi_type<FPTYPE>();
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
      d_local_g1_tensor.copy_(recv_g1_tensor_grad, d_local_g1_tensor.place(),
                              true);
    }
  }
#endif
#endif  // USE_MPI
  int64_t send_list_len = sendlist_tensor.numel();
  paddle::Tensor cpu_sendlist = paddle::empty(
      {send_list_len}, paddle::DataType::INT64, paddle::CPUPlace());
  cpu_sendlist.copy_(sendlist_tensor, paddle::CPUPlace(), true);
  int64_t* recvlist = cpu_sendlist.data<int64_t>();

  int nswap = sendproc_tensor.dims()[0];
  // swap send and recv here
  paddle::Tensor cpu_recvproc =
      paddle::empty({nswap}, paddle::DataType::INT32, paddle::CPUPlace());
  cpu_recvproc.copy_(recvproc_tensor, paddle::CPUPlace(), true);
  int* recvproc = cpu_recvproc.data<int>();

  paddle::Tensor cpu_sendproc =
      paddle::empty({nswap}, paddle::DataType::INT32, paddle::CPUPlace());
  cpu_sendproc.copy_(sendproc_tensor, paddle::CPUPlace(), true);
  int* sendproc = cpu_sendproc.data<int>();

  paddle::Tensor cpu_sendnum =
      paddle::empty({nswap}, paddle::DataType::INT32, paddle::CPUPlace());
  cpu_sendnum.copy_(sendnum_tensor, paddle::CPUPlace(), true);
  int* recvnum = cpu_sendnum.data<int>();

  paddle::Tensor cpu_recvnum =
      paddle::empty({nswap}, paddle::DataType::INT32, paddle::CPUPlace());
  cpu_recvnum.copy_(recvnum_tensor, paddle::CPUPlace(), true);
  int* sendnum = cpu_recvnum.data<int>();

  FPTYPE* local_g1 = d_local_g1_tensor.data<FPTYPE>();
  int tensor_size = d_local_g1_tensor.dims()[1];

  paddle::Tensor cpu_nlocal =
      paddle::empty({1}, paddle::DataType::INT32, paddle::CPUPlace());
  cpu_nlocal.copy_(nlocal_tensor, paddle::CPUPlace(), true);
  int nlocal = *cpu_nlocal.data<int>();

  paddle::Tensor cpu_nghost =
      paddle::empty({1}, paddle::DataType::INT32, paddle::CPUPlace());
  cpu_nghost.copy_(nghost_tensor, paddle::CPUPlace(), true);
  int nghost = *cpu_nghost.data<int>();
  int ntotal = nlocal + nghost;

  paddle::Tensor send_g1_tensor, recv_g1_tensor;
  FPTYPE *recv_g1 = nullptr, *send_g1 = nullptr;

  if (nswap != 0) {
    send_g1_tensor = d_local_g1_tensor;

    int max_recvnum =
        *(paddle::experimental::max(cpu_sendnum, {}, false).data<int>());
    recv_g1_tensor =
        paddle::empty({max_recvnum, tensor_size}, d_local_g1_tensor.dtype(),
                      d_local_g1_tensor.place());
    recv_g1 = recv_g1_tensor.data<FPTYPE>();
    send_g1 = send_g1_tensor.data<FPTYPE>() + ntotal * tensor_size;
  }

  for (int iswap = nswap - 1; iswap >= 0; --iswap) {
    int nrecv = recvnum[iswap];
    int nsend = sendnum[iswap];

    paddle::Tensor irecvlist;
    if (nrecv) {
      std::intptr_t addr = static_cast<std::intptr_t>(recvlist[iswap]);
      int* irecvlist_ptr = reinterpret_cast<int*>(addr);
      irecvlist =
          paddle::from_blob(irecvlist_ptr, {nrecv}, paddle::DataType::INT32,
                            paddle::DataLayout::NCHW, paddle::CPUPlace())
              .copy_to(d_local_g1_tensor.place(), true);
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
    recv_g1_tensor_grad.copy_(d_local_g1_tensor, recv_g1_tensor_grad.place(),
                              true);
  }
#endif
#endif
}

void Border_backward(const paddle::Tensor& sendlist_tensor,
                     const paddle::Tensor& sendproc_tensor,
                     const paddle::Tensor& recvproc_tensor,
                     const paddle::Tensor& sendnum_tensor,
                     const paddle::Tensor& recvnum_tensor,
                     const paddle::Tensor& g1_tensor,
                     const paddle::Tensor& communicator_tensor,
                     const paddle::Tensor& nlocal_tensor,
                     const paddle::Tensor& nghost_tensor,
                     paddle::Tensor& recv_g1_tensor_grad) {
  bool type_flag =
      (recv_g1_tensor_grad.dtype() == paddle::DataType::FLOAT64) ? true : false;
  if (type_flag) {
    Border_backward_t<double>(sendlist_tensor, sendproc_tensor, recvproc_tensor,
                              sendnum_tensor, recvnum_tensor, g1_tensor,
                              communicator_tensor, nlocal_tensor, nghost_tensor,
                              recv_g1_tensor_grad);
  } else {
    Border_backward_t<float>(sendlist_tensor, sendproc_tensor, recvproc_tensor,
                             sendnum_tensor, recvnum_tensor, g1_tensor,
                             communicator_tensor, nlocal_tensor, nghost_tensor,
                             recv_g1_tensor_grad);
  }
}

/**
 * @brief communicate the latest g1_tensor info to other lmp proc
 * @param[in]  sendlist_tensor list of atoms to send in each swap
 * @param[in]  sendproc_tensor proc to send to at each swap
 * @param[in]  recvproc_tensor proc to recv from at each swap
 * @param[in]  sendnum_tensor # of atoms to send in each swap
 * @param[in]  recvnum_tensor # of atoms to recv in each swap
 * @param[in]  g1_tensor tensor to store g1_tensor info
 * @param[in]  communicator_tensor MPI_comm data in lmp
 * @param[in]  nlocal_tensor # of local atoms
 * @param[in]  nghost_tensor # of nghost atoms
 * @param[out] recv_g1_tensor g1_tensor after communication
 **/
PD_BUILD_OP(border_op)
    .Inputs({"sendlist_tensor", "sendproc_tensor", "recvproc_tensor",
             "sendnum_tensor", "recvnum_tensor", "g1_tensor",
             "communicator_tensor", "nlocal_tensor", "nghost_tensor"})
    .Outputs({"recv_g1_tensor"})
    .SetKernelFn(PD_KERNEL(Border_forward))
    .SetInplaceMap({{"g1_tensor", "recv_g1_tensor"}});

PD_BUILD_GRAD_OP(border_op)
    .Inputs({"sendlist_tensor", "sendproc_tensor", "recvproc_tensor",
             "sendnum_tensor", "recvnum_tensor", "g1_tensor",
             "communicator_tensor", "nlocal_tensor", "nghost_tensor",
             paddle::Grad("recv_g1_tensor")})
    .Outputs({paddle::Grad("g1_tensor")})
    .SetInplaceMap({{paddle::Grad("recv_g1_tensor"),
                     paddle::Grad("g1_tensor")}})
    .SetKernelFn(PD_KERNEL(Border_backward));
