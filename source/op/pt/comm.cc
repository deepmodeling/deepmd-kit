// SPDX-License-Identifier: LGPL-3.0-or-later

#ifdef USE_MPI
#include <mpi.h>
#ifdef OMPI_MPI_H
#include <mpi-ext.h>
#endif
#endif
#include <torch/torch.h>

#include <cstdint>

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

class Border : public torch::autograd::Function<Border> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& sendlist_tensor,
      const torch::Tensor& sendproc_tensor,
      const torch::Tensor& recvproc_tensor,
      const torch::Tensor& sendnum_tensor,
      const torch::Tensor& recvnum_tensor,
      const torch::Tensor& g1,
      const torch::Tensor& communicator_tensor,
      const torch::Tensor& nlocal_tensor,
      const torch::Tensor& nghost_tensor) {
    bool type_flag = (g1.dtype() == torch::kDouble) ? true : false;
    if (type_flag) {
      return forward_t<double>(ctx, sendlist_tensor, sendproc_tensor,
                               recvproc_tensor, sendnum_tensor, recvnum_tensor,
                               g1, communicator_tensor, nlocal_tensor,
                               nghost_tensor);
    } else {
      return forward_t<float>(ctx, sendlist_tensor, sendproc_tensor,
                              recvproc_tensor, sendnum_tensor, recvnum_tensor,
                              g1, communicator_tensor, nlocal_tensor,
                              nghost_tensor);
    }
  }
  template <typename FPTYPE>
  static torch::autograd::variable_list forward_t(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& sendlist_tensor,
      const torch::Tensor& sendproc_tensor,
      const torch::Tensor& recvproc_tensor,
      const torch::Tensor& sendnum_tensor,
      const torch::Tensor& recvnum_tensor,
      const torch::Tensor& g1,
      const torch::Tensor& communicator_tensor,
      const torch::Tensor& nlocal_tensor,
      const torch::Tensor& nghost_tensor) {
    ctx->save_for_backward({sendlist_tensor, sendproc_tensor, recvproc_tensor,
                            sendnum_tensor, recvnum_tensor, communicator_tensor,
                            nlocal_tensor, nghost_tensor});
    int** sendlist = reinterpret_cast<int**>(sendlist_tensor.data_ptr());
    int* sendproc = sendproc_tensor.data_ptr<int>();
    int* recvproc = recvproc_tensor.data_ptr<int>();
    int* sendnum = sendnum_tensor.data_ptr<int>();
    int* recvnum = recvnum_tensor.data_ptr<int>();
    int tensor_size = g1.size(1);
    int nswap = sendproc_tensor.size(0);

    int nlocal = nlocal_tensor.item<int>();
    int nghost = nghost_tensor.item<int>();
    int ntotal = nlocal + nghost;
    torch::Tensor recv_g1_tensor = g1;

#ifdef USE_MPI
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    int cuda_aware = 0;
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
#ifndef NO_CUDA_AWARE
      cuda_aware = MPIX_Query_cuda_support();
#endif
      if (cuda_aware == 0) {
        recv_g1_tensor = torch::empty_like(g1).to(torch::kCPU);
        recv_g1_tensor.copy_(g1);
      }
    }
#endif
#endif
    FPTYPE* recv_g1 = recv_g1_tensor.data_ptr<FPTYPE>() + nlocal * tensor_size;
    auto int32_options = torch::TensorOptions().dtype(torch::kInt32);
    for (int iswap = 0; iswap < nswap; ++iswap) {
      int nrecv = recvnum[iswap];
      int nsend = sendnum[iswap];
      torch::Tensor isendlist;
      torch::Tensor send_g1_tensor;
      FPTYPE* send_g1;
      if (nsend != 0) {
        isendlist = torch::from_blob(sendlist[iswap], {nsend}, int32_options)
                        .to(recv_g1_tensor.device());
        send_g1_tensor = recv_g1_tensor.index_select(0, isendlist);
        send_g1 = send_g1_tensor.data_ptr<FPTYPE>();
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
        // Self-send branch: choose the host-vs-device memcpy based on
        // where the data actually lives, not on MPI state. The buffer
        // we read/write is ``recv_g1_tensor`` whose device is either
        // (a) the original ``g1`` device, or (b) CPU after the
        // non-cuda-aware MPI fallback above. Reading that device
        // directly is the only correct dispatch for build configs
        // where USE_MPI is on but the call site uses CPU tensors
        // (e.g. unit tests of border_op without MPI init).
        if (recv_g1_tensor.is_cuda()) {
          gpuMemcpy(recv_g1, send_g1,
                    (unsigned long)nsend * tensor_size * sizeof(FPTYPE),
                    gpuMemcpyDeviceToDevice);
        } else {
          memcpy(recv_g1, send_g1,
                 (unsigned long)nsend * tensor_size * sizeof(FPTYPE));
        }
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
    // Only copy back when ``recv_g1_tensor`` was actually moved to a
    // different device above (the cuda_aware==0 CPU fallback). When
    // ``recv_g1_tensor`` still aliases ``g1`` no copy is needed; the
    // is_alias_of check is the precise correctness condition and works
    // for both CUDA and CPU call sites.
    if (!recv_g1_tensor.is_alias_of(g1)) {
      g1.copy_(recv_g1_tensor);
    }
#endif
#endif
    return {g1};
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_variables = ctx->get_saved_variables();
    torch::Tensor sendlist_tensor = saved_variables[0];
    torch::Tensor sendproc_tensor = saved_variables[1];
    torch::Tensor recvproc_tensor = saved_variables[2];
    torch::Tensor sendnum_tensor = saved_variables[3];
    torch::Tensor recvnum_tensor = saved_variables[4];
    torch::Tensor communicator_tensor = saved_variables[5];
    torch::Tensor nlocal_tensor = saved_variables[6];
    torch::Tensor nghost_tensor = saved_variables[7];
    torch::Tensor d_in = border_op_backward_dispatch(
        sendlist_tensor, sendproc_tensor, recvproc_tensor, sendnum_tensor,
        recvnum_tensor, grad_output[0], communicator_tensor, nlocal_tensor,
        nghost_tensor);
    return {torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), d_in,
            torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor()};
  }

  // Forward declaration; defined as a free function below so it can be
  // registered as a separate torch op (deepmd::border_op_backward) used by
  // the pt_expt opaque-op autograd wrapper.
  static torch::Tensor border_op_backward_dispatch(
      const torch::Tensor& sendlist_tensor,
      const torch::Tensor& sendproc_tensor,
      const torch::Tensor& recvproc_tensor,
      const torch::Tensor& sendnum_tensor,
      const torch::Tensor& recvnum_tensor,
      const torch::Tensor& grad_g1,
      const torch::Tensor& communicator_tensor,
      const torch::Tensor& nlocal_tensor,
      const torch::Tensor& nghost_tensor);

  template <typename FPTYPE>
  static torch::Tensor backward_t(const torch::Tensor& sendlist_tensor,
                                  const torch::Tensor& sendproc_tensor,
                                  const torch::Tensor& recvproc_tensor,
                                  const torch::Tensor& sendnum_tensor,
                                  const torch::Tensor& recvnum_tensor,
                                  const torch::Tensor& grad_g1,
                                  const torch::Tensor& communicator_tensor,
                                  const torch::Tensor& nlocal_tensor,
                                  const torch::Tensor& nghost_tensor) {
    torch::Tensor d_local_g1_tensor = grad_g1.contiguous();
#ifdef USE_MPI
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    int world_size = 0;
    int cuda_aware = 0;
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
#ifndef NO_CUDA_AWARE
      cuda_aware = MPIX_Query_cuda_support();
#endif
      if (cuda_aware == 0) {
        d_local_g1_tensor = torch::empty_like(grad_g1).to(torch::kCPU);
        d_local_g1_tensor.copy_(grad_g1);
      }
    }
#endif
#endif
    int** recvlist = reinterpret_cast<int**>(sendlist_tensor.data_ptr());
    // swap send and recv here
    int* recvproc = sendproc_tensor.data_ptr<int>();
    int* sendproc = recvproc_tensor.data_ptr<int>();
    int* recvnum = sendnum_tensor.data_ptr<int>();
    int* sendnum = recvnum_tensor.data_ptr<int>();

    FPTYPE* local_g1 = d_local_g1_tensor.data_ptr<FPTYPE>();
    int tensor_size = d_local_g1_tensor.size(1);
    int nswap = sendproc_tensor.size(0);

    int nlocal = nlocal_tensor.item<int>();
    int nghost = nghost_tensor.item<int>();
    int ntotal = nlocal + nghost;
    torch::Tensor send_g1_tensor;
    torch::Tensor recv_g1_tensor;
    FPTYPE* recv_g1;
    FPTYPE* send_g1;
    if (nswap != 0) {
      send_g1_tensor = d_local_g1_tensor;
      int max_recvnum = sendnum_tensor.max().item<int>();
      auto options = torch::TensorOptions()
                         .dtype(d_local_g1_tensor.dtype())
                         .device(d_local_g1_tensor.device());
      recv_g1_tensor = torch::empty({max_recvnum, tensor_size}, options);
      recv_g1 = recv_g1_tensor.data_ptr<FPTYPE>();
      send_g1 = send_g1_tensor.data_ptr<FPTYPE>() + ntotal * tensor_size;
    }

    int end = ntotal;
    auto int32_options = torch::TensorOptions().dtype(torch::kInt32);
    for (int iswap = nswap - 1; iswap >= 0; --iswap) {
      int nrecv = recvnum[iswap];
      int nsend = sendnum[iswap];

      torch::Tensor irecvlist;
      if (nrecv) {
        irecvlist = torch::from_blob(recvlist[iswap], {nrecv}, int32_options)
                        .to(d_local_g1_tensor.device());
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
          // Self-send branch: dispatch on the actual device of the
          // ``recv_g1_tensor`` buffer, not on MPI state. Same rationale
          // as the forward kernel — USE_MPI builds may be called with
          // CPU tensors (unit tests of border_op_backward) where the
          // gpuMemcpy path silently fails with cudaErrorInvalidValue
          // and leaves recv_g1 uninitialized.
          if (recv_g1_tensor.is_cuda()) {
            gpuMemcpy(recv_g1, send_g1,
                      (unsigned long)nrecv * tensor_size * sizeof(FPTYPE),
                      gpuMemcpyDeviceToDevice);
          } else {
            memcpy(recv_g1, send_g1,
                   (unsigned long)nrecv * tensor_size * sizeof(FPTYPE));
          }
#else
        memcpy(recv_g1, send_g1,
               (unsigned long)nrecv * tensor_size * sizeof(FPTYPE));
#endif
        }
#ifdef USE_MPI
      }
#endif
      if (nrecv) {
        d_local_g1_tensor.index_add_(0, irecvlist,
                                     recv_g1_tensor.slice(0, 0, nrecv));
      }
    }
#ifdef USE_MPI
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
    // Move result back to the device of the input grad only when
    // ``d_local_g1_tensor`` was actually moved to a different device
    // above (the cuda_aware==0 CPU fallback). The is_alias_of check
    // is the precise correctness condition and works for both CUDA
    // and CPU call sites (no-op when the tensor still aliases input).
    if (!d_local_g1_tensor.is_alias_of(grad_g1)) {
      d_local_g1_tensor = d_local_g1_tensor.to(grad_g1.device());
    }
#endif
#endif
    return d_local_g1_tensor;
  }

#ifdef USE_MPI
  static void unpack_communicator(const torch::Tensor& communicator_tensor,
                                  MPI_Comm& mpi_comm) {
#ifdef OMPI_MPI_H
    std::int64_t* communicator = communicator_tensor.data_ptr<std::int64_t>();
#else
    std::int64_t* ptr = communicator_tensor.data_ptr<std::int64_t>();
    int* communicator = reinterpret_cast<int*>(ptr);
#endif
    mpi_comm = reinterpret_cast<MPI_Comm>(*communicator);
  }
#endif
};
std::vector<torch::Tensor> border_op(const torch::Tensor& sendlist_tensor,
                                     const torch::Tensor& sendproc_tensor,
                                     const torch::Tensor& recvproc_tensor,
                                     const torch::Tensor& sendnum_tensor,
                                     const torch::Tensor& recvnum_tensor,
                                     const torch::Tensor& g1_tensor,
                                     const torch::Tensor& communicator_tensor,
                                     const torch::Tensor& nlocal_tensor,
                                     const torch::Tensor& nghost_tensor)

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
{
  return Border::apply(sendlist_tensor, sendproc_tensor, recvproc_tensor,
                       sendnum_tensor, recvnum_tensor, g1_tensor,
                       communicator_tensor, nlocal_tensor, nghost_tensor);
}

// Define Border::border_op_backward_dispatch out-of-line so the type-flag
// dispatch can refer to the templated backward_t members declared in the
// class.
torch::Tensor Border::border_op_backward_dispatch(
    const torch::Tensor& sendlist_tensor,
    const torch::Tensor& sendproc_tensor,
    const torch::Tensor& recvproc_tensor,
    const torch::Tensor& sendnum_tensor,
    const torch::Tensor& recvnum_tensor,
    const torch::Tensor& grad_g1,
    const torch::Tensor& communicator_tensor,
    const torch::Tensor& nlocal_tensor,
    const torch::Tensor& nghost_tensor) {
  bool type_flag = (grad_g1.dtype() == torch::kDouble);
  if (type_flag) {
    return backward_t<double>(sendlist_tensor, sendproc_tensor, recvproc_tensor,
                              sendnum_tensor, recvnum_tensor, grad_g1,
                              communicator_tensor, nlocal_tensor,
                              nghost_tensor);
  } else {
    return backward_t<float>(sendlist_tensor, sendproc_tensor, recvproc_tensor,
                             sendnum_tensor, recvnum_tensor, grad_g1,
                             communicator_tensor, nlocal_tensor, nghost_tensor);
  }
}

/**
 * @brief Standalone backward of border_op for use by pt_expt's opaque-op
 * autograd wrapper. Performs the symmetric MPI exchange that the autograd
 * Border::backward applies, but without an autograd context — comm tensors
 * are passed directly so the op can be registered as a torch op and
 * embedded in an AOTInductor graph.
 *
 * The comm topology is symmetric: the same sendlist/sendnum/recvnum buffers
 * encode the forward exchange; backward simply swaps send <-> recv and
 * accumulates gradients into the local atom slots.
 *
 * @param[in]  sendlist_tensor  send-list pointer-array (forward direction)
 * @param[in]  sendproc_tensor  send-proc IDs (forward direction)
 * @param[in]  recvproc_tensor  recv-proc IDs (forward direction)
 * @param[in]  sendnum_tensor   atoms sent per swap (forward direction)
 * @param[in]  recvnum_tensor   atoms received per swap (forward direction)
 * @param[in]  grad_g1          upstream gradient w.r.t. g1 of forward
 * @param[in]  communicator_tensor MPI communicator handle as int64
 * @param[in]  nlocal_tensor    number of local atoms (per rank)
 * @param[in]  nghost_tensor    number of ghost atoms (per rank)
 * @return d_in (gradient w.r.t. forward g1 input), same shape as grad_g1.
 */
torch::Tensor border_op_backward(const torch::Tensor& sendlist_tensor,
                                 const torch::Tensor& sendproc_tensor,
                                 const torch::Tensor& recvproc_tensor,
                                 const torch::Tensor& sendnum_tensor,
                                 const torch::Tensor& recvnum_tensor,
                                 const torch::Tensor& grad_g1,
                                 const torch::Tensor& communicator_tensor,
                                 const torch::Tensor& nlocal_tensor,
                                 const torch::Tensor& nghost_tensor) {
  return Border::border_op_backward_dispatch(
      sendlist_tensor, sendproc_tensor, recvproc_tensor, sendnum_tensor,
      recvnum_tensor, grad_g1, communicator_tensor, nlocal_tensor,
      nghost_tensor);
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def("border_op", border_op);
  m.def("border_op_backward", border_op_backward);
}

// ============================================================================
// Opaque wrappers for the pt_expt (.pt2 / AOTInductor) export path.
//
// ``deepmd::border_op`` and ``deepmd::border_op_backward`` are registered
// without an explicit dispatch key, which makes them
// ``CompositeImplicitAutograd`` ops.  ``torch.export`` decomposes such ops
// during tracing — i.e., it tries to inline the C++ kernel — and that
// fails because the kernel calls ``data_ptr()`` on FakeTensors.
//
// These ``deepmd_export::*`` wrappers are registered with explicit
// ``CPU`` and ``CUDA`` dispatch keys so ``torch.export`` records them as
// opaque external calls in the graph.  The .pt2 archive embeds the call
// sites; at runtime the dispatcher routes back to the underlying
// ``deepmd::*`` op.  Both clones because ``deepmd::border_op`` returns
// the same tensor it modified in place, which violates AOTInductor's
// no-aliasing rule for graph outputs.
//
// Python (``deepmd/pt_expt/utils/comm.py``) layers ``register_fake`` and
// ``register_autograd`` on top of these C++-defined ops so traced graphs
// can run their fake/backward.
// ============================================================================

namespace {
torch::Tensor border_op_export(const torch::Tensor& sendlist_tensor,
                               const torch::Tensor& sendproc_tensor,
                               const torch::Tensor& recvproc_tensor,
                               const torch::Tensor& sendnum_tensor,
                               const torch::Tensor& recvnum_tensor,
                               const torch::Tensor& g1_tensor,
                               const torch::Tensor& communicator_tensor,
                               const torch::Tensor& nlocal_tensor,
                               const torch::Tensor& nghost_tensor) {
  auto out = border_op(sendlist_tensor, sendproc_tensor, recvproc_tensor,
                       sendnum_tensor, recvnum_tensor, g1_tensor,
                       communicator_tensor, nlocal_tensor, nghost_tensor);
  // border_op returns {g1_tensor} — a list whose first element aliases
  // g1_tensor. Clone for AOTI graph-output correctness.
  return out.empty() ? torch::empty_like(g1_tensor) : out[0].clone();
}

torch::Tensor border_op_backward_export(
    const torch::Tensor& sendlist_tensor,
    const torch::Tensor& sendproc_tensor,
    const torch::Tensor& recvproc_tensor,
    const torch::Tensor& sendnum_tensor,
    const torch::Tensor& recvnum_tensor,
    const torch::Tensor& grad_g1,
    const torch::Tensor& communicator_tensor,
    const torch::Tensor& nlocal_tensor,
    const torch::Tensor& nghost_tensor) {
  return border_op_backward(sendlist_tensor, sendproc_tensor, recvproc_tensor,
                            sendnum_tensor, recvnum_tensor, grad_g1,
                            communicator_tensor, nlocal_tensor, nghost_tensor)
      .clone();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(deepmd_export, m) {
  m.def(
      "border_op(Tensor sendlist, Tensor sendproc, Tensor recvproc, "
      "Tensor sendnum, Tensor recvnum, Tensor g1, Tensor communicator, "
      "Tensor nlocal, Tensor nghost) -> Tensor");
  m.def(
      "border_op_backward(Tensor sendlist, Tensor sendproc, Tensor recvproc, "
      "Tensor sendnum, Tensor recvnum, Tensor grad_g1, Tensor communicator, "
      "Tensor nlocal, Tensor nghost) -> Tensor");
}

// Register CPU + CUDA implementations under explicit dispatch keys so
// torch.export sees opaque external calls (vs CompositeImplicitAutograd
// which gets decomposed during trace).
TORCH_LIBRARY_IMPL(deepmd_export, CPU, m) {
  m.impl("border_op", border_op_export);
  m.impl("border_op_backward", border_op_backward_export);
}
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
TORCH_LIBRARY_IMPL(deepmd_export, CUDA, m) {
  m.impl("border_op", border_op_export);
  m.impl("border_op_backward", border_op_backward_export);
}
#endif
