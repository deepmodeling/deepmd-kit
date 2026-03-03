// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#ifdef BUILD_PYTORCH
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "common.h"
#include "neighbor_list.h"

namespace deepmd {

/**
 * @brief Build comm_dict tensors from sendlist/sendnum/recvnum buffers.
 *
 * This is the shared tensor-building logic for all PyTorch backends
 * (DeepPotPT, DeepSpinPT). Backend-specific entries (e.g. has_spin)
 * should be added by the caller after this function returns.
 *
 * @param[out] comm_dict The communication dictionary to populate.
 * @param[in] lmp_list The LAMMPS neighbor list (for sendproc/recvproc/world).
 * @param[in] sendlist Pointer array (int**) for each swap's send list.
 * @param[in] sendnum Number of send atoms per swap.
 * @param[in] recvnum Number of recv atoms per swap.
 */
inline void build_comm_dict(torch::Dict<std::string, torch::Tensor>& comm_dict,
                            const InputNlist& lmp_list,
                            int** sendlist,
                            int* sendnum,
                            int* recvnum) {
  int nswap = lmp_list.nswap;
  auto int32_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
  auto int64_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);

  torch::Tensor sendlist_tensor =
      torch::from_blob(static_cast<void*>(sendlist), {nswap}, int64_option);
  torch::Tensor sendnum_tensor =
      torch::from_blob(sendnum, {nswap}, int32_option);
  torch::Tensor recvnum_tensor =
      torch::from_blob(recvnum, {nswap}, int32_option);
  torch::Tensor sendproc_tensor =
      torch::from_blob(lmp_list.sendproc, {nswap}, int32_option);
  torch::Tensor recvproc_tensor =
      torch::from_blob(lmp_list.recvproc, {nswap}, int32_option);

  torch::Tensor communicator_tensor;
  static std::int64_t null_communicator = 0;
  if (lmp_list.world == nullptr) {
    communicator_tensor =
        torch::from_blob(&null_communicator, {1}, torch::kInt64);
  } else {
    communicator_tensor =
        torch::from_blob(const_cast<void*>(lmp_list.world), {1}, torch::kInt64);
  }

  comm_dict.insert_or_assign("send_list", sendlist_tensor);
  comm_dict.insert_or_assign("send_proc", sendproc_tensor);
  comm_dict.insert_or_assign("recv_proc", recvproc_tensor);
  comm_dict.insert_or_assign("send_num", sendnum_tensor);
  comm_dict.insert_or_assign("recv_num", recvnum_tensor);
  comm_dict.insert_or_assign("communicator", communicator_tensor);
}

/**
 * @brief Build comm_dict with sendlist remapping for virtual (NULL-type) atoms.
 *
 * Calls remap_comm_sendlist() to remap indices through fwd_map, then
 * build_comm_dict() to create tensors. Backend-specific entries (e.g.
 * has_spin) should be added by the caller after this function returns.
 *
 * @param[out] comm_dict The communication dictionary to populate.
 * @param[in] lmp_list The LAMMPS neighbor list containing communication info.
 * @param[in] fwd_map Map from original atom index to real-atom index (-1 for
 *            virtual atoms).
 * @param[out] remapped_sendlist Storage for remapped send lists (kept alive for
 *             tensor lifetime).
 * @param[out] remapped_sendlist_ptrs Pointer array into remapped_sendlist.
 * @param[out] remapped_sendnum Remapped send counts per swap.
 * @param[out] remapped_recvnum Remapped recv counts per swap.
 */
inline void build_comm_dict_with_virtual_atoms(
    torch::Dict<std::string, torch::Tensor>& comm_dict,
    const InputNlist& lmp_list,
    const std::vector<int>& fwd_map,
    std::vector<std::vector<int>>& remapped_sendlist,
    std::vector<int*>& remapped_sendlist_ptrs,
    std::vector<int>& remapped_sendnum,
    std::vector<int>& remapped_recvnum) {
  remap_comm_sendlist(remapped_sendlist, remapped_sendnum, remapped_recvnum,
                      lmp_list, fwd_map);
  int nswap = lmp_list.nswap;
  remapped_sendlist_ptrs.resize(nswap);
  for (int s = 0; s < nswap; ++s) {
    remapped_sendlist_ptrs[s] = remapped_sendlist[s].data();
  }
  build_comm_dict(comm_dict, lmp_list, remapped_sendlist_ptrs.data(),
                  remapped_sendnum.data(), remapped_recvnum.data());
}

}  // namespace deepmd

#endif  // BUILD_PYTORCH
