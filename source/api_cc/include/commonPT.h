// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#ifdef BUILD_PYTORCH
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
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

/**
 * @brief Flatten a jagged neighbor list into a [1, nloc, nnei] tensor.
 *
 * Each row in @p data may have a different number of neighbors.  Short rows
 * are padded with -1.  The output width is max(min_nnei, max_row_length).
 * No truncation or distance sorting is done — the model's format_nlist
 * handles that on-device.
 *
 * If @p min_nnei is 0 (the default used by the .pth callers) and every row
 * is empty (no atom has any neighbor — fully-dissociated system), the
 * output shape is [1, nloc, 0].  PyTorch accepts zero-sized dimensions, and
 * the eager `_format_nlist` pads it back up to sum(sel).  .pt2 callers
 * always pass @p min_nnei = sum(sel) > 0, so the output width is at least
 * sum(sel) for them.
 *
 * @param data      Jagged neighbor list: data[i] holds neighbor indices
 *                  for local atom i.
 * @param min_nnei  Minimum width of the nnei dimension.  For .pt2 models
 *                  this should be sum(sel) from the model metadata, because
 *                  torch.export marks nnei >= sum(sel) as a dynamic constraint.
 *                  For .pth models 0 (the default) is fine.
 */
inline torch::Tensor createNlistTensor(
    const std::vector<std::vector<int>>& data, int min_nnei = 0) {
  int nloc = static_cast<int>(data.size());
  int nnei = min_nnei;
  for (int ii = 0; ii < nloc; ++ii) {
    nnei = std::max(nnei, static_cast<int>(data[ii].size()));
  }
  std::vector<int> flat_data(static_cast<size_t>(nloc) * nnei, -1);
  for (int ii = 0; ii < nloc; ++ii) {
    for (size_t jj = 0; jj < data[ii].size(); ++jj) {
      flat_data[static_cast<size_t>(ii) * nnei + jj] = data[ii][jj];
    }
  }
  torch::Tensor flat_tensor = torch::tensor(flat_data, torch::kInt32);
  return flat_tensor.view({1, nloc, nnei});
}

struct EdgeTensorPack {
  torch::Tensor edge_index;
  torch::Tensor edge_vec;
  torch::Tensor edge_index_ext;
  torch::Tensor edge_mask;
};

/**
 * @brief Build compact edge tensors from a neighbor list.
 *
 * The returned tensors are aligned by edge:
 * - edge_index uses flattened local-atom indices and drives descriptor message
 *   passing.
 * - edge_index_ext uses flattened extended-atom indices and drives force and
 *   virial scatter.
 * - edge_mask marks physical edges. When geometry is requested, two masked
 *   dummy edges are appended so the exported graph never observes a singular
 *   edge dimension.
 *
 * @param nlist Neighbor-list rows.  By default row i is center atom i; callers
 *   that compact LAMMPS rows must pass row_centers.
 * @param coord Extended coordinates shaped as nall x 3.
 * @param mapping Extended-to-local atom map with length nall.
 * @param nloc Number of local atoms.
 * @param nall Number of extended atoms.
 * @param device Target device for the returned tensors.
 * @param with_geometry Whether to also materialize edge_vec, edge_mask and
 *   model-input dummy edges. The cached LAMMPS path keeps only the real skin
 *   topology and compacts it on-device every step, so it passes ``false``.
 *   The returned edge_vec and edge_mask are left undefined in that case.
 * @param row_centers Optional center atom index for each neighbor-list row.
 */
template <typename VALUETYPE>
inline EdgeTensorPack createEdgeTensors(
    const std::vector<std::vector<int>>& nlist,
    const std::vector<VALUETYPE>& coord,
    const std::vector<std::int64_t>& mapping,
    const int nloc,
    const int nall,
    const torch::Device& device,
    const bool with_geometry = true,
    const std::vector<int>* row_centers = nullptr) {
  std::vector<std::int64_t> src;
  std::vector<std::int64_t> dst;
  std::vector<std::int64_t> src_ext;
  std::vector<std::int64_t> dst_ext;
  std::vector<VALUETYPE> edge_vec;
  size_t reserve_size = with_geometry ? 2 : 0;
  for (const auto& row : nlist) {
    reserve_size += row.size();
  }
  src.reserve(reserve_size);
  dst.reserve(reserve_size);
  src_ext.reserve(reserve_size);
  dst_ext.reserve(reserve_size);
  if (with_geometry) {
    edge_vec.reserve(reserve_size * 3);
  }

  // Real edges: use row_centers when LAMMPS has compacted away empty rows.
  for (int ii = 0; ii < static_cast<int>(nlist.size()); ++ii) {
    if (row_centers != nullptr &&
        static_cast<size_t>(ii) >= row_centers->size()) {
      continue;
    }
    const int center =
        row_centers == nullptr ? ii : (*row_centers)[static_cast<size_t>(ii)];
    if (center < 0 || center >= nloc || center >= nall) {
      continue;
    }
    const size_t center_offset = static_cast<size_t>(center) * 3;
    for (const int jj : nlist[ii]) {
      if (jj < 0 || jj >= nall) {
        continue;
      }
      const std::int64_t src_local = mapping[static_cast<size_t>(jj)];
      if (src_local < 0 || src_local >= nloc) {
        continue;
      }
      const size_t neighbor_offset = static_cast<size_t>(jj) * 3;
      const VALUETYPE dx = coord[neighbor_offset] - coord[center_offset];
      const VALUETYPE dy =
          coord[neighbor_offset + 1] - coord[center_offset + 1];
      const VALUETYPE dz =
          coord[neighbor_offset + 2] - coord[center_offset + 2];
      const VALUETYPE rr = dx * dx + dy * dy + dz * dz;
      if (rr <= static_cast<VALUETYPE>(1e-10)) {
        continue;
      }
      src.push_back(src_local);
      dst.push_back(center);
      src_ext.push_back(jj);
      dst_ext.push_back(center);
      if (with_geometry) {
        edge_vec.push_back(dx);
        edge_vec.push_back(dy);
        edge_vec.push_back(dz);
      }
    }
  }

  const size_t real_edges = src.size();
  if (with_geometry) {
    // Dummy edges keep exported edge tensors non-empty without affecting
    // output.
    for (int ii = 0; ii < 2; ++ii) {
      src.push_back(0);
      dst.push_back(0);
      src_ext.push_back(0);
      dst_ext.push_back(0);
      edge_vec.push_back(0);
      edge_vec.push_back(0);
      edge_vec.push_back(0);
    }
  }
  const size_t nedge = src.size();
  std::vector<std::int64_t> edge_index(2 * nedge);
  std::vector<std::int64_t> edge_index_ext(2 * nedge);
  // Materialize local-owner and extended scatter index spaces side by side.
  for (size_t ii = 0; ii < nedge; ++ii) {
    edge_index[ii] = src[ii];
    edge_index[nedge + ii] = dst[ii];
    edge_index_ext[ii] = src_ext[ii];
    edge_index_ext[nedge + ii] = dst_ext[ii];
  }

  auto int_options = torch::TensorOptions().dtype(torch::kInt64);
  EdgeTensorPack pack;
  if (nedge == 0) {
    pack.edge_index = torch::empty({2, 0}, int_options).to(device);
    pack.edge_index_ext = torch::empty({2, 0}, int_options).to(device);
  } else {
    pack.edge_index =
        torch::from_blob(edge_index.data(),
                         {2, static_cast<std::int64_t>(nedge)}, int_options)
            .clone()
            .to(device);
    pack.edge_index_ext =
        torch::from_blob(edge_index_ext.data(),
                         {2, static_cast<std::int64_t>(nedge)}, int_options)
            .clone()
            .to(device);
  }
  if (with_geometry) {
    pack.edge_vec =
        torch::from_blob(
            edge_vec.data(), {static_cast<std::int64_t>(nedge), 3},
            torch::TensorOptions().dtype(std::is_same<VALUETYPE, float>::value
                                             ? torch::kFloat32
                                             : torch::kFloat64))
            .clone()
            .to(device);
    std::vector<std::uint8_t> edge_mask(nedge, 0);
    std::fill(edge_mask.begin(), edge_mask.begin() + real_edges,
              static_cast<std::uint8_t>(1));
    pack.edge_mask =
        torch::from_blob(edge_mask.data(), {static_cast<std::int64_t>(nedge)},
                         torch::TensorOptions().dtype(torch::kUInt8))
            .clone()
            .to(torch::kBool)
            .to(device);
  }
  return pack;
}

/**
 * @brief Compact a cached LAMMPS skin topology to the current cutoff edge set.
 *
 * LAMMPS rebuilds neighbor topology only when its skin list is refreshed.  The
 * SeZM lower graph, however, should see only the current model-cutoff edges.
 * This helper keeps the cached skin topology immutable, recomputes edge
 * vectors from the current coordinates on the target device, filters by
 * ``rr <= rcut**2``, then appends the two masked dummy edges required by the
 * exported graph contract.
 */
inline EdgeTensorPack compactEdgeTensors(const torch::Tensor& edge_index,
                                         const torch::Tensor& edge_index_ext,
                                         const torch::Tensor& coord,
                                         const double rcut) {
  const auto coord_flat = coord.reshape({-1, 3});
  const auto src_ext = edge_index_ext.select(0, 0);
  const auto dst_ext = edge_index_ext.select(0, 1);
  const auto edge_vec_all =
      coord_flat.index_select(0, src_ext) - coord_flat.index_select(0, dst_ext);
  const auto rr = (edge_vec_all * edge_vec_all).sum(1);
  const auto keep = (rr > 1e-10) & (rr <= rcut * rcut);
  const auto real_idx = torch::nonzero(keep).reshape({-1});

  EdgeTensorPack pack;
  const auto real_edge_index = edge_index.index_select(1, real_idx);
  const auto real_edge_index_ext = edge_index_ext.index_select(1, real_idx);
  const auto real_edge_vec = edge_vec_all.index_select(0, real_idx);

  const auto dummy_index = torch::zeros({2, 2}, edge_index.options());
  const auto dummy_vec = torch::zeros({2, 3}, edge_vec_all.options());
  pack.edge_index = torch::cat({real_edge_index, dummy_index}, 1);
  pack.edge_index_ext = torch::cat({real_edge_index_ext, dummy_index}, 1);
  pack.edge_vec = torch::cat({real_edge_vec, dummy_vec}, 0);

  const auto real_mask = torch::ones(
      {real_idx.size(0)},
      torch::TensorOptions().dtype(torch::kBool).device(coord.device()));
  const auto dummy_mask = torch::zeros({2}, real_mask.options());
  pack.edge_mask = torch::cat({real_mask, dummy_mask}, 0);
  return pack;
}

}  // namespace deepmd

#endif  // BUILD_PYTORCH
