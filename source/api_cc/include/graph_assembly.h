// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "commonPT.h"
#include "errors.h"

namespace deepmd {

/**
 * @brief Destination-grouped skin topology, cached across neighbor rebuilds.
 *
 * A molecular-dynamics host rebuilds its neighbor list every few tens of
 * steps and reuses it in between, so the topology is cached while the geometry
 * is not. The cached form is compressed-sparse-row over the graph's node axis
 * rather than a flat edge list, because the host list already groups its
 * neighbours by center: keeping that grouping means the per-step assembly
 * needs a prefix sum instead of a sort.
 *
 * ``source_ext`` indexes the extended atoms and drives the geometry;
 * ``source_node`` indexes the graph nodes, which for a single-rank folded
 * graph is the local owner of the same atom. Both are 32-bit because a
 * neighbor list that a host can hold has fewer than two billion atoms, and
 * halving the index traffic matters for a pass whose arithmetic is three
 * subtractions per edge.
 */
struct SkinTopology {
  /// Offsets into the edge arrays, one per node plus the total.
  std::vector<std::int64_t> row_ptr;
  /// Extended index of each edge's source atom.
  std::vector<std::int32_t> source_ext;
  /// Node index of each edge's source atom.
  std::vector<std::int32_t> source_node;
  /// Number of nodes the row pointers span.
  std::int64_t node_count = 0;

  /// Return whether a topology has been built.
  bool empty() const { return row_ptr.empty(); }
};

/**
 * @brief Build the cached skin topology from a host neighbor list.
 *
 * Mirrors the contracts of ``createEdgeTensors``: rows may be compacted, so a
 * row's center comes from ``row_centers`` when present; a neighbour outside
 * the extended set is dropped; and ``fold_to_local`` selects between folding
 * ghost sources onto their local owners (single-rank message passing) and
 * indexing the extended atoms directly (multi-rank, where ghost features are
 * exchanged instead).
 *
 * @param nlist Neighbor-list rows holding extended indices.
 * @param mapping Extended-to-local owner map, length ``nall``.
 * @param nloc Number of local atoms.
 * @param nall Number of extended atoms.
 * @param node_count Size of the graph's node axis.
 * @param row_centers Center atom of each row, or null when row i centers on i.
 * @param fold_to_local Whether ghost sources fold onto their local owners.
 *
 * @return The destination-grouped skin topology.
 */
inline SkinTopology buildSkinTopology(
    const std::vector<std::vector<int>>& nlist,
    const std::vector<std::int64_t>& mapping,
    const int nloc,
    const int nall,
    const std::int64_t node_count,
    const std::vector<int>* row_centers,
    const bool fold_to_local) {
  if (fold_to_local && mapping.size() < static_cast<size_t>(nall)) {
    throw deepmd::deepmd_exception(
        "folding ghost neighbours onto their local owners needs an owner for "
        "each of the " +
        std::to_string(nall) + " extended atoms, but the mapping holds " +
        std::to_string(mapping.size()) +
        "; under LAMMPS this is what 'atom_modify map yes' supplies");
  }
  const std::int64_t row_count = static_cast<std::int64_t>(nlist.size());

  // === Step 1. Count the neighbours each node contributes ===
  // A row centers on one node, and no node owns two rows, so counting by
  // center and filling by center are both race-free over rows.
  SkinTopology topology;
  topology.node_count = node_count;
  topology.row_ptr.assign(node_count + 1, 0);
  std::vector<std::int64_t> center_of_row(row_count, -1);
  for (std::int64_t row = 0; row < row_count; ++row) {
    if (row_centers != nullptr &&
        static_cast<size_t>(row) >= row_centers->size()) {
      continue;
    }
    const std::int64_t center =
        row_centers == nullptr ? row : (*row_centers)[static_cast<size_t>(row)];
    if (center < 0 || center >= nloc || center >= nall ||
        center >= node_count) {
      continue;
    }
    center_of_row[row] = center;
    std::int64_t kept = 0;
    for (const int neighbor : nlist[static_cast<size_t>(row)]) {
      if (neighbor >= 0 && neighbor < nall) {
        ++kept;
      }
    }
    topology.row_ptr[center + 1] = kept;
  }
  for (std::int64_t node = 0; node < node_count; ++node) {
    topology.row_ptr[node + 1] += topology.row_ptr[node];
  }

  // === Step 2. Fill the source arrays into each node's range ===
  const std::int64_t edge_count = topology.row_ptr[node_count];
  topology.source_ext.resize(edge_count);
  topology.source_node.resize(edge_count);
  at::parallel_for(0, row_count, 1, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t row = begin; row < end; ++row) {
      const std::int64_t center = center_of_row[row];
      if (center < 0) {
        continue;
      }
      std::int64_t cursor = topology.row_ptr[center];
      for (const int neighbor : nlist[static_cast<size_t>(row)]) {
        if (neighbor < 0 || neighbor >= nall) {
          continue;
        }
        std::int64_t source = neighbor;
        if (fold_to_local) {
          source = mapping[static_cast<size_t>(neighbor)];
          // Folding is single-domain, where every extended atom has an owner
          // among the local ones. An owner outside that range marks a mapping
          // that was never filled, not a neighbour to skip: skipping would
          // discard the whole halo and leave a quietly incomplete graph.
          if (source < 0 || source >= nloc) {
            throw deepmd::deepmd_exception(
                "extended atom " + std::to_string(neighbor) + " of " +
                std::to_string(nall) + " maps to owner " +
                std::to_string(source) + ", which is not one of the " +
                std::to_string(nloc) +
                " local atoms; under LAMMPS an owner for every extended atom "
                "is what 'atom_modify map yes' supplies");
          }
        }
        topology.source_ext[cursor] = static_cast<std::int32_t>(neighbor);
        topology.source_node[cursor] = static_cast<std::int32_t>(source);
        ++cursor;
      }
    }
  });
  return topology;
}

/**
 * @brief Storage the assembly reuses across steps.
 *
 * The survivors of one chunk of nodes are staged here before they are copied
 * into the payload, so that the geometry is gathered once rather than once to
 * count and once to write. A chunk stages into the region its own skin edges
 * occupy, which is an upper bound on its survivors and needs no bookkeeping of
 * its own.
 */
struct GraphAssemblyScratch {
  /// Surviving edge count of each node, then its offsets.
  std::vector<std::int64_t> row_ptr;
};

namespace detail {

/// Return whether an edge survives the cutoff and the type exclusion.
template <typename VALUETYPE>
inline bool edge_survives(const VALUETYPE* coord,
                          const std::int64_t source_ext,
                          const std::int64_t center_ext,
                          const double rcut_squared,
                          double& dx,
                          double& dy,
                          double& dz) {
  dx = static_cast<double>(coord[source_ext * 3]) -
       static_cast<double>(coord[center_ext * 3]);
  dy = static_cast<double>(coord[source_ext * 3 + 1]) -
       static_cast<double>(coord[center_ext * 3 + 1]);
  dz = static_cast<double>(coord[source_ext * 3 + 2]) -
       static_cast<double>(coord[center_ext * 3 + 2]);
  const double rr = dx * dx + dy * dy + dz * dz;
  return rr > 1e-10 && rr <= rcut_squared;
}

/// Return whether a type pair is kept, given a flat keep table or none.
inline bool pair_kept(const int* keep_table,
                      const std::int64_t* atype,
                      const std::int64_t source_node,
                      const std::int64_t center,
                      const int ntypes) {
  if (keep_table == nullptr) {
    return true;
  }
  const std::int64_t source_type =
      std::max<std::int64_t>(atype[source_node], 0);
  const std::int64_t center_type = std::max<std::int64_t>(atype[center], 0);
  return keep_table[center_type * (ntypes + 1) + source_type] != 0;
}

}  // namespace detail

/**
 * @brief Assemble the destination-major neighbor graph for one geometry.
 *
 * Replaces a chain of roughly twenty tensor operations -- gather, difference,
 * norm, comparison, ``nonzero``, three ``index_select``, three ``cat``, a cast
 * and three sorts -- with one threaded pass over the cached topology. The chain
 * moved about half a gigabyte per step on a production system and several of
 * its stages were single-threaded.
 *
 * Two passes over the cached topology cost less than one pass that stages its
 * survivors: the coordinates a host-sized neighbor list addresses stay
 * resident in L2, so gathering a candidate twice is cheaper than writing it to
 * a staging buffer and copying it back. Measured on an 8000-atom step, the two
 * passes together take 0.56 ms against 2.09 ms for the staged form.
 *
 * The cutoff filter and the model-level type exclusion are one predicate, so
 * an excluded edge is never allocated rather than allocated and masked. That
 * is what the sort-based path achieved by moving masked edges into a suffix
 * outside every row, and it is exactly equivalent for a consumer that reads
 * the row pointers.
 *
 * Two masked edges terminate the payload so that the exported graph never
 * observes an empty edge axis.
 *
 * @param topology Cached skin topology, destination-grouped.
 * @param coord Extended coordinates, length ``3 * nall``, in the index space
 *   of ``topology.source_ext``.
 * @param atype Node types, length ``topology.node_count``; read only when a
 *   keep table is supplied.
 * @param keep_table Flat ``(ntypes + 1)^2`` type-pair keep table, or null.
 * @param ntypes Number of real atom types.
 * @param rcut Model cutoff.
 * @param edge_vec_fp32 Whether the payload carries float32 displacements.
 * @param with_source_csr Whether the source-major permutation is needed.
 * @param device Target device for the returned tensors.
 * @param scratch Reused count and offset storage.
 *
 * @return The graph pack, with an identity destination permutation.
 */
template <typename VALUETYPE>
inline GraphTensorPack assembleGraph(const SkinTopology& topology,
                                     const VALUETYPE* coord,
                                     const std::int64_t* atype,
                                     const int* keep_table,
                                     const int ntypes,
                                     const double rcut,
                                     const bool edge_vec_fp32,
                                     const bool with_source_csr,
                                     const torch::Device& device,
                                     GraphAssemblyScratch& scratch) {
  const std::int64_t node_count = topology.node_count;
  const double rcut_squared = rcut * rcut;
  const std::int32_t* source_ext = topology.source_ext.data();
  const std::int32_t* source_node = topology.source_node.data();
  const std::int64_t* skin_row_ptr = topology.row_ptr.data();

  // === Step 1. Count the survivors of each node ===
  scratch.row_ptr.resize(node_count + 1);
  std::int64_t* row_ptr = scratch.row_ptr.data();
  at::parallel_for(0, node_count, 1, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t node = begin; node < end; ++node) {
      std::int64_t kept = 0;
      double dx = 0;
      double dy = 0;
      double dz = 0;
      for (std::int64_t edge = skin_row_ptr[node];
           edge < skin_row_ptr[node + 1]; ++edge) {
        if (detail::edge_survives(coord, source_ext[edge], node, rcut_squared,
                                  dx, dy, dz) &&
            detail::pair_kept(keep_table, atype, source_node[edge], node,
                              ntypes)) {
          ++kept;
        }
      }
      row_ptr[node + 1] = kept;
    }
  });
  row_ptr[0] = 0;
  for (std::int64_t node = 0; node < node_count; ++node) {
    row_ptr[node + 1] += row_ptr[node];
  }
  const std::int64_t real_edges = row_ptr[node_count];
  const std::int64_t edge_count = real_edges + 2;

  // === Step 2. Allocate the payload and write it in place ===
  const auto index_options = torch::TensorOptions().dtype(torch::kInt64);
  const auto vec_options = torch::TensorOptions().dtype(
      edge_vec_fp32 ? torch::kFloat32 : torch::kFloat64);
  at::Tensor edge_index = torch::empty({2, edge_count}, index_options);
  at::Tensor edge_vec = torch::empty({edge_count, 3}, vec_options);
  at::Tensor edge_mask =
      torch::empty({edge_count}, torch::TensorOptions().dtype(torch::kBool));
  std::int64_t* source_out = edge_index.data_ptr<std::int64_t>();
  std::int64_t* destination_out = source_out + edge_count;
  bool* mask_out = edge_mask.data_ptr<bool>();
  float* vec_f32 = edge_vec_fp32 ? edge_vec.data_ptr<float>() : nullptr;
  double* vec_f64 = edge_vec_fp32 ? nullptr : edge_vec.data_ptr<double>();

  at::parallel_for(0, node_count, 1, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t node = begin; node < end; ++node) {
      std::int64_t cursor = row_ptr[node];
      double dx = 0;
      double dy = 0;
      double dz = 0;
      for (std::int64_t edge = skin_row_ptr[node];
           edge < skin_row_ptr[node + 1]; ++edge) {
        if (!detail::edge_survives(coord, source_ext[edge], node, rcut_squared,
                                   dx, dy, dz) ||
            !detail::pair_kept(keep_table, atype, source_node[edge], node,
                               ntypes)) {
          continue;
        }
        source_out[cursor] = source_node[edge];
        destination_out[cursor] = node;
        mask_out[cursor] = true;
        if (vec_f32 != nullptr) {
          vec_f32[cursor * 3] = static_cast<float>(dx);
          vec_f32[cursor * 3 + 1] = static_cast<float>(dy);
          vec_f32[cursor * 3 + 2] = static_cast<float>(dz);
        } else {
          vec_f64[cursor * 3] = dx;
          vec_f64[cursor * 3 + 1] = dy;
          vec_f64[cursor * 3 + 2] = dz;
        }
        ++cursor;
      }
    }
  });
  for (std::int64_t slot = real_edges; slot < edge_count; ++slot) {
    source_out[slot] = 0;
    destination_out[slot] = 0;
    mask_out[slot] = false;
    if (vec_f32 != nullptr) {
      vec_f32[slot * 3] = vec_f32[slot * 3 + 1] = vec_f32[slot * 3 + 2] = 0.0F;
    } else {
      vec_f64[slot * 3] = vec_f64[slot * 3 + 1] = vec_f64[slot * 3 + 2] = 0.0;
    }
  }

  // === Step 3. Publish the row pointers and the two permutations ===
  GraphTensorPack pack;
  pack.edge_index = edge_index.to(device);
  pack.edge_vec = edge_vec.to(device);
  pack.edge_mask = edge_mask.to(device);
  // Destination grouping is structural here, so the permutation is the
  // identity and is left empty: the consumers read the rows directly, and
  // materializing it would cost eight bytes per edge -- 157 MB on a
  // 125,000-atom system -- of pure redundancy, allocated and filled every step.
  pack.destination_order = torch::empty({0}, index_options).to(device);
  pack.destination_row_ptr =
      torch::from_blob(row_ptr, {node_count + 1}, index_options)
          .clone()
          .to(device);
  if (with_source_csr) {
    // The operator library owns the source view: its grouping is the same
    // threaded counting sort the Python graph builders use, and the payload
    // already satisfies its destination-major precondition.
    using BuildGraphCSR =
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(
            torch::Tensor, c10::SymInt, c10::SymInt);
    static const auto build_graph_csr =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("deepmd::build_graph_csr", "")
            .typed<BuildGraphCSR>();
    torch::Tensor unused_order;
    torch::Tensor unused_row_ptr;
    std::tie(unused_order, unused_row_ptr, pack.source_order,
             pack.source_row_ptr) =
        build_graph_csr.call(edge_index, c10::SymInt(node_count),
                             c10::SymInt(real_edges));
    pack.source_order = pack.source_order.to(device);
    pack.source_row_ptr = pack.source_row_ptr.to(device);
  } else {
    // The consumer never reads the source views; the axes still have to exist.
    pack.source_row_ptr = pack.destination_row_ptr;
    pack.source_order = pack.destination_order;
  }
  return pack;
}

}  // namespace deepmd
