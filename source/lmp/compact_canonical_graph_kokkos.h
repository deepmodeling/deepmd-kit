// SPDX-License-Identifier: LGPL-3.0-or-later
// The compact canonical graph builder is available when the LAMMPS Kokkos
// package is enabled.
#ifdef LMP_KOKKOS

#ifndef LMP_COMPACT_CANONICAL_GRAPH_KOKKOS_H
#define LMP_COMPACT_CANONICAL_GRAPH_KOKKOS_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "kokkos_type.h"
#include "neigh_list_kokkos.h"
#include "neighbor.h"
#include "pointers.h"

namespace LAMMPS_NS {

// Device-resident model node set and compact canonical neighbor graph.
//
// A ``.pt2`` artifact frozen with the compact canonical lower consumes a
// dual-CSR topology over a contiguous range of model nodes: uint32 source
// indices, float32 edge vectors, an int64 row pointer per node in each
// direction, and a permutation that reorders the destination-major edge list
// into source-major order. This component derives all of it from the Kokkos
// device neighbor list and leaves it on the device, so a pair style hands the
// model raw device pointers and never stages the graph through the host.
//
// The node set is the model's view of the atoms. Atom types that map to no
// model element (NULL in ``pair_coeff``) are virtual: they carry no node, so
// the surviving atoms are compacted into a contiguous node index range, which
// degenerates to the identity when no type maps to NULL. A single rank folds
// every ghost onto the local atom that owns it and works on the minimum-image
// node set, which requires the box to be thicker than twice the cutoff along
// every periodic direction. Domain decomposition instead gives every real
// ghost a node of its own -- the extended node set -- and leaves the pair
// style to fold the ghost outputs onto their owners by reverse communication.
//
// build() is one function, and public, because CUDA forbids extended device
// lambdas inside non-public members: its stages cannot be split into private
// helpers.
template <class DeviceType>
class CompactCanonicalGraphKokkos : protected Pointers {
 public:
  typedef ArrayTypes<DeviceType> AT;

  CompactCanonicalGraphKokkos(class LAMMPS* lmp)
      : Pointers(lmp),
        nloc_model(0),
        nnode_model(0),
        has_null_types(false),
        storage_count(0),
        execution_space(ExecutionSpaceFromDevice<DeviceType>::space),
        extended_nodes(false),
        edge_capacity(0) {}

  // Cache the LAMMPS type (1-based) -> model type map on the device and select
  // the node-set mode; ``extended`` gives every real ghost a node of its own.
  // The map comes from the pair style's coeff(), so this runs once the styles
  // know their model, before the first graph is built.
  void setup(const std::vector<int>& type_idx_map, bool extended) {
    host_type_map = type_idx_map;
    extended_nodes = extended;
    has_null_types = false;
    const int ntypes = static_cast<int>(host_type_map.size());
    d_type_map =
        Kokkos::View<int*, DeviceType>("compact_canonical:type_map", ntypes);
    auto h_type_map = Kokkos::create_mirror_view(d_type_map);
    for (int t = 0; t < ntypes; ++t) {
      if (host_type_map[t] < 0) {
        has_null_types = true;  // some LAMMPS type is a virtual (NULL) atom
      }
      h_type_map(t) = host_type_map[t];
    }
    Kokkos::deep_copy(d_type_map, h_type_map);
  }

  // Rebuild the atom -> node compaction, which only moves when the neighbor
  // list is rebuilt or the atom count outgrows the maps. A pair style that
  // builds its own graph over the same node set calls this directly; build()
  // calls it for the compact canonical graph.
  void refresh_nodes() {
    const int nlocal = atom->nlocal;
    const int nall = atom->nlocal + atom->nghost;

    if (neighbor->ago != 0 && (int)k_loc2model.extent(0) >= nall) {
      return;
    }
    if ((int)k_candidate_to_model.extent(0) < nall) {
      k_candidate_to_model =
          DAT::tdual_int_1d("compact_canonical:candidate_to_model", nall);
    }
    if ((int)k_loc2model.extent(0) < nall) {
      k_loc2model = DAT::tdual_int_1d("compact_canonical:loc2model", nall);
      k_model2loc = DAT::tdual_int_1d("compact_canonical:model2loc", nall);
    }
    atomKK->sync(Host, TAG_MASK | TYPE_MASK);
    auto h_loc2model = k_loc2model.view_host();
    auto h_model2loc = k_model2loc.view_host();
    const int* lmp_type = atom->type;
    int m = 0;
    for (int i = 0; i < nlocal; ++i) {
      if (host_type_map[lmp_type[i] - 1] >= 0) {
        h_loc2model(i) = m;
        h_model2loc(m) = i;
        ++m;
      } else {
        h_loc2model(i) = -1;
      }
    }
    nloc_model = m;
    for (int j = nlocal; j < nall; ++j) {
      if (extended_nodes && host_type_map[lmp_type[j] - 1] >= 0) {
        h_loc2model(j) = m;
        h_model2loc(m) = j;
        ++m;
      } else {
        h_loc2model(j) = -1;
      }
    }
    nnode_model = m;

    // Resolve each candidate atom to its model node once, on the host. In the
    // folded representation a ghost contributes to the node of the local atom
    // that owns it, so the resolution is a composition of the ownership map
    // with the model map; the extended representation gives ghosts their own
    // nodes and the composition degenerates to the model map. Collapsing it
    // here leaves the device traversal, which visits every candidate of every
    // center, with a single gather.
    auto h_candidate_to_model = k_candidate_to_model.view_host();
    if (extended_nodes) {
      for (int j = 0; j < nall; ++j) {
        h_candidate_to_model(j) = h_loc2model(j);
      }
    } else {
      for (int j = 0; j < nall; ++j) {
        const int owner = (j < nlocal) ? j : atom->map(atom->tag[j]);
        h_candidate_to_model(j) = owner < 0 ? -1 : h_loc2model(owner);
      }
    }
    k_candidate_to_model.template modify<LMPHostType>();
    k_candidate_to_model.template sync<DeviceType>();
    d_candidate_to_model = k_candidate_to_model.template view<DeviceType>();
    k_loc2model.template modify<LMPHostType>();
    k_loc2model.template sync<DeviceType>();
    d_loc2model = k_loc2model.template view<DeviceType>();
    k_model2loc.template modify<LMPHostType>();
    k_model2loc.template sync<DeviceType>();
    d_model2loc = k_model2loc.template view<DeviceType>();
  }

  // Build the compact canonical graph of the current configuration from the
  // Kokkos full neighbor list. Bond vectors are the center-to-neighbor
  // displacements divided by ``dist_unit_cvt_factor``, the same conversion the
  // pair style applies to coordinates it hands the model.
  void build(class NeighList* list,
             double cutoff,
             double dist_unit_cvt_factor) {
    refresh_nodes();

    auto* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
    const int inum = k_list->inum;
    auto d_numneigh = k_list->d_numneigh;
    auto d_neighbors = k_list->d_neighbors;
    auto d_ilist = k_list->d_ilist;
    auto loc2model = d_loc2model;
    auto candidate_to_model = d_candidate_to_model;
    auto model2loc = d_model2loc;
    const double cutsq = cutoff * cutoff;
    const double inv_dist = 1.0 / dist_unit_cvt_factor;
    const int node_count_int = nnode_model;
    const std::size_t node_count = static_cast<std::size_t>(node_count_int);
    const int nall = atom->nlocal + atom->nghost;

    // === Node types in the artifact's index layout ===
    atomKK->sync(execution_space, TYPE_MASK);
    auto type = atomKK->k_type.template view<DeviceType>();
    auto type_map = d_type_map;
    if ((int)d_model_type.extent(0) < node_count_int) {
      d_model_type = Kokkos::View<std::int64_t*, DeviceType>(
          "compact_canonical:model_type", nall);
    }
    auto model_type = d_model_type;
    Kokkos::parallel_for(
        "compact_canonical:node_type",
        Kokkos::RangePolicy<DeviceType>(0, node_count_int),
        KOKKOS_LAMBDA(const int m) {
          model_type(m) = type_map(type(model2loc(m)) - 1);
        });

    atomKK->sync(execution_space, X_MASK);
    auto x = atomKK->k_x.template view<DeviceType>();

    if (d_destination_row_ptr.extent(0) < node_count + 1) {
      d_destination_row_ptr = Kokkos::View<std::int64_t*, DeviceType>(
          "compact_canonical:destination_row_ptr", node_count + 1);
      d_source_counts = Kokkos::View<std::uint32_t*, DeviceType>(
          "compact_canonical:source_counts", node_count);
      d_source_row_ptr = Kokkos::View<std::int64_t*, DeviceType>(
          "compact_canonical:source_row_ptr", node_count + 1);
      d_source_cursor = Kokkos::View<std::uint32_t*, DeviceType>(
          "compact_canonical:source_cursor", node_count);
    }
    Kokkos::deep_copy(d_destination_row_ptr, std::int64_t{0});
    Kokkos::deep_copy(d_source_counts, std::uint32_t{0});
    if (node_count_int == 0) {
      storage_count = min_storage_edges;
      return;
    }
    auto destination_row_ptr = d_destination_row_ptr;
    auto source_counts = d_source_counts;

    // === Destination-major CSR: per-node edge count, then its prefix sum ===
    Kokkos::parallel_for(
        "compact_canonical:count", Kokkos::RangePolicy<DeviceType>(0, inum),
        KOKKOS_LAMBDA(const int ii) {
          const int i = d_ilist(ii);
          const int mi = loc2model(i);
          if (mi < 0) {
            return;
          }
          const double xi = x(i, 0);
          const double yi = x(i, 1);
          const double zi = x(i, 2);
          const int jnum = d_numneigh(i);
          std::int64_t count = 0;
          for (int jj = 0; jj < jnum; ++jj) {
            const int j = d_neighbors(i, jj) & NEIGHMASK;
            const int mj = candidate_to_model(j);
            if (mj < 0) {
              continue;
            }
            const double dx = x(j, 0) - xi;
            const double dy = x(j, 1) - yi;
            const double dz = x(j, 2) - zi;
            if (dx * dx + dy * dy + dz * dz < cutsq) {
              ++count;
            }
          }
          destination_row_ptr(mi) = count;
        });

    Kokkos::parallel_scan(
        "compact_canonical:destination_scan",
        Kokkos::RangePolicy<DeviceType>(0, node_count_int),
        KOKKOS_LAMBDA(const int node, std::int64_t& update, const bool final) {
          const std::int64_t count = destination_row_ptr(node);
          if (final) {
            destination_row_ptr(node) = update;
          }
          update += count;
          if (final && node == node_count_int - 1) {
            destination_row_ptr(node_count_int) = update;
          }
        });
    std::int64_t edge_count = 0;
    Kokkos::deep_copy(edge_count,
                      Kokkos::subview(d_destination_row_ptr, node_count_int));
    storage_count = std::max<std::int64_t>(edge_count, min_storage_edges);
    if (static_cast<std::uint64_t>(storage_count) >
        std::numeric_limits<std::uint32_t>::max()) {
      error->one(FLERR,
                 "Compact canonical graph exceeds the uint32 edge-index range");
    }
    const std::size_t required = static_cast<std::size_t>(storage_count);
    if (edge_capacity < required) {
      // Thermal cutoff-count fluctuations are much smaller than the historical
      // 12.5% geometric-growth reserve. A 2% reserve avoids repeated allocation
      // while preventing unused edge storage from retaining several GiB at
      // billion-edge scale.
      const std::size_t slack = required / 50 + 64;
      if (required > std::numeric_limits<std::size_t>::max() - slack) {
        error->one(FLERR, "Compact canonical graph capacity overflows size_t");
      }
      edge_capacity = required + slack;
      d_source = Kokkos::View<std::uint32_t*, DeviceType>(
          "compact_canonical:source", edge_capacity);
      d_edge_vec = Kokkos::View<float*, DeviceType>(
          "compact_canonical:edge_vec", edge_capacity * 3);
      d_source_order = Kokkos::View<std::uint32_t*, DeviceType>(
          "compact_canonical:source_order", edge_capacity);
    }

    auto source = d_source;
    auto edge_vec = d_edge_vec;
    // === Destination-major fill, one warp per center ===
    // A thread-per-center fill writes each surviving edge at an offset private
    // to its center, so the lanes of a warp scatter their twelve-byte edge
    // vectors across thirty-two unrelated rows. Cooperating on one center
    // instead sends consecutive survivors to consecutive slots, which coalesces
    // the dominant store stream. Candidates are taken a warp at a time and each
    // lane writes at its exclusive prefix within the warp, so the edge order is
    // the candidate order a serial fill produces.
    constexpr int neighbor_lanes = 32;  // lanes cooperating on one center
    using team_policy = Kokkos::TeamPolicy<DeviceType>;
    using member_type = typename team_policy::member_type;
    using lane_scratch =
        Kokkos::View<int*, typename DeviceType::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using vector_scratch =
        Kokkos::View<float*, typename DeviceType::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const int scratch_bytes = lane_scratch::shmem_size(neighbor_lanes) +
                              vector_scratch::shmem_size(3 * neighbor_lanes);
    Kokkos::parallel_for(
        "compact_canonical:fill",
        team_policy(inum, neighbor_lanes)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
        KOKKOS_LAMBDA(const member_type& team) {
          const int i = d_ilist(team.league_rank());
          const int mi = loc2model(i);
          if (mi < 0) {
            return;
          }
          lane_scratch node(team.team_scratch(0), neighbor_lanes);
          vector_scratch vec(team.team_scratch(0), 3 * neighbor_lanes);

          const double xi = x(i, 0);
          const double yi = x(i, 1);
          const double zi = x(i, 2);
          const int jnum = d_numneigh(i);
          const int lane = team.team_rank();
          std::int64_t edge = destination_row_ptr(mi);
          for (int base = 0; base < jnum; base += neighbor_lanes) {
            const int jj = base + lane;
            int mj = -1;
            if (jj < jnum) {
              const int j = d_neighbors(i, jj) & NEIGHMASK;
              mj = candidate_to_model(j);
              if (mj >= 0) {
                const double dx = x(j, 0) - xi;
                const double dy = x(j, 1) - yi;
                const double dz = x(j, 2) - zi;
                if (dx * dx + dy * dy + dz * dz < cutsq) {
                  vec(3 * lane + 0) = static_cast<float>(dx * inv_dist);
                  vec(3 * lane + 1) = static_cast<float>(dy * inv_dist);
                  vec(3 * lane + 2) = static_cast<float>(dz * inv_dist);
                } else {
                  mj = -1;
                }
              }
            }
            node(lane) = mj;
            team.team_barrier();

            // Compact the survivors of this warp of candidates: consecutive
            // survivors take consecutive slots, so the stores of a warp fall in
            // one contiguous span of the edge arrays.
            std::int64_t kept = 0;
            Kokkos::parallel_scan(
                Kokkos::TeamThreadRange(team, neighbor_lanes),
                [&](const int slot, std::int64_t& offset, const bool final) {
                  const int target = node(slot);
                  if (final && target >= 0) {
                    const std::int64_t position = edge + offset;
                    source(position) = static_cast<std::uint32_t>(target);
                    edge_vec(3 * position + 0) = vec(3 * slot + 0);
                    edge_vec(3 * position + 1) = vec(3 * slot + 1);
                    edge_vec(3 * position + 2) = vec(3 * slot + 2);
                    Kokkos::atomic_fetch_add(&source_counts(target),
                                             std::uint32_t{1});
                  }
                  offset += target >= 0 ? 1 : 0;
                },
                kept);
            edge += kept;
            team.team_barrier();
          }
        });

    // === Source-major CSR and the permutation into source order ===
    auto source_row_ptr = d_source_row_ptr;
    Kokkos::parallel_scan(
        "compact_canonical:source_scan",
        Kokkos::RangePolicy<DeviceType>(0, node_count_int),
        KOKKOS_LAMBDA(const int node, std::int64_t& update, const bool final) {
          const std::int64_t count =
              static_cast<std::int64_t>(source_counts(node));
          if (final) {
            source_row_ptr(node) = update;
          }
          update += count;
          if (final && node == node_count_int - 1) {
            source_row_ptr(node_count_int) = update;
          }
        });
    auto source_cursor = d_source_cursor;
    Kokkos::parallel_for(
        "compact_canonical:source_cursor",
        Kokkos::RangePolicy<DeviceType>(0, node_count_int),
        KOKKOS_LAMBDA(const int node) {
          source_cursor(node) =
              static_cast<std::uint32_t>(source_row_ptr(node));
        });
    auto source_order = d_source_order;
    Kokkos::parallel_for(
        "compact_canonical:source_scatter",
        Kokkos::RangePolicy<DeviceType, Kokkos::IndexType<std::int64_t>>(
            0, edge_count),
        KOKKOS_LAMBDA(const std::int64_t edge) {
          const auto position = Kokkos::atomic_fetch_add(
              &source_cursor(source(edge)), std::uint32_t{1});
          source_order(position) = static_cast<std::uint32_t>(edge);
        });

    // === Guard edges past the physical end of the graph ===
    if (storage_count > edge_count) {
      Kokkos::parallel_for(
          "compact_canonical:guards",
          Kokkos::RangePolicy<DeviceType, Kokkos::IndexType<std::int64_t>>(
              edge_count, storage_count),
          KOKKOS_LAMBDA(const std::int64_t edge) {
            source(edge) = std::uint32_t{0};
            edge_vec(3 * edge + 0) = 0.0f;
            edge_vec(3 * edge + 1) = 0.0f;
            edge_vec(3 * edge + 2) = 0.0f;
            source_order(edge) = static_cast<std::uint32_t>(edge);
          });
    }
  }

  // === Node set, valid after refresh_nodes() ===
  int nloc_model;   // real local model nodes; the energy is summed over these
  int nnode_model;  // total model nodes (== nloc_model folded; + ghost
                    // extended)
  bool has_null_types;  // some LAMMPS type maps to no model element
  // LAMMPS type (1-based) -> model type, resident on the device.
  Kokkos::View<int*, DeviceType> d_type_map;
  DAT::tdual_int_1d k_model2loc;      // (nall) model node index -> atom index
  typename AT::t_int_1d d_loc2model;  // (nall) atom -> model node index, or -1
  typename AT::t_int_1d d_model2loc;
  // (nall) candidate atom -> model node index, or -1, with a ghost already
  // folded onto its owner when the node set is not extended.
  typename AT::t_int_1d d_candidate_to_model;

  // === Compact canonical artifact, valid after build() ===
  // The edge arrays are padded to ``storage_count`` rows: the traced program
  // declares its edge axis with a lower bound of two, so a graph with fewer
  // physical edges is completed with guard rows that carry a zero bond vector
  // on node zero.
  std::int64_t storage_count;
  Kokkos::View<std::int64_t*, DeviceType> d_model_type;  // (nnode_model)
  Kokkos::View<std::uint32_t*, DeviceType> d_source;     // (storage_count)
  Kokkos::View<float*, DeviceType> d_edge_vec;           // (3 * storage_count)
  Kokkos::View<std::int64_t*, DeviceType> d_destination_row_ptr;
  Kokkos::View<std::int64_t*, DeviceType> d_source_row_ptr;
  Kokkos::View<std::uint32_t*, DeviceType> d_source_order;

 private:
  // Shortest edge axis the traced program accepts.
  static constexpr std::int64_t min_storage_edges = 2;

  ExecutionSpace execution_space;
  std::vector<int> host_type_map;
  bool extended_nodes;
  std::size_t edge_capacity;

  DAT::tdual_int_1d k_loc2model;
  DAT::tdual_int_1d k_candidate_to_model;
  // Per-node counts are bounded by the LAMMPS neighbor limit. CSR offsets
  // remain int64 so the compact graph retains its full global edge range.
  Kokkos::View<std::uint32_t*, DeviceType> d_source_counts;
  Kokkos::View<std::uint32_t*, DeviceType> d_source_cursor;
};

}  // namespace LAMMPS_NS

#endif

#endif  // LMP_KOKKOS
