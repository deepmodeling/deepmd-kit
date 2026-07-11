// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef LMP_KOKKOS
#include "pair_deepmd_kokkos.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

using namespace LAMMPS_NS;

template <class DeviceType>
PairDeepMDKokkos<DeviceType>::PairDeepMDKokkos(LAMMPS* lmp)
    : PairDeepMD(lmp),
      has_null_types(false),
      multi_rank(false),
      nloc_model(0),
      nnode_model(0),
      edge_capacity(0),
      edge_vec_fp32(false),
      canonical_graph(false),
      device_path_ok(false),
      reverse_virial(false),
      reverse_used_host(false) {
  respa_enable = 0;
  kokkosable = 1;
  atomKK = (AtomKokkos*)atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  reverse_comm_device = 1;
}

template <class DeviceType>
PairDeepMDKokkos<DeviceType>::~PairDeepMDKokkos() {
  if (copymode) {
    return;
  }
  memoryKK->destroy_kokkos(k_eatom, eatom);
}

template <class DeviceType>
int PairDeepMDKokkos<DeviceType>::pack_reverse_comm(int n,
                                                    int first,
                                                    double* buf) {
  if (reverse_virial) {
    auto h_reverse = k_reverse_virial.view_host();
    int m = 0;
    const int last = first + n;
    for (int i = first; i < last; ++i) {
      for (int k = 0; k < 9; ++k) {
        buf[m++] = h_reverse(9 * i + k);
      }
    }
    return m;
  }
  reverse_used_host = true;
  atomKK->sync(Host, F_MASK);
  double** f = atom->f;
  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; ++i) {
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
  }
  return m;
}

template <class DeviceType>
void PairDeepMDKokkos<DeviceType>::unpack_reverse_comm(int n,
                                                       int* list,
                                                       double* buf) {
  if (reverse_virial) {
    k_reverse_virial.modify_host();
    auto h_reverse = k_reverse_virial.view_host();
    int m = 0;
    for (int i = 0; i < n; ++i) {
      const int j = list[i];
      for (int k = 0; k < 9; ++k) {
        h_reverse(9 * j + k) += buf[m++];
      }
    }
    return;
  }
  reverse_used_host = true;
  atomKK->sync(Host, F_MASK);
  double** f = atom->f;
  int m = 0;
  for (int i = 0; i < n; ++i) {
    const int j = list[i];
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
  }
  atomKK->modified(Host, F_MASK);
}

template <class DeviceType>
int PairDeepMDKokkos<DeviceType>::pack_reverse_comm_kokkos(
    int n, int first, DAT::tdual_double_1d& buf) {
  auto d_buf = buf.template view<DeviceType>();
  if (reverse_virial) {
    auto reverse_virial_data = k_reverse_virial.template view<DeviceType>();
    const int first_i = first;
    Kokkos::parallel_for(
        "deepmd/kk:pack_rev_virial", Kokkos::RangePolicy<DeviceType>(0, n),
        KOKKOS_LAMBDA(const int i) {
          for (int k = 0; k < 9; ++k) {
            d_buf(9 * i + k) = reverse_virial_data(9 * (first_i + i) + k);
          }
        });
    return n * 9;
  }
  auto f = atomKK->k_f.template view<DeviceType>();
  const int first_i = first;
  Kokkos::parallel_for(
      "deepmd/kk:pack_rev", Kokkos::RangePolicy<DeviceType>(0, n),
      KOKKOS_LAMBDA(const int i) {
        d_buf(3 * i + 0) = f(first_i + i, 0);
        d_buf(3 * i + 1) = f(first_i + i, 1);
        d_buf(3 * i + 2) = f(first_i + i, 2);
      });
  return n * 3;
}

template <class DeviceType>
void PairDeepMDKokkos<DeviceType>::unpack_reverse_comm_kokkos(
    int n, DAT::tdual_int_1d list, DAT::tdual_double_1d& buf) {
  auto d_buf = buf.template view<DeviceType>();
  auto d_list = list.template view<DeviceType>();
  if (reverse_virial) {
    k_reverse_virial.template modify<DeviceType>();
    auto reverse_virial_data = k_reverse_virial.template view<DeviceType>();
    Kokkos::parallel_for(
        "deepmd/kk:unpack_rev_virial", Kokkos::RangePolicy<DeviceType>(0, n),
        KOKKOS_LAMBDA(const int i) {
          const int j = d_list(i);
          for (int k = 0; k < 9; ++k) {
            reverse_virial_data(9 * j + k) += d_buf(9 * i + k);
          }
        });
    return;
  }
  auto f = atomKK->k_f.template view<DeviceType>();
  Kokkos::parallel_for(
      "deepmd/kk:unpack_rev", Kokkos::RangePolicy<DeviceType>(0, n),
      KOKKOS_LAMBDA(const int i) {
        const int j = d_list(i);
        f(j, 0) += d_buf(3 * i + 0);
        f(j, 1) += d_buf(3 * i + 1);
        f(j, 2) += d_buf(3 * i + 2);
      });
}

template <class DeviceType>
void PairDeepMDKokkos<DeviceType>::init_style() {
  // Base setup and the full neighbor-list request.
  PairDeepMD::init_style();

#ifndef DP_USE_CXX_API
  error->all(FLERR,
             "pair style deepmd/kk requires the C++ API build of DeePMD "
             "(compute_edges_gpu is unavailable through the C API).");
#endif
  // The device edge path requires a GPU execution space and a single model.
  if (std::is_same<DeviceType, LMPHostType>::value) {
    error->all(FLERR, "pair style deepmd/kk runs on the GPU backend only.");
  }
  device_path_ok = deep_pot.supports_device_edge_inference();
  if (!device_path_ok) {
    error->all(
        FLERR,
        "pair style deepmd/kk requires an edge-input or graph-input .pt2 "
        "artifact; use pair style deepmd for an nlist artifact.");
  }
  // Domain decomposition uses the extended (local + ghost) node set: the model
  // computes per-node forces and the reverse communication folds the ghost
  // forces onto their owners. A single rank uses the folded minimum-image node
  // set. Message-passing models additionally receive communication metadata
  // through compute_edges_gpu.
  multi_rank = (comm->nprocs > 1);
  if (numb_models != 1) {
    error->all(FLERR, "pair style deepmd/kk does not support model deviation.");
  }
  // A local edge graph folds ghost neighbours onto their local owner through
  // the atom map; without it the fold returns -1 and corrupts the graph.
  if (atom->map_style == Atom::MAP_NONE) {
    error->all(FLERR,
               "pair style deepmd/kk needs an atom map; add 'atom_modify map "
               "yes' to the input.");
  }
  // Runtime frame (fparam) and per-atom (aparam) parameters are threaded to the
  // device edge path in compute(); only a runtime charge/spin override is not,
  // as compute_edges_gpu draws charge_spin from the model's stored default.
  if (!charge_spin.empty()) {
    error->all(FLERR,
               "pair style deepmd/kk uses the model's stored default "
               "charge_spin; a runtime charge_spin is not supported.");
  }

  // Route the base full request to the Kokkos device neighbor build.
  auto request = neighbor->find_request(this);
  request->set_kokkos_device(std::is_same<DeviceType, LMPDeviceType>::value);
  request->set_kokkos_host(false);
  request->enable_full();
  edge_vec_fp32 = deep_pot.uses_fp32_edge_vectors();
  canonical_graph = deep_pot.uses_canonical_graph_inference();
  // Force exchange transfers three values per atom. Centroid per-atom virial
  // uses nine values even though the Kokkos full-list request runs newton off,
  // so comm_reverse_off reserves the classic host buffer for the wider mode.
  comm_reverse = 3;
  comm_reverse_off = 9;

  // Cache the LAMMPS-type -> model-type map on the device (indexed by
  // type - 1); type_idx_map is populated by the base coeff().
  const int ntypes = static_cast<int>(type_idx_map.size());
  d_type_map = Kokkos::View<int*, DeviceType>("deepmd/kk:type_map", ntypes);
  auto h_type_map = Kokkos::create_mirror_view(d_type_map);
  has_null_types = false;
  for (int t = 0; t < ntypes; ++t) {
    if (type_idx_map[t] < 0) {
      has_null_types = true;  // some LAMMPS type is a virtual (NULL) atom
    }
    h_type_map(t) = type_idx_map[t];
  }
  Kokkos::deep_copy(d_type_map, h_type_map);
}

#ifdef DP_USE_CXX_API
template <class DeviceType>
void PairDeepMDKokkos<DeviceType>::prepare_model_nodes() {
  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;

  if (neighbor->ago == 0 || (int)k_loc2model.extent(0) < nall) {
    if ((int)k_owner.extent(0) < nall) {
      k_owner = DAT::tdual_int_1d("deepmd/kk:owner", nall);
    }
    if ((int)k_loc2model.extent(0) < nall) {
      k_loc2model = DAT::tdual_int_1d("deepmd/kk:loc2model", nall);
      k_model2loc = DAT::tdual_int_1d("deepmd/kk:model2loc", nall);
    }
    atomKK->sync(Host, TAG_MASK | TYPE_MASK);
    auto h_owner = k_owner.view_host();
    for (int jj = 0; jj < nall; ++jj) {
      h_owner(jj) = (jj < nlocal) ? jj : atom->map(atom->tag[jj]);
    }
    auto h_loc2model = k_loc2model.view_host();
    auto h_model2loc = k_model2loc.view_host();
    const int* lmp_type = atom->type;
    int m = 0;
    for (int i = 0; i < nlocal; ++i) {
      if (type_idx_map[lmp_type[i] - 1] >= 0) {
        h_loc2model(i) = m;
        h_model2loc(m) = i;
        ++m;
      } else {
        h_loc2model(i) = -1;
      }
    }
    nloc_model = m;
    for (int j = nlocal; j < nall; ++j) {
      if (multi_rank && type_idx_map[lmp_type[j] - 1] >= 0) {
        h_loc2model(j) = m;
        h_model2loc(m) = j;
        ++m;
      } else {
        h_loc2model(j) = -1;
      }
    }
    nnode_model = m;
    k_owner.template modify<LMPHostType>();
    k_owner.template sync<DeviceType>();
    d_owner = k_owner.template view<DeviceType>();
    k_loc2model.template modify<LMPHostType>();
    k_loc2model.template sync<DeviceType>();
    d_loc2model = k_loc2model.template view<DeviceType>();
    k_model2loc.template modify<LMPHostType>();
    k_model2loc.template sync<DeviceType>();
    d_model2loc = k_model2loc.template view<DeviceType>();
  }

  atomKK->sync(execution_space, TYPE_MASK);
  auto type = atomKK->k_type.template view<DeviceType>();
  auto type_map = d_type_map;
  auto model2loc = d_model2loc;
  if (canonical_graph) {
    if ((int)d_model_type_i64.extent(0) < nnode_model) {
      d_model_type_i64 = Kokkos::View<std::int64_t*, DeviceType>(
          "deepmd/kk:model_type_i64", nall);
    }
    auto model_type = d_model_type_i64;
    Kokkos::parallel_for(
        "deepmd/kk:mtype_i64", Kokkos::RangePolicy<DeviceType>(0, nnode_model),
        KOKKOS_LAMBDA(const int m) {
          model_type(m) = type_map(type(model2loc(m)) - 1);
        });
  } else {
    if ((int)d_model_type.extent(0) < nnode_model) {
      d_model_type =
          Kokkos::View<int*, DeviceType>("deepmd/kk:model_type", nall);
    }
    auto model_type = d_model_type;
    Kokkos::parallel_for(
        "deepmd/kk:mtype", Kokkos::RangePolicy<DeviceType>(0, nnode_model),
        KOKKOS_LAMBDA(const int m) {
          model_type(m) = type_map(type(model2loc(m)) - 1);
        });
  }
}

template <class DeviceType>
int PairDeepMDKokkos<DeviceType>::build_edges_device() {
  const int nlocal = atom->nlocal;
  prepare_model_nodes();

  // === Neighbor list and atom views on the device ===
  NeighListKokkos<DeviceType>* k_list =
      static_cast<NeighListKokkos<DeviceType>*>(list);
  const int inum = k_list->inum;
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  atomKK->sync(execution_space, X_MASK);
  auto x = atomKK->k_x.template view<DeviceType>();

  const double cut = cutoff;
  const double cutsq = cut * cut;
  const bool multi = multi_rank;
  auto owner = d_owner;
  auto loc2model = d_loc2model;
  auto model2loc = d_model2loc;

  if ((int)d_edge_offset.extent(0) < nlocal + 1) {
    d_edge_offset = Kokkos::View<std::int64_t*, DeviceType>(
        "deepmd/kk:edge_offset", nlocal + 1);
  }
  auto edge_offset = d_edge_offset;

  // === Pass 1: per-center edge count (0 for a virtual center) ===
  // A neighbour is the ghost's owner node (folded) or the ghost's own node
  // (extended); a neighbour imaging a virtual atom is skipped.
  Kokkos::parallel_for(
      "deepmd/kk:count", Kokkos::RangePolicy<DeviceType>(0, inum),
      KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist(ii);
        if (loc2model(i) < 0) {
          edge_offset(i) = 0;
          return;
        }
        const double xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);
        const int jnum = d_numneigh(i);
        int c = 0;
        for (int jj = 0; jj < jnum; ++jj) {
          int j = d_neighbors(i, jj);
          j &= NEIGHMASK;
          if (loc2model(multi ? j : owner(j)) < 0) {
            continue;
          }
          const double dx = x(j, 0) - xi, dy = x(j, 1) - yi, dz = x(j, 2) - zi;
          if (dx * dx + dy * dy + dz * dz < cutsq) {
            ++c;
          }
        }
        edge_offset(i) = c;
      });

  // === Exclusive prefix sum of the counts -> edge offsets, total = nedge ===
  // An empty subdomain (nlocal == 0) has no edges and must not read the scan
  // sentinel, which the empty scan would leave uninitialized.
  std::int64_t nedge_total = 0;
  if (nlocal > 0) {
    Kokkos::parallel_scan(
        "deepmd/kk:scan", Kokkos::RangePolicy<DeviceType>(0, nlocal),
        KOKKOS_LAMBDA(const int i, std::int64_t& update, const bool final) {
          const std::int64_t c = edge_offset(i);
          if (final) {
            edge_offset(i) = update;
          }
          update += c;
          if (final && i == nlocal - 1) {
            edge_offset(nlocal) = update;
          }
        });
    Kokkos::deep_copy(nedge_total, Kokkos::subview(d_edge_offset, nlocal));
  }
  if (nedge_total > std::numeric_limits<int>::max()) {
    error->one(
        FLERR,
        "The DeePMD Kokkos edge count exceeds the int32 graph-index limit");
  }
  const int nedge = static_cast<int>(nedge_total);

  // Keep the edge buffers allocated (non-null) even with no physical edges, so
  // an isolated-atom step still runs the model for its per-atom energy bias.
  const int want = nedge > 0 ? nedge : 1;
  if (edge_capacity < want) {
    const std::int64_t grown = static_cast<std::int64_t>(want) + want / 8 + 64;
    edge_capacity = static_cast<int>(grown > std::numeric_limits<int>::max()
                                         ? std::numeric_limits<int>::max()
                                         : grown);
    // Size in size_t: at multi-million-atom scale 3 * edge_capacity exceeds the
    // 32-bit range even though the edge count itself still fits int.
    d_edge_index = Kokkos::View<int*, DeviceType>(
        "deepmd/kk:edge_index", static_cast<std::size_t>(edge_capacity) * 2);
    if (edge_vec_fp32) {
      d_edge_vec_float = Kokkos::View<float*, DeviceType>(
          "deepmd/kk:edge_vec_float",
          static_cast<std::size_t>(edge_capacity) * 3);
      d_edge_vec = Kokkos::View<double*, DeviceType>();
    } else {
      d_edge_vec = Kokkos::View<double*, DeviceType>(
          "deepmd/kk:edge_vec", static_cast<std::size_t>(edge_capacity) * 3);
      d_edge_vec_float = Kokkos::View<float*, DeviceType>();
    }
  }
  auto edge_index = d_edge_index;
  auto edge_vec = d_edge_vec;
  auto edge_vec_float = d_edge_vec_float;
  const bool write_fp32_edge = edge_vec_fp32;
  const double inv_dist = 1.0 / dist_unit_cvt_factor;
  const std::int64_t nedge_l = nedge;

  // === Pass 2: emit model-space edges (src = neighbour node, dst = center
  // node; bond = x[j] - x[i]) ===
  Kokkos::parallel_for(
      "deepmd/kk:fill", Kokkos::RangePolicy<DeviceType>(0, inum),
      KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist(ii);
        const int mi = loc2model(i);
        if (mi < 0) {
          return;
        }
        const double xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);
        const int jnum = d_numneigh(i);
        std::int64_t e = edge_offset(i);
        for (int jj = 0; jj < jnum; ++jj) {
          int j = d_neighbors(i, jj);
          j &= NEIGHMASK;
          const int mj = loc2model(multi ? j : owner(j));
          if (mj < 0) {
            continue;
          }
          const double dx = x(j, 0) - xi, dy = x(j, 1) - yi, dz = x(j, 2) - zi;
          if (dx * dx + dy * dy + dz * dz < cutsq) {
            edge_index(e) = mj;            // src (model node)
            edge_index(nedge_l + e) = mi;  // dst (model node)
            if (write_fp32_edge) {
              edge_vec_float(3 * e + 0) = static_cast<float>(dx * inv_dist);
              edge_vec_float(3 * e + 1) = static_cast<float>(dy * inv_dist);
              edge_vec_float(3 * e + 2) = static_cast<float>(dz * inv_dist);
            } else {
              edge_vec(3 * e + 0) = dx * inv_dist;
              edge_vec(3 * e + 1) = dy * inv_dist;
              edge_vec(3 * e + 2) = dz * inv_dist;
            }
            ++e;
          }
        }
      });

  // Compacted coordinates in model-node order for edge-input models (the graph
  // lower ignores coordinates); only needed when virtual atoms compact them.
  if (has_null_types) {
    if ((int)d_coord_model.extent(0) < 3 * nnode_model) {
      d_coord_model = Kokkos::View<double*, DeviceType>("deepmd/kk:coord_model",
                                                        3 * nnode_model);
    }
    auto coord_model = d_coord_model;
    Kokkos::parallel_for(
        "deepmd/kk:coord", Kokkos::RangePolicy<DeviceType>(0, nnode_model),
        KOKKOS_LAMBDA(const int m) {
          const int i = model2loc(m);
          coord_model(3 * m + 0) = x(i, 0);
          coord_model(3 * m + 1) = x(i, 1);
          coord_model(3 * m + 2) = x(i, 2);
        });
  }
  return nedge;
}

template <class DeviceType>
std::int64_t PairDeepMDKokkos<DeviceType>::build_canonical_edges_device(
    CompactCanonicalGraphWorkspace<DeviceType>& workspace) {
  prepare_model_nodes();

  auto* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  const int inum = k_list->inum;
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;
  atomKK->sync(execution_space, X_MASK);
  auto x = atomKK->k_x.template view<DeviceType>();
  auto owner = d_owner;
  auto loc2model = d_loc2model;
  const bool multi = multi_rank;
  const double cutsq = cutoff * cutoff;
  const double inv_dist = 1.0 / dist_unit_cvt_factor;
  const int node_count_int = nnode_model;
  const std::size_t node_count = static_cast<std::size_t>(node_count_int);

  if (workspace.destination_row_ptr.extent(0) < node_count + 1) {
    workspace.destination_row_ptr = Kokkos::View<std::int64_t*, DeviceType>(
        "deepmd/kk:canonical_destination_row_ptr", node_count + 1);
    workspace.source_counts = Kokkos::View<std::int64_t*, DeviceType>(
        "deepmd/kk:canonical_source_counts", node_count);
    workspace.source_row_ptr = Kokkos::View<std::int64_t*, DeviceType>(
        "deepmd/kk:canonical_source_row_ptr", node_count + 1);
    workspace.source_cursor = Kokkos::View<std::int64_t*, DeviceType>(
        "deepmd/kk:canonical_source_cursor", node_count);
  }
  Kokkos::deep_copy(workspace.destination_row_ptr, std::int64_t{0});
  Kokkos::deep_copy(workspace.source_counts, std::int64_t{0});
  if (node_count_int == 0) {
    return 0;
  }
  auto destination_row_ptr = workspace.destination_row_ptr;
  auto source_counts = workspace.source_counts;

  Kokkos::parallel_for(
      "deepmd/kk:canonical_count", Kokkos::RangePolicy<DeviceType>(0, inum),
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
          int j = d_neighbors(i, jj) & NEIGHMASK;
          const int mj = loc2model(multi ? j : owner(j));
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
      "deepmd/kk:canonical_destination_scan",
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
  Kokkos::deep_copy(edge_count, Kokkos::subview(workspace.destination_row_ptr,
                                                node_count_int));
  const std::int64_t storage_count = std::max<std::int64_t>(edge_count, 2);
  const std::size_t required = static_cast<std::size_t>(storage_count);
  if (workspace.edge_capacity < required) {
    const std::size_t slack = required / 8 + 64;
    if (required > std::numeric_limits<std::size_t>::max() - slack) {
      error->one(FLERR, "Compact DPA1 graph capacity overflows size_t");
    }
    workspace.edge_capacity = required + slack;
    workspace.source = Kokkos::View<std::int64_t*, DeviceType>(
        "deepmd/kk:canonical_source", workspace.edge_capacity);
    workspace.edge_vec = Kokkos::View<float*, DeviceType>(
        "deepmd/kk:canonical_edge_vec", workspace.edge_capacity * 3);
    workspace.source_order = Kokkos::View<std::int64_t*, DeviceType>(
        "deepmd/kk:canonical_source_order", workspace.edge_capacity);
  }

  auto source = workspace.source;
  auto edge_vec = workspace.edge_vec;
  Kokkos::parallel_for(
      "deepmd/kk:canonical_fill", Kokkos::RangePolicy<DeviceType>(0, inum),
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
        std::int64_t edge = destination_row_ptr(mi);
        for (int jj = 0; jj < jnum; ++jj) {
          int j = d_neighbors(i, jj) & NEIGHMASK;
          const int mj = loc2model(multi ? j : owner(j));
          if (mj < 0) {
            continue;
          }
          const double dx = x(j, 0) - xi;
          const double dy = x(j, 1) - yi;
          const double dz = x(j, 2) - zi;
          if (dx * dx + dy * dy + dz * dz < cutsq) {
            source(edge) = static_cast<std::int64_t>(mj);
            edge_vec(3 * edge + 0) = static_cast<float>(dx * inv_dist);
            edge_vec(3 * edge + 1) = static_cast<float>(dy * inv_dist);
            edge_vec(3 * edge + 2) = static_cast<float>(dz * inv_dist);
            Kokkos::atomic_fetch_add(&source_counts(mj), std::int64_t{1});
            ++edge;
          }
        }
      });

  auto source_row_ptr = workspace.source_row_ptr;
  Kokkos::parallel_scan(
      "deepmd/kk:canonical_source_scan",
      Kokkos::RangePolicy<DeviceType>(0, node_count_int),
      KOKKOS_LAMBDA(const int node, std::int64_t& update, const bool final) {
        const std::int64_t count = source_counts(node);
        if (final) {
          source_row_ptr(node) = update;
        }
        update += count;
        if (final && node == node_count_int - 1) {
          source_row_ptr(node_count_int) = update;
        }
      });
  Kokkos::deep_copy(
      workspace.source_cursor,
      Kokkos::subview(workspace.source_row_ptr,
                      std::make_pair(std::int64_t{0}, static_cast<std::int64_t>(
                                                          node_count_int))));
  auto source_cursor = workspace.source_cursor;
  auto source_order = workspace.source_order;
  Kokkos::parallel_for(
      "deepmd/kk:canonical_source_scatter",
      Kokkos::RangePolicy<DeviceType, Kokkos::IndexType<std::int64_t>>(
          0, edge_count),
      KOKKOS_LAMBDA(const std::int64_t edge) {
        const auto position =
            Kokkos::atomic_fetch_add(&source_cursor(source(edge)), 1LL);
        source_order(position) = edge;
      });
  if (storage_count > edge_count) {
    Kokkos::parallel_for(
        "deepmd/kk:canonical_guards",
        Kokkos::RangePolicy<DeviceType, Kokkos::IndexType<std::int64_t>>(
            edge_count, storage_count),
        KOKKOS_LAMBDA(const std::int64_t edge) {
          source(edge) = 0;
          edge_vec(3 * edge + 0) = 0.0f;
          edge_vec(3 * edge + 1) = 0.0f;
          edge_vec(3 * edge + 2) = 0.0f;
          source_order(edge) = edge;
        });
  }
  return edge_count;
}
#endif

template <class DeviceType>
void PairDeepMDKokkos<DeviceType>::compute(int eflag, int vflag) {
#ifdef DP_USE_CXX_API
  if (!device_path_ok) {
    error->all(FLERR,
               "pair style deepmd/kk cannot execute this model input schema.");
  }
  ev_init(eflag, vflag);
  if (vflag_atom) {
    error->all(FLERR,
               "6-element atomic virial is not supported. Use compute "
               "centroid/stress/atom command for 9-element atomic virial.");
  }

  const int nlocal = atom->nlocal;
  // Per-atom energy is scattered on the device into a DualView that aliases the
  // base Pair ``eatom`` array; (re)allocate it here as the standard Kokkos
  // pair styles do. The centroid per-atom virial has no Kokkos device path, so
  // it is filled on the host below.
  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "deepmd/kk:eatom");
    d_eatom = k_eatom.template view<DeviceType>();
  }
  std::int64_t nedge = 0;
  if (canonical_graph) {
    nedge = build_canonical_edges_device(canonical_workspace);
  } else {
    nedge = build_edges_device();
  }
  const int nloc_m = nloc_model;    // local model nodes (energy)
  const int nnode_m = nnode_model;  // total model nodes (force / virial)

  // Energy is per local node; force / virial span the model node set, which is
  // the local atoms (folded) or local + real ghost atoms (extended, up to
  // nall). The two extents grow independently: under domain decomposition
  // ``nlocal`` and ``nall`` need not move together, so a shared guard could
  // leave the energy buffer short when ``nlocal`` grows while ``nall`` does
  // not.
  const int nall = atom->nlocal + atom->nghost;
  if ((int)d_atom_energy.extent(0) < nlocal) {
    d_atom_energy =
        Kokkos::View<double*, DeviceType>("deepmd/kk:atom_energy", nlocal);
  }
  if ((int)d_out_force.extent(0) < 3 * nall) {
    d_out_force =
        Kokkos::View<double*, DeviceType>("deepmd/kk:out_force", 3 * nall);
    d_atom_virial =
        Kokkos::View<double*, DeviceType>("deepmd/kk:atom_virial", 9 * nall);
  }
  if (cvflag_atom && multi_rank && (int)k_reverse_virial.extent(0) < 9 * nall) {
    k_reverse_virial =
        DAT::tdual_double_1d("deepmd/kk:reverse_virial", 9 * nall);
  }
  Kokkos::deep_copy(d_out_force, 0.0);
  Kokkos::deep_copy(d_atom_energy, 0.0);
  Kokkos::deep_copy(d_atom_virial, 0.0);

  atomKK->sync(execution_space, X_MASK | TYPE_MASK);
  auto x = atomKK->k_x.template view<DeviceType>();

  // Runtime frame (fparam) and per-atom (aparam) parameters, built per step
  // from the same sources as the standalone pair (compute / fix / ttm or a
  // uniform setting). Empty vectors fall back to the model's stored defaults.
  std::vector<double> aparam_step;
  if (do_compute_aparam) {
    make_aparam_from_compute(aparam_step);
  } else if (aparam.size() > 0) {
    make_uniform_aparam(aparam_step, aparam, nlocal);
  } else if (do_ttm) {
#ifdef USE_TTM
    if (dim_aparam > 0) {
      make_ttm_aparam(aparam_step);
    } else if (dim_fparam > 0) {
      make_ttm_fparam(fparam);
    }
#endif
  }
  if (do_compute_fparam) {
    make_fparam_from_compute(fparam);
  } else if (do_fix_fparam) {
    make_fparam_from_fix(fparam);
  }

  // ``aparam`` is built in LAMMPS local order; when virtual atoms drop nodes it
  // must be compacted into model-node order (the first ``nloc_model`` nodes) so
  // it aligns with the atoms the model consumes.
  if (has_null_types && dim_aparam > 0 && !aparam_step.empty()) {
    auto h_m2l = k_model2loc.view_host();
    std::vector<double> aparam_model(static_cast<std::size_t>(nloc_m) *
                                     dim_aparam);
    for (int m = 0; m < nloc_m; ++m) {
      const int i = h_m2l(m);
      for (int k = 0; k < dim_aparam; ++k) {
        aparam_model[static_cast<std::size_t>(m) * dim_aparam + k] =
            aparam_step[static_cast<std::size_t>(i) * dim_aparam + k];
      }
    }
    aparam_step.swap(aparam_model);
  }

  // Send/recv swap metadata for a message-passing model under domain
  // decomposition: ghost features are exchanged across ranks inside the forward
  // pass. It is passed only when the raw LAMMPS atom indices in the swap lists
  // match the model-node indices, i.e. when no virtual (NULL-type) atoms
  // compact the node set; otherwise the extended edge-input path is rejected.
  deepmd_compat::InputNlist comm_list;
  const deepmd_compat::InputNlist* comm_ptr = nullptr;
  if (multi_rank && !has_null_types) {
    comm_list = make_comm_nlist();
    comm_ptr = &comm_list;
  }

  if ((canonical_graph && nnode_m > 0) || nloc_m > 0 || comm_ptr != nullptr) {
    // Fully device-resident inference: raw device pointers in and out. The
    // edge buffers are produced on the Kokkos stream and consumed by the model
    // on PyTorch's stream, and the outputs flow back to the Kokkos scatter, so
    // the two runtimes are bracketed by explicit synchronization: fence the
    // Kokkos work before the model reads the edges, and synchronize the device
    // after so the scatter sees the finished model outputs.
    Kokkos::fence();
    // Coordinates are model-node order: the compacted buffer when virtual atoms
    // are present, else the local coordinates directly (the graph lower ignores
    // them; edge-input models consume them).
    const double* coord_ptr = has_null_types ? d_coord_model.data() : x.data();
    try {
      if (canonical_graph) {
        const std::int64_t storage_count = std::max<std::int64_t>(nedge, 2);
        auto& workspace = canonical_workspace;
        deep_pot.compute_canonical_graph_gpu(
            d_atom_energy.data(), d_out_force.data(), d_atom_virial.data(),
            d_model_type_i64.data(), workspace.source.data(),
            workspace.edge_vec.data(), workspace.destination_row_ptr.data(),
            workspace.source_row_ptr.data(), workspace.source_order.data(),
            nloc_m, nnode_m, storage_count);
      } else if (edge_vec_fp32) {
        deep_pot.compute_edges_gpu(
            d_atom_energy.data(), d_out_force.data(), d_atom_virial.data(),
            coord_ptr, d_model_type.data(), d_edge_index.data(),
            d_edge_vec_float.data(), nloc_m, static_cast<int>(nedge), fparam,
            aparam_step, nnode_m, comm_ptr);
      } else {
        deep_pot.compute_edges_gpu(
            d_atom_energy.data(), d_out_force.data(), d_atom_virial.data(),
            coord_ptr, d_model_type.data(), d_edge_index.data(),
            d_edge_vec.data(), nloc_m, static_cast<int>(nedge), fparam,
            aparam_step, nnode_m, comm_ptr);
      }
    } catch (deepmd_compat::deepmd_exception& e) {
      error->one(FLERR, e.what());
    }
  }

  // === Scatter the model-node forces onto their atoms ===
  // ``model2loc`` maps a model node back to its LAMMPS atom (the identity when
  // there are no virtual atoms); virtual atoms receive no contribution. For the
  // extended multi-domain set the nodes past ``nloc_m`` are ghosts, whose
  // forces are written to the ghost slots and folded onto their owners by the
  // reverse communication that the KOKKOS package (which forces 'newton off'
  // with a full list) would otherwise skip.
  // The scatter remains device-resident. If LAMMPS selects classic host
  // communication, the host pack/unpack methods synchronize the force DualView
  // and the completed fold is copied back once after all communication stages.
  auto model2loc = d_model2loc;
  const double fscale = scale[1][1] * force_unit_cvt_factor;
  reverse_virial = false;
  reverse_used_host = false;
  // The KOKKOS package runs 'newton off', so the integrator's force_clear only
  // zeros the local forces (f[0, nlocal)); the ghost slots f[nlocal, nall) are
  // left untouched. The extended scatter writes ghost slots and folds them onto
  // their owners by reverse communication, so those slots must be zeroed first,
  // or their contribution accumulates across steps.
  atomKK->sync(execution_space, F_MASK);
  auto f = atomKK->k_f.template view<DeviceType>();
  auto out_force = d_out_force;
  if (multi_rank) {
    Kokkos::parallel_for(
        "deepmd/kk:clear_ghost_f",
        Kokkos::RangePolicy<DeviceType>(nlocal, nall),
        KOKKOS_LAMBDA(const int m) {
          f(m, 0) = 0.0;
          f(m, 1) = 0.0;
          f(m, 2) = 0.0;
        });
  }
  Kokkos::parallel_for(
      "deepmd/kk:scatter_f", Kokkos::RangePolicy<DeviceType>(0, nnode_m),
      KOKKOS_LAMBDA(const int m) {
        const int i = model2loc(m);
        f(i, 0) += fscale * out_force(3 * m + 0);
        f(i, 1) += fscale * out_force(3 * m + 1);
        f(i, 2) += fscale * out_force(3 * m + 2);
      });
  atomKK->modified(execution_space, F_MASK);
  if (multi_rank) {
    comm->reverse_comm(this, 3);
    if (reverse_used_host) {
      atomKK->sync(execution_space, F_MASK);
    }
  }

  if (eflag_global) {
    auto atom_energy = d_atom_energy;
    double e_sum = 0.0;
    Kokkos::parallel_reduce(
        "deepmd/kk:esum", Kokkos::RangePolicy<DeviceType>(0, nloc_m),
        KOKKOS_LAMBDA(const int m, double& acc) { acc += atom_energy(m); },
        e_sum);
    eng_vdwl += scale[1][1] * e_sum * ener_unit_cvt_factor;
  }

  if (vflag_global) {
    // Sum the per-node 9-component virial and map to the LAMMPS global 6
    // (xx, yy, zz, xy, xz, yz), matching the standalone pair's index map. The
    // sum spans all nodes (local + extended ghost) so the reduction equals the
    // model's reduced virial for this rank's local-centered edges.
    auto atom_virial = d_atom_virial;
    const int comp[6] = {0, 4, 8, 3, 6, 7};
    for (int k = 0; k < 6; ++k) {
      const int off = comp[k];
      double vsum = 0.0;
      Kokkos::parallel_reduce(
          "deepmd/kk:vsum", Kokkos::RangePolicy<DeviceType>(0, nnode_m),
          KOKKOS_LAMBDA(const int m, double& acc) {
            acc += atom_virial(9 * m + off);
          },
          vsum);
      virial[k] += scale[1][1] * vsum * ener_unit_cvt_factor;
    }
  }

  if (eflag_atom) {
    auto atom_energy = d_atom_energy;
    auto eatom_v = d_eatom;
    const double escale = scale[1][1] * ener_unit_cvt_factor;
    Kokkos::deep_copy(d_eatom, 0.0);  // virtual atoms keep zero energy
    Kokkos::parallel_for(
        "deepmd/kk:eatom", Kokkos::RangePolicy<DeviceType>(0, nloc_m),
        KOKKOS_LAMBDA(const int m) {
          eatom_v(model2loc(m)) = escale * atom_energy(m);
        });
    k_eatom.template modify<DeviceType>();
    k_eatom.sync_host();
  }

  if (cvflag_atom) {
    // Centroid per-atom virial is reported on owned atoms. Contributions
    // carried by extended ghost nodes are folded to their owners explicitly
    // because the KOKKOS full-list path runs with newton pair disabled.
    auto h_av = Kokkos::create_mirror_view(d_atom_virial);
    Kokkos::deep_copy(h_av, d_atom_virial);
    auto h_m2l = k_model2loc.view_host();
    const double vscale = scale[1][1] * ener_unit_cvt_factor;
    const int map9[9] = {0, 4, 8, 3, 6, 7, 1, 2, 5};
    for (int m = 0; m < nloc_m; ++m) {
      const int ii = h_m2l(m);
      for (int k = 0; k < 9; ++k) {
        cvatom[ii][k] += vscale * h_av(9 * m + map9[k]);
      }
    }
    if (multi_rank) {
      reverse_virial = true;
      k_reverse_virial.modify_host();
      auto h_reverse = k_reverse_virial.view_host();
      Kokkos::deep_copy(h_reverse, 0.0);
      for (int m = nloc_m; m < nnode_m; ++m) {
        const int ii = h_m2l(m);
        for (int k = 0; k < 9; ++k) {
          h_reverse(9 * ii + k) = vscale * h_av(9 * m + map9[k]);
        }
      }
      k_reverse_virial.template sync<DeviceType>();
      comm->reverse_comm(this, 9);
      k_reverse_virial.sync_host();
      for (int i = 0; i < nlocal; ++i) {
        for (int k = 0; k < 9; ++k) {
          cvatom[i][k] += h_reverse(9 * i + k);
        }
      }
      reverse_virial = false;
    }
  }
#else
  PairDeepMD::compute(eflag, vflag);
#endif
}

namespace LAMMPS_NS {
template class PairDeepMDKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairDeepMDKokkos<LMPHostType>;
#endif
}  // namespace LAMMPS_NS

#endif  // LMP_KOKKOS
