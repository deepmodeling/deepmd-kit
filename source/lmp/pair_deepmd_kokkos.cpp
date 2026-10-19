// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef LMP_KOKKOS
#include "pair_deepmd_kokkos.h"

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
      compact_graph(lmp),
      multi_rank(false),
      edge_capacity(0),
      edge_vec_fp32(false),
      canonical_graph(false),
      canonical_source_csr(false),
      long_range(false),
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
    int n, int first, DeepMDKokkosCommBuffer& buf) {
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
    int n, DAT::tdual_int_1d list, DeepMDKokkosCommBuffer& buf) {
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

  // The device edge path requires a GPU execution space and a single model.
  if (std::is_same<DeviceType, LMPHostType>::value) {
    error->all(FLERR, "pair style deepmd/kk runs on the GPU backend only.");
  }
  // The constructor runs before LAMMPS confirms that KOKKOS is enabled, so
  // the virial of the graph is allocated here, once: its address stays fixed
  // from run to run, as the replay of a graph without halo nodes keys on it.
  if (!d_graph_virial.is_allocated()) {
    d_graph_virial =
        Kokkos::View<double[9], DeviceType>("deepmd/kk:graph_virial");
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
  // device edge path in compute().  A charge/spin condition needs no threading:
  // compute_edges_gpu draws it from the model, and settings() has already fixed
  // the model on the condition the pair_style line asked for.

  // Route the base full request to the Kokkos device neighbor build.
  auto request = neighbor->find_request(this);
  request->set_kokkos_device(std::is_same<DeviceType, LMPDeviceType>::value);
  request->set_kokkos_host(false);
  request->enable_full();
  edge_vec_fp32 = deep_pot.uses_fp32_edge_vectors();
  canonical_graph = deep_pot.uses_canonical_graph_inference();
  canonical_source_csr = deep_pot.canonical_graph_requires_source_csr();
  // PairDeepMD::init_style() has confined a model with a long-range module to
  // a complete non-periodic system.
  long_range = deep_pot.has_long_range();
  // Force exchange transfers three values per atom. Centroid per-atom virial
  // uses nine values even though the Kokkos full-list request runs newton off,
  // so comm_reverse_off reserves the classic host buffer for the wider mode.
  comm_reverse = 3;
  comm_reverse_off = 9;

  // The model node set is shared by both input schemas; type_idx_map is
  // populated by the base coeff().
  compact_graph.setup(type_idx_map, multi_rank);
}

template <class DeviceType>
int PairDeepMDKokkos<DeviceType>::build_edges_device() {
  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;
  compact_graph.refresh_nodes();
  const int nnode_model = compact_graph.nnode_model;

  // === Neighbor list and atom views on the device ===
  NeighListKokkos<DeviceType>* k_list =
      static_cast<NeighListKokkos<DeviceType>*>(list);
  const int inum = k_list->inum;
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  atomKK->sync(execution_space, X_MASK | TYPE_MASK);
  auto x = atomKK->k_x.template view<DeviceType>();

  const double cut = cutoff;
  const double cutsq = cut * cut;
  auto loc2model = compact_graph.d_loc2model;
  auto candidate_to_model = compact_graph.d_candidate_to_model;
  auto model2loc = compact_graph.d_model2loc;

  // === Node types in the edge-input schema's index layout ===
  auto type = atomKK->k_type.template view<DeviceType>();
  auto type_map = compact_graph.d_type_map;
  if ((int)d_model_type.extent(0) < nnode_model) {
    d_model_type = Kokkos::View<int*, DeviceType>("deepmd/kk:model_type", nall);
  }
  auto model_type = d_model_type;
  Kokkos::parallel_for(
      "deepmd/kk:mtype", Kokkos::RangePolicy<DeviceType>(0, nnode_model),
      KOKKOS_LAMBDA(const int m) {
        model_type(m) = type_map(type(model2loc(m)) - 1);
      });

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
          if (candidate_to_model(j) < 0) {
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
          const int mj = candidate_to_model(j);
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
  return nedge;
}

template <class DeviceType>
const double* PairDeepMDKokkos<DeviceType>::model_node_positions() {
  atomKK->sync(execution_space, X_MASK);
  auto x = atomKK->k_x.template view<DeviceType>();
  // Without virtual atoms the model nodes are the leading atoms, whose
  // coordinates are already in the length unit of the model at a conversion
  // factor of one.
  if (!compact_graph.has_null_types && dist_unit_cvt_factor == 1.0) {
    return x.data();
  }
  const int nnode_model = compact_graph.nnode_model;
  if ((int)d_coord_model.extent(0) < 3 * nnode_model) {
    d_coord_model = Kokkos::View<double*, DeviceType>("deepmd/kk:coord_model",
                                                      3 * nnode_model);
  }
  auto coord_model = d_coord_model;
  auto model2loc = compact_graph.d_model2loc;
  const double inv_dist = 1.0 / dist_unit_cvt_factor;
  Kokkos::parallel_for(
      "deepmd/kk:coord", Kokkos::RangePolicy<DeviceType>(0, nnode_model),
      KOKKOS_LAMBDA(const int m) {
        const int i = model2loc(m);
        coord_model(3 * m + 0) = x(i, 0) * inv_dist;
        coord_model(3 * m + 1) = x(i, 1) * inv_dist;
        coord_model(3 * m + 2) = x(i, 2) * inv_dist;
      });
  return coord_model.data();
}

template <class DeviceType>
void PairDeepMDKokkos<DeviceType>::compute(int eflag, int vflag) {
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
  int nedge = 0;
  // The source CSR feeds the source-attributed per-atom virial, and every
  // force assembly of an artifact that groups edges by source.
  const bool source_csr = canonical_source_csr || cvflag_atom;
  if (canonical_graph) {
    compact_graph.build(list, cutoff, dist_unit_cvt_factor, source_csr);
  } else {
    nedge = build_edges_device();
  }
  const int nloc_m = compact_graph.nloc_model;    // local nodes (energy)
  const int nnode_m = compact_graph.nnode_model;  // all nodes (force / virial)

  // Energy is per local node; force / virial span the model node set, which is
  // the local atoms (folded) or local + real ghost atoms (extended). The two
  // extents grow independently: under domain decomposition ``nlocal`` and the
  // node count need not move together, so a shared guard could leave the
  // energy buffer short when ``nlocal`` grows while the node count does not.
  // Sized by the node set rather than by every atom, the buffers of a folded
  // run keep their addresses while the ghost count fluctuates, which lets an
  // unchanged evaluation be replayed.
  const int nall = atom->nlocal + atom->nghost;
  if ((int)d_atom_energy.extent(0) < nlocal) {
    d_atom_energy =
        Kokkos::View<double*, DeviceType>("deepmd/kk:atom_energy", nlocal);
  }
  if ((int)d_out_force.extent(0) < 3 * nnode_m) {
    d_out_force =
        Kokkos::View<double*, DeviceType>("deepmd/kk:out_force", 3 * nnode_m);
    d_atom_virial =
        Kokkos::View<double*, DeviceType>("deepmd/kk:atom_virial", 9 * nnode_m);
  }
  if (cvflag_atom && multi_rank && (int)k_reverse_virial.extent(0) < 9 * nall) {
    k_reverse_virial =
        DAT::tdual_double_1d("deepmd/kk:reverse_virial", 9 * nall);
  }
  // The canonical evaluation overwrites every output entry the step reads,
  // so only its absence leaves outputs to clear; the virial of the graph is
  // read on the canonical path alone.
  if (!canonical_graph || nnode_m == 0) {
    Kokkos::deep_copy(d_out_force, 0.0);
    Kokkos::deep_copy(d_atom_energy, 0.0);
    Kokkos::deep_copy(d_atom_virial, 0.0);
  }
  if (canonical_graph && nnode_m == 0) {
    Kokkos::deep_copy(d_graph_virial, 0.0);
  }

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
  if (compact_graph.has_null_types && dim_aparam > 0 && !aparam_step.empty()) {
    auto h_m2l = compact_graph.k_model2loc.view_host();
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
  if (multi_rank && !compact_graph.has_null_types) {
    comm_list = make_comm_nlist();
    comm_ptr = &comm_list;
  }

  // A model with a long-range module evaluates the system over all the ranks
  // that hold it, so on several ranks every rank calls it at every step, one
  // without nodes included, and passes it the communicator.
  void* const world_ptr =
      long_range && multi_rank ? static_cast<void*>(&world) : nullptr;
  if ((canonical_graph && (nnode_m > 0 || world_ptr != nullptr)) ||
      (!canonical_graph && (nloc_m > 0 || comm_ptr != nullptr))) {
    // Fully device-resident inference: raw device pointers in and out. The
    // compact canonical model is enqueued on the Kokkos stream that builds its
    // graph and scatters its forces, so the three stay ordered without a host
    // synchronization. The edge-input models, and every model of an execution
    // space without a CUDA stream to share, run on PyTorch's own stream, so
    // the Kokkos work is fenced before the model reads the edges, and the
    // model synchronizes the device before returning its outputs.
    // Every edge-input model takes the positions of the model nodes, and a
    // compact canonical model reads them only with a long-range module. Their
    // gather is enqueued before the fence, and on the canonical path on the
    // stream that the model shares.
    const double* coord_ptr =
        (!canonical_graph || long_range) ? model_node_positions() : nullptr;
    void* const model_stream =
        compact_canonical_stream(typename DeviceType::execution_space());
    if (!canonical_graph || model_stream == nullptr) {
      Kokkos::fence();
    }
    try {
      if (canonical_graph) {
        deep_pot.compute_canonical_graph_gpu(
            d_atom_energy.data(), d_out_force.data(), d_graph_virial.data(),
            cvflag_atom ? d_atom_virial.data() : nullptr,
            compact_graph.d_model_type.data(), compact_graph.d_source.data(),
            compact_graph.d_edge_vec.data(),
            compact_graph.d_destination_row_ptr.data(),
            source_csr ? compact_graph.d_source_row_ptr.data() : nullptr,
            source_csr ? compact_graph.d_source_order.data() : nullptr,
            coord_ptr, world_ptr, nloc_m, nnode_m, compact_graph.storage_count,
            model_stream);
      } else if (edge_vec_fp32) {
        deep_pot.compute_edges_gpu(d_atom_energy.data(), d_out_force.data(),
                                   d_atom_virial.data(), coord_ptr,
                                   d_model_type.data(), d_edge_index.data(),
                                   d_edge_vec_float.data(), nloc_m, nedge,
                                   fparam, aparam_step, nnode_m, comm_ptr);
      } else {
        deep_pot.compute_edges_gpu(d_atom_energy.data(), d_out_force.data(),
                                   d_atom_virial.data(), coord_ptr,
                                   d_model_type.data(), d_edge_index.data(),
                                   d_edge_vec.data(), nloc_m, nedge, fparam,
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
  // The scatter and the fold stay device-resident: the reverse communication
  // packs and unpacks on the device (CompactCanonicalGhostFold). Only a
  // communication style other than the Kokkos brick communication calls the
  // host pack/unpack methods, which synchronize the force DualView, and the
  // completed fold is then copied back once.
  auto model2loc = compact_graph.d_model2loc;
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
    ghost_fold.fold(comm, world, this, 3);
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
    // Map the 9-component virial to the LAMMPS global 6 (xx, yy, zz, xy, xz,
    // yz), matching the standalone pair's index map. The canonical artifact
    // returns it for the whole graph; otherwise the per-node virial is summed
    // over all nodes (local + extended ghost), which equals the model's
    // reduced virial for this rank's local-centered edges.
    const int comp[6] = {0, 4, 8, 3, 6, 7};
    if (canonical_graph) {
      auto h_virial = Kokkos::create_mirror_view(d_graph_virial);
      Kokkos::deep_copy(h_virial, d_graph_virial);
      for (int k = 0; k < 6; ++k) {
        virial[k] += scale[1][1] * h_virial(comp[k]) * ener_unit_cvt_factor;
      }
    } else {
      auto atom_virial = d_atom_virial;
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
    auto h_m2l = compact_graph.k_model2loc.view_host();
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
      ghost_fold.fold(comm, world, this, 9);
      k_reverse_virial.sync_host();
      for (int i = 0; i < nlocal; ++i) {
        for (int k = 0; k < 9; ++k) {
          cvatom[i][k] += h_reverse(9 * i + k);
        }
      }
      reverse_virial = false;
    }
  }
}

namespace LAMMPS_NS {
template class PairDeepMDKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairDeepMDKokkos<LMPHostType>;
#endif
}  // namespace LAMMPS_NS

#endif  // LMP_KOKKOS
