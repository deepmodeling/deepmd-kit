// SPDX-License-Identifier: LGPL-3.0-or-later
// The device pair style is available when the LAMMPS Kokkos package is enabled.
#ifdef LMP_KOKKOS

#ifndef LAMMPS_VERSION_NUMBER
#error Please define LAMMPS_VERSION_NUMBER to yyyymmdd
#endif

#ifdef PAIR_CLASS
// clang-format off
PairStyle(deepmd/kk,PairDeepMDKokkos<LMPDeviceType>);
PairStyle(deepmd/kk/device,PairDeepMDKokkos<LMPDeviceType>);
PairStyle(deepmd/kk/host,PairDeepMDKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_DEEPMD_KOKKOS_H
#define LMP_PAIR_DEEPMD_KOKKOS_H

#include <cstddef>
#include <cstdint>

#include "kokkos_base.h"
#include "kokkos_type.h"
#include "neigh_list_kokkos.h"
#include "pair_deepmd.h"

namespace LAMMPS_NS {

template <class DeviceType>
struct CompactCanonicalGraphWorkspace {
  Kokkos::View<std::int64_t*, DeviceType> source;
  Kokkos::View<float*, DeviceType> edge_vec;
  Kokkos::View<std::int64_t*, DeviceType> destination_row_ptr;
  Kokkos::View<std::int64_t*, DeviceType> source_counts;
  Kokkos::View<std::int64_t*, DeviceType> source_row_ptr;
  Kokkos::View<std::int64_t*, DeviceType> source_cursor;
  Kokkos::View<std::int64_t*, DeviceType> source_order;
  std::size_t edge_capacity = 0;
};

// GPU-resident inference for exported ``.pt2`` models whose forward consumes
// an explicit edge graph: both the graph-input form (a compact, unpadded
// neighbor graph) and the edge-input form. Both are dispatched through
// ``DeepPot::compute_edges_gpu``.
//
// The neighbor list, the compact edge schema and the model outputs all stay
// on the device: the edge graph is built from the Kokkos device neighbor
// list, handed to ``compute_edges_gpu`` as raw device pointers, and the
// returned per-atom force / energy / virial are scattered back into the
// Kokkos atom arrays without any host round-trip. This removes the per-step
// host coordinate marshaling and the host-device transfers of the standalone
// ``pair_style deepmd`` path.
//
// A single rank uses the folded minimum-image node set (box thickness
// > 2 * cutoff along every periodic direction); domain decomposition uses the
// extended local-plus-ghost node set and folds ghost forces onto their owners
// through reverse communication.
template <class DeviceType>
class PairDeepMDKokkos : public PairDeepMD, public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  PairDeepMDKokkos(class LAMMPS*);
  ~PairDeepMDKokkos() override;

  void compute(int, int) override;
  void init_style() override;
  // Fold extended (ghost) node outputs onto their owners. The KOKKOS package
  // forces 'newton off' with a full neighbor list, disabling the integrator's
  // automatic reverse communication, so the extended multi-domain path drives
  // it explicitly for force and centroid per-atom virial. The Kokkos overrides
  // run device-resident with GPU-aware MPI; the plain overrides serve the
  // host-staged path.
  int pack_reverse_comm(int, int, double*) override;
  void unpack_reverse_comm(int, int*, double*) override;
  int pack_reverse_comm_kokkos(int, int, DAT::tdual_double_1d&) override;
  void unpack_reverse_comm_kokkos(int,
                                  DAT::tdual_int_1d,
                                  DAT::tdual_double_1d&) override;

  // Build the device edge graph from the Kokkos full neighbor list, returning
  // the edge count. A single rank folds ghosts onto local owners (minimum
  // image); domain decomposition keeps the extended local-plus-ghost node set.
  // Public because it launches extended device lambdas, which CUDA forbids
  // inside non-public members.
  void prepare_model_nodes();
  int build_edges_device();
  std::int64_t build_canonical_edges_device(
      CompactCanonicalGraphWorkspace<DeviceType>& workspace);

 protected:
  // LAMMPS type (1-based) -> model type, resident on the device.
  Kokkos::View<int*, DeviceType> d_type_map;
  Kokkos::View<int*, DeviceType>
      d_model_type;  // (nnode_model) type per model node
  Kokkos::View<std::int64_t*, DeviceType>
      d_model_type_i64;  // compact canonical artifact type per model node
  // Ghost -> local owner fold, rebuilt on the host at each neighbor rebuild.
  DAT::tdual_int_1d k_owner;
  typename AT::t_int_1d d_owner;

  // Virtual-atom (NULL type) compaction, rebuilt with the neighbor list: the
  // model sees only the local atoms with a real model type, so ``model2loc``
  // lists those local indices and ``loc2model`` inverts it (-1 for virtual).
  // When no type maps to NULL the compaction is the identity.
  bool has_null_types;
  bool multi_rank;  // domain-decomposed run -> extended (local+ghost) node set
  int nloc_model;   // real local model nodes; the energy is summed over these
  int nnode_model;  // total model nodes (== nloc_model folded; + ghost
                    // extended)
  DAT::tdual_int_1d k_loc2model;  // (nall) atom -> model node index, or -1
  DAT::tdual_int_1d k_model2loc;  // (nall) model node index -> atom index
  typename AT::t_int_1d d_loc2model;
  typename AT::t_int_1d d_model2loc;
  Kokkos::View<double*, DeviceType>
      d_coord_model;  // (3 * nnode_model), NULL case

  // Compact edge schema: edge_index is [2 * nedge] (src rows then dst rows),
  // edge_vec is [3 * nedge]; offsets is the per-atom exclusive edge prefix.
  Kokkos::View<std::int64_t*, DeviceType> d_edge_offset;  // (nlocal + 1)
  Kokkos::View<int*, DeviceType> d_edge_index;            // (2 * nedge)
  Kokkos::View<double*, DeviceType> d_edge_vec;           // (3 * nedge)
  Kokkos::View<float*, DeviceType>
      d_edge_vec_float;  // (3 * nedge), compressed graph ABI
  CompactCanonicalGraphWorkspace<DeviceType> canonical_workspace;

  // Model outputs on the device. Energy is per local atom; force and virial
  // span the model node set (up to ``nall`` under domain decomposition).
  Kokkos::View<double*, DeviceType> d_atom_energy;  // (nlocal)
  Kokkos::View<double*, DeviceType> d_out_force;    // (3 * nall)
  Kokkos::View<double*, DeviceType> d_atom_virial;  // (9 * nall)
  DAT::tdual_double_1d
      k_reverse_virial;  // (9 * nall), atom-order ghost contributions

  // Per-atom energy accumulator (aliases the base Pair ``eatom`` host array so
  // downstream per-atom computes/dumps see it after the device-to-host sync).
  DAT::ttransform_kkacc_1d k_eatom;
  typename AT::t_kkacc_1d d_eatom;

  int edge_capacity;       // allocated edges in d_edge_index / d_edge_vec
  bool edge_vec_fp32;      // model graph ABI consumes edge vectors in fp32
  bool canonical_graph;    // compact source-only graph artifact
  bool device_path_ok;     // resolved once in init_style
  bool reverse_virial;     // reverse communication operates on centroid virial
  bool reverse_used_host;  // force reverse communication selected host staging
};

}  // namespace LAMMPS_NS

#endif
#endif

#endif  // LMP_KOKKOS
