// SPDX-License-Identifier: LGPL-3.0-or-later
// The device pair style is available when the LAMMPS Kokkos package is enabled.
#ifdef LMP_KOKKOS

#ifndef LAMMPS_VERSION_NUMBER
#error Please define LAMMPS_VERSION_NUMBER to yyyymmdd
#endif

#ifdef PAIR_CLASS
// clang-format off
PairStyle(dpa4spin/kk,PairDPA4SpinKokkos<LMPDeviceType>);
PairStyle(dpa4spin/kk/device,PairDPA4SpinKokkos<LMPDeviceType>);
PairStyle(dpa4spin/kk/host,PairDPA4SpinKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_DPA4SPIN_KOKKOS_H
#define LMP_PAIR_DPA4SPIN_KOKKOS_H

#include <cstddef>
#include <cstdint>

#include "compact_canonical_graph_kokkos.h"
#include "kokkos_base.h"
#include "kokkos_type.h"
#include "neigh_list_kokkos.h"
#include "pair_dpa4spin.h"

namespace LAMMPS_NS {

// LAMMPS 22Jul2025 exposes reverse-communication buffers as X_FLOAT; starting
// with 10Sep2025, Kokkos pair styles use a fixed double buffer.
#if LAMMPS_VERSION_NUMBER < 20250910
using DPA4SpinKokkosCommBuffer = DAT::tdual_xfloat_1d;
#else
using DPA4SpinKokkosCommBuffer = DAT::tdual_double_1d;
#endif

// GPU-resident inference for exported native-spin ``.pt2`` models whose forward
// consumes the compact canonical graph: a dual-CSR neighbor topology with
// uint32 indices and float32 edge vectors, plus the per-node magnetic moment.
// It is dispatched through ``DeepSpin::compute_canonical_graph_gpu``.
//
// The neighbor list, the compact graph, the moment and the model outputs all
// stay on the device: the graph is built from the Kokkos device neighbor list,
// the moment is gathered from the Kokkos ``sp`` array, both are handed to
// ``compute_canonical_graph_gpu`` as raw device pointers, and the returned
// per-atom force, magnetic force, energy and virial are scattered back into the
// Kokkos atom arrays without any host round-trip. This removes the per-step
// host coordinate and moment marshaling of the ``dpa4spin`` path.
//
// A single rank uses the folded minimum-image node set (box thickness
// > 2 * cutoff along every periodic direction); domain decomposition uses the
// extended local-plus-ghost node set and folds ghost force and magnetic force
// onto their owners through reverse communication.
template <class DeviceType>
class PairDPA4SpinKokkos : public PairDPA4Spin, public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  PairDPA4SpinKokkos(class LAMMPS*);
  ~PairDPA4SpinKokkos() override;

  void compute(int, int) override;
  void init_style() override;
  // Fold extended (ghost) node outputs onto their owners. The KOKKOS package
  // forces 'newton off' with a full neighbor list, disabling the integrator's
  // automatic reverse communication, so the extended multi-domain path drives
  // it explicitly for force, magnetic force and centroid per-atom virial. The
  // Kokkos overrides run device-resident with GPU-aware MPI; the plain
  // overrides serve the host-staged path.
  int pack_reverse_comm(int, int, double*) override;
  void unpack_reverse_comm(int, int*, double*) override;
  int pack_reverse_comm_kokkos(int, int, DPA4SpinKokkosCommBuffer&) override;
  void unpack_reverse_comm_kokkos(int,
                                  DAT::tdual_int_1d,
                                  DPA4SpinKokkosCommBuffer&) override;

  // Gather the per-node magnetic moment from the Kokkos ``sp`` array. Public
  // because it launches an extended device lambda, which CUDA forbids inside
  // non-public members.
  void gather_moment_device();

 protected:
  // Model node set and the compact canonical graph over it.
  CompactCanonicalGraphKokkos<DeviceType> compact_graph;
  bool multi_rank;  // domain-decomposed run -> extended (local+ghost) node set
  // (3 * nall) per-node magnetic moment in the ABI's float32 layout, the
  // LAMMPS unit direction scaled by its magnitude.
  Kokkos::View<float*, DeviceType> d_moment;

  // Model outputs on the device. Energy is per local atom; force, magnetic
  // force and virial span the model node set (up to ``nall`` under domain
  // decomposition).
  Kokkos::View<double*, DeviceType> d_atom_energy;    // (nlocal)
  Kokkos::View<double*, DeviceType> d_out_force;      // (3 * nall)
  Kokkos::View<double*, DeviceType> d_out_force_mag;  // (3 * nall)
  Kokkos::View<double*, DeviceType> d_atom_virial;    // (9 * nall)
  DAT::tdual_double_1d
      k_reverse_virial;  // (9 * nall), atom-order ghost contributions

  // Per-atom energy accumulator (aliases the base Pair ``eatom`` host array so
  // downstream per-atom computes/dumps see it after the device-to-host sync).
  // The transformed accumulator view was added in the 10Sep2025 release.
#if LAMMPS_VERSION_NUMBER < 20250910
  DAT::tdual_double_1d k_eatom;
  typename AT::t_double_1d d_eatom;
#else
  DAT::ttransform_kkacc_1d k_eatom;
  typename AT::t_kkacc_1d d_eatom;
#endif

  bool reverse_virial;     // reverse communication operates on centroid virial
  bool reverse_used_host;  // force reverse communication selected host staging
};

}  // namespace LAMMPS_NS

#endif
#endif

#endif  // LMP_KOKKOS
