// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef LMP_KOKKOS
#include "pair_dpa4spin_kokkos.h"

#include <type_traits>

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

namespace {
// Reduced Planck constant in eV.ps. The model reports the magnetic force as
// the energy gradient with respect to the magnetic moment, while LAMMPS stores
// the precession force; the two differ by the moment magnitude over hbar.
constexpr double kHBar = 6.5821191e-04;
}  // namespace

template <class DeviceType>
PairDPA4SpinKokkos<DeviceType>::PairDPA4SpinKokkos(LAMMPS* lmp)
    : PairDPA4Spin(lmp),
      compact_graph(lmp),
      multi_rank(false),
      reverse_virial(false),
      reverse_used_host(false) {
  respa_enable = 0;
  kokkosable = 1;
  atomKK = (AtomKokkos*)atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | TYPE_MASK | SP_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | FM_MASK | ENERGY_MASK | VIRIAL_MASK;
  reverse_comm_device = 1;
}

template <class DeviceType>
PairDPA4SpinKokkos<DeviceType>::~PairDPA4SpinKokkos() {
  if (copymode) {
    return;
  }
  memoryKK->destroy_kokkos(k_eatom, eatom);
}

template <class DeviceType>
int PairDPA4SpinKokkos<DeviceType>::pack_reverse_comm(int n,
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
  atomKK->sync(Host, F_MASK | FM_MASK);
  double** f = atom->f;
  double** fm = atom->fm;
  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; ++i) {
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
    buf[m++] = fm[i][0];
    buf[m++] = fm[i][1];
    buf[m++] = fm[i][2];
  }
  return m;
}

template <class DeviceType>
void PairDPA4SpinKokkos<DeviceType>::unpack_reverse_comm(int n,
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
  atomKK->sync(Host, F_MASK | FM_MASK);
  double** f = atom->f;
  double** fm = atom->fm;
  int m = 0;
  for (int i = 0; i < n; ++i) {
    const int j = list[i];
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
    fm[j][0] += buf[m++];
    fm[j][1] += buf[m++];
    fm[j][2] += buf[m++];
  }
  atomKK->modified(Host, F_MASK | FM_MASK);
}

template <class DeviceType>
int PairDPA4SpinKokkos<DeviceType>::pack_reverse_comm_kokkos(
    int n, int first, DPA4SpinKokkosCommBuffer& buf) {
  auto d_buf = buf.template view<DeviceType>();
  if (reverse_virial) {
    auto reverse_virial_data = k_reverse_virial.template view<DeviceType>();
    const int first_i = first;
    Kokkos::parallel_for(
        "dpa4spin/kk:pack_rev_virial", Kokkos::RangePolicy<DeviceType>(0, n),
        KOKKOS_LAMBDA(const int i) {
          for (int k = 0; k < 9; ++k) {
            d_buf(9 * i + k) = reverse_virial_data(9 * (first_i + i) + k);
          }
        });
    return n * 9;
  }
  auto f = atomKK->k_f.template view<DeviceType>();
  auto fm = atomKK->k_fm.template view<DeviceType>();
  const int first_i = first;
  Kokkos::parallel_for(
      "dpa4spin/kk:pack_rev", Kokkos::RangePolicy<DeviceType>(0, n),
      KOKKOS_LAMBDA(const int i) {
        d_buf(6 * i + 0) = f(first_i + i, 0);
        d_buf(6 * i + 1) = f(first_i + i, 1);
        d_buf(6 * i + 2) = f(first_i + i, 2);
        d_buf(6 * i + 3) = fm(first_i + i, 0);
        d_buf(6 * i + 4) = fm(first_i + i, 1);
        d_buf(6 * i + 5) = fm(first_i + i, 2);
      });
  return n * 6;
}

template <class DeviceType>
void PairDPA4SpinKokkos<DeviceType>::unpack_reverse_comm_kokkos(
    int n, DAT::tdual_int_1d list, DPA4SpinKokkosCommBuffer& buf) {
  auto d_buf = buf.template view<DeviceType>();
  auto d_list = list.template view<DeviceType>();
  if (reverse_virial) {
    k_reverse_virial.template modify<DeviceType>();
    auto reverse_virial_data = k_reverse_virial.template view<DeviceType>();
    Kokkos::parallel_for(
        "dpa4spin/kk:unpack_rev_virial", Kokkos::RangePolicy<DeviceType>(0, n),
        KOKKOS_LAMBDA(const int i) {
          const int j = d_list(i);
          for (int k = 0; k < 9; ++k) {
            reverse_virial_data(9 * j + k) += d_buf(9 * i + k);
          }
        });
    return;
  }
  auto f = atomKK->k_f.template view<DeviceType>();
  auto fm = atomKK->k_fm.template view<DeviceType>();
  Kokkos::parallel_for(
      "dpa4spin/kk:unpack_rev", Kokkos::RangePolicy<DeviceType>(0, n),
      KOKKOS_LAMBDA(const int i) {
        const int j = d_list(i);
        f(j, 0) += d_buf(6 * i + 0);
        f(j, 1) += d_buf(6 * i + 1);
        f(j, 2) += d_buf(6 * i + 2);
        fm(j, 0) += d_buf(6 * i + 3);
        fm(j, 1) += d_buf(6 * i + 4);
        fm(j, 2) += d_buf(6 * i + 5);
      });
}

template <class DeviceType>
void PairDPA4SpinKokkos<DeviceType>::init_style() {
  // Full neighbor-list request and the shared native-spin contract.
  PairDPA4Spin::init_style();

  // The device-resident entry point runs the model on an accelerator, so the
  // Kokkos host execution space has nothing to hand it.
  if (std::is_same<DeviceType, LMPHostType>::value) {
    error->all(FLERR, "pair style dpa4spin/kk runs on the GPU backend only.");
  }
  // Device residency is what the compact canonical ABI exists for: it is the
  // only schema whose inputs this style can hand over without a host round
  // trip.
  if (!deep_spin.uses_canonical_graph_inference()) {
    error->all(FLERR,
               "pair style dpa4spin/kk requires a model frozen with the "
               "compact canonical graph lower; a model frozen with the graph "
               "lower is served by pair style dpa4spin.");
  }
  // Domain decomposition uses the extended (local + ghost) node set: the model
  // computes per-node force and magnetic force and the reverse communication
  // folds the ghost contributions onto their owners. A single rank uses the
  // folded minimum-image node set.
  multi_rank = (comm->nprocs > 1);

  // Route the base full request to the Kokkos device neighbor build.
  auto request = neighbor->find_request(this);
  request->set_kokkos_device(std::is_same<DeviceType, LMPDeviceType>::value);
  request->set_kokkos_host(false);
  request->enable_full();
  // Force and magnetic force exchange three values each per atom. Centroid
  // per-atom virial uses nine values even though the Kokkos full-list request
  // runs newton off, so comm_reverse_off reserves the classic host buffer for
  // the wider mode.
  comm_reverse = 6;
  comm_reverse_off = 9;

  // The model node set is shared with the graph builder; type_idx_map is
  // populated by the base coeff().
  compact_graph.setup(type_idx_map, multi_rank);
}

template <class DeviceType>
void PairDPA4SpinKokkos<DeviceType>::gather_moment_device() {
  const int nall = atom->nlocal + atom->nghost;
  const int nnode_model = compact_graph.nnode_model;
  if ((int)d_moment.extent(0) < 3 * nnode_model) {
    d_moment = Kokkos::View<float*, DeviceType>("dpa4spin/kk:moment", 3 * nall);
  }
  // LAMMPS stores the moment as a unit direction and its magnitude; the
  // artifact consumes the product. Ghost rows repeat their owner's value,
  // which the ``sp`` forward communication has already refreshed, so the
  // moment needs no exchange of its own.
  atomKK->sync(execution_space, SP_MASK);
  auto sp = atomKK->k_sp.template view<DeviceType>();
  auto moment = d_moment;
  auto model2loc = compact_graph.d_model2loc;
  Kokkos::parallel_for(
      "dpa4spin/kk:moment", Kokkos::RangePolicy<DeviceType>(0, nnode_model),
      KOKKOS_LAMBDA(const int m) {
        const int i = model2loc(m);
        const auto norm = sp(i, 3);
        moment(3 * m + 0) = static_cast<float>(sp(i, 0) * norm);
        moment(3 * m + 1) = static_cast<float>(sp(i, 1) * norm);
        moment(3 * m + 2) = static_cast<float>(sp(i, 2) * norm);
      });
}

template <class DeviceType>
void PairDPA4SpinKokkos<DeviceType>::compute(int eflag, int vflag) {
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
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "dpa4spin/kk:eatom");
    d_eatom = k_eatom.template view<DeviceType>();
  }
  compact_graph.build(list, cutoff, dist_unit_cvt_factor);
  gather_moment_device();
  const int nloc_m = compact_graph.nloc_model;    // local nodes (energy)
  const int nnode_m = compact_graph.nnode_model;  // all nodes (force / virial)

  // Energy is per local node; force, magnetic force and virial span the model
  // node set, which is the local atoms (folded) or local + real ghost atoms
  // (extended, up to nall). The two extents grow independently: under domain
  // decomposition ``nlocal`` and ``nall`` need not move together, so a shared
  // guard could leave the energy buffer short when ``nlocal`` grows while
  // ``nall`` does not.
  const int nall = atom->nlocal + atom->nghost;
  if ((int)d_atom_energy.extent(0) < nlocal) {
    d_atom_energy =
        Kokkos::View<double*, DeviceType>("dpa4spin/kk:atom_energy", nlocal);
  }
  if ((int)d_out_force.extent(0) < 3 * nall) {
    d_out_force =
        Kokkos::View<double*, DeviceType>("dpa4spin/kk:out_force", 3 * nall);
    d_out_force_mag = Kokkos::View<double*, DeviceType>(
        "dpa4spin/kk:out_force_mag", 3 * nall);
    d_atom_virial =
        Kokkos::View<double*, DeviceType>("dpa4spin/kk:atom_virial", 9 * nall);
  }
  if (cvflag_atom && multi_rank && (int)k_reverse_virial.extent(0) < 9 * nall) {
    k_reverse_virial =
        DAT::tdual_double_1d("dpa4spin/kk:reverse_virial", 9 * nall);
  }
  Kokkos::deep_copy(d_out_force, 0.0);
  Kokkos::deep_copy(d_out_force_mag, 0.0);
  Kokkos::deep_copy(d_atom_energy, 0.0);
  Kokkos::deep_copy(d_atom_virial, 0.0);

  if (nnode_m > 0) {
    // Fully device-resident inference: raw device pointers in and out. The
    // graph and the moment are produced on the Kokkos stream and consumed by
    // the model on PyTorch's stream, and the outputs flow back to the Kokkos
    // scatter, so the two runtimes are bracketed by explicit synchronization:
    // fence the Kokkos work before the model reads its inputs, and synchronize
    // the device after so the scatter sees the finished model outputs.
    Kokkos::fence();
    try {
      deep_spin.compute_canonical_graph_gpu(
          d_atom_energy.data(), d_out_force.data(), d_out_force_mag.data(),
          d_atom_virial.data(), compact_graph.d_model_type.data(),
          compact_graph.d_source.data(), compact_graph.d_edge_vec.data(),
          compact_graph.d_destination_row_ptr.data(),
          compact_graph.d_source_row_ptr.data(),
          compact_graph.d_source_order.data(), d_moment.data(), nloc_m, nnode_m,
          compact_graph.storage_count);
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
  // communication, the host pack/unpack methods synchronize the force
  // DualViews and the completed fold is copied back once after all
  // communication stages.
  auto model2loc = compact_graph.d_model2loc;
  const double fscale = scale[1][1] * force_unit_cvt_factor;
  const double fmscale = scale[1][1] * force_unit_cvt_factor / kHBar;
  reverse_virial = false;
  reverse_used_host = false;
  // The KOKKOS package runs 'newton off', so the integrator's force_clear only
  // zeros the local force and magnetic force (indices [0, nlocal)); the ghost
  // slots [nlocal, nall) are left untouched. The extended scatter writes ghost
  // slots and folds them onto their owners by reverse communication, so those
  // slots must be zeroed first, or their contribution accumulates across steps.
  atomKK->sync(execution_space, F_MASK | FM_MASK | SP_MASK);
  auto f = atomKK->k_f.template view<DeviceType>();
  auto fm = atomKK->k_fm.template view<DeviceType>();
  auto sp = atomKK->k_sp.template view<DeviceType>();
  auto out_force = d_out_force;
  auto out_force_mag = d_out_force_mag;
  if (multi_rank) {
    Kokkos::parallel_for(
        "dpa4spin/kk:clear_ghost_f",
        Kokkos::RangePolicy<DeviceType>(nlocal, nall),
        KOKKOS_LAMBDA(const int m) {
          f(m, 0) = 0.0;
          f(m, 1) = 0.0;
          f(m, 2) = 0.0;
          fm(m, 0) = 0.0;
          fm(m, 1) = 0.0;
          fm(m, 2) = 0.0;
        });
  }
  Kokkos::parallel_for(
      "dpa4spin/kk:scatter_f", Kokkos::RangePolicy<DeviceType>(0, nnode_m),
      KOKKOS_LAMBDA(const int m) {
        const int i = model2loc(m);
        // The precession force carried by ``fm`` is the energy gradient with
        // respect to the moment, rescaled by the moment magnitude over hbar.
        const double moment = fmscale * sp(i, 3);
        f(i, 0) += fscale * out_force(3 * m + 0);
        f(i, 1) += fscale * out_force(3 * m + 1);
        f(i, 2) += fscale * out_force(3 * m + 2);
        fm(i, 0) += moment * out_force_mag(3 * m + 0);
        fm(i, 1) += moment * out_force_mag(3 * m + 1);
        fm(i, 2) += moment * out_force_mag(3 * m + 2);
      });
  atomKK->modified(execution_space, F_MASK | FM_MASK);
  if (multi_rank) {
    comm->reverse_comm(this, 6);
    if (reverse_used_host) {
      atomKK->sync(execution_space, F_MASK | FM_MASK);
    }
  }

  if (eflag_global) {
    auto atom_energy = d_atom_energy;
    double e_sum = 0.0;
    Kokkos::parallel_reduce(
        "dpa4spin/kk:esum", Kokkos::RangePolicy<DeviceType>(0, nloc_m),
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
          "dpa4spin/kk:vsum", Kokkos::RangePolicy<DeviceType>(0, nnode_m),
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
        "dpa4spin/kk:eatom", Kokkos::RangePolicy<DeviceType>(0, nloc_m),
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
}

namespace LAMMPS_NS {
template class PairDPA4SpinKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairDPA4SpinKokkos<LMPHostType>;
#endif
}  // namespace LAMMPS_NS

#endif  // LMP_KOKKOS
