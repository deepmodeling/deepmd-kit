// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#ifdef BUILD_PYTORCH
// AOTInductor package loader requires a header that may not exist on all
// platforms (e.g. macOS x86_64).  Disable pt_expt support when missing.
#if __has_include(<torch/csrc/inductor/aoti_package/model_package_loader.h>)
#define BUILD_PT_EXPT 1
#else
#define BUILD_PT_EXPT 0
#endif

#if BUILD_PT_EXPT

#include <torch/torch.h>

#include "DeepPot.h"

// Forward-declare to keep TempFile out of public header. Defined in
// commonPTExpt.h.
namespace deepmd::ptexpt {
class TempFile;
}

namespace torch::inductor {
class AOTIModelPackageLoader;
}

namespace deepmd {
/**
 * @brief PyTorch Exportable (AOTInductor .pt2) implementation for Deep
 *Potential.
 **/
class DeepPotPTExpt : public DeepPotBackend {
 public:
  /**
   * @brief DP constructor without initialization.
   **/
  DeepPotPTExpt();
  virtual ~DeepPotPTExpt();
  /**
   * @brief DP constructor with initialization.
   * @param[in] model The name of the .pt2 model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  DeepPotPTExpt(const std::string& model,
                const int& gpu_rank = 0,
                const std::string& file_content = "");
  /**
   * @brief Initialize the DP.
   * @param[in] model The name of the .pt2 model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "");

 private:
  /**
   * @brief Evaluate with nlist (LAMMPS path — extended forces).
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& lmp_list,
               const int& ago,
               const std::vector<VALUETYPE>& fparam,
               const std::vector<VALUETYPE>& aparam,
               const std::vector<double>& charge_spin,
               const bool atomic);
  /**
   * @brief Evaluate without nlist (standalone — builds nlist, folds back).
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam,
               const std::vector<VALUETYPE>& aparam,
               const std::vector<double>& charge_spin,
               const bool atomic);

 public:
  double cutoff() const {
    assert(inited);
    return rcut;
  };
  int numb_types() const {
    assert(inited);
    return ntypes;
  };
  int numb_types_spin() const {
    assert(inited);
    return 0;
  };
  int dim_fparam() const {
    assert(inited);
    return dfparam;
  };
  int dim_aparam() const {
    assert(inited);
    return daparam;
  };
  int dim_chg_spin() const override {
    assert(inited);
    return dchgspin;
  };
  void get_type_map(std::string& type_map);
  bool is_aparam_nall() const {
    assert(inited);
    return aparam_nall;
  };
  bool has_default_fparam() const {
    assert(inited);
    return has_default_fparam_;
  };

  // forward to template class (no charge_spin — uses default_chg_spin_
  // fallback)
  void computew(std::vector<double>& ener,
                std::vector<double>& force,
                std::vector<double>& virial,
                std::vector<double>& atom_energy,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const std::vector<double>& fparam,
                const std::vector<double>& aparam,
                const bool atomic);
  void computew(std::vector<double>& ener,
                std::vector<float>& force,
                std::vector<float>& virial,
                std::vector<float>& atom_energy,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const std::vector<float>& fparam,
                const std::vector<float>& aparam,
                const bool atomic);
  void computew(std::vector<double>& ener,
                std::vector<double>& force,
                std::vector<double>& virial,
                std::vector<double>& atom_energy,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const int nghost,
                const InputNlist& inlist,
                const int& ago,
                const std::vector<double>& fparam,
                const std::vector<double>& aparam,
                const bool atomic);
  void computew(std::vector<double>& ener,
                std::vector<float>& force,
                std::vector<float>& virial,
                std::vector<float>& atom_energy,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const int nghost,
                const InputNlist& inlist,
                const int& ago,
                const std::vector<float>& fparam,
                const std::vector<float>& aparam,
                const bool atomic);
  void computew_mixed_type(std::vector<double>& ener,
                           std::vector<double>& force,
                           std::vector<double>& virial,
                           std::vector<double>& atom_energy,
                           std::vector<double>& atom_virial,
                           const int& nframes,
                           const std::vector<double>& coord,
                           const std::vector<int>& atype,
                           const std::vector<double>& box,
                           const std::vector<double>& fparam,
                           const std::vector<double>& aparam,
                           const bool atomic);
  void computew_mixed_type(std::vector<double>& ener,
                           std::vector<float>& force,
                           std::vector<float>& virial,
                           std::vector<float>& atom_energy,
                           std::vector<float>& atom_virial,
                           const int& nframes,
                           const std::vector<float>& coord,
                           const std::vector<int>& atype,
                           const std::vector<float>& box,
                           const std::vector<float>& fparam,
                           const std::vector<float>& aparam,
                           const bool atomic);

  // charge_spin overloads — pass runtime charge/spin per call
  void computew(std::vector<double>& ener,
                std::vector<double>& force,
                std::vector<double>& virial,
                std::vector<double>& atom_energy,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const std::vector<double>& fparam,
                const std::vector<double>& aparam,
                const std::vector<double>& charge_spin,
                const bool atomic) override;
  void computew(std::vector<double>& ener,
                std::vector<float>& force,
                std::vector<float>& virial,
                std::vector<float>& atom_energy,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const std::vector<float>& fparam,
                const std::vector<float>& aparam,
                const std::vector<double>& charge_spin,
                const bool atomic) override;
  void computew(std::vector<double>& ener,
                std::vector<double>& force,
                std::vector<double>& virial,
                std::vector<double>& atom_energy,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const int nghost,
                const InputNlist& inlist,
                const int& ago,
                const std::vector<double>& fparam,
                const std::vector<double>& aparam,
                const std::vector<double>& charge_spin,
                const bool atomic) override;
  void computew(std::vector<double>& ener,
                std::vector<float>& force,
                std::vector<float>& virial,
                std::vector<float>& atom_energy,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const int nghost,
                const InputNlist& inlist,
                const int& ago,
                const std::vector<float>& fparam,
                const std::vector<float>& aparam,
                const std::vector<double>& charge_spin,
                const bool atomic) override;
  void computew_mixed_type(std::vector<double>& ener,
                           std::vector<double>& force,
                           std::vector<double>& virial,
                           std::vector<double>& atom_energy,
                           std::vector<double>& atom_virial,
                           const int& nframes,
                           const std::vector<double>& coord,
                           const std::vector<int>& atype,
                           const std::vector<double>& box,
                           const std::vector<double>& fparam,
                           const std::vector<double>& aparam,
                           const std::vector<double>& charge_spin,
                           const bool atomic) override;
  void computew_mixed_type(std::vector<double>& ener,
                           std::vector<float>& force,
                           std::vector<float>& virial,
                           std::vector<float>& atom_energy,
                           std::vector<float>& atom_virial,
                           const int& nframes,
                           const std::vector<float>& coord,
                           const std::vector<int>& atype,
                           const std::vector<float>& box,
                           const std::vector<float>& fparam,
                           const std::vector<float>& aparam,
                           const std::vector<double>& charge_spin,
                           const bool atomic) override;

  /**
   * @brief Fully device-resident edge inference for single-domain SeZM/DPA4.
   *
   * Runs the exported model directly on a GPU-built compact edge schema,
   * keeping coordinates, the edge graph and the outputs on the device.  All
   * pointers reference GPU memory on the model's device.  ``edge_index`` is the
   * flattened [2, nedge] local edge graph (row 0 = neighbor/source, row 1 =
   * center/destination); ``edge_vec`` is the matching minimum-image bond vector
   * ``r_neighbor - r_center``.  Outputs are written device-to-device.
   *
   * @param[out] d_atom_energy Per-atom energy, GPU [nloc].
   * @param[out] d_force Per-atom force, GPU [nloc * 3] row-major.
   * @param[out] d_atom_virial Per-atom virial, GPU [nloc * 9] row-major.
   * @param[in] d_coord Local coordinates, GPU [nloc * 3] row-major.
   * @param[in] d_atype Local atom types, GPU [nloc].
   * @param[in] d_edge_index Local edge graph, GPU [2 * nedge].
   * @param[in] d_edge_vec Minimum-image bond vectors, GPU [nedge * 3].
   * @param[in] nloc Number of local atoms.
   * @param[in] nedge Number of physical edges (dummy edges added internally).
   */
  void compute_edges_gpu(double* d_atom_energy,
                         double* d_force,
                         double* d_atom_virial,
                         const double* d_coord,
                         const int* d_atype,
                         const int* d_edge_index,
                         const double* d_edge_vec,
                         const int nloc,
                         const int nedge) override;

 private:
  bool inited;
  int ntypes;
  int dfparam;
  int daparam;
  int dchgspin;
  bool aparam_nall;
  bool has_default_fparam_;
  std::vector<double> default_fparam_;
  std::vector<double> default_chg_spin_;
  double rcut;
  int gpu_id;
  bool gpu_enabled;
  std::vector<std::string> type_map;
  std::vector<std::string> output_keys;  // sorted internal output key names
  bool do_atomic_virial;  // whether model was exported with atomic virial corr
  int nnei;               // expected nlist nnei dimension (= sum(sel))
  bool lower_input_is_edge_ = false;
  bool lower_input_is_graph_ = false;
  NeighborListData nlist_data;
  at::Tensor mapping_tensor;           // cached mapping tensor (LAMMPS path)
  std::vector<std::int64_t> mapping_;  // cached mapping vector (LAMMPS path)
  at::Tensor firstneigh_tensor;        // cached nlist tensor (LAMMPS path)
  at::Tensor edge_index_tensor;        // cached local edge graph (LAMMPS path)
  at::Tensor edge_index_ext_tensor;  // cached extended edge graph (LAMMPS path)
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
  // Optional second AOTInductor artifact for the multi-rank GNN code
  // path (Phase 4).  Loaded only if the .pt2 metadata reports
  // ``has_comm_artifact == true`` AND the model has GNN message
  // passing.  ``with_comm_tempfile_`` owns the extracted nested .pt2
  // for the lifetime of ``with_comm_loader``.
  bool has_comm_artifact_ = false;
  // Whether the regular .pt2 graph consumes the mapping tensor for
  // ghost-feature gather (true for any message-passing descriptor:
  // DPA2/DPA3/hybrids; false for se_e2_a/DPA1/etc.).  Mirrors the
  // descriptor's ``has_message_passing()`` API; read from the
  // ``has_message_passing`` metadata field.  Defaults to false for
  // pre-PR .pt2 archives that lack the field so non-GNN archives
  // continue to work; GNN archives must be regenerated to opt into
  // the fail-fast guard against the silent-corruption bug.
  bool has_message_passing_ = false;
  // Whether the collective empty-rank preflight (allreduce of the minimum
  // owned+ghost node count over the LAMMPS communicator, graph with-comm
  // route) has PASSED for the current neighbor topology.  Reset on every
  // ``ago == 0`` rebuild: the node count shares the lifetime of the cached
  // nlist/mapping/edge topology, so re-running the collective on cache-hit
  // (``ago > 0``) force calls added a global synchronization per MD step
  // without any added protection.
  bool graph_comm_preflight_done_ = false;
  // Device-resident (ntypes+1)^2 model-level pair-type keep table, uploaded
  // ONCE in ``init`` from the ``pair_exclude_types`` metadata field (see
  // ``deepmd::buildPairExcludeTable``).  An UNDEFINED tensor => no model-level
  // exclusion (identity).  The device is fixed at ``init`` (``gpu_id`` /
  // ``gpu_enabled``), so the seam helpers ``index_select`` it directly with no
  // per-step CPU clone / H2D upload.  Exclusion is a BUILD-time transform
  // (decision #18/A4): the C++ ingestion seam is the single application site
  // (``applyPairExclusion`` graph / ``applyPairExclusionNlist`` dense); the
  // exported .pt2 lowers consume pre-excluded inputs and never re-apply it.
  torch::Tensor pair_exclude_table_;
  std::unique_ptr<deepmd::ptexpt::TempFile> with_comm_tempfile_;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> with_comm_loader;

  /**
   * @brief Multi-frame loop for standalone compute (no nlist).
   */
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute_nframes(ENERGYVTYPE& ener,
                       std::vector<VALUETYPE>& force,
                       std::vector<VALUETYPE>& virial,
                       std::vector<VALUETYPE>& atom_energy,
                       std::vector<VALUETYPE>& atom_virial,
                       const int nframes,
                       const std::vector<VALUETYPE>& coord,
                       const std::vector<int>& atype,
                       const std::vector<VALUETYPE>& box,
                       const std::vector<VALUETYPE>& fparam,
                       const std::vector<VALUETYPE>& aparam,
                       const std::vector<double>& charge_spin,
                       const bool atomic);

  /**
   * @brief Mixed-type compute implementation (loops over frames).
   */
  template <typename VALUETYPE>
  void compute_mixed_type_impl(std::vector<double>& ener,
                               std::vector<VALUETYPE>& force,
                               std::vector<VALUETYPE>& virial,
                               std::vector<VALUETYPE>& atom_energy,
                               std::vector<VALUETYPE>& atom_virial,
                               const int& nframes,
                               const std::vector<VALUETYPE>& coord,
                               const std::vector<int>& atype,
                               const std::vector<VALUETYPE>& box,
                               const std::vector<VALUETYPE>& fparam,
                               const std::vector<VALUETYPE>& aparam,
                               const std::vector<double>& charge_spin,
                               const bool atomic);

  /**
   * @brief Run the .pt2 model and return flat output tensors.
   * @param[in] coord Extended coordinates tensor.
   * @param[in] atype Extended atom types tensor.
   * @param[in] nlist Neighbor list tensor.
   * @param[in] mapping Mapping tensor.
   * @param[in] fparam Frame parameter tensor (or empty).
   * @param[in] aparam Atomic parameter tensor (or empty).
   * @param[in] charge_spin Charge/spin tensor (or empty).
   * @return Vector of output tensors in sorted key order.
   */
  std::vector<torch::Tensor> run_model(const torch::Tensor& coord,
                                       const torch::Tensor& atype,
                                       const torch::Tensor& nlist,
                                       const torch::Tensor& mapping,
                                       const torch::Tensor& fparam,
                                       const torch::Tensor& aparam,
                                       const torch::Tensor& charge_spin);

  std::vector<torch::Tensor> run_model_edges(
      const torch::Tensor& coord,
      const torch::Tensor& atype,
      const torch::Tensor& edge_index,
      const torch::Tensor& edge_vec,
      const torch::Tensor& edge_scatter_index,
      const torch::Tensor& edge_mask,
      const torch::Tensor& fparam,
      const torch::Tensor& aparam,
      const torch::Tensor& charge_spin);

  /**
   * @brief Run a NeighborGraph-schema ``.pt2`` (lower_input_kind="graph").
   *
   * Positional AOTI input order matches the Python export ABI:
   * ``(atype, n_node, edge_index, edge_vec, edge_mask, [fparam], [aparam],
   * [charge_spin])``.  Unlike the edge schema there is no ``coord`` and no
   * ``edge_scatter_index`` input; node count is carried by ``n_node`` and the
   * geometry is fully described by ``edge_vec``.
   *
   * @param[in] atype Per-node local types, shape ``(N,)`` int64.
   * @param[in] n_node Per-frame node count, shape ``(nf,)`` int64.
   * @param[in] edge_index Folded edge graph ``(2, E)`` int64 [src, dst].
   * @param[in] edge_vec Edge vectors ``(E, 3)`` (neighbour - center).
   * @param[in] edge_mask Physical-edge mask ``(E,)`` bool.
   */
  std::vector<torch::Tensor> run_model_graph(const torch::Tensor& atype,
                                             const torch::Tensor& n_node,
                                             const torch::Tensor& edge_index,
                                             const torch::Tensor& edge_vec,
                                             const torch::Tensor& edge_mask,
                                             const torch::Tensor& fparam,
                                             const torch::Tensor& aparam,
                                             const torch::Tensor& charge_spin);

  /**
   * @brief Run the with-comm .pt2 artifact with comm tensors appended.
   *
   * @param[in] base 4-6 base inputs (coord, atype, nlist, mapping,
   *            fparam?, aparam?) — same as ``run_model``.
   * @param[in] charge_spin Charge/spin tensor (or empty).
   * @param[in] comm_tensors 8 comm tensors in canonical positional
   *            order: send_list, send_proc, recv_proc, send_num,
   *            recv_num, communicator, nlocal, nghost.
   */
  std::vector<torch::Tensor> run_model_with_comm(
      const torch::Tensor& coord,
      const torch::Tensor& atype,
      const torch::Tensor& nlist,
      const torch::Tensor& mapping,
      const torch::Tensor& fparam,
      const torch::Tensor& aparam,
      const torch::Tensor& charge_spin,
      const std::vector<at::Tensor>& comm_tensors);

  /**
   * @brief Run the with-comm edge (SeZM) ``.pt2`` artifact with comm tensors.
   *
   * The edge schema indexes the extended node set, so ``edge_index`` and
   * ``edge_scatter_index`` coincide. ``atype`` carries owned atoms (fitting,
   * energy read-out) while ``extended_atype`` embeds ghost neighbours.
   *
   * @param[in] comm_tensors 8 comm tensors in canonical positional order:
   *            send_list, send_proc, recv_proc, send_num, recv_num,
   *            communicator, nlocal, nghost.
   */
  std::vector<torch::Tensor> run_model_edges_with_comm(
      const torch::Tensor& coord,
      const torch::Tensor& atype,
      const torch::Tensor& extended_atype,
      const torch::Tensor& edge_index,
      const torch::Tensor& edge_vec,
      const torch::Tensor& edge_scatter_index,
      const torch::Tensor& edge_mask,
      const torch::Tensor& fparam,
      const torch::Tensor& aparam,
      const torch::Tensor& charge_spin,
      const std::vector<at::Tensor>& comm_tensors);

  /**
   * @brief Run the with-comm NeighborGraph (message-passing, e.g. DPA2/DPA3)
   *        ``.pt2`` artifact with comm tensors appended.
   *
   * Positional AOTI input order mirrors ``run_model_graph``: ``(atype,
   * n_node, edge_index, edge_vec, edge_mask, [fparam], [aparam],
   * [charge_spin], comm_tensors...)``.  The graph is built on the
   * EXTENDED region (``fold_to_local=false``): ghost nodes are distinct and
   * their embeddings are filled in-place by ``border_op`` inside the
   * exported artifact, exactly as the with-comm dense/edge artifacts do.
   *
   * @param[in] atype Per-node extended-region types, shape ``(N,)`` int64,
   *            N == nall_real (owned prefix first, then halo).
   * @param[in] n_node Per-frame node count, shape ``(nf,)`` int64.
   * @param[in] edge_index Extended-region edge graph ``(2, E)`` int64
   *            [src, dst]; ghost-node indices retained (NOT folded to
   *            owners -- ``fold_to_local=false``).
   * @param[in] edge_vec Edge vectors ``(E, 3)`` (neighbour - center).
   * @param[in] edge_mask Physical-edge mask ``(E,)`` bool.
   * @param[in] aparam Atomic parameters on the flat EXTENDED node axis,
   *            shape ``(N, dim_aparam)`` (owned prefix filled, halo rows
   *            zero-padded -- ghost fitting outputs are masked inside the
   *            artifact), or an empty tensor when ``dim_aparam == 0``.
   * @param[in] comm_tensors 8 comm tensors in canonical positional order:
   *            send_list, send_proc, recv_proc, send_num, recv_num,
   *            communicator, nlocal, nghost.  All 8 stay on CPU (host
   *            control metadata for the opaque ``border_op``, symmetric
   *            with the dense with-comm artifact).
   * @param[in] n_local (1,) int64 owned-node count consumed IN-GRAPH by the
   *            owned-node energy mask; moved to the model device here (its
   *            consumer is a device kernel after ``move_to_device_pass``).
   *            Same value as the ``nlocal`` comm tensor -- the two inputs
   *            separate the device-compute role from the host-MPI-control
   *            role, so ``border_op`` never pulls device scalars.
   */
  std::vector<torch::Tensor> run_model_graph_with_comm(
      const torch::Tensor& atype,
      const torch::Tensor& n_node,
      const torch::Tensor& edge_index,
      const torch::Tensor& edge_vec,
      const torch::Tensor& edge_mask,
      const torch::Tensor& fparam,
      const torch::Tensor& aparam,
      const torch::Tensor& charge_spin,
      const std::vector<at::Tensor>& comm_tensors,
      const torch::Tensor& n_local);

  /**
   * @brief Extract outputs from flat tensor list using output_keys.
   */
  void extract_outputs(std::map<std::string, torch::Tensor>& output_map,
                       const std::vector<torch::Tensor>& flat_outputs);

  /**
   * @brief Translate PyTorch exceptions to DeePMD-kit exceptions.
   */
  void translate_error(std::function<void()> f);
};

}  // namespace deepmd

#endif  // BUILD_PT_EXPT
#endif  // BUILD_PYTORCH
