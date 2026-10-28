// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#ifdef BUILD_PYTORCH
// The AOTInductor package loader header is absent on some platforms (e.g.
// macOS x86_64); the native-spin compact backend is compiled out there.
#if __has_include(<torch/csrc/inductor/aoti_package/model_package_loader.h>)
#define BUILD_PT_EXPT_NATIVE_SPIN 1
#else
#define BUILD_PT_EXPT_NATIVE_SPIN 0
#endif

#if BUILD_PT_EXPT_NATIVE_SPIN

#include <torch/torch.h>

#include "DeepSpin.h"

// Forward-declare to keep the private header out of the public one. Defined in
// commonPTExpt.h.
namespace deepmd::ptexpt {
class ChargeStateFold;
}

namespace torch::inductor {
class AOTIModelPackageLoader;
}

namespace deepmd {

struct GraphTensorPack;
struct CanonicalGraphTensorPack;

/**
 * @brief PyTorch Exportable (AOTInductor .pt2) backend for the native spin
 * scheme.
 *
 * Native spin treats the magnetic moment as an equivariant descriptor input:
 * there are no virtual atoms and no atom doubling, so a node of the neighbor
 * graph is exactly one atom and the magnetic force is a per-node output next
 * to the conservative force. The scheme, declared by the archive as
 * ``spin_scheme == "native"``, is what selects this class; the lower-forward
 * schema the artifact was frozen with is an internal branch of it.
 *
 * Two schemas are served, both owned by
 * ``deepmd.pt_expt.model.native_spin_model.NativeSpinEnergyModel``:
 *
 * - ``lower_input_kind == "graph"``, the general NeighborGraph ABI
 *   (``forward_lower_graph_exportable``): the ten topology tensors, the
 *   per-node moment at positional index 10, then the conditional tail of
 *   frame parameter, atomic parameter and charge/spin condition. Any
 *   graph-lower descriptor can be frozen this way.
 * - ``lower_input_kind == "dpa4c_canonical"``, the compact deployment ABI
 *   (``forward_lower_canonical_graph_exportable``): the eight dual-CSR graph
 *   tensors -- uint32 topology and float32 edge vectors -- and the moment at
 *   positional index 8, with no conditional tail.
 *
 * Three entry points share the selected forward:
 *
 * - the standalone host ``computew`` (builds its own neighbor list),
 * - the LAMMPS host ``computew`` (consumes an ``InputNlist``),
 * - :meth:`compute_canonical_graph_gpu`, which takes an already device-resident
 *   graph and moment and writes its outputs device-to-device. Device residency
 *   is what the compact ABI exists for, so that entry point requires it.
 *
 * A single rank folds ghost neighbours onto their local owners, so the graph
 * carries ``nloc`` nodes. Domain decomposition keeps the extended
 * local-plus-ghost node set instead, so ghost force and magnetic force rows
 * survive to be folded onto their owners by LAMMPS reverse communication. That
 * layout gives a ghost node no owner to draw intermediate features from, so a
 * descriptor that exchanges them is confined to a single rank.
 **/
class NativeSpinPTExpt : public DeepSpinBackend {
 public:
  NativeSpinPTExpt();
  ~NativeSpinPTExpt() override;
  NativeSpinPTExpt(const std::string& model,
                   const int& gpu_rank = 0,
                   const std::string& file_content = "");
  /**
   * @brief Load a native-spin .pt2 archive frozen with either graph schema.
   * @param[in] model Path of the .pt2 model file.
   * @param[in] gpu_rank The GPU rank.
   * @param[in] file_content Unsupported for .pt2; must be empty.
   **/
  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "") override;

  double cutoff() const override {
    assert(inited);
    return rcut;
  };
  int numb_types() const override {
    assert(inited);
    return ntypes;
  };
  int numb_types_spin() const override {
    assert(inited);
    return ntypes_spin;
  };
  int dim_fparam() const override {
    assert(inited);
    return dfparam;
  };
  int dim_aparam() const override {
    assert(inited);
    return daparam;
  };
  /**
   * @brief The width of a charge/spin condition this model accepts.
   *
   * This is the width a caller names a condition with, which is not in
   * general the width of the conditioning input of the compiled forward:
   * compression folds the condition into frozen tables and so removes it from
   * the argument list, leaving a model that still serves a condition through
   * the fold shipped beside the inference lower.
   **/
  int dim_chg_spin() const override {
    assert(inited);
    return settable_chgspin;
  };
  /**
   * @brief Fix the charge/spin condition served for the rest of the run.
   *
   * The condition reaches the model by one of two routes, and this sets both
   * so that the caller does not depend on how the model was frozen.  It
   * becomes the condition of every later forward pass that is not given one
   * explicitly.  A compressed descriptor additionally carries the condition
   * inside frozen tables that the compiled lower holds as constants; when the
   * archive ships the fold that rebuilds them, it runs here and the resulting
   * tables are written over those constants.
   *
   * Intended to be called once, before inference, since overwriting the
   * constants of a loaded module is not safe to interleave with a forward
   * pass.
   *
   * @param[in] charge_spin The condition, of length ``dim_chg_spin()``.
   **/
  void set_charge_spin(const std::vector<double>& charge_spin) override;
  void get_type_map(std::string& type_map) override;
  bool is_aparam_nall() const override { return false; };
  bool has_default_fparam() const override {
    assert(inited);
    return has_default_fparam_;
  };
  std::vector<bool> get_use_spin() const override {
    assert(inited);
    return use_spin_;
  };

  void computew(std::vector<double>& ener,
                std::vector<double>& force,
                std::vector<double>& force_mag,
                std::vector<double>& virial,
                std::vector<double>& atom_energy,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<double>& spin,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const std::vector<double>& fparam,
                const std::vector<double>& aparam,
                const bool atomic) override;
  void computew(std::vector<double>& ener,
                std::vector<float>& force,
                std::vector<float>& force_mag,
                std::vector<float>& virial,
                std::vector<float>& atom_energy,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<float>& spin,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const std::vector<float>& fparam,
                const std::vector<float>& aparam,
                const bool atomic) override;
  void computew(std::vector<double>& ener,
                std::vector<double>& force,
                std::vector<double>& force_mag,
                std::vector<double>& virial,
                std::vector<double>& atom_energy,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<double>& spin,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const int nghost,
                const InputNlist& inlist,
                const int& ago,
                const std::vector<double>& fparam,
                const std::vector<double>& aparam,
                const bool atomic) override;
  void computew(std::vector<double>& ener,
                std::vector<float>& force,
                std::vector<float>& force_mag,
                std::vector<float>& virial,
                std::vector<float>& atom_energy,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<float>& spin,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const int nghost,
                const InputNlist& inlist,
                const int& ago,
                const std::vector<float>& fparam,
                const std::vector<float>& aparam,
                const bool atomic) override;

  // Charge/spin-aware overloads.  This backend serves the condition in force
  // rather than marshalling one per call, so a condition named here is
  // checked against that state and rejected when it names another; the
  // inherited defaults would drop it without a word.  An empty condition
  // selects the state in force.
  void computew(std::vector<double>& ener,
                std::vector<double>& force,
                std::vector<double>& force_mag,
                std::vector<double>& virial,
                std::vector<double>& atom_energy,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<double>& spin,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const std::vector<double>& fparam,
                const std::vector<double>& aparam,
                const std::vector<double>& charge_spin,
                const bool atomic) override;
  void computew(std::vector<double>& ener,
                std::vector<float>& force,
                std::vector<float>& force_mag,
                std::vector<float>& virial,
                std::vector<float>& atom_energy,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<float>& spin,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const std::vector<float>& fparam,
                const std::vector<float>& aparam,
                const std::vector<double>& charge_spin,
                const bool atomic) override;
  void computew(std::vector<double>& ener,
                std::vector<double>& force,
                std::vector<double>& force_mag,
                std::vector<double>& virial,
                std::vector<double>& atom_energy,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<double>& spin,
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
                std::vector<float>& force_mag,
                std::vector<float>& virial,
                std::vector<float>& atom_energy,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<float>& spin,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const int nghost,
                const InputNlist& inlist,
                const int& ago,
                const std::vector<float>& fparam,
                const std::vector<float>& aparam,
                const std::vector<double>& charge_spin,
                const bool atomic) override;

  /**
   * @brief Fully device-resident inference on a compact canonical graph.
   *
   * All pointers reference accelerator memory on the model device and every
   * output is written device-to-device. ``edge_storage`` is the allocated edge
   * capacity; the physical edge count is the last entry of
   * ``d_destination_row_ptr`` and the tail beyond it is ignored by the
   * artifact.
   *
   * @param[out] d_atom_energy Per-atom energy, GPU [nloc].
   * @param[out] d_force Per-node force, GPU [nall_nodes * 3] row-major.
   * @param[out] d_force_mag Per-node magnetic force, GPU [nall_nodes * 3]
   *   row-major.
   * @param[out] d_atom_virial Per-node virial, GPU [nall_nodes * 9] row-major.
   * @param[in] d_atype Per-node atom types, GPU [nall_nodes].
   * @param[in] d_source Source-node index per edge, GPU [edge_storage].
   * @param[in] d_edge_vec Destination-major edge vectors, GPU
   *   [edge_storage * 3].
   * @param[in] d_destination_row_ptr Destination CSR offsets, GPU
   *   [nall_nodes + 1].
   * @param[in] d_source_row_ptr Source CSR offsets, GPU [nall_nodes + 1].
   * @param[in] d_source_order Source-grouped edge positions, GPU
   *   [edge_storage].
   * @param[in] d_spin Per-node magnetic moment, GPU [nall_nodes * 3]
   *   row-major; ghost rows carry their owner's moment.
   * @param[in] nloc Number of local atoms.
   * @param[in] nall_nodes Graph node count (local + ghost).
   * @param[in] edge_storage Allocated edge capacity.
   */
  void compute_canonical_graph_gpu(double* d_atom_energy,
                                   double* d_force,
                                   double* d_force_mag,
                                   double* d_atom_virial,
                                   const std::int64_t* d_atype,
                                   const std::uint32_t* d_source,
                                   const float* d_edge_vec,
                                   const std::int64_t* d_destination_row_ptr,
                                   const std::int64_t* d_source_row_ptr,
                                   const std::uint32_t* d_source_order,
                                   const float* d_spin,
                                   const int nloc,
                                   const int nall_nodes,
                                   const std::int64_t edge_storage) override;

  bool uses_canonical_graph_inference() const override;

  bool uses_native_spin_scheme() const override;

 private:
  /**
   * @brief Evaluate with a pre-built neighbor list (LAMMPS path).
   *
   * The caller supplies the extended coordinates, so the cell plays no part
   * here. Returns extended per-atom force and magnetic force so that LAMMPS
   * reverse communication folds the ghost rows onto their owners.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& force_mag,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
               const std::vector<int>& atype,
               const int nghost,
               const InputNlist& lmp_list,
               const int& ago,
               const std::vector<VALUETYPE>& fparam,
               const std::vector<VALUETYPE>& aparam,
               const std::vector<double>& charge_spin,
               const bool atomic);
  /**
   * @brief Evaluate frames that arrive without a neighbor list.
   *
   * The inputs carry the frame count implicitly, in the length of the
   * coordinates; the outputs are the frames' results laid end to end. Every
   * frame brings its own cell and therefore its own ghost set, so they are
   * evaluated one at a time.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& force_mag,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam,
               const std::vector<VALUETYPE>& aparam,
               const std::vector<double>& charge_spin,
               const bool atomic);

  /**
   * @brief Whether a call may name the charge state it wants served.
   *
   * The two artifacts this class serves carry the condition differently. An
   * uncompressed one keeps it in the argument list of its compiled forward,
   * where a state reaches the evaluation that names it. A compressed one
   * folds it into frozen tables, which are rebuilt by ``set_charge_spin``
   * and stand for the whole run, so a call can only restate what they hold.
   * The width of the forward's argument tells the two apart.
   **/
  bool reads_charge_spin_per_call() const { return dchgspin > 0; }

  /**
   * @brief Evaluate one frame: build a neighbor list, then fold ghost
   * contributions back onto their local owners.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute_frame(ENERGYVTYPE& ener,
                     std::vector<VALUETYPE>& force,
                     std::vector<VALUETYPE>& force_mag,
                     std::vector<VALUETYPE>& virial,
                     std::vector<VALUETYPE>& atom_energy,
                     std::vector<VALUETYPE>& atom_virial,
                     const std::vector<VALUETYPE>& coord,
                     const std::vector<VALUETYPE>& spin,
                     const std::vector<int>& atype,
                     const std::vector<VALUETYPE>& box,
                     const std::vector<VALUETYPE>& fparam,
                     const std::vector<VALUETYPE>& aparam,
                     const std::vector<double>& charge_spin,
                     const bool atomic);

  /**
   * @brief Run the nine-input compact canonical native-spin forward.
   *
   * Positional order: the eight dual-CSR graph tensors, then the per-node
   * moment at index 8, which is the last slot -- the compact lower is traced
   * without a conditional tail.
   */
  std::vector<torch::Tensor> run_model_canonical(
      const CanonicalGraphTensorPack& graph, const torch::Tensor& spin);

  /**
   * @brief Run the NeighborGraph native-spin forward.
   *
   * Positional order: the ten NeighborGraph tensors, the per-node moment at
   * index 10, then the conditional tail -- the frame parameter, the atomic
   * parameter and the charge/spin condition, each present only when the model
   * declares a non-zero width.
   */
  std::vector<torch::Tensor> run_model_graph(const GraphTensorPack& graph,
                                             const torch::Tensor& spin,
                                             const torch::Tensor& fparam,
                                             const torch::Tensor& aparam,
                                             const torch::Tensor& charge_spin);

  /**
   * @brief Apply model-level pair exclusion, canonicalize the payload and run.
   *
   * The shared tail of both host paths: pair exclusion is a build-time
   * transform applied exactly once here, the destination-major
   * canonicalization follows, and the ABI branch decides whether the payload
   * is narrowed to the compact dual-CSR form or fed to the NeighborGraph
   * forward. The returned map holds the artifact's public output keys with the
   * per-atom virial in its ``(N, 9)`` layout.
   *
   * The charge/spin condition is not marshalled per call: the NeighborGraph
   * forward reads the condition currently in force, and the compact one
   * carries it in its frozen tables.
   *
   * @param[in,out] graph Graph payload for ``node_count`` nodes; consumed in
   *   place by the canonicalization.
   * @param[in] node_count Number of graph nodes.
   * @param[in] nloc Number of owned nodes, the prefix of the node axis.
   * @param[in] spin Per-node moment, shape ``(node_count, 3)``.
   * @param[in] fparam Frame parameter, ``dim_fparam`` values, or empty when
   *   the model carries a default or declares no width.
   * @param[in] aparam Atomic parameter, ``nloc * dim_aparam`` values, or empty
   *   when the model declares no width.
   */
  std::map<std::string, torch::Tensor> run_graph_payload(
      GraphTensorPack& graph,
      const std::int64_t node_count,
      const std::int64_t nloc,
      const torch::Tensor& spin,
      const std::vector<double>& fparam,
      const std::vector<double>& aparam,
      const std::vector<double>& charge_spin);

  /**
   * @brief Bind the flat artifact outputs to their metadata key names.
   */
  void extract_outputs(std::map<std::string, torch::Tensor>& output_map,
                       const std::vector<torch::Tensor>& flat_outputs);

  /**
   * @brief Translate PyTorch exceptions into DeePMD-kit exceptions.
   */
  void translate_error(std::function<void()> f);

  bool inited;
  // Every width below is a property of the loaded archive.  They read as zero
  // until ``init`` has run, so that a query on an uninitialised backend
  // answers "none" rather than whatever the allocation happened to hold.
  int ntypes = 0;
  int ntypes_spin = 0;
  int dfparam = 0;
  int daparam = 0;
  // Conditioning width of the compiled forward's argument list.  Zero for a
  // model that carries no condition, and also for a compressed one, whose
  // condition lives in frozen tables rather than in an input.  Every gate that
  // decides whether to hand the forward a condition tensor reads this.
  int dchgspin = 0;
  // Width of a charge state this model can be given; see ``dim_chg_spin()``.
  int settable_chgspin = 0;
  bool has_default_fparam_;
  std::vector<double> default_fparam_;
  // The condition served by every forward pass that is not given one
  // explicitly, initialised from the archive and replaced by
  // ``set_charge_spin``.
  std::vector<double> default_chg_spin_;
  /** Half-open row range of each charge-state value, from the archive. */
  std::vector<std::pair<double, double> > chg_spin_table_ranges_;
  double rcut;
  int gpu_id;
  bool gpu_enabled;
  // Which of the two schemas the loaded artifact declares: the compact
  // dual-CSR ABI when true, the general NeighborGraph ABI when false.
  bool canonical_abi_ = false;
  // Edge-vector precision the NeighborGraph artifact was traced with, read
  // from the ``graph_edge_dtype`` metadata field. The compact ABI is float32
  // by definition and validates that at load.
  bool graph_edge_fp32_ = false;
  // Whether the descriptor reads intermediate features of neighbouring nodes,
  // which an extended-region graph cannot supply for a ghost node.
  bool has_message_passing_ = false;
  std::vector<bool> use_spin_;
  std::vector<std::string> type_map;
  std::vector<std::string> output_keys;  // sorted internal output key names
  // Device-resident (ntypes+1)^2 model-level pair-type keep table, uploaded
  // once in ``init`` from the ``pair_exclude_types`` metadata field. An
  // UNDEFINED tensor means no exclusion. Exclusion belongs to the graph build:
  // the ingestion seam applies it exactly once and the exported lower consumes
  // a pre-excluded payload.
  torch::Tensor pair_exclude_table_;
  // Cached LAMMPS skin topology, rebuilt whenever ``ago == 0``. The
  // model-cutoff edge set is recomputed from it on-device every step.
  NeighborListData nlist_data;
  std::vector<std::int64_t> mapping_;
  at::Tensor edge_index_tensor;      // node-space edges (folded or extended)
  at::Tensor edge_index_ext_tensor;  // extended-atom edges, for the geometry
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
  // The charge/spin condition a compressed descriptor folded into the
  // constants of its lower, re-runnable so that a condition chosen at runtime
  // can be served.  Null for an uncompressed model, which reads its condition
  // as an ordinary input and needs no rebuild.
  std::unique_ptr<deepmd::ptexpt::ChargeStateFold> charge_state_fold_;
};

}  // namespace deepmd

#endif  // BUILD_PT_EXPT_NATIVE_SPIN
#endif  // BUILD_PYTORCH
