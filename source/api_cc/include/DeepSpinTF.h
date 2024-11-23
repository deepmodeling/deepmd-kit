// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "DeepSpin.h"
#include "common.h"
#include "commonTF.h"
#include "neighbor_list.h"

namespace deepmd {
/**
 * @brief TensorFlow implementation for Deep Potential.
 **/
class DeepSpinTF : public DeepSpinBackend {
 public:
  /**
   * @brief DP constructor without initialization.
   **/
  DeepSpinTF();
  virtual ~DeepSpinTF();
  /**
   * @brief DP constructor with initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  DeepSpinTF(const std::string& model,
             const int& gpu_rank = 0,
             const std::string& file_content = "");
  /**
   * @brief Initialize the DP.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "");

 private:
  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] atomic Whether to compute atomic energy and virial.
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
               const bool atomic);
  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] lmp_list The input neighbour list.
   * @param[in] ago Update the internal neighbour list if ago is 0.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] atomic Whether to compute atomic energy and virial.
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
               const int nghost,
               const InputNlist& lmp_list,
               const int& ago,
               const std::vector<VALUETYPE>& fparam,
               const std::vector<VALUETYPE>& aparam,
               const bool atomic);

 public:
  /**
   * @brief Get the cutoff radius.
   * @return The cutoff radius.
   **/
  double cutoff() const {
    assert(inited);
    return rcut;
  };
  /**
   * @brief Get the number of types.
   * @return The number of types.
   **/
  int numb_types() const {
    assert(inited);
    return ntypes;
  };
  /**
   * @brief Get the number of types with spin.
   * @return The number of types with spin.
   **/
  int numb_types_spin() const {
    assert(inited);
    return ntypes_spin;
  };
  /**
   * @brief Get the dimension of the frame parameter.
   * @return The dimension of the frame parameter.
   **/
  int dim_fparam() const {
    assert(inited);
    return dfparam;
  };
  /**
   * @brief Get the dimension of the atomic parameter.
   * @return The dimension of the atomic parameter.
   **/
  int dim_aparam() const {
    assert(inited);
    return daparam;
  };
  /**
   * @brief Get the type map (element name of the atom types) of this model.
   * @param[out] type_map The type map of this model.
   **/
  void get_type_map(std::string& type_map);

  /**
   * @brief Get whether the atom dimension of aparam is nall instead of fparam.
   * @param[out] aparam_nall whether the atom dimension of aparam is nall
   *instead of fparam.
   **/
  bool is_aparam_nall() const {
    assert(inited);
    return aparam_nall;
  };

  // forward to template class
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
                const bool atomic);
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
                const bool atomic);
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
                const bool atomic);
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
                const bool atomic);

  template <typename VALUETYPE>
  void extend(int& extend_inum,
              std::vector<int>& extend_ilist,
              std::vector<int>& extend_numneigh,
              std::vector<std::vector<int>>& extend_neigh,
              std::vector<int*>& extend_firstneigh,
              std::vector<VALUETYPE>& extend_dcoord,
              std::vector<int>& extend_atype,
              int& extend_nghost,
              std::map<int, int>& new_idx_map,
              std::map<int, int>& old_idx_map,
              const InputNlist& lmp_list,
              const std::vector<VALUETYPE>& dcoord,
              const std::vector<int>& atype,
              const int nghost,
              const std::vector<VALUETYPE>& spin,
              const int numb_types,
              const int numb_types_spin);

  template <typename VALUETYPE>
  void extend_nlist(std::vector<VALUETYPE>& extend_dcoord,
                    std::vector<int>& extend_atype,
                    const std::vector<VALUETYPE>& dcoord_,
                    const std::vector<VALUETYPE>& dspin_,
                    const std::vector<int>& datype_);

  void cum_sum(std::map<int, int>&, std::map<int, int>&);

 private:
  tensorflow::Session* session;
  int num_intra_nthreads, num_inter_nthreads;
  tensorflow::GraphDef* graph_def;
  bool inited;
  template <class VT>
  VT get_scalar(const std::string& name) const;
  template <class VT>
  void get_vector(std::vector<VT>& vec, const std::string& name) const;

  double rcut;
  int dtype;
  double cell_size;
  std::string model_type;
  std::string model_version;
  int ntypes;
  int ntypes_spin;
  std::vector<double> virtual_len;
  std::vector<double> spin_norm;
  int extend_inum;
  std::vector<int> extend_ilist;
  std::vector<int> extend_numneigh;
  std::vector<std::vector<int>> extend_neigh;
  std::vector<int*> extend_firstneigh;
  // std::vector<double> extend_dcoord;
  std::vector<int> extend_dtype;
  int extend_nghost;
  // for spin systems, search new index of atoms by their old index
  std::map<int, int> new_idx_map;
  std::map<int, int> old_idx_map;
  int dfparam;
  int daparam;
  bool aparam_nall;
  /**
   * @brief Validate the size of frame and atomic parameters.
   * @param[in] nframes The number of frames.
   * @param[in] nloc The number of local atoms.
   * @param[in] fparam The frame parameter.
   * @param[in] aparam The atomic parameter.
   * @tparam VALUETYPE The type of the parameters, double or float.
   */
  template <typename VALUETYPE>
  void validate_fparam_aparam(const int& nframes,
                              const int& nloc,
                              const std::vector<VALUETYPE>& fparam,
                              const std::vector<VALUETYPE>& aparam) const;
  /**
   * @brief Tile the frame or atomic parameters if there is only
   * a single frame of frame or atomic parameters.
   * @param[out] out_param The tiled frame or atomic parameters.
   * @param[in] nframes The number of frames.
   * @param[in] dparam The dimension of the frame or atomic parameters in a
   * frame.
   * @param[in] param The frame or atomic parameters.
   * @tparam VALUETYPE The type of the parameters, double or float.
   */
  template <typename VALUETYPE>
  void tile_fparam_aparam(std::vector<VALUETYPE>& out_param,
                          const int& nframes,
                          const int& dparam,
                          const std::vector<VALUETYPE>& param) const;
  // copy neighbor list info from host
  bool init_nbor;
  std::vector<int> sec_a;
  NeighborListData nlist_data;
  InputNlist nlist;
  AtomMap atommap;
};

}  // namespace deepmd
