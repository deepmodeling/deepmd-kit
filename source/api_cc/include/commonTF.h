// SPDX-License-Identifier: LGPL-3.0-or-later
#include <string>
#include <vector>

#ifdef TF_PRIVATE
#include "tf_private.h"
#else
#include "tf_public.h"
#endif

namespace deepmd {
/**
 * @brief Check TensorFlow status. Exit if not OK.
 * @param[in] status TensorFlow status.
 **/
void check_status(const tensorflow::Status& status);

/**
 * @brief Get the value of a tensor.
 * @param[in] session TensorFlow session.
 * @param[in] name The name of the tensor.
 * @param[in] scope The scope of the tensor.
 * @return The value of the tensor.
 **/
template <typename VT>
VT session_get_scalar(tensorflow::Session* session,
                      const std::string name,
                      const std::string scope = "");

/**
 * @brief Get the vector of a tensor.
 * @param[out] o_vec The output vector.
 * @param[in] session TensorFlow session.
 * @param[in] name The name of the tensor.
 * @param[in] scope The scope of the tensor.
 **/
template <typename VT>
void session_get_vector(std::vector<VT>& o_vec,
                        tensorflow::Session* session,
                        const std::string name_,
                        const std::string scope = "");

/**
 * @brief Get the type of a tensor.
 * @param[in] session TensorFlow session.
 * @param[in] name The name of the tensor.
 * @param[in] scope The scope of the tensor.
 * @return The type of the tensor as int.
 **/
int session_get_dtype(tensorflow::Session* session,
                      const std::string name,
                      const std::string scope = "");

/**
 * @brief Get input tensors.
 * @param[out] input_tensors Input tensors.
 * @param[in] dcoord_ Coordinates of atoms.
 * @param[in] ntypes Number of atom types.
 * @param[in] datype_ Atom types.
 * @param[in] dbox Box matrix.
 * @param[in] cell_size Cell size.
 * @param[in] fparam_ Frame parameters.
 * @param[in] aparam_ Atom parameters.
 * @param[in] atommap Atom map.
 * @param[in] scope The scope of the tensors.
 * @param[in] aparam_nall Whether the atomic dimesion of atomic parameters is
 * nall.
 */
template <typename MODELTYPE, typename VALUETYPE>
int session_input_tensors(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const std::vector<VALUETYPE>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    const double& cell_size,
    const std::vector<VALUETYPE>& fparam_,
    const std::vector<VALUETYPE>& aparam_,
    const deepmd::AtomMap& atommap,
    const std::string scope = "",
    const bool aparam_nall = false);

/**
 * @brief Get input tensors.
 * @param[out] input_tensors Input tensors.
 * @param[in] dcoord_ Coordinates of atoms.
 * @param[in] ntypes Number of atom types.
 * @param[in] datype_ Atom types.
 * @param[in] dlist Neighbor list.
 * @param[in] fparam_ Frame parameters.
 * @param[in] aparam_ Atom parameters.
 * @param[in] atommap Atom map.
 * @param[in] nghost Number of ghost atoms.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[in] scope The scope of the tensors.
 * @param[in] aparam_nall Whether the atomic dimesion of atomic parameters is
 * nall.
 */
template <typename MODELTYPE, typename VALUETYPE>
int session_input_tensors(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const std::vector<VALUETYPE>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    InputNlist& dlist,
    const std::vector<VALUETYPE>& fparam_,
    const std::vector<VALUETYPE>& aparam_,
    const deepmd::AtomMap& atommap,
    const int nghost,
    const int ago,
    const std::string scope = "",
    const bool aparam_nall = false);

/**
 * @brief Get input tensors for mixed type.
 * @param[out] input_tensors Input tensors.
 * @param[in] nframes Number of frames.
 * @param[in] dcoord_ Coordinates of atoms.
 * @param[in] ntypes Number of atom types.
 * @param[in] datype_ Atom types.
 * @param[in] dlist Neighbor list.
 * @param[in] fparam_ Frame parameters.
 * @param[in] aparam_ Atom parameters.
 * @param[in] atommap Atom map.
 * @param[in] nghost Number of ghost atoms.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[in] scope The scope of the tensors.
 * @param[in] aparam_nall Whether the atomic dimesion of atomic parameters is
 * nall.
 */
template <typename MODELTYPE, typename VALUETYPE>
int session_input_tensors_mixed_type(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const int& nframes,
    const std::vector<VALUETYPE>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    const double& cell_size,
    const std::vector<VALUETYPE>& fparam_,
    const std::vector<VALUETYPE>& aparam_,
    const deepmd::AtomMap& atommap,
    const std::string scope = "",
    const bool aparam_nall = false);

}  // namespace deepmd
