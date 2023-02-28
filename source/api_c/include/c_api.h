#pragma once
#ifdef __cplusplus
extern "C" {
#endif

/** 
 * @brief Neighbor list.
*/
typedef struct DP_Nlist DP_Nlist;

/**
* @brief Create a new neighbor list.
* @param[in] inum_ Number of core region atoms
* @param[in] Array stores the core region atom's index
* @param[in] Array stores the core region atom's neighbor atom number
* @param[in] Array stores the core region atom's neighbor index
* @returns A pointer to the neighbor list.
**/
extern DP_Nlist* DP_NewNlist(
  int inum_, 
  int * ilist_,
  int * numneigh_, 
  int ** firstneigh_);

/**
* @brief The deep potential.
**/
typedef struct DP_DeepPot DP_DeepPot;

/**
* @brief DP constructor with initialization.
* @param[in] c_model The name of the frozen model file.
* @returns A pointer to the deep potential.
**/
extern DP_DeepPot* DP_NewDeepPot(const char* c_model);

/**
* @brief Evaluate the energy, force and virial by using a DP. (double version)
* @param[in] dp The DP to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[out] energy Output energy.
* @param[out] force Output force. The array should be of size natoms x 3.
* @param[out] virial Output virial. The array should be of size 9.
* @param[out] atomic_energy Output atomic energy. The array should be of size natoms.
* @param[out] atomic_virial Output atomic virial. The array should be of size natoms x 9.
* @warning The output arrays should be allocated before calling this function. Pass NULL if not required.
  **/
extern void DP_DeepPotCompute (
  DP_DeepPot* dp,
  const int natom,
  const double* coord,
  const int* atype,
  const double* cell,
  double* energy,
  double* force,
  double* virial,
  double* atomic_energy,
  double* atomic_virial
  );

/**
* @brief Evaluate the energy, force and virial by using a DP. (float version)
* @param[in] dp The DP to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[out] energy Output energy.
* @param[out] force Output force. The array should be of size natoms x 3.
* @param[out] virial Output virial. The array should be of size 9.
* @param[out] atomic_energy Output atomic energy. The array should be of size natoms.
* @param[out] atomic_virial Output atomic virial. The array should be of size natoms x 9.
* @warning The output arrays should be allocated before calling this function. Pass NULL if not required.
  **/
extern void DP_DeepPotComputef (
  DP_DeepPot* dp,
  const int natom,
  const float* coord,
  const int* atype,
  const float* cell,
  double* energy,
  float* force,
  float* virial,
  float* atomic_energy,
  float* atomic_virial
  );

/**
* @brief Evaluate the energy, force and virial by using a DP with the neighbor list. (double version)
* @param[in] dp The DP to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[in] nghost The number of ghost atoms.
* @param[in] nlist The neighbor list.
* @param[in] ago Update the internal neighbour list if ago is 0.
* @param[out] energy Output energy.
* @param[out] force Output force. The array should be of size natoms x 3.
* @param[out] virial Output virial. The array should be of size 9.
* @param[out] atomic_energy Output atomic energy. The array should be of size natoms.
* @param[out] atomic_virial Output atomic virial. The array should be of size natoms x 9.
* @warning The output arrays should be allocated before calling this function. Pass NULL if not required.
  **/
extern void DP_DeepPotComputeNList (
  DP_DeepPot* dp,
  const int natom,
  const double* coord,
  const int* atype,
  const double* cell,
  const int nghost,
  const DP_Nlist* nlist,
  const int ago,
  double* energy,
  double* force,
  double* virial,
  double* atomic_energy,
  double* atomic_virial
  );

/**
* @brief Evaluate the energy, force and virial by using a DP with the neighbor list. (float version)
* @param[in] dp The DP to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[in] nghost The number of ghost atoms.
* @param[in] nlist The neighbor list.
* @param[in] ago Update the internal neighbour list if ago is 0.
* @param[out] energy Output energy.
* @param[out] force Output force. The array should be of size natoms x 3.
* @param[out] virial Output virial. The array should be of size 9.
* @param[out] atomic_energy Output atomic energy. The array should be of size natoms.
* @param[out] atomic_virial Output atomic virial. The array should be of size natoms x 9.
* @warning The output arrays should be allocated before calling this function. Pass NULL if not required.
  **/
extern void DP_DeepPotComputeNListf (
  DP_DeepPot* dp,
  const int natom,
  const float* coord,
  const int* atype,
  const float* cell,
  const int nghost,
  const DP_Nlist* nlist,
  const int ago,
  double* energy,
  float* force,
  float* virial,
  float* atomic_energy,
  float* atomic_virial
  );

/**
* @brief The deep potential model deviation.
**/
typedef struct DP_DeepPotModelDevi DP_DeepPotModelDevi;

/**
* @brief DP model deviation constructor with initialization.
* @param[in] c_models The array of the name of the frozen model file.
* @param[in] nmodels The number of models.
**/
extern DP_DeepPotModelDevi* DP_NewDeepPotModelDevi(const char** c_models, int n_models);

/**
* @brief Evaluate the energy, force and virial by using a DP model deviation with neighbor list. (double version)
* @param[in] dp The DP model deviation to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[in] nghost The number of ghost atoms.
* @param[in] nlist The neighbor list.
* @param[in] ago Update the internal neighbour list if ago is 0.
* @param[out] energy Output energy.
* @param[out] force Output force. The array should be of size natoms x 3.
* @param[out] virial Output virial. The array should be of size 9.
* @param[out] atomic_energy Output atomic energy. The array should be of size natoms.
* @param[out] atomic_virial Output atomic virial. The array should be of size natoms x 9.
* @warning The output arrays should be allocated before calling this function. Pass NULL if not required.
  **/
extern void DP_DeepPotModelDeviComputeNList (
  DP_DeepPotModelDevi* dp,
  const int natom,
  const double* coord,
  const int* atype,
  const double* cell,
  const int nghost,
  const DP_Nlist* nlist,
  const int ago,
  double* energy,
  double* force,
  double* virial,
  double* atomic_energy,
  double* atomic_virial
  );

/**
* @brief Evaluate the energy, force and virial by using a DP model deviation with neighbor list. (float version)
* @param[in] dp The DP model deviation to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[in] nghost The number of ghost atoms.
* @param[in] nlist The neighbor list.
* @param[in] ago Update the internal neighbour list if ago is 0.
* @param[out] energy Output energy.
* @param[out] force Output force. The array should be of size natoms x 3.
* @param[out] virial Output virial. The array should be of size 9.
* @param[out] atomic_energy Output atomic energy. The array should be of size natoms.
* @param[out] atomic_virial Output atomic virial. The array should be of size natoms x 9.
* @warning The output arrays should be allocated before calling this function. Pass NULL if not required.
  **/
extern void DP_DeepPotModelDeviComputeNListf (
  DP_DeepPotModelDevi* dp,
  const int natom,
  const float* coord,
  const int* atype,
  const float* cell,
  const int nghost,
  const DP_Nlist* nlist,
  const int ago,
  double* energy,
  float* force,
  float* virial,
  float* atomic_energy,
  float* atomic_virial
  );

/**
 * @brief Get the type map of a DP model deviation.
 * @param[in] dp The DP model deviation to use.
 * @return The cutoff radius.
*/
double DP_DeepPotModelDeviGetCutoff(DP_DeepPotModelDevi* dp);

/**
 * @brief Get the type map of a DP model deviation.
 * @param[in] dp The DP model deviation to use.
 * @return The number of types of the DP model deviation.
*/
int DP_DeepPotModelDeviGetNumbTypes(DP_DeepPotModelDevi* dp);

/**
 * @brief Get the type map of a DP.
 * @param[in] dp The DP to use.
 * @return The cutoff radius.
*/
double DP_DeepPotGetCutoff(DP_DeepPot* dp);

/**
 * @brief Get the type map of a DP.
 * @param[in] dp The DP to use.
 * @return The number of types of the DP.
*/
int DP_DeepPotGetNumbTypes(DP_DeepPot* dp);

/**
 * @brief Get the type map of a DP.
 * @param[in] dp The DP to use.
 * @return The type map of the DP.
*/
const char* DP_DeepPotGetTypeMap(DP_DeepPot* dp);

/**
* @brief The deep tensor.
**/
typedef struct DP_DeepTensor DP_DeepTensor;

/**
* @brief Deep Tensor constructor with initialization.
* @param[in] c_model The name of the frozen model file.
* @returns A pointer to the deep tensor.
**/
extern DP_DeepTensor* DP_NewDeepTensor(const char* c_model);

/**
* @brief Evaluate the tensor by using a DP. (double version)
* @param[in] dt The Deep Tensor to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[out] tensor Output tensor.
  **/
extern void DP_DeepTensorComputeTensor (
  DP_DeepTensor* dt,
  const int natom,
  const double* coord,
  const int* atype,
  const double* cell,
  double** tensor,
  int* size
  );

/**
* @brief Evaluate the tensor by using a DP. (float version)
* @param[in] dt The Deep Tensor to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[out] tensor Output tensor.
* @param[out] size Output size of the tensor.
  **/
extern void DP_DeepTensorComputeTensorf (
  DP_DeepTensor* dt,
  const int natom,
  const float* coord,
  const int* atype,
  const float* cell,
  float** tensor,
  int* size
  );

/**
* @brief Evaluate the tensor by using a DP with the neighbor list. (double version)
* @param[in] dt The Deep Tensor to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[in] nghost The number of ghost atoms.
* @param[in] nlist The neighbor list.
* @param[out] tensor Output tensor.
* @param[out] size Output size of the tensor.
  **/
extern void DP_DeepTensorComputeTensorNList (
  DP_DeepTensor* dt,
  const int natom,
  const double* coord,
  const int* atype,
  const double* cell,
  const int nghost,
  const DP_Nlist* nlist,
  double** tensor,
  int* size
  );

/**
* @brief Evaluate the tensor by using a DP with the neighbor list. (float version)
* @param[in] dt The Deep Tensor to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[in] nghost The number of ghost atoms.
* @param[in] nlist The neighbor list.
* @param[out] tensor Output tensor.
* @param[out] size Output size of the tensor.
  **/
extern void DP_DeepTensorComputeTensorNListf (
  DP_DeepTensor* dt,
  const int natom,
  const float* coord,
  const int* atype,
  const float* cell,
  const int nghost,
  const DP_Nlist* nlist,
  float** tensor,
  int* size
  );

/**
* @brief Evaluate the global tensor, force and virial by using a DP. (double version)
* @param[in] dt The Deep Tensor to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[out] global_tensor Output global tensor.
* @param[out] force Output force. The array should be of size natoms x 3.
* @param[out] virial Output virial. The array should be of size 9.
* @param[out] atomic_tensor Output atomic tensor. The array should be of size natoms.
* @param[out] atomic_virial Output atomic virial. The array should be of size natoms x 9.
* @param[out] size_at Output size of atomic tensor.
* @warning The output arrays should be allocated before calling this function. Pass NULL if not required.
  **/
extern void DP_DeepTensorCompute (
  DP_DeepTensor* dt,
  const int natom,
  const double* coord,
  const int* atype,
  const double* cell,
  double* global_tensor,
  double* force,
  double* virial,
  double** atomic_tensor,
  double* atomic_virial,
  int* size_at
  );

/**
* @brief Evaluate the global tensor, force and virial by using a DP. (float version)
* @param[in] dt The Deep Tensor to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[out] global_tensor Output global tensor.
* @param[out] force Output force. The array should be of size natoms x 3.
* @param[out] virial Output virial. The array should be of size 9.
* @param[out] atomic_tensor Output atomic tensor. The array should be of size natoms.
* @param[out] atomic_virial Output atomic virial. The array should be of size natoms x 9.
* @param[out] size_at Output size of atomic tensor.
* @warning The output arrays should be allocated before calling this function. Pass NULL if not required.
  **/
extern void DP_DeepTensorComputef (
  DP_DeepTensor* dt,
  const int natom,
  const float* coord,
  const int* atype,
  const float* cell,
  float* global_tensor,
  float* force,
  float* virial,
  float** atomic_tensor,
  float* atomic_virial,
  int* size_at
  );

/**
* @brief Evaluate the global tensor, force and virial by using a DP with the neighbor list. (double version)
* @param[in] dt The Deep Tensor to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[in] nghost The number of ghost atoms.
* @param[in] nlist The neighbor list.
* @param[out] global_tensor Output global tensor.
* @param[out] force Output force. The array should be of size natoms x 3.
* @param[out] virial Output virial. The array should be of size 9.
* @param[out] atomic_tensor Output atomic tensor. The array should be of size natoms.
* @param[out] atomic_virial Output atomic virial. The array should be of size natoms x 9.
* @param[out] size_at Output size of atomic tensor.
* @warning The output arrays should be allocated before calling this function. Pass NULL if not required.
  **/
extern void DP_DeepTensorComputeNList (
  DP_DeepTensor* dt,
  const int natom,
  const double* coord,
  const int* atype,
  const double* cell,
  const int nghost,
  const DP_Nlist* nlist,
  double* global_tensor,
  double* force,
  double* virial,
  double** atomic_tensor,
  double* atomic_virial,
  int* size_at
  );

/**
* @brief Evaluate the global tensor, force and virial by using a DP with the neighbor list. (float version)
* @param[in] dt The Deep Tensor to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size 9. Pass NULL if pbc is not used.
* @param[in] nghost The number of ghost atoms.
* @param[in] nlist The neighbor list.
* @param[out] global_tensor Output global tensor.
* @param[out] force Output force. The array should be of size natoms x 3.
* @param[out] virial Output virial. The array should be of size 9.
* @param[out] atomic_tensor Output atomic tensor. The array should be of size natoms.
* @param[out] atomic_virial Output atomic virial. The array should be of size natoms x 9.
* @param[out] size_at Output size of atomic tensor.
* @warning The output arrays should be allocated before calling this function. Pass NULL if not required.
  **/
extern void DP_DeepTensorComputeNListf (
  DP_DeepTensor* dt,
  const int natom,
  const float* coord,
  const int* atype,
  const float* cell,
  const int nghost,
  const DP_Nlist* nlist,
  float* global_tensor,
  float* force,
  float* virial,
  float** atomic_tensor,
  float* atomic_virial,
  int* size_at
  );

/**
 * @brief Get the type map of a Deep Tensor.
 * @param[in] dt The Deep Tensor to use.
 * @return The cutoff radius.
*/
double DP_DeepTensorGetCutoff(DP_DeepTensor* dt);

/**
 * @brief Get the type map of a Deep Tensor.
 * @param[in] dt The Deep Tensor to use.
 * @return The number of types of the Deep Tensor.
*/
int DP_DeepTensorGetNumbTypes(DP_DeepTensor* dt);

/**
 * @brief Get the output dimension of a Deep Tensor.
 * @param[in] dt The Deep Tensor to use.
 * @return The output dimension of the Deep Tensor.
*/
int DP_DeepTensorGetOutputDim(DP_DeepTensor* dt);

/**
 * @brief Get sel types of a Deep Tensor.
 * @param[in] dt The Deep Tensor to use.
 * @return The sel types
*/
int* DP_DeepTensorGetSelTypes(DP_DeepTensor* dt);

/**
 * @brief Get the number of sel types of a Deep Tensor.
 * @param[in] dt The Deep Tensor to use.
 * @return The number of sel types
*/
int DP_DeepTensorGetNumbSelTypes(DP_DeepTensor* dt);

/**
* @brief Convert PBtxt to PB.
* @param[in] c_pbtxt The name of the PBtxt file.
* @param[in] c_pb The name of the PB file.
  **/
extern void DP_ConvertPbtxtToPb(
  const char* c_pbtxt,
  const char* c_pb
  );

#ifdef __cplusplus
} /* end extern "C" */
#endif