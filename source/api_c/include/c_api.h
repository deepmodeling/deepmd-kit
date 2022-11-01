#ifdef __cplusplus
extern "C" {
#endif

/**
* @brief The deep potential.
**/
typedef struct DP_DeepPot DP_DeepPot;

/**
* @brief DP constructor with initialization.
* @param[in] c_model The name of the frozen model file.
**/
extern DP_DeepPot* DP_NewDeepPot(const char* c_model);

/**
* @brief Evaluate the energy, force and virial by using a DP. (double version)
* @param[in] dp The DP to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size nframes x 9.
* @param[out] energy Output energy.
* @param[out] force Output force.
* @param[out] virial Output virial.
  **/
extern void DP_DeepPotCompute (
  DP_DeepPot* dp,
  const int natom,
  const double* coord,
  const int* atype,
  const double* cell,
  double* energy,
  double* force,
  double* virial
  );

/**
* @brief Evaluate the energy, force and virial by using a DP. (float version)
* @param[in] dp The DP to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size nframes x 9.
* @param[out] energy Output energy.
* @param[out] force Output force.
* @param[out] virial Output virial.
  **/
extern void DP_DeepPotComputef (
  DP_DeepPot* dp,
  const int natom,
  const float* coord,
  const int* atype,
  const float* cell,
  double* energy,
  float* force,
  float* virial
  );

#ifdef __cplusplus
} /* end extern "C" */
#endif