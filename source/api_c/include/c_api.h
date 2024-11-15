// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#ifdef __cplusplus
extern "C" {
#else
// for C99
#include <stdbool.h>
#endif

/** @file */

/** C API version. Bumped whenever the API is changed.
 * @since API version 22
 */
#define DP_C_API_VERSION 25

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
extern DP_Nlist* DP_NewNlist(int inum_,
                             int* ilist_,
                             int* numneigh_,
                             int** firstneigh_);
/**
 * @brief Create a new neighbor list with communication capabilities.
 * @details This function extends DP_NewNlist by adding support for parallel
 * communication, allowing the neighbor list to be used in distributed
 * environments.
 * @param[in] inum_ Number of core region atoms.
 * @param[in] ilist_ Array storing the core region atom's index.
 * @param[in] numneigh_ Array storing the core region atom's neighbor atom
 * number.
 * @param[in] firstneigh_ Array storing the core region atom's neighbor index.
 * @param[in] nswap Number of swaps to be performed in communication.
 * @param[in] sendnum Array storing the number of atoms to send for each swap.
 * @param[in] recvnum Array storing the number of atoms to receive for each
 * swap.
 * @param[in] firstrecv Index of the first receive operation for each swap.
 * @param[in] sendlist List of atoms to be sent for each swap.
 * @param[in] sendproc Array of processor IDs to send atoms to for each swap.
 * @param[in] recvproc Array of processor IDs from which atoms are received for
 * each swap.
 * @param[in] world Pointer to the MPI communicator or similar communication
 * world used for the operation.
 * @returns A pointer to the initialized neighbor list with communication
 * capabilities.
 */
extern DP_Nlist* DP_NewNlist_comm(int inum_,
                                  int* ilist_,
                                  int* numneigh_,
                                  int** firstneigh_,
                                  int nswap,
                                  int* sendnum,
                                  int* recvnum,
                                  int* firstrecv,
                                  int** sendlist,
                                  int* sendproc,
                                  int* recvproc,
                                  void* world);

/**
 * @brief Set mask for a neighbor list.
 *
 * @param nl Neighbor list.
 * @param mask mask.
 * @since API version 23
 *
 **/
extern void DP_NlistSetMask(DP_Nlist* nl, int mask);

/**
 * @brief Set mapping for a neighbor list.
 *
 * @param nl Neighbor list.
 * @param mapping mapping from all atoms to real atoms, in size nall.
 * @since API version 25
 *
 **/
extern void DP_NlistSetMapping(DP_Nlist* nl, int* mapping);

/**
 * @brief Delete a neighbor list.
 *
 * @param nl Neighbor list to delete.
 *
 **/
extern void DP_DeleteNlist(DP_Nlist* nl);

/**
 * @brief Check if there is any exceptions throw.
 *
 * @param dp The neighbor list to use.
 * @return const char* error message.
 */
const char* DP_NlistCheckOK(DP_Nlist* dp);

/**
 * @brief The deep potential base model.
 * @since API version 24
 **/
typedef struct DP_DeepBaseModel DP_DeepBaseModel;

/**
 * @brief Delete a Deep Potential Base Model.
 *
 * @param dp Deep Potential Base Model to delete.
 * @since API version 24
 */
extern void DP_DeleteDeepBaseModel(DP_DeepBaseModel* dp);

/**
 * @brief The deep potential base model deviation.
 * @since API version 24
 **/
typedef struct DP_DeepBaseModelDevi DP_DeepBaseModelDevi;

/**
 * @brief Delete a Deep Potential Base Model Deviation.
 *
 * @param dp Deep Potential Base Model Deviation to delete.
 * @since API version 24
 */
extern void DP_DeleteDeepBaseModelDevi(DP_DeepBaseModelDevi* dp);

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
 * @brief DP constructor with initialization.
 *
 * @param c_model The name of the frozen model file.
 * @param gpu_rank The rank of the GPU.
 * @param c_file_content Broken implementation. Use
 * DP_NewDeepPotWithParam2 instead.
 * @return DP_DeepPot* A pointer to the deep potential.
 */
extern DP_DeepPot* DP_NewDeepPotWithParam(const char* c_model,
                                          const int gpu_rank,
                                          const char* c_file_content);

/**
 * @brief DP constructor with initialization.
 * @version 2
 * @param c_model The name of the frozen model file.
 * @param gpu_rank The rank of the GPU.
 * @param c_file_content The content of the model file.
 * @param size_file_content The size of the model file.
 * @return DP_DeepPot* A pointer to the deep potential.
 */
extern DP_DeepPot* DP_NewDeepPotWithParam2(const char* c_model,
                                           const int gpu_rank,
                                           const char* c_file_content,
                                           const int size_file_content);

/**
 * @brief Delete a Deep Potential.
 *
 * @param dp Deep Potential to delete.
 */
extern void DP_DeleteDeepPot(DP_DeepPot* dp);

/**
 * @brief The deep potential spin model.
 * @since API version 24
 **/
typedef struct DP_DeepSpin DP_DeepSpin;

/**
 * @brief DP constructor with initialization.
 * @param[in] c_model The name of the frozen model file.
 * @returns A pointer to the deep potential spin model.
 * @since API version 24
 **/
extern DP_DeepSpin* DP_NewDeepSpin(const char* c_model);

/**
 * @brief DP constructor with initialization.
 * @version 2
 * @param c_model The name of the frozen model file.
 * @param gpu_rank The rank of the GPU.
 * @param c_file_content The content of the model file.
 * @param size_file_content The size of the model file.
 * @return DP_DeepSpin* A pointer to the deep potential spin model.
 * @since API version 24
 */
extern DP_DeepSpin* DP_NewDeepSpinWithParam2(const char* c_model,
                                             const int gpu_rank,
                                             const char* c_file_content,
                                             const int size_file_content);

/**
 * @brief Delete a Deep Potential Spin Model.
 *
 * @param dp Deep Potential Spin Model to delete.
 * @since API version 24
 */
extern void DP_DeleteDeepSpin(DP_DeepSpin* dp);

// deprecated interface version1
/**
 * @brief Evaluate the energy, force and virial by using a DP. (double version)
 * @attention The number of frames is assumed to be 1.
 * @param[in] dp The DP to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotCompute(DP_DeepPot* dp,
                              const int natom,
                              const double* coord,
                              const int* atype,
                              const double* cell,
                              double* energy,
                              double* force,
                              double* virial,
                              double* atomic_energy,
                              double* atomic_virial);

/**
 * @brief Evaluate the energy, force and virial by using a DP. (float version)
 * @attention The number of frames is assumed to be 1.
 * @param[in] dp The DP to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotComputef(DP_DeepPot* dp,
                               const int natom,
                               const float* coord,
                               const int* atype,
                               const float* cell,
                               double* energy,
                               float* force,
                               float* virial,
                               float* atomic_energy,
                               float* atomic_virial);

/**
 * @brief Evaluate the energy, force and virial by using a DP with the neighbor
 *list. (double version)
 * @attention The number of frames is assumed to be 1.
 * @param[in] dp The DP to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotComputeNList(DP_DeepPot* dp,
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
                                   double* atomic_virial);

/**
 * @brief Evaluate the energy, force and virial by using a DP with the neighbor
 *list. (float version)
 * @attention The number of frames is assumed to be 1.
 * @param[in] dp The DP to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotComputeNListf(DP_DeepPot* dp,
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
                                    float* atomic_virial);

/**
 * @brief Evaluate the energy, force and virial by using a DP. (double version)
 * @version 2
 * @param[in] dp The DP to use.
 * @param[in] nframes The number of frames.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 * natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotCompute2(DP_DeepPot* dp,
                               const int nframes,
                               const int natom,
                               const double* coord,
                               const int* atype,
                               const double* cell,
                               const double* fparam,
                               const double* aparam,
                               double* energy,
                               double* force,
                               double* virial,
                               double* atomic_energy,
                               double* atomic_virial);

/**
 * @brief Evaluate the energy, force, magnetic force and virial by using a DP
 * spin model. (double version)
 * @version 2
 * @param[in] dp The DP spin model to use.
 * @param[in] nframes The number of frames.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should be
 *of size nframes x natoms x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 * natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] force_mag Output magnetic force on each atom. The array should be
 * of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 * @since API version 24
 **/
extern void DP_DeepSpinCompute2(DP_DeepSpin* dp,
                                const int nframes,
                                const int natom,
                                const double* coord,
                                const double* spin,
                                const int* atype,
                                const double* cell,
                                const double* fparam,
                                const double* aparam,
                                double* energy,
                                double* force,
                                double* force_mag,
                                double* virial,
                                double* atomic_energy,
                                double* atomic_virial);

/**
 * @brief Evaluate the energy, force and virial by using a DP. (float version)
 * @version 2
 * @param[in] dp The DP to use.
 * @param[in] nframes The number of frames.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 * natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotComputef2(DP_DeepPot* dp,
                                const int nframes,
                                const int natom,
                                const float* coord,
                                const int* atype,
                                const float* cell,
                                const float* fparam,
                                const float* aparam,
                                double* energy,
                                float* force,
                                float* virial,
                                float* atomic_energy,
                                float* atomic_virial);

/**
 * @brief Evaluate the energy, force, magnetic force and virial by using a DP
 * spin model. (float version)
 * @version 2
 * @param[in] dp The DP to use.
 * @param[in] nframes The number of frames.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should be
 *of size nframes x natoms x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 * natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] force_mag Output magnetic force on each atom. The array should be
 * of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 * @since API version 24
 **/
extern void DP_DeepSpinComputef2(DP_DeepSpin* dp,
                                 const int nframes,
                                 const int natom,
                                 const float* coord,
                                 const float* spin,
                                 const int* atype,
                                 const float* cell,
                                 const float* fparam,
                                 const float* aparam,
                                 double* energy,
                                 float* force,
                                 float* force_mag,
                                 float* virial,
                                 float* atomic_energy,
                                 float* atomic_virial);

/**
 * @brief Evaluate the energy, force and virial by using a DP with the neighbor
 *list. (double version)
 * @version 2
 * @param[in] dp The DP to use.
 * @param[in] nframes The number of frames.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 * natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotComputeNList2(DP_DeepPot* dp,
                                    const int nframes,
                                    const int natom,
                                    const double* coord,
                                    const int* atype,
                                    const double* cell,
                                    const int nghost,
                                    const DP_Nlist* nlist,
                                    const int ago,
                                    const double* fparam,
                                    const double* aparam,
                                    double* energy,
                                    double* force,
                                    double* virial,
                                    double* atomic_energy,
                                    double* atomic_virial);

/**
 * @brief Evaluate the energy, force, magnetic force and virial by using a DP
 *spin model with the neighbor list. (double version)
 * @version 2
 * @param[in] dp The DP spin model to use.
 * @param[in] nframes The number of frames.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should be
 *of size nframes x natoms x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 * natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] force_mag Output magnetic force on each atom. The array should be
 * of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 * @since API version 24
 **/
extern void DP_DeepSpinComputeNList2(DP_DeepSpin* dp,
                                     const int nframes,
                                     const int natoms,
                                     const double* coord,
                                     const double* spin,
                                     const int* atype,
                                     const double* cell,
                                     const int nghost,
                                     const DP_Nlist* nlist,
                                     const int ago,
                                     const double* fparam,
                                     const double* aparam,
                                     double* energy,
                                     double* force,
                                     double* force_mag,
                                     double* virial,
                                     double* atomic_energy,
                                     double* atomic_virial);

/**
 * @brief Evaluate the energy, force and virial by using a DP with the neighbor
 *list. (float version)
 * @version 2
 * @param[in] dp The DP to use.
 * @param[in] nframes The number of frames.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 * natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotComputeNListf2(DP_DeepPot* dp,
                                     const int nframes,
                                     const int natom,
                                     const float* coord,
                                     const int* atype,
                                     const float* cell,
                                     const int nghost,
                                     const DP_Nlist* nlist,
                                     const int ago,
                                     const float* fparam,
                                     const float* aparam,
                                     double* energy,
                                     float* force,
                                     float* virial,
                                     float* atomic_energy,
                                     float* atomic_virial);

/**
 * @brief Evaluate the energy, force, magnetic force and virial by using a DP
 *spin model with the neighbor list. (float version)
 * @version 2
 * @param[in] dp The DP spin model to use.
 * @param[in] nframes The number of frames.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should be
 *of size nframes x natoms x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 * natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] force_mag Output magnetic force on each atom. The array should be
 * of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 * @since API version 24
 **/
extern void DP_DeepSpinComputeNListf2(DP_DeepSpin* dp,
                                      const int nframes,
                                      const int natoms,
                                      const float* coord,
                                      const float* spin,
                                      const int* atype,
                                      const float* cell,
                                      const int nghost,
                                      const DP_Nlist* nlist,
                                      const int ago,
                                      const float* fparam,
                                      const float* aparam,
                                      double* energy,
                                      float* force,
                                      float* force_mag,
                                      float* virial,
                                      float* atomic_energy,
                                      float* atomic_virial);

/**
 * @brief Evaluate the energy, force and virial by using a DP with the mixed
 *type. (double version)
 * @param[in] dp The DP to use.
 * @param[in] nframes The number of frames.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain nframes x natoms
 *ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 * natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotComputeMixedType(DP_DeepPot* dp,
                                       const int nframes,
                                       const int natoms,
                                       const double* coord,
                                       const int* atype,
                                       const double* cell,
                                       const double* fparam,
                                       const double* aparam,
                                       double* energy,
                                       double* force,
                                       double* virial,
                                       double* atomic_energy,
                                       double* atomic_virial);
/**
 * @brief Evaluate the energy, force and virial by using a DP with the mixed
 *type. (float version)
 * @param[in] dp The DP to use.
 * @param[in] nframes The number of frames.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain nframes x natoms
 *ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 * natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotComputeMixedTypef(DP_DeepPot* dp,
                                        const int nframes,
                                        const int natoms,
                                        const float* coord,
                                        const int* atype,
                                        const float* cell,
                                        const float* fparam,
                                        const float* aparam,
                                        double* energy,
                                        float* force,
                                        float* virial,
                                        float* atomic_energy,
                                        float* atomic_virial);

/**
 * @brief The deep potential model deviation.
 **/
typedef struct DP_DeepPotModelDevi DP_DeepPotModelDevi;

/**
 * @brief DP model deviation constructor with initialization.
 * @param[in] c_models The array of the name of the frozen model file.
 * @param[in] nmodels The number of models.
 **/
extern DP_DeepPotModelDevi* DP_NewDeepPotModelDevi(const char** c_models,
                                                   int n_models);

/**
 * @brief DP model deviation constructor with initialization.
 *
 * @param[in] c_models The array of the name of the frozen model file.
 * @param[in] nmodels The number of models.
 * @param[in] gpu_rank The rank of the GPU.
 * @param[in] c_file_contents The contents of the model file.
 * @param[in] n_file_contents The number of the contents of the model file.
 * @param[in] size_file_contents The sizes of the contents of the model file.
 * @return DP_DeepPotModelDevi* A pointer to the deep potential model deviation.
 */
extern DP_DeepPotModelDevi* DP_NewDeepPotModelDeviWithParam(
    const char** c_model,
    const int n_models,
    const int gpu_rank,
    const char** c_file_contents,
    const int n_file_contents,
    const int* size_file_contents);

/**
 * @brief Delete a Deep Potential Model Deviation.
 *
 * @param dp Deep Potential to delete.
 */
extern void DP_DeleteDeepPotModelDevi(DP_DeepPotModelDevi* dp);

/**
 * @brief The deep potential spin model deviation.
 * @since API version 24
 **/
typedef struct DP_DeepSpinModelDevi DP_DeepSpinModelDevi;

/**
 * @brief DP spin model deviation constructor with initialization.
 * @param[in] c_models The array of the name of the frozen model file.
 * @param[in] nmodels The number of models.
 * @since API version 24
 **/
extern DP_DeepSpinModelDevi* DP_NewDeepSpinModelDevi(const char** c_models,
                                                     int n_models);

/**
 * @brief DP spin model deviation constructor with initialization.
 *
 * @param[in] c_models The array of the name of the frozen model file.
 * @param[in] nmodels The number of models.
 * @param[in] gpu_rank The rank of the GPU.
 * @param[in] c_file_contents The contents of the model file.
 * @param[in] n_file_contents The number of the contents of the model file.
 * @param[in] size_file_contents The sizes of the contents of the model file.
 * @return DP_DeepSpinModelDevi* A pointer to the deep potential model
 * deviation.
 * @since API version 24
 */
extern DP_DeepSpinModelDevi* DP_NewDeepSpinModelDeviWithParam(
    const char** c_model,
    const int n_models,
    const int gpu_rank,
    const char** c_file_contents,
    const int n_file_contents,
    const int* size_file_contents);

/**
 * @brief Delete a Deep Potential Spin Model Deviation.
 *
 * @param dp Deep Potential Spin Model to delete.
 * @since API version 24
 */
extern void DP_DeleteDeepSpinModelDevi(DP_DeepSpinModelDevi* dp);

// deprecated interface version1
/**
 * @brief Evaluate the energy, force and virial by using a DP model deviation
 *with neighbor list. (double version)
 * @param[in] dp The DP model deviation to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotModelDeviCompute(DP_DeepPotModelDevi* dp,
                                       const int natom,
                                       const double* coord,
                                       const int* atype,
                                       const double* cell,
                                       double* energy,
                                       double* force,
                                       double* virial,
                                       double* atomic_energy,
                                       double* atomic_virial);

/**
 * @brief Evaluate the energy, force and virial by using a DP model deviation
 *with neighbor list. (float version)
 * @param[in] dp The DP model deviation to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotModelDeviComputef(DP_DeepPotModelDevi* dp,
                                        const int natom,
                                        const float* coord,
                                        const int* atype,
                                        const float* cell,
                                        double* energy,
                                        float* force,
                                        float* virial,
                                        float* atomic_energy,
                                        float* atomic_virial);

/**
 * @brief Evaluate the energy, force and virial by using a DP model deviation
 *with neighbor list. (double version)
 * @version 2
 * @param[in] dp The DP model deviation to use.
 * @param[in] nframes The number of frames. Only support 1 for now.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 *natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
void DP_DeepPotModelDeviCompute2(DP_DeepPotModelDevi* dp,
                                 const int nframes,
                                 const int natoms,
                                 const double* coord,
                                 const int* atype,
                                 const double* cell,
                                 const double* fparam,
                                 const double* aparam,
                                 double* energy,
                                 double* force,
                                 double* virial,
                                 double* atomic_energy,
                                 double* atomic_virial);

/**
 * @brief Evaluate the energy, force, magnetic force and virial by using a DP
 * spin model deviation. (double version)
 * @version 2
 * @param[in] dp The DP spin model deviation to use.
 * @param[in] nframes The number of frames. Only support 1 for now.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should be
 *of size nframes x natoms x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 *natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] force_mag Output magnetic force on each atom. The array should be
 * of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 * @since API version 24
 **/
void DP_DeepSpinModelDeviCompute2(DP_DeepSpinModelDevi* dp,
                                  const int nframes,
                                  const int natoms,
                                  const double* coord,
                                  const double* spin,
                                  const int* atype,
                                  const double* cell,
                                  const double* fparam,
                                  const double* aparam,
                                  double* energy,
                                  double* force,
                                  double* force_mag,
                                  double* virial,
                                  double* atomic_energy,
                                  double* atomic_virial);
/**
 * @brief Evaluate the energy, force and virial by using a DP model deviation
 *with neighbor list. (float version)
 * @version 2
 * @param[in] dp The DP model deviation to use.
 * @param[in] nframes The number of frames. Only support 1 for now.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 *natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
void DP_DeepPotModelDeviComputef2(DP_DeepPotModelDevi* dp,
                                  const int nframes,
                                  const int natoms,
                                  const float* coord,
                                  const int* atype,
                                  const float* cell,
                                  const float* fparam,
                                  const float* aparam,
                                  double* energy,
                                  float* force,
                                  float* virial,
                                  float* atomic_energy,
                                  float* atomic_virial);

/**
 * @brief Evaluate the energy, force, magnetic force and virial by using a DP
 * spin model deviation. (float version)
 * @version 2
 * @param[in] dp The DP spin model deviation to use.
 * @param[in] nframes The number of frames. Only support 1 for now.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should be
 *of size nframes x natoms x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 *natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] force_mag Output magnetic force on each atom. The array should be
 * of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 * @since API version 24
 **/
void DP_DeepSpinModelDeviComputef2(DP_DeepSpinModelDevi* dp,
                                   const int nframes,
                                   const int natoms,
                                   const float* coord,
                                   const float* spin,
                                   const int* atype,
                                   const float* cell,
                                   const float* fparam,
                                   const float* aparam,
                                   double* energy,
                                   float* force,
                                   float* force_mag,
                                   float* virial,
                                   float* atomic_energy,
                                   float* atomic_virial);

// deprecated interface version1
/**
 * @brief Evaluate the energy, force and virial by using a DP model deviation
 *with neighbor list. (double version)
 * @param[in] dp The DP model deviation to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotModelDeviComputeNList(DP_DeepPotModelDevi* dp,
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
                                            double* atomic_virial);

/**
 * @brief Evaluate the energy, force and virial by using a DP model deviation
 *with neighbor list. (float version)
 * @param[in] dp The DP model deviation to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepPotModelDeviComputeNListf(DP_DeepPotModelDevi* dp,
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
                                             float* atomic_virial);

/**
 * @brief Evaluate the energy, force and virial by using a DP model deviation
 *with neighbor list. (double version)
 * @version 2
 * @param[in] dp The DP model deviation to use.
 * @param[in] nframes The number of frames. Only support 1 for now.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 *natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
void DP_DeepPotModelDeviComputeNList2(DP_DeepPotModelDevi* dp,
                                      const int nframes,
                                      const int natoms,
                                      const double* coord,
                                      const int* atype,
                                      const double* cell,
                                      const int nghost,
                                      const DP_Nlist* nlist,
                                      const int ago,
                                      const double* fparam,
                                      const double* aparam,
                                      double* energy,
                                      double* force,
                                      double* virial,
                                      double* atomic_energy,
                                      double* atomic_virial);

/**
 * @brief Evaluate the energy, force, magnetic force and virial by using a DP
 *spin model deviation with neighbor list. (double version)
 * @version 2
 * @param[in] dp The DP model deviation to use.
 * @param[in] nframes The number of frames. Only support 1 for now.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should be
 *of size nframes x natoms x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 *natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] force_mag Output magnetic force on each atom. The array should be
 * of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 * @since API version 24
 **/
void DP_DeepSpinModelDeviComputeNList2(DP_DeepSpinModelDevi* dp,
                                       const int nframes,
                                       const int natoms,
                                       const double* coord,
                                       const double* spin,
                                       const int* atype,
                                       const double* cell,
                                       const int nghost,
                                       const DP_Nlist* nlist,
                                       const int ago,
                                       const double* fparam,
                                       const double* aparam,
                                       double* energy,
                                       double* force,
                                       double* force_mag,
                                       double* virial,
                                       double* atomic_energy,
                                       double* atomic_virial);

/**
 * @brief Evaluate the energy, force and virial by using a DP model deviation
 *with neighbor list. (float version)
 * @version 2
 * @param[in] dp The DP model deviation to use.
 * @param[in] nframes The number of frames. Only support 1 for now.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 *natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
void DP_DeepPotModelDeviComputeNListf2(DP_DeepPotModelDevi* dp,
                                       const int nframes,
                                       const int natoms,
                                       const float* coord,
                                       const int* atype,
                                       const float* cell,
                                       const int nghost,
                                       const DP_Nlist* nlist,
                                       const int ago,
                                       const float* fparam,
                                       const float* aparam,
                                       double* energy,
                                       float* force,
                                       float* virial,
                                       float* atomic_energy,
                                       float* atomic_virial);

/**
 * @brief Evaluate the energy, force, magnetic force and virial by using a DP
 *spin model deviation with neighbor list. (float version)
 * @version 2
 * @param[in] dp The DP model deviation to use.
 * @param[in] nframes The number of frames. Only support 1 for now.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should be
 *of size nframes x natoms x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[in] ago Update the internal neighbour list if ago is 0.
 * @param[in] fparam The frame parameters. The array can be of size nframes x
 *dim_fparam.
 * @param[in] aparam The atom parameters. The array can be of size nframes x
 *natoms x dim_aparam.
 * @param[out] energy Output energy.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] force_mag Output magnetic force on each atom. The array should be
 * of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_energy Output atomic energy. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 * @since API version 24
 **/
void DP_DeepSpinModelDeviComputeNListf2(DP_DeepSpinModelDevi* dp,
                                        const int nframes,
                                        const int natoms,
                                        const float* coord,
                                        const float* spin,
                                        const int* atype,
                                        const float* cell,
                                        const int nghost,
                                        const DP_Nlist* nlist,
                                        const int ago,
                                        const float* fparam,
                                        const float* aparam,
                                        double* energy,
                                        float* force,
                                        float* force_mag,
                                        float* virial,
                                        float* atomic_energy,
                                        float* atomic_virial);

// Deep Base Model methods
/**
 * @brief Get the cutoff of a DP.
 * @param[in] dpbase The DP to use.
 * @return The cutoff radius.
 * @since API version 24
 */
double DP_DeepBaseModelGetCutoff(DP_DeepBaseModel* dpbase);

/**
 * @brief Get the number of types of a DP.
 * @param[in] dpbase The DP to use.
 * @return The number of types of the DP.
 * @since API version 24
 */
int DP_DeepBaseModelGetNumbTypes(DP_DeepBaseModel* dpbase);

/**
 * @brief Get the number of types with spin of a DP.
 * @param[in] dpbase The DP to use.
 * @return The number of types with spin of the DP.
 * @since API version 24
 */
int DP_DeepBaseModelGetNumbTypesSpin(DP_DeepBaseModel* dpbase);

/**
 * @brief Get the dimension of frame parameters of a DP.
 * @param[in] dpbase The DP to use.
 * @return The dimension of frame parameters of the DP.
 * @since API version 24
 */
int DP_DeepBaseModelGetDimFParam(DP_DeepBaseModel* dpbase);

/**
 * @brief Get the dimension of atomic parameters of a DP.
 * @param[in] dpbase The DP to use.
 * @return The dimension of atomic parameters of the DP.
 * @since API version 24
 */
int DP_DeepBaseModelGetDimAParam(DP_DeepBaseModel* dpbase);

/**
 * @brief Check whether the atomic dimension of atomic parameters is nall
 * instead of nloc.
 *
 * @param[in] dpbase The DP to use.
 * @return true the atomic dimension of atomic parameters is nall
 * @return false the atomic dimension of atomic parameters is nloc
 * @since API version 24
 */
bool DP_DeepBaseModelIsAParamNAll(DP_DeepBaseModel* dpbase);

/**
 * @brief Get the type map of a DP.
 * @param[in] dpbase The DP to use.
 * @return The type map of the DP.
 * @since API version 24
 */
const char* DP_DeepBaseModelGetTypeMap(DP_DeepBaseModel* dpbase);

/**
 * @brief Check if there is any exceptions throw.
 *
 * @param dpbase The DP to use.
 * @return const char* error message.
 * @since API version 24
 */
const char* DP_DeepBaseModelCheckOK(DP_DeepBaseModel* dpbase);

/**
 * @brief Get the dimension of frame parameters of a DP Model Deviation.
 * @param[in] dpbase The DP Model Deviation to use.
 * @return The dimension of frame parameters of the DP Model Deviation.
 * @since API version 24
 */
int DP_DeepBaseModelDeviGetDimFParam(DP_DeepBaseModelDevi* dpbase);
/**
 * @brief Get the dimension of atomic parameters of a DP Model Deviation.
 * @param[in] dpbase The DP Model Deviation to use.
 * @return The dimension of atomic parameters of the DP Model Deviation.
 * @since API version 24
 */
int DP_DeepBaseModelDeviGetDimAParam(DP_DeepBaseModelDevi* dpbase);

/**
 * @brief Check whether the atomic dimension of atomic parameters is nall
 * instead of nloc.
 *
 * @param[in] dpbase The DP Model Deviation to use.
 * @return true the atomic dimension of atomic parameters is nall
 * @return false the atomic dimension of atomic parameters is nloc
 * @since API version 24
 */
bool DP_DeepBaseModelDeviIsAParamNAll(DP_DeepBaseModelDevi* dpbase);

/**
 * @brief Get the type map of a DP model deviation.
 * @param[in] dpbase The DP model deviation to use.
 * @return The cutoff radius.
 * @since API version 24
 */
double DP_DeepBaseModelDeviGetCutoff(DP_DeepBaseModelDevi* dpbase);

/**
 * @brief Get the number of types of a DP model deviation.
 * @param[in] dpbase The DP model deviation to use.
 * @return The number of types of the DP model deviation.
 * @since API version 24
 */
int DP_DeepBaseModelDeviGetNumbTypes(DP_DeepBaseModelDevi* dpbase);

/**
 * @brief Get the number of types with spin of a DP model deviation.
 * @param[in] dpbase The DP model deviation to use.
 * @return The number of types with spin of the DP model deviation.
 * @since API version 24
 */
int DP_DeepBaseModelDeviGetNumbTypesSpin(DP_DeepBaseModelDevi* dpbase);

/**
 * @brief Check if there is any exceptions throw.
 *
 * @param dpbase The DP model deviation to use.
 * @return const char* error message.
 * @since API version 24
 */
const char* DP_DeepBaseModelDeviCheckOK(DP_DeepBaseModelDevi* dpbase);

// DeepPot methods for c_api
/**
 * @brief Get the cutoff of a DP.
 * @param[in] dp The DP to use.
 * @return The cutoff radius.
 */
double DP_DeepPotGetCutoff(DP_DeepPot* dp);

/**
 * @brief Get the number of types of a DP.
 * @param[in] dp The DP to use.
 * @return The number of types of the DP.
 */
int DP_DeepPotGetNumbTypes(DP_DeepPot* dp);

/**
 * @brief Get the number of types with spin of a DP.
 * @param[in] dp The DP to use.
 * @return The number of types with spin of the DP.
 */
int DP_DeepPotGetNumbTypesSpin(DP_DeepPot* dp);

/**
 * @brief Get the dimension of frame parameters of a DP.
 * @param[in] dp The DP to use.
 * @return The dimension of frame parameters of the DP.
 */
int DP_DeepPotGetDimFParam(DP_DeepPot* dp);

/**
 * @brief Get the dimension of atomic parameters of a DP.
 * @param[in] dp The DP to use.
 * @return The dimension of atomic parameters of the DP.
 */
int DP_DeepPotGetDimAParam(DP_DeepPot* dp);

/**
 * @brief Check whether the atomic dimension of atomic parameters is nall
 * instead of nloc.
 *
 * @param[in] dp The DP to use.
 * @return true the atomic dimension of atomic parameters is nall
 * @return false the atomic dimension of atomic parameters is nloc
 */
bool DP_DeepPotIsAParamNAll(DP_DeepPot* dp);

/**
 * @brief Get the type map of a DP.
 * @param[in] dp The DP to use.
 * @return The type map of the DP.
 */
const char* DP_DeepPotGetTypeMap(DP_DeepPot* dp);

/**
 * @brief Check if there is any exceptions throw.
 *
 * @param dp The DP to use.
 * @return const char* error message.
 */
const char* DP_DeepPotCheckOK(DP_DeepPot* dp);

/**
 * @brief Get the dimension of frame parameters of a DP Model Deviation.
 * @param[in] dp The DP Model Deviation to use.
 * @return The dimension of frame parameters of the DP Model Deviation.
 */
int DP_DeepPotModelDeviGetDimFParam(DP_DeepPotModelDevi* dp);
/**
 * @brief Get the dimension of atomic parameters of a DP Model Deviation.
 * @param[in] dp The DP Model Deviation to use.
 * @return The dimension of atomic parameters of the DP Model Deviation.
 */
int DP_DeepPotModelDeviGetDimAParam(DP_DeepPotModelDevi* dp);

/**
 * @brief Check whether the atomic dimension of atomic parameters is nall
 * instead of nloc.
 *
 * @param[in] dp The DP Model Deviation to use.
 * @return true the atomic dimension of atomic parameters is nall
 * @return false the atomic dimension of atomic parameters is nloc
 */
bool DP_DeepPotModelDeviIsAParamNAll(DP_DeepPotModelDevi* dp);

/**
 * @brief Get the type map of a DP model deviation.
 * @param[in] dp The DP model deviation to use.
 * @return The cutoff radius.
 */
double DP_DeepPotModelDeviGetCutoff(DP_DeepPotModelDevi* dp);

/**
 * @brief Get the number of types of a DP model deviation.
 * @param[in] dp The DP model deviation to use.
 * @return The number of types of the DP model deviation.
 */
int DP_DeepPotModelDeviGetNumbTypes(DP_DeepPotModelDevi* dp);

/**
 * @brief Get the number of types with spin of a DP model deviation.
 * @param[in] dp The DP model deviation to use.
 * @return The number of types with spin of the DP model deviation.
 */
int DP_DeepPotModelDeviGetNumbTypesSpin(DP_DeepPotModelDevi* dp);

/**
 * @brief Check if there is any exceptions throw.
 *
 * @param dp The DP model deviation to use.
 * @return const char* error message.
 */
const char* DP_DeepPotModelDeviCheckOK(DP_DeepPotModelDevi* dp);

// DeepSpin methods for c_api
/**
 * @brief Get the cutoff of a DP Spin Model.
 * @param[in] dp The DP Spin Model to use.
 * @return The cutoff radius.
 * @since API version 24
 */
double DP_DeepSpinGetCutoff(DP_DeepSpin* dp);

/**
 * @brief Get the number of types of a DP Spin Model.
 * @param[in] dp The DP Spin Model to use.
 * @return The number of types of the DP Spin Model.
 * @since API version 24
 */
int DP_DeepSpinGetNumbTypes(DP_DeepSpin* dp);

/**
 * @brief Get the number of types with spin of a DP Spin Model.
 * @param[in] dp The DP Spin Model to use.
 * @return The number of types with spin of the DP Spin Model.
 * @since API version 24
 */
int DP_DeepSpinGetNumbTypesSpin(DP_DeepSpin* dp);

/**
 * @brief Get the dimension of frame parameters of a DP Spin Model.
 * @param[in] dp The DP Spin Model to use.
 * @return The dimension of frame parameters of the DP Spin Model.
 * @since API version 24
 */
int DP_DeepSpinGetDimFParam(DP_DeepSpin* dp);

/**
 * @brief Get the dimension of atomic parameters of a DP Spin Model.
 * @param[in] dp The DP Spin Model to use.
 * @return The dimension of atomic parameters of the DP Spin Model.
 * @since API version 24
 */
int DP_DeepSpinGetDimAParam(DP_DeepSpin* dp);

/**
 * @brief Check whether the atomic dimension of atomic parameters is nall
 * instead of nloc.
 *
 * @param[in] dp The DP Spin Model to use.
 * @return true the atomic dimension of atomic parameters is nall
 * @return false the atomic dimension of atomic parameters is nloc
 * @since API version 24
 */
bool DP_DeepSpinIsAParamNAll(DP_DeepSpin* dp);

/**
 * @brief Get the type map of a DP Spin Model.
 * @param[in] dp The DP Spin Model to use.
 * @return The type map of the DP Spin Model.
 * @since API version 24
 */
const char* DP_DeepSpinGetTypeMap(DP_DeepSpin* dp);

/**
 * @brief Check if there is any exceptions throw.
 *
 * @param dp The DP Spin Model to use.
 * @return const char* error message.
 * @since API version 24
 */
const char* DP_DeepSpinCheckOK(DP_DeepSpin* dp);

/**
 * @brief Get the dimension of frame parameters of a DP Spin Model Deviation.
 * @param[in] dp The DP Spin Model Deviation to use.
 * @return The dimension of frame parameters of the DP Spin Model Deviation.
 * @since API version 24
 */
int DP_DeepSpinModelDeviGetDimFParam(DP_DeepSpinModelDevi* dp);
/**
 * @brief Get the dimension of atomic parameters of a DP Spin Model Deviation.
 * @param[in] dp The DP Spin Model Deviation to use.
 * @return The dimension of atomic parameters of the DP Spin Model Deviation.
 * @since API version 24
 */
int DP_DeepSpinModelDeviGetDimAParam(DP_DeepSpinModelDevi* dp);

/**
 * @brief Check whether the atomic dimension of atomic parameters is nall
 * instead of nloc.
 *
 * @param[in] dp The DP Spin Model Deviation to use.
 * @return true the atomic dimension of atomic parameters is nall
 * @return false the atomic dimension of atomic parameters is nloc
 * @since API version 24
 */
bool DP_DeepSpinModelDeviIsAParamNAll(DP_DeepSpinModelDevi* dp);

/**
 * @brief Get the type map of a DP model deviation.
 * @param[in] dp The DP model deviation to use.
 * @return The cutoff radius.
 * @since API version 24
 */
double DP_DeepSpinModelDeviGetCutoff(DP_DeepSpinModelDevi* dp);

/**
 * @brief Get the number of types of a DP model deviation.
 * @param[in] dp The DP model deviation to use.
 * @return The number of types of the DP model deviation.
 * @since API version 24
 */
int DP_DeepSpinModelDeviGetNumbTypes(DP_DeepSpinModelDevi* dp);

/**
 * @brief Get the number of types with spin of a DP model deviation.
 * @param[in] dp The DP model deviation to use.
 * @return The number of types with spin of the DP model deviation.
 * @since API version 24
 */
int DP_DeepSpinModelDeviGetNumbTypesSpin(DP_DeepSpinModelDevi* dp);

/**
 * @brief Check if there is any exceptions throw.
 *
 * @param dp The DP model deviation to use.
 * @return const char* error message.
 * @since API version 24
 */
const char* DP_DeepSpinModelDeviCheckOK(DP_DeepSpinModelDevi* dp);

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
 * @brief Deep Tensor constructor with initialization.
 *
 * @param c_model The name of the frozen model file.
 * @param gpu_rank The rank of the GPU.
 * @param c_name_scope The name scope.
 * @return DP_DeepTensor* A pointer to the deep tensor.
 */
extern DP_DeepTensor* DP_NewDeepTensorWithParam(const char* c_model,
                                                const int gpu_rank,
                                                const char* c_name_scope);

/**
 * @brief Delete a Deep Tensor.
 *
 * @param dp Deep Tensor to delete.
 */
extern void DP_DeleteDeepTensor(DP_DeepTensor* dt);

/**
 * @brief Evaluate the tensor by using a DP. (double version)
 * @param[in] dt The Deep Tensor to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[out] tensor Output tensor.
 **/
extern void DP_DeepTensorComputeTensor(DP_DeepTensor* dt,
                                       const int natom,
                                       const double* coord,
                                       const int* atype,
                                       const double* cell,
                                       double** tensor,
                                       int* size);

/**
 * @brief Evaluate the tensor by using a DP. (float version)
 * @param[in] dt The Deep Tensor to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[out] tensor Output tensor.
 * @param[out] size Output size of the tensor.
 **/
extern void DP_DeepTensorComputeTensorf(DP_DeepTensor* dt,
                                        const int natom,
                                        const float* coord,
                                        const int* atype,
                                        const float* cell,
                                        float** tensor,
                                        int* size);

/**
 * @brief Evaluate the tensor by using a DP with the neighbor list. (double
 *version)
 * @param[in] dt The Deep Tensor to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[out] tensor Output tensor.
 * @param[out] size Output size of the tensor.
 **/
extern void DP_DeepTensorComputeTensorNList(DP_DeepTensor* dt,
                                            const int natom,
                                            const double* coord,
                                            const int* atype,
                                            const double* cell,
                                            const int nghost,
                                            const DP_Nlist* nlist,
                                            double** tensor,
                                            int* size);

/**
 * @brief Evaluate the tensor by using a DP with the neighbor list. (float
 *version)
 * @param[in] dt The Deep Tensor to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[out] tensor Output tensor.
 * @param[out] size Output size of the tensor.
 **/
extern void DP_DeepTensorComputeTensorNListf(DP_DeepTensor* dt,
                                             const int natom,
                                             const float* coord,
                                             const int* atype,
                                             const float* cell,
                                             const int nghost,
                                             const DP_Nlist* nlist,
                                             float** tensor,
                                             int* size);

/**
 * @brief Evaluate the global tensor, force and virial by using a DP. (double
 *version)
 * @param[in] dt The Deep Tensor to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[out] global_tensor Output global tensor.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_tensor Output atomic tensor. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @param[out] size_at Output size of atomic tensor.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepTensorCompute(DP_DeepTensor* dt,
                                 const int natom,
                                 const double* coord,
                                 const int* atype,
                                 const double* cell,
                                 double* global_tensor,
                                 double* force,
                                 double* virial,
                                 double** atomic_tensor,
                                 double* atomic_virial,
                                 int* size_at);

/**
 * @brief Evaluate the global tensor, force and virial by using a DP. (float
 *version)
 * @param[in] dt The Deep Tensor to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[out] global_tensor Output global tensor.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_tensor Output atomic tensor. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @param[out] size_at Output size of atomic tensor.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepTensorComputef(DP_DeepTensor* dt,
                                  const int natom,
                                  const float* coord,
                                  const int* atype,
                                  const float* cell,
                                  float* global_tensor,
                                  float* force,
                                  float* virial,
                                  float** atomic_tensor,
                                  float* atomic_virial,
                                  int* size_at);

/**
 * @brief Evaluate the global tensor, force and virial by using a DP with the
 *neighbor list. (double version)
 * @param[in] dt The Deep Tensor to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[out] global_tensor Output global tensor.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_tensor Output atomic tensor. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @param[out] size_at Output size of atomic tensor.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepTensorComputeNList(DP_DeepTensor* dt,
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
                                      int* size_at);

/**
 * @brief Evaluate the global tensor, force and virial by using a DP with the
 *neighbor list. (float version)
 * @param[in] dt The Deep Tensor to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] box The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[out] global_tensor Output global tensor.
 * @param[out] force Output force. The array should be of size natoms x 3.
 * @param[out] virial Output virial. The array should be of size 9.
 * @param[out] atomic_tensor Output atomic tensor. The array should be of size
 *natoms.
 * @param[out] atomic_virial Output atomic virial. The array should be of size
 *natoms x 9.
 * @param[out] size_at Output size of atomic tensor.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DeepTensorComputeNListf(DP_DeepTensor* dt,
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
                                       int* size_at);

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
 * @brief Get the type map of a Deep Tensor.
 * @param[in] dt The Deep Tensor to use.
 * @return The type map of the Deep Tensor.
 */
const char* DP_DeepTensorGetTypeMap(DP_DeepTensor* dt);

/**
 * @brief Check if there is any exceptions throw.
 *
 * @param dt The Deep Tensor to use.
 * @return const char* error message.
 */
const char* DP_DeepTensorCheckOK(DP_DeepTensor* dt);

/**
 * @brief The dipole charge modifier.
 **/
typedef struct DP_DipoleChargeModifier DP_DipoleChargeModifier;

/**
 * @brief Dipole charge modifier constructor with initialization.
 * @param[in] c_model The name of the frozen model file.
 * @returns A pointer to the dipole charge modifier.
 **/
extern DP_DipoleChargeModifier* DP_NewDipoleChargeModifier(const char* c_model);

/**
 * @brief Dipole charge modifier constructor with initialization.
 *
 * @param c_model The name of the frozen model file.
 * @param gpu_rank The rank of the GPU.
 * @param c_name_scope The name scope.
 * @return DP_DipoleChargeModifier* A pointer to the dipole charge modifier.
 */
extern DP_DipoleChargeModifier* DP_NewDipoleChargeModifierWithParam(
    const char* c_model, const int gpu_rank, const char* c_name_scope);

/**
 * @brief Delete a Dipole Charge Modifier.
 *
 * @param dp Dipole Charge Modifier to delete.
 */
extern void DP_DeleteDipoleChargeModifier(DP_DipoleChargeModifier* dcm);

/**
 * @brief Evaluate the force and virial correction by using a dipole charge
 *modifier with the neighbor list. (double version)
 * @param[in] dcm The dipole charge modifier to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size nall
 *x 3.
 * @param[in] atype The atom types. The array should contain nall ints.
 * @param[in] cell The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] pairs The pairs of atoms. The list should contain npairs pairs of
 *ints.
 * @param[in] npairs The number of pairs.
 * @param[in] delef_ The electric field on each atom. The array should be of
 *size nframes x nloc x 3.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[out] dfcorr_ Output force correction. The array should be of size
 *nall x 3.
 * @param[out] dvcorr_ Output virial correction. The array should be of size 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DipoleChargeModifierComputeNList(DP_DipoleChargeModifier* dcm,
                                                const int natom,
                                                const double* coord,
                                                const int* atype,
                                                const double* cell,
                                                const int* pairs,
                                                const int npairs,
                                                const double* delef_,
                                                const int nghost,
                                                const DP_Nlist* nlist,
                                                double* dfcorr_,
                                                double* dvcorr_);

/**
 * @brief Evaluate the force and virial correction by using a dipole charge
 *modifier with the neighbor list. (float version)
 * @param[in] dcm The dipole charge modifier to use.
 * @param[in] natoms The number of atoms.
 * @param[in] coord The coordinates of atoms. The array should be of size natoms
 *x 3.
 * @param[in] atype The atom types. The array should contain natoms ints.
 * @param[in] cell The cell of the region. The array should be of size 9. Pass
 *NULL if pbc is not used.
 * @param[in] pairs The pairs of atoms. The list should contain npairs pairs of
 *ints.
 * @param[in] npairs The number of pairs.
 * @param[in] delef_ The electric field on each atom. The array should be of
 *size nframes x natoms x 3.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nlist The neighbor list.
 * @param[out] dfcorr_ Output force correction. The array should be of size
 *natoms x 3.
 * @param[out] dvcorr_ Output virial correction. The array should be of size 9.
 * @warning The output arrays should be allocated before calling this function.
 *Pass NULL if not required.
 **/
extern void DP_DipoleChargeModifierComputeNListf(DP_DipoleChargeModifier* dcm,
                                                 const int natom,
                                                 const float* coord,
                                                 const int* atype,
                                                 const float* cell,
                                                 const int* pairs,
                                                 const int npairs,
                                                 const float* delef_,
                                                 const int nghost,
                                                 const DP_Nlist* nlist,
                                                 float* dfcorr_,
                                                 float* dvcorr_);

/**
 * @brief Get the type map of a DipoleChargeModifier.
 * @param[in] dcm The DipoleChargeModifier to use.
 * @return The cutoff radius.
 */
double DP_DipoleChargeModifierGetCutoff(DP_DipoleChargeModifier* dt);

/**
 * @brief Get the type map of a DipoleChargeModifier.
 * @param[in] dcm The DipoleChargeModifier to use.
 * @return The number of types of the DipoleChargeModifier.
 */
int DP_DipoleChargeModifierGetNumbTypes(DP_DipoleChargeModifier* dt);

/**
 * @brief Get sel types of a DipoleChargeModifier.
 * @param[in] dcm The DipoleChargeModifier to use.
 * @return The sel types
 */
int* DP_DipoleChargeModifierGetSelTypes(DP_DipoleChargeModifier* dt);

/**
 * @brief Get the number of sel types of a DipoleChargeModifier.
 * @param[in] dcm The DipoleChargeModifier to use.
 * @return The number of sel types
 */
int DP_DipoleChargeModifierGetNumbSelTypes(DP_DipoleChargeModifier* dt);

/**
 * @brief Check if there is any exceptions throw.
 *
 * @param dcm The DipoleChargeModifier to use.
 * @return const char* error message.
 */
const char* DP_DipoleChargeModifierCheckOK(DP_DipoleChargeModifier* dcm);

/**
 * @brief Convert PBtxt to PB.
 * @param[in] c_pbtxt The name of the PBtxt file.
 * @param[in] c_pb The name of the PB file.
 **/
extern void DP_ConvertPbtxtToPb(const char* c_pbtxt, const char* c_pb);

/**
 * @brief Print the summary of DeePMD-kit, including the version and the build
 * information.
 * @param[in] c_pre The prefix to each line.
 */
extern void DP_PrintSummary(const char* c_pre);

/**
 * @brief Read a file to a char array.
 * @param[in] c_model The name of the file.
 * @return const char* The char array.
 */
const char* DP_ReadFileToChar(const char* c_model);

/**
 * @brief Read a file to a char array. This version can handle string with '\0'
 * @version 2
 * @param[in] c_model The name of the file.
 * @param[out] size The size of the char array.
 * @return const char* The char array.
 */
const char* DP_ReadFileToChar2(const char* c_model, int* size);

/**
 * @brief Get forward and backward map of selected atoms by
 * atom types.
 * @param[in] natoms The number of atoms.
 * @param[in] atype The atom types of all atoms.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] nsel_type The number of selected atom types.
 * @param[in] sel_type The selected atom types.
 * @param[out] fwd_map The forward map with size natoms.
 * @param[out] nreal The number of selected real atoms.
 * @param[out] bkw_map The backward map with size nreal.
 * @param[out] nghost_real The number of selected ghost atoms.
 */
void DP_SelectByType(const int natoms,
                     const int* atype,
                     const int nghost,
                     const int nsel_type,
                     const int* sel_type,
                     int* fwd_map,
                     int* nreal,
                     int* bkw_map,
                     int* nghost_real);

/**
 * @brief Apply the given map to a vector. Assume nframes is 1.
 * @param[in] in The input vector.
 * @param[in] fwd_map The map.
 * @param[in] stride The stride of the input vector.
 * @param[in] nall1 The number of atoms in the input vector.
 * @param[out] nall2 The number of atoms in the output vector.
 * @param[out] out The output vector.
 */
void DP_SelectMapInt(const int* in,
                     const int* fwd_map,
                     const int stride,
                     const int nall1,
                     const int nall2,
                     int* out);

/**
 * @brief Destroy a char array.
 *
 * @param c_str The char array.
 */
void DP_DeleteChar(const char* c_str);

#ifdef __cplusplus
} /* end extern "C" */
#endif
