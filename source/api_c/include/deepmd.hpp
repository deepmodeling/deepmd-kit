/*
Header-only DeePMD-kit C++ 11 library

This header-only library provides a C++ 11 interface to the DeePMD-kit C API.
*/

#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <cassert>

#include "c_api.h"

template <typename FPTYPE>
inline void _DP_DeepPotCompute(
    DP_DeepPot *dp,
    const int natom,
    const FPTYPE *coord,
    const int *atype,
    const FPTYPE *cell,
    double *energy,
    FPTYPE *force,
    FPTYPE *virial,
    FPTYPE *atomic_energy,
    FPTYPE *atomic_virial);

template <>
inline void _DP_DeepPotCompute<double>(
    DP_DeepPot *dp,
    const int natom,
    const double *coord,
    const int *atype,
    const double *cell,
    double *energy,
    double *force,
    double *virial,
    double *atomic_energy,
    double *atomic_virial)
{
    DP_DeepPotCompute(dp, natom, coord, atype, cell, energy, force, virial, atomic_energy, atomic_virial);
}

template <>
inline void _DP_DeepPotCompute<float>(
    DP_DeepPot *dp,
    const int natom,
    const float *coord,
    const int *atype,
    const float *cell,
    double *energy,
    float *force,
    float *virial,
    float *atomic_energy,
    float *atomic_virial)
{
    DP_DeepPotComputef(dp, natom, coord, atype, cell, energy, force, virial, atomic_energy, atomic_virial);
}

namespace deepmd
{
    namespace hpp
    {
        /**
         * @brief Convert pbtxt to pb.
         * @param[in] fn_pb_txt Filename of the pb txt file.
         * @param[in] fn_pb Filename of the pb file.
         **/
        void
        convert_pbtxt_to_pb(std::string fn_pb_txt, std::string fn_pb)
        {
            DP_ConvertPbtxtToPb(fn_pb_txt.c_str(), fn_pb.c_str());
        };
        /**
         * @brief Deep Potential.
         **/
        class DeepPot
        {
        public:
            /**
             * @brief DP constructor without initialization.
             **/
            DeepPot() : dp(nullptr) {};
            ~DeepPot(){};
            /**
             * @brief DP constructor with initialization.
             * @param[in] model The name of the frozen model file.
             **/
            DeepPot(const std::string &model) : dp(nullptr)
            {
                init(model);
            };
            /**
             * @brief Initialize the DP.
             * @param[in] model The name of the frozen model file.
             **/
            void init(const std::string &model)
            {
                if (dp)
                {
                    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do nothing at the second call of initializer" << std::endl;
                    return;
                }
                dp = DP_NewDeepPot(model.c_str());
            };

            /**
             * @brief Evaluate the energy, force and virial by using this DP.
             * @param[out] ener The system energy.
             * @param[out] force The force on each atom.
             * @param[out] virial The virial.
             * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
             * @param[in] atype The atom types. The list should contain natoms ints.
             * @param[in] box The cell of the region. The array should be of size nframes x 9 (PBC) or empty (no PBC).
             **/
            template <typename VALUETYPE>
            void compute(double &ener,
                         std::vector<VALUETYPE> &force,
                         std::vector<VALUETYPE> &virial,
                         const std::vector<VALUETYPE> &coord,
                         const std::vector<int> &atype,
                         const std::vector<VALUETYPE> &box)
            {
                unsigned int natoms = atype.size();
                assert(natoms * 3 == coord.size());
                if (!box.empty()) {
                    assert(box.size() == 9);
                }
                const VALUETYPE *coord_ = &coord[0];
                const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
                const int *atype_ = &atype[0];
                double *ener_ = &ener;
                force.resize(natoms * 3);
                virial.resize(9);
                VALUETYPE *force_ = &force[0];
                VALUETYPE *virial_ = &virial[0];

                _DP_DeepPotCompute<VALUETYPE>(dp, natoms, coord_, atype_, box_, ener_, force_, virial_, nullptr, nullptr);
            };
            /**
             * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial by using this DP.
             * @param[out] ener The system energy.
             * @param[out] force The force on each atom.
             * @param[out] virial The virial.
             * @param[out] atom_energy The atomic energy.
             * @param[out] atom_virial The atomic virial.
             * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
             * @param[in] atype The atom types. The list should contain natoms ints.
             * @param[in] box The cell of the region. The array should be of size nframes x 9 (PBC) or empty (no PBC).
             **/
            template <typename VALUETYPE>
            void compute(double &ener,
                         std::vector<VALUETYPE> &force,
                         std::vector<VALUETYPE> &virial,
                         std::vector<VALUETYPE> &atom_energy,
                         std::vector<VALUETYPE> &atom_virial,
                         const std::vector<VALUETYPE> &coord,
                         const std::vector<int> &atype,
                         const std::vector<VALUETYPE> &box)
            {
                unsigned int natoms = atype.size();
                assert(natoms * 3 == coord.size());
                if (!box.empty()) {
                    assert(box.size() == 9);
                }
                const VALUETYPE *coord_ = &coord[0];
                const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
                const int *atype_ = &atype[0];

                double *ener_ = &ener;
                force.resize(natoms * 3);
                virial.resize(9);
                atom_energy.resize(natoms);
                atom_virial.resize(natoms * 9);
                VALUETYPE *force_ = &force[0];
                VALUETYPE *virial_ = &virial[0];
                VALUETYPE *atomic_ener_ = &atom_energy[0];
                VALUETYPE *atomic_virial_ = &atom_virial[0];

                _DP_DeepPotCompute<VALUETYPE>(dp, natoms, coord_, atype_, box_, ener_, force_, virial_, atomic_ener_, atomic_virial_);
            };
            /**
             * @brief Get the cutoff radius.
             * @return The cutoff radius.
             **/
            double cutoff() const
            {
                assert(dp);
                return DP_DeepPotGetCutoff(dp);
            };
            /**
             * @brief Get the number of types.
             * @return The number of types.
             **/
            int numb_types() const
            {
                assert(dp);
                return DP_DeepPotGetNumbTypes(dp);
            };
            /**
             * @brief Get the type map (element name of the atom types) of this model.
             * @param[out] type_map The type map of this model.
             **/
            void get_type_map(std::string &type_map)
            {
                const char *type_map_c = DP_DeepPotGetTypeMap(dp);
                type_map.assign(type_map_c);
                delete[] type_map_c;
            };

        private:
            DP_DeepPot *dp;
        };
    }
}
