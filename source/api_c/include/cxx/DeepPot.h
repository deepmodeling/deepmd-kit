#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <cassert>

#include "c_api.h"
#include "common.h"
#include "neighbor_list.h"


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
    /**
     * @brief Deep Potential.
     **/
    class DeepPot
    {
    public:
        /**
         * @brief DP constructor without initialization.
         **/
        DeepPot();
        ~DeepPot();
        /**
         * @brief DP constructor with initialization.
         * @param[in] model The name of the frozen model file.
         * @param[in] gpu_rank The GPU rank. Default is 0.
         * @param[in] file_content The content of the model file. If it is not empty, DP will read from the string instead of the file.
         **/
        DeepPot(const std::string &model, const int &gpu_rank = 0, const std::string &file_content = "")
        {
            init(model, gpu_rank, file_content);
        };
        /**
         * @brief Initialize the DP.
         * @param[in] model The name of the frozen model file.
         * @param[in] gpu_rank The GPU rank. Default is 0.
         * @param[in] file_content The content of the model file. If it is not empty, DP will read from the string instead of the file.
         **/
        void init(const std::string &model, const int &gpu_rank = 0, const std::string &file_content = "")
        {
            if (inited)
            {
                std::cerr << "WARNING: deepmd-kit should not be initialized twice, do nothing at the second call of initializer" << std::endl;
                return;
            }
            if (gpu_rank != 0 || !file_content.empty())
                throw deepmd::deepmd_exception("Not implemented!");
            dp = DP_NewDeepPot(model.c_str());
            inited = true;
        };
        /**
         * @brief Print the DP summary to the screen.
         * @param[in] pre The prefix to each line.
         **/
        void print_summary(const std::string &pre) const
        {
            throw deepmd::deepmd_exception("Not implemented!");
        };

        /**
         * @brief Evaluate the energy, force and virial by using this DP.
         * @param[out] ener The system energy.
         * @param[out] force The force on each atom.
         * @param[out] virial The virial.
         * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
         * @param[in] atype The atom types. The list should contain natoms ints.
         * @param[in] box The cell of the region. The array should be of size nframes x 9.
         * @param[in] fparam The frame parameter. The array can be of size :
         * nframes x dim_fparam.
         * dim_fparam. Then all frames are assumed to be provided with the same fparam.
         * @param[in] aparam The atomic parameter The array can be of size :
         * nframes x natoms x dim_aparam.
         * natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
         * dim_aparam. Then all frames and atoms are provided with the same aparam.
         **/
        template <typename VALUETYPE>
        void compute(ENERGYTYPE &ener,
                     std::vector<VALUETYPE> &force,
                     std::vector<VALUETYPE> &virial,
                     const std::vector<VALUETYPE> &coord,
                     const std::vector<int> &atype,
                     const std::vector<VALUETYPE> &box,
                     const std::vector<VALUETYPE> &fparam = std::vector<VALUETYPE>(),
                     const std::vector<VALUETYPE> &aparam = std::vector<VALUETYPE>())
        {
            if (!fparam.empty() || aparam.empty())
            {
                throw deepmd::deepmd_exception("Not implemented!");
            }
            int natoms = coord.size() / 3;
            const VALUETYPE *coord_ = &coord[0];
            const VALUETYPE *box_ = &box[0];
            const int *atype_ = &atype[0];

            double *ener_ = new double;
            VALUETYPE *force_ = new VALUETYPE[natoms * 3];
            VALUETYPE *virial_ = new VALUETYPE[9];

            _DP_DeepPotCompute<VALUETYPE>(dp, natoms, coord_, atype_, box_, ener_, force_, virial_, nullptr, nullptr);

            ener = *ener_;
            force.assign(force_, force_ + natoms * 3);
            virial.assign(virial_, virial_ + 9);
        };
        /**
         * @brief Evaluate the energy, force and virial by using this DP.
         * @param[out] ener The system energy.
         * @param[out] force The force on each atom.
         * @param[out] virial The virial.
         * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
         * @param[in] atype The atom types. The list should contain natoms ints.
         * @param[in] box The cell of the region. The array should be of size nframes x 9.
         * @param[in] nghost The number of ghost atoms.
         * @param[in] inlist The input neighbour list.
         * @param[in] ago Update the internal neighbour list if ago is 0.
         * @param[in] fparam The frame parameter. The array can be of size :
         * nframes x dim_fparam.
         * dim_fparam. Then all frames are assumed to be provided with the same fparam.
         * @param[in] aparam The atomic parameter The array can be of size :
         * nframes x natoms x dim_aparam.
         * natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
         * dim_aparam. Then all frames and atoms are provided with the same aparam.
         **/
        template <typename VALUETYPE>
        void compute(ENERGYTYPE &ener,
                     std::vector<VALUETYPE> &force,
                     std::vector<VALUETYPE> &virial,
                     const std::vector<VALUETYPE> &coord,
                     const std::vector<int> &atype,
                     const std::vector<VALUETYPE> &box,
                     const int nghost,
                     const InputNlist &inlist,
                     const int &ago,
                     const std::vector<VALUETYPE> &fparam = std::vector<VALUETYPE>(),
                     const std::vector<VALUETYPE> &aparam = std::vector<VALUETYPE>())
        {
            throw deepmd::deepmd_exception("Not implemented!");
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
         * @param[in] box The cell of the region. The array should be of size nframes x 9.
         * @param[in] fparam The frame parameter. The array can be of size :
         * nframes x dim_fparam.
         * dim_fparam. Then all frames are assumed to be provided with the same fparam.
         * @param[in] aparam The atomic parameter The array can be of size :
         * nframes x natoms x dim_aparam.
         * natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
         * dim_aparam. Then all frames and atoms are provided with the same aparam.
         **/
        template <typename VALUETYPE>
        void compute(ENERGYTYPE &ener,
                     std::vector<VALUETYPE> &force,
                     std::vector<VALUETYPE> &virial,
                     std::vector<VALUETYPE> &atom_energy,
                     std::vector<VALUETYPE> &atom_virial,
                     const std::vector<VALUETYPE> &coord,
                     const std::vector<int> &atype,
                     const std::vector<VALUETYPE> &box,
                     const std::vector<VALUETYPE> &fparam = std::vector<VALUETYPE>(),
                     const std::vector<VALUETYPE> &aparam = std::vector<VALUETYPE>())
        {
            if (!fparam.empty() || aparam.empty())
            {
                throw deepmd::deepmd_exception("Not implemented!");
            }
            int natoms = coord.size() / 3;
            const VALUETYPE *coord_ = &coord[0];
            const VALUETYPE *box_ = &box[0];
            const int *atype_ = &atype[0];

            double *ener_ = new double;
            VALUETYPE *force_ = new VALUETYPE[natoms * 3];
            VALUETYPE *virial_ = new VALUETYPE[9];
            VALUETYPE *atomic_ener_ = new VALUETYPE[natoms];
            VALUETYPE *atomic_virial_ = new VALUETYPE[natoms * 9];

            _DP_DeepPotCompute<VALUETYPE>(dp, natoms, coord_, atype_, box_, ener_, force_, virial_, atomic_ener_, atomic_virial_);

            ener = *ener_;
            force.assign(force_, force_ + natoms * 3);
            virial.assign(virial_, virial_ + 9);
            atom_energy.assign(atomic_ener_, atomic_ener_ + natoms);
            atom_virial.assign(atomic_virial_, atomic_virial_ + natoms * 9);
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
         * @param[in] box The cell of the region. The array should be of size nframes x 9.
         * @param[in] nghost The number of ghost atoms.
         * @param[in] lmp_list The input neighbour list.
         * @param[in] ago Update the internal neighbour list if ago is 0.
         * @param[in] fparam The frame parameter. The array can be of size :
         * nframes x dim_fparam.
         * dim_fparam. Then all frames are assumed to be provided with the same fparam.
         * @param[in] aparam The atomic parameter The array can be of size :
         * nframes x natoms x dim_aparam.
         * natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
         * dim_aparam. Then all frames and atoms are provided with the same aparam.
         **/
        template <typename VALUETYPE>
        void compute(ENERGYTYPE &ener,
                     std::vector<VALUETYPE> &force,
                     std::vector<VALUETYPE> &virial,
                     std::vector<VALUETYPE> &atom_energy,
                     std::vector<VALUETYPE> &atom_virial,
                     const std::vector<VALUETYPE> &coord,
                     const std::vector<int> &atype,
                     const std::vector<VALUETYPE> &box,
                     const int nghost,
                     const InputNlist &lmp_list,
                     const int &ago,
                     const std::vector<VALUETYPE> &fparam = std::vector<VALUETYPE>(),
                     const std::vector<VALUETYPE> &aparam = std::vector<VALUETYPE>())
        {
            throw deepmd::deepmd_exception("Not implemented!");
        };
        /**
         * @brief Get the cutoff radius.
         * @return The cutoff radius.
         **/
        double cutoff() const
        {
            assert(inited);
            return DP_DeepPotGetCutoff(dp);
        };
        /**
         * @brief Get the number of types.
         * @return The number of types.
         **/
        int numb_types() const
        {
            assert(inited);
            return DP_DeepPotGetNumbTypes(dp);
        };
        /**
         * @brief Get the dimension of the frame parameter.
         * @return The dimension of the frame parameter.
         **/
        int dim_fparam() const
        {
            throw deepmd::deepmd_exception("Not implemented!");
            return 0;
        };
        /**
         * @brief Get the dimension of the atomic parameter.
         * @return The dimension of the atomic parameter.
         **/
        int dim_aparam() const
        {
            throw deepmd::deepmd_exception("Not implemented!");
            return 0;
        };
        /**
         * @brief Get the type map (element name of the atom types) of this model.
         * @param[out] type_map The type map of this model.
         **/
        void get_type_map(std::string &type_map)
        {
            const char *type_map_c = DP_DeepPotGetTypeMap(dp);
            type_map = type_map_c;
            delete[] type_map_c;
        };

    private:
        DP_DeepPot *dp;
        bool inited;
    };
}
