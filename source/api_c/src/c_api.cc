#include "c_api.h"

#include <vector>
#include <string>
#include <numeric>
#include "c_api_internal.h"
#include "common.h"
#include "DeepPot.h"

extern "C" {

DP_Nlist::DP_Nlist(deepmd::InputNlist& nl) : nl(nl) {}

DP_Nlist* DP_NewNlist(
    int inum_, 
    int * ilist_,
    int * numneigh_, 
    int ** firstneigh_){
    deepmd::InputNlist nl(inum_, ilist_, numneigh_, firstneigh_);
    DP_Nlist* new_nl = new DP_Nlist(nl);
    return new_nl;
}

DP_DeepPot::DP_DeepPot(deepmd::DeepPot& dp)
    : dp(dp) {}

DP_DeepPot* DP_NewDeepPot(const char* c_model) {
    std::string model(c_model);
    deepmd::DeepPot dp(model);
    DP_DeepPot* new_dp = new DP_DeepPot(dp);
    return new_dp;
}

DP_DeepPotModelDevi::DP_DeepPotModelDevi(deepmd::DeepPotModelDevi& dp)
    : dp(dp) {}

DP_DeepPotModelDevi* DP_NewDeepPotModelDevi(const char** c_models, int n_models) {
    std::vector<std::string> model(c_models, c_models + n_models);
    deepmd::DeepPotModelDevi dp(model);
    DP_DeepPotModelDevi* new_dp = new DP_DeepPotModelDevi(dp);
    return new_dp;
}

} // extern "C"

template <typename VALUETYPE>
inline
void DP_DeepPotCompute_variant (
    DP_DeepPot* dp,
    const int natoms,
    const VALUETYPE* coord,
    const int* atype,
    const VALUETYPE* cell,
    double* energy,
    VALUETYPE* force,
    VALUETYPE* virial,
    VALUETYPE* atomic_energy,
    VALUETYPE* atomic_virial
    ) {
    // init C++ vectors from C arrays
    std::vector<VALUETYPE> coord_(coord, coord+natoms*3);
    std::vector<int> atype_(atype, atype+natoms);
    std::vector<VALUETYPE> cell_;
    if (cell) {
        // pbc
        cell_.assign(cell, cell+9);
    }
    double e;
    std::vector<VALUETYPE> f, v, ae, av;

    dp->dp.compute(e, f, v, ae, av, coord_, atype_, cell_);
    // copy from C++ vectors to C arrays, if not NULL pointer
    if(energy) *energy = e;
    if(force) std::copy(f.begin(), f.end(), force);
    if(virial) std::copy(v.begin(), v.end(), virial);
    if(atomic_energy) std::copy(ae.begin(), ae.end(), atomic_energy);
    if(atomic_virial) std::copy(av.begin(), av.end(), atomic_virial);
}

template
void DP_DeepPotCompute_variant <double> (
    DP_DeepPot* dp,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    double* energy,
    double* force,
    double* virial,
    double* atomic_energy,
    double* atomic_virial
    );

template
void DP_DeepPotCompute_variant <float> (
    DP_DeepPot* dp,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    double* energy,
    float* force,
    float* virial,
    float* atomic_energy,
    float* atomic_virial
    );

template <typename VALUETYPE>
inline
void DP_DeepPotComputeNList_variant (
    DP_DeepPot* dp,
    const int natoms,
    const VALUETYPE* coord,
    const int* atype,
    const VALUETYPE* cell,
    const int nghost,
    const DP_Nlist* nlist,
    const int ago,
    double* energy,
    VALUETYPE* force,
    VALUETYPE* virial,
    VALUETYPE* atomic_energy,
    VALUETYPE* atomic_virial
    ) {
    // init C++ vectors from C arrays
    std::vector<VALUETYPE> coord_(coord, coord+natoms*3);
    std::vector<int> atype_(atype, atype+natoms);
    std::vector<VALUETYPE> cell_;
    if (cell) {
        // pbc
        cell_.assign(cell, cell+9);
    }
    double e;
    std::vector<VALUETYPE> f, v, ae, av;

    dp->dp.compute(e, f, v, ae, av, coord_, atype_, cell_, nghost, nlist->nl, ago);
    // copy from C++ vectors to C arrays, if not NULL pointer
    if(energy) *energy = e;
    if(force) std::copy(f.begin(), f.end(), force);
    if(virial) std::copy(v.begin(), v.end(), virial);
    if(atomic_energy) std::copy(ae.begin(), ae.end(), atomic_energy);
    if(atomic_virial) std::copy(av.begin(), av.end(), atomic_virial);
}

template
void DP_DeepPotComputeNList_variant <double> (
    DP_DeepPot* dp,
    const int natoms,
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

template
void DP_DeepPotComputeNList_variant <float> (
    DP_DeepPot* dp,
    const int natoms,
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

template <typename VALUETYPE>
inline
void flatten_vector(std::vector<VALUETYPE> & onedv, const std::vector<std::vector<VALUETYPE>>& twodv) {
    onedv.clear();
    for (size_t ii = 0; ii < twodv.size(); ++ii) {
        onedv.insert(onedv.end(), twodv[ii].begin(), twodv[ii].end());
    }
}


template <typename VALUETYPE>
void DP_DeepPotModelDeviComputeNlist_variant (
    DP_DeepPotModelDevi* dp,
    const int natoms,
    const VALUETYPE* coord,
    const int* atype,
    const VALUETYPE* cell,
    const int nghost,
    const DP_Nlist* nlist,
    const int ago,
    double* energy,
    VALUETYPE* force,
    VALUETYPE* virial,
    VALUETYPE* atomic_energy,
    VALUETYPE* atomic_virial
    ) {
    // init C++ vectors from C arrays
    std::vector<VALUETYPE> coord_(coord, coord+natoms*3);
    std::vector<int> atype_(atype, atype+natoms);
    std::vector<VALUETYPE> cell_;
    if (cell) {
        // pbc
        cell_.assign(cell, cell+9);
    }
    // different from DeepPot
    std::vector<double> e;
    std::vector<std::vector<VALUETYPE>> f, v, ae, av;

    dp->dp.compute(e, f, v, ae, av, coord_, atype_, cell_, nghost, nlist->nl, ago);
    // 2D vector to 2D array, flatten first    
    if(energy) {
        std::copy(e.begin(), e.end(), energy);
    }
    if(force) {
        std::vector<VALUETYPE> f_flat;
        flatten_vector(f_flat, f);
        std::copy(f_flat.begin(), f_flat.end(), force);
    }
    if(virial) {
        std::vector<VALUETYPE> v_flat;
        flatten_vector(v_flat, v);
        std::copy(v_flat.begin(), v_flat.end(), virial);
    }
    if(atomic_energy) {
        std::vector<VALUETYPE> ae_flat;
        flatten_vector(ae_flat, ae);
        std::copy(ae_flat.begin(), ae_flat.end(), atomic_energy);
    }
    if(atomic_virial) {
        std::vector<VALUETYPE> av_flat;
        flatten_vector(av_flat, av);
        std::copy(av_flat.begin(), av_flat.end(), atomic_virial);
    }
}

template
void DP_DeepPotModelDeviComputeNlist_variant <double> (
    DP_DeepPotModelDevi* dp,
    const int natoms,
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

template
void DP_DeepPotModelDeviComputeNlist_variant <float> (
    DP_DeepPotModelDevi* dp,
    const int natoms,
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

extern "C" {

void DP_DeepPotCompute (
    DP_DeepPot* dp,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    double* energy,
    double* force,
    double* virial,
    double* atomic_energy,
    double* atomic_virial
    ) {
    DP_DeepPotCompute_variant<double>(dp, natoms, coord, atype, cell, energy, force, virial, atomic_energy, atomic_virial);
}

void DP_DeepPotComputef (
    DP_DeepPot* dp,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    double* energy,
    float* force,
    float* virial,
    float* atomic_energy,
    float* atomic_virial
    ) {
    DP_DeepPotCompute_variant<float>(dp, natoms, coord, atype, cell, energy, force, virial, atomic_energy, atomic_virial);
}

void DP_DeepPotComputeNList (
    DP_DeepPot* dp,
    const int natoms,
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
    ) {
    DP_DeepPotComputeNList_variant<double>(dp, natoms, coord, atype, cell, nghost, nlist, ago, energy, force, virial, atomic_energy, atomic_virial);
}

void DP_DeepPotComputeNListf (
    DP_DeepPot* dp,
    const int natoms,
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
    ) {
    DP_DeepPotComputeNList_variant<float>(dp, natoms, coord, atype, cell, nghost, nlist, ago, energy, force, virial, atomic_energy, atomic_virial);
}

const char* DP_DeepPotGetTypeMap(
    DP_DeepPot* dp
    ) {
    std::string type_map;
    dp->dp.get_type_map(type_map);
    // copy from string to char*
    const std::string::size_type size = type_map.size();
    // +1 for '\0'
    char *buffer = new char[size + 1];
    std::copy(type_map.begin(), type_map.end(), buffer);
    buffer[size] = '\0';
    return buffer;
}

double DP_DeepPotGetCutoff(
    DP_DeepPot* dp
    ) {
    return dp->dp.cutoff();
}

int DP_DeepPotGetNumbTypes(
    DP_DeepPot* dp
    ) {
    return dp->dp.numb_types();
}

void DP_DeepPotModelDeviComputeNlist (
    DP_DeepPotModelDevi* dp,
    const int natoms,
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
    ) {
    DP_DeepPotModelDeviComputeNlist_variant<double>(dp, natoms, coord, atype, cell, nghost, nlist, ago, energy, force, virial, atomic_energy, atomic_virial);
}

void DP_DeepPotModelDeviComputeNlistf (
    DP_DeepPotModelDevi* dp,
    const int natoms,
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
    ) {
    DP_DeepPotModelDeviComputeNlist_variant<float>(dp, natoms, coord, atype, cell, nghost, nlist, ago, energy, force, virial, atomic_energy, atomic_virial);
}

double DP_DeepPotModelDeviGetCutoff(
    DP_DeepPotModelDevi* dp
    ) {
    return dp->dp.cutoff();
}

int DP_DeepPotModelDeviGetNumbTypes(
    DP_DeepPotModelDevi* dp
    ) {
    return dp->dp.numb_types();
}

void DP_ConvertPbtxtToPb(
    const char* c_pbtxt,
    const char* c_pb
    ) {
    std::string pbtxt(c_pbtxt);
    std::string pb(c_pb);
    deepmd::convert_pbtxt_to_pb(pbtxt, pb);
}

} // extern "C"