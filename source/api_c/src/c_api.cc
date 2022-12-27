#include "c_api.h"

#include <vector>
#include <string>
#include <numeric>
#include "c_api_internal.h"
#include "common.h"
#include "DeepPot.h"
#include "DeepTensor.h"
#include "DataModifier.h"

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

DP_DeepPot* DP_NewDeepPotWithParam(
        const char* c_model, const int gpu_rank, const char* c_file_content) {
    std::string model(c_model);
    std::string file_content(c_file_content);
    deepmd::DeepPot dp(model, gpu_rank, file_content);
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

DP_DeepTensor::DP_DeepTensor(deepmd::DeepTensor& dt)
    : dt(dt) {}

DP_DeepTensor* DP_NewDeepTensor(const char* c_model) {
    std::string model(c_model);
    deepmd::DeepTensor dt(model);
    DP_DeepTensor* new_dt = new DP_DeepTensor(dt);
    return new_dt;
}

DP_DeepTensor* DP_NewDeepTensorWithParam(
        const char* c_model, const int gpu_rank, const char* c_name_scope) {
    std::string model(c_model);
    std::string name_scope(c_name_scope);
    deepmd::DeepTensor dt(model, gpu_rank, name_scope);
    DP_DeepTensor* new_dt = new DP_DeepTensor(dt);
    return new_dt;
}

DP_DipoleChargeModifier::DP_DipoleChargeModifier(deepmd::DipoleChargeModifier& dcm)
    : dcm(dcm) {}

DP_DipoleChargeModifier* DP_NewDipoleChargeModifier(const char* c_model) {
    std::string model(c_model);
    deepmd::DipoleChargeModifier dcm(model);
    DP_DipoleChargeModifier* new_dcm = new DP_DipoleChargeModifier(dcm);
    return new_dcm;
}

DP_DipoleChargeModifier* DP_NewDipoleChargeModifierWithParam(
        const char* c_model, const int gpu_rank, const char* c_name_scope) {
    std::string model(c_model);
    std::string name_scope(c_name_scope);
    deepmd::DipoleChargeModifier dcm(model, gpu_rank, name_scope);
    DP_DipoleChargeModifier* new_dcm = new DP_DipoleChargeModifier(dcm);
    return new_dcm;
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
void DP_DeepPotModelDeviComputeNList_variant (
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
void DP_DeepPotModelDeviComputeNList_variant <double> (
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
void DP_DeepPotModelDeviComputeNList_variant <float> (
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

template <typename VALUETYPE>
inline
void DP_DeepTensorComputeTensor_variant (
    DP_DeepTensor* dt,
    const int natoms,
    const VALUETYPE* coord,
    const int* atype,
    const VALUETYPE* cell,
    VALUETYPE** tensor,
    int* size
    ) {
    // init C++ vectors from C arrays
    std::vector<VALUETYPE> coord_(coord, coord+natoms*3);
    std::vector<int> atype_(atype, atype+natoms);
    std::vector<VALUETYPE> cell_;
    if (cell) {
        // pbc
        cell_.assign(cell, cell+9);
    }
    std::vector<VALUETYPE> t;

    dt->dt.compute(t, coord_, atype_, cell_);
    // do not know the size of tensor in advance...
    *tensor = new VALUETYPE[t.size()];
    std::copy(t.begin(), t.end(), *tensor);
    *size = t.size();
}

template
void DP_DeepTensorComputeTensor_variant <double> (
    DP_DeepTensor* dt,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    double** tensor,
    int* size
    );

template
void DP_DeepTensorComputeTensor_variant <float> (
    DP_DeepTensor* dt,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    float** tensor,
    int* size
    );

template <typename VALUETYPE>
inline
void DP_DeepTensorComputeTensorNList_variant (
    DP_DeepTensor* dt,
    const int natoms,
    const VALUETYPE* coord,
    const int* atype,
    const VALUETYPE* cell,
    const int nghost,
    const DP_Nlist* nlist,
    VALUETYPE** tensor,
    int* size
    ) {
    // init C++ vectors from C arrays
    std::vector<VALUETYPE> coord_(coord, coord+natoms*3);
    std::vector<int> atype_(atype, atype+natoms);
    std::vector<VALUETYPE> cell_;
    if (cell) {
        // pbc
        cell_.assign(cell, cell+9);
    }
    std::vector<VALUETYPE> t;

    dt->dt.compute(t, coord_, atype_, cell_, nghost, nlist->nl);
    // do not know the size of tensor in advance...
    *tensor = new VALUETYPE[t.size()];
    std::copy(t.begin(), t.end(), *tensor);
    *size = t.size();
}

template
void DP_DeepTensorComputeTensorNList_variant <double> (
    DP_DeepTensor* dt,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    const int nghost,
    const DP_Nlist* nlist,
    double** tensor,
    int* size
    );

template
void DP_DeepTensorComputeTensorNList_variant <float> (
    DP_DeepTensor* dt,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    const int nghost,
    const DP_Nlist* nlist,
    float** tensor,
    int* size
    );

template <typename VALUETYPE>
inline
void DP_DeepTensorCompute_variant (
    DP_DeepTensor* dt,
    const int natoms,
    const VALUETYPE* coord,
    const int* atype,
    const VALUETYPE* cell,
    VALUETYPE* global_tensor,
    VALUETYPE* force,
    VALUETYPE* virial,
    VALUETYPE** atomic_tensor,
    VALUETYPE* atomic_virial,
    int* size_at
    ) {
    // init C++ vectors from C arrays
    std::vector<VALUETYPE> coord_(coord, coord+natoms*3);
    std::vector<int> atype_(atype, atype+natoms);
    std::vector<VALUETYPE> cell_;
    if (cell) {
        // pbc
        cell_.assign(cell, cell+9);
    }
    std::vector<VALUETYPE> t, f, v, at, av;

    dt->dt.compute(t, f, v, at, av, coord_, atype_, cell_);
    // copy from C++ vectors to C arrays, if not NULL pointer
    if(global_tensor) std::copy(t.begin(), t.end(), global_tensor);
    if(force) std::copy(f.begin(), f.end(), force);
    if(virial) std::copy(v.begin(), v.end(), virial);
    if(atomic_virial) std::copy(av.begin(), av.end(), atomic_virial);
    // do not know the size of atomic tensor in advance...
    if(atomic_tensor) {
        *atomic_tensor = new VALUETYPE[at.size()];
        std::copy(at.begin(), at.end(), *atomic_tensor);
    }
    if(size_at) *size_at = at.size();
}

template
void DP_DeepTensorCompute_variant <double> (
    DP_DeepTensor* dt,
    const int natoms,
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

template
void DP_DeepTensorCompute_variant <float> (
    DP_DeepTensor* dt,
    const int natoms,
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

template <typename VALUETYPE>
inline
void DP_DeepTensorComputeNList_variant (
    DP_DeepTensor* dt,
    const int natoms,
    const VALUETYPE* coord,
    const int* atype,
    const VALUETYPE* cell,
    const int nghost,
    const DP_Nlist* nlist,
    VALUETYPE* global_tensor,
    VALUETYPE* force,
    VALUETYPE* virial,
    VALUETYPE** atomic_tensor,
    VALUETYPE* atomic_virial,
    int* size_at
    ) {
    // init C++ vectors from C arrays
    std::vector<VALUETYPE> coord_(coord, coord+natoms*3);
    std::vector<int> atype_(atype, atype+natoms);
    std::vector<VALUETYPE> cell_;
    if (cell) {
        // pbc
        cell_.assign(cell, cell+9);
    }
    std::vector<VALUETYPE> t, f, v, at, av;

    dt->dt.compute(t, f, v, at, av, coord_, atype_, cell_, nghost, nlist->nl);
    // copy from C++ vectors to C arrays, if not NULL pointer
    if(global_tensor) std::copy(t.begin(), t.end(), global_tensor);
    if(force) std::copy(f.begin(), f.end(), force);
    if(virial) std::copy(v.begin(), v.end(), virial);
    if(atomic_virial) std::copy(av.begin(), av.end(), atomic_virial);
    // do not know the size of atomic tensor in advance...
    if(atomic_tensor) {
        *atomic_tensor = new VALUETYPE[at.size()];
        std::copy(at.begin(), at.end(), *atomic_tensor);
    }
    if(size_at) *size_at = at.size();
}

template
void DP_DeepTensorComputeNList_variant <double> (
    DP_DeepTensor* dt,
    const int natoms,
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

template
void DP_DeepTensorComputeNList_variant <float> (
    DP_DeepTensor* dt,
    const int natoms,
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

template <typename VALUETYPE>
inline
void DP_DipoleChargeModifierComputeNList_variant (
  DP_DipoleChargeModifier* dcm,
  const int natoms,
  const VALUETYPE* coord,
  const int* atype,
  const VALUETYPE* cell,
  const int* pairs,
  const int npairs,
  const VALUETYPE* delef,
  const int nghost,
  const DP_Nlist* nlist,
  VALUETYPE* dfcorr_,
  VALUETYPE* dvcorr_
  ){
    // init C++ vectors from C arrays
    std::vector<VALUETYPE> coord_(coord, coord+natoms*3);
    std::vector<int> atype_(atype, atype+natoms);
    std::vector<VALUETYPE> cell_;
    if (cell) {
        // pbc
        cell_.assign(cell, cell+9);
    }
    // pairs
    std::vector<std::pair<int, int> > pairs_;
    for (int i = 0; i < npairs; i++) {
        pairs_.push_back(std::make_pair(pairs[i*2], pairs[i*2+1]));
    }
    std::vector<VALUETYPE> delef_(delef, delef+natoms*3);
    std::vector<VALUETYPE> df, dv;

    dcm->dcm.compute(df, dv, coord_, atype_, cell_, pairs_, delef_, nghost, nlist->nl);
    // copy from C++ vectors to C arrays, if not NULL pointer
    if(dfcorr_) std::copy(df.begin(), df.end(), dfcorr_);
    if(dvcorr_) std::copy(dv.begin(), dv.end(), dvcorr_);
}

template
void DP_DipoleChargeModifierComputeNList_variant <double> (
  DP_DipoleChargeModifier* dcm,
  const int natoms,
  const double* coord,
  const int* atype,
  const double* cell,
  const int* pairs,
  const int npairs,
  const double* delef,
  const int nghost,
  const DP_Nlist* nlist,
  double* dfcorr_,
  double* dvcorr_
  );

template
void DP_DipoleChargeModifierComputeNList_variant <float> (
  DP_DipoleChargeModifier* dcm,
  const int natoms,
  const float* coord,
  const int* atype,
  const float* cell,
  const int* pairs,
  const int npairs,
  const float* delef,
  const int nghost,
  const DP_Nlist* nlist,
  float* dfcorr_,
  float* dvcorr_
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

void DP_DeepPotModelDeviComputeNList (
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
    DP_DeepPotModelDeviComputeNList_variant<double>(dp, natoms, coord, atype, cell, nghost, nlist, ago, energy, force, virial, atomic_energy, atomic_virial);
}

void DP_DeepPotModelDeviComputeNListf (
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
    DP_DeepPotModelDeviComputeNList_variant<float>(dp, natoms, coord, atype, cell, nghost, nlist, ago, energy, force, virial, atomic_energy, atomic_virial);
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

void DP_DeepTensorComputeTensor (
    DP_DeepTensor* dt,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    double** tensor,
    int* size
    ) {
    DP_DeepTensorComputeTensor_variant<double>(dt, natoms, coord, atype, cell, tensor, size);
}

void DP_DeepTensorComputeTensorf (
    DP_DeepTensor* dt,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    float** tensor,
    int* size
    ) {
    DP_DeepTensorComputeTensor_variant<float>(dt, natoms, coord, atype, cell, tensor, size);
}

void DP_DeepTensorComputeTensorNList (
    DP_DeepTensor* dt,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    const int nghost,
    const DP_Nlist* nlist,
    double** tensor,
    int* size
    ) {
    DP_DeepTensorComputeTensorNList_variant<double>(dt, natoms, coord, atype, cell, nghost, nlist, tensor, size);
}

void DP_DeepTensorComputeTensorNListf (
    DP_DeepTensor* dt,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    const int nghost,
    const DP_Nlist* nlist,
    float** tensor,
    int* size
    ) {
    DP_DeepTensorComputeTensorNList_variant<float>(dt, natoms, coord, atype, cell, nghost, nlist, tensor, size);
}

void DP_DeepTensorCompute (
    DP_DeepTensor* dt,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    double* global_tensor,
    double* force,
    double* virial,
    double** atomic_tensor,
    double* atomic_virial,
    int* size_at
    ) {
    DP_DeepTensorCompute_variant<double>(dt, natoms, coord, atype, cell, global_tensor, force, virial, atomic_tensor, atomic_virial, size_at);
}

void DP_DeepTensorComputef (
    DP_DeepTensor* dt,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    float* global_tensor,
    float* force,
    float* virial,
    float** atomic_tensor,
    float* atomic_virial,
    int* size_at
    ) {
    DP_DeepTensorCompute_variant<float>(dt, natoms, coord, atype, cell, global_tensor, force, virial, atomic_tensor, atomic_virial, size_at);
}

void DP_DeepTensorComputeNList (
    DP_DeepTensor* dt,
    const int natoms,
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
    ) {
    DP_DeepTensorComputeNList_variant<double>(dt, natoms, coord, atype, cell, nghost, nlist, global_tensor, force, virial, atomic_tensor, atomic_virial, size_at);
}

void DP_DeepTensorComputeNListf (
    DP_DeepTensor* dt,
    const int natoms,
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
    ) {
    DP_DeepTensorComputeNList_variant<float>(dt, natoms, coord, atype, cell, nghost, nlist, global_tensor, force, virial, atomic_tensor, atomic_virial, size_at);
}

double DP_DeepTensorGetCutoff(
    DP_DeepTensor* dt
    ) {
    return dt->dt.cutoff();
}

int DP_DeepTensorGetNumbTypes(
    DP_DeepTensor* dt
    ) {
    return dt->dt.numb_types();
}

int DP_DeepTensorGetOutputDim(
    DP_DeepTensor* dt
    ) {
    return dt->dt.output_dim();
}

int* DP_DeepTensorGetSelTypes(
    DP_DeepTensor* dt
    ) {
    return (int*) &(dt->dt.sel_types())[0];
}

int DP_DeepTensorGetNumbSelTypes(
    DP_DeepTensor* dt
    ) {
    return dt->dt.sel_types().size();
}

void DP_DipoleChargeModifierComputeNList (
  DP_DipoleChargeModifier* dcm,
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
  double* dvcorr_
  ){
    DP_DipoleChargeModifierComputeNList_variant<double>(dcm, natom, coord, atype, cell, pairs, npairs, delef_, nghost, nlist, dfcorr_, dvcorr_);
}

void DP_DipoleChargeModifierComputeNListf (
  DP_DipoleChargeModifier* dcm,
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
  float* dvcorr_
  ){
    DP_DipoleChargeModifierComputeNList_variant<float>(dcm, natom, coord, atype, cell, pairs, npairs, delef_, nghost, nlist, dfcorr_, dvcorr_);
}

double DP_DipoleChargeModifierGetCutoff(
    DP_DipoleChargeModifier* dcm
    ) {
    return dcm->dcm.cutoff();
}

int DP_DipoleChargeModifierGetNumbTypes(
    DP_DipoleChargeModifier* dcm
    ) {
    return dcm->dcm.numb_types();
}

int* DP_DipoleChargeModifierGetSelTypes(
    DP_DipoleChargeModifier* dcm
    ) {
    return (int*) &(dcm->dcm.sel_types())[0];
}

int DP_DipoleChargeModifierGetNumbSelTypes(
    DP_DipoleChargeModifier* dcm
    ) {
    return dcm->dcm.sel_types().size();
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