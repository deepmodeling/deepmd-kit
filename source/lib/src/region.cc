#include <stdexcept>
#include <algorithm>
#include "region.h"
#define BOXT_DIM 9

using namespace deepmd;

template<typename FPTYPE>
Region<FPTYPE>::
Region()
{
  boxt = new FPTYPE[BOXT_DIM];
  rec_boxt = new FPTYPE[BOXT_DIM];
}

template<typename FPTYPE>
Region<FPTYPE>::
~Region()
{
  delete [] boxt;
  delete [] rec_boxt;
}

template struct Region<double>;
template struct Region<float>;

template<typename FPTYPE>
inline FPTYPE
compute_volume(const FPTYPE * boxt)
{
  FPTYPE volume =
      boxt[0*3+0] * (boxt[1*3+1]*boxt[2*3+2] - boxt[2*3+1]*boxt[1*3+2]) - 
      boxt[0*3+1] * (boxt[1*3+0]*boxt[2*3+2] - boxt[2*3+0]*boxt[1*3+2]) +
      boxt[0*3+2] * (boxt[1*3+0]*boxt[2*3+1] - boxt[2*3+0]*boxt[1*3+1]);
  if (volume < 0) {
    throw std::runtime_error("Negative volume detected. Please make sure the simulation cell obeys the right-hand rule.");
  }
  return volume;
}

template<typename FPTYPE>
inline void
compute_rec_boxt(
    FPTYPE * rec_boxt,
    const FPTYPE * boxt)
{
  FPTYPE volumei = static_cast<FPTYPE>(1.) / compute_volume(boxt);
  rec_boxt[0*3+0] =( boxt[1*3+1]*boxt[2*3+2] - boxt[2*3+1]*boxt[1*3+2]) * volumei;
  rec_boxt[1*3+1] =( boxt[0*3+0]*boxt[2*3+2] - boxt[2*3+0]*boxt[0*3+2]) * volumei;
  rec_boxt[2*3+2] =( boxt[0*3+0]*boxt[1*3+1] - boxt[1*3+0]*boxt[0*3+1]) * volumei;
  rec_boxt[0*3+1] =(-boxt[1*3+0]*boxt[2*3+2] + boxt[2*3+0]*boxt[1*3+2]) * volumei;
  rec_boxt[0*3+2] =( boxt[1*3+0]*boxt[2*3+1] - boxt[2*3+0]*boxt[1*3+1]) * volumei;
  rec_boxt[1*3+0] =(-boxt[0*3+1]*boxt[2*3+2] + boxt[2*3+1]*boxt[0*3+2]) * volumei;
  rec_boxt[1*3+2] =(-boxt[0*3+0]*boxt[2*3+1] + boxt[2*3+0]*boxt[0*3+1]) * volumei;
  rec_boxt[2*3+0] =( boxt[0*3+1]*boxt[1*3+2] - boxt[1*3+1]*boxt[0*3+2]) * volumei;
  rec_boxt[2*3+1] =(-boxt[0*3+0]*boxt[1*3+2] + boxt[1*3+0]*boxt[0*3+2]) * volumei;
}

template<typename FPTYPE>
inline void
tensor_dot_vec (
    FPTYPE * o_v,
    const FPTYPE * i_t,
    const FPTYPE * i_v)
{
  o_v[0] = i_v[0] * i_t[0*3+0] + i_v[1] * i_t[0*3+1] + i_v[2] * i_t[0*3+2];
  o_v[1] = i_v[0] * i_t[1*3+0] + i_v[1] * i_t[1*3+1] + i_v[2] * i_t[1*3+2];
  o_v[2] = i_v[0] * i_t[2*3+0] + i_v[1] * i_t[2*3+1] + i_v[2] * i_t[2*3+2];
}

template<typename FPTYPE>
inline void
tensor_t_dot_vec (
    FPTYPE * o_v,
    const FPTYPE * i_t,
    const FPTYPE * i_v)
{
  o_v[0] = i_v[0] * i_t[0*3+0] + i_v[1] * i_t[1*3+0] + i_v[2] * i_t[2*3+0];
  o_v[1] = i_v[0] * i_t[0*3+1] + i_v[1] * i_t[1*3+1] + i_v[2] * i_t[2*3+1];
  o_v[2] = i_v[0] * i_t[0*3+2] + i_v[1] * i_t[1*3+2] + i_v[2] * i_t[2*3+2];
}

template<typename FPTYPE>
void
deepmd::
init_region_cpu(
    Region<FPTYPE> & region,
    const FPTYPE * boxt)
{
  std::copy(boxt, boxt+BOXT_DIM, region.boxt);
  compute_rec_boxt(region.rec_boxt, region.boxt);
}

template<typename FPTYPE>
void
deepmd::
convert_to_inter_cpu(
    FPTYPE * ri, 
    const Region<FPTYPE> & region,
    const FPTYPE * rp)
{
  tensor_dot_vec(ri, region.rec_boxt, rp);
}

template<typename FPTYPE>
void
deepmd::
convert_to_phys_cpu(
    FPTYPE * rp, 
    const Region<FPTYPE> & region,
    const FPTYPE * ri)
{
  tensor_t_dot_vec(rp, region.boxt, ri);
}

template<typename FPTYPE>
FPTYPE
deepmd::
volume_cpu(
    const Region<FPTYPE> & region)
{
  return compute_volume(region.boxt);
}

template
void 
deepmd::
init_region_cpu<double>(
    deepmd::Region<double> & region,
    const double * boxt);

template
void 
deepmd::
init_region_cpu<float>(
    deepmd::Region<float> & region,
    const float * boxt);

template
void
deepmd::
convert_to_inter_cpu<double>(
    double * ri, 
    const deepmd::Region<double> & region,
    const double * rp);

template
void
deepmd::
convert_to_inter_cpu<float>(
    float * ri, 
    const deepmd::Region<float> & region,
    const float * rp);

template
void
deepmd::
convert_to_phys_cpu<double>(
    double * ri, 
    const deepmd::Region<double> & region,
    const double * rp);

template
void
deepmd::
convert_to_phys_cpu<float>(
    float * ri, 
    const deepmd::Region<float> & region,
    const float * rp);

template
double
deepmd::
volume_cpu<double>(
    const deepmd::Region<double> & region);

template
float
deepmd::
volume_cpu<float>(
    const deepmd::Region<float> & region);
