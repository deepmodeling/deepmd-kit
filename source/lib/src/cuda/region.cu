#include "device.h"
#include "gpu_cuda.h"
#include "region.h"

template<typename FPTYPE>
__device__ inline void tensorDotVector(FPTYPE *o_v, const FPTYPE *i_v, const FPTYPE *i_t)
{
    o_v[0] = i_v[0] * i_t[0*3+0] + i_v[1] * i_t[0*3+1] + i_v[2] * i_t[0*3+2];
    o_v[1] = i_v[0] * i_t[1*3+0] + i_v[1] * i_t[1*3+1] + i_v[2] * i_t[1*3+2];
    o_v[2] = i_v[0] * i_t[2*3+0] + i_v[1] * i_t[2*3+1] + i_v[2] * i_t[2*3+2];
}
template<typename FPTYPE>
__device__ inline void tensorTransDotVector(FPTYPE *o_v, const FPTYPE *i_v, const FPTYPE *i_t)
{
    o_v[0] = i_v[0] * i_t[0*3+0] + i_v[1] * i_t[1*3+0] + i_v[2] * i_t[2*3+0];
    o_v[1] = i_v[0] * i_t[0*3+1] + i_v[1] * i_t[1*3+1] + i_v[2] * i_t[2*3+1];
    o_v[2] = i_v[0] * i_t[0*3+2] + i_v[1] * i_t[1*3+2] + i_v[2] * i_t[2*3+2];
}
template<typename FPTYPE>
__device__ inline void phys2Inter(FPTYPE *inter, const FPTYPE *phys, const FPTYPE *rec_boxt)
{
    tensorDotVector(inter, phys, rec_boxt);
}
template<typename FPTYPE>
__device__ inline void inter2Phys(FPTYPE *phys, const FPTYPE *inter, const FPTYPE *boxt)
{
    tensorTransDotVector(phys, inter, boxt);
}
template<typename FPTYPE>
__device__ inline FPTYPE compute_volume(const FPTYPE * boxt)
{
    FPTYPE volume =
    boxt[0*3+0] * (boxt[1*3+1]*boxt[2*3+2] - boxt[2*3+1]*boxt[1*3+2]) - 
    boxt[0*3+1] * (boxt[1*3+0]*boxt[2*3+2] - boxt[2*3+0]*boxt[1*3+2]) +
    boxt[0*3+2] * (boxt[1*3+0]*boxt[2*3+1] - boxt[2*3+0]*boxt[1*3+1]);
    return volume;
}

template<typename FPTYPE>
__global__ void _phys2Inter(FPTYPE *inter, const FPTYPE *phys, const FPTYPE *rec_boxt)
{
    phys2Inter(inter, phys, rec_boxt);
}

template<typename FPTYPE>
__global__ void _inter2Phys(FPTYPE *phys, const FPTYPE *inter, const FPTYPE *boxt)
{
    inter2Phys(phys, inter, boxt);
}

template<typename FPTYPE>
__global__ void _compute_volume(FPTYPE * volume, const FPTYPE * boxt)
{
    volume[0] = compute_volume(boxt);
}

namespace deepmd {
//only for unittest
template<typename FPTYPE>
void
convert_to_inter_gpu(
    FPTYPE * ri, 
    const Region<FPTYPE> & region,
    const FPTYPE * rp)
{
    _phys2Inter<<<1, 1>>>(ri, rp, region.rec_boxt);
}

template<typename FPTYPE>
void
convert_to_phys_gpu(
    FPTYPE * rp, 
    const Region<FPTYPE> & region,
    const FPTYPE * ri)
{
    _inter2Phys<<<1, 1>>>(rp, ri, region.boxt);
}

template<typename FPTYPE>
void
volume_gpu(FPTYPE * volume, const Region<FPTYPE> & region)
{
    _compute_volume<<<1, 1>>>(volume, region.boxt);
}

template void convert_to_inter_gpu<float>(float * ri, const Region<float> & region, const float * rp);
template void convert_to_inter_gpu<double>(double * ri, const Region<double> & region, const double * rp);
template void convert_to_phys_gpu<float>(float * rp, const Region<float> & region, const float * ri);
template void convert_to_phys_gpu<double>(double * rp, const Region<double> & region, const double * ri);
template void volume_gpu<float>(float * volume, const Region<float> & region);
template void volume_gpu<double>(double * volume, const Region<double> & region);
}