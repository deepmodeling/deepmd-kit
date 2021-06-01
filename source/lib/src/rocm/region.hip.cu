#include "hip/hip_runtime.h"
#include "device.h"
#include "region.h"
#include "region.cuh"

template<typename FPTYPE>
__global__ void _phys2Inter(
    FPTYPE *inter, 
    const FPTYPE *phys, 
    const FPTYPE *rec_boxt)
{
    phys2Inter(inter, phys, rec_boxt);
}

template<typename FPTYPE>
__global__ void _inter2Phys(
    FPTYPE *phys, 
    const FPTYPE *inter, 
    const FPTYPE *boxt)
{
    inter2Phys(phys, inter, boxt);
}

template<typename FPTYPE>
__global__ void _compute_volume(
    FPTYPE * volume, 
    const FPTYPE * boxt)
{
    volume[0] = compute_volume(boxt);
}

namespace deepmd {
//only for unittest
template<typename FPTYPE>
void
convert_to_inter_gpu_rocm(
    FPTYPE * ri, 
    const Region<FPTYPE> & region,
    const FPTYPE * rp)
{
    hipLaunchKernelGGL(_phys2Inter, 1, 1, 0, 0, ri, rp, region.rec_boxt);
}

template<typename FPTYPE>
void
convert_to_phys_gpu_rocm(
    FPTYPE * rp, 
    const Region<FPTYPE> & region,
    const FPTYPE * ri)
{
    hipLaunchKernelGGL(_inter2Phys, 1, 1, 0, 0, rp, ri, region.boxt);
}

template<typename FPTYPE>
void
volume_gpu_rocm(
    FPTYPE * volume,
    const Region<FPTYPE> & region)
{
    hipLaunchKernelGGL(_compute_volume, 1, 1, 0, 0, volume, region.boxt);
}

template void convert_to_inter_gpu_rocm<float>(float * ri, const Region<float> & region, const float * rp);
template void convert_to_inter_gpu_rocm<double>(double * ri, const Region<double> & region, const double * rp);
template void convert_to_phys_gpu_rocm<float>(float * rp, const Region<float> & region, const float * ri);
template void convert_to_phys_gpu_rocm<double>(double * rp, const Region<double> & region, const double * ri);
template void volume_gpu_rocm<float>(float * volume, const Region<float> & region);
template void volume_gpu_rocm<double>(double * volume, const Region<double> & region);
}