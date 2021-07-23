#pragma once

template<typename FPTYPE>
__device__ inline void tensorDotVector(
    FPTYPE *o_v, 
    const FPTYPE *i_v, 
    const FPTYPE *i_t)
{
    o_v[0] = i_v[0] * i_t[0*3+0] + i_v[1] * i_t[0*3+1] + i_v[2] * i_t[0*3+2];
    o_v[1] = i_v[0] * i_t[1*3+0] + i_v[1] * i_t[1*3+1] + i_v[2] * i_t[1*3+2];
    o_v[2] = i_v[0] * i_t[2*3+0] + i_v[1] * i_t[2*3+1] + i_v[2] * i_t[2*3+2];
}
template<typename FPTYPE>
__device__ inline void tensorTransDotVector(
    FPTYPE *o_v, 
    const FPTYPE *i_v, 
    const FPTYPE *i_t)
{
    o_v[0] = i_v[0] * i_t[0*3+0] + i_v[1] * i_t[1*3+0] + i_v[2] * i_t[2*3+0];
    o_v[1] = i_v[0] * i_t[0*3+1] + i_v[1] * i_t[1*3+1] + i_v[2] * i_t[2*3+1];
    o_v[2] = i_v[0] * i_t[0*3+2] + i_v[1] * i_t[1*3+2] + i_v[2] * i_t[2*3+2];
}
template<typename FPTYPE>
__device__ inline void phys2Inter(
    FPTYPE *inter, 
    const FPTYPE *phys, 
    const FPTYPE *rec_boxt)
{
    tensorDotVector(inter, phys, rec_boxt);
}
template<typename FPTYPE>
__device__ inline void inter2Phys(
    FPTYPE *phys, 
    const FPTYPE *inter, 
    const FPTYPE *boxt)
{
    tensorTransDotVector(phys, inter, boxt);
}
template<typename FPTYPE>
__device__ inline FPTYPE compute_volume(
    const FPTYPE * boxt)
{
    FPTYPE volume =
    boxt[0*3+0] * (boxt[1*3+1]*boxt[2*3+2] - boxt[2*3+1]*boxt[1*3+2]) - 
    boxt[0*3+1] * (boxt[1*3+0]*boxt[2*3+2] - boxt[2*3+0]*boxt[1*3+2]) +
    boxt[0*3+2] * (boxt[1*3+0]*boxt[2*3+1] - boxt[2*3+0]*boxt[1*3+1]);
    return volume;
}