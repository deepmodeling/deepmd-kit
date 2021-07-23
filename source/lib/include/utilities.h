#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <cmath>

namespace deepmd{

void cum_sum(
    std::vector<int> & sec, 
    const std::vector<int> & n_sel);

template <typename TYPE>
inline TYPE
dot1 (const TYPE* r0, const TYPE* r1)
{
  return r0[0] * r1[0];
}

template <typename TYPE>
inline TYPE
dot2 (const TYPE* r0, const TYPE* r1)
{
  return r0[0] * r1[0] + r0[1] * r1[1];
}

template <typename TYPE>
inline TYPE
dot3 (const TYPE* r0, const TYPE* r1)
{
  return r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2];
}

template <typename TYPE>
inline TYPE
dot4 (const TYPE* r0, const TYPE* r1)
{
  return r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2] + r0[3] * r1[3];
}

template <typename TYPE>
inline void 
dotmv3 (TYPE * vec_o, const TYPE * tensor, const TYPE * vec_i)
{
  vec_o[0] = dot3(tensor+0, vec_i);
  vec_o[1] = dot3(tensor+3, vec_i);
  vec_o[2] = dot3(tensor+6, vec_i);
}

template <typename TYPE>
inline void
cprod (const TYPE * r0,
       const TYPE * r1,
       TYPE* r2)
{
  r2[0] = r0[1] * r1[2] - r0[2] * r1[1];
  r2[1] = r0[2] * r1[0] - r0[0] * r1[2];
  r2[2] = r0[0] * r1[1] - r0[1] * r1[0];
}

template <typename TYPE>
inline TYPE invsqrt (const TYPE x);

template <>
inline double
invsqrt<double> (const double x) 
{
  return 1./sqrt (x);
}

template <>
inline float
invsqrt<float> (const float x) 
{
  return 1./sqrtf (x);
}

}
