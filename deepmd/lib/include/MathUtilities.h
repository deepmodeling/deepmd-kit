#ifndef __MathUtilities_h_wanghan__
#define __MathUtilities_h_wanghan__

#include <vector>
#include <cmath>

using namespace std;

  namespace MathUtilities 
  {

    template <typename TYPE>
    TYPE det1d (const TYPE * tensor);

    template <typename TYPE>
    TYPE det2d (const TYPE * tensor);

    template <typename TYPE>
    TYPE det3d (const TYPE * tensor);
    
    template <typename TYPE>
    TYPE det4d (const TYPE * tensor);

    template<typename TYPE, int NN>
    struct det 
    {
      inline TYPE operator () (const TYPE * tensor) const;
    };

    template <typename TYPE>
    TYPE max (const TYPE & v0, const TYPE & v1);

    template <typename TYPE>
    TYPE min (const TYPE & v0, const TYPE & v1);

    template <typename TYPE>
    void dot (TYPE * vec_o, const TYPE * tensor, const TYPE * vec_i);

    template <typename TYPE>
    TYPE dot (const TYPE& x0, const TYPE& y0, const TYPE& z0,
	      const TYPE& x1, const TYPE& y1, const TYPE& z1);

    template <typename TYPE>
    TYPE dot (const TYPE* r0, const TYPE* r1);

    template <typename TYPE>
    void cprod (const TYPE& x0, const TYPE& y0, const TYPE& z0,
		const TYPE& x1, const TYPE& y1, const TYPE& z1,
		TYPE& x2, TYPE& y2, TYPE& z2);

    template <typename TYPE>
    void cprod (const TYPE * r0, const TYPE * r1, TYPE* r2);

    template <typename TYPE>
    TYPE cos (const TYPE& x0, const TYPE& y0, const TYPE& z0,
	      const TYPE& x1, const TYPE& y1, const TYPE& z1);

    template <typename TYPE>
    TYPE angle (const TYPE& x0, const TYPE& y0, const TYPE& z0,
		const TYPE& x1, const TYPE& y1, const TYPE& z1);

    template <typename TYPE>
    inline TYPE invsqrt (const TYPE x);

    template <typename TYPE>
    inline TYPE msp_sqrt (const TYPE x);
    
    template<typename TYPE, typename VALUETYPE>
    inline int searchVec (const vector<TYPE> & vec,
			  const int sta_,
			  const int end_,
			  const VALUETYPE & val);
    template<typename TYPE, typename VALUETYPE>
    inline int lowerBound (const vector<TYPE> & vec,
			   const int sta_,
			   const int end_,
			   const VALUETYPE & val);
    template<typename TYPE, typename VALUETYPE>
    inline int upperBound (const vector<TYPE> & vec,
			   const int sta_,
			   const int end_,
			   const VALUETYPE & val);
    template<typename TYPE, typename VALUETYPE, typename COMPARE>
    inline int upperBound (const vector<TYPE> & vec,
			   const int sta_,
			   const int end_,
			   const VALUETYPE & val,
			   const COMPARE & less);
  };

#include "MathUtilities_Impl.h"

#endif
