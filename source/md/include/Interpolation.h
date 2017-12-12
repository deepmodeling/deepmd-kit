#ifndef __wanghan__Interpolation_h__
#define __wanghan__Interpolation_h__

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "Poly.h"

namespace Interpolation {
// linear interpolations
    void pieceLinearInterpol (const double & a,  const double & b, 
			      const double & va, const double & vb,
			      Poly & p);
    void piecewiseLinear (const std::vector<double > & x,
			  const std::vector<double > & y,
			  PiecewisePoly & ps);
// spline interpolations
    void pieceHermiteInterpol (const double & a,  const double & b,
			       const double & va, const double & vb,
			       const double & da, const double & db,
			       Poly & p) ;  
    void pieceSecondDerivativeInterpol (const double & a,  const double & b,
					const double & va, const double & vb,
					const double & dda,const double & ddb,
					Poly & p);
    void piece6OrderInterpol (const double & a,   const double & b,
			      const double & va,  const double & vb,
			      const double & da,  const double & db,
			      const double & dda, const double & ddb,
			      Poly & p);

    bool spline (const std::vector<double > & x,
		 const std::vector<double > & y,
		 PiecewisePoly & ps);
    bool spline (const std::vector<double >::const_iterator xbegin,
		 const std::vector<double >::const_iterator xend,
		 const std::vector<double >::const_iterator ybegin,
		 PiecewisePoly & ps);
    bool splinePeriodic (const std::vector<double > & x,
			 const std::vector<double > & y,
			 PiecewisePoly & ps);
    bool solverForSplinePeriodic (
	const std::vector<double >::const_iterator & lbegin,
	const std::vector<double >::const_iterator & lend,
	const std::vector<double >::iterator & ubegin, 
	const std::vector<double >::iterator & uend);
    void secondDerivativeInterpol (
	const std::vector<double >::const_iterator & xbegin,
	const std::vector<double >::const_iterator & xend,
	const std::vector<double >::const_iterator & vbegin,
	const std::vector<double >::const_iterator & ddbegin,
	PiecewisePoly & ps);
  
}

#endif




