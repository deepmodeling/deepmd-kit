#include "ZMFunctions.h"
#include "UnitManager.h"
#include <cmath>
#include <iostream>

#define M_inv2 (0.5)
#define M_inv4 (0.25)
#define M_inv8 (0.125)
#define M_inv16 (0.06250000000000000000)
#define M_inv48 (.02083333333333333333)

static double
f (const double & r)
{
  return 1./r;
}

static double
D1f (const double & r)
{
  return -1./(r*r);
}

static double
D2f (const double & r)
{
  return 2./(r*r*r);
}

static double
D3f (const double & r)
{
  return -6./(r*r*r*r);
}

static double
D4f (const double & r)
{
  return 24./(r*r*r*r*r);
}

static double
g (const double & alpha,
   const double & r)
{
  return erfc(alpha * r);
}

static double
D1g (const double & alpha,
    const double & r)
{
  double tmp = alpha * r;
  return - M_2_SQRTPI * alpha * exp (-tmp * tmp);
}

static double
D2g (const double & alpha,
     const double & r)
{
  double tmp = alpha * r;
  return M_2_SQRTPI * 2 * alpha * alpha * alpha * r * exp (-tmp * tmp);
}

static double
D3g (const double & alpha,
     const double & r)
{
  double tmp = alpha * r;
  return M_2_SQRTPI * 2 * alpha * alpha * alpha * (1. - 2. * tmp * tmp) * exp (-tmp * tmp);
}

static double
D4g (const double & alpha,
     const double & r)
{
  double tmp = alpha * r;
  double alpha5 = alpha * alpha;
  alpha5 = alpha5 * alpha5 * alpha;
  return M_2_SQRTPI * 4. * alpha5 * (-3. + 2. * tmp * tmp) * r * exp (-tmp * tmp);
}


double ZeroMultipole::
funcV (const double & alpha,
       const double & r)
{
  return f(r) * g(alpha, r);
}

double ZeroMultipole::
funcD1V (const double & alpha,
	 const double & r)
{
  return D1f(r) * g(alpha, r) + f(r) * D1g(alpha, r);
}

double ZeroMultipole::
funcD2V (const double & alpha,
	 const double & r)
{
  return D2f(r) * g(alpha, r) + 2. * D1f(r) * D1g(alpha, r) + f(r) * D2g(alpha, r);
}

double ZeroMultipole::
funcD3V (const double & alpha,
	 const double & r)
{
  return D3f(r) * g(alpha, r) + 3. * D2f(r) * D1g(alpha, r) + 3. * D1f(r) * D2g(alpha, r) + f(r) * D3g(alpha, r);
}

double ZeroMultipole::
funcD4V (const double & alpha,
	 const double & r)
{
  return D4f(r) * g(alpha, r) + 4. * D3f(r) * D1g(alpha, r) + 6. * D2f(r) * D2g(alpha, r) + 4. * D1f(r) * D3g(alpha, r) + f(r) * D4g(alpha, r);
}


void ZeroMultipole::
calCoefficients (const int & ll,
		 const double & alpha,
		 const double & rc,
		 vector<double > & coeff)
{
  coeff.clear ();
  coeff.resize (ll+1);
  double b0, b1, b2, b3, b4;
  double invrc, invrc2, invrc3, invrc4;
  double rc2;
      
  switch (ll) {
  case 0:
      b0 = funcV (alpha,rc);
      coeff[0] = b0;
      break;
  case 1:
      b0 = funcV (alpha,rc);
      b1 = funcD1V (alpha,rc);
      coeff[0] = b0 - M_inv2 * b1 * rc;
      coeff[1] = M_inv2 * b1 / rc;
      break;
  case 2:
      b0 = funcV (alpha,rc);
      b1 = funcD1V (alpha,rc);
      b2 = funcD2V (alpha,rc);
      invrc = 1./rc;
      coeff[0] = M_inv8 * b2 * rc * rc - 5.*M_inv8 * b1 * rc + b0;
      coeff[1] = 3.*M_inv4 * b1 * invrc - M_inv4 * b2;
      coeff[2] = M_inv8 * b2 * invrc * invrc - M_inv8 * b1 * invrc * invrc * invrc;
      break;
  case 3:
      b0 = funcV (alpha,rc);
      b1 = funcD1V (alpha,rc);
      b2 = funcD2V (alpha,rc);
      b3 = funcD3V (alpha,rc);
      invrc = 1./rc;
      invrc2 = invrc * invrc;
      coeff[0] = - M_inv48 * b3 * rc * rc * rc + 3.*M_inv16 * b2 * rc * rc - 11.*M_inv16 * b1 * rc + b0;
      coeff[1] = 15.*M_inv16 * b1 * invrc - 7.*M_inv16 * b2 + M_inv16 * b3 * rc;
      coeff[2] = 5.*M_inv16 * b2 * invrc2 - 5.*M_inv16 * b1 * invrc2 * invrc - M_inv16 * b3 * invrc;
      coeff[3] = M_inv16 * b1 * invrc2 * invrc2 * invrc - M_inv16 * b2 * invrc2 * invrc2 + M_inv48 * b3 * invrc2 * invrc;
      break;
  case 4:
      b0 = funcV (alpha,rc);
      b1 = funcD1V (alpha,rc);
      b2 = funcD2V (alpha,rc);
      b3 = funcD3V (alpha,rc);
      b4 = funcD4V (alpha,rc);
      rc2 = rc * rc;
      invrc = 1./rc;
      invrc2 = invrc * invrc;
      invrc3 = invrc2 * invrc;
      invrc4 = invrc2 * invrc2;
      coeff[0] = 1./384. * b4 * rc2 * rc2 - 7./192. * b3 * rc2 * rc + 29./128. * b2 * rc2 - 93./128. * b1 * rc + b0;
      coeff[1] = 35./32. * b1 * invrc - 19./32. * b2 - 1./96. * b4 * rc2 + M_inv8 * b3 * rc;
      coeff[2] = 1./64. * b4 - 35./64. * b1 * invrc3 + 35./64. * b2 * invrc2 - 5./32. * b3 * invrc;
      coeff[3] = 7./32. * b1 * invrc4 * invrc - 7./32. * b2 * invrc4 + 1./12. * b3 * invrc3 - 1./96. * b4 * invrc2;
      coeff[4] = 5./128. * b2 * invrc4 * invrc2 - 5./128. * b1 * invrc4 * invrc3 - 1./64. * b3 * invrc4 * invrc + 1./384 * b4 * invrc4;
      break;
  default:
      cerr << "ll larger than 4 is not implemented" << endl;
      break;
  }
}


ZeroMultipole::Potential::
Potential ()
    : alpha(0), rc(1.0), ll(0)
{
  calCoefficients (ll, alpha, rc, coeff);
}

ZeroMultipole::Potential::
Potential (const int & ll,
	   const double & alpha,
	   const double & rc)
{
  reinit (ll, alpha, rc);
}

void ZeroMultipole::Potential::
reinit (const int & ll_,
	const double & alpha_,
	const double & rc_)
{
  ll = ll_;
  alpha = alpha_;
  rc = rc_;  
  calCoefficients (ll, alpha, rc, coeff);
}

double ZeroMultipole::Potential::
pot (const double & rr)
{
  if (rr > rc) return 0.;
  double tmp0 = funcV (alpha, rr);
  // double tmp0 = 0.;
  double tmp1 = coeff.back();
  for (int ii = ll-1; ii >= 0; --ii){
    tmp1 = tmp1 * rr * rr + coeff[ii];
  }
  return tmp0 - tmp1;
}

double ZeroMultipole::Potential::
ulpot (const double & rr)
{
  return pot(rr) + coeff[0];
}


double ZeroMultipole::Potential::
mpotp (const double & rr) {
  if (rr > rc) return 0.;
  double tmp0 = - funcD1V (alpha, rr);
  double tmp1 = 2 * ll * coeff[ll];
  for (int ii = ll-1; ii >= 1; --ii){
    tmp1 = tmp1 * rr * rr + coeff[ii] * 2 * ii;
  }
  return tmp0 + tmp1 * rr;
}

double ZeroMultipole::Potential::
mulpotp (const double & rr)
{
  return mpotp (rr);
}



double ZeroMultipole::Potential::
energyCorr (const vector<double > & charges) const
{
  double sum = 0.;
  double factor = UnitManager::ElectrostaticConvertion;
  for (unsigned ii = 0; ii < charges.size(); ++ii){
    sum += charges[ii] * charges[ii];
  }
  
  // return - (coeff[0] * 0.5 + alpha / sqrt(M_PI)) * sum;
  return - (coeff[0] * 0.5 + alpha / sqrt(M_PI)) * sum * factor;
}






