#include "Integrator.h"
#include <cassert>

template <typename VALUETYPE>
void 
Integrator<VALUETYPE>::
stepVeloc (vector<VALUETYPE > & vv,
	   const vector<VALUETYPE > & ff,
	   const vector<VALUETYPE > & mass, 
	   const double & dt, 
	   const vector<int > & freez) const
{
  int natoms = ff.size() / 3;
  for (int kk = 0; kk < natoms; ++kk){
    VALUETYPE invmdt =  dt / (mass[kk] * massConst);
    vv[kk*3+0] += ff[kk*3+0] * invmdt;
    vv[kk*3+1] += ff[kk*3+1] * invmdt;
    vv[kk*3+2] += ff[kk*3+2] * invmdt;
  }  
  for (unsigned ii = 0; ii < freez.size(); ++ii){
    int kk = freez[ii];
    vv[kk*3+0] = 0;
    vv[kk*3+1] = 0;
    vv[kk*3+2] = 0;
  }
}

template <typename VALUETYPE>
void 
Integrator<VALUETYPE>::
stepCoord (vector<VALUETYPE > & rr,
	   const vector<VALUETYPE > & vv, 
	   const double & dt) const
{
  for (unsigned kk = 0; kk < vv.size(); ++kk){
    rr[kk] += dt * vv[kk];
  }  
}


template <typename VALUETYPE> 
ThermostatLangevin<VALUETYPE>::
ThermostatLangevin (const VALUETYPE	T_,
		    const VALUETYPE	tau_,
		    const long long int	seed)
{
  reinit (T_, tau_, seed);
}


template <typename VALUETYPE> 
void 
ThermostatLangevin<VALUETYPE>::
reinit (const VALUETYPE		T_,
	const VALUETYPE		tau_,
	const long long int	seed)
{
  gaussian.set_seed (seed);
  temperature = T_;
  kT = UnitManager::BoltzmannConstant * T_;
  gamma = 1./tau_;
  VALUETYPE twogammakT = 2. * gamma * kT;
  sigma = 1./sqrt (twogammakT) * twogammakT;
  sigmainvsqrt2gamma = VALUETYPE(sigma / sqrt (2. * gamma));  
}


template <typename VALUETYPE>
void
ThermostatLangevin<VALUETYPE>::
stepOU (vector<VALUETYPE> & vv,
	const vector<VALUETYPE > & mass,
	const double & dt, 
	const vector<int > & freez) const
{
  VALUETYPE emgammat = exp (-gamma * dt);
  VALUETYPE sqrt1memgammat2 = sqrt (1. - emgammat * emgammat);
  VALUETYPE prefR = sigmainvsqrt2gamma * sqrt1memgammat2;

  int numb_part =  mass.size();
  assert (int(vv.size() ) == 3 * numb_part);

  double * all_rands = (double *) malloc (sizeof(double) * numb_part * 3);
  gaussian.gen (all_rands, numb_part*3);

  for (int kk = 0; kk < numb_part; ++kk){
    VALUETYPE sm = mass[kk] * UnitManager::IntegratorMassConstant;
    VALUETYPE invsqrtm = 1./sqrt (sm);
    vv[kk*3+0] = emgammat * vv[kk*3+0] + prefR * invsqrtm * all_rands[kk*3+0];
    vv[kk*3+1] = emgammat * vv[kk*3+1] + prefR * invsqrtm * all_rands[kk*3+1];
    vv[kk*3+2] = emgammat * vv[kk*3+2] + prefR * invsqrtm * all_rands[kk*3+2];
  }
  for (unsigned ii = 0; ii < freez.size(); ++ii){
    int kk = freez[ii];
    vv[kk*3+0] = 0;
    vv[kk*3+1] = 0;
    vv[kk*3+2] = 0;
  }

  free (all_rands);
}

template class Integrator<float>;
template class Integrator<double>;
template class ThermostatLangevin<float>;
template class ThermostatLangevin<double>;
