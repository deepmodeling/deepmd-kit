#pragma once

#include "Gaussian.h"
#include "UnitManager.h"

#include <vector>
using namespace std;

template <typename VALUETYPE>
class Integrator
{
public:
  Integrator () 
      : massConst (UnitManager::IntegratorMassConstant) {};
public:
  void stepVeloc (vector<VALUETYPE > & vv,
		  const vector<VALUETYPE > & ff,
		  const vector<VALUETYPE > & mass, 
		  const double & dt, 
		  const vector<int > & freez = vector<int> ()) const;
  void stepCoord (vector<VALUETYPE > & rr,
		  const vector<VALUETYPE > & vv, 
		  const double & dt) const;
private:
  VALUETYPE massConst;
};

template <typename VALUETYPE> 
class ThermostatLangevin 
{
public:
  ThermostatLangevin (const VALUETYPE		T = 300.,
		      const VALUETYPE		tau = 1.,
		      const long long int	seed = 0);
  void reinit (const VALUETYPE		T = 300.,
	       const VALUETYPE		tau = 1.,
	       const long long int	seed = 0);
  void stepOU (vector<VALUETYPE> & vv,
	       const vector<VALUETYPE > & mass,
	       const double & dt, 
	       const vector<int > & freez = vector<int> ()) const;
private:
  mutable Gaussian	gaussian;
  string	scheme;
  VALUETYPE	temperature;
  VALUETYPE	gamma;
  VALUETYPE	sigma;
  VALUETYPE	kT;
  VALUETYPE	sigmainvsqrt2gamma;
};
