// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <vector>

#include "Gaussian.h"
#include "UnitManager.h"

template <typename VALUETYPE>
class Integrator {
 public:
  Integrator() : massConst(UnitManager::IntegratorMassConstant){};

 public:
  void stepVeloc(std::vector<VALUETYPE>& vv,
                 const std::vector<VALUETYPE>& ff,
                 const std::vector<VALUETYPE>& mass,
                 const double& dt,
                 const std::vector<int>& freez = std::vector<int>()) const;
  void stepCoord(std::vector<VALUETYPE>& rr,
                 const std::vector<VALUETYPE>& vv,
                 const double& dt) const;

 private:
  VALUETYPE massConst;
};

template <typename VALUETYPE>
class ThermostatLangevin {
 public:
  ThermostatLangevin(const VALUETYPE T = 300.,
                     const VALUETYPE tau = 1.,
                     const long long int seed = 0);
  void reinit(const VALUETYPE T = 300.,
              const VALUETYPE tau = 1.,
              const long long int seed = 0);
  void stepOU(std::vector<VALUETYPE>& vv,
              const std::vector<VALUETYPE>& mass,
              const double& dt,
              const std::vector<int>& freez = std::vector<int>()) const;

 private:
  mutable Gaussian gaussian;
  std::string scheme;
  VALUETYPE temperature;
  VALUETYPE gamma;
  VALUETYPE sigma;
  VALUETYPE kT;
  VALUETYPE sigmainvsqrt2gamma;
};
