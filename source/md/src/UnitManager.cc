#include "UnitManager.h"
#include <cmath>

// unit independent constants
double UnitManager::Degree2Radian		= M_PI / 180.;
double UnitManager::Radian2Degree		= 180. / M_PI;
// unit dependent
double UnitManager::IntegratorMassConstant	= 1.;
double UnitManager::PressureConstant		= 16.60539040;
double UnitManager::BoltzmannConstant		= 8.31445986144858164e-3;
double UnitManager::ElectrostaticConvertion	= 138.93545756169981341199;

string UnitManager::unit_names[] =
{
  "biology",
  "metal",
  "unitless"
};

void
UnitManager::
set (const string & unit)
{
  if (unit == "metal"){
    IntegratorMassConstant	= 1.03642695707516506071e-4;
    PressureConstant		= 1.602176621e6;
    BoltzmannConstant		= 8.6173303e-5;
    ElectrostaticConvertion	= 14.39964535475696995031;
  }
  else if (unit == "unitless"){
    IntegratorMassConstant	= 1.;
    PressureConstant		= 1.;
    BoltzmannConstant		= 1.;
    ElectrostaticConvertion	= 1.;
  }
}

