#pragma once

#include <string>
using namespace std;

class UnitManager
{
protected:
  UnitManager () {};
public:
  static double Degree2Radian;
  static double Radian2Degree;

  static double IntegratorMassConstant;
  static double PressureConstant;
  static double BoltzmannConstant;
  static double ElectrostaticConvertion;

  static double DefaultTableUpperLimit;
  static double DefaultTableStep;
  static double DefaultTableExtension;
  static void set (const string & name_of_system);
private :
  static string	unit_names[];
};
