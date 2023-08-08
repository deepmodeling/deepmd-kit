// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <string>

class UnitManager {
 protected:
  UnitManager(){};

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
  static void set(const std::string& name_of_system);

 private:
  static std::string unit_names[];
};
