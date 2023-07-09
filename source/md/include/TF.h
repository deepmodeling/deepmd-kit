// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <string>
#include <vector>

#include "AdWeight.h"
#include "common.h"

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else
typedef float VALUETYPE;
#endif

class TF {
 public:
  TF(const std::string& filename);

 public:
  void apply(std::vector<VALUETYPE>& force,
             const std::vector<VALUETYPE>& coord,
             const AdWeight& weight) const;

 private:
  VALUETYPE meas(const VALUETYPE& xx) const;
  std::vector<double> data;
  double hh;
  double xup;
};
