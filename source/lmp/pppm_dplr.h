// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(pppm/dplr, PPPMDPLR)
// clang-format on
#else

#ifndef LMP_PPPM_DPLR_H
#define LMP_PPPM_DPLR_H

#define FLOAT_PREC double

#include <iostream>
#include <vector>

#include "pppm.h"

namespace LAMMPS_NS {

class PPPMDPLR : public PPPM {
 public:
#if LAMMPS_VERSION_NUMBER < 20181109
  // See lammps/lammps#1165
  PPPMDPLR(class LAMMPS *, int, char **);
#else
  PPPMDPLR(class LAMMPS *);
#endif
  ~PPPMDPLR() override{};
  void init() override;
  const std::vector<double> &get_fele() const { return fele; };

 protected:
  void compute(int, int) override;
  void fieldforce_ik() override;
  void fieldforce_ad() override;

 private:
  std::vector<double> fele;
};

}  // namespace LAMMPS_NS

#endif
#endif
