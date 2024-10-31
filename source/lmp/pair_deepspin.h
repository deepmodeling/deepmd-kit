// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef LAMMPS_VERSION_NUMBER
#error Please define LAMMPS_VERSION_NUMBER to yyyymmdd
#endif

#ifdef PAIR_CLASS

PairStyle(deepspin, PairDeepSpin)

#else

#ifndef LMP_PAIR_NNP_SPIN_H
#define LMP_PAIR_NNP_SPIN_H

#include "pair_base.h"
#include <fstream>
#include <iostream>
#include <map>

#include "comm_brick.h"
#define FLOAT_PREC double

namespace LAMMPS_NS {
class CommBrickDeepSpin : public CommBrick {
  friend class PairDeepSpin;
};
class PairDeepSpin : public PairDeepMDBase {
 public:
  PairDeepSpin(class LAMMPS *);
  ~PairDeepSpin() override;
  void compute(int, int) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  
 private:
  CommBrickDeepSpin *commdata_;
};

}  // namespace LAMMPS_NS

#endif
#endif
