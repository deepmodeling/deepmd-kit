#ifdef COMPUTE_CLASS

ComputeStyle(deeptensor/atom,ComputeDeeptensorAtom)

#else

#ifndef LMP_COMPUTE_DEEPTENSOR_ATOM_H
#define LMP_COMPUTE_DEEPTENSOR_ATOM_H

#include "compute.h"
#include "pair_deepmd.h"
#ifdef LMPPLUGIN
#include "DeepTensor.h"
#else
#include "deepmd/DeepTensor.h"
#endif

namespace LAMMPS_NS {

class ComputeDeeptensorAtom : public Compute {
 public:
  ComputeDeeptensorAtom(class LAMMPS *, int, char **);
  ~ComputeDeeptensorAtom() override;
  void init() override;
  void compute_peratom() override;
  double memory_usage() override;
  void init_list(int, class NeighList *) override;

 private:
  int nmax;
  double **tensor;
  PairDeepMD dp;
  class NeighList *list;
  deepmd::DeepTensor dt;
  std::vector<int > sel_types;
};

}

#endif
#endif

