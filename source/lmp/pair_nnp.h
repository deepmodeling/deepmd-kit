/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(deepmd,PairNNP)

#else

#ifndef LMP_PAIR_NNP_H
#define LMP_PAIR_NNP_H

#include "pair.h"
#include "NNPInter.h"
#include <iostream>
#include <fstream>

namespace LAMMPS_NS {

class PairNNP : public Pair {
 public:
  PairNNP(class LAMMPS *);
  virtual ~PairNNP();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int i, int j);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);

 protected:  
  virtual void allocate();

private:  
  NNPInter nnp_inter;
  NNPInterModelDevi nnp_inter_model_devi;
  unsigned numb_models;
  double cutoff;
  vector<vector<double > > all_force;
  ofstream fp;
  int out_freq;
  string out_file;
};

}

#endif
#endif
