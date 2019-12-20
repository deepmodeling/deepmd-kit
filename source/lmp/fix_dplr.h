#ifdef FIX_CLASS

FixStyle(dplr,FixDPLR)

#else

#ifndef LMP_FIX_DPLR_H
#define LMP_FIX_DPLR_H

#include <stdio.h>
#include "fix.h"
#include "pair_nnp.h"
#include "DeepTensor.h"
#include "DataModifier.h"

#ifdef HIGH_PREC
#define FLOAT_PREC double
#else
#define FLOAT_PREC float
#endif

namespace LAMMPS_NS {
  class FixDPLR : public Fix {
public:
    FixDPLR(class LAMMPS *, int, char **);
    virtual ~FixDPLR() {};
    int setmask();
    void init();
    void setup(int);
    void post_integrate();
    void pre_force(int);
    void post_force(int);
    int pack_reverse_comm(int, int, double *);
    void unpack_reverse_comm(int, int *, double *);
private:
    PairNNP * pair_nnp;
    DeepTensor dpt;
    DataModifier dtm;
    string model;
    int ntypes;
    vector<int > sel_type;
    vector<int > dpl_type;
    vector<int > bond_type;
    map<int,int > type_asso;
    map<int,int > bk_type_asso;
    vector<FLOAT_PREC> dipole_recd;
    vector<double> dfcorr_buff;
    void get_valid_pairs(vector<pair<int,int> >& pairs);
  };
}

#endif // LMP_FIX_DPLR_H
#endif // FIX_CLASS
