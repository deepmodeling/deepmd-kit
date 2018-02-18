#include <iostream>
#include <iomanip>
#include "memory.h"
#include "error.h"

#include "Convert.h"
#include "pair_nnp.h"

using namespace LAMMPS_NS;
using namespace std;

PairNNP::PairNNP(LAMMPS *lmp) 
    : Pair(lmp)
      
{
  respa_enable = 0;
  writedata = 0;
  cutoff = 0.;
}

PairNNP::~PairNNP()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairNNP::compute(int eflag, int vflag)
{
  if (eflag || vflag) ev_setup(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int natoms = atom->nlocal;

  vector<int > lmp_type (natoms);
  for (int ii = 0; ii < natoms; ++ii){
    lmp_type[ii] = type[ii];
  }
  Convert cvt (lmp_type);
  
  double dener (0);
  vector<double > dforce (natoms * 3);
  vector<double > dforce_tmp (natoms * 3);
  vector<double > dvirial (9, 0);
  vector<double > dcoord (natoms * 3, 0.);
  vector<double > dcoord_tmp (natoms * 3, 0.);
  vector<int > dtype = cvt.get_type();
  vector<double > dbox (9, 0) ;

  // get box
  dbox[0] = domain->h[0];	// xx
  dbox[4] = domain->h[1];	// yy
  dbox[8] = domain->h[2];	// zz
  dbox[7] = domain->h[3];	// zy
  dbox[6] = domain->h[4];	// zx
  dbox[3] = domain->h[5];	// yx

  // get coord
  for (int ii = 0; ii < natoms; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      dcoord_tmp[ii*3+dd] = x[ii][dd] - domain->boxlo[dd];
    }
  }
  cvt.forward (dcoord, dcoord_tmp, 3);
  
  // compute
  nnp_inter.compute (dener, dforce_tmp, dvirial, dcoord, dtype, dbox);
  
  // get force
  cvt.backward (dforce, dforce_tmp, 3);
  for (int ii = 0; ii < natoms; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      f[ii][dd] += dforce[3*ii+dd];
    }
  }
  
  // accumulate energy and virial
  if (eflag) eng_vdwl += dener;
  if (vflag) {
    virial[0] += 1.0 * dvirial[0];
    virial[1] += 1.0 * dvirial[4];
    virial[2] += 1.0 * dvirial[8];
    virial[3] += 1.0 * dvirial[3];
    virial[4] += 1.0 * dvirial[6];
    virial[5] += 1.0 * dvirial[7];
  }
}


void PairNNP::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;
  for (int i = 1; i <= n; i++)
    setflag[i][i] = 1;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
}

void PairNNP::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  nnp_inter.init (arg[0]);
  cutoff = nnp_inter.cutoff ();
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNNP::coeff(int narg, char **arg)
{
  if (!allocated) {
    allocate();
  }
}


double PairNNP::init_one(int i, int j)
{
  return cutoff;
}

