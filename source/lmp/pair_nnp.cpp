#include <iostream>
#include <iomanip>
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "update.h"
#include "output.h"
#include "error.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"

#include "pair_nnp.h"

using namespace LAMMPS_NS;
using namespace std;

static void 
ana_st (double & max, 
	double & min, 
	double & sum, 
	const vector<double> & vec, 
	const int & nloc) 
{
  if (vec.size() == 0) return;
  max = vec[0];
  min = vec[0];
  sum = vec[0];
  for (unsigned ii = 1; ii < nloc; ++ii){
    if (vec[ii] > max) max = vec[ii];
    if (vec[ii] < min) min = vec[ii];
    sum += vec[ii];
  }
}

PairNNP::PairNNP(LAMMPS *lmp) 
    : Pair(lmp)
      
{
  respa_enable = 0;
  writedata = 0;
  cutoff = 0.;
  numb_models = 0;
  out_freq = 0;

  // set comm size needed by this Pair
  comm_reverse = 1;
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
  if (numb_models == 0) return;
  if (eflag || vflag) ev_setup(eflag,vflag);
  bool do_ghost = true;
  
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = 0;
  if (do_ghost) {
    nghost = atom->nghost;
  }
  int nall = nlocal + nghost;
  int newton_pair = force->newton_pair;

  vector<int > dtype (nall);
  for (int ii = 0; ii < nall; ++ii){
    dtype[ii] = type[ii] - 1;
  }  

  double dener (0);
  vector<double > dforce (nall * 3);
  vector<double > dvirial (9, 0);
  vector<double > dcoord (nall * 3, 0.);
  vector<double > dbox (9, 0) ;

  // get box
  dbox[0] = domain->h[0];	// xx
  dbox[4] = domain->h[1];	// yy
  dbox[8] = domain->h[2];	// zz
  dbox[7] = domain->h[3];	// zy
  dbox[6] = domain->h[4];	// zx
  dbox[3] = domain->h[5];	// yx

  // get coord
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      dcoord[ii*3+dd] = x[ii][dd] - domain->boxlo[dd];
    }
  }
  
  // compute
  if (do_ghost) {
    LammpsNeighborList lmp_list (list->inum, list->ilist, list->numneigh, list->firstneigh);
    if (numb_models == 1) {
      if ( ! (eflag_atom || vflag_atom) ) {      
	nnp_inter.compute (dener, dforce, dvirial, dcoord, dtype, dbox, nghost, lmp_list);
      }
      // do atomic energy and virial
      else {
	vector<double > deatom (nall * 1, 0);
	vector<double > dvatom (nall * 9, 0);
	nnp_inter.compute (dener, dforce, dvirial, deatom, dvatom, dcoord, dtype, dbox, nghost, lmp_list);
	if (eflag_atom) {
	  for (int ii = 0; ii < nlocal; ++ii) eatom[ii] += deatom[ii];
	}
	if (vflag_atom) {
	  for (int ii = 0; ii < nall; ++ii){
	    vatom[ii][0] += 1.0 * dvatom[9*ii+0];
	    vatom[ii][1] += 1.0 * dvatom[9*ii+4];
	    vatom[ii][2] += 1.0 * dvatom[9*ii+8];
	    vatom[ii][3] += 1.0 * dvatom[9*ii+3];
	    vatom[ii][4] += 1.0 * dvatom[9*ii+6];
	    vatom[ii][5] += 1.0 * dvatom[9*ii+7];
	  }
	}
      }
    }
    else {
      if ( (eflag_atom || vflag_atom) ) {
	error->all(FLERR,"Model devi mode does not compute atomic energy nor virial");
      }
      vector<double> 		all_energy;
      vector<vector<double>> 	all_virial;	       
      nnp_inter_model_devi.compute(all_energy, all_force, all_virial, dcoord, dtype, dbox, nghost, lmp_list);      
      nnp_inter_model_devi.compute_avg (dener, all_energy);
      nnp_inter_model_devi.compute_avg (dforce, all_force);
      nnp_inter_model_devi.compute_avg (dvirial, all_virial);
      if (out_freq > 0 && update->ntimestep % out_freq == 0) {
	int rank = comm->me;
	if (newton_pair) {
	  comm->reverse_comm_pair(this);
	}
	vector<double> tmp_avg_f;
	vector<double> std_f;
	nnp_inter_model_devi.compute_avg (tmp_avg_f, all_force);  
	nnp_inter_model_devi.compute_std_f (std_f, tmp_avg_f, all_force);
	double min = 0, max = 0, avg = 0;      
	ana_st(max, min, avg, std_f, nlocal);
	double all_min = 0, all_max = 0, all_avg = 0;
	int all_nlocal = 0;
	MPI_Reduce (&min, &all_min, 1, MPI_DOUBLE, MPI_MIN, 0, world);
	MPI_Reduce (&max, &all_max, 1, MPI_DOUBLE, MPI_MAX, 0, world);
	MPI_Reduce (&avg, &all_avg, 1, MPI_DOUBLE, MPI_SUM, 0, world);
	MPI_Reduce (&nlocal, &all_nlocal, 1, MPI_INT, MPI_SUM, 0, world);
	all_avg /= double(all_nlocal);
	if (rank == 0) {
	  fp << setw(12) << update->ntimestep 
	     << " " << setw(18) << all_max 
	     << " " << setw(18) << all_min
	     << " " << setw(18) << all_avg
	     << endl;
	}
      }
    }
  }
  else {
    if (numb_models == 1) {
      nnp_inter.compute (dener, dforce, dvirial, dcoord, dtype, dbox, nghost);
    }
    else {
      error->all(FLERR,"Serial version does not support model devi");
    }
  }

  // get force
  for (int ii = 0; ii < nall; ++ii){
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
  if (narg <= 0) error->all(FLERR,"Illegal pair_style command");

  if (narg == 1) {
    nnp_inter.init (arg[0]);
    cutoff = nnp_inter.cutoff ();
    numb_models = 1;
  }
  else {
    if (narg < 4) {
      error->all(FLERR,"Illegal pair_style command\nusage:\npair_style deepmd model1 model2 [models...] out_freq out_file\n");
    }    
    vector<string> models;
    for (int ii = 0; ii < narg-2; ++ii){
      models.push_back(arg[ii]);
    }
    out_freq = atoi(arg[narg-2]);
    if (out_freq < 0) error->all(FLERR,"Illegal out_freq, should be >= 0");
    out_file = string(arg[narg-1]);

    nnp_inter_model_devi.init(models);
    cutoff = nnp_inter_model_devi.cutoff();
    numb_models = models.size();
    fp.open (out_file);
    fp << scientific;
    fp << "#"
       << setw(12-1) << "step" 
       << setw(18+1) << "max"
       << setw(18+1) << "min"
       << setw(18+1) << "avg"
       << endl;
  }  
  comm_reverse = numb_models * 3;
  all_force.resize(numb_models);
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


void PairNNP::init_style()
{
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  // neighbor->requests[irequest]->full = 1;  
  // neighbor->requests[irequest]->newton = 2;  
}


double PairNNP::init_one(int i, int j)
{
  return cutoff;
}


/* ---------------------------------------------------------------------- */

int PairNNP::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    for (int dd = 0; dd < numb_models; ++dd){
      buf[m++] = all_force[dd][3*i+0];
      buf[m++] = all_force[dd][3*i+1];
      buf[m++] = all_force[dd][3*i+2];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairNNP::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (int dd = 0; dd < numb_models; ++dd){
      all_force[dd][3*j+0] += buf[m++];
      all_force[dd][3*j+1] += buf[m++];
      all_force[dd][3*j+2] += buf[m++];
    }
  }
}
