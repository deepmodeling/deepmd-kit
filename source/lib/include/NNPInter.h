#pragma once

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <vector>
#include "version.h"

using namespace tensorflow;
using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
typedef double ENERGYTYPE;
#else 
typedef float  VALUETYPE;
typedef double ENERGYTYPE;
#endif

struct LammpsNeighborList 
{
  int inum;
  const int * ilist;
  const int * numneigh;
  const int *const* firstneigh;
  LammpsNeighborList (int inum_, 
		      const int * ilist_,
		      const int * numneigh_, 
		      const int *const* firstneigh_) 
      : inum(inum_), ilist(ilist_), numneigh(numneigh_), firstneigh(firstneigh_)
      {
      }
};

struct InternalNeighborList 
{
  int * pilist;
  int * pjrange;
  int * pjlist;
  vector<int > ilist;
  vector<int > jrange;
  vector<int > jlist;
  void clear () {ilist.clear(); jrange.clear(); jlist.clear();}
  void make_ptrs () {
    pilist = &ilist[0]; pjrange = &jrange[0]; pjlist = &jlist[0];
  }
};

class NNPInter 
{
public:
  NNPInter () ;
  NNPInter  (const string & model);
  void init (const string & model);
  void print_summary(const string &pre) const;
public:
  void compute (ENERGYTYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box, 
		const int			nghost = 0);
  void compute (ENERGYTYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box, 
		const int			nghost,
		const LammpsNeighborList &	lmp_list);
  void compute (ENERGYTYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		vector<VALUETYPE> &		atom_energy,
		vector<VALUETYPE> &		atom_virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box);
  void compute (ENERGYTYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		vector<VALUETYPE> &		atom_energy,
		vector<VALUETYPE> &		atom_virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box, 
		const int			nghost, 
		const LammpsNeighborList &	lmp_list);
  VALUETYPE cutoff () const {assert(inited); return rcut;};
  int numb_types () const {assert(inited); return ntypes;};
private:
  Session* session;
  int num_intra_nthreads, num_inter_nthreads;
  GraphDef graph_def;
  bool inited;
  VALUETYPE get_rcut () const;
  int get_ntypes () const;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  int ntypes;
};

class NNPInterModelDevi
{
public:
  NNPInterModelDevi () ;
  NNPInterModelDevi  (const vector<string> & models);
  void init (const vector<string> & models);
public:
  void compute (ENERGYTYPE &			ener,
  		vector<VALUETYPE> &		force,
  		vector<VALUETYPE> &		virial,
  		vector<VALUETYPE> &		model_devi,
  		const vector<VALUETYPE> &	coord,
  		const vector<int> &		atype,
  		const vector<VALUETYPE> &	box);
  void compute (vector<ENERGYTYPE> &		all_ener,
		vector<vector<VALUETYPE> > &	all_force,
		vector<vector<VALUETYPE> > &	all_virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box,
		const int			nghost,
		const LammpsNeighborList &	lmp_list);
  void compute (vector<ENERGYTYPE> &		all_ener,
		vector<vector<VALUETYPE> > &	all_force,
		vector<vector<VALUETYPE> > &	all_virial,
		vector<vector<VALUETYPE> > &	all_atom_energy,
		vector<vector<VALUETYPE> > &	all_atom_virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box,
		const int			nghost,
		const LammpsNeighborList &	lmp_list);
  VALUETYPE cutoff () const {assert(inited); return rcut;};
  int numb_types () const {assert(inited); return ntypes;};
#ifndef HIGH_PREC
  void compute_avg (ENERGYTYPE &		dener,
		    const vector<ENERGYTYPE > &	all_energy);
#endif
  void compute_avg (VALUETYPE &			dener,
		    const vector<VALUETYPE > &	all_energy);
  void compute_avg (vector<VALUETYPE> &		avg,
		    const vector<vector<VALUETYPE> > &	xx);
  void compute_std_e (vector<VALUETYPE> &		std,
		      const vector<VALUETYPE> &		avg,
		      const vector<vector<VALUETYPE> >&	xx);
  void compute_std_f (vector<VALUETYPE> &		std,
		      const vector<VALUETYPE> &		avg,
		      const vector<vector<VALUETYPE> >& xx);
private:
  unsigned numb_models;
  vector<Session*> sessions;
  int num_intra_nthreads, num_inter_nthreads;
  vector<GraphDef> graph_defs;
  bool inited;
  VALUETYPE get_rcut () const;
  int get_ntypes () const;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  int ntypes;
};


