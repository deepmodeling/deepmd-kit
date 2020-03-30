#pragma once

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "NNPAtomMap.h"
#include <vector>
#include "version.h"

typedef double compute_t;
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
  ~NNPInter() ;
  NNPInter  (const string & model, const int & gpu_rank = 0);
  void init (const string & model, const int & gpu_rank = 0);
  void print_summary(const string &pre) const;
public:
  void compute (ENERGYTYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box, 
		const int			nghost = 0,
		const vector<VALUETYPE>	&	fparam = vector<VALUETYPE>(),
		const vector<VALUETYPE>	&	aparam = vector<VALUETYPE>());
  void compute (ENERGYTYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box, 
		const int			nghost,
		const LammpsNeighborList &	lmp_list,
		const int				&   ago,
		const vector<VALUETYPE>	&	fparam = vector<VALUETYPE>(),
		const vector<VALUETYPE>	&	aparam = vector<VALUETYPE>());
  void compute (ENERGYTYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		vector<VALUETYPE> &		atom_energy,
		vector<VALUETYPE> &		atom_virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box,
		const vector<VALUETYPE>	&	fparam = vector<VALUETYPE>(),
		const vector<VALUETYPE>	&	aparam = vector<VALUETYPE>());
  void compute (ENERGYTYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		vector<VALUETYPE> &		atom_energy,
		vector<VALUETYPE> &		atom_virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box, 
		const int			nghost, 
		const LammpsNeighborList &	lmp_list,
		const int 				&   ago,
		const vector<VALUETYPE>	&	fparam = vector<VALUETYPE>(),
		const vector<VALUETYPE>	&	aparam = vector<VALUETYPE>());
  VALUETYPE cutoff () const {assert(inited); return rcut;};
  int numb_types () const {assert(inited); return ntypes;};
  int dim_fparam () const {assert(inited); return dfparam;};
  int dim_aparam () const {assert(inited); return daparam;};
private:
  Session* session;
  int num_intra_nthreads, num_inter_nthreads;
  GraphDef graph_def;
  bool inited;
  template<class VT> VT get_scalar(const string & name) const;
  // VALUETYPE get_rcut () const;
  // int get_ntypes () const;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  int ntypes;
  int dfparam;
  int daparam;
  void validate_fparam_aparam(const int & nloc,
			      const vector<VALUETYPE> &fparam,
			      const vector<VALUETYPE> &aparam)const ;

  // copy neighbor list info from host
  bool init_nbor;
  std::vector<int> sec_a;
  compute_t *array_double;
  InternalNeighborList nlist;
  NNPAtomMap<VALUETYPE> nnpmap;
  unsigned long long *array_longlong;
  int *ilist, *jrange, *jlist, *array_int;
  int ilist_size, jrange_size, jlist_size;
  int arr_int_size, arr_ll_size, arr_dou_size;

  // function used for neighbor list copy
  vector<int> get_sel_a() const;
#ifdef USE_CUDA_TOOLKIT
  void update_nbor(const InternalNeighborList & nlist, const int nloc);
#endif
};

class NNPInterModelDevi
{
public:
  NNPInterModelDevi () ;
  ~NNPInterModelDevi() ;
  NNPInterModelDevi  (const vector<string> & models, const int & gpu_rank = 0);
  void init (const vector<string> & models, const int & gpu_rank = 0);
public:
  void compute (ENERGYTYPE &			ener,
  		vector<VALUETYPE> &		force,
  		vector<VALUETYPE> &		virial,
  		vector<VALUETYPE> &		model_devi,
  		const vector<VALUETYPE> &	coord,
  		const vector<int> &		atype,
  		const vector<VALUETYPE> &	box,
		const vector<VALUETYPE>	&	fparam = vector<VALUETYPE>(),
		const vector<VALUETYPE>	&	aparam = vector<VALUETYPE>());
  void compute (vector<ENERGYTYPE> &		all_ener,
		vector<vector<VALUETYPE> > &	all_force,
		vector<vector<VALUETYPE> > &	all_virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box,
		const int			nghost,
		const LammpsNeighborList &	lmp_list,
		const int 				&   ago,
		const vector<VALUETYPE>	&	fparam = vector<VALUETYPE>(),
		const vector<VALUETYPE>	&	aparam = vector<VALUETYPE>());
  void compute (vector<ENERGYTYPE> &		all_ener,
		vector<vector<VALUETYPE> > &	all_force,
		vector<vector<VALUETYPE> > &	all_virial,
		vector<vector<VALUETYPE> > &	all_atom_energy,
		vector<vector<VALUETYPE> > &	all_atom_virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box,
		const int			nghost,
		const LammpsNeighborList &	lmp_list,
		const int 				&   ago,
		const vector<VALUETYPE>	&	fparam = vector<VALUETYPE>(),
		const vector<VALUETYPE>	&	aparam = vector<VALUETYPE>());
  VALUETYPE cutoff () const {assert(inited); return rcut;};
  int numb_types () const {assert(inited); return ntypes;};
  int dim_fparam () const {assert(inited); return dfparam;};
  int dim_aparam () const {assert(inited); return daparam;};
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
  void compute_relative_std_f (vector<VALUETYPE> &		std,
		      const vector<VALUETYPE> &		avg,
		      const VALUETYPE eps);
private:
  unsigned numb_models;
  vector<Session*> sessions;
  int num_intra_nthreads, num_inter_nthreads;
  vector<GraphDef> graph_defs;
  bool inited;
  template<class VT> VT get_scalar(const string name) const;
  // VALUETYPE get_rcut () const;
  // int get_ntypes () const;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  int ntypes;
  int dfparam;
  int daparam;
  void validate_fparam_aparam(const int & nloc,
			      const vector<VALUETYPE> &fparam,
			      const vector<VALUETYPE> &aparam)const ;

  // copy neighbor list info from host
  bool init_nbor;
  compute_t *array_double;
  vector<vector<int> > sec;
  InternalNeighborList nlist;
  NNPAtomMap<VALUETYPE> nnpmap;
  unsigned long long *array_longlong;
  int max_sec_size = 0, max_sec_back = 0;
  int *ilist, *jrange, *jlist, *array_int;
  int ilist_size, jrange_size, jlist_size, arr_int_size, arr_ll_size, arr_dou_size;

  // function used for nborlist copy
  void get_max_sec();
  vector<vector<int> > get_sel() const;
  void cum_sum(const std::vector<std::vector<int32> > n_sel);
#ifdef USE_CUDA_TOOLKIT
  void update_nbor(const InternalNeighborList & nlist, const int nloc);
#endif
};


