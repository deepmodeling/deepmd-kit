#pragma once

#include "common.h"
typedef double compute_t;

class NNPInter 
{
public:
  NNPInter () ;
  ~NNPInter() ;
  NNPInter  (const std::string & model, const int & gpu_rank = 0, const std::string & file_content = "");
  void init (const std::string & model, const int & gpu_rank = 0, const std::string & file_content = "");
  void print_summary(const std::string &pre) const;
public:
  void compute (ENERGYTYPE &			ener,
		std::vector<VALUETYPE> &	force,
		std::vector<VALUETYPE> &	virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box, 
		const int			nghost = 0,
		const std::vector<VALUETYPE>&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>&	aparam = std::vector<VALUETYPE>());
  void compute (ENERGYTYPE &			ener,
		std::vector<VALUETYPE> &	force,
		std::vector<VALUETYPE> &	virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box, 
		const int			nghost,
		const LammpsNeighborList &	lmp_list,
		const int&			ago,
		const std::vector<VALUETYPE>&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>&	aparam = std::vector<VALUETYPE>());
  void compute (ENERGYTYPE &			ener,
		std::vector<VALUETYPE> &	force,
		std::vector<VALUETYPE> &	virial,
		std::vector<VALUETYPE> &	atom_energy,
		std::vector<VALUETYPE> &	atom_virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box,
		const std::vector<VALUETYPE>&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>&	aparam = std::vector<VALUETYPE>());
  void compute (ENERGYTYPE &			ener,
		std::vector<VALUETYPE> &	force,
		std::vector<VALUETYPE> &	virial,
		std::vector<VALUETYPE> &	atom_energy,
		std::vector<VALUETYPE> &	atom_virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box, 
		const int			nghost, 
		const LammpsNeighborList &	lmp_list,
		const int&			ago,
		const std::vector<VALUETYPE>&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>&	aparam = std::vector<VALUETYPE>());
  VALUETYPE cutoff () const {assert(inited); return rcut;};
  int numb_types () const {assert(inited); return ntypes;};
  int dim_fparam () const {assert(inited); return dfparam;};
  int dim_aparam () const {assert(inited); return daparam;};
  void get_type_map (std::string & type_map);
private:
  Session* session;
  int num_intra_nthreads, num_inter_nthreads;
  GraphDef graph_def;
  bool inited;
  template<class VT> VT get_scalar(const std::string & name) const;
  // VALUETYPE get_rcut () const;
  // int get_ntypes () const;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  int ntypes;
  int dfparam;
  int daparam;
  void validate_fparam_aparam(const int & nloc,
			      const std::vector<VALUETYPE> &fparam,
			      const std::vector<VALUETYPE> &aparam)const ;
  void compute_inner (ENERGYTYPE &			ener,
		      std::vector<VALUETYPE> &		force,
		      std::vector<VALUETYPE> &		virial,
		      const std::vector<VALUETYPE> &	coord,
		      const std::vector<int> &		atype,
		      const std::vector<VALUETYPE> &	box, 
		      const int				nghost,
		      const int &			ago,
		      const std::vector<VALUETYPE>&	fparam = std::vector<VALUETYPE>(),
		      const std::vector<VALUETYPE>&	aparam = std::vector<VALUETYPE>());

  // copy neighbor list info from host
  bool init_nbor;
  std::vector<int> sec_a;
  compute_t *array_double;
  InternalNeighborList nlist;
  NNPAtomMap<VALUETYPE> nnpmap;
  int *ilist, *jrange, *jlist;
  int ilist_size, jrange_size, jlist_size;

  // function used for neighbor list copy
  std::vector<int> get_sel_a() const;
};

class NNPInterModelDevi
{
public:
  NNPInterModelDevi () ;
  ~NNPInterModelDevi() ;
  NNPInterModelDevi  (const std::vector<std::string> & models, const int & gpu_rank = 0, const std::vector<std::string> & file_contents = std::vector<std::string>());
  void init (const std::vector<std::string> & models, const int & gpu_rank = 0, const std::vector<std::string> & file_contents = std::vector<std::string>());
public:
  void compute (ENERGYTYPE &			ener,
  		std::vector<VALUETYPE> &		force,
  		std::vector<VALUETYPE> &		virial,
  		std::vector<VALUETYPE> &		model_devi,
  		const std::vector<VALUETYPE> &	coord,
  		const std::vector<int> &		atype,
  		const std::vector<VALUETYPE> &	box,
		const std::vector<VALUETYPE>	&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>	&	aparam = std::vector<VALUETYPE>());
  void compute (std::vector<ENERGYTYPE> &		all_ener,
		std::vector<std::vector<VALUETYPE> > &	all_force,
		std::vector<std::vector<VALUETYPE> > &	all_virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &		atype,
		const std::vector<VALUETYPE> &	box,
		const int			nghost,
		const LammpsNeighborList &	lmp_list,
		const int 				&   ago,
		const std::vector<VALUETYPE>	&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>	&	aparam = std::vector<VALUETYPE>());
  void compute (std::vector<ENERGYTYPE> &		all_ener,
		std::vector<std::vector<VALUETYPE> > &	all_force,
		std::vector<std::vector<VALUETYPE> > &	all_virial,
		std::vector<std::vector<VALUETYPE> > &	all_atom_energy,
		std::vector<std::vector<VALUETYPE> > &	all_atom_virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &		atype,
		const std::vector<VALUETYPE> &	box,
		const int			nghost,
		const LammpsNeighborList &	lmp_list,
		const int 				&   ago,
		const std::vector<VALUETYPE>	&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>	&	aparam = std::vector<VALUETYPE>());
  VALUETYPE cutoff () const {assert(inited); return rcut;};
  int numb_types () const {assert(inited); return ntypes;};
  int dim_fparam () const {assert(inited); return dfparam;};
  int dim_aparam () const {assert(inited); return daparam;};
#ifndef HIGH_PREC
  void compute_avg (ENERGYTYPE &		dener,
		    const std::vector<ENERGYTYPE > &	all_energy);
#endif
  void compute_avg (VALUETYPE &			dener,
		    const std::vector<VALUETYPE > &	all_energy);
  void compute_avg (std::vector<VALUETYPE> &		avg,
		    const std::vector<std::vector<VALUETYPE> > &	xx);
  void compute_std_e (std::vector<VALUETYPE> &		std,
		      const std::vector<VALUETYPE> &		avg,
		      const std::vector<std::vector<VALUETYPE> >&	xx);
  void compute_std_f (std::vector<VALUETYPE> &		std,
		      const std::vector<VALUETYPE> &		avg,
		      const std::vector<std::vector<VALUETYPE> >& xx);
  void compute_relative_std_f (std::vector<VALUETYPE> &		std,
		      const std::vector<VALUETYPE> &		avg,
		      const VALUETYPE eps);
private:
  unsigned numb_models;
  std::vector<Session*> sessions;
  int num_intra_nthreads, num_inter_nthreads;
  std::vector<GraphDef> graph_defs;
  bool inited;
  template<class VT> VT get_scalar(const std::string name) const;
  // VALUETYPE get_rcut () const;
  // int get_ntypes () const;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  int ntypes;
  int dfparam;
  int daparam;
  void validate_fparam_aparam(const int & nloc,
			      const std::vector<VALUETYPE> &fparam,
			      const std::vector<VALUETYPE> &aparam)const ;

  // copy neighbor list info from host
  bool init_nbor;
  compute_t *array_double;
  std::vector<std::vector<int> > sec;
  InternalNeighborList nlist;
  NNPAtomMap<VALUETYPE> nnpmap;
  int *ilist, *jrange, *jlist;
  int ilist_size, jrange_size, jlist_size;

  // function used for nborlist copy
  std::vector<std::vector<int> > get_sel() const;
  void cum_sum(const std::vector<std::vector<int32> > n_sel);
};


