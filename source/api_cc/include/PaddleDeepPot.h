#pragma once
#include "paddle/include/paddle_inference_api.h"
#include "paddle_common.h"
#include "neighbor_list.h"

namespace deepmd{
class PaddleDeepPot
{
public:
  PaddleDeepPot() ;
  ~PaddleDeepPot() ;
  PaddleDeepPot(std::string & prog_str, std::string & params_str);
  void init (std::string & prog_str, std::string & params_str);
  void print_summary(const std::string &pre) const;
public:
  void compute (ENERGYTYPE &			ener,
		std::vector<VALUETYPE> &	force,
		std::vector<VALUETYPE> &	virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box, 
		const std::vector<VALUETYPE>&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>&	aparam = std::vector<VALUETYPE>());
  void compute (ENERGYTYPE &			ener,
		std::vector<VALUETYPE> &	force,
		std::vector<VALUETYPE> &	virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box, 
		const int			nghost,
		const InputNlist &		inlist,
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
		const InputNlist &	lmp_list,
		const int&			ago,
		const std::vector<VALUETYPE>&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>&	aparam = std::vector<VALUETYPE>());
  VALUETYPE cutoff () const {assert(inited); return rcut;};
  int numb_types () const {assert(inited); return ntypes;};
  int dim_fparam () const {assert(inited); return dfparam;};
  int dim_aparam () const {assert(inited); return daparam;};
  void get_type_map (std::string & type_map);
  void validate_fparam_aparam(const int & nloc,
			      const std::vector<VALUETYPE> &fparam,
			      const std::vector<VALUETYPE> &aparam,
			      const int &dfparam,
                              const int &daparam)const ;
public:
  paddle_infer::Config config;
  std::shared_ptr<paddle_infer::Predictor> predictor;
  int math_lib_num_threads;
  bool inited;
  template<class VT> VT get_scalar(const std::string & name) const;

  VALUETYPE rcut = 6.0;
  VALUETYPE cell_size;
  int ntypes = 2;
  int dfparam;
  int daparam;
  
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

  bool init_nbor;
  std::vector<int> sec_a;
  NeighborListData nlist_data;
  InputNlist nlist;
  AtomMap<VALUETYPE> atommap;

  std::vector<int> get_sel_a() const;
};

class PaddleDeepPotModelDevi
{
public:
  PaddleDeepPotModelDevi () ;
  ~PaddleDeepPotModelDevi() ;
  PaddleDeepPotModelDevi  (std::vector<std::string> & prog_strs, std::vector<std::string> & params_strs);
  void init (std::vector<std::string>& prog_strs, std::vector<std::string>& params_strs);
public:
  void compute (std::vector<ENERGYTYPE> &		all_ener,
		std::vector<std::vector<VALUETYPE> > &	all_force,
		std::vector<std::vector<VALUETYPE> > &	all_virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &		atype,
		const std::vector<VALUETYPE> &	box,
		const int			nghost,
		const InputNlist &	lmp_list,
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
		const InputNlist &	lmp_list,
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
  void validate_fparam_aparam(const int & nloc,
			      const std::vector<VALUETYPE> &fparam,
			      const std::vector<VALUETYPE> &aparam,
			      const int &dfparam,
                              const int &daparam)const ;
public:
  unsigned numb_models;
  std::vector<paddle_infer::Config> configs;
  int math_lib_num_threads;
  std::vector<std::shared_ptr<paddle_infer::Predictor>> predictors;
  bool inited;
  template<class VT> VT get_scalar(const std::string name) const;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  std::string model_type;
  std::string model_version;
  int ntypes;
  int dfparam;
  int daparam;
  

  bool init_nbor;
  std::vector<std::vector<int> > sec;
  deepmd::AtomMap<VALUETYPE> atommap;
  NeighborListData nlist_data;
  InputNlist nlist;

  std::vector<std::vector<int> > get_sel() const;
  void cum_sum(const std::vector<std::vector<int> > n_sel);
};
}
