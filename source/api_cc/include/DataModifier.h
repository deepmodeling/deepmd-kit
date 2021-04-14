#pragma once

#include "DeepPot.h"

namespace deepmd{
class DipoleChargeModifier
{
public:
  DipoleChargeModifier();
  DipoleChargeModifier(const std::string & model, 
	       const int & gpu_rank = 0, 
	       const std::string & name_scope = "");
  ~DipoleChargeModifier () {};
  void init (const std::string & model, 
	     const int & gpu_rank = 0, 
	     const std::string & name_scope = "");
  void print_summary(const std::string &pre) const;
public:
  void compute (std::vector<VALUETYPE> &		dfcorr_,
		std::vector<VALUETYPE> &		dvcorr_,
		const std::vector<VALUETYPE> &	dcoord_,
		const std::vector<int> &		datype_,
		const std::vector<VALUETYPE> &	dbox, 
		const std::vector<std::pair<int,int>> &	pairs,
		const std::vector<VALUETYPE> &	delef_, 
		const int			nghost,
		const InputNlist &	lmp_list);
  VALUETYPE cutoff () const {assert(inited); return rcut;};
  int numb_types () const {assert(inited); return ntypes;};
  std::vector<int> sel_types () const {assert(inited); return sel_type;};
private:
  tensorflow::Session* session;
  std::string name_scope, name_prefix;
  int num_intra_nthreads, num_inter_nthreads;
  tensorflow::GraphDef graph_def;
  bool inited;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  int ntypes;
  std::string model_type;
  std::vector<int> sel_type;
  template<class VT> VT get_scalar(const std::string & name) const;
  template<class VT> void get_vector(std::vector<VT> & vec, const std::string & name) const;
  void run_model (std::vector<VALUETYPE> &		dforce,
		  std::vector<VALUETYPE> &		dvirial,
		  tensorflow::Session *			session,
		  const std::vector<std::pair<std::string, tensorflow::Tensor>> & input_tensors,
		  const AtomMap<VALUETYPE> &	atommap,
		  const int			nghost);
};
}

