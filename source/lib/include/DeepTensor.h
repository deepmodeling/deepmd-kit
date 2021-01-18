#pragma once

#include "NNPInter.h"

class DeepTensor
{
public:
  DeepTensor();
  DeepTensor(const string & model, 
	     const int & gpu_rank = 0, 
	     const string &name_scope = "");
  void init (const string & model, 
	     const int & gpu_rank = 0, 
	     const string &name_scope = "");
  void print_summary(const string &pre) const;
public:
  void compute (std::vector<VALUETYPE> &	value,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box,
		const int			nghost = 0);
  void compute (std::vector<VALUETYPE> &	value,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box, 
		const int			nghost,
		const LammpsNeighborList &	lmp_list);
  VALUETYPE cutoff () const {assert(inited); return rcut;};
  int numb_types () const {assert(inited); return ntypes;};
  int output_dim () const {assert(inited); return odim;};
  const std::vector<int> & sel_types () const {assert(inited); return sel_type;};
private:
  Session* session;
  string name_scope;
  int num_intra_nthreads, num_inter_nthreads;
  GraphDef graph_def;
  bool inited;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  int ntypes;
  string model_type;
  int odim;
  std::vector<int> sel_type;
  template<class VT> VT get_scalar(const string & name) const;
  template<class VT> void get_vector (std::vector<VT> & vec, const string & name) const;
  void run_model (std::vector<VALUETYPE> &		d_tensor_,
		  Session *				session, 
		  const std::vector<std::pair<string, Tensor>> & input_tensors,
		  const NNPAtomMap<VALUETYPE> &		nnpmap, 
		  const int				nghost = 0);
  void compute_inner (std::vector<VALUETYPE> &		value,
		      const std::vector<VALUETYPE> &	coord,
		      const std::vector<int> &		atype,
		      const std::vector<VALUETYPE> &	box,
		      const int				nghost = 0);
  void compute_inner (std::vector<VALUETYPE> &		value,
		      const std::vector<VALUETYPE> &	coord,
		      const std::vector<int> &		atype,
		      const std::vector<VALUETYPE> &	box, 
		      const int				nghost,
		      const InternalNeighborList&	lmp_list);
};

