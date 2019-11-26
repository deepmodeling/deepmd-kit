#include "DeepTensor.h"

DeepTensor::
DeepTensor()
    : inited (false)
{
}

DeepTensor::
DeepTensor(const string & model, 
	   const int & gpu_rank, 
	   const string &name_scope_)
    : inited (false), name_scope(name_scope_)
{
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  init(model, gpu_rank);  
}

void
DeepTensor::
init (const string & model, 
      const int & gpu_rank, 
      const string &name_scope_)
{
  assert (!inited);
  name_scope = name_scope_;
  SessionOptions options;
  options.config.set_inter_op_parallelism_threads(num_inter_nthreads);
  options.config.set_intra_op_parallelism_threads(num_intra_nthreads);
  checkStatus (NewSession(options, &session));
  checkStatus (ReadBinaryProto(Env::Default(), model, &graph_def));
  checkStatus (session->Create(graph_def));  
  rcut = get_scalar<VALUETYPE>("descrpt_attr/rcut");
  cell_size = rcut;
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  model_type = get_scalar<string>("model_attr/model_type");
  odim = get_scalar<int>("model_attr/output_dim");
  get_vector<int>(sel_type, "model_attr/sel_type");
  inited = true;
}

template<class VT>
VT
DeepTensor::
get_scalar (const string & name) const
{
  return session_get_scalar<VT>(session, name, name_scope);
}

template<class VT>
void
DeepTensor::
get_vector (vector<VT> & vec, const string & name) const
{
  session_get_vector<VT>(vec, session, name, name_scope);
}

void 
DeepTensor::
run_model (vector<VALUETYPE> &		d_tensor_,
	   Session *			session, 
	   const std::vector<std::pair<string, Tensor>> & input_tensors,
	   const NNPAtomMap<VALUETYPE> &nnpmap, 
	   const int			nghost)
{
  unsigned nloc = nnpmap.get_type().size();
  unsigned nall = nloc + nghost;
  if (nloc == 0) {
    // return empty
    d_tensor_.clear();
    return;
  }

  std::vector<Tensor> output_tensors;
  checkStatus (session->Run(input_tensors, 
			    {name_prefix(name_scope) + "o_" + model_type},
			    {}, 
			    &output_tensors));
  
  Tensor output_t = output_tensors[0];
  assert (output_t.dims() == 1), "dim of output tensor should be 1";
  int o_size = output_t.dim_size(0);

  auto ot = output_t.flat<VALUETYPE> ();

  vector<VALUETYPE> d_tensor (o_size);
  for (unsigned ii = 0; ii < o_size; ++ii){
    d_tensor[ii] = ot(ii);
  }
  d_tensor_ = d_tensor;
}


void
DeepTensor::
compute (vector<VALUETYPE> &		dtensor_,
	 const vector<VALUETYPE> &	dcoord_,
	 const vector<int> &		datype_,
	 const vector<VALUETYPE> &	dbox, 
	 const int			nghost)
{
  vector<VALUETYPE> dcoord;
  vector<int> datype, fwd_map, bkw_map;
  int nghost_real;
  select_real_atoms(fwd_map, bkw_map, nghost_real, dcoord_, datype_, nghost, ntypes);
  // resize to nall_real
  dcoord.resize(bkw_map.size() * 3);
  datype.resize(bkw_map.size());
  // fwd map
  select_map<VALUETYPE>(dcoord, dcoord_, fwd_map, 3);
  select_map<int>(datype, datype_, fwd_map, 1);
  compute_inner(dtensor_, dcoord, datype, dbox, nghost_real);
}

void
DeepTensor::
compute (vector<VALUETYPE> &		dtensor_,
	 const vector<VALUETYPE> &	dcoord_,
	 const vector<int> &		datype_,
	 const vector<VALUETYPE> &	dbox, 
	 const int			nghost,
	 const LammpsNeighborList &	lmp_list)
{
  vector<VALUETYPE> dcoord;
  vector<int> datype, fwd_map, bkw_map;
  int nghost_real;
  select_real_atoms(fwd_map, bkw_map, nghost_real, dcoord_, datype_, nghost, ntypes);
  // resize to nall_real
  dcoord.resize(bkw_map.size() * 3);
  datype.resize(bkw_map.size());
  // fwd map
  select_map<VALUETYPE>(dcoord, dcoord_, fwd_map, 3);
  select_map<int>(datype, datype_, fwd_map, 1);
  // internal nlist
  InternalNeighborList nlist;
  convert_nlist_lmp_internal(nlist, lmp_list);
  shuffle_nlist_exclude_empty(nlist, fwd_map);  
  compute_inner(dtensor_, dcoord, datype, dbox, nghost_real, nlist);
}


void
DeepTensor::
compute_inner (vector<VALUETYPE> &		dtensor_,
	       const vector<VALUETYPE> &	dcoord_,
	       const vector<int> &		datype_,
	       const vector<VALUETYPE> &	dbox, 
	       const int			nghost)
{
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  NNPAtomMap<VALUETYPE> nnpmap (datype_.begin(), datype_.begin() + nloc);
  assert (nloc == nnpmap.get_type().size());

  std::vector<std::pair<string, Tensor>> input_tensors;
  int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, cell_size, vector<VALUETYPE>(), vector<VALUETYPE>(), nnpmap, nghost, name_scope);
  assert (ret == nloc);

  run_model (dtensor_, session, input_tensors, nnpmap, nghost);
}

void
DeepTensor::
compute_inner (vector<VALUETYPE> &		dtensor_,
	       const vector<VALUETYPE> &	dcoord_,
	       const vector<int> &		datype_,
	       const vector<VALUETYPE> &	dbox, 
	       const int			nghost,
	       const InternalNeighborList &	nlist_)
{
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  NNPAtomMap<VALUETYPE> nnpmap (datype_.begin(), datype_.begin() + nloc);
  assert (nloc == nnpmap.get_type().size());

  InternalNeighborList nlist(nlist_);
  shuffle_nlist (nlist, nnpmap);

  std::vector<std::pair<string, Tensor>> input_tensors;
  int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, nlist, vector<VALUETYPE>(), vector<VALUETYPE>(), nnpmap, nghost, name_scope);
  assert (nloc == ret);

  run_model (dtensor_, session, input_tensors, nnpmap, nghost);
}
