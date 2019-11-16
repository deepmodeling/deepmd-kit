#include "DeepTensor.h"

DeepTensor::
DeepTensor(const string & model, const int & gpu_rank)
    : inited (false)
{
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  init(model, gpu_rank);  
}

void
DeepTensor::
init (const string & model, const int & gpu_rank)
{
  assert (!inited);
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
  inited = true;
}

template<class VT>
VT
DeepTensor::
get_scalar (const string & name) const
{
  return session_get_scalar<VT>(session, name);
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
    // no backward map needed
    d_tensor_.resize(nall * odim);
    fill(d_tensor_.begin(), d_tensor_.end(), 0.0);
    return;
  }

  std::vector<Tensor> output_tensors;
  checkStatus (session->Run(input_tensors, 
			    {"o_" + model_type},
			    {}, 
			    &output_tensors));
  
  Tensor output_t = output_tensors[0];

  auto ot = output_t.flat<VALUETYPE> ();

  vector<VALUETYPE> d_tensor (nall * odim);
  for (unsigned ii = 0; ii < nall * odim; ++ii){
    d_tensor[ii] = ot(ii);
  }
  d_tensor_ = d_tensor;
  nnpmap.backward (d_tensor_.begin(), d_tensor.begin(), odim);
}

void
DeepTensor::
compute (vector<VALUETYPE> &		dtensor_,
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
  int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, cell_size, vector<VALUETYPE>(), vector<VALUETYPE>(), nnpmap, nghost);
  assert (ret == nloc);

  run_model (dtensor_, session, input_tensors, nnpmap, nghost);
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
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  NNPAtomMap<VALUETYPE> nnpmap (datype_.begin(), datype_.begin() + nloc);
  assert (nloc == nnpmap.get_type().size());

  InternalNeighborList nlist;
  convert_nlist_lmp_internal (nlist, lmp_list);
  shuffle_nlist (nlist, nnpmap);

  std::vector<std::pair<string, Tensor>> input_tensors;
  int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, nlist, vector<VALUETYPE>(), vector<VALUETYPE>(), nnpmap, nghost);
  assert (nloc == ret);

  run_model (dtensor_, session, input_tensors, nnpmap, nghost);
}
