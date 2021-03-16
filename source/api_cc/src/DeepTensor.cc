#include "DeepTensor.h"

using namespace deepmd;
using namespace tensorflow;

DeepTensor::
DeepTensor()
    : inited (false)
{
}

DeepTensor::
DeepTensor(const std::string & model, 
	   const int & gpu_rank, 
	   const std::string &name_scope_)
    : inited (false), name_scope(name_scope_)
{
  init(model, gpu_rank);  
}

void
DeepTensor::
init (const std::string & model, 
      const int & gpu_rank, 
      const std::string &name_scope_)
{
  if (inited){
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do nothing at the second call of initializer" << std::endl;
    return ;
  }
  name_scope = name_scope_;
  SessionOptions options;
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  options.config.set_inter_op_parallelism_threads(num_inter_nthreads);
  options.config.set_intra_op_parallelism_threads(num_intra_nthreads);
  deepmd::check_status (NewSession(options, &session));
  deepmd::check_status (ReadBinaryProto(Env::Default(), model, &graph_def));
  deepmd::check_status (session->Create(graph_def));  
  rcut = get_scalar<VALUETYPE>("descrpt_attr/rcut");
  cell_size = rcut;
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  odim = get_scalar<int>("model_attr/output_dim");
  get_vector<int>(sel_type, "model_attr/sel_type");
  model_type = get_scalar<STRINGTYPE>("model_attr/model_type");
  model_version = get_scalar<STRINGTYPE>("model_attr/model_version");
  if(! model_compatable(model_version)){
    throw std::runtime_error(
	"incompatable model: version " + model_version 
	+ " in graph, but version " + global_model_version 
	+ " supported ");
  }
  inited = true;
}

template<class VT>
VT
DeepTensor::
get_scalar (const std::string & name) const
{
  return session_get_scalar<VT>(session, name, name_scope);
}

template<class VT>
void
DeepTensor::
get_vector (std::vector<VT> & vec, const std::string & name) const
{
  session_get_vector<VT>(vec, session, name, name_scope);
}

void 
DeepTensor::
run_model (std::vector<VALUETYPE> &	d_tensor_,
	   Session *			session, 
	   const std::vector<std::pair<std::string, Tensor>> & input_tensors,
	   const AtomMap<VALUETYPE> &atommap, 
	   const int			nghost)
{
  unsigned nloc = atommap.get_type().size();
  unsigned nall = nloc + nghost;
  if (nloc == 0) {
    // return empty
    d_tensor_.clear();
    return;
  }

  std::vector<Tensor> output_tensors;
  deepmd::check_status (session->Run(input_tensors, 
			    {name_prefix(name_scope) + "o_" + model_type},
			    {}, 
			    &output_tensors));
  
  Tensor output_t = output_tensors[0];
  assert (output_t.dims() == 1), "dim of output tensor should be 1";
  int o_size = output_t.dim_size(0);

  auto ot = output_t.flat<VALUETYPE> ();

  std::vector<VALUETYPE> d_tensor (o_size);
  for (unsigned ii = 0; ii < o_size; ++ii){
    d_tensor[ii] = ot(ii);
  }
  d_tensor_ = d_tensor;
}


void
DeepTensor::
compute (std::vector<VALUETYPE> &	dtensor_,
	 const std::vector<VALUETYPE> &	dcoord_,
	 const std::vector<int> &	datype_,
	 const std::vector<VALUETYPE> &	dbox)
{
  std::vector<VALUETYPE> dcoord;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real;
  select_real_atoms(fwd_map, bkw_map, nghost_real, dcoord_, datype_, 0, ntypes);
  assert(nghost_real == 0);
  // resize to nall_real
  dcoord.resize(bkw_map.size() * 3);
  datype.resize(bkw_map.size());
  // fwd map
  select_map<VALUETYPE>(dcoord, dcoord_, fwd_map, 3);
  select_map<int>(datype, datype_, fwd_map, 1);
  compute_inner(dtensor_, dcoord, datype, dbox);
}

void
DeepTensor::
compute (std::vector<VALUETYPE> &	dtensor_,
	 const std::vector<VALUETYPE> &	dcoord_,
	 const std::vector<int> &	datype_,
	 const std::vector<VALUETYPE> &	dbox, 
	 const int			nghost,
	 const InputNlist &	lmp_list)
{
  std::vector<VALUETYPE> dcoord;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real;
  select_real_atoms(fwd_map, bkw_map, nghost_real, dcoord_, datype_, nghost, ntypes);
  // resize to nall_real
  dcoord.resize(bkw_map.size() * 3);
  datype.resize(bkw_map.size());
  // fwd map
  select_map<VALUETYPE>(dcoord, dcoord_, fwd_map, 3);
  select_map<int>(datype, datype_, fwd_map, 1);
  // internal nlist
  NeighborListData nlist_data;
  nlist_data.copy_from_nlist(lmp_list);
  nlist_data.shuffle_exclude_empty(fwd_map);  
  InputNlist nlist;
  nlist_data.make_inlist(nlist);
  compute_inner(dtensor_, dcoord, datype, dbox, nghost_real, nlist);
}


void
DeepTensor::
compute_inner (std::vector<VALUETYPE> &		dtensor_,
	       const std::vector<VALUETYPE> &	dcoord_,
	       const std::vector<int> &		datype_,
	       const std::vector<VALUETYPE> &	dbox)
{
  int nall = dcoord_.size() / 3;
  int nloc = nall;
  AtomMap<VALUETYPE> atommap (datype_.begin(), datype_.begin() + nloc);
  assert (nloc == atommap.get_type().size());

  std::vector<std::pair<std::string, Tensor>> input_tensors;
  int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, cell_size, std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap, name_scope);
  assert (ret == nloc);

  run_model (dtensor_, session, input_tensors, atommap);
}

void
DeepTensor::
compute_inner (std::vector<VALUETYPE> &		dtensor_,
	       const std::vector<VALUETYPE> &	dcoord_,
	       const std::vector<int> &		datype_,
	       const std::vector<VALUETYPE> &	dbox, 
	       const int			nghost,
	       const InputNlist &	nlist_)
{
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  AtomMap<VALUETYPE> atommap (datype_.begin(), datype_.begin() + nloc);
  assert (nloc == atommap.get_type().size());

  NeighborListData nlist_data;
  nlist_data.copy_from_nlist(nlist_);
  nlist_data.shuffle(atommap);
  InputNlist nlist;
  nlist_data.make_inlist(nlist);

  std::vector<std::pair<std::string, Tensor>> input_tensors;
  int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, nlist, std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap, nghost, 0, name_scope);
  assert (nloc == ret);

  run_model (dtensor_, session, input_tensors, atommap, nghost);
}
