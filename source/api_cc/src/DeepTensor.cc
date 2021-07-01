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

void 
DeepTensor::
print_summary(const std::string &pre) const
{
  std::cout << pre << "installed to:       " + global_install_prefix << std::endl;
  std::cout << pre << "source:             " + global_git_summ << std::endl;
  std::cout << pre << "source branch:      " + global_git_branch << std::endl;
  std::cout << pre << "source commit:      " + global_git_hash << std::endl;
  std::cout << pre << "source commit at:   " + global_git_date << std::endl;
  std::cout << pre << "surpport model ver.:" + global_model_version << std::endl;
  std::cout << pre << "build float prec:   " + global_float_prec << std::endl;
  std::cout << pre << "build with tf inc:  " + global_tf_include_dir << std::endl;
  std::cout << pre << "build with tf lib:  " + global_tf_lib << std::endl;
  std::cout << pre << "set tf intra_op_parallelism_threads: " <<  num_intra_nthreads << std::endl;
  std::cout << pre << "set tf inter_op_parallelism_threads: " <<  num_inter_nthreads << std::endl;
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
		  const std::vector<int> &	sel_fwd,
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
  // Yixiao: newer model may output rank 2 tensor [nframes x (natoms x noutdim)]
  // assert (output_t.dims() == 1), "dim of output tensor should be 1";
  auto ot = output_t.flat<VALUETYPE> ();
  // this is an Eigen Tensor
  int o_size = ot.size();

  std::vector<VALUETYPE> d_tensor (o_size);
  for (unsigned ii = 0; ii < o_size; ++ii){
    d_tensor[ii] = ot(ii);
  }
  // now we map the type-sorted sel-atom tensor back to original order
  // first we have to get the type-sorted select map
  std::vector<int> sel_srt = sel_fwd;
  select_map<int>(sel_srt, sel_fwd, atommap.get_fwd_map(), 1);
  // remove those -1 that correspond to discarded atoms
  std::remove(sel_srt.begin(), sel_srt.end(), -1);
  // now map the tensor back
  d_tensor_.resize(o_size);
  select_map<VALUETYPE>(d_tensor_, d_tensor, sel_srt, odim);
}

void
DeepTensor::
run_model (std::vector<VALUETYPE> &		dglobal_tensor_,
		  std::vector<VALUETYPE> &	dforce_,
		  std::vector<VALUETYPE> &	dvirial_,
		  std::vector<VALUETYPE> &	datom_tensor_,
		  std::vector<VALUETYPE> &	datom_virial_,
		  tensorflow::Session *			session, 
		  const std::vector<std::pair<std::string, tensorflow::Tensor>> & input_tensors,
		  const AtomMap<VALUETYPE> &		atommap, 
		  const std::vector<int> &		sel_fwd,
		  const int				nghost)
{
  unsigned nloc = atommap.get_type().size();
  unsigned nall = nloc + nghost;
  unsigned nsel = nloc - std::count(sel_fwd.begin(), sel_fwd.end(), -1);
  if (nloc == 0) {
    // return empty
    dglobal_tensor_.clear();
    dforce_.clear();
    dvirial_.clear();
    return;
  }

  std::vector<Tensor> output_tensors;
  deepmd::check_status (session->Run(input_tensors, 
			    {name_prefix(name_scope) + "o_global_" + model_type, 
			     name_prefix(name_scope) + "o_force", 
			     name_prefix(name_scope) + "o_virial", 
			     name_prefix(name_scope) + "o_" + model_type,
			     name_prefix(name_scope) + "o_atom_virial"},
			    {}, 
			    &output_tensors));

  Tensor output_gt = output_tensors[0];
  Tensor output_f = output_tensors[1];
  Tensor output_v = output_tensors[2];
  Tensor output_at = output_tensors[3];
  Tensor output_av = output_tensors[4];
  // this is the new model, output has to be rank 2 tensor
  assert (output_gt.dims() == 2), "dim of output tensor should be 2";
  assert (output_f.dims() == 2), "dim of output tensor should be 2";
  assert (output_v.dims() == 2), "dim of output tensor should be 2";
  assert (output_at.dims() == 2), "dim of output tensor should be 2";
  assert (output_av.dims() == 2), "dim of output tensor should be 2";
  // also check the tensor shapes
  assert (output_gt.dim_size(0) == 1), "nframes should match";
  assert (output_gt.dim_size(1) == odim), "dof of global tensor should be odim";  
  assert (output_f.dim_size(0) == 1), "nframes should match";
  assert (output_f.dim_size(1) == odim * nall * 3), "dof of force should be odim * nall * 3";
  assert (output_v.dim_size(0) == 1), "nframes should match";
  assert (output_v.dim_size(1) == odim * 9), "dof of virial should be odim * 9";
  assert (output_at.dim_size(0) == 1), "nframes should match";
  assert (output_at.dim_size(1) == nsel * odim), "dof of atomic tensor should be nsel * odim";  
  assert (output_av.dim_size(0) == 1), "nframes should match";
  assert (output_av.dim_size(1) == odim * nall * 9), "dof of atomic virial should be odim * nall * 9";  

  auto ogt = output_gt.flat <ENERGYTYPE> ();
  auto of = output_f.flat <VALUETYPE> ();
  auto ov = output_v.flat <VALUETYPE> ();
  auto oat = output_at.flat<VALUETYPE> ();
  auto oav = output_av.flat<VALUETYPE> ();

  // global tensor
  dglobal_tensor_.resize(odim);
  for (unsigned ii = 0; ii < odim; ++ii){
    dglobal_tensor_[ii] = ogt(ii);
  }

  // component-wise force
  std::vector<VALUETYPE> dforce (3 * nall * odim);
  for (unsigned ii = 0; ii < odim * nall * 3; ++ii){
    dforce[ii] = of(ii);
  }
  dforce_ = dforce;
  for (unsigned dd = 0; dd < odim; ++dd){
    atommap.backward (dforce_.begin() + (dd * nall * 3), dforce.begin() + (dd * nall * 3), 3);
  }

  // component-wise virial
  dvirial_.resize(odim * 9);
  for (unsigned ii = 0; ii < odim * 9; ++ii){
    dvirial_[ii] = ov(ii);
  }
  
  // atomic tensor
  std::vector<VALUETYPE> datom_tensor (nsel * odim);
  for (unsigned ii = 0; ii < nsel * odim; ++ii){
    datom_tensor[ii] = oat(ii);
  }
  std::vector<int> sel_srt = sel_fwd;
  select_map<int>(sel_srt, sel_fwd, atommap.get_fwd_map(), 1);
  std::remove(sel_srt.begin(), sel_srt.end(), -1);
  datom_tensor_.resize(nsel * odim);
  select_map<VALUETYPE>(datom_tensor_, datom_tensor, sel_srt, odim);

  // component-wise atomic virial
  std::vector<VALUETYPE> datom_virial (9 * nall * odim);
  for (unsigned ii = 0; ii < odim * nall * 9; ++ii){
    datom_virial[ii] = oav(ii);
  }
  datom_virial_ = datom_virial;
  for (unsigned dd = 0; dd < odim; ++dd){
    atommap.backward (datom_virial_.begin() + (dd * nall * 9), datom_virial.begin() + (dd * nall * 9), 9);
  }
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
compute (std::vector<VALUETYPE> &	dglobal_tensor_,
	 std::vector<VALUETYPE> &	dforce_,
	 std::vector<VALUETYPE> &	dvirial_,
	 const std::vector<VALUETYPE> &	dcoord_,
	 const std::vector<int> &	datype_,
	 const std::vector<VALUETYPE> &	dbox)
{
  std::vector<VALUETYPE> tmp_at_, tmp_av_;
  compute(dglobal_tensor_, dforce_, dvirial_, tmp_at_, tmp_av_, dcoord_, datype_, dbox);
}

void
DeepTensor::
compute (std::vector<VALUETYPE> &	dglobal_tensor_,
	 std::vector<VALUETYPE> &	dforce_,
	 std::vector<VALUETYPE> &	dvirial_,
	 const std::vector<VALUETYPE> &	dcoord_,
	 const std::vector<int> &	datype_,
	 const std::vector<VALUETYPE> &	dbox, 
	 const int			nghost,
	 const InputNlist &	lmp_list)
{
  std::vector<VALUETYPE> tmp_at_, tmp_av_;
  compute(dglobal_tensor_, dforce_, dvirial_, tmp_at_, tmp_av_, dcoord_, datype_, dbox, nghost, lmp_list);
}

void
DeepTensor::
compute (std::vector<VALUETYPE> &	dglobal_tensor_,
	 std::vector<VALUETYPE> &	dforce_,
	 std::vector<VALUETYPE> &	dvirial_,
	 std::vector<VALUETYPE> &	datom_tensor_,
	 std::vector<VALUETYPE> &	datom_virial_,
	 const std::vector<VALUETYPE> &	dcoord_,
	 const std::vector<int> &	datype_,
	 const std::vector<VALUETYPE> &	dbox)
{
  std::vector<VALUETYPE> dcoord, dforce, datom_virial;
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
  compute_inner(dglobal_tensor_, dforce, dvirial_, datom_tensor_, datom_virial, dcoord, datype, dbox);
  // bkw map
  dforce_.resize(odim * fwd_map.size() * 3);
  for(int kk = 0; kk < odim; ++kk){
    select_map<VALUETYPE>(dforce_.begin() + kk * fwd_map.size() * 3, dforce.begin() + kk * bkw_map.size() * 3, bkw_map, 3);
  }
  datom_virial_.resize(odim * fwd_map.size() * 9);
  for(int kk = 0; kk < odim; ++kk){
    select_map<VALUETYPE>(datom_virial_.begin() + kk * fwd_map.size() * 9, datom_virial.begin() + kk * bkw_map.size() * 9, bkw_map, 9);
  }
}

void
DeepTensor::
compute (std::vector<VALUETYPE> &	dglobal_tensor_,
	 std::vector<VALUETYPE> &	dforce_,
	 std::vector<VALUETYPE> &	dvirial_,
	 std::vector<VALUETYPE> &	datom_tensor_,
	 std::vector<VALUETYPE> &	datom_virial_,
	 const std::vector<VALUETYPE> &	dcoord_,
	 const std::vector<int> &	datype_,
	 const std::vector<VALUETYPE> &	dbox, 
	 const int			nghost,
	 const InputNlist &	lmp_list)
{
  std::vector<VALUETYPE> dcoord, dforce, datom_virial;
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
  compute_inner(dglobal_tensor_, dforce, dvirial_, datom_tensor_, datom_virial, dcoord, datype, dbox, nghost_real, nlist);
  // bkw map
  dforce_.resize(odim * fwd_map.size() * 3);
  for(int kk = 0; kk < odim; ++kk){
    select_map<VALUETYPE>(dforce_.begin() + kk * fwd_map.size() * 3, dforce.begin() + kk * bkw_map.size() * 3, bkw_map, 3);
  }
  datom_virial_.resize(odim * fwd_map.size() * 9);
  for(int kk = 0; kk < odim; ++kk){
    select_map<VALUETYPE>(datom_virial_.begin() + kk * fwd_map.size() * 9, datom_virial.begin() + kk * bkw_map.size() * 9, bkw_map, 9);
  }
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
  
  std::vector<int> sel_fwd, sel_bkw;
  int nghost_sel;
  // this gives the raw selection map, will pass to run model
  select_by_type(sel_fwd, sel_bkw, nghost_sel, dcoord_, datype_, 0, sel_type);

  std::vector<std::pair<std::string, Tensor>> input_tensors;
  int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, cell_size, std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap, name_scope);
  assert (ret == nloc);

  run_model (dtensor_, session, input_tensors, atommap, sel_fwd);
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

  std::vector<int> sel_fwd, sel_bkw;
  int nghost_sel;
  // this gives the raw selection map, will pass to run model
  select_by_type(sel_fwd, sel_bkw, nghost_sel, dcoord_, datype_, nghost, sel_type);
  sel_fwd.resize(nloc);

  NeighborListData nlist_data;
  nlist_data.copy_from_nlist(nlist_);
  nlist_data.shuffle(atommap);
  InputNlist nlist;
  nlist_data.make_inlist(nlist);

  std::vector<std::pair<std::string, Tensor>> input_tensors;
  int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, nlist, std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap, nghost, 0, name_scope);
  assert (nloc == ret);

  run_model (dtensor_, session, input_tensors, atommap, sel_fwd, nghost);
}

void
DeepTensor::
compute_inner (std::vector<VALUETYPE> &		dglobal_tensor_,
	       std::vector<VALUETYPE> &	dforce_,
	       std::vector<VALUETYPE> &	dvirial_,
	       std::vector<VALUETYPE> &	datom_tensor_,
	       std::vector<VALUETYPE> &	datom_virial_,
	       const std::vector<VALUETYPE> &	dcoord_,
	       const std::vector<int> &		datype_,
	       const std::vector<VALUETYPE> &	dbox)
{
  int nall = dcoord_.size() / 3;
  int nloc = nall;
  AtomMap<VALUETYPE> atommap (datype_.begin(), datype_.begin() + nloc);
  assert (nloc == atommap.get_type().size());
  
  std::vector<int> sel_fwd, sel_bkw;
  int nghost_sel;
  // this gives the raw selection map, will pass to run model
  select_by_type(sel_fwd, sel_bkw, nghost_sel, dcoord_, datype_, 0, sel_type);

  std::vector<std::pair<std::string, Tensor>> input_tensors;
  int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, cell_size, std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap, name_scope);
  assert (ret == nloc);

  run_model (dglobal_tensor_, dforce_, dvirial_, datom_tensor_, datom_virial_, session, input_tensors, atommap, sel_fwd);
}

void
DeepTensor::
compute_inner (std::vector<VALUETYPE> &		dglobal_tensor_,
	       std::vector<VALUETYPE> &	dforce_,
	       std::vector<VALUETYPE> &	dvirial_,
	       std::vector<VALUETYPE> &	datom_tensor_,
	       std::vector<VALUETYPE> &	datom_virial_,
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

  std::vector<int> sel_fwd, sel_bkw;
  int nghost_sel;
  // this gives the raw selection map, will pass to run model
  select_by_type(sel_fwd, sel_bkw, nghost_sel, dcoord_, datype_, nghost, sel_type);
  sel_fwd.resize(nloc);

  NeighborListData nlist_data;
  nlist_data.copy_from_nlist(nlist_);
  nlist_data.shuffle(atommap);
  InputNlist nlist;
  nlist_data.make_inlist(nlist);

  std::vector<std::pair<std::string, Tensor>> input_tensors;
  int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, nlist, std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap, nghost, 0, name_scope);
  assert (nloc == ret);

  run_model (dglobal_tensor_, dforce_, dvirial_, datom_tensor_, datom_virial_, session, input_tensors, atommap, sel_fwd, nghost);
}

