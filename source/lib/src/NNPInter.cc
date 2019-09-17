#include "NNPInter.h"
#include "NNPAtomMap.h"
#include "SimulationRegion.h"
#include <stdexcept>	

static
void
checkStatus(const tensorflow::Status& status) {
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(1);
  }
}

static void
convert_nlist_lmp_internal (InternalNeighborList & list,
			    const LammpsNeighborList & lmp_list) 
{
  list.clear();
  int total_num_nei = 0;
  int inum = lmp_list.inum;
  for (int ii = 0; ii < inum; ++ii){
    total_num_nei += lmp_list.numneigh[ii];
  }
  list.ilist.resize(inum);
  list.jrange.resize(inum+1);
  list.jlist.resize(total_num_nei);
  memcpy(&list.ilist[0], lmp_list.ilist, inum*sizeof(int));
  list.jrange[0] = 0;
  for (int ii = 0; ii < inum; ++ii){
    int jnum = lmp_list.numneigh[ii];
    list.jrange[ii+1] = list.jrange[ii] + jnum;
    const int * jlist = lmp_list.firstneigh[ii];
    memcpy(&(list.jlist[list.jrange[ii]]), jlist, jnum*sizeof(int));
  }
}

static void
shuffle_nlist (InternalNeighborList & list, 
	       const NNPAtomMap<VALUETYPE> & map)
{
  const vector<int> & fwd_map = map.get_fwd_map();
  int nloc = fwd_map.size();
  for (unsigned ii = 0; ii < list.ilist.size(); ++ii){
    if (list.ilist[ii] < nloc) {
      list.ilist[ii] = fwd_map[list.ilist[ii]];
    }
  }
  for (unsigned ii = 0; ii < list.jlist.size(); ++ii){
    if (list.jlist[ii] < nloc) {
      list.jlist[ii] = fwd_map[list.jlist[ii]];
    }
  }
}

static int
make_input_tensors (std::vector<std::pair<string, Tensor>> & input_tensors,
		    const vector<VALUETYPE> &	dcoord_,
		    const int &			ntypes,
		    const vector<int> &		datype_,
		    const vector<VALUETYPE> &	dbox, 
		    const VALUETYPE &		cell_size,
    		    const vector<VALUETYPE>	fparam_,
		    const NNPAtomMap<VALUETYPE>&nnpmap,
		    const int			nghost = 0)
{
  bool b_ghost = (nghost != 0);
  
  assert (dbox.size() == 9);

  int nframes = 1;
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  assert (nall == datype_.size());

  vector<int > datype = nnpmap.get_type();
  vector<int > type_count (ntypes, 0);
  for (unsigned ii = 0; ii < datype.size(); ++ii){
    type_count[datype[ii]] ++;
  }
  datype.insert (datype.end(), datype_.begin() + nloc, datype_.end());

  SimulationRegion<VALUETYPE> region;
  vector<double > dbox_(9);
  for (int dd = 0; dd < 9; ++dd) dbox_[dd] = dbox[dd];
  region.reinitBox (&dbox_[0]);
  double box_l[3];
  region.toFaceDistance (box_l);
  
  vector<int > ncell (3, 2);
  for (int dd = 0; dd < 3; ++dd){
    ncell[dd] = box_l[dd] / cell_size;
    if (ncell[dd] < 2) ncell[dd] = 2;
  }
  vector<int > next(3, 0);
  for (int dd = 0; dd < 3; ++dd){
    double cellh = box_l[dd] / ncell[dd];
    next[dd] = cellh / cell_size;
    if (next[dd] * cellh < cell_size) next[dd]++;
    assert (next[dd] * cellh >= cell_size);
  }

  TensorShape coord_shape ;
  coord_shape.AddDim (nframes);
  coord_shape.AddDim (nall * 3);
  TensorShape type_shape ;
  type_shape.AddDim (nframes);
  type_shape.AddDim (nall);
  TensorShape box_shape ;
  box_shape.AddDim (nframes);
  box_shape.AddDim (9);
  TensorShape mesh_shape ;
  if (!b_ghost){
    mesh_shape.AddDim (6);
  }
  else {
    mesh_shape.AddDim (12);
  }
  TensorShape natoms_shape ;
  natoms_shape.AddDim (2 + ntypes);
  TensorShape fparam_shape ;
  fparam_shape.AddDim (nframes);
  fparam_shape.AddDim (fparam_.size());
  
#ifdef HIGH_PREC
  Tensor coord_tensor	(DT_DOUBLE, coord_shape);
  Tensor box_tensor	(DT_DOUBLE, box_shape);
  Tensor fparam_tensor  (DT_DOUBLE, fparam_shape);
#else
  Tensor coord_tensor	(DT_FLOAT, coord_shape);
  Tensor box_tensor	(DT_FLOAT, box_shape);
  Tensor fparam_tensor  (DT_FLOAT, fparam_shape);
#endif
  Tensor type_tensor	(DT_INT32, type_shape);
  Tensor mesh_tensor	(DT_INT32, mesh_shape);
  Tensor natoms_tensor	(DT_INT32, natoms_shape);

  auto coord = coord_tensor.matrix<VALUETYPE> ();
  auto type = type_tensor.matrix<int> ();
  auto box = box_tensor.matrix<VALUETYPE> ();
  auto mesh = mesh_tensor.flat<int> ();
  auto natoms = natoms_tensor.flat<int> ();  
  auto fparam = fparam_tensor.matrix<VALUETYPE> ();

  vector<VALUETYPE> dcoord (dcoord_);
  nnpmap.forward (dcoord.begin(), dcoord_.begin(), 3);
  
  for (int ii = 0; ii < nframes; ++ii){
    for (int jj = 0; jj < nall * 3; ++jj){
      coord(ii, jj) = dcoord[jj];
    }
    for (int jj = 0; jj < 9; ++jj){
      box(ii, jj) = dbox[jj];
    }
    for (int jj = 0; jj < nall; ++jj){
      type(ii, jj) = datype[jj];
    }
    for (int jj = 0; jj < fparam_.size(); ++jj){
      fparam(ii, jj) = fparam_[jj];
    }
  }
  mesh (1-1) = 0;
  mesh (2-1) = 0;
  mesh (3-1) = 0;
  mesh (4-1) = ncell[0];
  mesh (5-1) = ncell[1];
  mesh (6-1) = ncell[2];
  if (b_ghost){
    mesh(7-1) = -next[0];
    mesh(8-1) = -next[1];
    mesh(9-1) = -next[2];
    mesh(10-1) = ncell[0] + next[0];
    mesh(11-1) = ncell[1] + next[1];
    mesh(12-1) = ncell[2] + next[2];
  }
  natoms (0) = nloc;
  natoms (1) = nall;
  for (int ii = 0; ii < ntypes; ++ii) natoms(ii+2) = type_count[ii];

  if (fparam_.size() == 0) {
    input_tensors = {
      {"t_coord",	coord_tensor}, 
      {"t_type",	type_tensor},
      {"t_box",		box_tensor},
      {"t_mesh",	mesh_tensor},
      {"t_natoms",	natoms_tensor},
    };  
  }
  else {
    input_tensors = {
      {"t_coord",	coord_tensor}, 
      {"t_type",	type_tensor},
      {"t_box",		box_tensor},
      {"t_mesh",	mesh_tensor},
      {"t_natoms",	natoms_tensor},
      {"t_fparam",	fparam_tensor},
    };  
  }

  return nloc;
}

static int
make_input_tensors (std::vector<std::pair<string, Tensor>> & input_tensors,
		    const vector<VALUETYPE> &	dcoord_,
		    const int &			ntypes,
		    const vector<int> &		datype_,
		    const vector<VALUETYPE> &	dbox,		    
		    InternalNeighborList &	dlist, 
    		    const vector<VALUETYPE>	fparam_,
		    const NNPAtomMap<VALUETYPE>&nnpmap,
    		    const int			nghost)
{
  assert (dbox.size() == 9);

  int nframes = 1;
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  assert (nall == datype_.size());

  vector<int > datype = nnpmap.get_type();
  vector<int > type_count (ntypes, 0);
  for (unsigned ii = 0; ii < datype.size(); ++ii){
    type_count[datype[ii]] ++;
  }
  datype.insert (datype.end(), datype_.begin() + nloc, datype_.end());

  TensorShape coord_shape ;
  coord_shape.AddDim (nframes);
  coord_shape.AddDim (nall * 3);
  TensorShape type_shape ;
  type_shape.AddDim (nframes);
  type_shape.AddDim (nall);
  TensorShape box_shape ;
  box_shape.AddDim (nframes);
  box_shape.AddDim (9);
  TensorShape mesh_shape ;
  mesh_shape.AddDim (16);
  TensorShape natoms_shape ;
  natoms_shape.AddDim (2 + ntypes);
  TensorShape fparam_shape ;
  fparam_shape.AddDim (nframes);
  fparam_shape.AddDim (fparam_.size());
  
#ifdef HIGH_PREC
  Tensor coord_tensor	(DT_DOUBLE, coord_shape);
  Tensor box_tensor	(DT_DOUBLE, box_shape);
  Tensor fparam_tensor  (DT_DOUBLE, fparam_shape);
#else
  Tensor coord_tensor	(DT_FLOAT, coord_shape);
  Tensor box_tensor	(DT_FLOAT, box_shape);
  Tensor fparam_tensor  (DT_FLOAT, fparam_shape);
#endif
  Tensor type_tensor	(DT_INT32, type_shape);
  Tensor mesh_tensor	(DT_INT32, mesh_shape);
  Tensor natoms_tensor	(DT_INT32, natoms_shape);

  auto coord = coord_tensor.matrix<VALUETYPE> ();
  auto type = type_tensor.matrix<int> ();
  auto box = box_tensor.matrix<VALUETYPE> ();
  auto mesh = mesh_tensor.flat<int> ();
  auto natoms = natoms_tensor.flat<int> ();
  auto fparam = fparam_tensor.matrix<VALUETYPE> ();

  vector<VALUETYPE> dcoord (dcoord_);
  nnpmap.forward (dcoord.begin(), dcoord_.begin(), 3);
  
  for (int ii = 0; ii < nframes; ++ii){
    for (int jj = 0; jj < nall * 3; ++jj){
      coord(ii, jj) = dcoord[jj];
    }
    for (int jj = 0; jj < 9; ++jj){
      box(ii, jj) = dbox[jj];
    }
    for (int jj = 0; jj < nall; ++jj){
      type(ii, jj) = datype[jj];
    }
    for (int jj = 0; jj < fparam_.size(); ++jj){
      fparam(ii, jj) = fparam_[jj];
    }
  }
  
  for (int ii = 0; ii < 16; ++ii) mesh(ii) = 0;
  
  mesh (0) = sizeof(int *) / sizeof(int);
  assert (mesh(0) * sizeof(int) == sizeof(int *));
  const int & stride = mesh(0);
  mesh (1) = dlist.ilist.size();
  assert (mesh(1) == nloc);
  assert (stride <= 4);
  dlist.make_ptrs();
  memcpy (&mesh(4), &(dlist.pilist), sizeof(int *));
  memcpy (&mesh(8), &(dlist.pjrange), sizeof(int *));
  memcpy (&mesh(12), &(dlist.pjlist), sizeof(int *));

  natoms (0) = nloc;
  natoms (1) = nall;
  for (int ii = 0; ii < ntypes; ++ii) natoms(ii+2) = type_count[ii];

  if (fparam_.size() == 0) {
    input_tensors = {
      {"t_coord",	coord_tensor}, 
      {"t_type",	type_tensor},
      {"t_box",		box_tensor},
      {"t_mesh",	mesh_tensor},
      {"t_natoms",	natoms_tensor},
    };  
  }
  else {
    input_tensors = {
      {"t_coord",	coord_tensor}, 
      {"t_type",	type_tensor},
      {"t_box",		box_tensor},
      {"t_mesh",	mesh_tensor},
      {"t_natoms",	natoms_tensor},
      {"t_fparam",	fparam_tensor},
    };  
  }

  return nloc;
}

static void 
run_model (ENERGYTYPE &			dener,
	   vector<VALUETYPE> &		dforce_,
	   vector<VALUETYPE> &		dvirial,	   
	   Session *			session, 
	   const std::vector<std::pair<string, Tensor>> & input_tensors,
	   const NNPAtomMap<VALUETYPE> &	nnpmap, 
	   const int			nghost = 0)
{
  unsigned nloc = nnpmap.get_type().size();
  unsigned nall = nloc + nghost;
  if (nloc == 0) {
    dener = 0;
    // no backward map needed
    // dforce of size nall * 3
    dforce_.resize(nall * 3);
    fill(dforce_.begin(), dforce_.end(), 0.0);
    // dvirial of size 9
    dvirial.resize(9);
    fill(dvirial.begin(), dvirial.end(), 0.0);
    return;
  }

  std::vector<Tensor> output_tensors;

  checkStatus (session->Run(input_tensors, 
			    {"o_energy", "o_force", "o_virial"}, 
			    {}, 
			    &output_tensors));
  
  Tensor output_e = output_tensors[0];
  Tensor output_f = output_tensors[1];
  Tensor output_v = output_tensors[2];

  auto oe = output_e.flat <ENERGYTYPE> ();
  auto of = output_f.flat <VALUETYPE> ();
  auto ov = output_v.flat <VALUETYPE> ();

  dener = oe(0);
  vector<VALUETYPE> dforce (3 * nall);
  dvirial.resize (9);
  for (unsigned ii = 0; ii < nall * 3; ++ii){
    dforce[ii] = of(ii);
  }
  for (unsigned ii = 0; ii < 9; ++ii){
    dvirial[ii] = ov(ii);
  }
  dforce_ = dforce;
  nnpmap.backward (dforce_.begin(), dforce.begin(), 3);
}

static void 
run_model (ENERGYTYPE &			dener,
	   vector<VALUETYPE> &		dforce_,
	   vector<VALUETYPE> &		dvirial,	   
	   vector<VALUETYPE> &		datom_energy_,
	   vector<VALUETYPE> &		datom_virial_,
	   Session *			session, 
	   const std::vector<std::pair<string, Tensor>> & input_tensors,
	   const NNPAtomMap<VALUETYPE> &nnpmap, 
	   const int			nghost = 0)
{
  unsigned nloc = nnpmap.get_type().size();
  unsigned nall = nloc + nghost;
  if (nloc == 0) {
    dener = 0;
    // no backward map needed
    // dforce of size nall * 3
    dforce_.resize(nall * 3);
    fill(dforce_.begin(), dforce_.end(), 0.0);
    // dvirial of size 9
    dvirial.resize(9);
    fill(dvirial.begin(), dvirial.end(), 0.0);
    // datom_energy_ of size nall
    datom_energy_.resize(nall);
    fill(datom_energy_.begin(), datom_energy_.end(), 0.0);
    // datom_virial_ of size nall * 9
    datom_virial_.resize(nall * 9);
    fill(datom_virial_.begin(), datom_virial_.end(), 0.0);
    return;
  }

  std::vector<Tensor> output_tensors;

  checkStatus (session->Run(input_tensors, 
			    {"o_energy", "o_force", "o_virial", "o_atom_energy", "o_atom_virial"}, 
			    {}, 
			    &output_tensors));

  Tensor output_e = output_tensors[0];
  Tensor output_f = output_tensors[1];
  Tensor output_v = output_tensors[2];
  Tensor output_ae = output_tensors[3];
  Tensor output_av = output_tensors[4];

  auto oe = output_e.flat <ENERGYTYPE> ();
  auto of = output_f.flat <VALUETYPE> ();
  auto ov = output_v.flat <VALUETYPE> ();
  auto oae = output_ae.flat <VALUETYPE> ();
  auto oav = output_av.flat <VALUETYPE> ();

  dener = oe(0);
  vector<VALUETYPE> dforce (3 * nall);
  vector<VALUETYPE> datom_energy (nall, 0);
  vector<VALUETYPE> datom_virial (9 * nall);
  dvirial.resize (9);
  for (int ii = 0; ii < nall * 3; ++ii){
    dforce[ii] = of(ii);
  }
  for (int ii = 0; ii < nloc; ++ii){
    datom_energy[ii] = oae(ii);
  }
  for (int ii = 0; ii < nall * 9; ++ii){
    datom_virial[ii] = oav(ii);
  }
  for (int ii = 0; ii < 9; ++ii){
    dvirial[ii] = ov(ii);
  }
  dforce_ = dforce;
  datom_energy_ = datom_energy;
  datom_virial_ = datom_virial;
  nnpmap.backward (dforce_.begin(), dforce.begin(), 3);
  nnpmap.backward (datom_energy_.begin(), datom_energy.begin(), 1);
  nnpmap.backward (datom_virial_.begin(), datom_virial.begin(), 9);
}

static void
get_env_nthreads(int & num_intra_nthreads,
		 int & num_inter_nthreads)
{
  num_intra_nthreads = 0;
  num_inter_nthreads = 0;
  const char* env_intra_nthreads = std::getenv("OMP_NUM_THREADS");
  const char* env_inter_nthreads = std::getenv("TF_INTER_OP_PARALLELISM_THREADS");
  if (env_intra_nthreads && 
      string(env_intra_nthreads) != string("") && 
      atoi(env_intra_nthreads) >= 0
      ) {
    num_intra_nthreads = atoi(env_intra_nthreads);
  }
  if (env_inter_nthreads && 
      string(env_inter_nthreads) != string("") &&
      atoi(env_inter_nthreads) >= 0
      ) {
    num_inter_nthreads = atoi(env_inter_nthreads);
  }
}


NNPInter::
NNPInter ()
    : inited (false)
{
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
}

NNPInter::
NNPInter (const string & model, const int & gpu_rank)
    : inited (false)
{
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  init(model, gpu_rank);  
}

#ifdef USE_CUDA_TOOLKIT
void
NNPInter::
init (const string & model, const int & gpu_rank)
{
  assert (!inited);
  SessionOptions options;
  options.config.set_inter_op_parallelism_threads(num_inter_nthreads);
  options.config.set_intra_op_parallelism_threads(num_intra_nthreads);
  options.config.set_allow_soft_placement(true);
  options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.9);
  options.config.mutable_gpu_options()->set_allow_growth(true);

  checkStatus (ReadBinaryProto(Env::Default(), model, &graph_def));
  int gpu_num = -1;
  cudaGetDeviceCount(&gpu_num);
  // std::cout << "current number of devices: " << gpu_num << std::endl;
  // set device to GPU only when at least GPU is found
  if (gpu_num > 0) {
    std::string str = "/gpu:";
    str += std::to_string(gpu_rank % gpu_num);
    graph::SetDefaultDevice(str, &graph_def);
    // std::cout << "current device rank: " << str << std::endl;
  }
  checkStatus (NewSession(options, &session));
  checkStatus (session->Create(graph_def));
  rcut = get_scalar<VALUETYPE>("descrpt_attr/rcut");
  cell_size = rcut;
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  dfparam = get_scalar<int>("fitting_attr/dfparam");
  assert(rcut == get_rcut());
  assert(ntypes == get_ntypes());
  if (dfparam < 0) dfparam = 0;
  inited = true;
}
#else
void
NNPInter::
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
  dfparam = get_scalar<int>("fitting_attr/dfparam");
  assert(rcut == get_rcut());
  assert(ntypes == get_ntypes());
  if (dfparam < 0) dfparam = 0;
  // rcut = get_rcut();
  // ntypes = get_ntypes();
  // dfparam = get_dfparam();
  inited = true;
}
#endif

void 
NNPInter::
print_summary(const string &pre) const
{
  cout << pre << "installed to:       " + global_install_prefix << endl;
  cout << pre << "source:             " + global_git_summ << endl;
  cout << pre << "source brach:       " + global_git_branch << endl;
  cout << pre << "source commit:      " + global_git_hash << endl;
  cout << pre << "source commit at:   " + global_git_date << endl;
  cout << pre << "build float prec:   " + global_float_prec << endl;
  cout << pre << "build with tf inc:  " + global_tf_include_dir << endl;
  cout << pre << "build with tf lib:  " + global_tf_lib << endl;
  cout << pre << "set tf intra_op_parallelism_threads: " <<  num_intra_nthreads << endl;
  cout << pre << "set tf inter_op_parallelism_threads: " <<  num_inter_nthreads << endl;
}

template<class VT>
VT
NNPInter::
get_scalar (const string & name) const
{
  std::vector<Tensor> output_tensors;
  checkStatus (session->Run(std::vector<std::pair<string, Tensor>> ({}), 
			    {name.c_str()}, 
			    {}, 
			    &output_tensors));
  Tensor output_rc = output_tensors[0];
  auto orc = output_rc.flat <VT> ();
  return orc(0);
}

void
NNPInter::
compute (ENERGYTYPE &			dener,
	 vector<VALUETYPE> &		dforce_,
	 vector<VALUETYPE> &		dvirial,
	 const vector<VALUETYPE> &	dcoord_,
	 const vector<int> &		datype_,
	 const vector<VALUETYPE> &	dbox, 
	 const int			nghost,
	 const vector<VALUETYPE>	fparam)
{
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  NNPAtomMap<VALUETYPE> nnpmap (datype_.begin(), datype_.begin() + nloc);
  assert (nloc == nnpmap.get_type().size());
  if (fparam.size() != dfparam) {
    throw std::runtime_error("the dim of frame parameter provided is not consistent with what the model uses");
  }

  std::vector<std::pair<string, Tensor>> input_tensors;
  int ret = make_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, cell_size, fparam, nnpmap, nghost);
  assert (ret == nloc);

  run_model (dener, dforce_, dvirial, session, input_tensors, nnpmap, nghost);
}

void
NNPInter::
compute (ENERGYTYPE &			dener,
	 vector<VALUETYPE> &		dforce_,
	 vector<VALUETYPE> &		dvirial,
	 const vector<VALUETYPE> &	dcoord_,
	 const vector<int> &		datype_,
	 const vector<VALUETYPE> &	dbox, 
	 const int			nghost,
	 const LammpsNeighborList &	lmp_list,
	 const vector<VALUETYPE>	fparam)
{
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  NNPAtomMap<VALUETYPE> nnpmap (datype_.begin(), datype_.begin() + nloc);
  assert (nloc == nnpmap.get_type().size());
  if (fparam.size() != dfparam) {
    throw std::runtime_error("the dim of frame parameter provided is not consistent with what the model uses");
  }

  InternalNeighborList nlist;
  convert_nlist_lmp_internal (nlist, lmp_list);
  shuffle_nlist (nlist, nnpmap);

  std::vector<std::pair<string, Tensor>> input_tensors;
  int ret = make_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, nlist, fparam, nnpmap, nghost);
  assert (nloc == ret);

  run_model (dener, dforce_, dvirial, session, input_tensors, nnpmap, nghost);
}


void
NNPInter::
compute (ENERGYTYPE &			dener,
	 vector<VALUETYPE> &		dforce_,
	 vector<VALUETYPE> &		dvirial,
	 vector<VALUETYPE> &		datom_energy_,
	 vector<VALUETYPE> &		datom_virial_,
	 const vector<VALUETYPE> &	dcoord_,
	 const vector<int> &		datype_,
	 const vector<VALUETYPE> &	dbox,
	 const vector<VALUETYPE>	fparam)
{
  NNPAtomMap<VALUETYPE> nnpmap (datype_.begin(), datype_.end());
  if (fparam.size() != dfparam) {
    throw std::runtime_error("the dim of frame parameter provided is not consistent with what the model uses");
  }

  std::vector<std::pair<string, Tensor>> input_tensors;
  int nloc = make_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, cell_size, fparam, nnpmap);

  run_model (dener, dforce_, dvirial, datom_energy_, datom_virial_, session, input_tensors, nnpmap);
}



void
NNPInter::
compute (ENERGYTYPE &			dener,
	 vector<VALUETYPE> &		dforce_,
	 vector<VALUETYPE> &		dvirial,
	 vector<VALUETYPE> &		datom_energy_,
	 vector<VALUETYPE> &		datom_virial_,
	 const vector<VALUETYPE> &	dcoord_,
	 const vector<int> &		datype_,
	 const vector<VALUETYPE> &	dbox, 
	 const int			nghost, 
	 const LammpsNeighborList &	lmp_list,
	 const vector<VALUETYPE>	fparam)
{
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  NNPAtomMap<VALUETYPE> nnpmap (datype_.begin(), datype_.begin() + nloc);
  assert (nloc == nnpmap.get_type().size());
  if (fparam.size() != dfparam) {
    throw std::runtime_error("the dim of frame parameter provided is not consistent with what the model uses");
  }

  InternalNeighborList nlist;
  convert_nlist_lmp_internal (nlist, lmp_list);
  shuffle_nlist (nlist, nnpmap);

  std::vector<std::pair<string, Tensor>> input_tensors;
  int ret = make_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, nlist, fparam, nnpmap, nghost);
  assert (nloc == ret);

  run_model (dener, dforce_, dvirial, datom_energy_, datom_virial_, session, input_tensors, nnpmap, nghost);
}




NNPInterModelDevi::
NNPInterModelDevi ()
    : inited (false), 
      numb_models (0)
{
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
}

NNPInterModelDevi::
NNPInterModelDevi (const vector<string> & models, const int & gpu_rank)
    : inited (false), 
      numb_models (0)
{
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  init(models, gpu_rank);
}

#ifdef USE_CUDA_TOOLKIT
void
NNPInterModelDevi::
init (const vector<string> & models, const int & gpu_rank)
{
  assert (!inited);
  numb_models = models.size();
  sessions.resize(numb_models);
  graph_defs.resize(numb_models);
  SessionOptions options;
  options.config.set_inter_op_parallelism_threads(num_inter_nthreads);
  options.config.set_intra_op_parallelism_threads(num_intra_nthreads);
  options.config.set_allow_soft_placement(true);
  options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.9);
  options.config.mutable_gpu_options()->set_allow_growth(true);
  
  for (unsigned ii = 0; ii < numb_models; ++ii){
    checkStatus (ReadBinaryProto(Env::Default(), models[ii], &graph_defs[ii]));
  }
  int gpu_num = -1;
  cudaGetDeviceCount(&gpu_num);
  // std::cout << "current number of devices: " << gpu_num << std::endl;
  for (unsigned ii = 0; ii < numb_models; ++ii){
    // set device to GPU only when at least GPU is found
    if (gpu_num > 0) {
      std::string str = "/gpu:";
      str += std::to_string(gpu_rank % gpu_num);
      graph::SetDefaultDevice(str, &graph_defs[ii]);
      // std::cout << "current device rank: " << str << std::endl;
    }
    checkStatus (NewSession(options, &(sessions[ii])));
    checkStatus (sessions[ii]->Create(graph_defs[ii]));
  }
  rcut = get_scalar<VALUETYPE>("descrpt_attr/rcut");
  cell_size = rcut;
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  dfparam = get_scalar<int>("fitting_attr/dfparam");
  if (dfparam < 0) dfparam = 0;
  // rcut = get_rcut();
  // cell_size = rcut;
  // ntypes = get_ntypes();
  inited = true;
}
#else
void
NNPInterModelDevi::
init (const vector<string> & models, const int & gpu_rank)
{
  assert (!inited);
  numb_models = models.size();
  sessions.resize(numb_models);
  graph_defs.resize(numb_models);
  SessionOptions options;
  options.config.set_inter_op_parallelism_threads(num_inter_nthreads);
  options.config.set_intra_op_parallelism_threads(num_intra_nthreads);
  for (unsigned ii = 0; ii < numb_models; ++ii){
    checkStatus (NewSession(options, &(sessions[ii])));
    checkStatus (ReadBinaryProto(Env::Default(), models[ii], &graph_defs[ii]));
    checkStatus (sessions[ii]->Create(graph_defs[ii]));
  }
  rcut = get_scalar<VALUETYPE>("descrpt_attr/rcut");
  cell_size = rcut;
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  dfparam = get_scalar<int>("fitting_attr/dfparam");
  if (dfparam < 0) dfparam = 0;
  // rcut = get_rcut();
  // cell_size = rcut;
  // ntypes = get_ntypes();
  inited = true;
}
#endif

template<class VT>
VT
NNPInterModelDevi::
get_scalar(const string name) const 
{
  VT myrcut = 0;
  for (unsigned ii = 0; ii < numb_models; ++ii){
    std::vector<Tensor> output_tensors;
    checkStatus (sessions[ii]->Run(std::vector<std::pair<string, Tensor>> ({}), 
				   {name.c_str()}, 
				   {}, 
				   &output_tensors));
    Tensor output_rc = output_tensors[0];
    auto orc = output_rc.flat <VT> ();
    if (ii == 0){
      myrcut = orc(0);
    }
    else {
      assert (myrcut == orc(0));
    }
  }
  return myrcut;
}

void
NNPInterModelDevi::
compute (ENERGYTYPE &			dener,
	 vector<VALUETYPE> &		dforce_,
	 vector<VALUETYPE> &		dvirial,
	 vector<VALUETYPE> &		model_devi,
	 const vector<VALUETYPE> &	dcoord_,
	 const vector<int> &		datype_,
	 const vector<VALUETYPE> &	dbox,
	 const vector<VALUETYPE>	fparam)
{
  if (numb_models == 0) return;
  if (fparam.size() != dfparam) {
    throw std::runtime_error("the dim of frame parameter provided is not consistent with what the model uses");
  }

  NNPAtomMap<VALUETYPE> nnpmap (datype_.begin(), datype_.end());

  std::vector<std::pair<string, Tensor>> input_tensors;
  int nloc = make_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, cell_size, fparam, nnpmap);

  vector<ENERGYTYPE > all_energy (numb_models);
  vector<vector<VALUETYPE > > all_force (numb_models);
  vector<vector<VALUETYPE > > all_virial (numb_models);

  for (unsigned ii = 0; ii < numb_models; ++ii){
    run_model (all_energy[ii], all_force[ii], all_virial[ii], sessions[ii], input_tensors, nnpmap);
  }

  dener = 0;
  for (unsigned ii = 0; ii < numb_models; ++ii){
    dener += all_energy[ii];
  }
  dener /= VALUETYPE(numb_models);
  compute_avg (dvirial, all_virial);  
  compute_avg (dforce_, all_force);
  
  compute_std_f (model_devi, dforce_, all_force);
  
  // for (unsigned ii = 0; ii < numb_models; ++ii){
  //   cout << all_force[ii][573] << " " << all_force[ii][574] << " " << all_force[ii][575] << endl;
  // }
  // cout << dforce_[573] << " " 
  //      << dforce_[574] << " " 
  //      << dforce_[575] << " " 
  //      << model_devi[191] << endl;
}

void
NNPInterModelDevi::
compute (vector<ENERGYTYPE> &		all_energy,
	 vector<vector<VALUETYPE>> &	all_force,
	 vector<vector<VALUETYPE>> &	all_virial,
	 const vector<VALUETYPE> &	dcoord_,
	 const vector<int> &		datype_,
	 const vector<VALUETYPE> &	dbox,
	 const int			nghost,
	 const LammpsNeighborList &	lmp_list,
	 const vector<VALUETYPE>	fparam)
{
  if (numb_models == 0) return;
  if (fparam.size() != dfparam) {
    throw std::runtime_error("the dim of frame parameter provided is not consistent with what the model uses");
  }
  
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  NNPAtomMap<VALUETYPE> nnpmap (datype_.begin(), datype_.begin() + nloc);
  assert (nloc == nnpmap.get_type().size());

  InternalNeighborList nlist;
  convert_nlist_lmp_internal (nlist, lmp_list);
  shuffle_nlist (nlist, nnpmap);

  std::vector<std::pair<string, Tensor>> input_tensors;
  int ret = make_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, nlist, fparam, nnpmap, nghost);
  assert (nloc == ret);

  all_energy.resize (numb_models);
  all_force.resize (numb_models);
  all_virial.resize (numb_models);

  for (unsigned ii = 0; ii < numb_models; ++ii){
    run_model (all_energy[ii], all_force[ii], all_virial[ii], sessions[ii], input_tensors, nnpmap, nghost);
  }
}

void
NNPInterModelDevi::
compute (vector<ENERGYTYPE> &			all_energy,
	 vector<vector<VALUETYPE>> &		all_force,
	 vector<vector<VALUETYPE>> &		all_virial,
	 vector<vector<VALUETYPE>> &		all_atom_energy,
	 vector<vector<VALUETYPE>> &		all_atom_virial,
	 const vector<VALUETYPE> &		dcoord_,
	 const vector<int> &			datype_,
	 const vector<VALUETYPE> &		dbox,
	 const int				nghost,
	 const LammpsNeighborList &		lmp_list,
	 const vector<VALUETYPE>		fparam)
{
  if (numb_models == 0) return;
  if (fparam.size() != dfparam) {
    throw std::runtime_error("the dim of frame parameter provided is not consistent with what the model uses");
  }
  
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  NNPAtomMap<VALUETYPE> nnpmap (datype_.begin(), datype_.begin() + nloc);
  assert (nloc == nnpmap.get_type().size());

  InternalNeighborList nlist;
  convert_nlist_lmp_internal (nlist, lmp_list);
  shuffle_nlist (nlist, nnpmap);

  std::vector<std::pair<string, Tensor>> input_tensors;
  int ret = make_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, nlist, fparam, nnpmap, nghost);
  assert (nloc == ret);

  all_energy.resize (numb_models);
  all_force .resize (numb_models);
  all_virial.resize (numb_models);
  all_atom_energy.resize (numb_models);
  all_atom_virial.resize (numb_models);  

  for (unsigned ii = 0; ii < numb_models; ++ii){
    run_model (all_energy[ii], all_force[ii], all_virial[ii], all_atom_energy[ii], all_atom_virial[ii], sessions[ii], input_tensors, nnpmap, nghost);
  }
}

void
NNPInterModelDevi::
compute_avg (VALUETYPE &		dener, 
	     const vector<VALUETYPE > &	all_energy) 
{
  assert (all_energy.size() == numb_models);
  if (numb_models == 0) return;

  dener = 0;
  for (unsigned ii = 0; ii < numb_models; ++ii){
    dener += all_energy[ii];
  }
  dener /= (VALUETYPE)(numb_models);  
}

#ifndef HIGH_PREC
void
NNPInterModelDevi::
compute_avg (ENERGYTYPE &		dener, 
	     const vector<ENERGYTYPE >&	all_energy) 
{
  assert (all_energy.size() == numb_models);
  if (numb_models == 0) return;

  dener = 0;
  for (unsigned ii = 0; ii < numb_models; ++ii){
    dener += all_energy[ii];
  }
  dener /= (ENERGYTYPE)(numb_models);  
}
#endif

void
NNPInterModelDevi::
compute_avg (vector<VALUETYPE> &		avg, 
	     const vector<vector<VALUETYPE> > &	xx) 
{
  assert (xx.size() == numb_models);
  if (numb_models == 0) return;
  
  avg.resize(xx[0].size());
  fill (avg.begin(), avg.end(), VALUETYPE(0.));
  
  for (unsigned ii = 0; ii < numb_models; ++ii){
    for (unsigned jj = 0; jj < avg.size(); ++jj){
      avg[jj] += xx[ii][jj];
    }
  }

  for (unsigned jj = 0; jj < avg.size(); ++jj){
    avg[jj] /= VALUETYPE(numb_models);
  }
}


// void
// NNPInterModelDevi::
// compute_std (VALUETYPE &		std, 
// 	     const VALUETYPE &		avg, 
// 	     const vector<VALUETYPE >&	xx)
// {
//   std = 0;
//   assert(xx.size() == numb_models);
//   for (unsigned jj = 0; jj < xx.size(); ++jj){
//     std += (xx[jj] - avg) * (xx[jj] - avg);
//   }
//   std = sqrt(std / VALUETYPE(numb_models));
//   // std = sqrt(std / VALUETYPE(numb_models-));
// }

void
NNPInterModelDevi::
compute_std_e (vector<VALUETYPE> &		std, 
	       const vector<VALUETYPE> &	avg, 
	       const vector<vector<VALUETYPE> >&xx)  
{
  assert (xx.size() == numb_models);
  if (numb_models == 0) return;

  unsigned ndof = avg.size();
  unsigned nloc = ndof;
  assert (nloc == ndof);
  
  std.resize(nloc);
  fill (std.begin(), std.end(), VALUETYPE(0.));
  
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    for (unsigned jj = 0 ; jj < nloc; ++jj){
      const VALUETYPE * tmp_f = &(xx[ii][jj]);
      const VALUETYPE * tmp_avg = &(avg[jj]);
      VALUETYPE vdiff = xx[ii][jj] - avg[jj];
      std[jj] += vdiff * vdiff;
    }
  }

  for (unsigned jj = 0; jj < nloc; ++jj){
    std[jj] = sqrt(std[jj] / VALUETYPE(numb_models));
    // std[jj] = sqrt(std[jj] / VALUETYPE(numb_models-1));
  }
}

void
NNPInterModelDevi::
compute_std_f (vector<VALUETYPE> &		std, 
	       const vector<VALUETYPE> &	avg, 
	       const vector<vector<VALUETYPE> >&xx)  
{
  assert (xx.size() == numb_models);
  if (numb_models == 0) return;

  unsigned ndof = avg.size();
  unsigned nloc = ndof / 3;
  assert (nloc * 3 == ndof);
  
  std.resize(nloc);
  fill (std.begin(), std.end(), VALUETYPE(0.));
  
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    for (unsigned jj = 0 ; jj < nloc; ++jj){
      const VALUETYPE * tmp_f = &(xx[ii][jj*3]);
      const VALUETYPE * tmp_avg = &(avg[jj*3]);
      VALUETYPE vdiff[3];
      vdiff[0] = tmp_f[0] - tmp_avg[0];
      vdiff[1] = tmp_f[1] - tmp_avg[1];
      vdiff[2] = tmp_f[2] - tmp_avg[2];
      std[jj] += MathUtilities::dot(vdiff, vdiff);
    }
  }

  for (unsigned jj = 0; jj < nloc; ++jj){
    std[jj] = sqrt(std[jj] / VALUETYPE(numb_models));
    // std[jj] = sqrt(std[jj] / VALUETYPE(numb_models-1));
  }
}

