#include "common.h"
#include "AtomMap.h"
#include "device.h"

void 
select_by_type(std::vector<int> & fwd_map,
	       std::vector<int> & bkw_map,
	       int & nghost_real, 
	       const std::vector<VALUETYPE> & dcoord_, 
	       const std::vector<int> & datype_,
	       const int & nghost,
	       const std::vector<int> & sel_type_)
{
  std::vector<int> sel_type (sel_type_);
  sort(sel_type.begin(), sel_type.end());  
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  int nloc_real = 0;
  nghost_real = 0;
  fwd_map.resize(nall);
  bkw_map.clear();
  bkw_map.reserve(nall);  
  int cc = 0;
  for (int ii = 0; ii < nall; ++ii){
    // exclude virtual sites
    // select the type with id < ntypes
    if (binary_search(sel_type.begin(), sel_type.end(), datype_[ii])){
      bkw_map.push_back(ii);
      if (ii < nloc) {
	nloc_real += 1;
      }
      else{
	nghost_real += 1;
      }
      fwd_map[ii] = cc;
      cc ++;
    }
    else{
      fwd_map[ii] = -1;
    }
  }  
  assert((nloc_real+nghost_real) == bkw_map.size());  
}	       


void
select_real_atoms(std::vector<int> & fwd_map,
		  std::vector<int> & bkw_map,
		  int & nghost_real,
		  const std::vector<VALUETYPE> & dcoord_, 
		  const std::vector<int> & datype_,
		  const int & nghost,
		  const int & ntypes)
{
  std::vector<int > sel_type;
  for (int ii = 0; ii < ntypes; ++ii){
    sel_type.push_back(ii);
  }
  select_by_type(fwd_map, bkw_map, nghost_real, dcoord_, datype_, nghost, sel_type);
}


void
NeighborListData::
copy_from_nlist(const InputNlist & inlist)
{
  int inum = inlist.inum;
  ilist.resize(inum);
  jlist.resize(inum);
  memcpy(&ilist[0], inlist.ilist, inum*sizeof(int));
  for(int ii = 0; ii < inum; ++ii){
    int jnum = inlist.numneigh[ii];
    jlist[ii].resize(jnum);
    memcpy(&jlist[ii][0], inlist.firstneigh[ii], jnum*sizeof(int));
  }
}


void
NeighborListData::
shuffle(const AtomMap<VALUETYPE> & map)
{
  const std::vector<int> & fwd_map = map.get_fwd_map();
  shuffle(fwd_map);
}

void
NeighborListData::
shuffle(const std::vector<int> & fwd_map)
{
  int nloc = fwd_map.size();
  for(unsigned ii = 0; ii < ilist.size(); ++ii){
    if(ilist[ii] < nloc){
      ilist[ii] = fwd_map[ilist[ii]];
    }
  }
  for(unsigned ii = 0; ii < jlist.size(); ++ii){
    for(unsigned jj = 0; jj < jlist[ii].size(); ++jj){
      if(jlist[ii][jj] < nloc){
	jlist[ii][jj] = fwd_map[jlist[ii][jj]];
      }
    }
  }
}

void
NeighborListData::
shuffle_exclude_empty (const std::vector<int> & fwd_map)
{
  shuffle(fwd_map);
  std::vector<int > new_ilist;
  std::vector<std::vector<int> > new_jlist;
  new_ilist.reserve(ilist.size());
  new_jlist.reserve(jlist.size());
  for(int ii = 0; ii < ilist.size(); ++ii){
    if(ilist[ii] >= 0){
      new_ilist.push_back(ilist[ii]);
    }
  }
  int new_inum = new_ilist.size();
  for(int ii = 0; ii < jlist.size(); ++ii){
    if(ilist[ii] >= 0){
      std::vector<int> tmp_jlist;
      tmp_jlist.reserve(jlist[ii].size());
      for(int jj = 0; jj < jlist[ii].size(); ++jj){
	if(jlist[ii][jj] >= 0){
	  tmp_jlist.push_back(jlist[ii][jj]);
	}
      }
      new_jlist.push_back(tmp_jlist);
    }
  }
  ilist = new_ilist;
  jlist = new_jlist;
}

void 
NeighborListData::
make_inlist(InputNlist & inlist)
{
  int nloc = ilist.size();
  numneigh.resize(nloc);
  firstneigh.resize(nloc);
  for(int ii = 0; ii < nloc; ++ii){
    numneigh[ii] = jlist[ii].size();
    firstneigh[ii] = &jlist[ii][0];
  }
  inlist.inum = nloc;
  inlist.ilist = &ilist[0];
  inlist.numneigh = &numneigh[0];
  inlist.firstneigh = &firstneigh[0];
}

void
checkStatus(const tensorflow::Status& status) {
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(1);
  }
}

void
get_env_nthreads(int & num_intra_nthreads,
		 int & num_inter_nthreads)
{
  num_intra_nthreads = 0;
  num_inter_nthreads = 0;
  const char* env_intra_nthreads = std::getenv("TF_INTRA_OP_PARALLELISM_THREADS");
  const char* env_inter_nthreads = std::getenv("TF_INTER_OP_PARALLELISM_THREADS");
  if (env_intra_nthreads && 
      std::string(env_intra_nthreads) != std::string("") && 
      atoi(env_intra_nthreads) >= 0
      ) {
    num_intra_nthreads = atoi(env_intra_nthreads);
  }
  if (env_inter_nthreads && 
      std::string(env_inter_nthreads) != std::string("") &&
      atoi(env_inter_nthreads) >= 0
      ) {
    num_inter_nthreads = atoi(env_inter_nthreads);
  }
}

std::string
name_prefix(const std::string & scope)
{
  std::string prefix = "";
  if (scope != ""){
    prefix = scope + "/";
  }
  return prefix;
}

int
session_input_tensors (std::vector<std::pair<std::string, Tensor>> & input_tensors,
		       const std::vector<VALUETYPE> &	dcoord_,
		       const int &			ntypes,
		       const std::vector<int> &		datype_,
		       const std::vector<VALUETYPE> &	dbox, 
		       const VALUETYPE &		cell_size,
		       const std::vector<VALUETYPE> &	fparam_,
		       const std::vector<VALUETYPE> &	aparam_,
		       const AtomMap<VALUETYPE>&	atommap,
		       const std::string		scope)
{
  bool b_pbc = (dbox.size() == 9);

  int nframes = 1;
  int nall = dcoord_.size() / 3;
  int nloc = nall;
  assert (nall == datype_.size());

  std::vector<int > datype = atommap.get_type();
  std::vector<int > type_count (ntypes, 0);
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
  if (b_pbc){
    mesh_shape.AddDim(6);
  }
  else {
    mesh_shape.AddDim(0);
  }
  TensorShape natoms_shape ;
  natoms_shape.AddDim (2 + ntypes);
  TensorShape fparam_shape ;
  fparam_shape.AddDim (nframes);
  fparam_shape.AddDim (fparam_.size());
  TensorShape aparam_shape ;
  aparam_shape.AddDim (nframes);
  aparam_shape.AddDim (aparam_.size());
  
#ifdef HIGH_PREC
  Tensor coord_tensor	(DT_DOUBLE, coord_shape);
  Tensor box_tensor	(DT_DOUBLE, box_shape);
  Tensor fparam_tensor  (DT_DOUBLE, fparam_shape);
  Tensor aparam_tensor  (DT_DOUBLE, aparam_shape);
#else
  Tensor coord_tensor	(DT_FLOAT, coord_shape);
  Tensor box_tensor	(DT_FLOAT, box_shape);
  Tensor fparam_tensor  (DT_FLOAT, fparam_shape);
  Tensor aparam_tensor  (DT_FLOAT, aparam_shape);
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
  auto aparam = aparam_tensor.matrix<VALUETYPE> ();

  std::vector<VALUETYPE> dcoord (dcoord_);
  atommap.forward (dcoord.begin(), dcoord_.begin(), 3);
  
  for (int ii = 0; ii < nframes; ++ii){
    for (int jj = 0; jj < nall * 3; ++jj){
      coord(ii, jj) = dcoord[jj];
    }
    if(b_pbc){
      for (int jj = 0; jj < 9; ++jj){
	box(ii, jj) = dbox[jj];
      }
    }
    else{
      for (int jj = 0; jj < 9; ++jj){
	box(ii, jj) = 0.;
      }
    }
    for (int jj = 0; jj < nall; ++jj){
      type(ii, jj) = datype[jj];
    }
    for (int jj = 0; jj < fparam_.size(); ++jj){
      fparam(ii, jj) = fparam_[jj];
    }
    for (int jj = 0; jj < aparam_.size(); ++jj){
      aparam(ii, jj) = aparam_[jj];
    }
  }
  if (b_pbc){
    mesh (1-1) = 0;
    mesh (2-1) = 0;
    mesh (3-1) = 0;
    mesh (4-1) = 0;
    mesh (5-1) = 0;
    mesh (6-1) = 0;
  }
  natoms (0) = nloc;
  natoms (1) = nall;
  for (int ii = 0; ii < ntypes; ++ii) natoms(ii+2) = type_count[ii];

  std::string prefix = "";
  if (scope != ""){
    prefix = scope + "/";
  }
  input_tensors = {
    {prefix+"t_coord",	coord_tensor}, 
    {prefix+"t_type",	type_tensor},
    {prefix+"t_box",	box_tensor},
    {prefix+"t_mesh",	mesh_tensor},
    {prefix+"t_natoms",	natoms_tensor},
  };  
  if (fparam_.size() > 0) {
    input_tensors.push_back({prefix+"t_fparam", fparam_tensor});
  }
  if (aparam_.size() > 0) {
    input_tensors.push_back({prefix+"t_aparam", aparam_tensor});
  }
  return nloc;
}

int
session_input_tensors (std::vector<std::pair<std::string, Tensor>> & input_tensors,
		       const std::vector<VALUETYPE> &	dcoord_,
		       const int &			ntypes,
		       const std::vector<int> &		datype_,
		       const std::vector<VALUETYPE> &	dbox,		    
		       InputNlist &		dlist, 
		       const std::vector<VALUETYPE> &	fparam_,
		       const std::vector<VALUETYPE> &	aparam_,
		       const AtomMap<VALUETYPE>&	atommap,
		       const int			nghost,
           const int      ago,
		       const std::string			scope)
{
  assert (dbox.size() == 9);

  int nframes = 1;
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  assert (nall == datype_.size());  

  std::vector<int > datype = atommap.get_type();
  std::vector<int > type_count (ntypes, 0);
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
  TensorShape aparam_shape ;
  aparam_shape.AddDim (nframes);
  aparam_shape.AddDim (aparam_.size());
  
#ifdef HIGH_PREC
  Tensor coord_tensor	(DT_DOUBLE, coord_shape);
  Tensor box_tensor	(DT_DOUBLE, box_shape);
  Tensor fparam_tensor  (DT_DOUBLE, fparam_shape);
  Tensor aparam_tensor  (DT_DOUBLE, aparam_shape);
#else
  Tensor coord_tensor	(DT_FLOAT, coord_shape);
  Tensor box_tensor	(DT_FLOAT, box_shape);
  Tensor fparam_tensor  (DT_FLOAT, fparam_shape);
  Tensor aparam_tensor  (DT_FLOAT, aparam_shape);
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
  auto aparam = aparam_tensor.matrix<VALUETYPE> ();

  std::vector<VALUETYPE> dcoord (dcoord_);
  atommap.forward (dcoord.begin(), dcoord_.begin(), 3);
  
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
    for (int jj = 0; jj < aparam_.size(); ++jj){
      aparam(ii, jj) = aparam_[jj];
    }
  }
  
  for (int ii = 0; ii < 16; ++ii) mesh(ii) = 0;
  
  const int stride = sizeof(int *) / sizeof(int);
  assert (stride * sizeof(int) == sizeof(int *));
  assert (stride <= 4);
  mesh (0) = ago;
  mesh (1) = dlist.inum;
  mesh (2) = 0;
  mesh (3) = 0;
  memcpy (&mesh(4),  &(dlist.ilist), sizeof(int *));
  memcpy (&mesh(8),  &(dlist.numneigh), sizeof(int *));
  memcpy (&mesh(12), &(dlist.firstneigh), sizeof(int **));

  natoms (0) = nloc;
  natoms (1) = nall;
  for (int ii = 0; ii < ntypes; ++ii) natoms(ii+2) = type_count[ii];

  std::string prefix = "";
  if (scope != ""){
    prefix = scope + "/";
  }
  input_tensors = {
    {prefix+"t_coord",	coord_tensor}, 
    {prefix+"t_type",	type_tensor},
    {prefix+"t_box",	box_tensor},
    {prefix+"t_mesh",	mesh_tensor},
    {prefix+"t_natoms",natoms_tensor},
  };  
  if (fparam_.size() > 0) {
    input_tensors.push_back({prefix+"t_fparam", fparam_tensor});
  }
  if (aparam_.size() > 0) {
    input_tensors.push_back({prefix+"t_aparam", aparam_tensor});
  }
  return nloc;
}

