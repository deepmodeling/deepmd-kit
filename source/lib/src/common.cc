#include "common.h"
#include "NNPAtomMap.h"
#include "SimulationRegion.h"

void 
select_by_type(vector<int> & fwd_map,
	       vector<int> & bkw_map,
	       int & nghost_real, 
	       const vector<VALUETYPE> & dcoord_, 
	       const vector<int> & datype_,
	       const int & nghost,
	       const vector<int> & sel_type_)
{
  vector<int> sel_type (sel_type_);
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
select_real_atoms(vector<int> & fwd_map,
		  vector<int> & bkw_map,
		  int & nghost_real,
		  const vector<VALUETYPE> & dcoord_, 
		  const vector<int> & datype_,
		  const int & nghost,
		  const int & ntypes)
{
  vector<int > sel_type;
  for (int ii = 0; ii < ntypes; ++ii){
    sel_type.push_back(ii);
  }
  select_by_type(fwd_map, bkw_map, nghost_real, dcoord_, datype_, nghost, sel_type);
}

void
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

void
shuffle_nlist (InternalNeighborList & list, 
	       const NNPAtomMap<VALUETYPE> & map)
{
  const vector<int> & fwd_map = map.get_fwd_map();
  shuffle_nlist(list, fwd_map);
}

void
shuffle_nlist (InternalNeighborList & list, 
	       const vector<int> & fwd_map)
{
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

void
shuffle_nlist_exclude_empty (InternalNeighborList & list, 
			     const vector<int> & fwd_map)
{
  int old_nloc = fwd_map.size();
  shuffle_nlist(list, fwd_map);
  vector<int> new_ilist, new_jrange, new_jlist, new_icount;
  new_ilist.reserve(list.ilist.size());
  new_icount.reserve(list.ilist.size());
  new_jrange.reserve(list.jrange.size());
  new_jlist.reserve(list.jlist.size());
  for(int ii = 0; ii < list.ilist.size(); ++ii){
    if(list.ilist[ii] >= 0){
      new_ilist.push_back(list.ilist[ii]);
    }
  }
  new_jrange.resize(new_ilist.size()+1);
  new_jrange[0] = 0;
  int ci = 0;
  for(int ii = 0; ii < list.ilist.size(); ++ii){
    if (list.ilist[ii] < 0) continue;
    int js = list.jrange[ii];
    int je = list.jrange[ii+1];
    int cc = 0;
    for (int jj = js; jj < je; ++jj){
      if (list.jlist[jj] >= 0) {
	new_jlist.push_back(list.jlist[jj]);
	cc++;
      }      
    }
    new_jrange[ci+1] = new_jrange[ci] + cc;
    ci ++;
  }
  list.ilist = new_ilist;
  list.jrange = new_jrange;
  list.jlist = new_jlist;
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

string
name_prefix(const string & scope)
{
  string prefix = "";
  if (scope != ""){
    prefix = scope + "/";
  }
  return prefix;
}

int
session_input_tensors (std::vector<std::pair<string, Tensor>> & input_tensors,
		       const vector<VALUETYPE> &	dcoord_,
		       const int &			ntypes,
		       const vector<int> &		datype_,
		       const vector<VALUETYPE> &	dbox, 
		       const VALUETYPE &		cell_size,
		       const vector<VALUETYPE> &	fparam_,
		       const vector<VALUETYPE> &	aparam_,
		       const NNPAtomMap<VALUETYPE>&	nnpmap,
		       const int			nghost, 
		       const string			scope)
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
    for (int jj = 0; jj < aparam_.size(); ++jj){
      aparam(ii, jj) = aparam_[jj];
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

  string prefix = "";
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
session_input_tensors (std::vector<std::pair<string, Tensor>> & input_tensors,
		       const vector<VALUETYPE> &	dcoord_,
		       const int &			ntypes,
		       const vector<int> &		datype_,
		       const vector<VALUETYPE> &	dbox,		    
		       InternalNeighborList &		dlist, 
		       const vector<VALUETYPE> &	fparam_,
		       const vector<VALUETYPE> &	aparam_,
		       const NNPAtomMap<VALUETYPE>&	nnpmap,
		       const int			nghost,
		       const string			scope)
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
    for (int jj = 0; jj < aparam_.size(); ++jj){
      aparam(ii, jj) = aparam_[jj];
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

  string prefix = "";
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

int
session_input_tensors (
    vector<std::pair<string, Tensor>>   &   input_tensors,
    const vector<VALUETYPE>             &	  dcoord_,
    const int                           &   ntypes,
    const vector<int>                   &	  datype_,
    const vector<VALUETYPE>             &	  dbox,
    const int                           *   ilist, 
    const int                           *   jrange,
    const int                           *   jlist,
    const vector<VALUETYPE>	            &   fparam_,
    const vector<VALUETYPE>	            &   aparam_,
    const NNPAtomMap<VALUETYPE>         &   nnpmap,
    const int			                      &   nghost)
{
    assert (dbox.size() == 9);

    int nframes = 1;
    int nall = dcoord_.size() / 3;
    int nloc = nall - nghost;
    assert (nall == datype_.size());

    vector<int > datype = nnpmap.get_type();
    vector<int > type_count (ntypes, 0);
    for (unsigned ii = 0; ii < datype.size(); ++ii) {
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
    TensorShape mesh_shape;
    mesh_shape.AddDim (16);
    TensorShape natoms_shape;
    natoms_shape.AddDim (2 + ntypes);
    TensorShape fparam_shape;
    fparam_shape.AddDim (nframes);
    fparam_shape.AddDim (fparam_.size());
    TensorShape aparam_shape ;
    aparam_shape.AddDim (nframes);
    aparam_shape.AddDim (aparam_.size());

    #ifdef HIGH_PREC
        Tensor coord_tensor	(DT_DOUBLE, coord_shape);
        Tensor box_tensor	(DT_DOUBLE, box_shape);
        Tensor fparam_tensor(DT_DOUBLE, fparam_shape);
        Tensor aparam_tensor(DT_DOUBLE, fparam_shape);
    #else
        Tensor coord_tensor	(DT_FLOAT, coord_shape);
        Tensor box_tensor	(DT_FLOAT, box_shape);
        Tensor fparam_tensor(DT_FLOAT, fparam_shape);
        Tensor aparam_tensor(DT_FLOAT, fparam_shape);
    #endif
    Tensor type_tensor	(DT_INT32, type_shape);
    Tensor mesh_tensor	(DT_INT32, mesh_shape);
    Tensor natoms_tensor(DT_INT32, natoms_shape);

    auto coord = coord_tensor.matrix<VALUETYPE> ();
    auto type = type_tensor.matrix<int> ();
    auto box = box_tensor.matrix<VALUETYPE> ();
    auto mesh = mesh_tensor.flat<int> ();
    auto natoms = natoms_tensor.flat<int> ();
    auto fparam = fparam_tensor.matrix<VALUETYPE> ();
    auto aparam = aparam_tensor.matrix<VALUETYPE> ();

    vector<VALUETYPE> dcoord (dcoord_);
    nnpmap.forward (dcoord.begin(), dcoord_.begin(), 3);

    for (int ii = 0; ii < nframes; ++ii) {
        for (int jj = 0; jj < nall * 3; ++jj) {
            coord(ii, jj) = dcoord[jj];
        }
        for (int jj = 0; jj < 9; ++jj) {
            box(ii, jj) = dbox[jj];
        }
        for (int jj = 0; jj < nall; ++jj) {
            type(ii, jj) = datype[jj];
        }
        for (int jj = 0; jj < fparam_.size(); ++jj) {
            fparam(ii, jj) = fparam_[jj];
        }
        for (int jj = 0; jj < aparam_.size(); ++jj) {
            aparam(ii, jj) = aparam_[jj];
        }
    }
    
    for (int ii = 0; ii < 16; ++ii) mesh(ii) = 0;
    
    mesh (0) = sizeof(int *) / sizeof(int);
    assert (mesh(0) * sizeof(int) == sizeof(int *));
    const int & stride = mesh(0);
    // mesh (1) = dlist.ilist.size();
    mesh (1) = nloc;
    assert (mesh(1) == nloc);
    assert (stride <= 4);
    memcpy (&mesh(4), &(ilist), sizeof(int *));
    memcpy (&mesh(8), &(jrange), sizeof(int *));
    memcpy (&mesh(12), &(jlist), sizeof(int *));

    natoms (0) = nloc;
    natoms (1) = nall;
    for (int ii = 0; ii < ntypes; ++ii) natoms(ii+2) = type_count[ii];
    
    input_tensors = {
        {"t_coord",	coord_tensor}, 
        {"t_type",	type_tensor},
        {"t_box",		box_tensor},
        {"t_mesh",	mesh_tensor},
        {"t_natoms",	natoms_tensor},
    };  
    if (fparam_.size() > 0) {
        input_tensors.push_back({"t_fparam", fparam_tensor});
    }
    if (aparam_.size() > 0) {
        input_tensors.push_back({"t_aparam", aparam_tensor});
    }
    return nloc;
}
