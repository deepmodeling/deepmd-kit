#include "common.h"
#include "NNPAtomMap.h"
#include "SimulationRegion.h"

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
  std::cout << "in session_input_tensors 1 -------------" << endl;
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
           const int      ago,
		       const string			scope)
{
  std::cout << "in session_input_tensors 2 -------------" << endl;

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
  
  const int stride = sizeof(int *) / sizeof(int);
  assert (stride * sizeof(int) == sizeof(int *));
  assert (stride <= 4);
  mesh (0) = ago;
  mesh (1) = dlist.ilist.size();
  mesh (2) = dlist.jrange.size();
  mesh (3) = dlist.jlist.size();
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

  std::cout << "in session_input_tensors 3 -------------" << endl;

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

    std::cout << "in session_input_tensors 4 -------------" << endl;

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
