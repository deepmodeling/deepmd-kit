#include "common.h"
#include "AtomMap.h"
#include "device.h"
#include <dlfcn.h>
#include <fcntl.h>
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#if (defined __ARM_ARCH) || (defined PLATFORM_AARCH64)
#include <arm_neon.h>
#endif

using namespace tensorflow;

static std::vector<std::string>
split(const std::string &input_, 
      const std::string &delimiter)
{
  std::string input = input_;
  size_t pos = 0;
  std::vector<std::string> res;
  while ((pos = input.find(delimiter)) != std::string::npos) {
    res.push_back(input.substr(0, pos));
    input.erase(0, pos + delimiter.length());
  }
  res.push_back(input);
  return res;
}

bool
deepmd::
model_compatable(
    std::string & model_version)
{
  std::vector<std::string> words_mv = split(model_version, ".");
  std::vector<std::string> words_gmv = split(global_model_version, ".");
  if(words_mv.size() != 2){
    throw deepmd::deepmd_exception("invalid graph model version string " + model_version);
  }
  if(words_gmv.size() != 2){
    throw deepmd::deepmd_exception("invalid supported model version string " + global_model_version);
  }
  int model_version_major = atoi(words_mv[0].c_str());
  int model_version_minor = atoi(words_mv[1].c_str());
  int MODEL_VERSION_MAJOR = atoi(words_gmv[0].c_str());
  int MODEL_VERSION_MINOR = atoi(words_gmv[1].c_str());
  if(model_version_major != MODEL_VERSION_MAJOR ||
     model_version_minor >  MODEL_VERSION_MINOR){
    return false;
  }
  else{
    return true;
  }
}

void 
deepmd::
select_by_type(std::vector<int> & fwd_map,
	       std::vector<int> & bkw_map,
	       int & nghost_real, 
	       const std::vector<deepmd::VALUETYPE> & dcoord_, 
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
deepmd::
select_real_atoms(std::vector<int> & fwd_map,
		  std::vector<int> & bkw_map,
		  int & nghost_real,
		  const std::vector<deepmd::VALUETYPE> & dcoord_, 
		  const std::vector<int> & datype_,
		  const int & nghost,
		  const int & ntypes)
{
  std::vector<int > sel_type;
  for (int ii = 0; ii < ntypes; ++ii){
    sel_type.push_back(ii);
  }
  deepmd::select_by_type(fwd_map, bkw_map, nghost_real, dcoord_, datype_, nghost, sel_type);
}


void
deepmd::NeighborListData::
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
deepmd::NeighborListData::
shuffle(const AtomMap<deepmd::VALUETYPE> & map)
{
  const std::vector<int> & fwd_map = map.get_fwd_map();
  shuffle(fwd_map);
}

void
deepmd::NeighborListData::
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
deepmd::NeighborListData::
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
deepmd::NeighborListData::
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
deepmd::
check_status(const tensorflow::Status& status) {
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    throw deepmd::tf_exception(status.ToString());
  }
}

void
throw_env_not_set_warning(std::string env_name)
{
  std::cerr << "DeePMD-kit WARNING: Environmental variable " << env_name << " is not set. "
    << "Tune " << env_name << " for the best performance."
    << std::endl;
}

void
deepmd::
get_env_nthreads(int & num_intra_nthreads,
		 int & num_inter_nthreads)
{
  num_intra_nthreads = 0;
  num_inter_nthreads = 0;
  const char* env_intra_nthreads = std::getenv("TF_INTRA_OP_PARALLELISM_THREADS");
  const char* env_inter_nthreads = std::getenv("TF_INTER_OP_PARALLELISM_THREADS");
  const char* env_omp_nthreads = std::getenv("OMP_NUM_THREADS");
  if (env_intra_nthreads && 
      std::string(env_intra_nthreads) != std::string("") && 
      atoi(env_intra_nthreads) >= 0
      ) {
    num_intra_nthreads = atoi(env_intra_nthreads);
  } else {
    throw_env_not_set_warning("TF_INTRA_OP_PARALLELISM_THREADS");
  }
  if (env_inter_nthreads && 
      std::string(env_inter_nthreads) != std::string("") &&
      atoi(env_inter_nthreads) >= 0
      ) {
    num_inter_nthreads = atoi(env_inter_nthreads);
  } else {
    throw_env_not_set_warning("TF_INTER_OP_PARALLELISM_THREADS");
  }
  if (!(env_omp_nthreads && 
      std::string(env_omp_nthreads) != std::string("") &&
      atoi(env_omp_nthreads) >= 0
      )) {
    throw_env_not_set_warning("OMP_NUM_THREADS");
  }
}

void
deepmd::
load_op_library()
{
  tensorflow::Env* env = tensorflow::Env::Default();
  std::string dso_path = env->FormatLibraryFileName("deepmd_op", "");
  void* dso_handle = dlopen(dso_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!dso_handle) {
    throw deepmd::deepmd_exception(dso_path + " is not found! You can add the library directory to LD_LIBRARY_PATH");
  }
}

std::string
deepmd::
name_prefix(const std::string & scope)
{
  std::string prefix = "";
  if (scope != ""){
    prefix = scope + "/";
  }
  return prefix;
}

int
deepmd::
session_input_tensors (
    std::vector<std::pair<std::string, Tensor>> & input_tensors,
    const std::vector<deepmd::VALUETYPE> &	dcoord_,
    const int &					ntypes,
    const std::vector<int> &			datype_,
    const std::vector<deepmd::VALUETYPE> &	dbox, 
    const deepmd::VALUETYPE &			cell_size,
    const std::vector<deepmd::VALUETYPE> &	fparam_,
    const std::vector<deepmd::VALUETYPE> &	aparam_,
    const deepmd::AtomMap<deepmd::VALUETYPE>&	atommap,
    const std::string				scope)
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

  auto coord = coord_tensor.matrix<deepmd::VALUETYPE> ();
  auto type = type_tensor.matrix<int> ();
  auto box = box_tensor.matrix<deepmd::VALUETYPE> ();
  auto mesh = mesh_tensor.flat<int> ();
  auto natoms = natoms_tensor.flat<int> ();  
  auto fparam = fparam_tensor.matrix<deepmd::VALUETYPE> ();
  auto aparam = aparam_tensor.matrix<deepmd::VALUETYPE> ();

  std::vector<deepmd::VALUETYPE> dcoord (dcoord_);
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
deepmd::
session_input_tensors (
    std::vector<std::pair<std::string, Tensor>> & input_tensors,
    const std::vector<deepmd::VALUETYPE> &	dcoord_,
    const int &					ntypes,
    const std::vector<int> &			datype_,
    const std::vector<deepmd::VALUETYPE> &	dbox,		    
    InputNlist &				dlist, 
    const std::vector<deepmd::VALUETYPE> &	fparam_,
    const std::vector<deepmd::VALUETYPE> &	aparam_,
    const deepmd::AtomMap<deepmd::VALUETYPE>&	atommap,
    const int					nghost,
    const int					ago,
    const std::string				scope)
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

  auto coord = coord_tensor.matrix<deepmd::VALUETYPE> ();
  auto type = type_tensor.matrix<int> ();
  auto box = box_tensor.matrix<deepmd::VALUETYPE> ();
  auto mesh = mesh_tensor.flat<int> ();
  auto natoms = natoms_tensor.flat<int> ();
  auto fparam = fparam_tensor.matrix<deepmd::VALUETYPE> ();
  auto aparam = aparam_tensor.matrix<deepmd::VALUETYPE> ();

  std::vector<deepmd::VALUETYPE> dcoord (dcoord_);
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

#if HUAWEI_ASCEND
int
deepmd::
session_input_tensors (
    std::vector<std::pair<std::string, Tensor>> & input_tensors,
    const std::vector<deepmd::VALUETYPE> &	dcoord_,
    const int &					ntypes,
    const std::vector<int> &			datype_,
    const std::vector<deepmd::VALUETYPE> &	dbox,		    
    InputNlist &				dlist, 
    const std::vector<deepmd::VALUETYPE> &	fparam_,
    const std::vector<deepmd::VALUETYPE> &	aparam_,
    const deepmd::AtomMap<deepmd::VALUETYPE>&	atommap,
    const int					nghost,
    const int					ago,
    const std::vector<int > &		natoms_padding,
    const std::string				scope)
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

  TensorShape box_shape ;
  box_shape.AddDim (nframes);
  box_shape.AddDim (9);
  TensorShape mesh_shape ;
  mesh_shape.AddDim (nloc * 1026 + 1);
  TensorShape natoms_shape ;
  natoms_shape.AddDim (2 + ntypes);
  TensorShape fparam_shape ;
  fparam_shape.AddDim (nframes);
  fparam_shape.AddDim (fparam_.size());
  TensorShape aparam_shape ;
  aparam_shape.AddDim (nframes);
  aparam_shape.AddDim (aparam_.size());
  
#ifdef HIGH_PREC
  Tensor box_tensor	(DT_DOUBLE, box_shape);
  Tensor fparam_tensor  (DT_DOUBLE, fparam_shape);
  Tensor aparam_tensor  (DT_DOUBLE, aparam_shape);
#else
  Tensor box_tensor	(DT_FLOAT, box_shape);
  Tensor fparam_tensor  (DT_FLOAT, fparam_shape);
  Tensor aparam_tensor  (DT_FLOAT, aparam_shape);
#endif
  Tensor mesh_tensor	(DT_INT32, mesh_shape);
  Tensor natoms_tensor	(DT_INT32, natoms_shape);

  auto box = box_tensor.matrix<deepmd::VALUETYPE> ();
  auto natoms = natoms_tensor.flat<int> ();
  auto fparam = fparam_tensor.matrix<deepmd::VALUETYPE> ();
  auto aparam = aparam_tensor.matrix<deepmd::VALUETYPE> ();

  std::vector<deepmd::VALUETYPE> dcoord (dcoord_);
  atommap.forward (dcoord.begin(), dcoord_.begin(), 3);

  for (int ii = 0; ii < nframes; ++ii){
    for (int jj = 0; jj < 9; ++jj){
      box(ii, jj) = dbox[jj];
    }
    for (int jj = 0; jj < fparam_.size(); ++jj){
      fparam(ii, jj) = fparam_[jj];
    }
    for (int jj = 0; jj < aparam_.size(); ++jj){
      aparam(ii, jj) = aparam_[jj];
    }
  }

  int nall_padding = natoms_padding[1];
  int nloc_padding = natoms_padding[0];
  std::vector<int> type_count_padding;
  type_count_padding.assign(natoms_padding.begin()+2, natoms_padding.end());
  TensorShape coord_padding_shape ;
  coord_padding_shape.AddDim (nframes);
  coord_padding_shape.AddDim (nall_padding * 3);
  TensorShape type_padding_shape ;
  type_padding_shape.AddDim (nframes);
  type_padding_shape.AddDim (nall_padding);
  Tensor coord_padding_tensor	(DT_FLOAT, coord_padding_shape);
  Tensor type_padding_tensor	(DT_INT32, type_padding_shape);
  
  // padding coord and type
  auto coord_padding = coord_padding_tensor.flat<float> ();
  auto type_padding = type_padding_tensor.flat<int> ();
#if (defined __ARM_ARCH) || (defined PLATFORM_AARCH64)
  float32x4_t temp_coord = vdupq_n_f32(0.0f);
  int32x4_t temp_type = vdupq_n_s32(-1);
  #pragma omp parallel for
  for (uint jj = 0; jj < nall_padding - 4; jj += 4) {
      vst1q_s32(type_padding.data() + jj, temp_type);
      vst1q_f32(coord_padding.data() + jj*3, temp_coord);
      vst1q_f32(coord_padding.data() + jj*3 + 1, temp_coord);
      vst1q_f32(coord_padding.data() + jj*3 + 2, temp_coord);
  }
  int curr_type = (nall_padding - 4)/4;
  for (uint jj = curr_type * 4; jj < nall_padding; jj += 1) {
      type_padding.data()[jj] = -1;
      coord_padding.data()[jj*3] = 0.0;
      coord_padding.data()[jj*3 + 1] = 0.0;
      coord_padding.data()[jj*3 + 2] = 0.0;
  }
#else
    #pragma omp parallel for
    for (int jj = 0; jj < nall_padding; ++jj){
        type_padding(jj) = -1;
        coord_padding(jj*3) = 0.0;
        coord_padding(jj*3 + 1) = 0.0;
        coord_padding(jj*3 + 2) = 0.0;
    }
#endif

  auto *coord_padding_p = coord_padding_tensor.flat<float> ().data();
  auto *type_padding_p = type_padding_tensor.flat<int> ().data();
  int offset1 = 0;
  int offset2 = 0;
  for (long int jj = 0; jj < type_count.size(); jj += 1){
    #pragma omp parallel sections
    {
      #pragma omp section
      {
        memcpy (coord_padding_p+offset1*3,  dcoord.data()+offset2*3, type_count[jj]*3*sizeof(float));
      }
      #pragma omp section
      {
        memcpy (type_padding_p+offset1,  datype.data()+offset2, type_count[jj]*sizeof(int));
      }
    }
    offset1 += type_count_padding[jj];
    offset2 += type_count[jj];
  }
  #pragma omp parallel sections
    {
      #pragma omp section
      {
        memcpy (coord_padding_p+offset1*3,  dcoord.data()+offset2*3, (nall-nloc)*3*sizeof(float));
      }
      #pragma omp section
      {
        memcpy (type_padding_p+offset1,  datype.data()+offset2, (nall-nloc)*sizeof(int));
      }
    }

  const int stride = sizeof(int *) / sizeof(int);
  assert (stride * sizeof(int) == sizeof(int *));
  assert (stride <= 4);

  std::string prefix = "";
  if (scope != ""){
    prefix = scope + "/";
  }
  input_tensors = {
    {prefix+"t_coord",	coord_padding_tensor},
    {prefix+"t_type",	type_padding_tensor},
    {prefix+"t_box",	box_tensor},
  };
  if (ago == 0) {
      // padding mesh data and add offset
      int _max_nbor_size = deepmd::max_numneigh(dlist);
      int neigh_len;
      if (_max_nbor_size < 1024) {
          neigh_len = 1024;
      }
      else if (_max_nbor_size < 2048) {
          neigh_len = 2048;
      }
      else {
          neigh_len = 4096;
      }
      // Current max neigh_len is set as 1024 in Assign op
      assert (neigh_len == 1024);
      // generate offset mapping
      std::vector<int > offset_mapping (nall, 0);
      int offset_padding = 0;
      int offset = 0;
      for (unsigned jj = 0; jj < type_count.size(); jj += 1){
          for (int kk = 0; kk < type_count[jj]; kk += 1){
              offset_mapping[offset + kk] = offset_padding + kk;
          }
          offset += type_count[jj];
          offset_padding += type_count_padding[jj];
      }
      assert (nloc == offset);
      for (int kk = 0; kk < (nall-nloc); kk += 1){
          offset_mapping[offset + kk] = offset_padding + kk;
      }
      TensorShape mesh_padding_shape;
      int mesh_dim = nloc_padding * (neigh_len + 2 )+ 1;
      mesh_padding_shape.AddDim (mesh_dim);
      Tensor mesh_padding_tensor	(DT_INT32, mesh_padding_shape);
      auto mesh_padding = mesh_padding_tensor.flat<int> ();
#if (defined __ARM_ARCH) || (defined PLATFORM_AARCH64)
      int32x4_t temp = vdupq_n_s32(-1);
      #pragma omp parallel for
      for (uint i = 0; i < mesh_dim - 4; i += 4) {
          vst1q_s32(mesh_padding.data() + i, temp);
      }
      int curr_mesh = mesh_dim / 4;
      #pragma omp parallel for
      for (uint i = curr_mesh * 4; i < mesh_dim; i += 1) {
          mesh_padding.data()[i] = -1;
      }
#else
      #pragma omp parallel for
      for (int atom_i = 0; atom_i < mesh_dim; ++atom_i) {
          mesh_padding(atom_i) = -1;
      }
#endif
      mesh_padding(0) = nloc_padding;
      #pragma omp parallel for
      for (int atom_i = 0; atom_i < nloc_padding; atom_i++) {
          mesh_padding(atom_i + 1) = atom_i;
      }
      #pragma omp parallel for
      for (int atom_i = 0; atom_i < dlist.inum; atom_i++) {
          int offset_atom = offset_mapping[dlist.ilist[atom_i]];
          mesh_padding(nloc_padding + offset_atom + 1) = dlist.numneigh[atom_i];
          int offset_mesh = 2*nloc_padding + offset_atom*neigh_len;
          for (int nei_i = 0; nei_i < dlist.numneigh[atom_i]; ++nei_i) {
              int offset_idx = offset_mapping[dlist.firstneigh[atom_i][nei_i]];
              mesh_padding(offset_mesh + nei_i + 1) = offset_idx;
          }
      }
      input_tensors.push_back({prefix+"t_mesh",	mesh_padding_tensor});
  }
  if (fparam_.size() > 0) {
    input_tensors.push_back({prefix+"t_fparam", fparam_tensor});
  }
  if (aparam_.size() > 0) {
    input_tensors.push_back({prefix+"t_aparam", aparam_tensor});
  }
  return nloc;
}
#endif //HUAWEI_ASCEND

template<typename VT>
VT
deepmd::
session_get_scalar(Session* session, const std::string name_, const std::string scope) 
{
  std::string name = name_;
  if (scope != "") {
    name = scope + "/" + name;
  }
  std::vector<Tensor> output_tensors;
  deepmd::check_status (session->Run(std::vector<std::pair<std::string, Tensor>> ({}), 
			    {name.c_str()}, 
			    {}, 
			    &output_tensors));
  Tensor output_rc = output_tensors[0];
  auto orc = output_rc.flat <VT> ();
  return orc(0);
}

template<typename VT>
void
deepmd::
session_get_vector(std::vector<VT> & o_vec, Session* session, const std::string name_, const std::string scope) 
{
  std::string name = name_;
  if (scope != "") {
    name = scope + "/" + name;
  }
  std::vector<Tensor> output_tensors;
  deepmd::check_status (session->Run(std::vector<std::pair<std::string, Tensor>> ({}), 
			    {name.c_str()}, 
			    {}, 
			    &output_tensors));
  Tensor output_rc = output_tensors[0];
  assert(1 == output_rc.shape().dims());
  int dof = output_rc.shape().dim_size(0);
  o_vec.resize(dof);
  auto orc = output_rc.flat <VT> ();
  for (int ii = 0; ii < dof; ++ii){
    o_vec[ii] = orc(ii);
  }  
}


template<typename VT>
void 
deepmd::
select_map(std::vector<VT> & out,
	   const std::vector<VT > & in,
	   const std::vector<int > & idx_map, 
	   const int & stride)
{
#ifdef DEBUG
  assert(in.size() / stride * stride == in.size()), "in size should be multiples of stride"
#endif
  for (int ii = 0; ii < in.size() / stride; ++ii){
#ifdef DEBUG
    assert(ii < idx_map.size()), "idx goes over the idx map size";
    assert(idx_map[ii] < out.size()), "mappped idx goes over the out size";
#endif
    if (idx_map[ii] >= 0) {
      int to_ii = idx_map[ii];
      for (int dd = 0; dd < stride; ++dd){
	out[to_ii * stride + dd] = in[ii * stride + dd];
      }
    }
  }
}

template<typename VT>
void 
deepmd::
select_map(typename std::vector<VT >::iterator out,
	   const typename std::vector<VT >::const_iterator in, 
	   const std::vector<int > & idx_map, 
	   const int & stride)
{
  for (int ii = 0; ii < idx_map.size(); ++ii){
    if (idx_map[ii] >= 0) {
      int to_ii = idx_map[ii];
      for (int dd = 0; dd < stride; ++dd){
	*(out + to_ii * stride + dd) = *(in + ii * stride + dd);
      }
    }
  }
}

// sel_map(_,_,fwd_map,_) == sel_map_inv(_,_,bkw_map,_)
template<typename VT>
void 
deepmd::
select_map_inv(std::vector<VT> & out,
	   const std::vector<VT > & in,
	   const std::vector<int > & idx_map, 
	   const int & stride)
{
#ifdef DEBUG
  assert(in.size() / stride * stride == in.size()), "in size should be multiples of stride"
#endif
  for (int ii = 0; ii < out.size() / stride; ++ii){
#ifdef DEBUG
    assert(ii < idx_map.size()), "idx goes over the idx map size";
    assert(idx_map[ii] < in.size()), "from idx goes over the in size";
#endif
    if (idx_map[ii] >= 0) {
      int from_ii = idx_map[ii];
      for (int dd = 0; dd < stride; ++dd){
	out[ii * stride + dd] = in[from_ii * stride + dd];
      }
    }
  }
}

template<typename VT>
void 
deepmd::
select_map_inv(typename std::vector<VT >::iterator out,
	   const typename std::vector<VT >::const_iterator in, 
	   const std::vector<int > & idx_map, 
	   const int & stride)
{
  for (int ii = 0; ii < idx_map.size(); ++ii){
    if (idx_map[ii] >= 0) {
      int from_ii = idx_map[ii];
      for (int dd = 0; dd < stride; ++dd){
	*(out + ii * stride + dd) = *(in + from_ii * stride + dd);
      }
    }
  }
}


template
int
deepmd::
session_get_scalar<int>(Session*, const std::string, const std::string);

template
void
deepmd::
session_get_vector<int>(std::vector<int> &, Session*, const std::string, const std::string);

template
void 
deepmd::
select_map<int>(
    std::vector<int> & out,
    const std::vector<int > & in,
    const std::vector<int > & idx_map, 
    const int & stride);

template
void 
deepmd::
select_map<int>(
    typename std::vector<int >::iterator out,
    const typename std::vector<int >::const_iterator in, 
    const std::vector<int > & idx_map, 
    const int & stride);

template
void 
deepmd::
select_map_inv<int>(
    std::vector<int> & out,
    const std::vector<int > & in,
    const std::vector<int > & idx_map, 
    const int & stride);

template
void 
deepmd::
select_map_inv<int>(
    typename std::vector<int >::iterator out,
    const typename std::vector<int >::const_iterator in, 
    const std::vector<int > & idx_map, 
    const int & stride);


template
float
deepmd::
session_get_scalar<float>(Session*, const std::string, const std::string);

template
void
deepmd::
session_get_vector<float>(std::vector<float> &, Session*, const std::string, const std::string);

template
void 
deepmd::
select_map<float>(
    std::vector<float> & out,
    const std::vector<float > & in,
    const std::vector<int > & idx_map, 
    const int & stride);

template
void 
deepmd::
select_map<float>(
    typename std::vector<float >::iterator out,
    const typename std::vector<float >::const_iterator in, 
    const std::vector<int > & idx_map, 
    const int & stride);

template
void 
deepmd::
select_map_inv<float>(
    std::vector<float> & out,
    const std::vector<float > & in,
    const std::vector<int > & idx_map, 
    const int & stride);

template
void 
deepmd::
select_map_inv<float>(
    typename std::vector<float >::iterator out,
    const typename std::vector<float >::const_iterator in, 
    const std::vector<int > & idx_map, 
    const int & stride);


template
double
deepmd::
session_get_scalar<double>(Session*, const std::string, const std::string);

template
void
deepmd::
session_get_vector<double>(std::vector<double> &, Session*, const std::string, const std::string);

template
void 
deepmd::
select_map<double>(
    std::vector<double> & out,
    const std::vector<double > & in,
    const std::vector<int > & idx_map, 
    const int & stride);

template
void 
deepmd::
select_map<double >(
    typename std::vector<double >::iterator out,
    const typename std::vector<double >::const_iterator in, 
    const std::vector<int > & idx_map, 
    const int & stride);

template
void 
deepmd::
select_map_inv<double>(
    std::vector<double> & out,
    const std::vector<double > & in,
    const std::vector<int > & idx_map, 
    const int & stride);

template
void 
deepmd::
select_map_inv<double >(
    typename std::vector<double >::iterator out,
    const typename std::vector<double >::const_iterator in, 
    const std::vector<int > & idx_map, 
    const int & stride);


template
deepmd::STRINGTYPE
deepmd::
session_get_scalar<deepmd::STRINGTYPE>(Session*, const std::string, const std::string);

template
void
deepmd::
session_get_vector<deepmd::STRINGTYPE>(std::vector<deepmd::STRINGTYPE> &, Session*, const std::string, const std::string);

template
void 
deepmd::
select_map<deepmd::STRINGTYPE>(
    std::vector<deepmd::STRINGTYPE> & out,
    const std::vector<deepmd::STRINGTYPE > & in,
    const std::vector<int > & idx_map, 
    const int & stride);

template
void 
deepmd::
select_map<deepmd::STRINGTYPE >(
    typename std::vector<deepmd::STRINGTYPE >::iterator out,
    const typename std::vector<deepmd::STRINGTYPE >::const_iterator in, 
    const std::vector<int > & idx_map, 
    const int & stride);

template
void 
deepmd::
select_map_inv<deepmd::STRINGTYPE>(
    std::vector<deepmd::STRINGTYPE> & out,
    const std::vector<deepmd::STRINGTYPE > & in,
    const std::vector<int > & idx_map, 
    const int & stride);

template
void 
deepmd::
select_map_inv<deepmd::STRINGTYPE >(
    typename std::vector<deepmd::STRINGTYPE >::iterator out,
    const typename std::vector<deepmd::STRINGTYPE >::const_iterator in, 
    const std::vector<int > & idx_map, 
    const int & stride);


void
deepmd::
read_file_to_string(std::string model, std::string & file_content)
{
  deepmd::check_status(tensorflow::ReadFileToString(tensorflow::Env::Default(), model, &file_content));
}


void
deepmd::
convert_pbtxt_to_pb(std::string fn_pb_txt, std::string fn_pb)
{
    int fd = open(fn_pb_txt.c_str(), O_RDONLY);
    tensorflow::protobuf::io::ZeroCopyInputStream* input = new tensorflow::protobuf::io::FileInputStream(fd);
    tensorflow::GraphDef graph_def;
    tensorflow::protobuf::TextFormat::Parse(input, &graph_def);
    delete input;
    std::fstream output(fn_pb, std::ios::out | std::ios::trunc | std::ios::binary);
    graph_def.SerializeToOstream(&output);
}
