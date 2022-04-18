#include "paddle_common.h"
#include "AtomMap.h"
#include "device.h"

template <typename T>
void SetInputs(const std::vector<T>& param_vec, const std::vector<int> param_shape, const std::string& input_name,
                     const std::shared_ptr<paddle_infer::Predictor>& predictor) {
    auto tensor = predictor->GetInputHandle(input_name);
    tensor->Reshape(param_shape);
    tensor->CopyFromCpu(param_vec.data());
}

int deepmd::get_math_lib_num_threads()
{
  int num_nthreads = 1; // By default use 1 thread
  const char* str_num_nthreads = std::getenv("OMP_NUM_THREADS");
  if (str_num_nthreads && std::string(str_num_nthreads)!=std::string("") && atoi(str_num_nthreads) >= 0) {
    num_nthreads = atoi(str_num_nthreads);
  }
  return num_nthreads;
}

int deepmd::paddle_input_tensors (
    std::shared_ptr<paddle_infer::Predictor> predictor,
    const std::vector<deepmd::VALUETYPE> &	dcoord_,
    const int &					ntypes,
    const std::vector<int> &			datype_,
    const std::vector<deepmd::VALUETYPE> &	dbox, 
    const deepmd::VALUETYPE &			cell_size,
    const std::vector<deepmd::VALUETYPE> &	fparam_,
    const std::vector<deepmd::VALUETYPE> &	aparam_,
    const deepmd::AtomMap<deepmd::VALUETYPE>&	atommap){

  // Deal with input
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

  // Calculate params shape
  std::vector<int> coord_shape = {nframes, nall * 3};
  std::vector<int> type_shape = {nframes, nall};
  std::vector<int> natoms_shape = {2 + ntypes};
  std::vector<int> box_shape = {nframes, 9};
  std::vector<int> mesh_shape;
  if (b_pbc==true){
    mesh_shape.push_back(6);
  } else{
    mesh_shape.push_back(0);
  }
  assert (mesh_shape.size() == 1);
  assert (natoms_shape.size() == 1);
  std::vector<int> fparam_shape = {nframes, static_cast<int>(fparam_.size())};
  std::vector<int> aparam_shape = {nframes, static_cast<int>(aparam_.size())};

  std::vector<deepmd::VALUETYPE> dcoord (dcoord_);
  atommap.forward(dcoord.begin(), dcoord_.begin(), 3);
  std::vector<deepmd::VALUETYPE> coord(nframes * nall * 3, 0.0);
  std::vector<deepmd::VALUETYPE> box(nframes * 9, 0.0);
  std::vector<int> type(nframes * nall, 0), mesh(mesh_shape[0]), natoms(natoms_shape[0]);
  
  for (int i = 0; i < nframes; i++){
    for (int j = 0; j < dcoord.size(); j++){
      coord[i * nframes + j] = dcoord[j];
    }
  
    for (int j = 0; j < dbox.size(); j++){
      if (b_pbc){
        box[i * nframes + j] = dbox[j];
      }else{
        box[i * nframes + j] = 0.0;
      }
    }

    for (int j = 0; j < nall; j++){
      type[i * nframes + j] = datype[j];
    }

    // for (int jj = 0; jj < fparam_.size(); ++jj){
    //   fparam(ii, jj) = fparam_[jj];
    // }
    // for (int jj = 0; jj < aparam_.size(); ++jj){
    //   aparam(ii, jj) = aparam_[jj];
    // }
  }

  if (b_pbc){
    fill(mesh.begin(), mesh.end(), 0);
  }
  natoms[0] = nloc;
  natoms[1] = nall;
  for (int k = 0; k < ntypes; ++k) natoms[k+2] = type_count[k];

  auto input_names = predictor->GetInputNames();
  SetInputs<deepmd::VALUETYPE>(coord, coord_shape, input_names[0], predictor);
  SetInputs<int>(type, type_shape, input_names[1], predictor);
  SetInputs<int>(natoms, natoms_shape, input_names[2], predictor);
  SetInputs<deepmd::VALUETYPE>(box, box_shape, input_names[3], predictor);
  SetInputs<int>(mesh, mesh_shape, input_names[4], predictor);
  return nloc;
}

int deepmd::paddle_input_tensors (
    std::shared_ptr<paddle_infer::Predictor> predictor,
    const std::vector<deepmd::VALUETYPE> &	dcoord_,
    const int &					ntypes,
    const std::vector<int> &			datype_,
    const std::vector<deepmd::VALUETYPE> &	dbox,		    
    deepmd::InputNlist &				dlist, 
    const std::vector<deepmd::VALUETYPE> &	fparam_,
    const std::vector<deepmd::VALUETYPE> &	aparam_,
    const deepmd::AtomMap<deepmd::VALUETYPE>&	atommap,
    const int					nghost,
    const int					ago)
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
  // Calculate params shape
  std::vector<int> coord_shape = {nframes, nall * 3};
  std::vector<int> type_shape = {nframes, nall};
  std::vector<int> box_shape = {nframes, 9};
  std::vector<int> mesh_shape = {16}; // mesh_shape.AddDim (16);
  std::vector<int> natoms_shape = {2 + ntypes};
  std::vector<int> fparam_shape = {nframes, static_cast<int>(fparam_.size())};
  std::vector<int> aparam_shape = {nframes, static_cast<int>(aparam_.size())};

  std::vector<deepmd::VALUETYPE> dcoord (dcoord_);
  atommap.forward (dcoord.begin(), dcoord_.begin(), 3);

  std::vector<deepmd::VALUETYPE> coord(nframes * nall * 3, 0.0), box(nframes * 9, 0.0);
  std::vector<int> type(nframes * nall, 0), mesh(mesh_shape[0], 0), natoms(natoms_shape[0], 0);

  for (int i = 0; i < nframes; i++){
    for (int j = 0; j < nall * 3; ++j){
      coord[i * nframes + j]= dcoord[j];
    }
    for (int j = 0; j < 9; ++j){
      box[i * nframes + j] = dbox[j];
    }
    for (int j = 0; j < nall; ++j){
      type[i * nframes + j] = datype[j];
    }
    // for (int j = 0; j < fparam_.size(); ++j){
    //   fparam[i * nframes + j] = fparam_[j];
    // }
    // for (int j = 0; j < aparam_.size(); ++j){
    //   aparam[i * nframes + j] = aparam_[j];
    // }
  }
  for (int i = 0; i < 16; ++i) mesh[i] = 0;
  
  const int stride = sizeof(int *) / sizeof(int);
  assert (stride * sizeof(int) == sizeof(int *));
  assert (stride <= 4);
  mesh[0] = ago;
  mesh[1] = dlist.inum;
  mesh[2] = 0;
  mesh[3] = 0;
  memcpy(&mesh[4],  &(dlist.ilist), sizeof(int *));
  memcpy(&mesh[8],  &(dlist.numneigh), sizeof(int *));
  memcpy(&mesh[12], &(dlist.firstneigh), sizeof(int **));

  natoms[0] = nloc;
  natoms[1] = nall;
  for (int i = 0; i < ntypes; ++i) 
    natoms[i+2] = type_count[i];

  auto input_names = predictor->GetInputNames();
  SetInputs<float>(coord, coord_shape, input_names[0], predictor);
  SetInputs<int>(type, type_shape, input_names[1], predictor);
  SetInputs<int>(natoms, natoms_shape, input_names[2], predictor);
  SetInputs<float>(box, box_shape, input_names[3], predictor);
  SetInputs<int>(mesh, mesh_shape, input_names[4], predictor);
  return nloc;
}
