#include "NNPInter.h"
#include "NNPAtomMap.h"
#include "SimulationRegion.h"

static
void
checkStatus(const tensorflow::Status& status) {
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(1);
  }
}

NNPInter::
NNPInter ()
    : inited (false)
{
}

NNPInter::
NNPInter (const string & model)
{
  checkStatus (NewSession(SessionOptions(), &session));
  checkStatus (ReadBinaryProto(Env::Default(), model, &graph_def));
  checkStatus (session->Create(graph_def));  
  rcut = get_rcut();
  cell_size = rcut;
  inited = true;
}

void
NNPInter::
init (const string & model)
{
  assert (!inited);
  checkStatus (NewSession(SessionOptions(), &session));
  checkStatus (ReadBinaryProto(Env::Default(), model, &graph_def));
  checkStatus (session->Create(graph_def));  
  rcut = get_rcut();
  cell_size = rcut;
  inited = true;
}

VALUETYPE
NNPInter::
get_rcut () const
{
  std::vector<Tensor> output_tensors;
  checkStatus (session->Run(std::vector<std::pair<string, Tensor>> ({}), 
			    {"t_rcut"}, 
			    {}, 
			    &output_tensors));
  Tensor output_rc = output_tensors[0];
  auto orc = output_rc.flat <VALUETYPE> ();
  return orc(0);
}

void
NNPInter::
compute (VALUETYPE &			dener,
	 vector<VALUETYPE> &		dforce_,
	 vector<VALUETYPE> &		dvirial,
	 const vector<VALUETYPE> &	dcoord_,
	 const vector<int> &		datype_,
	 const vector<VALUETYPE> &	dbox)
{
  assert (dbox.size() == 9);

  int nframes = 1;
  int nloc = dcoord_.size() / 3;
  assert (nloc == datype_.size());

  NNPAtomMap<VALUETYPE> nnpmap (datype_);
  vector<int > datype = nnpmap.get_type();

  int ntypes = datype.back () + 1;
  vector<int > type_count (ntypes, 0);
  for (unsigned ii = 0; ii < datype.size(); ++ii){
    type_count[datype[ii]] ++;
  }

  SimulationRegion<VALUETYPE> region;
  region.reinitBox (&dbox[0]);
  double box_l[3];
  region.toFaceDistance (box_l);
  
  vector<int > ncell (3, 2);
  for (int dd = 0; dd < 3; ++dd){
    ncell[dd] = box_l[dd] / cell_size;
    if (ncell[dd] < 2) ncell[dd] = 2;
  }

  TensorShape coord_shape ;
  coord_shape.AddDim (nframes);
  coord_shape.AddDim (nloc * 3);
  TensorShape type_shape ;
  type_shape.AddDim (nframes);
  type_shape.AddDim (nloc);
  TensorShape box_shape ;
  box_shape.AddDim (nframes);
  box_shape.AddDim (9);
  TensorShape mesh_shape ;
  mesh_shape.AddDim (6);
  TensorShape natoms_shape ;
  natoms_shape.AddDim (2 + ntypes);
  
#ifdef HIGH_PREC
  Tensor coord_tensor	(DT_DOUBLE, coord_shape);
  Tensor type_tensor	(DT_INT32, type_shape);
  Tensor box_tensor	(DT_DOUBLE, box_shape);
  Tensor mesh_tensor	(DT_INT32, mesh_shape);
  Tensor natoms_tensor	(DT_INT32, natoms_shape);
#else
  Tensor coord_tensor	(DT_FLOAT, coord_shape);
  Tensor type_tensor	(DT_INT32, type_shape);
  Tensor box_tensor	(DT_FLOAT, box_shape);
  Tensor mesh_tensor	(DT_INT32, mesh_shape);
  Tensor natoms_tensor	(DT_INT32, natoms_shape);
#endif

  auto coord = coord_tensor.matrix<VALUETYPE> ();
  auto type = type_tensor.matrix<int> ();
  auto box = box_tensor.matrix<VALUETYPE> ();
  auto mesh = mesh_tensor.flat<int> ();
  auto natoms = natoms_tensor.flat<int> ();


  vector<VALUETYPE> dcoord (dcoord_.size());
  nnpmap.forward (dcoord, dcoord_, 3);
  for (int ii = 0; ii < nframes; ++ii){
    for (int jj = 0; jj < nloc * 3; ++jj){
      coord(ii, jj) = dcoord[jj];
    }
    for (int jj = 0; jj < 9; ++jj){
      box(ii, jj) = dbox[jj];
    }
    for (int jj = 0; jj < nloc; ++jj){
      type(ii, jj) = datype[jj];
    }
  }
  mesh (1-1) = 0;
  mesh (2-1) = 0;
  mesh (3-1) = 0;
  mesh (4-1) = ncell[0];
  mesh (5-1) = ncell[1];
  mesh (6-1) = ncell[2];
  natoms (0) = nloc;
  natoms (1) = nloc;
  for (int ii = 0; ii < ntypes; ++ii) natoms(ii+2) = type_count[ii];

  std::vector<std::pair<string, Tensor>> input_tensors = {
    {"t_coord",	coord_tensor}, 
    {"t_type",	type_tensor},
    {"t_box",	box_tensor},
    {"t_mesh",	mesh_tensor},
    {"t_natoms", natoms_tensor},
  };
  std::vector<Tensor> output_tensors;

  checkStatus (session->Run(input_tensors, 
			    {"energy_test", "force_test", "virial_test"}, 
			    {}, 
			    &output_tensors));
  
  Tensor output_e = output_tensors[0];
  Tensor output_f = output_tensors[1];
  Tensor output_v = output_tensors[2];

  auto oe = output_e.flat <VALUETYPE> ();
  auto of = output_f.flat <VALUETYPE> ();
  auto ov = output_v.flat <VALUETYPE> ();

  dener = oe(0);
  vector<VALUETYPE> dforce (3 * nloc);
  dvirial.resize (9);
  for (int ii = 0; ii < nloc * 3; ++ii){
    dforce[ii] = of(ii);
  }
  for (int ii = 0; ii < 9; ++ii){
    dvirial[ii] = ov(ii);
  }
  dforce_.resize (dforce.size());
  nnpmap.backward (dforce_, dforce, 3);
}

