#include "common.h"
#include "Integrator.h"
#include "NNPInter.h"
#include "Statistics.h"

#include "Trajectory.h"
#include "GroFileManager.h"
#include "XyzFileManager.h"
#include "Convert.h"

#include "json.hpp"
using json = nlohmann::json;

#include <fstream>

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else 
typedef float  VALUETYPE;
#endif

void 
print_vec (const vector<VALUETYPE > & vec)
{
  int nloc = vec.size() / 3;
  for (int kk = 0; kk < nloc; ++kk){
    for (int dd = 0; dd < 3; ++dd){
      cout << vec[kk*3+dd] << " \t " ;
    }
    cout << endl;
  }
}

int main(int argc, char * argv[])
{
  UnitManager::set ("metal");

  if (argc == 0) {
    cerr << "usage " << endl;
    cerr << argv[0] << " input_script " << endl;
    return 1;
  }

  ifstream fp (argv[1]);
  json jdata;
  fp >> jdata;
  cout << "# using data base" << endl;
  cout << setw(4) << jdata << endl;

  int nframes = 1;

  vector<VALUETYPE> dcoord, dveloc, dbox, dmass;
  vector<int> dtype;
  vector<int> freez;
  
  // load_raw (dcoord, dtype, dbox);
  // dveloc.resize(dcoord.size(), 0.);
  string conf_format = jdata["conf_format"];
  string conf_file = jdata["conf_file"];
  vector<int > resdindex, atomindex;
  vector<string> resdname, atomname;
  vector<vector<double > > posi, velo, tmp_forc;
  vector<double> boxsize;
  if (conf_format == "gro") {
    GroFileManager::read (conf_file, resdindex, resdname, atomname, atomindex, posi, velo, boxsize);
  }
  else if (conf_format == "xyz"){
    XyzFileManager::read (conf_file, atomname, posi, velo, tmp_forc, boxsize);
    if (velo.size() == 0) {
      for (unsigned ii = 0; ii < posi.size(); ++ii){
	velo.push_back (vector<double > (3, 0.));
      }
    }
    // convert to nanometer
    for (unsigned ii = 0; ii < posi.size(); ++ii){
      for (unsigned dd = 0; dd < 3; ++dd){
	posi[ii][dd] *= .1;
	velo[ii][dd] *= .1;
      }
    }
    for (unsigned dd = 0; dd < 9; ++dd){
      boxsize[dd] *= .1;
    }
    for (unsigned ii = 0; ii < posi.size(); ++ii){
      resdindex.push_back (ii+1);
      atomindex.push_back (ii+1);
    }
    resdname = atomname;
  }
  else {
    cerr << "unknow conf file format: " << conf_format << endl;
    return 1;
  }
  map<string, int> name_type_map = jdata["atom_type"];
  map<string, VALUETYPE> name_mass_map = jdata["atom_mass"];
  map<string, VALUETYPE> name_charge_map;
  if (jdata.find ("atom_charge") == jdata.end()) {
    for (map<string, VALUETYPE>::iterator iter = name_mass_map.begin();
	 iter != name_mass_map.end(); 
	 ++iter ){
      name_charge_map[iter->first] = 0.;
    }
  }
  else {
    map<string, VALUETYPE> name_charge_map_tmp = jdata["atom_charge"];
    name_charge_map = name_charge_map_tmp;
  }
  if (jdata.find ("freeze_atoms") != jdata.end()){
    freez = jdata["freeze_atoms"].get<vector<int> > ();
  }

  // convert but do not sort
  Convert<VALUETYPE> cvt (atomname, name_type_map, name_mass_map, name_charge_map, false);
  cvt.gro2nnp (dcoord, dveloc, dbox, posi, velo, boxsize);
  dtype = cvt.get_type();
  dmass = cvt.get_mass();

  int nloc = dtype.size();
  SimulationRegion<double> region;
  region.reinitBox (&dbox[0]);
  normalize_coord<VALUETYPE> (dcoord, region);

  vector<VALUETYPE > dforce (nloc * 3, 0.);
  vector<VALUETYPE > dae (nloc * 1, 0.);
  vector<VALUETYPE > dav (nloc * 9, 0.);
  vector<VALUETYPE > dvirial (9, 0.0);
  VALUETYPE dener = 0;

  string graph_file = jdata["graph_file"];
  VALUETYPE dt = jdata["dt"];
  int nsteps = jdata["nsteps"];
  int nener = jdata["ener_freq"];
  int nxtc = jdata["xtc_freq"];
  int ntrr = jdata["trr_freq"];
  string ener_file = jdata["ener_file"];
  string xtc_file = jdata["xtc_file"];
  string trr_file = jdata["trr_file"];
  double temperature = jdata["T"];
  double tau_t = jdata["tau_T"];
  long long int seed = 0;
  if (jdata.find ("rand_seed") != jdata.end()) {
    seed = jdata["rand_seed"];
  }
  bool print_f = false;
  if (jdata.find ("print_force") != jdata.end()) {
    print_f = jdata["print_force"];
  }

  Integrator<VALUETYPE> inte;
  ThermostatLangevin<VALUETYPE> thm (temperature, tau_t, seed);
  NNPInter nnp (graph_file);
  
  Statistics<VALUETYPE> st;
  XtcSaver sxtc (xtc_file.c_str(), nloc);
  TrrSaver strr (trr_file.c_str(), nloc);
  
  // compute force at step 0
  nnp.compute (dener, dforce, dvirial, dcoord, dtype, dbox);
  // change virial to gromacs convention
  for (int ii = 0; ii < 9; ++ii) dvirial[ii] *= -0.5;
  st.record (dener, dvirial, dveloc, dmass, region);
  ofstream efout (ener_file);
  ofstream pforce;
  if (print_f) pforce.open ("force.out");
  st.print_head (efout);
  st.print (efout, 0, 0);

  for (int ii = 0; ii < nsteps; ++ii){
    inte.stepVeloc (dveloc, dforce, dmass, 0.5*dt, freez);
    inte.stepCoord (dcoord, dveloc, 0.5*dt);
    thm.stepOU (dveloc, dmass, dt, freez);
    inte.stepCoord (dcoord, dveloc, 0.5*dt);
    normalize_coord<VALUETYPE> (dcoord, region);
    nnp.compute (dener, dforce, dvirial, dae, dav, dcoord, dtype, dbox);
    // change virial to gromacs convention
    for (int ii = 0; ii < 9; ++ii) dvirial[ii] *= -0.5;
    inte.stepVeloc (dveloc, dforce, dmass, 0.5*dt, freez);
    if ((ii + 1) % nener == 0) {
      st.record (dener, dvirial, dveloc, dmass, region);
      st.print (efout, ii+1, (ii+1) * dt);
      efout.flush();
    }
    if (nxtc > 0 && (ii + 1) % nxtc == 0){
      cvt.nnp2gro (posi, velo, boxsize, dcoord, dveloc, dbox);
      sxtc.save (ii+1, (ii+1) * dt, posi, boxsize);
    }
    if (ntrr > 0 && (ii + 1) % ntrr == 0){
      cvt.nnp2gro (posi, velo, boxsize, dcoord, dveloc, dbox);
      strr.save (ii+1, (ii+1) * dt, posi, velo, vector<vector<VALUETYPE> > (), boxsize);
      if (print_f) {
	for (int jj = 0;  jj < dforce.size(); ++jj) {
	  pforce << dforce[jj] << " " ;
	}
	pforce << endl;
      }
    }    
  }
  
  cvt.nnp2gro (posi, velo, boxsize, dcoord, dveloc, dbox);
  GroFileManager::write ("out.gro", resdindex, resdname, atomname, atomindex, posi, velo, boxsize);
  // ofstream oxyz ("out.xyz");
  // oxyz << nloc << endl;
  // oxyz << setprecision(12) ;
  // for (int ii = 0; ii < dbox.size(); ++ii) {
  //   oxyz << dbox[ii] * 1 << " " ;
  // }
  // oxyz << endl;
  // for (int ii = 0; ii < posi.size(); ++ii){
  //   oxyz << atomname[ii] << " \t" ;
  //   for (int dd = 0; dd < 3; ++dd){
  //     oxyz << posi[ii][dd] * 10 << " ";
  //   }
  //   oxyz << endl;
  // }
  
  return 0;
}
