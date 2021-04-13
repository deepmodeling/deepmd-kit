#include <fstream>
#include <cstdint>
#include "sockets.h"
#include "DeepPot.h"
#include "Convert.h"
#include "XyzFileManager.h"
#include "SimulationRegion.h"

#include "json.hpp"
using json = nlohmann::json;


// using namespace std;

// bohr -> angstrom
const double cvt_len  = 0.52917721;
const double icvt_len = 1./cvt_len;
// hatree -> eV
const double cvt_ener  = 27.21138602;
const double icvt_ener = 1./cvt_ener;
// hatree/Bohr -> eV / angstrom
const double cvt_f  = cvt_ener / cvt_len;
const double icvt_f = 1./cvt_f;

char *trimwhitespace(char *str)
{
  char *end;
  // Trim leading space
  while(isspace((unsigned char)*str)) str++;
  if(*str == 0)  // All spaces?
    return str;
  // Trim trailing space
  end = str + strlen(str) - 1;
  while(end > str && isspace((unsigned char)*end)) end--;
  // Write new null terminator
  *(end+1) = 0;
  return str;
}

void 
normalize_coord (std::vector<double > & coord,
		 const SimulationRegion<double > & region)
{
  int natoms = coord.size() / 3;

  for (int ii = 0; ii < natoms; ++ii){
    double inter[3];
    region.phys2Inter (inter, &coord[3*ii]);
    for (int dd = 0; dd < 3; ++dd){
      inter[dd] -= int(floor(inter[dd]));
      if      (inter[dd] < 0 ) inter[dd] += 1.;
      else if (inter[dd] >= 1) inter[dd] -= 1.;
    }
    region.inter2Phys (&coord[3*ii], inter);
  }
}

int main(int argc, char * argv[])
{
  if (argc == 1) {
    std::cerr << "usage " << std::endl;
    std::cerr << argv[0] << " input_script " << std::endl;
    return 1;
  }

  std::ifstream fp (argv[1]);
  json jdata;
  fp >> jdata;
  std::cout << "# using data base" << std::endl;
  std::cout << std::setw(4) << jdata << std::endl;

  int socket;
  int inet = 1;
  if (jdata["use_unix"]) {
    inet = 0;
  }
  int port = jdata["port"];
  std::string host_str = jdata["host"];
  const char * host = host_str.c_str();
  std::string graph_file = jdata["graph_file"];
  std::string coord_file = jdata["coord_file"];
  std::map<std::string, int> name_type_map = jdata["atom_type"];
  bool b_verb = jdata["verbose"];
  
  std::vector<std::string > atom_name;  
  {
    std::vector<std::vector<double > >  posi;
    std::vector<std::vector<double > >  velo;
    std::vector<std::vector<double > >  forc;
    XyzFileManager::read (coord_file, atom_name, posi, velo, forc);
  }

  Convert<double> cvt (atom_name, name_type_map);
  deepmd::DeepPot nnp_inter (graph_file);
  
  enum { _MSGLEN = 12 };
  int MSGLEN = _MSGLEN;
  char header [_MSGLEN+1] = {'\0'};
  bool hasdata = false;
  int32_t cbuf = 0;
  char initbuffer[2048];
  double cell_h [9];
  double cell_ih[9];
  int32_t natoms = -1;
  double dener (0);
  std::vector<double > dforce;
  std::vector<double > dforce_tmp;
  std::vector<double > dvirial (9, 0);
  std::vector<double > dcoord ;
  std::vector<double > dcoord_tmp ;
  std::vector<int > dtype = cvt.get_type();
  std::vector<double > dbox (9, 0) ;
  SimulationRegion<double > region;
  double * msg_buff = NULL;
  double ener;
  double virial[9];
  char msg_needinit[]	= "NEEDINIT    ";
  char msg_havedata[]	= "HAVEDATA    ";
  char msg_ready[]	= "READY       ";
  char msg_forceready[] = "FORCEREADY  ";
  char msg_nothing[]	= "nothing";
  
  open_socket_ (&socket, &inet, &port, host);
  
  bool isinit = true;

  while (true) {
    readbuffer_ (&socket, header, MSGLEN);
    std::string header_str (trimwhitespace(header));
    if (b_verb) std::cout << "# get header " << header_str << std::endl;

    if (header_str == "STATUS"){
      if (! isinit) {
	writebuffer_ (&socket, msg_needinit, MSGLEN);
	if (b_verb) std::cout << "# send back  " << "NEEDINIT" << std::endl;
      }
      else if (hasdata) {
	writebuffer_ (&socket, msg_havedata, MSGLEN);
	if (b_verb) std::cout << "# send back  " << "HAVEDATA" << std::endl;
      }
      else {
	writebuffer_ (&socket, msg_ready, MSGLEN);
	if (b_verb) std::cout << "# send back  " << "READY" << std::endl;
      }
    }
    else if (header_str == "INIT") {
      assert (4 == sizeof(int32_t));
      readbuffer_ (&socket, (char *)(&cbuf), sizeof(int32_t));
      readbuffer_ (&socket, initbuffer, cbuf);
      if (b_verb) std::cout << "Init sys from wrapper, using " << initbuffer << std::endl;
    }
    else if (header_str == "POSDATA"){
      assert (8 == sizeof(double));
      
      // get box
      readbuffer_ (&socket, (char *)(cell_h),  9*sizeof(double));
      readbuffer_ (&socket, (char *)(cell_ih), 9*sizeof(double));
      for (int dd = 0; dd < 9; ++dd){
	dbox[dd] = cell_h[dd] * cvt_len;
      }
      region.reinitBox (&dbox[0]);
      
      // get number of atoms
      readbuffer_ (&socket, (char *)(&cbuf), sizeof(int32_t));
      if (natoms < 0) {
	natoms = cbuf;
	if (b_verb) std::cout << "# get number of atoms in system: " << natoms << std::endl;
	
	dcoord.resize (3 * natoms);
	dforce.resize (3 * natoms, 0);
	dcoord_tmp.resize (3 * natoms);
	dforce_tmp.resize (3 * natoms, 0);
	msg_buff = new double [3 * natoms];
      }
      
      // get coord
      readbuffer_ (&socket, (char *)(msg_buff), natoms * 3 * sizeof(double));
      for (int ii = 0; ii < natoms * 3; ++ii){
	dcoord_tmp[ii] = msg_buff[ii] * cvt_len;
      }
      cvt.forward (dcoord, dcoord_tmp, 3);
      normalize_coord (dcoord, region);

      // nnp over writes ener, force and virial
#ifdef HIGH_PREC
      nnp_inter.compute (dener, dforce_tmp, dvirial, dcoord, dtype, dbox);   
#else 
      // model in float prec
      std::vector<float> dcoord_(dcoord.size());
      std::vector<float> dbox_(dbox.size());
      for (unsigned dd = 0; dd < dcoord.size(); ++dd) dcoord_[dd] = dcoord[dd];
      for (unsigned dd = 0; dd < dbox.size(); ++dd) dbox_[dd] = dbox[dd];
      std::vector<float> dforce_(dforce.size(), 0);
      std::vector<float> dvirial_(dvirial.size(), 0);
      double dener_ = 0;
      nnp_inter.compute (dener_, dforce_, dvirial_, dcoord_, dtype, dbox_);   
      for (unsigned dd = 0; dd < dforce.size(); ++dd) dforce_tmp[dd] = dforce_[dd];	
      for (unsigned dd = 0; dd < dvirial.size(); ++dd) dvirial[dd] = dvirial_[dd];	
      dener = dener_;      
#endif
      cvt.backward (dforce, dforce_tmp, 3);
      hasdata = true;
    }
    else if (header_str == "GETFORCE"){
      ener = dener * icvt_ener;
      for (int ii = 0; ii < natoms * 3; ++ii){
	msg_buff[ii] = dforce[ii] * icvt_f;
      }
      for (int ii = 0; ii < 9; ++ii){
	virial[ii] = dvirial[ii] * icvt_ener * (1.0);
      }
      if (b_verb) std::cout << "# energy of sys. : " << std::scientific << std::setprecision(10) << dener << std::endl;
      writebuffer_ (&socket, msg_forceready, MSGLEN);
      writebuffer_ (&socket, (char *)(&ener), sizeof(double));
      writebuffer_ (&socket, (char *)(&natoms), sizeof(int32_t));
      writebuffer_ (&socket, (char *)(msg_buff), 3 * natoms * sizeof(double));
      writebuffer_ (&socket, (char *)(virial), 9 * sizeof(double));
      cbuf = 7;
      writebuffer_ (&socket, (char *)(&cbuf), sizeof(int32_t));
      writebuffer_ (&socket, msg_nothing, 7);
      hasdata = false;
    }
    else {
      std::cerr << "unexpected header " << std::endl;
      return 1;
    }
  }

  if (msg_buff != NULL){
    delete [] msg_buff;
  }
}
