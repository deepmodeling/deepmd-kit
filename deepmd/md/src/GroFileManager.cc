#include "GroFileManager.h"
#include <iterator>
#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;

class WrongFileFormat
{
};

bool GroFileManager::readTop (const std::string & filename,
			      std::vector<std::string > & molnames,
			      std::vector<int > & nmols)
{
  molnames.clear();
  nmols.clear();

  std::ifstream in(filename.c_str());
  if (in.bad()){
    std::cerr << "cannot open file " << filename << std::endl;
    return false;
  }
  char line [1024];
  std::string target ("[ molecules ]");
  bool find = false;
  while (!in.eof()){
    in.getline (line, 1024, '\n');
    if (target == std::string (line)) {
      find = true;
      break;
    }
  }
  if (!find){
    std::cerr << "cannot find [ molecules ] in file " << filename 
	      << ". please check there is no space after \"]\"\n";
    return false;
  }
  
//   while (!(in.getline (line, 1024, '\n')).eof()){
// //     if (line[0] == '['){
// //       break;
// //     }
// //     char name[1024];
// //     int number;
// //     sscanf (line, "%s%d", name, &number);
// //     molnames.push_back (std::string(name));
// //     nmols.push_back (number);
//   }
  

  std::string name;
  int number;
  while ( !(in >> name >> number).eof()){
    if (name[0] == '['){
      break;
    }
    if (name.empty()){
      break;
    }
//     std::cout << name << std::endl;
    molnames.push_back (name);
    nmols.push_back (number);
  }
  

  return true;
}



template <typename UnitaryFunction1, typename UnitaryFunction2,
	  typename UnitaryFunction3, typename UnitaryFunction4,
	  typename UnitaryFunction5, typename UnitaryFunction6>
bool GroFileManager::writePotenFile (const double & rmin, const double & rcut, 
				     const double & interval,
				     UnitaryFunction1 & f, UnitaryFunction2 & fp,
				     UnitaryFunction3 & g, UnitaryFunction4 & gp,
				     UnitaryFunction5 & h, UnitaryFunction6 & hp,
				     const std::string & filename)
{
  FILE * filep = fopen (filename.c_str(), "w");
  if(filep == NULL){
    std::cerr << "cannot open file " << filename << std::endl;
    return false;
  }

  double upper = rcut + 1;
  double nx;
  if ( int(upper / interval) != upper / interval)
    nx = int(upper / interval) + 1;
  else 
    nx = int(upper / interval);
  upper = interval * nx;
  
  int i = 0;
  for (i = 0; i <= nx + 1; ++i){
    double x = i * interval;
    if (x < rmin){
      fprintf (filep, "%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\n",
	       x, 0., 0., 0., 0., 0., 0.);
    }
    else {
      fprintf (filep, "%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\n",
	       x, f(x), -fp(x), g(x), -gp(x), h(x), -hp(x));
    }
  }
  
  fclose (filep);
  return true;
}



void GroFileManager::read (const std::string & name ,
			   std::vector<int > & resdindex,
			   std::vector<std::string > &  resdname,
			   std::vector<std::string > &  atomname,
			   std::vector<int > & atomindex,
			   std::vector<std::vector<double > > & posi,
			   std::vector<std::vector<double > > & velo,
			   std::vector<double > & boxsize_)
{
  FILE * fp = fopen (name.c_str(), "r");
  if (fp == NULL){
    std::cerr << "cannot open file " << name << std::endl;
    return;
  }
  while (fgetc(fp) != '\n');
  int npart;
  fscanf (fp, "%d\n", &npart);
  fclose (fp);
  
  resdindex.clear();
  resdname.clear();
  atomname.clear();
  atomindex.clear();
  posi.clear();
  velo.clear();
  vector<double > boxsize;
  boxsize.resize(3);

  fp = fopen (name.c_str(), "r");  
  while (fgetc(fp) != '\n');
  while (fgetc(fp) != '\n');
  char line[1024];
  for (int i = 0; i < npart; ++ i){
    fgets (line, 1024, fp);
    char tmp[1024];
    int tmpd;
    char tmps[1024];
    for (unsigned j = 0; j < 5; ++j){
      tmp[j] = line[j];
    }
    tmp[5] = '\0';
    if (sscanf (tmp, "%d", &tmpd) != 1){
      throw WrongFileFormat();
    }
    resdindex.push_back(tmpd);

    for (unsigned j = 0; j < 5; ++j){
      tmp[j] = line[j+5];
    }
    tmp[5] = '\0';
    if (sscanf (tmp, "%s", tmps) != 1){
      throw WrongFileFormat();
    }
    resdname.push_back (tmps);

    for (unsigned j = 0; j < 5; ++j){
      tmp[j] = line[j+10];
    }
    tmp[5] = '\0';
    if (sscanf (tmp, "%s", tmps) != 1){
      throw WrongFileFormat();
    }
    atomname.push_back (tmps);

    for (unsigned j = 0; j < 5; ++j){
      tmp[j] = line[j+15];
    }
    tmp[5] = '\0';
    if (sscanf (tmp, "%d", &tmpd) != 1){
      throw WrongFileFormat();
    }
    atomindex.push_back(tmpd);

    double a, b, c;
    double d, e, f;
    std::vector<double > tmpp(3);
    std::vector<double > tmpv(3);

    int tag = sscanf (&line[20], "%lf%lf%lf%lf%lf%lf", &a, &b, &c, &d, &e, &f);
    tmpp[0] = a;
    tmpp[1] = b;
    tmpp[2] = c;
    switch (tag){
    case 6:
	tmpv[0] = d;
	tmpv[1] = e;
	tmpv[2] = f;
	break;
    case 3:
	tmpv[0] = 0.f;
	tmpv[1] = 0.f;
	tmpv[2] = 0.f;
	break;
    default:
	throw WrongFileFormat();
    }

    posi.push_back(tmpp);
    velo.push_back(tmpv);
  }
  int tag = 0;
  double rbox[9];
  tag = fscanf (fp, "%lf%lf%lf%lf%lf%lf%lf%lf%lf",
		rbox+0, rbox+1, rbox+2,
		rbox+3, rbox+4, rbox+5,
		rbox+6, rbox+7, rbox+8);
  fclose (fp);

  boxsize_.resize (9, 0.);
  fill (boxsize_.begin(), boxsize_.end(), 0.);

  if (tag == 9){
    boxsize_[0] = rbox[0];
    boxsize_[4] = rbox[1];
    boxsize_[8] = rbox[2];
    boxsize_[0*3+1] = rbox[3];
    boxsize_[0*3+2] = rbox[4];
    boxsize_[1*3+0] = rbox[5];
    boxsize_[1*3+2] = rbox[6];
    boxsize_[2*3+0] = rbox[7];
    boxsize_[2*3+1] = rbox[8];
  }
  else {
    assert (tag == 3);
    boxsize_[0] = rbox[0];
    boxsize_[4] = rbox[1];
    boxsize_[8] = rbox[2];    
  }
}

void GroFileManager::write (const std::string & name ,
			    const std::vector<int > & resdindex,
			    const std::vector<std::string > &  resdname,
			    const std::vector<std::string > &  atomname,
			    const std::vector<int > & atomindex,
			    const std::vector<std::vector<double > > & posi,
			    const std::vector<std::vector<double > > & velo,
			    const std::vector<double > & boxsize)
{
  FILE * fp = fopen(name.c_str(), "w");
  if (fp == NULL){
    std::cerr << "cannot open file " << name << std::endl;
    return;
  }
  // std::copy (atomname.begin(), atomname.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
  
  fprintf (fp, "\n%d\n", int(resdindex.size()));
  for (int i = 0; i < int(resdindex.size()); ++i){
    fprintf (fp, "%5d%5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f\n",
	     resdindex[i] % 100000,
	     (char *)(resdname[i].c_str()),
	     (char *)(atomname[i].c_str()),
	     atomindex[i] % 100000, 
	     posi[i][0], posi[i][1], posi[i][2],
	     velo[i][0], velo[i][1], velo[i][2]);
  }
  // vector<double > box(3);
  // for (int ii = 0; ii < 3; ++ii) box[ii] = boxsize[3*ii+ii];
  if (boxsize.size() == 3){
    fprintf (fp, "%f %f %f\n", boxsize[0], boxsize[1], boxsize[2]);
  }
  else if (boxsize.size() == 9){    
    fprintf (fp, "%f %f %f %f %f %f %f %f %f \n",
	     boxsize[0*3+0], boxsize[1*3+1], boxsize[2*3+2],
	     boxsize[0*3+1], boxsize[0*3+2],
	     boxsize[1*3+0], boxsize[1*3+2],
	     boxsize[2*3+0], boxsize[2*3+1]);
  }

  fclose (fp);
}


struct F 
{
  double operator () (double x)
      {
	return 1./x;
      }
}
    ;
struct Zero
{
  double operator () (double x)
      {
	return 0;
      }
}
    ;

