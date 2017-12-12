#ifndef __MDFileManager_Trajectory_h_wanghan__
#define __MDFileManager_Trajectory_h_wanghan__

// #include "Defines.h"
#include "xdrfile/xdrfile.h"
#include "xdrfile/xdrfile_xtc.h"
#include "xdrfile/xdrfile_trr.h"
#include <vector> 

using namespace std;

class XtcSaver
{
public:
  XtcSaver () : inited(false), prec(1000) {};
  ~XtcSaver ();
  XtcSaver (const char * filename,
	    const int & natoms);
  bool reinit (const char * filename,
	       const int & natoms);
public:
  void save (const int & step,
	     const double & time,
	     const vector<vector<double > > & frame, 
	     const vector<double > & box);
private:
  XDRFILE *xd;
  int natoms;
  rvec * xx;
  float prec;
  bool inited;
  void clear ();
};

class TrrSaver
{
public:
  TrrSaver () : inited(false), lambda(0) {};
  ~TrrSaver ();
  TrrSaver (const char * filename,
	    const int & natoms);
  bool reinit (const char * filename,
	       const int & natoms);
public:
  void save (const int & step,
	     const double & time,
	     const vector<vector<double > > & ixx, 
	     const vector<vector<double > > & ivv, 
	     const vector<vector<double > > & iff, 
	     const vector<double > & box);
private:
  XDRFILE *xd;
  int natoms;
  rvec * xx, *vv, *ff;
  float lambda;
  bool inited;
  void clear ();
};

#endif
