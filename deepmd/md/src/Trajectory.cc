#include "Trajectory.h"
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cassert>

bool
XtcSaver::
reinit (const char * filename,
	const int & natoms_)
{
  char tmpname[2048];
  strncpy (tmpname, filename, 2047);
  
  xd = xdrfile_open (filename, "w");
  if (xd == NULL){
    std::cerr << "cannot open file " << filename << std::endl;
    return false;
  }
  natoms = natoms_;

  xx = (rvec *) malloc (sizeof(rvec) * natoms);
  inited = true;
  return true;  
}

XtcSaver::
~XtcSaver ()
{
  clear();
}

XtcSaver::
XtcSaver (const char * filename,
	  const int & natoms_)
    : inited(false), prec(1000)
{
  reinit (filename, natoms_);
}

void
XtcSaver::
clear ()
{
  if (inited){
    free (xx);
    xdrfile_close (xd);
    inited = false;
  }
}

void
XtcSaver::
save (const int & step,
      const double & time,
      const vector<vector<double > > & frame, 
      const vector<double > & box)
{
  assert (box.size() == 9);
  assert (inited);
  matrix tmpBox;
  for (int dd0 = 0; dd0 < 3; ++dd0){
    for (int dd1 = 0; dd1 < 3; ++dd1){
      tmpBox[dd0][dd1] = 0;
    }
  }
  for (int dd = 0; dd < 3; ++dd){
    tmpBox[dd][dd] = box[3*dd+dd];
  }
  for (int ii = 0; ii < frame.size(); ++ii){
    for (int dd = 0; dd < 3; ++dd) xx[ii][dd] = frame[ii][dd];
  }
  write_xtc (xd, natoms, step, time, tmpBox, xx, prec);
}


bool
TrrSaver::
reinit (const char * filename,
	const int & natoms_)
{
  char tmpname[2048];
  strncpy (tmpname, filename, 2047);
  
  xd = xdrfile_open (filename, "w");
  if (xd == NULL){
    std::cerr << "cannot open file " << filename << std::endl;
    return false;
  }
  natoms = natoms_;

  xx = (rvec *) malloc (sizeof(rvec) * natoms);
  vv = (rvec *) malloc (sizeof(rvec) * natoms);
  ff = (rvec *) malloc (sizeof(rvec) * natoms);
  for (int ii = 0; ii < natoms; ++ii){
    for (int dd = 0; dd < 3; ++dd) {
      vv[ii][dd] = 0;
      ff[ii][dd] = 0;
    }
  }
  inited = true;
  return true;  
}

TrrSaver::
~TrrSaver ()
{
  clear();
}

TrrSaver::
TrrSaver (const char * filename,
	  const int & natoms_)
    : inited(false), lambda(0)
{
  reinit (filename, natoms_);
}

void
TrrSaver::
clear ()
{
  if (inited){
    free (xx);
    free (vv);
    free (ff);
    xdrfile_close (xd);
    inited = false;
  }
}

void
TrrSaver::
save (const int & step,
      const double & time,
      const vector<vector<double > > & ixx, 
      const vector<vector<double > > & ivv, 
      const vector<vector<double > > & iff, 
      const vector<double > & box)
{
  assert (box.size() == 9);
  assert (inited);
  matrix tmpBox;
  for (int dd0 = 0; dd0 < 3; ++dd0){
    for (int dd1 = 0; dd1 < 3; ++dd1){
      tmpBox[dd0][dd1] = box[3*dd0 + dd1];
    }
  }
  for (int ii = 0; ii < ixx.size(); ++ii){
    for (int dd = 0; dd < 3; ++dd) xx[ii][dd] = ixx[ii][dd];
  }
  for (int ii = 0; ii < natoms; ++ii){
    for (int dd = 0; dd < 3; ++dd) {
      vv[ii][dd] = 0;
      ff[ii][dd] = 0;
    }
  }
  for (int ii = 0; ii < ivv.size(); ++ii){
    for (int dd = 0; dd < 3; ++dd) vv[ii][dd] = ivv[ii][dd];
  }
  for (int ii = 0; ii < iff.size(); ++ii){
    for (int dd = 0; dd < 3; ++dd) ff[ii][dd] = iff[ii][dd];
  }
  write_trr (xd, natoms, step, time, lambda, tmpBox, xx, vv, ff);
}


