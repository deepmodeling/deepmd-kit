#include "MaxShift.h"
#include "common.h"

#include <cassert>

MaxShift::
MaxShift (const vector<VALUETYPE> & dcoord, 
	  const VALUETYPE & shell_)
{
  record = dcoord;
  shell = shell_;
  max_allow2 = shell * 0.5 * shell * 0.5;
}

VALUETYPE
MaxShift::
max_shift2 (const vector<VALUETYPE> & coord, 
	    const SimulationRegion<VALUETYPE> & region) 
{
  assert (coord.size() == record.size());
  int natoms = coord.size() / 3;
  
  VALUETYPE maxv = 0;
  
  for (int ii = 0; ii < natoms; ++ii){
    VALUETYPE diff[3];
    region.diffNearestNeighbor (&coord[ii*3], &record[ii*3], diff);
    VALUETYPE r2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
    if (r2 > maxv) maxv = r2;
  }

  return maxv;
}

bool 
MaxShift::
rebuild (const vector<VALUETYPE> & coord, 
	 const SimulationRegion<VALUETYPE> & region) 
{
  VALUETYPE maxv2 = max_shift2 (coord, region);
  if (maxv2 > max_allow2){
    record = coord;
    return true;
  }
  else {
    return false;
  }
}



