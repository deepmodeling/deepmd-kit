#include "HarmonicAngle.h"
#include "common.h"
#include <iostream> 
#include <cmath>
#include "mymath.h"

HarmonicAngle::
HarmonicAngle (const VALUETYPE & ka_,
	      const VALUETYPE & tt_)
    : ka(ka_), tt(tt_)
{
}

inline bool 
compute_variable (const VALUETYPE * rij,
		  const VALUETYPE * rkj,
		  VALUETYPE * var,
		  VALUETYPE * dvardcos,
		  VALUETYPE * cos_theta)
{
  *cos_theta = cos<VALUETYPE> (rij[0], rij[1], rij[2], rkj[0], rkj[1], rkj[2]);
  *var = acos (*cos_theta);
      
  VALUETYPE cos_theta2 = *cos_theta * *cos_theta;
  if (cos_theta2 >= 1) {
    *dvardcos = 1.;
    return false;
  }
  *dvardcos = - 1./sqrt(1. - cos_theta2);
  return true;
}
		  

void
HarmonicAngle::
compute (VALUETYPE &			ener,
	 vector<VALUETYPE> &		force,
	 vector<VALUETYPE> &		virial,
	 const vector<VALUETYPE> &	coord,
	 const vector<int> &		atype,
	 const SimulationRegion<VALUETYPE> &	region, 
	 const vector<int > &		alist)
{
  // all set zeros
  for (unsigned _ = 0; _ < alist.size(); _ += 3){
    int ii = alist[_];
    int jj = alist[_+1];
    int kk = alist[_+2];    

    VALUETYPE rij[3], rkj[3];
    region.diffNearestNeighbor (&coord[ii*3], &coord[jj*3], rij);      
    region.diffNearestNeighbor (&coord[kk*3], &coord[jj*3], rkj);      

    VALUETYPE var(0), dvardcos(0), cos_theta(0);    
    bool apply_force = compute_variable (rij, rkj, &var, &dvardcos, &cos_theta);

    VALUETYPE dudvar(0), angle_energy(0);
    VALUETYPE diff = var - tt;
    VALUETYPE pdiff = ka * diff;
    dudvar = - pdiff;
    angle_energy = VALUETYPE(0.5) * pdiff * diff;

    ener += angle_energy;

    // VALUETYPE fijx, fijy, fijz;
    // VALUETYPE fkjx, fkjy, fkjz;
    VALUETYPE fij[3];
    VALUETYPE fkj[3];

    if (apply_force) {    
      VALUETYPE dudcos = dudvar * dvardcos;
      VALUETYPE rij2 = dot<VALUETYPE> (rij, rij);
      VALUETYPE rkj2 = dot<VALUETYPE> (rkj, rkj);
      VALUETYPE invrij = 1./sqrt (rij2);
      VALUETYPE invrkj = 1./sqrt (rkj2);
      VALUETYPE invrij2 = invrij * invrij;
      VALUETYPE invrkj2 = invrkj * invrkj;
      VALUETYPE invrijrkj = invrij * invrkj;
      // can be further optimized:
      fij[0] = dudcos * (rkj[0] * invrijrkj - rij[0] * invrij2 * cos_theta);
      fij[1] = dudcos * (rkj[1] * invrijrkj - rij[1] * invrij2 * cos_theta);
      fij[2] = dudcos * (rkj[2] * invrijrkj - rij[2] * invrij2 * cos_theta);
      fkj[0] = dudcos * (rij[0] * invrijrkj - rkj[0] * invrkj2 * cos_theta);
      fkj[1] = dudcos * (rij[1] * invrijrkj - rkj[1] * invrkj2 * cos_theta);
      fkj[2] = dudcos * (rij[2] * invrijrkj - rkj[2] * invrkj2 * cos_theta);
    }
    else {
      fij[0] = fij[1] = fij[2] = fkj[0] = fkj[1] = fkj[2] = VALUETYPE(0);
    }

    force[3 * ii + 0] += fij[0];
    force[3 * ii + 1] += fij[1];
    force[3 * ii + 2] += fij[2];
    force[3 * kk + 0] += fkj[0];
    force[3 * kk + 1] += fkj[1];
    force[3 * kk + 2] += fkj[2];
    force[3 * jj + 0] -= fij[0] + fkj[0];
    force[3 * jj + 1] -= fij[1] + fkj[1];
    force[3 * jj + 2] -= fij[2] + fkj[2];
    for (int dd0 = 0; dd0 < 3; ++dd0) {
      for (int dd1 = 0; dd1 < 3; ++dd1) {
	virial[dd0*3+dd1] -= 0.5 * fij[dd0] * rij[dd1];
	virial[dd0*3+dd1] -= 0.5 * fkj[dd0] * rkj[dd1];
      }
    }
  }
}
