#include "AdWeight.h"
#include "CosSwitch.h"
#include <cmath>
#include <iostream>
#include <cassert>

AdWeight::
AdWeight (const VALUETYPE & pl){
  protect_level = pl;
}

void 
AdWeight::
sel_nn_atom (vector<VALUETYPE> & nn_coord,
	     vector<int> & nn_type,
	     vector<int> & nn_idx,
	     vector<int> & nn_tag,
	     const vector<VALUETYPE> & dcoord,
	     const vector<int> & dtype) const
{
  nn_coord.clear();
  nn_type.clear();
  nn_idx.clear();

  vector<int> & tag(nn_tag);
  zone_tag (tag, dcoord);
  for (int ii = 0; ii < tag.size(); ++ii){
    if (tag[ii] != 0){
      nn_coord.push_back (dcoord[3*ii+0]);
      nn_coord.push_back (dcoord[3*ii+1]);
      nn_coord.push_back (dcoord[3*ii+2]);
      nn_type .push_back (dtype[ii]);
      nn_idx  .push_back (ii);
    }
  }  
}

void 
AdWeight::
force_intpl (vector<VALUETYPE> & of,
	     const vector<VALUETYPE> & dcoord,
	     const vector<VALUETYPE> & ff_force,
	     const vector<VALUETYPE> & nn_force,
	     const vector<int> & nn_idx) const
{
  int nall = dcoord.size() / 3;
  
  vector<VALUETYPE> weight, weight_x;
  atom_weight (weight, weight_x, dcoord);
  assert (nall == weight.size());
  // for (unsigned ii = 0; ii < weight.size(); ++ii){
  //   cout << ii << " " << weight[ii] << " " << dcoord[ii*3] << endl;
  // }
  
  // cout << "of " << of.size() <<  endl;
  // cout << "dcoord " << dcoord.size() <<  endl;
  // cout << "ff_f " << ff_force.size() <<  endl;
  // cout << "nn_f " << nn_force.size() <<  endl;
  // cout << "nn_i " << nn_idx.size() <<  endl;
  // cout << "w " << weight.size() <<  endl;
  vector<VALUETYPE> nn_sum (3, 0.);
  vector<VALUETYPE> ff_sum (3, 0.);
  // for (int ii = 0; ii < ff_force.size() / 3; ++ii){
  //   for (int dd = 0; dd < 3; ++dd){
  //     ff_sum[dd] += ff_force[ii*3+dd];
  //   }
  // }
  // for (int ii = 0; ii < nn_force.size() / 3; ++ii){
  //   for (int dd = 0; dd < 3; ++dd){
  //     nn_sum[dd] += nn_force[ii*3+dd];
  //   }
  // }
  // cout << ff_sum[0]   << " "  << ff_sum[1]   << " "  << ff_sum[2]   << " " <<endl;
  // cout << nn_sum[0]   << " "  << nn_sum[1]   << " "  << nn_sum[2]   << " " <<endl;
  
  for (int ii = 0; ii < nn_idx.size(); ++ii){
    int idx = nn_idx[ii];
    for (int dd = 0; dd < 3; ++dd){
      // nn_sum[dd] += weight[idx] * nn_force[ii*3+dd];
      nn_sum[dd] +=  1 * nn_force[ii*3+dd];
      of[idx*3+dd] += weight[idx] * nn_force[ii*3+dd];
    }
    // cout << "nn " << dcoord[idx*3] << " " << weight[idx] << endl;
  }
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      of[ii*3+dd] += (1 - weight[ii]) * ff_force[ii*3+dd];
    }
    // cout << "ff " << dcoord[ii*3] << " " << 1-weight[ii] << endl;
  }

  for (int ii = 0; ii < of.size() / 3; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      ff_sum[dd] += ff_force[ii*3+dd];
    }
  }
  // cout << ff_sum[0]   << " "  << ff_sum[1]   << " "  << ff_sum[2]   << " " <<endl;
  // cout << nn_sum[0]   << " "  << nn_sum[1]   << " "  << nn_sum[2]   << " " <<endl;
  // cout << endl;
}


void 
AdWeight::
force_intpl (vector<VALUETYPE> & of,
	     const vector<VALUETYPE> & dcoord,
	     const vector<VALUETYPE> & ff_bd_force,
	     const vector<VALUETYPE> & ff_nb_force,
	     const vector<VALUETYPE> & nn_force,
	     const vector<int> & nn_idx) const
{
  int nall = dcoord.size() / 3;
  
  vector<VALUETYPE> weight, weight_x;
  atom_weight (weight, weight_x, dcoord);
  assert (nall == weight.size());

  vector<VALUETYPE> nn_sum (3, 0.);
  vector<VALUETYPE> ff_sum (3, 0.);
  
  for (int ii = 0; ii < nn_idx.size(); ++ii){
    int idx = nn_idx[ii];
    for (int dd = 0; dd < 3; ++dd){
      // nn_sum[dd] += weight[idx] * nn_force[ii*3+dd];
      // nn_sum[dd] +=  1 * nn_force[ii*3+dd];
      of[idx*3+dd] += weight[idx] * nn_force[ii*3+dd];
      // if (fabs(nn_force[ii*3+dd]) > 1e6) {
      // 	cout << " ii " << ii
      // 	     << " dd " << dd 
      // 	     << " coord " << dcoord[ii*3+dd]
      // 	     << " nn_f " << nn_force[ii*3+dd]
      // 	     << " ww " << weight[ii]
      // 	     << endl;
      // }
    }
    // cout << "nn " << dcoord[idx*3] << " " << weight[idx] << endl;
  }

  // double protect_level = 1e-3;
  // cout << "with protect_level " << protect_level << endl;
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      double pref = (1 - weight[ii]);
      if (fabs(pref) < protect_level) pref = protect_level;
      of[ii*3+dd] += pref * ff_bd_force[ii*3+dd];
      // if (fabs(ff_bd_force[ii*3+dd]) > 1e6) {
      // 	cout << " ii " << ii
      // 	     << " dd " << dd 
      // 	     << " coord " << dcoord[ii*3+dd]
      // 	     << " ff_f " << ff_bd_force[ii*3+dd]
      // 	     << " ww " << 1 - weight[ii]
      // 	     << endl;
      // }
    }
    // cout << "ff " << dcoord[ii*3] << " " << 1-weight[ii] << endl;
  }
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      of[ii*3+dd] += (1 - weight[ii]) * ff_nb_force[ii*3+dd];
      // if (fabs(ff_nb_force[ii*3+dd]) > 1e6) {
      // 	cout << " ii " << ii
      // 	     << " dd " << dd 
      // 	     << " coord " << dcoord[ii*3+dd]
      // 	     << " ff_f " << ff_nb_force[ii*3+dd]
      // 	     << " ww " << 1 - weight[ii]
      // 	     << endl;
      // }
    }
    // cout << "ff " << dcoord[ii*3] << " " << 1-weight[ii] << endl;
  }

  for (int ii = 0; ii < of.size() / 3; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      ff_sum[dd] += ff_bd_force[ii*3+dd];
    }
  }
  // cout << ff_sum[0]   << " "  << ff_sum[1]   << " "  << ff_sum[2]   << " " <<endl;
  // cout << nn_sum[0]   << " "  << nn_sum[1]   << " "  << nn_sum[2]   << " " <<endl;
  // cout << endl;
}
	     

SlabWeight::
SlabWeight (const vector<VALUETYPE> & box,
	    const VALUETYPE & rnn_,
	    const VALUETYPE & rhy_,
	    const VALUETYPE & rc_, 
	    const VALUETYPE & protect_level_)
  : AdWeight (protect_level_)
{
  assert (box.size() == 9);
  center.resize(3);
  for (int ii = 0; ii < 3; ++ii){
    center[ii] = 0.5 * box[3*ii+ii];
  }
  rnn = rnn_;
  rhy = rhy_;
  rc = rc_;
}


void
SlabWeight::
zone_tag (vector<int > & tag,
	  const vector<VALUETYPE> & coord) const
{
  int natoms = coord.size() / 3;
  tag.resize(natoms, 0);
  
  // slab axis x
  VALUETYPE radius = rnn + rhy;
  for (int ii = 0; ii < natoms; ++ii){
    VALUETYPE posi = fabs(coord[ii*3] - center[0]);
    if (posi < radius) {
      tag[ii] = 3;
    }
    else if (posi < radius + rc){
      tag[ii] = 2;
    }
    else if (posi < radius + rc * 2){
      tag[ii] = 1;
    }
    else {
      tag[ii] = 0;
    }
  }
}


// dirty hacking
void
SlabWeight::
atom_weight (vector<VALUETYPE > & weight,
	     vector<VALUETYPE > & weight_x,
	     const vector<VALUETYPE> & coord) const
{
  CosSwitch cs (rnn, rnn + rhy);
  
  int natoms = coord.size() / 3;
  weight.resize(natoms, 0);
  weight_x.resize(natoms, 0);
  // slab axis x
  // for (int ii = 0; ii < natoms; ++ii){
  //   VALUETYPE posi = fabs(coord[ii*3] - center[0]);
  //   cs.eval (weight[ii], posi);    
  //   // if (posi < radius){
  //   //   weight[ii] = 1.;
  //   // }
  //   // else {
  //   //   weight[ii] = 0.;
  //   // }
  // }
  // for (int ii = 0; ii < natoms/3; ++ii){
  //   VALUETYPE posi = fabs(coord[ii*3] - center[0]);
  //   cs.eval (weight[ii], posi);
  //   weight[natoms/3 + ii*2 + 0] = weight[ii];
  //   weight[natoms/3 + ii*2 + 1] = weight[ii];
  //   // weight_x
  //   weight_x[ii] = posi;
  //   weight_x[natoms/3 + ii*2 + 0] = posi;
  //   weight_x[natoms/3 + ii*2 + 1] = posi;    
  //   // if (posi < radius){
  //   //   weight[ii] = 1.;
  //   // }
  //   // else {
  //   //   weight[ii] = 0.;
  //   // }
  // }
  for (int ii = 0; ii < natoms; ii += 3){
    VALUETYPE posi = fabs (coord[ii*3] - center[0]);
    cs.eval (weight[ii], posi);
    weight[ii + 1] = weight[ii];
    weight[ii + 2] = weight[ii];
    // weight_x
    weight_x[ii] = posi;
    weight_x[ii + 1] = posi;
    weight_x[ii + 2] = posi;
  }
}

