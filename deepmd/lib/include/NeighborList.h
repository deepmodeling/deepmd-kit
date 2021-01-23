#pragma once

#include <algorithm>
#include <iterator>
#include <cassert>

#include "MathUtilities.h"
#include "SimulationRegion.h"

// build nlist by an extended grid
void
build_nlist (vector<vector<int > > &	nlist0,
	     vector<vector<int > > &	nlist1,
	     const vector<double > &	coord,
	     const int &		nloc,
	     const double &		rc0,
	     const double &		rc1,
	     const vector<int > &	nat_stt_,
	     const vector<int > &	nat_end_,
	     const vector<int > &	ext_stt_,
	     const vector<int > &	ext_end_,
	     const SimulationRegion<double> & region,
	     const vector<int > &	global_grid);

// build nlist by a grid for a periodic region
void
build_nlist (vector<vector<int > > &	nlist0,
	     vector<vector<int > > &	nlist1,
	     const vector<double > &	coord,
	     const double &		rc0,
	     const double &		rc1,
	     const vector<int > &	grid,
	     const SimulationRegion<double> & region);

// build nlist by a grid for a periodic region, atoms selected by sel0 and sel1
void
build_nlist (vector<vector<int > > &	nlist0,
	     vector<vector<int > > &	nlist1,
	     const vector<double > &	coord,
	     const vector<int> &	sel0,
	     const vector<int> &	sel1,
	     const double &		rc0,
	     const double &		rc1,
	     const vector<int > &	grid,
	     const SimulationRegion<double> & region);

// brute force (all-to-all distance computation) neighbor list building
// if region is NULL, open boundary is assumed,
// otherwise, periodic boundary condition is defined by region
void
build_nlist (vector<vector<int > > & nlist0,
	     vector<vector<int > > & nlist1,
	     const vector<double > & coord,
	     const double & rc0_,
	     const double & rc1_,
	     const SimulationRegion<double > * region = NULL);

// copy periodic images for the system
void 
copy_coord (vector<double > & out_c, 
	    vector<int > & out_t, 
	    vector<int > & mapping,
	    vector<int> & ncell,
	    vector<int> & ngcell,
	    const vector<double > & in_c,
	    const vector<int > & in_t,
	    const double & rc,
	    const SimulationRegion<double > & region);
