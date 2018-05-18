#pragma once

#include <algorithm>
#include <iterator>
#include <cassert>

#include "MathUtilities.h"
#include "SimulationRegion.h"

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
void
build_nlist (vector<vector<int > > &	nlist0,
	     vector<vector<int > > &	nlist1,
	     const vector<double > &	coord,
	     const double &		rc0,
	     const double &		rc1,
	     const vector<int > &	grid,
	     const SimulationRegion<double> & region);
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
