#include "NeighborList.h"
#include <iostream>
// #include <iomanip> 

using namespace std;
enum {
  MAX_WARN_IDX_OUT_OF_BOUND = 10,
};

bool 
is_loc (const vector<int> & idx, 
	const vector<int> & nat_stt,
	const vector<int> & nat_end)
{
  bool ret = true;
  for (int dd = 0; dd < 3; ++dd) ret = ret && idx[dd] >= nat_stt[dd];
  for (int dd = 0; dd < 3; ++dd) ret = ret && idx[dd] <  nat_end[dd];
  return ret;
}

int 
collapse_index (const vector<int> &	idx,
		const vector<int> &	size)
{
  return (idx[0] * size[1] + idx[1]) * size[2] + idx[2];
}

void
expand_index (vector<int > &		o_idx,
	      const int &		i_idx,
	      const vector<int> &	size)
{
  int tmp1 = i_idx / size[2];
  o_idx[2] = i_idx - tmp1 * size[2];
  o_idx[0] = tmp1 / size[1];
  o_idx[1] = tmp1 - o_idx[0] * size[1];
}

void 
build_clist (vector<vector<int > > &	clist,
	     const vector<double > &	coord,
	     const int &		nloc,
	     const vector<int > &	nat_stt,
	     const vector<int > &	nat_end,
	     const vector<int > &	ext_stt,
	     const vector<int > &	ext_end,
	     const SimulationRegion<double> & region,
	     const vector<int > &	global_grid)
{
  static int count_warning_loc_idx_lower = 0;
  static int count_warning_loc_idx_upper = 0;
  static int count_warning_ghost_idx_lower = 0;
  static int count_warning_ghost_idx_upper = 0;
  // compute region info, in terms of internal coord
  int nall = coord.size() / 3;
  vector<int> ext_ncell(3);
  for (int dd = 0; dd < 3; ++dd) ext_ncell[dd] = ext_end[dd] - ext_stt[dd];
  int ncell = ext_ncell[0] * ext_ncell[1] * ext_ncell[2];
  vector<double > cell_size (3);
  for (int dd = 0; dd < 3; ++dd) cell_size[dd] = 1./global_grid[dd];
  vector<double > nat_orig(3);
  for (int dd = 0; dd < 3; ++dd) nat_orig[dd] = nat_stt[dd] * cell_size[dd];
  vector<int> idx_orig_shift(3);
  for (int dd = 0; dd < 3; ++dd) idx_orig_shift[dd] = nat_stt[dd] - ext_stt[dd];
  
  // allocate the reserve the cell list
  clist.resize (ncell);
  int esti_natom_per_cell = nall / ncell + 10;
  for (unsigned ii = 0; ii < clist.size(); ++ii){
    clist[ii].clear();
    clist[ii].reserve (esti_natom_per_cell);
  }

  // build the cell list
  for (int ii = 0; ii < nloc; ++ii){
    double inter[3];
    region.phys2Inter (inter, &(coord[ii*3]));
    vector<int > idx(3);
    for (int dd = 0; dd < 3; ++dd){
      idx[dd] = (inter[dd] - nat_orig[dd]) / cell_size[dd];
      if (inter[dd] - nat_orig[dd] < 0.) idx[dd] --;
      if (idx[dd] < nat_stt[dd]) {
	if (count_warning_loc_idx_lower < MAX_WARN_IDX_OUT_OF_BOUND) {
	  cerr << "# warning: loc idx out of lower bound (ignored if warned for more than " << MAX_WARN_IDX_OUT_OF_BOUND << " times) " << endl;
	  count_warning_loc_idx_lower ++;
	}	
	idx[dd] = nat_stt[dd];
      }
      else if (idx[dd] >= nat_end[dd]) {
	if (count_warning_loc_idx_upper < MAX_WARN_IDX_OUT_OF_BOUND) {
	  cerr << "# warning: loc idx out of upper bound (ignored if warned for more than " << MAX_WARN_IDX_OUT_OF_BOUND << " times) " << endl;
	  count_warning_loc_idx_upper ++;
	}
	idx[dd] = nat_end[dd] - 1;
      }
      idx[dd] += idx_orig_shift[dd];
    }
    clist[collapse_index(idx, ext_ncell)].push_back (ii);
  }
  for (int ii = nloc; ii < nall; ++ii){
    double inter[3];
    region.phys2Inter (inter, &(coord[ii*3]));
    vector<int > idx(3);
    for (int dd = 0; dd < 3; ++dd){
      idx[dd] = (inter[dd] - nat_orig[dd]) / cell_size[dd];
      if (inter[dd] - nat_orig[dd] < 0.) idx[dd] --;
      if (idx[dd] < ext_stt[dd]) {
	if (count_warning_ghost_idx_lower < MAX_WARN_IDX_OUT_OF_BOUND &&
	    fabs((inter[dd] - nat_orig[dd]) - (ext_stt[dd] * cell_size[dd]))
	    > fabs(ext_stt[dd] * cell_size[dd]) * numeric_limits<double>::epsilon() * 5.
	    ) {
	  cerr << "# warning: ghost idx out of lower bound (ignored if warned for more than " << MAX_WARN_IDX_OUT_OF_BOUND << " times) " << endl;
	  count_warning_ghost_idx_lower ++;
	}
	idx[dd] = ext_stt[dd];
      }
      else if (idx[dd] >= ext_end[dd]) {
	if (count_warning_ghost_idx_upper < MAX_WARN_IDX_OUT_OF_BOUND) {
	  cerr << "# warning: ghost idx out of upper bound (ignored if warned for more than " << MAX_WARN_IDX_OUT_OF_BOUND << " times) " << endl;
	  count_warning_ghost_idx_upper ++;
	}
	idx[dd] = ext_end[dd] - 1;
      }
      idx[dd] += idx_orig_shift[dd];
    }
    clist[collapse_index(idx, ext_ncell)].push_back (ii);
  }
}

void 
build_clist (vector<vector<int > > &	clist,
	     const vector<double > &	coord,
	     const vector<int>  &	sel,
	     const vector<int > &	nat_stt,
	     const vector<int > &	nat_end,
	     const SimulationRegion<double> & region)
{
  static int count_warning_loc_idx_lower = 0;
  static int count_warning_loc_idx_upper = 0;
  // compute region info, in terms of internal coord
  int nall = coord.size() / 3;
  vector<int> nat_ncell(3);
  for (int dd = 0; dd < 3; ++dd) nat_ncell[dd] = nat_end[dd] - nat_stt[dd];
  int ncell = nat_ncell[0] * nat_ncell[1] * nat_ncell[2];
  vector<double > cell_size (3);
  for (int dd = 0; dd < 3; ++dd) cell_size[dd] = 1./nat_end[dd];
  vector<double > nat_orig(3);
  for (int dd = 0; dd < 3; ++dd) nat_orig[dd] = nat_stt[dd] * cell_size[dd];
  
  // allocate the reserve the cell list
  clist.resize (ncell);
  int esti_natom_per_cell = nall / ncell + 10;
  for (unsigned ii = 0; ii < clist.size(); ++ii){
    clist[ii].clear();
    clist[ii].reserve (esti_natom_per_cell);
  }

  // build the cell list
  for (unsigned _ = 0; _ < sel.size(); ++_){
    int ii = sel[_];
    double inter[3];
    region.phys2Inter (inter, &(coord[ii*3]));
    vector<int > idx(3);
    for (int dd = 0; dd < 3; ++dd){
      idx[dd] = (inter[dd] - nat_orig[dd]) / cell_size[dd];
      if (inter[dd] - nat_orig[dd] < 0.) idx[dd] --;
      if (idx[dd] < nat_stt[dd]) {
	if (count_warning_loc_idx_lower < MAX_WARN_IDX_OUT_OF_BOUND) {
	  cerr << "# warning: loc idx out of lower bound (ignored if warned for more than " << MAX_WARN_IDX_OUT_OF_BOUND << " times) " << endl;
	  count_warning_loc_idx_lower ++;
	}	
	idx[dd] = nat_stt[dd];
      }
      else if (idx[dd] >= nat_end[dd]) {
	if (count_warning_loc_idx_upper < MAX_WARN_IDX_OUT_OF_BOUND) {
	  cerr << "# warning: loc idx out of upper bound (ignored if warned for more than " << MAX_WARN_IDX_OUT_OF_BOUND << " times) " << endl;
	  count_warning_loc_idx_upper ++;
	}	
	idx[dd] = nat_end[dd] - 1;
      }
    }
    clist[collapse_index(idx, nat_ncell)].push_back (ii);
  }
}


void
build_nlist_cell (vector<vector<int> > &	nlist0,
		  vector<vector<int> > &	nlist1,
		  const int &			cidx,
		  const int &			tidx, 
		  const vector<vector<int > > &	clist,
		  const vector<double > &	coord,
		  const double &		rc02,
		  const double &		rc12,
		  const vector<int> &		shift = {0, 0, 0},
		  const vector<double > &	boxt = {0., 0., 0., 0., 0., 0., 0., 0., 0.})
{
  int nloc = nlist0.size();
  // loop over c (current) cell
  for (unsigned ii = 0; ii < clist[cidx].size(); ++ii){
    int i_idx = clist[cidx][ii];
    // assert (i_idx < nloc);
    // loop over t (target) cell
    for (unsigned jj = 0; jj < clist[tidx].size(); ++jj){
      int j_idx = clist[tidx][jj];
      if (cidx == tidx && j_idx <= i_idx) continue;
      double diff[3];
      for (int dd0 = 0; dd0 < 3; ++dd0) {
	diff[dd0] = coord[i_idx*3 + dd0] - coord[j_idx*3 + dd0];
	for (int dd1 = 0; dd1 < 3; ++dd1) {
	  diff[dd0] += shift[dd1] * boxt[3*dd1+dd0];
	}
      }
      double r2 = MathUtilities::dot<double> (diff, diff);
      if (r2 < rc02) {
	if (i_idx < nloc) nlist0[i_idx].push_back (j_idx);
	if (j_idx < nloc) nlist0[j_idx].push_back (i_idx);
      }
      else if (r2 < rc12) {
	if (i_idx < nloc) nlist1[i_idx].push_back (j_idx);
	if (j_idx < nloc) nlist1[j_idx].push_back (i_idx);
      }      
    }
  }
}

void
build_nlist_cell (vector<vector<int> > &	nlist0,
		  vector<vector<int> > &	nlist1,
		  const int &			cidx,
		  const int &			tidx, 
		  const vector<vector<int > > &	clist0,
		  const vector<vector<int > > &	clist1,
		  const vector<double > &	coord,
		  const double &		rc02,
		  const double &		rc12,
		  const vector<int> &		shift = {0, 0, 0},
		  const vector<double > &	boxt = {0., 0., 0., 0., 0., 0., 0., 0., 0.})
{
  // loop over c (current) cell
  for (unsigned ii = 0; ii < clist0[cidx].size(); ++ii){
    int i_idx = clist0[cidx][ii];
    if (i_idx >= nlist0.size()) continue;
    // loop over t (target) cell
    for (unsigned jj = 0; jj < clist1[tidx].size(); ++jj){
      int j_idx = clist1[tidx][jj];
      if (cidx == tidx && j_idx == i_idx) continue;
      double diff[3];
      for (int dd0 = 0; dd0 < 3; ++dd0) {
	diff[dd0] = coord[i_idx*3 + dd0] - coord[j_idx*3 + dd0];
	for (int dd1 = 0; dd1 < 3; ++dd1) {
	  diff[dd0] += shift[dd1] * boxt[3*dd1+dd0];
	}
      }
      double r2 = MathUtilities::dot<double> (diff, diff);
      if (r2 < rc02) {
	nlist0[i_idx].push_back (j_idx);
      }
      else if (r2 < rc12) {
	nlist1[i_idx].push_back (j_idx);
      }      
    }
  }
}

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
	     const vector<int > &	global_grid)
{
  // normalize the index
  // i require that the ext_stt = {0, 0, 0}
  vector<int > nat_stt (nat_stt_);
  vector<int > nat_end (nat_end_);
  vector<int > ext_stt (ext_stt_);
  vector<int > ext_end (ext_end_);
  
  // compute the clist
  vector<vector<int > > clist;
  build_clist (clist, coord, nloc, nat_stt, nat_end, ext_stt, ext_end, region, global_grid);

  // compute the region info
  int nall = coord.size() / 3;
  vector<int> ext_ncell(3);
  for (int dd = 0; dd < 3; ++dd) ext_ncell[dd] = ext_end[dd] - ext_stt[dd];

  // compute number of iter according to the cut-off
  assert (rc0 <= rc1);
  vector<int> niter (3);
  double to_face [3];
  region.toFaceDistance (to_face);
  for (int dd = 0; dd < 3; ++dd){
    double cell_size = to_face[dd] / nat_end[dd];
    niter[dd] = rc1 / cell_size;
    if (niter[dd] * cell_size < rc1) niter[dd] += 1;
    assert (niter[dd] * cell_size >= rc1);
  }
  // check the validity of the iters
  for (int dd = 0; dd < 3; ++dd){
    assert (nat_stt[dd] - niter[dd] >= ext_stt[dd]);
    assert (nat_end[dd] + niter[dd] <= ext_end[dd]);
  }

  // allocate the nlists
  double density = nloc / region.getVolume();
  nlist0.resize (nloc);
  for (int ii = 0; ii < nloc; ++ii){
    nlist0[ii].clear();
    int esti = 4./3. * 3.14 * (rc0*rc0*rc0) * density * 1.5 + 20;
    if (esti < 0) esti = 10;
    nlist0[ii].reserve ( esti );
  }  
  nlist1.resize (nloc);
  for (int ii = 0; ii < nloc; ++ii){
    nlist1[ii].clear();
    int esti = 4./3. * 3.14 * (rc1*rc1*rc1 - rc0*rc0*rc0) * density * 1.5 + 20;
    if (esti < 0) esti = 10;
    nlist1[ii].reserve ( esti );
  }

  // shift of the idx origin
  vector<int> idx_orig_shift(3);
  for (int dd = 0; dd < 3; ++dd) idx_orig_shift[dd] = nat_stt[dd] - ext_stt[dd];

  // compute the nlists
  double rc02 = 0;
  if (rc0 > 0) rc02 = rc0 * rc0;
  double rc12 = rc1 * rc1;
  vector<int> cidx(3);
  for (cidx[0] = nat_stt[0]; cidx[0] < nat_end[0]; ++cidx[0]){
    for (cidx[1] = nat_stt[1]; cidx[1] < nat_end[1]; ++cidx[1]){
      for (cidx[2] = nat_stt[2]; cidx[2] < nat_end[2]; ++cidx[2]){
	vector<int> mcidx(3);
	for (int dd = 0; dd < 3; ++dd) mcidx[dd] = cidx[dd] + idx_orig_shift[dd];
	int clp_cidx = collapse_index (mcidx, ext_ncell);
	vector<int> tidx(3);
	for (tidx[0] = cidx[0] - niter[0]; tidx[0] < cidx[0] + niter[0] + 1; ++tidx[0]) {
	  for (tidx[1] = cidx[1] - niter[1]; tidx[1] < cidx[1] + niter[1] + 1; ++tidx[1]) {
	    for (tidx[2] = cidx[2] - niter[2]; tidx[2] < cidx[2] + niter[2] + 1; ++tidx[2]) {
	      vector<int> mtidx(3);
	      for (int dd = 0; dd < 3; ++dd) mtidx[dd] = tidx[dd] + idx_orig_shift[dd];
	      int clp_tidx = collapse_index (mtidx, ext_ncell);
	      if (is_loc(tidx, nat_stt, nat_end) && clp_tidx < clp_cidx) continue;
	      build_nlist_cell (nlist0, nlist1, clp_cidx, clp_tidx, clist, coord, rc02, rc12);
	    }
	  }
	}
      }
    }
  }
}


// assume nat grid is the global grid. only used for serial simulations
void
build_nlist (vector<vector<int > > &	nlist0,
	     vector<vector<int > > &	nlist1,
	     const vector<double > &	coord,
	     const double &		rc0,
	     const double &		rc1,
	     const vector<int > &	grid,
	     const SimulationRegion<double> & region)
{
  // assuming nloc == nall
  int nloc = coord.size() / 3;
  // compute the clist
  vector<int> nat_stt(3, 0);
  vector<int> nat_end(grid);
  vector<vector<int > > clist;
  build_clist (clist, coord, nloc, nat_stt, nat_end, nat_stt, nat_end, region, nat_end);
  
  // compute the region info
  int nall = coord.size() / 3;
  vector<int> nat_ncell(3);
  for (int dd = 0; dd < 3; ++dd) nat_ncell[dd] = nat_end[dd] - nat_stt[dd];

  // compute number of iter according to the cut-off
  assert (rc0 <= rc1);
  vector<int> niter (3);
  double to_face [3];
  region.toFaceDistance (to_face);
  for (int dd = 0; dd < 3; ++dd){
    double cell_size = to_face[dd] / nat_end[dd];
    niter[dd] = rc1 / cell_size;
    if (niter[dd] * cell_size < rc1) niter[dd] += 1;
    assert (niter[dd] * cell_size >= rc1);
  }
  // check the validity of the iters
  for (int dd = 0; dd < 3; ++dd){
    assert (niter[dd] <= (nat_end[dd] - nat_stt[dd]) / 2);
  }

  // allocate the nlists
  double density = nall / region.getVolume();
  nlist0.resize (nloc);
  for (int ii = 0; ii < nloc; ++ii){
    nlist0[ii].clear();
    nlist0[ii].reserve ( 4./3. * 3.14 * (rc0*rc0*rc0) * density * 1.5 + 20);
  }  
  nlist1.resize (nloc);
  for (int ii = 0; ii < nloc; ++ii){
    nlist1[ii].clear();
    nlist1[ii].reserve ( 4./3. * 3.14 * (rc1*rc1*rc1 - rc0*rc0*rc0) * density * 1.5 + 20);
  }
  
  // physical cell size
  vector<double> phys_cs(9);
  for (int dd = 0; dd < 9; ++dd) phys_cs[dd] = region.getBoxTensor()[dd];

  // compute the nlists
  double rc02 = 0;
  if (rc0 > 0) rc02 = rc0 * rc0;
  double rc12 = rc1 * rc1;

#ifdef HALF_NEIGHBOR_LIST
  vector<int> cidx(3);
  for (cidx[0] = nat_stt[0]; cidx[0] < nat_end[0]; ++cidx[0]){
    for (cidx[1] = nat_stt[1]; cidx[1] < nat_end[1]; ++cidx[1]){
      for (cidx[2] = nat_stt[2]; cidx[2] < nat_end[2]; ++cidx[2]){
#else
  int idx_range[3];
  idx_range[0] = nat_end[0] - nat_stt[0];
  idx_range[1] = nat_end[1] - nat_stt[1];
  idx_range[2] = nat_end[2] - nat_stt[2];
  int idx_total = idx_range[0] * idx_range[1] * idx_range[2];
#pragma omp parallel for
  for (int tmpidx = 0; tmpidx < idx_total; ++tmpidx) {
    vector<int> cidx(3);
    cidx[0] = nat_stt[0] + tmpidx / (idx_range[1] * idx_range[2]);
    int tmpidx1 = tmpidx - cidx[0] * idx_range[1] * idx_range[2];
    cidx[1] = nat_stt[1] + tmpidx1 / idx_range[2];
    cidx[2] = nat_stt[2] + tmpidx1 - cidx[1] * idx_range[2];
    {
      {
#endif
	int clp_cidx = collapse_index (cidx, nat_ncell);
	vector<int> tidx(3);
	vector<int> stidx(3);
	vector<int> shift(3);
	for (tidx[0] = cidx[0] - niter[0]; tidx[0] < cidx[0] + niter[0] + 1; ++tidx[0]) {
	  shift[0] = 0;
	  if      (tidx[0] < 0)			shift[0] += 1;
	  else if (tidx[0] >= nat_ncell[0])	shift[0] -= 1;
	  stidx[0] = tidx[0] + shift[0] * nat_ncell[0];
	  for (tidx[1] = cidx[1] - niter[1]; tidx[1] < cidx[1] + niter[1] + 1; ++tidx[1]) {
	    shift[1] = 0;
	    if      (tidx[1] < 0)		shift[1] += 1;
	    else if (tidx[1] >= nat_ncell[1])	shift[1] -= 1;
	    stidx[1] = tidx[1] + shift[1] * nat_ncell[1];
	    for (tidx[2] = cidx[2] - niter[2]; tidx[2] < cidx[2] + niter[2] + 1; ++tidx[2]) {
	      shift[2] = 0;
	      if      (tidx[2] < 0)		shift[2] += 1;
	      else if (tidx[2] >= nat_ncell[2])	shift[2] -= 1;
	      stidx[2] = tidx[2] + shift[2] * nat_ncell[2];
	      int clp_tidx = collapse_index (stidx, nat_ncell);
#ifdef HALF_NEIGHBOR_LIST
	      if (clp_tidx < clp_cidx) continue;
	      build_nlist_cell (nlist0, nlist1, clp_cidx, clp_tidx, clist, coord, rc02, rc12, shift, phys_cs);
#else
	      build_nlist_cell (nlist0, nlist1, clp_cidx, clp_tidx, clist, clist, coord, rc02, rc12, shift, phys_cs);
#endif
	    }
	  }
	}
      }
    }
  }
}


void
build_nlist (vector<vector<int > > &	nlist0,
	     vector<vector<int > > &	nlist1,
	     const vector<double > &	coord,
	     const vector<int> &	sel0,
	     const vector<int> &	sel1,
	     const double &		rc0,
	     const double &		rc1,
	     const vector<int > &	grid,
	     const SimulationRegion<double> & region)
{
  int nloc = coord.size() / 3;
  // compute the clist
  vector<int> nat_stt(3, 0);
  vector<int> nat_end(grid);
  vector<vector<int > > clist0, clist1;
  build_clist (clist0, coord, sel0, nat_stt, nat_end, region);
  build_clist (clist1, coord, sel1, nat_stt, nat_end, region);
  
  // compute the region info
  int nall = coord.size() / 3;
  vector<int> nat_ncell(3);
  for (int dd = 0; dd < 3; ++dd) nat_ncell[dd] = nat_end[dd] - nat_stt[dd];

  // compute number of iter according to the cut-off
  assert (rc0 <= rc1);
  vector<int> niter (3);
  double to_face [3];
  region.toFaceDistance (to_face);
  for (int dd = 0; dd < 3; ++dd){
    double cell_size = to_face[dd] / nat_end[dd];
    niter[dd] = rc1 / cell_size;
    if (niter[dd] * cell_size < rc1) niter[dd] += 1;
    assert (niter[dd] * cell_size >= rc1);
  }
  // check the validity of the iters
  for (int dd = 0; dd < 3; ++dd){
    assert (niter[dd] <= (nat_end[dd] - nat_stt[dd]) / 2);
  }

  // allocate the nlists
  double density = nall / region.getVolume();
  nlist0.resize (nloc);
  for (int ii = 0; ii < nloc; ++ii){
    nlist0[ii].clear();
    nlist0[ii].reserve ( 4./3. * 3.14 * (rc0*rc0*rc0) * density * 1.5 + 20);
  }  
  nlist1.resize (nloc);
  for (int ii = 0; ii < nloc; ++ii){
    nlist1[ii].clear();
    nlist1[ii].reserve ( 4./3. * 3.14 * (rc1*rc1*rc1 - rc0*rc0*rc0) * density * 1.5 + 20);
  }
  
  // physical cell size
  vector<double> phys_cs(9);
  for (int dd = 0; dd < 9; ++dd) phys_cs[dd] = region.getBoxTensor()[dd];

  // compute the nlists
  double rc02 = 0;
  if (rc0 > 0) rc02 = rc0 * rc0;
  double rc12 = rc1 * rc1;
  vector<int> cidx(3);
  for (cidx[0] = nat_stt[0]; cidx[0] < nat_end[0]; ++cidx[0]){
    for (cidx[1] = nat_stt[1]; cidx[1] < nat_end[1]; ++cidx[1]){
      for (cidx[2] = nat_stt[2]; cidx[2] < nat_end[2]; ++cidx[2]){
	int clp_cidx = collapse_index (cidx, nat_ncell);
	vector<int> tidx(3);
	vector<int> stidx(3);
	vector<int> shift(3);
	for (tidx[0] = cidx[0] - niter[0]; tidx[0] < cidx[0] + niter[0] + 1; ++tidx[0]) {
	  shift[0] = 0;
	  if      (tidx[0] < 0)			shift[0] += 1;
	  else if (tidx[0] >= nat_ncell[0])	shift[0] -= 1;
	  stidx[0] = tidx[0] + shift[0] * nat_ncell[0];
	  for (tidx[1] = cidx[1] - niter[1]; tidx[1] < cidx[1] + niter[1] + 1; ++tidx[1]) {
	    shift[1] = 0;
	    if      (tidx[1] < 0)		shift[1] += 1;
	    else if (tidx[1] >= nat_ncell[1])	shift[1] -= 1;
	    stidx[1] = tidx[1] + shift[1] * nat_ncell[1];
	    for (tidx[2] = cidx[2] - niter[2]; tidx[2] < cidx[2] + niter[2] + 1; ++tidx[2]) {
	      shift[2] = 0;
	      if      (tidx[2] < 0)		shift[2] += 1;
	      else if (tidx[2] >= nat_ncell[2])	shift[2] -= 1;
	      stidx[2] = tidx[2] + shift[2] * nat_ncell[2];
	      int clp_tidx = collapse_index (stidx, nat_ncell);
	      build_nlist_cell (nlist0, nlist1, clp_cidx, clp_tidx, clist0, clist1, coord, rc02, rc12, shift, phys_cs);
	    }
	  }
	}
      }
    }
  }
}   

static int compute_pbc_shift (int idx, 
			      int ncell)
{
  int shift = 0;
  if (idx < 0) {
    shift = 1;
    while (idx + shift * ncell < 0) shift ++;
  }
  else if (idx >= ncell) {
    shift = -1;
    while (idx + shift * ncell >= ncell) shift --;
  }
  assert (idx + shift * ncell >= 0 && idx + shift * ncell < ncell);
  return shift;
}

void 
copy_coord (vector<double > & out_c, 
	    vector<int > & out_t, 
	    vector<int > & mapping,
	    vector<int> & ncell,
	    vector<int> & ngcell,
	    const vector<double > & in_c,
	    const vector<int > & in_t,
	    const double & rc,
	    const SimulationRegion<double > & region)
{
  int nloc = in_c.size() / 3;
  assert(nloc == in_t.size());

  ncell.resize(3);
  ngcell.resize(3);
  double to_face [3];
  double cell_size [3];
  region.toFaceDistance (to_face);
  for (int dd = 0; dd < 3; ++dd){
    ncell[dd]  = to_face[dd] / rc;
    if (ncell[dd] == 0) ncell[dd] = 1;
    cell_size[dd] = to_face[dd] / ncell[dd];
    ngcell[dd] = int(rc / cell_size[dd]) + 1;
    assert(cell_size[dd] * ngcell[dd] >= rc);
  }
  int total_ncell = (2 * ngcell[0] + ncell[0]) * (2 * ngcell[1] + ncell[1]) * (2 * ngcell[2] + ncell[2]);
  int loc_ncell = (ncell[0]) * (ncell[1]) * (ncell[2]);
  int esti_ntotal = total_ncell / loc_ncell * nloc + 10;

  // alloc
  out_c.reserve(esti_ntotal * 6);
  out_t.reserve(esti_ntotal * 2);
  mapping.reserve(esti_ntotal * 2);
  
  // build cell list
  vector<vector<int > > clist;
  vector<int> nat_stt(3, 0);
  build_clist (clist, in_c, nloc, nat_stt, ncell, nat_stt, ncell, region, ncell);

  // copy local atoms
  out_c.resize(nloc * 3);
  out_t.resize(nloc);
  mapping.resize(nloc);
  copy(in_c.begin(), in_c.end(), out_c.begin());
  copy(in_t.begin(), in_t.end(), out_t.begin());
  for (int ii = 0; ii < nloc; ++ii) mapping[ii] = ii;

  // push ghost
  vector<int> ii(3), jj(3), pbc_shift(3, 0);
  double pbc_shift_d[3];
  for (ii[0] = -ngcell[0]; ii[0] < ncell[0] + ngcell[0]; ++ii[0]){
    pbc_shift[0] = compute_pbc_shift(ii[0], ncell[0]);
    pbc_shift_d[0] = pbc_shift[0];
    jj[0] = ii[0] + pbc_shift[0] * ncell[0];
    for (ii[1] = -ngcell[1]; ii[1] < ncell[1] + ngcell[1]; ++ii[1]){
      pbc_shift[1] = compute_pbc_shift(ii[1], ncell[1]);
      pbc_shift_d[1] = pbc_shift[1];
      jj[1] = ii[1] + pbc_shift[1] * ncell[1];
      for (ii[2] = -ngcell[2]; ii[2] < ncell[2] + ngcell[2]; ++ii[2]){
	pbc_shift[2] = compute_pbc_shift(ii[2], ncell[2]);
	pbc_shift_d[2] = pbc_shift[2];
	jj[2] = ii[2] + pbc_shift[2] * ncell[2];
	// local cell, continue
	if (ii[0] >= 0 && ii[0] < ncell[0] &&
	    ii[1] >= 0 && ii[1] < ncell[1] &&
	    ii[2] >= 0 && ii[2] < ncell[2] ){
	  continue;
	}
	double shift_v [3];
	region.inter2Phys(shift_v, pbc_shift_d);
	int cell_idx = collapse_index(jj, ncell);
	vector<int> & cur_clist = clist[cell_idx];
	for (int kk = 0; kk < cur_clist.size(); ++kk){
	  int p_idx = cur_clist[kk];
	  double shifted_coord [3];
	  out_c.push_back(in_c[p_idx*3+0] - shift_v[0]);
	  out_c.push_back(in_c[p_idx*3+1] - shift_v[1]);
	  out_c.push_back(in_c[p_idx*3+2] - shift_v[2]);
	  out_t.push_back(in_t[p_idx]);
	  mapping.push_back(p_idx);
	  // double phys[3];
	  // for (int dd = 0; dd < 3; ++dd) phys[dd] = in_c[p_idx*3+dd] - shift_v[dd];
	  // double inter[3];
	  // region.phys2Inter(inter, phys);
	  // if (  inter[0] >= 0 && inter[0] < 1 &&
	  // 	inter[1] >= 0 && inter[1] < 1 &&
	  // 	inter[2] >= 0 && inter[2] < 1 ){
	  //   cout << out_c.size()  / 3 << " "
	  // 	 << inter[0] << " " 
	  // 	 << inter[1] << " " 
	  // 	 << inter[2] << " " 
	  // 	 << endl;
	  //   cout << "err here inner" << endl;
	  //   exit(1);
	  // }	  
	}
      }
    }
  }
}

