#include "neighbor_list.h"
#include "device.h"
#include <iostream>
// #include <iomanip> 

// using namespace std;
enum {
  MAX_WARN_IDX_OUT_OF_BOUND = 10,
};

bool 
is_loc (const std::vector<int> & idx, 
	const std::vector<int> & nat_stt,
	const std::vector<int> & nat_end)
{
  bool ret = true;
  for (int dd = 0; dd < 3; ++dd) ret = ret && idx[dd] >= nat_stt[dd];
  for (int dd = 0; dd < 3; ++dd) ret = ret && idx[dd] <  nat_end[dd];
  return ret;
}

int 
collapse_index (const std::vector<int> &	idx,
		const std::vector<int> &	size)
{
  return (idx[0] * size[1] + idx[1]) * size[2] + idx[2];
}

void
expand_index (std::vector<int > &		o_idx,
	      const int &		i_idx,
	      const std::vector<int> &	size)
{
  int tmp1 = i_idx / size[2];
  o_idx[2] = i_idx - tmp1 * size[2];
  o_idx[0] = tmp1 / size[1];
  o_idx[1] = tmp1 - o_idx[0] * size[1];
}

void 
build_clist (std::vector<std::vector<int > > &	clist,
	     const std::vector<double > &	coord,
	     const int &		nloc,
	     const std::vector<int > &	nat_stt,
	     const std::vector<int > &	nat_end,
	     const std::vector<int > &	ext_stt,
	     const std::vector<int > &	ext_end,
	     const SimulationRegion<double> & region,
	     const std::vector<int > &	global_grid)
{
  static int count_warning_loc_idx_lower = 0;
  static int count_warning_loc_idx_upper = 0;
  static int count_warning_ghost_idx_lower = 0;
  static int count_warning_ghost_idx_upper = 0;
  // compute region info, in terms of internal coord
  int nall = coord.size() / 3;
  std::vector<int> ext_ncell(3);
  for (int dd = 0; dd < 3; ++dd) ext_ncell[dd] = ext_end[dd] - ext_stt[dd];
  int ncell = ext_ncell[0] * ext_ncell[1] * ext_ncell[2];
  std::vector<double > cell_size (3);
  for (int dd = 0; dd < 3; ++dd) cell_size[dd] = 1./global_grid[dd];
  std::vector<double > nat_orig(3);
  for (int dd = 0; dd < 3; ++dd) nat_orig[dd] = nat_stt[dd] * cell_size[dd];
  std::vector<int> idx_orig_shift(3);
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
    std::vector<int > idx(3);
    for (int dd = 0; dd < 3; ++dd){
      idx[dd] = (inter[dd] - nat_orig[dd]) / cell_size[dd];
      if (inter[dd] - nat_orig[dd] < 0.) idx[dd] --;
      if (idx[dd] < nat_stt[dd]) {
	if (count_warning_loc_idx_lower < MAX_WARN_IDX_OUT_OF_BOUND) {
	  std::cerr << "# warning: loc idx out of lower bound (ignored if warned for more than " << MAX_WARN_IDX_OUT_OF_BOUND << " times) " << std::endl;
	  count_warning_loc_idx_lower ++;
	}	
	idx[dd] = nat_stt[dd];
      }
      else if (idx[dd] >= nat_end[dd]) {
	if (count_warning_loc_idx_upper < MAX_WARN_IDX_OUT_OF_BOUND) {
	  std::cerr << "# warning: loc idx out of upper bound (ignored if warned for more than " << MAX_WARN_IDX_OUT_OF_BOUND << " times) " << std::endl;
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
    std::vector<int > idx(3);
    for (int dd = 0; dd < 3; ++dd){
      idx[dd] = (inter[dd] - nat_orig[dd]) / cell_size[dd];
      if (inter[dd] - nat_orig[dd] < 0.) idx[dd] --;
      if (idx[dd] < ext_stt[dd]) {
	if (count_warning_ghost_idx_lower < MAX_WARN_IDX_OUT_OF_BOUND &&
	    fabs((inter[dd] - nat_orig[dd]) - (ext_stt[dd] * cell_size[dd]))
	    > fabs(ext_stt[dd] * cell_size[dd]) * std::numeric_limits<double>::epsilon() * 5.
	    ) {
	  std::cerr << "# warning: ghost idx out of lower bound (ignored if warned for more than " << MAX_WARN_IDX_OUT_OF_BOUND << " times) " << std::endl;
	  count_warning_ghost_idx_lower ++;
	}
	idx[dd] = ext_stt[dd];
      }
      else if (idx[dd] >= ext_end[dd]) {
	if (count_warning_ghost_idx_upper < MAX_WARN_IDX_OUT_OF_BOUND) {
	  std::cerr << "# warning: ghost idx out of upper bound (ignored if warned for more than " << MAX_WARN_IDX_OUT_OF_BOUND << " times) " << std::endl;
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
build_clist (std::vector<std::vector<int > > &	clist,
	     const std::vector<double > &	coord,
	     const std::vector<int>  &	sel,
	     const std::vector<int > &	nat_stt,
	     const std::vector<int > &	nat_end,
	     const SimulationRegion<double> & region)
{
  static int count_warning_loc_idx_lower = 0;
  static int count_warning_loc_idx_upper = 0;
  // compute region info, in terms of internal coord
  int nall = coord.size() / 3;
  std::vector<int> nat_ncell(3);
  for (int dd = 0; dd < 3; ++dd) nat_ncell[dd] = nat_end[dd] - nat_stt[dd];
  int ncell = nat_ncell[0] * nat_ncell[1] * nat_ncell[2];
  std::vector<double > cell_size (3);
  for (int dd = 0; dd < 3; ++dd) cell_size[dd] = 1./nat_end[dd];
  std::vector<double > nat_orig(3);
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
    std::vector<int > idx(3);
    for (int dd = 0; dd < 3; ++dd){
      idx[dd] = (inter[dd] - nat_orig[dd]) / cell_size[dd];
      if (inter[dd] - nat_orig[dd] < 0.) idx[dd] --;
      if (idx[dd] < nat_stt[dd]) {
	if (count_warning_loc_idx_lower < MAX_WARN_IDX_OUT_OF_BOUND) {
	  std::cerr << "# warning: loc idx out of lower bound (ignored if warned for more than " << MAX_WARN_IDX_OUT_OF_BOUND << " times) " << std::endl;
	  count_warning_loc_idx_lower ++;
	}	
	idx[dd] = nat_stt[dd];
      }
      else if (idx[dd] >= nat_end[dd]) {
	if (count_warning_loc_idx_upper < MAX_WARN_IDX_OUT_OF_BOUND) {
	  std::cerr << "# warning: loc idx out of upper bound (ignored if warned for more than " << MAX_WARN_IDX_OUT_OF_BOUND << " times) " << std::endl;
	  count_warning_loc_idx_upper ++;
	}	
	idx[dd] = nat_end[dd] - 1;
      }
    }
    clist[collapse_index(idx, nat_ncell)].push_back (ii);
  }
}


void
build_nlist_cell (std::vector<std::vector<int> > &	nlist0,
		  std::vector<std::vector<int> > &	nlist1,
		  const int &			cidx,
		  const int &			tidx, 
		  const std::vector<std::vector<int > > &	clist,
		  const std::vector<double > &	coord,
		  const double &		rc02,
		  const double &		rc12,
		  const std::vector<int> &		shift = {0, 0, 0},
		  const std::vector<double > &	boxt = {0., 0., 0., 0., 0., 0., 0., 0., 0.})
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
      double r2 = deepmd::dot3(diff, diff);
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
build_nlist_cell (std::vector<std::vector<int> > &	nlist0,
		  std::vector<std::vector<int> > &	nlist1,
		  const int &			cidx,
		  const int &			tidx, 
		  const std::vector<std::vector<int > > &	clist0,
		  const std::vector<std::vector<int > > &	clist1,
		  const std::vector<double > &	coord,
		  const double &		rc02,
		  const double &		rc12,
		  const std::vector<int> &		shift = {0, 0, 0},
		  const std::vector<double > &	boxt = {0., 0., 0., 0., 0., 0., 0., 0., 0.})
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
      double r2 = deepmd::dot3(diff, diff);
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
build_nlist (std::vector<std::vector<int > > &	nlist0,
	     std::vector<std::vector<int > > &	nlist1,
	     const std::vector<double > &	coord,
	     const int &		nloc,
	     const double &		rc0,
	     const double &		rc1,
	     const std::vector<int > &	nat_stt_,
	     const std::vector<int > &	nat_end_,
	     const std::vector<int > &	ext_stt_,
	     const std::vector<int > &	ext_end_,
	     const SimulationRegion<double> & region,
	     const std::vector<int > &	global_grid)
{
  // normalize the index
  // i require that the ext_stt = {0, 0, 0}
  std::vector<int > nat_stt (nat_stt_);
  std::vector<int > nat_end (nat_end_);
  std::vector<int > ext_stt (ext_stt_);
  std::vector<int > ext_end (ext_end_);
  
  // compute the clist
  std::vector<std::vector<int > > clist;
  build_clist (clist, coord, nloc, nat_stt, nat_end, ext_stt, ext_end, region, global_grid);

  // compute the region info
  int nall = coord.size() / 3;
  std::vector<int> ext_ncell(3);
  for (int dd = 0; dd < 3; ++dd) ext_ncell[dd] = ext_end[dd] - ext_stt[dd];

  // compute number of iter according to the cut-off
  assert (rc0 <= rc1);
  std::vector<int> niter (3);
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
  std::vector<int> idx_orig_shift(3);
  for (int dd = 0; dd < 3; ++dd) idx_orig_shift[dd] = nat_stt[dd] - ext_stt[dd];

  // compute the nlists
  double rc02 = 0;
  if (rc0 > 0) rc02 = rc0 * rc0;
  double rc12 = rc1 * rc1;
  std::vector<int> cidx(3);
  for (cidx[0] = nat_stt[0]; cidx[0] < nat_end[0]; ++cidx[0]){
    for (cidx[1] = nat_stt[1]; cidx[1] < nat_end[1]; ++cidx[1]){
      for (cidx[2] = nat_stt[2]; cidx[2] < nat_end[2]; ++cidx[2]){
	std::vector<int> mcidx(3);
	for (int dd = 0; dd < 3; ++dd) mcidx[dd] = cidx[dd] + idx_orig_shift[dd];
	int clp_cidx = collapse_index (mcidx, ext_ncell);
	std::vector<int> tidx(3);
	for (tidx[0] = cidx[0] - niter[0]; tidx[0] < cidx[0] + niter[0] + 1; ++tidx[0]) {
	  for (tidx[1] = cidx[1] - niter[1]; tidx[1] < cidx[1] + niter[1] + 1; ++tidx[1]) {
	    for (tidx[2] = cidx[2] - niter[2]; tidx[2] < cidx[2] + niter[2] + 1; ++tidx[2]) {
	      std::vector<int> mtidx(3);
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
build_nlist (std::vector<std::vector<int > > &	nlist0,
	     std::vector<std::vector<int > > &	nlist1,
	     const std::vector<double > &	coord,
	     const double &		rc0,
	     const double &		rc1,
	     const std::vector<int > &	grid,
	     const SimulationRegion<double> & region)
{
  // assuming nloc == nall
  int nloc = coord.size() / 3;
  // compute the clist
  std::vector<int> nat_stt(3, 0);
  std::vector<int> nat_end(grid);
  std::vector<std::vector<int > > clist;
  build_clist (clist, coord, nloc, nat_stt, nat_end, nat_stt, nat_end, region, nat_end);
  
  // compute the region info
  int nall = coord.size() / 3;
  std::vector<int> nat_ncell(3);
  for (int dd = 0; dd < 3; ++dd) nat_ncell[dd] = nat_end[dd] - nat_stt[dd];

  // compute number of iter according to the cut-off
  assert (rc0 <= rc1);
  std::vector<int> niter (3);
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
  std::vector<double> phys_cs(9);
  for (int dd = 0; dd < 9; ++dd) phys_cs[dd] = region.getBoxTensor()[dd];

  // compute the nlists
  double rc02 = 0;
  if (rc0 > 0) rc02 = rc0 * rc0;
  double rc12 = rc1 * rc1;

#ifdef HALF_NEIGHBOR_LIST
  std::vector<int> cidx(3);
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
    std::vector<int> cidx(3);
    cidx[0] = nat_stt[0] + tmpidx / (idx_range[1] * idx_range[2]);
    int tmpidx1 = tmpidx - cidx[0] * idx_range[1] * idx_range[2];
    cidx[1] = nat_stt[1] + tmpidx1 / idx_range[2];
    cidx[2] = nat_stt[2] + tmpidx1 - cidx[1] * idx_range[2];
    {
      {
#endif
	int clp_cidx = collapse_index (cidx, nat_ncell);
	std::vector<int> tidx(3);
	std::vector<int> stidx(3);
	std::vector<int> shift(3);
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
build_nlist (std::vector<std::vector<int > > &	nlist0,
	     std::vector<std::vector<int > > &	nlist1,
	     const std::vector<double > &	coord,
	     const std::vector<int> &	sel0,
	     const std::vector<int> &	sel1,
	     const double &		rc0,
	     const double &		rc1,
	     const std::vector<int > &	grid,
	     const SimulationRegion<double> & region)
{
  int nloc = coord.size() / 3;
  // compute the clist
  std::vector<int> nat_stt(3, 0);
  std::vector<int> nat_end(grid);
  std::vector<std::vector<int > > clist0, clist1;
  build_clist (clist0, coord, sel0, nat_stt, nat_end, region);
  build_clist (clist1, coord, sel1, nat_stt, nat_end, region);
  
  // compute the region info
  int nall = coord.size() / 3;
  std::vector<int> nat_ncell(3);
  for (int dd = 0; dd < 3; ++dd) nat_ncell[dd] = nat_end[dd] - nat_stt[dd];

  // compute number of iter according to the cut-off
  assert (rc0 <= rc1);
  std::vector<int> niter (3);
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
  std::vector<double> phys_cs(9);
  for (int dd = 0; dd < 9; ++dd) phys_cs[dd] = region.getBoxTensor()[dd];

  // compute the nlists
  double rc02 = 0;
  if (rc0 > 0) rc02 = rc0 * rc0;
  double rc12 = rc1 * rc1;
  std::vector<int> cidx(3);
  for (cidx[0] = nat_stt[0]; cidx[0] < nat_end[0]; ++cidx[0]){
    for (cidx[1] = nat_stt[1]; cidx[1] < nat_end[1]; ++cidx[1]){
      for (cidx[2] = nat_stt[2]; cidx[2] < nat_end[2]; ++cidx[2]){
	int clp_cidx = collapse_index (cidx, nat_ncell);
	std::vector<int> tidx(3);
	std::vector<int> stidx(3);
	std::vector<int> shift(3);
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


void
build_nlist (std::vector<std::vector<int > > & nlist0,
	     std::vector<std::vector<int > > & nlist1,
	     const std::vector<double > & posi3,
	     const double & rc0_,
	     const double & rc1_,
	     const SimulationRegion<double > * region)
{
  double rc0 (rc0_);
  double rc1 (rc1_);
  assert (rc0 <= rc1);
  double rc02 = rc0 * rc0;
  // negative rc0 means not applying rc0
  if (rc0 < 0) rc02 = 0;
  double rc12 = rc1 * rc1;

  unsigned natoms = posi3.size()/3;
  nlist0.clear();
  nlist1.clear();
  nlist0.resize(natoms);
  nlist1.resize(natoms);
  for (unsigned ii = 0; ii < natoms; ++ii){
    nlist0[ii].reserve (60);
    nlist1[ii].reserve (60);
  }
  for (unsigned ii = 0; ii < natoms; ++ii){
    for (unsigned jj = ii+1; jj < natoms; ++jj){
      double diff[3];
      if (region != NULL) {
	region->diffNearestNeighbor (posi3[jj*3+0], posi3[jj*3+1], posi3[jj*3+2],
				     posi3[ii*3+0], posi3[ii*3+1], posi3[ii*3+2],
				     diff[0], diff[1], diff[2]);
      }
      else {
	diff[0] = posi3[jj*3+0] - posi3[ii*3+0];
	diff[1] = posi3[jj*3+1] - posi3[ii*3+1];
	diff[2] = posi3[jj*3+2] - posi3[ii*3+2];
      }
      double r2 = deepmd::dot3(diff, diff);
      if (r2 < rc02) {
	nlist0[ii].push_back (jj);
	nlist0[jj].push_back (ii);
      }
      else if (r2 < rc12) {
	nlist1[ii].push_back (jj);
	nlist1[jj].push_back (ii);
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
copy_coord (std::vector<double > & out_c, 
	    std::vector<int > & out_t, 
	    std::vector<int > & mapping,
	    std::vector<int> & ncell,
	    std::vector<int> & ngcell,
	    const std::vector<double > & in_c,
	    const std::vector<int > & in_t,
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
  std::vector<std::vector<int > > clist;
  std::vector<int> nat_stt(3, 0);
  build_clist (clist, in_c, nloc, nat_stt, ncell, nat_stt, ncell, region, ncell);

  // copy local atoms
  out_c.resize(nloc * 3);
  out_t.resize(nloc);
  mapping.resize(nloc);
  copy(in_c.begin(), in_c.end(), out_c.begin());
  copy(in_t.begin(), in_t.end(), out_t.begin());
  for (int ii = 0; ii < nloc; ++ii) mapping[ii] = ii;

  // push ghost
  std::vector<int> ii(3), jj(3), pbc_shift(3, 0);
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
	std::vector<int> & cur_clist = clist[cell_idx];
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
	  //   std::cout << out_c.size()  / 3 << " "
	  // 	 << inter[0] << " " 
	  // 	 << inter[1] << " " 
	  // 	 << inter[2] << " " 
	  // 	 << std::endl;
	  //   std::cout << "err here inner" << std::endl;
	  //   exit(1);
	  // }	  
	}
      }
    }
  }
}

using namespace deepmd;

void
deepmd::
convert_nlist(
    InputNlist & to_nlist,
    std::vector<std::vector<int> > & from_nlist
    )
{
  to_nlist.inum = from_nlist.size();
  for(int ii = 0; ii < to_nlist.inum; ++ii){
    to_nlist.ilist[ii] = ii;
    to_nlist.numneigh[ii] = from_nlist[ii].size();
    to_nlist.firstneigh[ii] = &from_nlist[ii][0];
  }
}

int
deepmd::
max_numneigh(
    const InputNlist & nlist
    )
{
  int max_num = 0;
  for(int ii = 0; ii < nlist.inum; ++ii){
    if(nlist.numneigh[ii] > max_num) max_num = nlist.numneigh[ii];
  }
  return max_num;
}

template <typename FPTYPE>
int
deepmd::
build_nlist_cpu(
    InputNlist & nlist,
    int * max_list_size,
    const FPTYPE * c_cpy,
    const int & nloc, 
    const int & nall, 
    const int & mem_size_,
    const float & rcut)
{
  const int mem_size = mem_size_;
  *max_list_size = 0;
  nlist.inum = nloc;
  FPTYPE rcut2 = rcut * rcut;  
  std::vector<int> jlist;
  jlist.reserve(mem_size);  
  for(int ii = 0; ii < nlist.inum; ++ii){
    nlist.ilist[ii] = ii;
    jlist.clear();
    for(int jj = 0; jj < nall; ++jj){
      if(jj == ii) continue;
      FPTYPE diff[3];
      for(int dd = 0; dd < 3; ++dd){
	diff[dd] = c_cpy[ii*3+dd] - c_cpy[jj*3+dd];
      }
      FPTYPE diff2 = deepmd::dot3(diff, diff);
      if(diff2 < rcut2){
	jlist.push_back(jj);
      }
    }
    if(jlist.size() > mem_size){
      *max_list_size = jlist.size();
      return 1;      
    }
    else {
      int list_size = jlist.size();
      nlist.numneigh[ii] = list_size;
      if(list_size > *max_list_size) *max_list_size = list_size;
      std::copy(jlist.begin(), jlist.end(), nlist.firstneigh[ii]);
    }
  }
  return 0;
}

template
int
deepmd::
build_nlist_cpu<double>(
    InputNlist & nlist,
    int * max_list_size,
    const double * c_cpy,
    const int & nloc, 
    const int & nall, 
    const int & mem_size,
    const float & rcut);

template
int
deepmd::
build_nlist_cpu<float>(
    InputNlist & nlist,
    int * max_list_size,
    const float * c_cpy,
    const int & nloc, 
    const int & nall, 
    const int & mem_size,
    const float & rcut);

#if GOOGLE_CUDA
void deepmd::convert_nlist_gpu_cuda(
    InputNlist & gpu_nlist,
    InputNlist & cpu_nlist,
    int* & gpu_memory,
    const int & max_nbor_size)
{
  const int inum = cpu_nlist.inum;
  gpu_nlist.inum = inum;
  malloc_device_memory(gpu_nlist.ilist, inum);
  malloc_device_memory(gpu_nlist.numneigh, inum);
  malloc_device_memory(gpu_nlist.firstneigh, inum);
  memcpy_host_to_device(gpu_nlist.ilist, cpu_nlist.ilist, inum);
  memcpy_host_to_device(gpu_nlist.numneigh, cpu_nlist.numneigh, inum);
  int ** _firstneigh = NULL;
  _firstneigh = (int**)malloc(sizeof(int*) * inum);
  for (int ii = 0; ii < inum; ii++) {
    memcpy_host_to_device(gpu_memory + ii * max_nbor_size, cpu_nlist.firstneigh[ii], cpu_nlist.numneigh[ii]);
    _firstneigh[ii] = gpu_memory + ii * max_nbor_size;
  }
  memcpy_host_to_device(gpu_nlist.firstneigh, _firstneigh, inum);
  free(_firstneigh);
}

void deepmd::free_nlist_gpu_cuda(
    InputNlist & gpu_nlist)
{
  delete_device_memory(gpu_nlist.ilist);
  delete_device_memory(gpu_nlist.numneigh);
  delete_device_memory(gpu_nlist.firstneigh);
}
#endif // GOOGLE_CUDA
