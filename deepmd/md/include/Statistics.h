#pragma once

#include <vector>
#include "SimulationRegion.h"
using namespace std;

template <typename VALUETYPE>
class Statistics
{
public:
  Statistics (const VALUETYPE e_corr = 0,
	      const VALUETYPE p_corr = 0);
  void record (const VALUETYPE & ener,
	       const vector<VALUETYPE > & virial,
	       const vector<VALUETYPE > & veloc,
	       const vector<VALUETYPE > & mass, 
	       const SimulationRegion<VALUETYPE > & region);
public:
  double get_T () const;
  double get_V () const;
  double get_P () const;
  double get_E () const {return get_ekin() + get_epot();}; 
  double get_ekin () const {return r_kin_ener;}
  double get_epot () const {return r_pot_ener + e_corr;}
public:
  void print (ostream & os,
	      const int & step,
	      const double time) const;
  void print_head (ostream & os) const;
private: 
  int natoms;
  double r_ener;
  double r_pot_ener;
  double r_kin_ener;
  // vector<double> r_box;
  SimulationRegion<double > region;
  vector<double> r_vir;
  double e_corr;
  double p_corr;
}
    ;

