#include <iostream>
#include <fstream>
// #include <iomanip>
#include <stdio.h>

#include "Statistics.h"
#include "UnitManager.h"

template <typename VALUETYPE>
Statistics<VALUETYPE>::
Statistics (const VALUETYPE e_corr_,
	    const VALUETYPE p_corr_)
    : e_corr(e_corr_), p_corr(p_corr_)
{
}


template <typename VALUETYPE>
void
Statistics<VALUETYPE>::
record (const VALUETYPE & ener,
	const vector<VALUETYPE > & virial,
	const vector<VALUETYPE > & veloc,
	const vector<VALUETYPE > & mass, 
	const SimulationRegion<VALUETYPE > & region_)
{
  r_pot_ener = ener;
  r_vir.resize(9);
  for (unsigned ii = 0; ii < 9; ++ii){
    r_vir[ii] = virial[ii];
  }
  // r_box.resize(6);
  // for (unsigned ii = 0; ii < 6; ++ii){
  //   r_box[ii] = box[ii];
  // }
  region.reinitBox(region_.getBoxTensor());
  natoms = mass.size();
  r_kin_ener = 0;
  double pref = 0.5 * UnitManager::IntegratorMassConstant;
  for (int ii = 0; ii < natoms; ++ii){
    r_kin_ener += pref * mass[ii] * veloc[3*ii+0] * veloc[3*ii+0];
    r_kin_ener += pref * mass[ii] * veloc[3*ii+1] * veloc[3*ii+1];
    r_kin_ener += pref * mass[ii] * veloc[3*ii+2] * veloc[3*ii+2];
  }
}

template <typename VALUETYPE>
double
Statistics<VALUETYPE>::
get_T () const 
{
  return get_ekin () / (natoms * 3. * UnitManager::BoltzmannConstant) * 2.;
}

template <typename VALUETYPE>
double
Statistics<VALUETYPE>::
get_V () const 
{
  // return (r_box[1] - r_box[0]) * (r_box[3] - r_box[2]) * (r_box[5] - r_box[4]);
  return region.getVolume();
}

template <typename VALUETYPE>
double
Statistics<VALUETYPE>::
get_P () const 
{
  return (get_ekin() - (r_vir[0] + r_vir[4] + r_vir[8])) * 2./3. / get_V() * UnitManager::PressureConstant + p_corr;
}

template <typename VALUETYPE>
void
Statistics<VALUETYPE>::
print (ostream & os,
       const int & step,
       const double time) const  
{
  char tmps[65536];
  sprintf (tmps, "%13.4f %15.6f %15.6f %15.6f %15.6f %15.6f %15.6f %15.6f %15.6f\n", 
	   time,
	   get_ekin(),
	   get_epot(),
	   get_ekin() + get_epot(),
	   get_T(),
	   get_P(),
	   r_vir[0],
	   r_vir[4],
	   r_vir[8]);
  os << tmps ;
  // os << setw(7) << setprecision(6) << time << setprecision (8) << setfill (' ')
  //    << setw(15) << get_ekin() << " "
  //    << setw(15) << get_epot() << " "
  //    << setw(15) << get_ekin() + get_epot() << " "
  //    << setw(15) << get_T() << " "
  //    << setw(15) << get_P() << " "
  //    << setw(15) << r_vir[0] << " "
  //    << setw(15) << r_vir[4] << " "
  //    << setw(15) << r_vir[8] << " "
  //    << endl;
}

template <typename VALUETYPE>
void
Statistics<VALUETYPE>::
print_head (ostream & os) const  
{
  char tmps[65536];
  sprintf (tmps, "#%12s %15s %15s %15s %15s %15s %15s %15s %15s\n", 
	   "time",
	   "Kinetic",
	   "Potential",
	   "E_tot",
	   "Temperature",
	   "Pressure",
	   "Vxx",
	   "Vyy",
	   "Vzz");
  os << tmps ;
  // os << "#";
  // os << setw(6) <<  "time" << setfill (' ')
  //    << setw(15) << "Kinetic" << " "
  //    << setw(15) << "Potential" << " "
  //    << setw(15) << "E_tot" << " "
  //    << setw(15) << "Temperature" << " "
  //    << setw(15) << "Pressure" << " "
  //    << setw(15) << "Vxx" << " "
  //    << setw(15) << "Vyy" << " "
  //    << setw(15) << "Vzz" << " "
  //    << endl;
}


template class Statistics<float>;
template class Statistics<double>;

