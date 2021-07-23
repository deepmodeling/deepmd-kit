#pragma once

#include <vector>

using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else 
typedef float  VALUETYPE;
#endif

class AdWeight 
{
public:  
  AdWeight (const VALUETYPE & pl);
  virtual void 
  zone_tag (vector<int > & tag,
	    const vector<VALUETYPE> & coord) const = 0;
  virtual void
  atom_weight (vector<VALUETYPE > & weight,
	       vector<VALUETYPE > & weight_x,
	       const vector<VALUETYPE> & coord) const = 0;
  virtual vector<VALUETYPE> 
  get_center () const = 0;
  void 
  sel_nn_atom (vector<VALUETYPE> & nn_coord,
	       vector<int> & nn_type,
	       vector<int> & nn_idx,
	       vector<int> & nn_tag,
	       const vector<VALUETYPE> & dcoord,
	       const vector<int> & dtype) const;
  void 
  force_intpl (vector<VALUETYPE> & of,
	       const vector<VALUETYPE> & dcoord,
	       const vector<VALUETYPE> & ff_force,
	       const vector<VALUETYPE> & nn_force,
	       const vector<int> & nn_idx) const;
  void 
  force_intpl (vector<VALUETYPE> & of,
	       const vector<VALUETYPE> & dcoord,
	       const vector<VALUETYPE> & ff_bd_force,
	       const vector<VALUETYPE> & ff_nb_force,
	       const vector<VALUETYPE> & nn_force,
	       const vector<int> & nn_idx) const;
 private :
  VALUETYPE protect_level;
}
    ;


// slab model, axis x
class SlabWeight : public AdWeight
{
public:
  SlabWeight (const vector<VALUETYPE> & box,
	      const VALUETYPE & rnn,
	      const VALUETYPE & rhy,
	      const VALUETYPE & rc, 
	      const VALUETYPE & protect_level = 1e-3);
  virtual void 
  zone_tag (vector<int > & tag,
	    const vector<VALUETYPE> & coord) const;
  virtual void
  atom_weight (vector<VALUETYPE > & weight,
	       vector<VALUETYPE > & weight_x,
	       const vector<VALUETYPE> & coord) const;
  virtual vector<VALUETYPE> 
  get_center () const {return center;}
private:
  vector<VALUETYPE> center;
  VALUETYPE rnn;
  VALUETYPE rhy;
  VALUETYPE rc;
}
    ;


