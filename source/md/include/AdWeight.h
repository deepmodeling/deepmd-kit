// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <vector>

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else
typedef float VALUETYPE;
#endif

class AdWeight {
 public:
  AdWeight(const VALUETYPE& pl);
  virtual void zone_tag(std::vector<int>& tag,
                        const std::vector<VALUETYPE>& coord) const = 0;
  virtual void atom_weight(std::vector<VALUETYPE>& weight,
                           std::vector<VALUETYPE>& weight_x,
                           const std::vector<VALUETYPE>& coord) const = 0;
  virtual std::vector<VALUETYPE> get_center() const = 0;
  void sel_nn_atom(std::vector<VALUETYPE>& nn_coord,
                   std::vector<int>& nn_type,
                   std::vector<int>& nn_idx,
                   std::vector<int>& nn_tag,
                   const std::vector<VALUETYPE>& dcoord,
                   const std::vector<int>& dtype) const;
  void force_intpl(std::vector<VALUETYPE>& of,
                   const std::vector<VALUETYPE>& dcoord,
                   const std::vector<VALUETYPE>& ff_force,
                   const std::vector<VALUETYPE>& nn_force,
                   const std::vector<int>& nn_idx) const;
  void force_intpl(std::vector<VALUETYPE>& of,
                   const std::vector<VALUETYPE>& dcoord,
                   const std::vector<VALUETYPE>& ff_bd_force,
                   const std::vector<VALUETYPE>& ff_nb_force,
                   const std::vector<VALUETYPE>& nn_force,
                   const std::vector<int>& nn_idx) const;

 private:
  VALUETYPE protect_level;
};

// slab model, axis x
class SlabWeight : public AdWeight {
 public:
  SlabWeight(const std::vector<VALUETYPE>& box,
             const VALUETYPE& rnn,
             const VALUETYPE& rhy,
             const VALUETYPE& rc,
             const VALUETYPE& protect_level = 1e-3);
  virtual void zone_tag(std::vector<int>& tag,
                        const std::vector<VALUETYPE>& coord) const;
  virtual void atom_weight(std::vector<VALUETYPE>& weight,
                           std::vector<VALUETYPE>& weight_x,
                           const std::vector<VALUETYPE>& coord) const;
  virtual std::vector<VALUETYPE> get_center() const { return center; }

 private:
  std::vector<VALUETYPE> center;
  VALUETYPE rnn;
  VALUETYPE rhy;
  VALUETYPE rc;
};
