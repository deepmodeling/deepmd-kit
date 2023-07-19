// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <map>
#include <string>
#include <vector>

template <typename VALUETYPE>
class Convert {
 public:
  Convert(const std::vector<std::string>& atomname,
          std::map<std::string, int>& name_type_map,
          std::map<std::string, VALUETYPE>& name_mass_map,
          std::map<std::string, VALUETYPE>& name_charge_map,
          const bool sort = true);
  void gro2nnp(std::vector<VALUETYPE>& coord,
               std::vector<VALUETYPE>& veloc,
               std::vector<VALUETYPE>& box,
               const std::vector<std::vector<double> >& posi,
               const std::vector<std::vector<double> >& velo,
               const std::vector<double>& box_size) const;
  void nnp2gro(std::vector<std::vector<double> >& posi,
               std::vector<std::vector<double> >& velo,
               std::vector<double>& box_size,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& veloc,
               const std::vector<VALUETYPE>& box) const;
  void idx_gro2nnp(std::vector<int>& out, const std::vector<int>& in) const;
  void idx_nnp2gro(std::vector<int>& out, const std::vector<int>& in) const;
  const std::vector<int>& get_type() const { return atype; }
  const std::vector<VALUETYPE>& get_mass() const { return amass; }
  const std::vector<VALUETYPE>& get_charge() const { return acharge; }

 private:
  std::vector<int> idx_map_nnp2gro;
  std::vector<int> idx_map_gro2nnp;
  std::vector<int> atype;
  std::vector<VALUETYPE> amass;
  std::vector<VALUETYPE> acharge;
};
