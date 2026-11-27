// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef LMP_DEEPMD_MODEL_DEVIATION_H
#define LMP_DEEPMD_MODEL_DEVIATION_H

#include <algorithm>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "error.h"
#include "lmptype.h"
#include "pointers.h"

namespace LAMMPS_NS {

// Model deviation of an ensemble of models evaluated on one configuration.
//
// The first model of the ensemble drives the dynamics at every step; the
// others are evaluated only at the steps that write the deviation, every
// ``out_freq`` steps. At such a step the pair style stores the outputs of
// every model here, in the units of the models: the force (and, for a spin
// model, the magnetic force) of every local and ghost atom, and the virial of
// this rank. The forces form one row per atom in atom order, ``width()``
// values each, the layout of the reverse communication of the pair style,
// which folds the rows of the ghost atoms onto their owners. ``write`` then
// reduces the statistics over all ranks and appends one line to the output
// file: the maximum, minimum and root mean square over its nine components of
// the deviation of the virial, the maximum, minimum and average over the atoms
// of the deviation of the force and, for a spin model, of the magnetic force,
// optionally followed by the force deviation of every atom.
class ModelDeviation : protected Pointers {
 public:
  ModelDeviation(LAMMPS* lmp);

  // Whether ``word`` names an option of the deviation, which ends the list
  // of models on the pair_style line.
  static bool is_option(const std::string& word);
  // Restore the default options before a pair_style line is parsed.
  void reset();
  // Consume the option at ``arg[iarg]``: ``out_freq``, ``out_file``,
  // ``atomic``, ``relative`` or ``relative_v``. Returns the number of words
  // consumed, or zero when ``arg[iarg]`` names another keyword. ``is_key``
  // tells the keywords of the pair style, none of which can stand for the
  // level of a relative deviation. The levels are converted to the units of
  // the models by ``ener_unit_cvt_factor``, as the deviation itself is.
  int parse(int narg,
            char** arg,
            int iarg,
            double ener_unit_cvt_factor,
            bool (*is_key)(const std::string&));
  // Size the ensemble and, for an ensemble that writes its deviation, open
  // the output file on rank 0, appending to it when the run continues a
  // restart. ``magnetic`` adds the magnetic force. The conversion factors
  // take the outputs of the models to the units of LAMMPS.
  void setup(int nmodels,
             bool magnetic,
             bool append,
             double ener_unit_cvt_factor,
             double force_unit_cvt_factor);

  // Forget the neighbor list the further models hold, at the start of every
  // run, whose neighbor builds LAMMPS counts afresh.
  void start_run() { evaluated_builds = -1; }
  // Number of models of the ensemble.
  int models() const { return nmodels; }
  // Whether the current step evaluates every model and writes the deviation.
  bool due() const;
  // Neighbor-list age to hand the further models, which keep their own copy
  // of the list and see only the steps that are due: the age of the list if
  // they have evaluated it already, zero otherwise. Records the current list
  // as one they evaluate.
  int ensemble_ago();
  // Values per atom of the stored outputs: three per model, six with the
  // magnetic force.
  int width() const { return nmodels * (magnetic ? 6 : 3); }

  // Store the outputs of model ``model`` at the current step: the force
  // with shape (nall * 3), the virial with shape (9) and, for a spin model,
  // the magnetic force with shape (nall * 3). The first model sizes the rows.
  void store(int model,
             const std::vector<double>& force,
             const std::vector<double>& virial,
             const std::vector<double>& force_mag = {});
  // The stored rows, (nall * width()) in atom order.
  std::vector<double>& rows() { return rows_; }

  // Rows [first, first + n) of the stored outputs, and the accumulation of
  // such rows onto the atoms of ``list``.
  int pack_reverse(int n, int first, double* buf) const;
  void unpack_reverse(int n, const int* list, const double* buf);

  // Reduce the deviation of the stored outputs over all ranks and append it
  // to the output file.
  void write();

 private:
  int nmodels;
  bool magnetic;
  int out_freq;
  std::string out_file;
  bool out_each;
  bool out_rel;
  bool out_rel_v;
  double eps;
  double eps_v;
  double ener_unit;
  double force_unit;
  // Neighbor builds of the run when the further models were last evaluated,
  // which identifies the list they hold within the run; -1 before their first
  // evaluation in the run.
  bigint evaluated_builds;
  std::vector<double> rows_;
  std::vector<double> virials;
  std::ofstream fp;
};

// Load the models of an ensemble, named by ``paths``: the first into
// ``first``, which drives the dynamics, and the others into ``others``.
// ``init(model, path)`` loads one model. Every model reads the same frame and
// atomic parameter arrays, so their dimensions must agree. The atom type
// mapping is checked when pair_coeff selects the elements. The ensemble
// neighbor list covers the largest cutoff; each model selects its neighbors.
template <class Model, class Init>
double load_ensemble(const std::vector<std::string>& paths,
                     Model& first,
                     std::deque<Model>& others,
                     Init init,
                     const std::string& style,
                     Error* error) {
  init(first, paths[0]);
  others.clear();
  for (std::size_t ii = 1; ii < paths.size(); ++ii) {
    others.emplace_back();
    init(others.back(), paths[ii]);
  }
  double cutoff = first.cutoff();
  for (Model& model : others) {
    if (model.dim_fparam() != first.dim_fparam() ||
        model.dim_aparam() != first.dim_aparam()) {
      error->all(FLERR, "pair style " + style +
                            ": ensemble models must have the same frame and "
                            "atomic parameter dimensions.");
    }
    cutoff = std::max(cutoff, model.cutoff());
  }
  return cutoff;
}

// Every model receives the type indices selected by pair_coeff for the first
// model. Unselected elements may differ. A model with no element names still
// has numb_types() positional types. An out-of-range index is compatible
// only if every model ignores it.
template <class Model>
void check_ensemble_types(const std::vector<int>& mapping,
                          Model& first,
                          std::deque<Model>& others,
                          Error* error) {
  if (others.empty()) {
    return;
  }
  const auto types = [](Model& model) {
    std::string text;
    model.get_type_map(text);
    std::istringstream input(text);
    std::vector<std::string> names;
    for (std::string name; input >> name;) {
      names.push_back(name);
    }
    if (names.empty()) {
      names.resize(model.numb_types());
    }
    return names;
  };
  const auto first_types = types(first);
  for (auto& model : others) {
    const auto other_types = types(model);
    for (std::size_t ii = 0; ii < mapping.size(); ++ii) {
      if (mapping[ii] < 0) {
        continue;  // NULL in pair_coeff
      }
      const std::size_t type = mapping[ii];
      const bool present = type < first_types.size();
      if (present != (type < other_types.size()) ||
          (present && first_types[type] != other_types[type])) {
        error->all(
            FLERR,
            "Incompatible model ensemble: pair_coeff maps atom type " +
                std::to_string(ii + 1) + " to model index " +
                std::to_string(type) +
                ", which must designate the same element in every model.");
      }
    }
  }
}

}  // namespace LAMMPS_NS

#endif
