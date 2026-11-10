// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef LMP_DEEPMD_MODEL_DEVIATION_H
#define LMP_DEEPMD_MODEL_DEVIATION_H

#include <algorithm>
#include <deque>
#include <fstream>
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
// ``init(model, path)`` loads one model. Every model reads the atom types
// through the type mapping of the first one and the frame and atomic
// parameters the pair style builds for it, so the models must share its type
// map and the dimensions of those parameters. Returns the largest cutoff of
// the models, in their length unit, which the neighbor list of the ensemble
// covers; each model selects the neighbors within its own cutoff.
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
    init(others.emplace_back(), paths[ii]);
  }
  std::string type_map;
  first.get_type_map(type_map);
  double cutoff = first.cutoff();
  for (Model& model : others) {
    std::string other_type_map;
    model.get_type_map(other_type_map);
    if (other_type_map != type_map ||
        model.dim_fparam() != first.dim_fparam() ||
        model.dim_aparam() != first.dim_aparam()) {
      error->all(FLERR, "pair style " + style +
                            ": the models of an ensemble share the type map "
                            "and the frame and atomic parameter dimensions of "
                            "the first model, whose mapping of the atom types "
                            "and parameters they read.");
    }
    cutoff = std::max(cutoff, model.cutoff());
  }
  return cutoff;
}

}  // namespace LAMMPS_NS

#endif
