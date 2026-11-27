// SPDX-License-Identifier: LGPL-3.0-or-later
#include "model_deviation.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>

#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"

using namespace LAMMPS_NS;

namespace {
// Width of a column of the output file.
constexpr int kColumn = 18;

// Maximum, minimum and average over all atoms of a per-atom deviation, on
// rank 0. A rank without atoms contributes no extremum.
void reduce_deviation(const std::vector<double>& deviation,
                      const bigint natoms,
                      MPI_Comm world,
                      double reduced[3]) {
  double max = 0.0;
  double min = std::numeric_limits<double>::max();
  double sum = 0.0;
  for (const double value : deviation) {
    max = std::max(max, value);
    min = std::min(min, value);
    sum += value;
  }
  MPI_Reduce(&max, &reduced[0], 1, MPI_DOUBLE, MPI_MAX, 0, world);
  MPI_Reduce(&min, &reduced[1], 1, MPI_DOUBLE, MPI_MIN, 0, world);
  MPI_Reduce(&sum, &reduced[2], 1, MPI_DOUBLE, MPI_SUM, 0, world);
  reduced[2] /= static_cast<double>(natoms);
}
}  // namespace

ModelDeviation::ModelDeviation(LAMMPS* lmp)
    : Pointers(lmp),
      nmodels(1),
      magnetic(false),
      ener_unit(1.0),
      force_unit(1.0),
      evaluated_builds(-1) {
  reset();
}

bool ModelDeviation::is_option(const std::string& word) {
  return word == "out_freq" || word == "out_file" || word == "atomic" ||
         word == "relative" || word == "relative_v";
}

void ModelDeviation::reset() {
  out_freq = 100;
  out_file = "model_devi.out";
  out_each = false;
  out_rel = false;
  out_rel_v = false;
  eps = 0.0;
  eps_v = 0.0;
}

int ModelDeviation::parse(int narg,
                          char** arg,
                          int iarg,
                          double ener_unit_cvt_factor,
                          bool (*is_key)(const std::string&)) {
  const std::string word(arg[iarg]);
  if (word == "atomic") {
    out_each = true;
    return 1;
  }
  if (!is_option(word)) {
    return 0;
  }
  // The level of a relative deviation is a number, so a keyword in its place
  // leaves the level missing.
  const bool relative = word == "relative" || word == "relative_v";
  if (iarg + 1 >= narg || (relative && is_key(arg[iarg + 1]))) {
    error->all(FLERR, "Illegal " + word + ", not provided");
  }
  if (word == "out_freq") {
    out_freq = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
    if (out_freq < 0) {
      error->all(FLERR, "Illegal out_freq, should be >= 0");
    }
  } else if (word == "out_file") {
    out_file = arg[iarg + 1];
  } else if (word == "relative") {
    out_rel = true;
    eps =
        utils::numeric(FLERR, arg[iarg + 1], false, lmp) / ener_unit_cvt_factor;
  } else {
    out_rel_v = true;
    eps_v =
        utils::numeric(FLERR, arg[iarg + 1], false, lmp) / ener_unit_cvt_factor;
  }
  return 2;
}

void ModelDeviation::setup(int nmodels_,
                           bool magnetic_,
                           bool append,
                           double ener_unit_cvt_factor,
                           double force_unit_cvt_factor) {
  nmodels = nmodels_;
  magnetic = magnetic_;
  ener_unit = ener_unit_cvt_factor;
  force_unit = force_unit_cvt_factor;
  rows_.clear();
  virials.assign(9 * static_cast<std::size_t>(nmodels), 0.0);
  if (fp.is_open()) {
    fp.close();
  }
  if (comm->me != 0 || nmodels < 2 || out_freq == 0) {
    return;
  }
  if (append) {
    fp.open(out_file, std::ofstream::out | std::ofstream::app);
    fp << std::scientific;
    return;
  }
  fp.open(out_file);
  fp << std::scientific;
  // A spin model reports the deviation of the force on the atom ("fr") and
  // of the magnetic force ("fm").
  std::vector<std::string> quantities = {"v", magnetic ? "fr" : "f"};
  if (magnetic) {
    quantities.push_back("fm");
  }
  fp << "#" << std::setw(12 - 1) << "step";
  for (const std::string& quantity : quantities) {
    for (const char* statistic : {"max", "min", "avg"}) {
      fp << std::setw(kColumn + 1)
         << std::string(statistic) + "_devi_" + quantity;
    }
  }
  if (out_each) {
    // The atom count is not known when the header is written.
    fp << std::setw(kColumn + 1) << "atm_devi_f(N)";
  }
  fp << std::endl;
}

bool ModelDeviation::due() const {
  return nmodels > 1 && out_freq > 0 && update->ntimestep % out_freq == 0;
}

int ModelDeviation::ensemble_ago() {
  // The count of neighbor builds identifies the list in use within a run;
  // unlike the step of the last build, it does not repeat after a reset of
  // the step, and start_run() forgets it when a run restarts the count.
  const int ago = evaluated_builds == neighbor->ncalls ? neighbor->ago : 0;
  evaluated_builds = neighbor->ncalls;
  return ago;
}

void ModelDeviation::store(int model,
                           const std::vector<double>& force,
                           const std::vector<double>& virial,
                           const std::vector<double>& force_mag) {
  const std::size_t nall = force.size() / 3;
  const int stride = magnetic ? 6 : 3;
  if (model == 0) {
    rows_.assign(nall * width(), 0.0);
  }
  for (std::size_t ii = 0; ii < nall; ++ii) {
    double* row = rows_.data() + ii * width() + model * stride;
    for (int dd = 0; dd < 3; ++dd) {
      row[dd] = force[3 * ii + dd];
      if (magnetic) {
        row[3 + dd] = force_mag[3 * ii + dd];
      }
    }
  }
  std::copy(virial.begin(), virial.begin() + 9, virials.begin() + 9 * model);
}

int ModelDeviation::pack_reverse(int n, int first, double* buf) const {
  const auto begin = rows_.begin() + static_cast<std::size_t>(first) * width();
  std::copy(begin, begin + static_cast<std::size_t>(n) * width(), buf);
  return n * width();
}

void ModelDeviation::unpack_reverse(int n, const int* list, const double* buf) {
  for (int ii = 0; ii < n; ++ii) {
    double* row = rows_.data() + static_cast<std::size_t>(list[ii]) * width();
    for (int kk = 0; kk < width(); ++kk) {
      row[kk] += buf[ii * width() + kk];
    }
  }
}

void ModelDeviation::write() {
  // Non-const, as the MPI stubs of a serial build take non-const buffers.
  int nlocal = atom->nlocal;
  const bigint natoms = atom->natoms;
  const int stride = magnetic ? 6 : 3;

  // === Step 1. Deviation of the force and the magnetic force of each atom ===
  // The deviation of a 3-vector over the models is the root mean square of
  // its distance from the model average, divided by the norm of that average
  // plus the level ``eps`` for a relative deviation.
  const auto atom_deviation = [&](const int offset) {
    std::vector<double> deviation(nlocal);
    for (int ii = 0; ii < nlocal; ++ii) {
      const double* row = rows_.data() + static_cast<std::size_t>(ii) * width();
      double average[3] = {0.0, 0.0, 0.0};
      for (int kk = 0; kk < nmodels; ++kk) {
        for (int dd = 0; dd < 3; ++dd) {
          average[dd] += row[kk * stride + offset + dd] / nmodels;
        }
      }
      double variance = 0.0;
      for (int kk = 0; kk < nmodels; ++kk) {
        for (int dd = 0; dd < 3; ++dd) {
          const double diff = row[kk * stride + offset + dd] - average[dd];
          variance += diff * diff;
        }
      }
      deviation[ii] = std::sqrt(variance / nmodels);
      if (out_rel) {
        deviation[ii] /=
            std::sqrt(average[0] * average[0] + average[1] * average[1] +
                      average[2] * average[2]) +
            eps;
      }
    }
    return deviation;
  };
  std::vector<double> force_deviation = atom_deviation(0);
  double force_stats[3] = {0.0, 0.0, 0.0};
  reduce_deviation(force_deviation, natoms, world, force_stats);
  double force_mag_stats[3] = {0.0, 0.0, 0.0};
  if (magnetic) {
    reduce_deviation(atom_deviation(3), natoms, world, force_mag_stats);
  }

  // === Step 2. Deviation of the virial per atom ===
  // Every rank holds its share of the virial of each model, which the
  // reduction sums.
  std::vector<double> send(virials.size());
  std::vector<double> recv(virials.size());
  for (std::size_t ii = 0; ii < virials.size(); ++ii) {
    send[ii] = virials[ii] / static_cast<double>(natoms);
  }
  MPI_Reduce(send.data(), recv.data(), static_cast<int>(send.size()),
             MPI_DOUBLE, MPI_SUM, 0, world);
  if (comm->me == 0) {
    double virial_stats[3] = {0.0, std::numeric_limits<double>::max(), 0.0};
    for (int ii = 0; ii < 9; ++ii) {
      double average = 0.0;
      for (int kk = 0; kk < nmodels; ++kk) {
        average += recv[9 * kk + ii] / nmodels;
      }
      double variance = 0.0;
      for (int kk = 0; kk < nmodels; ++kk) {
        const double diff = recv[9 * kk + ii] - average;
        variance += diff * diff;
      }
      double deviation = std::sqrt(variance / nmodels);
      if (out_rel_v) {
        deviation /= std::abs(average) + eps_v;
      }
      virial_stats[0] = std::max(virial_stats[0], deviation);
      virial_stats[1] = std::min(virial_stats[1], deviation);
      virial_stats[2] += deviation * deviation;
    }
    virial_stats[2] = std::sqrt(virial_stats[2] / 9);
    fp << std::setw(12) << update->ntimestep;
    for (const double value : virial_stats) {
      fp << " " << std::setw(kColumn) << value * ener_unit;
    }
    for (const double value : force_stats) {
      fp << " " << std::setw(kColumn) << value * force_unit;
    }
    if (magnetic) {
      // The magnetic force is an energy per unit of the moment.
      for (const double value : force_mag_stats) {
        fp << " " << std::setw(kColumn) << value * ener_unit;
      }
    }
  }

  // === Step 3. Force deviation of every atom, in tag order ===
  if (out_each) {
    const int nprocs = comm->nprocs;
    std::vector<int> counts(nprocs), displacements(nprocs, 0);
    MPI_Gather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, world);
    for (int ii = 0; ii + 1 < nprocs; ++ii) {
      displacements[ii + 1] = displacements[ii] + counts[ii];
    }
    const bool root = comm->me == 0;
    std::vector<tagint> tags(root ? natoms : 0);
    std::vector<double> deviation(root ? natoms : 0);
    MPI_Gatherv(atom->tag, nlocal, MPI_LMP_TAGINT, tags.data(), counts.data(),
                displacements.data(), MPI_LMP_TAGINT, 0, world);
    MPI_Gatherv(force_deviation.data(), nlocal, MPI_DOUBLE, deviation.data(),
                counts.data(), displacements.data(), MPI_DOUBLE, 0, world);
    if (root) {
      // Tags need not be contiguous, after the deletion of atoms for one.
      std::vector<bigint> order(natoms);
      for (bigint ii = 0; ii < natoms; ++ii) {
        order[ii] = ii;
      }
      std::sort(
          order.begin(), order.end(),
          [&](const bigint a, const bigint b) { return tags[a] < tags[b]; });
      for (const bigint ii : order) {
        fp << " " << std::setw(kColumn) << deviation[ii] * force_unit;
      }
    }
  }
  if (comm->me == 0) {
    fp << std::endl;
  }
}
