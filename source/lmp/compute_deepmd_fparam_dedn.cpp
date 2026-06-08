// SPDX-License-Identifier: LGPL-3.0-or-later
#include "compute_deepmd_fparam_dedn.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "comm.h"
#include "compute.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "input.h"
#include "modify.h"
#include "update.h"
#include "variable.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeDeepmdFparamDedn::ComputeDeepmdFparamDedn(LAMMPS* lmp,
                                                 int narg,
                                                 char** arg)
    : Compute(lmp, narg, arg), source_index(-1), delta(1.0e-6), pair(nullptr) {
  if (narg < 4) {
    error->all(FLERR, "Illegal compute deepmd/fparam/dedn command");
  }

  std::string token = arg[3];
  auto lb = token.find('[');
  auto rb = token.find(']');
  if (lb != std::string::npos || rb != std::string::npos) {
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb + 1 ||
        rb != token.size() - 1) {
      error->all(FLERR, "Illegal source specification in compute command");
    }
    std::string idx = token.substr(lb + 1, rb - lb - 1);
    char* endptr = nullptr;
    errno = 0;
    long one_based = std::strtol(idx.c_str(), &endptr, 10);
    if (endptr == idx.c_str() || *endptr != '\0' || errno == ERANGE ||
        one_based < 1 ||
        one_based > static_cast<long>(std::numeric_limits<int>::max()) + 1L) {
      error->all(FLERR, "Source index must be a positive 1-based integer");
    }
    source_index = static_cast<int>(one_based - 1);
    token = token.substr(0, lb);
  }

  if (token.rfind("v_", 0) == 0) {
    source_type = SRC_VAR;
    source_id = token.substr(2);
    if (source_index >= 0) {
      error->all(FLERR,
                 "Variable source for compute deepmd/fparam/dedn must be "
                 "scalar");
    }
  } else if (token.rfind("c_", 0) == 0) {
    source_type = SRC_COMPUTE;
    source_id = token.substr(2);
  } else if (token.rfind("f_", 0) == 0) {
    source_type = SRC_FIX;
    source_id = token.substr(2);
  } else {
    error->all(FLERR, "Source must be a variable, compute, or fix reference");
  }

  int iarg = 4;
  if (iarg < narg) {
    delta = atof(arg[iarg]);
    ++iarg;
  }
  if (iarg != narg) {
    error->all(FLERR, "Illegal compute deepmd/fparam/dedn command");
  }
  if (delta <= 0.0) {
    error->all(FLERR, "delta must be > 0 in compute deepmd/fparam/dedn");
  }

  scalar_flag = 1;
  extscalar = 1;
  timeflag = 1;
}

/* ---------------------------------------------------------------------- */

ComputeDeepmdFparamDedn::~ComputeDeepmdFparamDedn() = default;

/* ---------------------------------------------------------------------- */

void ComputeDeepmdFparamDedn::init() {
  if (!force->pair) {
    error->all(FLERR,
               "compute deepmd/fparam/dedn requires an active pair style");
  }
  pair = dynamic_cast<PairDeepMD*>(force->pair);
  if (!pair) {
    error->all(
        FLERR,
        "compute deepmd/fparam/dedn currently requires pair_style deepmd");
  }
  if (pair->get_dim_fparam() != 1) {
    error->all(FLERR,
               "compute deepmd/fparam/dedn currently supports a single "
               "frame-parameter dimension only");
  }
}

/* ---------------------------------------------------------------------- */

double ComputeDeepmdFparamDedn::get_source_value() {
  if (source_type == SRC_VAR) {
    int ivar = input->variable->find(source_id.c_str());
    if (ivar < 0) {
      error->all(FLERR, "Variable source not found: " + source_id);
    }
    return input->variable->compute_equal(ivar);
  }

  if (source_type == SRC_COMPUTE) {
    int icompute = modify->find_compute(source_id);
    if (icompute < 0) {
      error->all(FLERR, "Compute source not found: " + source_id);
    }
    Compute* compute = modify->compute[icompute];
    if (!compute) {
      error->all(FLERR, "Compute source not found: " + source_id);
    }
    if (source_index < 0) {
      if (!compute->scalar_flag) {
        error->all(FLERR, "Compute source is not scalar: " + source_id);
      }
      if (!(compute->invoked_flag & Compute::INVOKED_SCALAR)) {
        compute->compute_scalar();
        compute->invoked_flag |= Compute::INVOKED_SCALAR;
      }
      return compute->scalar;
    }
    if (!compute->vector_flag) {
      error->all(FLERR, "Compute source is not vector-valued: " + source_id);
    }
    if (!(compute->invoked_flag & Compute::INVOKED_VECTOR)) {
      compute->compute_vector();
      compute->invoked_flag |= Compute::INVOKED_VECTOR;
    }
    if (source_index >= compute->size_vector) {
      error->all(FLERR, "Compute source index is out of range: " + source_id);
    }
    return compute->vector[source_index];
  }

  int ifix = modify->find_fix(source_id);
  if (ifix < 0) {
    error->all(FLERR, "Fix source not found: " + source_id);
  }
  Fix* fix = modify->fix[ifix];
  if (!fix) {
    error->all(FLERR, "Fix source not found: " + source_id);
  }
  if (source_index < 0) {
    if (!fix->scalar_flag) {
      error->all(FLERR, "Fix source is not scalar: " + source_id);
    }
    return fix->compute_scalar();
  }
  if (!fix->vector_flag) {
    error->all(FLERR, "Fix source is not vector-valued: " + source_id);
  }
  if (fix->size_vector <= source_index) {
    error->all(FLERR, "Fix source index is out of range: " + source_id);
  }
  return fix->compute_vector(source_index);
}

/* ---------------------------------------------------------------------- */

double ComputeDeepmdFparamDedn::compute_scalar() {
  invoked_scalar = update->ntimestep;
  int dim = 0;
  if (void* ptr = pair->extract("deepmd_dedn", dim)) {
    scalar = *static_cast<double*>(ptr);
    return scalar;
  }

  double fparam0 = get_source_value();
  std::vector<double> fparam_plus(1, fparam0 + delta);
  std::vector<double> fparam_minus(1, fparam0 - delta);

  double one = (pair->eval_energy_with_fparam(fparam_plus) -
                pair->eval_energy_with_fparam(fparam_minus)) /
               (2.0 * delta);

  MPI_Allreduce(&one, &scalar, 1, MPI_DOUBLE, MPI_SUM, world);
  return scalar;
}
