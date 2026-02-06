// SPDX-License-Identifier: LGPL-3.0-or-later
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_cboamd.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "output.h"
#include "respa.h"
#include "universe.h"
#include "update.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixCBOAMD::FixCBOAMD(LAMMPS* lmp, int narg, char** arg)
    : Fix(lmp, narg, arg),
      model_dipole(nullptr),
      model_polar(nullptr),
      deepmd_dipole(nullptr),
      deepmd_polar(nullptr),
      type_map(nullptr),
      ntypes(0),
      dipole(nullptr),
      polarizability(nullptr),
      forces_deepmd(nullptr),
      photons_enabled(false),
      nphoton(0),
      omega_photon(nullptr),
      lambda_photon(nullptr),
      lambda_vector(nullptr),
      qa(nullptr),
      pa(nullptr),
      fa(nullptr),
      ea(nullptr),
      dt(0.0),
      current_step(0),
      output_file(nullptr) {
  vector_flag = 1;
  extvector = 1;
  size_vector = 3;

  if (narg < 4) {
    error->all(FLERR, "Illegal fix cboamd command");
  }

  // Parse command line arguments
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "dipole") == 0) {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "Illegal fix cboamd command");
      }
      model_dipole = utils::strdup(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "polar") == 0) {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "Illegal fix cboamd command");
      }
      model_polar = utils::strdup(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "photons") == 0) {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "Illegal fix cboamd command");
      }
      photons_enabled = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "nphoton") == 0) {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "Illegal fix cboamd command");
      }
      nphoton = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "omega") == 0) {
      if (iarg + 1 + nphoton > narg) {
        error->all(FLERR, "Illegal fix cboamd command");
      }
      memory->create(omega_photon, nphoton, "fix_cboamd:omega_photon");
      for (int i = 0; i < nphoton; i++) {
        omega_photon[i] = utils::numeric(FLERR, arg[iarg + 1 + i], false, lmp);
      }
      iarg += 1 + nphoton;
    } else if (strcmp(arg[iarg], "lambda") == 0) {
      if (iarg + 1 + nphoton > narg) {
        error->all(FLERR, "Illegal fix cboamd command");
      }
      memory->create(lambda_photon, nphoton, "fix_cboamd:lambda_photon");
      for (int i = 0; i < nphoton; i++) {
        lambda_photon[i] = utils::numeric(FLERR, arg[iarg + 1 + i], false, lmp);
      }
      iarg += 1 + nphoton;
    } else if (strcmp(arg[iarg], "lambda_vector") == 0) {
      if (iarg + 1 + 3 * nphoton > narg) {
        error->all(FLERR, "Illegal fix cboamd command");
      }
      memory->create(lambda_vector, nphoton, 3, "fix_cboamd:lambda_vector");
      for (int i = 0; i < nphoton; i++) {
        for (int j = 0; j < 3; j++) {
          lambda_vector[i][j] =
              utils::numeric(FLERR, arg[iarg + 1 + 3 * i + j], false, lmp);
        }
      }
      iarg += 1 + 3 * nphoton;
    } else {
      error->all(FLERR, "Unknown fix cboamd keyword: {}", arg[iarg]);
    }
  }

  // Check required parameters
  if (!model_dipole) {
    error->all(FLERR, "fix cboamd requires dipole model");
  }
  // if (!model_polar) error->all(FLERR,"fix cboamd requires polarizability
  // model");
  if (photons_enabled && nphoton <= 0) {
    error->all(FLERR, "fix cboamd: nphoton must be > 0 when photons enabled");
  }
  if (photons_enabled && !omega_photon) {
    error->all(FLERR,
               "fix cboamd: omega must be specified when photons enabled");
  }
  if (photons_enabled && !lambda_photon) {
    error->all(FLERR,
               "fix cboamd: lambda must be specified when photons enabled");
  }
  if (photons_enabled && !lambda_vector) {
    error->all(
        FLERR,
        "fix cboamd: lambda_vector must be specified when photons enabled");
  }
  // if (dt <= 0.0) error->all(FLERR,"fix cboamd: dt must be > 0");

  // Set up arrays
  ntypes = atom->ntypes;
  memory->create(type_map, ntypes + 1, "fix_cboamd:type_map");
  for (int i = 1; i <= ntypes; i++) {
    type_map[i] = i - 1;  // Default mapping: type 1 -> 0, type 2 -> 1, etc.
  }

  // Allocate arrays
  memory->create(dipole, 3, "fix_cboamd:dipole");
  memory->create(polarizability, 9, "fix_cboamd:polarizability");
  memory->create(forces_deepmd, atom->nmax * 3, "fix_cboamd:forces_deepmd");

  if (photons_enabled) {
    memory->create(qa, nphoton, "fix_cboamd:qa");
    memory->create(pa, nphoton, "fix_cboamd:pa");
    memory->create(fa, nphoton, "fix_cboamd:fa");
    memory->create(ea, nphoton, "fix_cboamd:ea");

    // Initialize photon coordinates and momenta
    for (int i = 0; i < nphoton; i++) {
      qa[i] = 0.0;
      pa[i] = 0.0;
      fa[i] = 0.0;
      ea[i] = 0.0;
    }
  }

  // Initialize DeepMD models
  init_deepmd_models();

  // Open output file
  output_file = fopen("cboamd_output.dat", "w");
  if (!output_file) {
    error->all(FLERR, "Cannot open cboamd output file");
  }
  fprintf(output_file,
          "# Step Time Energy Dipole_x Dipole_y Dipole_z Pol_xx Pol_xy Pol_xz "
          "Pol_yx Pol_yy Pol_yz Pol_zx Pol_zy Pol_zz");
  if (photons_enabled) {
    for (int i = 0; i < nphoton; i++) {
      fprintf(output_file, " qa_%d pa_%d ea_%d fa_%d", i, i, i, i);
    }
  }
  fprintf(output_file, "\n");
}

/* ---------------------------------------------------------------------- */

FixCBOAMD::~FixCBOAMD() {
  cleanup_deepmd_models();

  if (model_dipole) {
    delete[] model_dipole;
  }
  if (model_polar) {
    delete[] model_polar;
  }

  memory->destroy(type_map);
  memory->destroy(dipole);
  memory->destroy(polarizability);
  memory->destroy(forces_deepmd);

  if (photons_enabled) {
    memory->destroy(omega_photon);
    memory->destroy(lambda_photon);
    memory->destroy(lambda_vector);
    memory->destroy(qa);
    memory->destroy(pa);
    memory->destroy(fa);
    memory->destroy(ea);
  }

  if (output_file) {
    fclose(output_file);
  }
}

/* ---------------------------------------------------------------------- */

int FixCBOAMD::setmask() {
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::init() {
  dt = update->dt;
  // Check if we have the right atom style
  if (!atom->tag_enable) {
    error->all(FLERR, "fix cboamd requires atom IDs");
  }

  // Set up neighbor list
  neighbor->add_request(this);
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::setup(int vflag) {
  post_force(vflag);
  // // Initialize DeepMD calculations
  // convert_coordinates_to_deepmd_format();
  // compute_deepmd_dipole();
  // // compute_deepmd_polarizability();

  // if (photons_enabled) {
  //   compute_cboa_forces();
  // }
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::initial_integrate(int /*vflag*/) {
  for (int alpha = 0; alpha < nphoton; alpha++) {
    // Update photon coordinates and momenta using Velocity Verlet
    pa[alpha] += 0.5 * fa[alpha] * dt * PS_TO_AU;
    qa[alpha] += pa[alpha] * dt * PS_TO_AU;
  }
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::final_integrate() {
  for (int alpha = 0; alpha < nphoton; alpha++) {
    // Update photon momenta using Velocity Verlet
    pa[alpha] += 0.5 * fa[alpha] * dt * PS_TO_AU;
  }
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::post_force(int vflag) {
  // Update photon coordinates if enabled
  // if (photons_enabled) {
  // update_photon_coordinates();
  // }

  // Compute DeepMD properties
  convert_coordinates_to_deepmd_format();
  compute_deepmd_dipole();
  // compute_deepmd_polarizability();

  // Compute CBOA forces if photons enabled
  if (photons_enabled) {
    compute_cboa_forces();
  }

  // Write output
  write_output();

  current_step++;
}

/* ---------------------------------------------------------------------- */

double FixCBOAMD::compute_scalar() {
  // Return total energy
  return 0.0;  // This would need to be implemented with actual energy value
}

/* ---------------------------------------------------------------------- */

double FixCBOAMD::compute_vector(int n) {
  // Return dipole components
  // if (n >= 0 && n < 3) return dipole[n];
  for (int i = 0; i < nphoton; i++) {
    if (n == i) {
      return pa[i];
    }
    if (n == nphoton + i) {
      return qa[i];
    }
    if (n == 2 * nphoton + i) {
      return ea[i];
    }
  }

  return 0.0;
}

/* ---------------------------------------------------------------------- */

double FixCBOAMD::compute_array(int i, int j) {
  // Return polarizability components
  if (i >= 0 && i < 3 && j >= 0 && j < 3) {
    return polarizability[i * 3 + j];
  }
  return 0.0;
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::write_restart(FILE* fp) {
  // Write restart data
  if (comm->me == 0) {
    int size = 0;
    if (photons_enabled) {
      size += 4 * nphoton;
    }
    fwrite(&size, sizeof(int), 1, fp);

    if (photons_enabled) {
      fwrite(qa, sizeof(double), nphoton, fp);
      fwrite(pa, sizeof(double), nphoton, fp);
      fwrite(fa, sizeof(double), nphoton, fp);
      fwrite(ea, sizeof(double), nphoton, fp);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::restart(char* buf) {
  // Read restart data
  int size;
  memcpy(&size, buf, sizeof(int));
  buf += sizeof(int);

  if (photons_enabled && size >= 4 * nphoton) {
    memcpy(qa, buf, nphoton * sizeof(double));
    buf += nphoton * sizeof(double);
    memcpy(pa, buf, nphoton * sizeof(double));
    buf += nphoton * sizeof(double);
    memcpy(fa, buf, nphoton * sizeof(double));
    buf += nphoton * sizeof(double);
    memcpy(ea, buf, nphoton * sizeof(double));
  }
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::grow_arrays(int nmax) {
  memory->grow(forces_deepmd, nmax * 3, "fix_cboamd:forces_deepmd");
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::copy_arrays(int i, int j, int delflag) {
  // Copy forces for atom i to atom j
  for (int k = 0; k < 3; k++) {
    forces_deepmd[j * 3 + k] = forces_deepmd[i * 3 + k];
  }
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::set_arrays(int i) {
  // Set forces for atom i
  for (int k = 0; k < 3; k++) {
    forces_deepmd[i * 3 + k] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

int FixCBOAMD::pack_exchange(int i, double* buf) {
  // Pack exchange data
  int n = 0;
  for (int k = 0; k < 3; k++) {
    buf[n++] = forces_deepmd[i * 3 + k];
  }
  return n;
}

/* ---------------------------------------------------------------------- */

int FixCBOAMD::unpack_exchange(int nlocal, double* buf) {
  // Unpack exchange data
  int n = 0;
  for (int k = 0; k < 3; k++) {
    forces_deepmd[nlocal * 3 + k] = buf[n++];
  }
  return n;
}

/* ---------------------------------------------------------------------- */

int FixCBOAMD::pack_restart(int i, double* buf) {
  // Pack restart data
  int n = 0;
  for (int k = 0; k < 3; k++) {
    buf[n++] = forces_deepmd[i * 3 + k];
  }
  return n;
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::unpack_restart(int nlocal, int nth) {
  // Unpack restart data
  // This method is called when restarting from a checkpoint
  // For now, we'll just initialize photon coordinates to zero
  if (photons_enabled) {
    for (int i = 0; i < nphoton; i++) {
      qa[i] = 0.0;
      pa[i] = 0.0;
      fa[i] = 0.0;
      ea[i] = 0.0;
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixCBOAMD::size_restart(int nlocal) { return 3; }

/* ---------------------------------------------------------------------- */

int FixCBOAMD::maxsize_restart() { return 3; }

/* ---------------------------------------------------------------------- */

void FixCBOAMD::init_deepmd_models() {
  try {
    // Initialize DeepMD models using C++ interface
    deepmd_dipole = new deepmd_compat::DeepTensor(model_dipole);
    if (model_polar) {
      deepmd_polar = new deepmd_compat::DeepTensor(model_polar);
    }

    if (comm->me == 0) {
      utils::logmesg(lmp, "DeepMD models initialized successfully:\n");
      utils::logmesg(lmp, "  Dipole: {}\n", model_dipole);
      if (model_polar) {
        utils::logmesg(lmp, "  Polarizability: {}\n", model_polar);
      }
    }
  } catch (const std::exception& e) {
    error->all(FLERR, "Failed to initialize DeepMD models: {}", e.what());
  }
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::cleanup_deepmd_models() {
  // Clean up DeepMD models
  if (deepmd_dipole) {
    delete deepmd_dipole;
    deepmd_dipole = nullptr;
  }
  if (deepmd_polar) {
    delete deepmd_polar;
    deepmd_polar = nullptr;
  }
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::compute_deepmd_dipole() {
  try {
    // Compute dipole using DeepMD

    // dipole_grad_deepmd is the negative gradient of dipole w.r.t. atomic
    // coordinates It is a flattened array of size 3 (dipole components) * N
    // (atoms) * 3 (atomic coordinate components)
    deepmd_dipole->compute(dipole_deepmd, dipole_grad_deepmd,
                           dipole_virial_deepmd, dipole_atom_deepmd,
                           dipole_atom_virial_deepmd, coords_deepmd,
                           atom_types_deepmd, cell_deepmd);
    // Extract dipole components (DeepMD returns in eV/A, convert to a.u.)
    // dipole[0] = dipole_deepmd[0] * ANGSTROM_TO_BOHR;
    // dipole[1] = dipole_deepmd[1] * ANGSTROM_TO_BOHR;
    // dipole[2] = dipole_deepmd[2] * ANGSTROM_TO_BOHR;
    dipole[0] = dipole_deepmd[0];
    dipole[1] = dipole_deepmd[1];
    dipole[2] = dipole_deepmd[2];
  } catch (const std::exception& e) {
    error->all(FLERR, "DeepMD dipole computation failed: {}", e.what());
  }
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::compute_deepmd_polarizability() {
  try {
    // Compute polarizability using DeepMD
    deepmd_polar->compute(polar_deepmd, polar_grad_deepmd, polar_virial_deepmd,
                          polar_atom_deepmd, polar_atom_virial_deepmd,
                          coords_deepmd, atom_types_deepmd, cell_deepmd);

    // Extract polarizability components (DeepMD returns in eV/A, convert to
    // a.u.)
    for (int i = 0; i < 9; i++) {
      polarizability[i] = polar_deepmd[i] * EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR;
    }
  } catch (const std::exception& e) {
    error->all(FLERR, "DeepMD polarizability computation failed: {}", e.what());
  }
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::update_photon_coordinates() {
  // Update photon coordinates using velocity Verlet
  for (int i = 0; i < nphoton; i++) {
    // Update position
    qa[i] += (pa[i] + fa[i] * dt * 0.5) * dt;

    // Store old force
    double fa_old = fa[i];

    // Update force (this would be computed based on dipole and polarizability)
    fa[i] = -ea[i] * omega_photon[i];

    // Update momentum
    pa[i] += (fa_old + fa[i]) * dt * 0.5;
  }
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::compute_cboa_forces() {
  // Compute cavity Born-Oppenheimer forces
  // This would implement the CBOA force calculation
  // For now, we'll just set dummy values

  for (int i = 0; i < nphoton; i++) {
    ea[i] = omega_photon[i] * qa[i];
    for (int j = 0; j < 3; j++) {
      ea[i] -= lambda_photon[i] * lambda_vector[i][j] * dipole[j];
    }
  }

  int nlocal = atom->nlocal;
  double** f = atom->f;

  for (int dp = 0; dp < 3; dp++) {  // dp is the direction of dipole or photon
    for (int i = 0; i < nlocal; i++) {
      for (int di = 0; di < 3;
           di++) {  // di is the direction of atomic coordinate
        int idx = dp * nlocal * 3 + i * 3 +
                  di;  // index in dipole_grad_deepmd: dipole component dp, atom
                       // i, coordinate di
        for (int alpha = 0; alpha < nphoton; alpha++) {
          // CBOA force contribution from photon alpha
          f[i][di] -= ea[alpha] * lambda_photon[alpha] *
                      lambda_vector[alpha][di] * dipole_grad_deepmd[idx] /
                      EV_TO_HARTREE;
        }
      }
    }
  }

  for (int alpha = 0; alpha < nphoton; alpha++) {
    fa[alpha] = -ea[alpha] * omega_photon[alpha];
  }
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::write_output() {
  // Write output to file
  if (comm->me == 0) {
    fprintf(output_file,
            "%d %.6f %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e "
            "%.6e %.6e",
            current_step, update->dt * current_step, dipole[0], dipole[1],
            dipole[2], polarizability[0], polarizability[1], polarizability[2],
            polarizability[3], polarizability[4], polarizability[5],
            polarizability[6], polarizability[7], polarizability[8]);

    if (photons_enabled) {
      for (int i = 0; i < nphoton; i++) {
        fprintf(output_file, " %.6e %.6e %.6e %.6e", qa[i], pa[i], ea[i],
                fa[i]);
      }
    }
    fprintf(output_file, "\n");
    fflush(output_file);
  }
}

/* ---------------------------------------------------------------------- */

void FixCBOAMD::convert_coordinates_to_deepmd_format() {
  int nlocal = atom->nlocal;
  int* type = atom->type;
  double** x = atom->x;
  double* boxlo = domain->boxlo;
  double* boxhi = domain->boxhi;

  // Resize vectors
  coords_deepmd.resize(nlocal * 3);
  atom_types_deepmd.resize(nlocal);
  cell_deepmd.resize(9);

  // Convert coordinates from LAMMPS to DeepMD format
  for (int i = 0; i < nlocal; i++) {
    coords_deepmd[i * 3] = x[i][0];      // x
    coords_deepmd[i * 3 + 1] = x[i][1];  // y
    coords_deepmd[i * 3 + 2] = x[i][2];  // z
    atom_types_deepmd[i] = type_map[type[i]];
  }

  // Set cell (assuming orthorhombic box)
  double lx = boxhi[0] - boxlo[0];
  double ly = boxhi[1] - boxlo[1];
  double lz = boxhi[2] - boxlo[2];

  cell_deepmd[0] = lx;
  cell_deepmd[1] = 0.0;
  cell_deepmd[2] = 0.0;
  cell_deepmd[3] = 0.0;
  cell_deepmd[4] = ly;
  cell_deepmd[5] = 0.0;
  cell_deepmd[6] = 0.0;
  cell_deepmd[7] = 0.0;
  cell_deepmd[8] = lz;
}
