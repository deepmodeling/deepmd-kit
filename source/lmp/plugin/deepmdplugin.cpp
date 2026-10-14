// SPDX-License-Identifier: LGPL-3.0-or-later
/**
 * See https://docs.lammps.org/Developer_plugins.html
 */
#include "compute_deepmd_fparam_dedn.h"
#include "compute_deeptensor_atom.h"
#include "deepmd_version.h"
#include "fix_cboamd.h"
#include "fix_dplr.h"
#include "lammpsplugin.h"
#ifdef LMP_KOKKOS
#include "pair_deepmd_kokkos.h"
#include "pair_dpa4spin_kokkos.h"
#endif
#include "pair_deepmd.h"
#include "pair_deepspin.h"
#include "version.h"
#if LAMMPS_VERSION_NUMBER >= 20220328
#include "pppm_dplr.h"
#endif

using namespace LAMMPS_NS;

static Pair* pairdeepmd(LAMMPS* lmp) { return new PairDeepMD(lmp); }
static Pair* pairdeepspin(LAMMPS* lmp) { return new PairDeepSpin(lmp); }

#ifdef LMP_KOKKOS
// Runtime plugins do not consume the PairStyle declarations used by LAMMPS'
// built-in package machinery, so register each Kokkos alias explicitly.
static Pair* pairdeepmdkokkosdevice(LAMMPS* lmp) {
  return new PairDeepMDKokkos<LMPDeviceType>(lmp);
}
static Pair* pairdeepmdkokkoshost(LAMMPS* lmp) {
  return new PairDeepMDKokkos<LMPHostType>(lmp);
}
static Pair* pairdpa4spinkokkosdevice(LAMMPS* lmp) {
  return new PairDPA4SpinKokkos<LMPDeviceType>(lmp);
}
static Pair* pairdpa4spinkokkoshost(LAMMPS* lmp) {
  return new PairDPA4SpinKokkos<LMPHostType>(lmp);
}
#endif

static Compute* computedeepmdtensoratom(LAMMPS* lmp, int narg, char** arg) {
  return new ComputeDeeptensorAtom(lmp, narg, arg);
}

static Compute* computedeepmdfparamdedn(LAMMPS* lmp, int narg, char** arg) {
  return new ComputeDeepmdFparamDedn(lmp, narg, arg);
}

static Fix* fixdplr(LAMMPS* lmp, int narg, char** arg) {
  return new FixDPLR(lmp, narg, arg);
}

static Fix* fixcboamd(LAMMPS* lmp, int narg, char** arg) {
  return new FixCBOAMD(lmp, narg, arg);
}

#if LAMMPS_VERSION_NUMBER >= 20220328
static KSpace* pppmdplr(LAMMPS* lmp) { return new PPPMDPLR(lmp); }
#endif

extern "C" void lammpsplugin_init(void* lmp, void* handle, void* regfunc) {
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc)regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "deepmd";
  plugin.info = "deepmd pair style " STR_GIT_SUMM;
  plugin.author = "Han Wang";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pairdeepmd;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);

#ifdef LMP_KOKKOS
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "deepmd/kk";
  plugin.info = "deepmd Kokkos pair style " STR_GIT_SUMM;
  plugin.author = "Tiancheng Li";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pairdeepmdkokkosdevice;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);

  plugin.name = "deepmd/kk/device";
  (*register_plugin)(&plugin, lmp);

  plugin.name = "deepmd/kk/host";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pairdeepmdkokkoshost;
  (*register_plugin)(&plugin, lmp);

  plugin.name = "dpa4spin/kk";
  plugin.info = "dpa4spin Kokkos pair style " STR_GIT_SUMM;
  plugin.author = "Tiancheng Li";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pairdpa4spinkokkosdevice;
  (*register_plugin)(&plugin, lmp);

  plugin.name = "dpa4spin/kk/device";
  (*register_plugin)(&plugin, lmp);

  plugin.name = "dpa4spin/kk/host";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pairdpa4spinkokkoshost;
  (*register_plugin)(&plugin, lmp);
#endif

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "deepspin";
  plugin.info = "deepspin pair style " STR_GIT_SUMM;
  plugin.author = "Duo Zhang";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pairdeepspin;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "compute";
  plugin.name = "deeptensor/atom";
  plugin.info = "compute deeptensor/atom " STR_GIT_SUMM;
  plugin.author = "Han Wang";
  plugin.creator.v2 = (lammpsplugin_factory2*)&computedeepmdtensoratom;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "compute";
  plugin.name = "deepmd/fparam/dedn";
  plugin.info = "compute deepmd/fparam/dedn " STR_GIT_SUMM;
  plugin.author = "Li Fu";
  plugin.creator.v2 = (lammpsplugin_factory2*)&computedeepmdfparamdedn;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "fix";
  plugin.name = "dplr";
  plugin.info = "fix dplr " STR_GIT_SUMM;
  plugin.author = "Han Wang";
  plugin.creator.v2 = (lammpsplugin_factory2*)&fixdplr;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "fix";
  plugin.name = "cboamd";
  plugin.info = "fix cboamd " STR_GIT_SUMM;
  plugin.author = "DeePMD-kit";
  plugin.creator.v2 = (lammpsplugin_factory2*)&fixcboamd;
  (*register_plugin)(&plugin, lmp);

#if LAMMPS_VERSION_NUMBER >= 20220328
  // lammps/lammps#
  plugin.style = "kspace";
  plugin.name = "pppm/dplr";
  plugin.info = "kspace pppm/dplr " STR_GIT_SUMM;
  plugin.author = "Han Wang";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pppmdplr;
  (*register_plugin)(&plugin, lmp);
#endif
}
