/**
* See https://docs.lammps.org/Developer_plugins.html
*/
#include "lammpsplugin.h"
#include "version.h"
#include "pair_deepmd.h"
#include "fix_dplr.h"
#include "compute_deeptensor_atom.h"

using namespace LAMMPS_NS;

static Pair *pairdeepmd(LAMMPS *lmp)
{
  return new PairDeepMD(lmp);
}

static Compute *computedeepmdtensoratom(LAMMPS *lmp, int narg, char **arg)
{
  return new ComputeDeeptensorAtom(lmp, narg, arg);
}

static Fix *fixdplr(LAMMPS *lmp, int narg, char **arg)
{
  return new FixDPLR(lmp, narg, arg);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "deepmd";
  plugin.info = "deepmd pair style v2.0";
  plugin.author = "Han Wang";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &pairdeepmd;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "compute";
  plugin.info = "compute deeptensor/atom v2.0";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &computedeepmdtensoratom;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "fix";
  plugin.info = "fix dplr v2.0";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &fixdplr;
  (*register_plugin)(&plugin, lmp);
}