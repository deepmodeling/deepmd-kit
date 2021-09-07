/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013-2020, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */

#include <iostream>
#include "gmxpre.h"

#include "config.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <array>

#include "gromacs/awh/awh.h"
#include "gromacs/domdec/dlbtiming.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/gpuhaloexchange.h"
#include "gromacs/domdec/partition.h"
#include "gromacs/essentialdynamics/edsam.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/ewald/pme_pp_comm_gpu.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nonbonded/nb_free_energy.h"
#include "gromacs/gmxlib/nonbonded/nb_kernel.h"
#include "gromacs/gmxlib/nonbonded/nonbonded.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/imd/imd.h"
#include "gromacs/listed_forces/disre.h"
#include "gromacs/listed_forces/gpubonded.h"
#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/listed_forces/manage_threading.h"
#include "gromacs/listed_forces/orires.h"
#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vecdump.h"
#include "gromacs/mdlib/calcmu.h"
#include "gromacs/mdlib/calcvir.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/enerdata_utils.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/forcerec.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/qmmm.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/mdtypes/state_propagator_data_gpu.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/mshift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pulling/pull.h"
#include "gromacs/pulling/pull_rotation.h"
#include "gromacs/timing/cyclecounter.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/timing/wallcyclereporting.h"
#include "gromacs/timing/walltime_accounting.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/fixedcapacityvector.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/sysinfo.h"
#include "deepmd/DeepPot.h"
#include "gromacs/mdlib/deepmd_plugin.h"

using gmx::AtomLocality;
using gmx::DomainLifetimeWorkload;
using gmx::ForceOutputs;
using gmx::InteractionLocality;
using gmx::SimulationWorkload;
using gmx::StepWorkload;

// TODO: this environment variable allows us to verify before release
// that on less common architectures the total cost of polling is not larger than
// a blocking wait (so polling does not introduce overhead when the static
// PME-first ordering would suffice).
static const bool c_disableAlternatingWait = (getenv("GMX_DISABLE_ALTERNATING_GPU_WAIT") != nullptr);

static void sum_forces(rvec f[], gmx::ArrayRef<const gmx::RVec> forceToAdd)
{
    const int end = forceToAdd.size();

    int gmx_unused nt = gmx_omp_nthreads_get(emntDefault);
#pragma omp parallel for num_threads(nt) schedule(static)
    for (int i = 0; i < end; i++)
    {
        rvec_inc(f[i], forceToAdd[i]);
    }
}

static void calc_virial(int                              start,
                        int                              homenr,
                        const rvec                       x[],
                        const gmx::ForceWithShiftForces& forceWithShiftForces,
                        tensor                           vir_part,
                        const t_graph*                   graph,
                        const matrix                     box,
                        t_nrnb*                          nrnb,
                        const t_forcerec*                fr,
                        int                              ePBC)
{
    /* The short-range virial from surrounding boxes */
    const rvec* fshift = as_rvec_array(forceWithShiftForces.shiftForces().data());
    calc_vir(SHIFTS, fr->shift_vec, fshift, vir_part, ePBC == epbcSCREW, box);
    inc_nrnb(nrnb, eNR_VIRIAL, SHIFTS);

    /* Calculate partial virial, for local atoms only, based on short range.
     * Total virial is computed in global_stat, called from do_md
     */
    const rvec* f = as_rvec_array(forceWithShiftForces.force().data());
    f_calc_vir(start, start + homenr, x, f, vir_part, graph, box);
    inc_nrnb(nrnb, eNR_VIRIAL, homenr);

    if (debug)
    {
        pr_rvecs(debug, 0, "vir_part", vir_part, DIM);
    }
}

static void pull_potential_wrapper(const t_commrec*               cr,
                                   const t_inputrec*              ir,
                                   const matrix                   box,
                                   gmx::ArrayRef<const gmx::RVec> x,
                                   gmx::ForceWithVirial*          force,
                                   const t_mdatoms*               mdatoms,
                                   gmx_enerdata_t*                enerd,
                                   pull_t*                        pull_work,
                                   const real*                    lambda,
                                   double                         t,
                                   gmx_wallcycle_t                wcycle)
{
    t_pbc pbc;
    real  dvdl;

    /* Calculate the center of mass forces, this requires communication,
     * which is why pull_potential is called close to other communication.
     */
    wallcycle_start(wcycle, ewcPULLPOT);
    set_pbc(&pbc, ir->ePBC, box);
    dvdl = 0;
    enerd->term[F_COM_PULL] += pull_potential(pull_work, mdatoms, &pbc, cr, t, lambda[efptRESTRAINT],
                                              as_rvec_array(x.data()), force, &dvdl);
    enerd->dvdl_lin[efptRESTRAINT] += dvdl;
    wallcycle_stop(wcycle, ewcPULLPOT);
}

static void pme_receive_force_ener(t_forcerec*           fr,
                                   const t_commrec*      cr,
                                   gmx::ForceWithVirial* forceWithVirial,
                                   gmx_enerdata_t*       enerd,
                                   bool                  useGpuPmePpComms,
                                   bool                  receivePmeForceToGpu,
                                   gmx_wallcycle_t       wcycle)
{
    real  e_q, e_lj, dvdl_q, dvdl_lj;
    float cycles_ppdpme, cycles_seppme;

    cycles_ppdpme = wallcycle_stop(wcycle, ewcPPDURINGPME);
    dd_cycles_add(cr->dd, cycles_ppdpme, ddCyclPPduringPME);

    /* In case of node-splitting, the PP nodes receive the long-range
     * forces, virial and energy from the PME nodes here.
     */
    wallcycle_start(wcycle, ewcPP_PMEWAITRECVF);
    dvdl_q  = 0;
    dvdl_lj = 0;
    gmx_pme_receive_f(fr->pmePpCommGpu.get(), cr, forceWithVirial, &e_q, &e_lj, &dvdl_q, &dvdl_lj,
                      useGpuPmePpComms, receivePmeForceToGpu, &cycles_seppme);
    enerd->term[F_COUL_RECIP] += e_q;
    enerd->term[F_LJ_RECIP] += e_lj;
    enerd->dvdl_lin[efptCOUL] += dvdl_q;
    enerd->dvdl_lin[efptVDW] += dvdl_lj;

    if (wcycle)
    {
        dd_cycles_add(cr->dd, cycles_seppme, ddCyclPME);
    }
    wallcycle_stop(wcycle, ewcPP_PMEWAITRECVF);
}

static void print_large_forces(FILE*            fp,
                               const t_mdatoms* md,
                               const t_commrec* cr,
                               int64_t          step,
                               real             forceTolerance,
                               const rvec*      x,
                               const rvec*      f)
{
    real       force2Tolerance = gmx::square(forceTolerance);
    gmx::index numNonFinite    = 0;
    for (int i = 0; i < md->homenr; i++)
    {
        real force2    = norm2(f[i]);
        bool nonFinite = !std::isfinite(force2);
        if (force2 >= force2Tolerance || nonFinite)
        {
            fprintf(fp, "step %" PRId64 " atom %6d  x %8.3f %8.3f %8.3f  force %12.5e\n", step,
                    ddglatnr(cr->dd, i), x[i][XX], x[i][YY], x[i][ZZ], std::sqrt(force2));
        }
        if (nonFinite)
        {
            numNonFinite++;
        }
    }
    if (numNonFinite > 0)
    {
        /* Note that with MPI this fatal call on one rank might interrupt
         * the printing on other ranks. But we can only avoid that with
         * an expensive MPI barrier that we would need at each step.
         */
        gmx_fatal(FARGS, "At step %" PRId64 " detected non-finite forces on %td atoms", step, numNonFinite);
    }
}

static void post_process_forces(const t_commrec*      cr,
                                int64_t               step,
                                t_nrnb*               nrnb,
                                gmx_wallcycle_t       wcycle,
                                const gmx_localtop_t* top,
                                const matrix          box,
                                const rvec            x[],
                                ForceOutputs*         forceOutputs,
                                tensor                vir_force,
                                const t_mdatoms*      mdatoms,
                                const t_graph*        graph,
                                const t_forcerec*     fr,
                                const gmx_vsite_t*    vsite,
                                const StepWorkload&   stepWork)
{
    rvec* f = as_rvec_array(forceOutputs->forceWithShiftForces().force().data());

    if (fr->haveDirectVirialContributions)
    {
        auto& forceWithVirial = forceOutputs->forceWithVirial();
        rvec* fDirectVir      = as_rvec_array(forceWithVirial.force_.data());

        if (vsite)
        {
            /* Spread the mesh force on virtual sites to the other particles...
             * This is parallellized. MPI communication is performed
             * if the constructing atoms aren't local.
             */
            matrix virial = { { 0 } };
            spread_vsite_f(vsite, x, fDirectVir, nullptr, stepWork.computeVirial, virial, nrnb,
                           &top->idef, fr->ePBC, fr->bMolPBC, graph, box, cr, wcycle);
            forceWithVirial.addVirialContribution(virial);
        }

        if (stepWork.computeVirial)
        {
            /* Now add the forces, this is local */
            sum_forces(f, forceWithVirial.force_);

            /* Add the direct virial contributions */
            GMX_ASSERT(
                    forceWithVirial.computeVirial_,
                    "forceWithVirial should request virial computation when we request the virial");
            m_add(vir_force, forceWithVirial.getVirial(), vir_force);

            if (debug)
            {
                pr_rvecs(debug, 0, "vir_force", vir_force, DIM);
            }
        }
    }

    if (fr->print_force >= 0)
    {
        print_large_forces(stderr, mdatoms, cr, step, fr->print_force, x, f);
    }
}

static void do_nb_verlet(t_forcerec*                fr,
                         const interaction_const_t* ic,
                         gmx_enerdata_t*            enerd,
                         const StepWorkload&        stepWork,
                         const InteractionLocality  ilocality,
                         const int                  clearF,
                         const int64_t              step,
                         t_nrnb*                    nrnb,
                         gmx_wallcycle_t            wcycle)
{
    if (!stepWork.computeNonbondedForces)
    {
        /* skip non-bonded calculation */
        return;
    }

    nonbonded_verlet_t* nbv = fr->nbv.get();

    /* GPU kernel launch overhead is already timed separately */
    if (fr->cutoff_scheme != ecutsVERLET)
    {
        gmx_incons("Invalid cut-off scheme passed!");
    }

    if (!nbv->useGpu())
    {
        /* When dynamic pair-list  pruning is requested, we need to prune
         * at nstlistPrune steps.
         */
        if (nbv->isDynamicPruningStepCpu(step))
        {
            /* Prune the pair-list beyond fr->ic->rlistPrune using
             * the current coordinates of the atoms.
             */
            wallcycle_sub_start(wcycle, ewcsNONBONDED_PRUNING);
            nbv->dispatchPruneKernelCpu(ilocality, fr->shift_vec);
            wallcycle_sub_stop(wcycle, ewcsNONBONDED_PRUNING);
        }
    }

    nbv->dispatchNonbondedKernel(ilocality, *ic, stepWork, clearF, *fr, enerd, nrnb);
}

static inline void clear_rvecs_omp(int n, rvec v[])
{
    int nth = gmx_omp_nthreads_get_simple_rvec_task(emntDefault, n);

    /* Note that we would like to avoid this conditional by putting it
     * into the omp pragma instead, but then we still take the full
     * omp parallel for overhead (at least with gcc5).
     */
    if (nth == 1)
    {
        for (int i = 0; i < n; i++)
        {
            clear_rvec(v[i]);
        }
    }
    else
    {
#pragma omp parallel for num_threads(nth) schedule(static)
        for (int i = 0; i < n; i++)
        {
            clear_rvec(v[i]);
        }
    }
}

/*! \brief Return an estimate of the average kinetic energy or 0 when unreliable
 *
 * \param groupOptions  Group options, containing T-coupling options
 */
static real averageKineticEnergyEstimate(const t_grpopts& groupOptions)
{
    real nrdfCoupled   = 0;
    real nrdfUncoupled = 0;
    real kineticEnergy = 0;
    for (int g = 0; g < groupOptions.ngtc; g++)
    {
        if (groupOptions.tau_t[g] >= 0)
        {
            nrdfCoupled += groupOptions.nrdf[g];
            kineticEnergy += groupOptions.nrdf[g] * 0.5 * groupOptions.ref_t[g] * BOLTZ;
        }
        else
        {
            nrdfUncoupled += groupOptions.nrdf[g];
        }
    }

    /* This conditional with > also catches nrdf=0 */
    if (nrdfCoupled > nrdfUncoupled)
    {
        return kineticEnergy * (nrdfCoupled + nrdfUncoupled) / nrdfCoupled;
    }
    else
    {
        return 0;
    }
}

/*! \brief This routine checks that the potential energy is finite.
 *
 * Always checks that the potential energy is finite. If step equals
 * inputrec.init_step also checks that the magnitude of the potential energy
 * is reasonable. Terminates with a fatal error when a check fails.
 * Note that passing this check does not guarantee finite forces,
 * since those use slightly different arithmetics. But in most cases
 * there is just a narrow coordinate range where forces are not finite
 * and energies are finite.
 *
 * \param[in] step      The step number, used for checking and printing
 * \param[in] enerd     The energy data; the non-bonded group energies need to be added to
 * enerd.term[F_EPOT] before calling this routine \param[in] inputrec  The input record
 */
static void checkPotentialEnergyValidity(int64_t step, const gmx_enerdata_t& enerd, const t_inputrec& inputrec)
{
    /* Threshold valid for comparing absolute potential energy against
     * the kinetic energy. Normally one should not consider absolute
     * potential energy values, but with a factor of one million
     * we should never get false positives.
     */
    constexpr real c_thresholdFactor = 1e6;

    bool energyIsNotFinite    = !std::isfinite(enerd.term[F_EPOT]);
    real averageKineticEnergy = 0;
    /* We only check for large potential energy at the initial step,
     * because that is by far the most likely step for this too occur
     * and because computing the average kinetic energy is not free.
     * Note: nstcalcenergy >> 1 often does not allow to catch large energies
     * before they become NaN.
     */
    if (step == inputrec.init_step && EI_DYNAMICS(inputrec.eI))
    {
        averageKineticEnergy = averageKineticEnergyEstimate(inputrec.opts);
    }

    if (energyIsNotFinite
        || (averageKineticEnergy > 0 && enerd.term[F_EPOT] > c_thresholdFactor * averageKineticEnergy))
    {
        gmx_fatal(
                FARGS,
                "Step %" PRId64
                ": The total potential energy is %g, which is %s. The LJ and electrostatic "
                "contributions to the energy are %g and %g, respectively. A %s potential energy "
                "can be caused by overlapping interactions in bonded interactions or very large%s "
                "coordinate values. Usually this is caused by a badly- or non-equilibrated initial "
                "configuration, incorrect interactions or parameters in the topology.",
                step, enerd.term[F_EPOT], energyIsNotFinite ? "not finite" : "extremely high",
                enerd.term[F_LJ], enerd.term[F_COUL_SR],
                energyIsNotFinite ? "non-finite" : "very high", energyIsNotFinite ? " or Nan" : "");
    }
}

/*! \brief Return true if there are special forces computed this step.
 *
 * The conditionals exactly correspond to those in computeSpecialForces().
 */
static bool haveSpecialForces(const t_inputrec&          inputrec,
                              const gmx::ForceProviders& forceProviders,
                              const pull_t*              pull_work,
                              const bool                 computeForces,
                              const gmx_edsam*           ed)
{

    return ((computeForces && forceProviders.hasForceProvider()) || // forceProviders
            (inputrec.bPull && pull_have_potential(pull_work)) ||   // pull
            inputrec.bRot ||                                        // enforced rotation
            (ed != nullptr) ||                                      // flooding
            (inputrec.bIMD && computeForces));                      // IMD
}

/*! \brief Compute forces and/or energies for special algorithms
 *
 * The intention is to collect all calls to algorithms that compute
 * forces on local atoms only and that do not contribute to the local
 * virial sum (but add their virial contribution separately).
 * Eventually these should likely all become ForceProviders.
 * Within this function the intention is to have algorithms that do
 * global communication at the end, so global barriers within the MD loop
 * are as close together as possible.
 *
 * \param[in]     fplog            The log file
 * \param[in]     cr               The communication record
 * \param[in]     inputrec         The input record
 * \param[in]     awh              The Awh module (nullptr if none in use).
 * \param[in]     enforcedRotation Enforced rotation module.
 * \param[in]     imdSession       The IMD session
 * \param[in]     pull_work        The pull work structure.
 * \param[in]     step             The current MD step
 * \param[in]     t                The current time
 * \param[in,out] wcycle           Wallcycle accounting struct
 * \param[in,out] forceProviders   Pointer to a list of force providers
 * \param[in]     box              The unit cell
 * \param[in]     x                The coordinates
 * \param[in]     mdatoms          Per atom properties
 * \param[in]     lambda           Array of free-energy lambda values
 * \param[in]     stepWork         Step schedule flags
 * \param[in,out] forceWithVirial  Force and virial buffers
 * \param[in,out] enerd            Energy buffer
 * \param[in,out] ed               Essential dynamics pointer
 * \param[in]     didNeighborSearch Tells if we did neighbor searching this step, used for ED sampling
 *
 * \todo Remove didNeighborSearch, which is used incorrectly.
 * \todo Convert all other algorithms called here to ForceProviders.
 */
static void computeSpecialForces(FILE*                          fplog,
                                 const t_commrec*               cr,
                                 const t_inputrec*              inputrec,
                                 gmx::Awh*                      awh,
                                 gmx_enfrot*                    enforcedRotation,
                                 gmx::ImdSession*               imdSession,
                                 pull_t*                        pull_work,
                                 int64_t                        step,
                                 double                         t,
                                 gmx_wallcycle_t                wcycle,
                                 gmx::ForceProviders*           forceProviders,
                                 const matrix                   box,
                                 gmx::ArrayRef<const gmx::RVec> x,
                                 const t_mdatoms*               mdatoms,
                                 real*                          lambda,
                                 const StepWorkload&            stepWork,
                                 gmx::ForceWithVirial*          forceWithVirial,
                                 gmx_enerdata_t*                enerd,
                                 gmx_edsam*                     ed,
                                 bool                           didNeighborSearch)
{
    /* NOTE: Currently all ForceProviders only provide forces.
     *       When they also provide energies, remove this conditional.
     */
    if (stepWork.computeForces)
    {
        gmx::ForceProviderInput  forceProviderInput(x, *mdatoms, t, box, *cr);
        gmx::ForceProviderOutput forceProviderOutput(forceWithVirial, enerd);

        /* Collect forces from modules */
        forceProviders->calculateForces(forceProviderInput, &forceProviderOutput);
    }

    if (inputrec->bPull && pull_have_potential(pull_work))
    {
        pull_potential_wrapper(cr, inputrec, box, x, forceWithVirial, mdatoms, enerd, pull_work,
                               lambda, t, wcycle);

        if (awh)
        {
            enerd->term[F_COM_PULL] += awh->applyBiasForcesAndUpdateBias(
                    inputrec->ePBC, *mdatoms, box, forceWithVirial, t, step, wcycle, fplog);
        }
    }

    rvec* f = as_rvec_array(forceWithVirial->force_.data());

    /* Add the forces from enforced rotation potentials (if any) */
    if (inputrec->bRot)
    {
        wallcycle_start(wcycle, ewcROTadd);
        enerd->term[F_COM_PULL] += add_rot_forces(enforcedRotation, f, cr, step, t);
        wallcycle_stop(wcycle, ewcROTadd);
    }

    if (ed)
    {
        /* Note that since init_edsam() is called after the initialization
         * of forcerec, edsam doesn't request the noVirSum force buffer.
         * Thus if no other algorithm (e.g. PME) requires it, the forces
         * here will contribute to the virial.
         */
        do_flood(cr, inputrec, as_rvec_array(x.data()), f, ed, box, step, didNeighborSearch);
    }

    /* Add forces from interactive molecular dynamics (IMD), if any */
    if (inputrec->bIMD && stepWork.computeForces)
    {
        imdSession->applyForces(f);
    }
}

/*! \brief Makes PME flags from StepWorkload data.
 *
 * \param[in]  stepWork     Step schedule flags
 * \returns                 PME flags
 */
static int makePmeFlags(const StepWorkload& stepWork)
{
    return GMX_PME_SPREAD | GMX_PME_SOLVE | (stepWork.computeVirial ? GMX_PME_CALC_ENER_VIR : 0)
           | (stepWork.computeEnergy ? GMX_PME_CALC_ENER_VIR : 0)
           | (stepWork.computeForces ? GMX_PME_CALC_F : 0);
}

/*! \brief Launch the prepare_step and spread stages of PME GPU.
 *
 * \param[in]  pmedata              The PME structure
 * \param[in]  box                  The box matrix
 * \param[in]  stepWork             Step schedule flags
 * \param[in]  pmeFlags             PME flags
 * \param[in]  xReadyOnDevice       Event synchronizer indicating that the coordinates are ready in
 * the device memory. \param[in]  wcycle               The wallcycle structure
 */
static inline void launchPmeGpuSpread(gmx_pme_t*            pmedata,
                                      const matrix          box,
                                      const StepWorkload&   stepWork,
                                      int                   pmeFlags,
                                      GpuEventSynchronizer* xReadyOnDevice,
                                      gmx_wallcycle_t       wcycle)
{
    pme_gpu_prepare_computation(pmedata, stepWork.haveDynamicBox, box, wcycle, pmeFlags,
                                stepWork.useGpuPmeFReduction);
    pme_gpu_launch_spread(pmedata, xReadyOnDevice, wcycle);
}

/*! \brief Launch the FFT and gather stages of PME GPU
 *
 * This function only implements setting the output forces (no accumulation).
 *
 * \param[in]  pmedata        The PME structure
 * \param[in]  wcycle         The wallcycle structure
 */
static void launchPmeGpuFftAndGather(gmx_pme_t* pmedata, gmx_wallcycle_t wcycle)
{
    pme_gpu_launch_complex_transforms(pmedata, wcycle);
    pme_gpu_launch_gather(pmedata, wcycle, PmeForceOutputHandling::Set);
}

/*! \brief
 *  Polling wait for either of the PME or nonbonded GPU tasks.
 *
 * Instead of a static order in waiting for GPU tasks, this function
 * polls checking which of the two tasks completes first, and does the
 * associated force buffer reduction overlapped with the other task.
 * By doing that, unlike static scheduling order, it can always overlap
 * one of the reductions, regardless of the GPU task completion order.
 *
 * \param[in]     nbv              Nonbonded verlet structure
 * \param[in,out] pmedata          PME module data
 * \param[in,out] forceOutputs     Output buffer for the forces and virial
 * \param[in,out] enerd            Energy data structure results are reduced into
 * \param[in]     stepWork         Step schedule flags
 * \param[in]     pmeFlags         PME flags
 * \param[in]     wcycle           The wallcycle structure
 */
static void alternatePmeNbGpuWaitReduce(nonbonded_verlet_t* nbv,
                                        gmx_pme_t*          pmedata,
                                        gmx::ForceOutputs*  forceOutputs,
                                        gmx_enerdata_t*     enerd,
                                        const StepWorkload& stepWork,
                                        int                 pmeFlags,
                                        gmx_wallcycle_t     wcycle)
{
    bool isPmeGpuDone = false;
    bool isNbGpuDone  = false;


    gmx::ForceWithShiftForces& forceWithShiftForces = forceOutputs->forceWithShiftForces();
    gmx::ForceWithVirial&      forceWithVirial      = forceOutputs->forceWithVirial();

    gmx::ArrayRef<const gmx::RVec> pmeGpuForces;

    while (!isPmeGpuDone || !isNbGpuDone)
    {
        if (!isPmeGpuDone)
        {
            GpuTaskCompletion completionType =
                    (isNbGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
            isPmeGpuDone = pme_gpu_try_finish_task(pmedata, pmeFlags, wcycle, &forceWithVirial,
                                                   enerd, completionType);
        }

        if (!isNbGpuDone)
        {
            GpuTaskCompletion completionType =
                    (isPmeGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
            isNbGpuDone = Nbnxm::gpu_try_finish_task(
                    nbv->gpu_nbv, stepWork, AtomLocality::Local, enerd->grpp.ener[egLJSR].data(),
                    enerd->grpp.ener[egCOULSR].data(), forceWithShiftForces.shiftForces(),
                    completionType, wcycle);

            if (isNbGpuDone)
            {
                nbv->atomdata_add_nbat_f_to_f(AtomLocality::Local, forceWithShiftForces.force());
            }
        }
    }
}

/*! \brief Set up the different force buffers; also does clearing.
 *
 * \param[in] fr        force record pointer
 * \param[in] pull_work The pull work object.
 * \param[in] inputrec  input record
 * \param[in] force     force array
 * \param[in] stepWork  Step schedule flags
 * \param[out] wcycle   wallcycle recording structure
 *
 * \returns             Cleared force output structure
 */
static ForceOutputs setupForceOutputs(t_forcerec*                         fr,
                                      pull_t*                             pull_work,
                                      const t_inputrec&                   inputrec,
                                      gmx::ArrayRefWithPadding<gmx::RVec> force,
                                      const StepWorkload&                 stepWork,
                                      gmx_wallcycle_t                     wcycle)
{
    wallcycle_sub_start(wcycle, ewcsCLEAR_FORCE_BUFFER);

    /* NOTE: We assume fr->shiftForces is all zeros here */
    gmx::ForceWithShiftForces forceWithShiftForces(force, stepWork.computeVirial, fr->shiftForces);

    if (stepWork.computeForces)
    {
        /* Clear the short- and long-range forces */
        clear_rvecs_omp(fr->natoms_force_constr, as_rvec_array(forceWithShiftForces.force().data()));
    }

    /* If we need to compute the virial, we might need a separate
     * force buffer for algorithms for which the virial is calculated
     * directly, such as PME. Otherwise, forceWithVirial uses the
     * the same force (f in legacy calls) buffer as other algorithms.
     */
    const bool useSeparateForceWithVirialBuffer =
            (stepWork.computeForces && (stepWork.computeVirial && fr->haveDirectVirialContributions));
    /* forceWithVirial uses the local atom range only */
    gmx::ForceWithVirial forceWithVirial(useSeparateForceWithVirialBuffer ? fr->forceBufferForDirectVirialContributions
                                                                          : force.unpaddedArrayRef(),
                                         stepWork.computeVirial);

    if (useSeparateForceWithVirialBuffer)
    {
        /* TODO: update comment
         * We only compute forces on local atoms. Note that vsites can
         * spread to non-local atoms, but that part of the buffer is
         * cleared separately in the vsite spreading code.
         */
        clear_rvecs_omp(forceWithVirial.force_.size(), as_rvec_array(forceWithVirial.force_.data()));
    }

    if (inputrec.bPull && pull_have_constraint(pull_work))
    {
        clear_pull_forces(pull_work);
    }

    wallcycle_sub_stop(wcycle, ewcsCLEAR_FORCE_BUFFER);

    return ForceOutputs(forceWithShiftForces, forceWithVirial);
}


/*! \brief Set up flags that have the lifetime of the domain indicating what type of work is there to compute.
 */
static DomainLifetimeWorkload setupDomainLifetimeWorkload(const t_inputrec&         inputrec,
                                                          const t_forcerec&         fr,
                                                          const pull_t*             pull_work,
                                                          const gmx_edsam*          ed,
                                                          const t_idef&             idef,
                                                          const t_fcdata&           fcd,
                                                          const t_mdatoms&          mdatoms,
                                                          const SimulationWorkload& simulationWork,
                                                          const StepWorkload&       stepWork)
{
    DomainLifetimeWorkload domainWork;
    // Note that haveSpecialForces is constant over the whole run
    domainWork.haveSpecialForces =
            haveSpecialForces(inputrec, *fr.forceProviders, pull_work, stepWork.computeForces, ed);
    domainWork.haveCpuBondedWork  = haveCpuBondeds(fr);
    domainWork.haveGpuBondedWork  = ((fr.gpuBonded != nullptr) && fr.gpuBonded->haveInteractions());
    domainWork.haveRestraintsWork = haveRestraints(idef, fcd);
    domainWork.haveCpuListedForceWork = haveCpuListedForces(fr, idef, fcd);
    // Note that haveFreeEnergyWork is constant over the whole run
    domainWork.haveFreeEnergyWork = (fr.efep != efepNO && mdatoms.nPerturbed != 0);
    // We assume we have local force work if there are CPU
    // force tasks including PME or nonbondeds.
    domainWork.haveCpuLocalForceWork =
            domainWork.haveSpecialForces || domainWork.haveCpuListedForceWork
            || domainWork.haveFreeEnergyWork || simulationWork.useCpuNonbonded || simulationWork.useCpuPme
            || simulationWork.haveEwaldSurfaceContribution || inputrec.nwall > 0;

    return domainWork;
}

/*! \brief Set up force flag stuct from the force bitmask.
 *
 * \param[in]      legacyFlags          Force bitmask flags used to construct the new flags
 * \param[in]      isNonbondedOn        Global override, if false forces to turn off all nonbonded calculation.
 * \param[in]      simulationWork       Simulation workload description.
 * \param[in]      rankHasPmeDuty       If this rank computes PME.
 *
 * \returns New Stepworkload description.
 */
static StepWorkload setupStepWorkload(const int                 legacyFlags,
                                      const bool                isNonbondedOn,
                                      const SimulationWorkload& simulationWork,
                                      const bool                rankHasPmeDuty)
{
    StepWorkload flags;
    flags.stateChanged           = ((legacyFlags & GMX_FORCE_STATECHANGED) != 0);
    flags.haveDynamicBox         = ((legacyFlags & GMX_FORCE_DYNAMICBOX) != 0);
    flags.doNeighborSearch       = ((legacyFlags & GMX_FORCE_NS) != 0);
    flags.computeVirial          = ((legacyFlags & GMX_FORCE_VIRIAL) != 0);
    flags.computeEnergy          = ((legacyFlags & GMX_FORCE_ENERGY) != 0);
    flags.computeForces          = ((legacyFlags & GMX_FORCE_FORCES) != 0);
    flags.computeListedForces    = ((legacyFlags & GMX_FORCE_LISTED) != 0);
    flags.computeNonbondedForces = ((legacyFlags & GMX_FORCE_NONBONDED) != 0) && isNonbondedOn;
    flags.computeDhdl            = ((legacyFlags & GMX_FORCE_DHDL) != 0);

    if (simulationWork.useGpuBufferOps)
    {
        GMX_ASSERT(simulationWork.useGpuNonbonded,
                   "Can only offload buffer ops if nonbonded computation is also offloaded");
    }
    flags.useGpuXBufferOps = simulationWork.useGpuBufferOps;
    // on virial steps the CPU reduction path is taken
    flags.useGpuFBufferOps    = simulationWork.useGpuBufferOps && !flags.computeVirial;
    flags.useGpuPmeFReduction = flags.useGpuFBufferOps
                                && (simulationWork.useGpuPme
                                    && (rankHasPmeDuty || simulationWork.useGpuPmePpCommunication));

    return flags;
}


/* \brief Launch end-of-step GPU tasks: buffer clearing and rolling pruning.
 *
 * TODO: eliminate the \p useGpuNonbonded and \p useGpuNonbonded when these are
 * incorporated in DomainLifetimeWorkload.
 */
static void launchGpuEndOfStepTasks(nonbonded_verlet_t*               nbv,
                                    gmx::GpuBonded*                   gpuBonded,
                                    gmx_pme_t*                        pmedata,
                                    gmx_enerdata_t*                   enerd,
                                    const gmx::MdrunScheduleWorkload& runScheduleWork,
                                    bool                              useGpuNonbonded,
                                    bool                              useGpuPme,
                                    int64_t                           step,
                                    gmx_wallcycle_t                   wcycle)
{
    if (useGpuNonbonded)
    {
        /* Launch pruning before buffer clearing because the API overhead of the
         * clear kernel launches can leave the GPU idle while it could be running
         * the prune kernel.
         */
        if (nbv->isDynamicPruningStepGpu(step))
        {
            nbv->dispatchPruneKernelGpu(step);
        }

        /* now clear the GPU outputs while we finish the step on the CPU */
        wallcycle_start_nocount(wcycle, ewcLAUNCH_GPU);
        wallcycle_sub_start_nocount(wcycle, ewcsLAUNCH_GPU_NONBONDED);
        Nbnxm::gpu_clear_outputs(nbv->gpu_nbv, runScheduleWork.stepWork.computeVirial);
        wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_NONBONDED);
        wallcycle_stop(wcycle, ewcLAUNCH_GPU);
    }

    if (useGpuPme)
    {
        pme_gpu_reinit_computation(pmedata, wcycle);
    }

    if (runScheduleWork.domainWork.haveGpuBondedWork && runScheduleWork.stepWork.computeEnergy)
    {
        // in principle this should be included in the DD balancing region,
        // but generally it is infrequent so we'll omit it for the sake of
        // simpler code
        gpuBonded->waitAccumulateEnergyTerms(enerd);

        gpuBonded->clearEnergies();
    }
}


void do_force(FILE*                               fplog,
              const t_commrec*                    cr,
              const gmx_multisim_t*               ms,
              const t_inputrec*                   inputrec,
              gmx::Awh*                           awh,
              gmx_enfrot*                         enforcedRotation,
              gmx::ImdSession*                    imdSession,
              pull_t*                             pull_work,
              int64_t                             step,
              t_nrnb*                             nrnb,
              gmx_wallcycle_t                     wcycle,
              const gmx_localtop_t*               top,
              const matrix                        box,
              gmx::ArrayRefWithPadding<gmx::RVec> x,
              history_t*                          hist,
              gmx::ArrayRefWithPadding<gmx::RVec> force,
              tensor                              vir_force,
              const t_mdatoms*                    mdatoms,
              gmx_enerdata_t*                     enerd,
              t_fcdata*                           fcd,
              gmx::ArrayRef<real>                 lambda,
              t_graph*                            graph,
              t_forcerec*                         fr,
              gmx::MdrunScheduleWorkload*         runScheduleWork,
              const gmx_vsite_t*                  vsite,
              rvec                                mu_tot,
              double                              t,
              gmx_edsam*                          ed,
              int                                 legacyFlags,
              const DDBalanceRegionHandler&       ddBalanceRegionHandler)
{
    int                          i, j;
    double                       mu[2 * DIM];
    nonbonded_verlet_t*          nbv      = fr->nbv.get();
    interaction_const_t*         ic       = fr->ic;
    gmx::StatePropagatorDataGpu* stateGpu = fr->stateGpu;

    // TODO remove the code below when the legacy flags are not in use anymore
    /* modify force flag if not doing nonbonded */
    if (!fr->bNonbonded)
    {
        legacyFlags &= ~GMX_FORCE_NONBONDED;
    }

    const SimulationWorkload& simulationWork = runScheduleWork->simulationWork;


    runScheduleWork->stepWork    = setupStepWorkload(legacyFlags, fr->bNonbonded, simulationWork,
                                                  thisRankHasDuty(cr, DUTY_PME));
    const StepWorkload& stepWork = runScheduleWork->stepWork;


    const bool useGpuPmeOnThisRank = simulationWork.useGpuPme && thisRankHasDuty(cr, DUTY_PME);
    const int  pmeFlags            = makePmeFlags(stepWork);

    // Switches on whether to use GPU for position and force buffer operations
    // TODO consider all possible combinations of triggers, and how to combine optimally in each case.
    const BufferOpsUseGpu useGpuXBufOps =
            stepWork.useGpuXBufferOps ? BufferOpsUseGpu::True : BufferOpsUseGpu::False;
    // GPU Force buffer ops are disabled on virial steps, because the virial calc is not yet ported to GPU
    const BufferOpsUseGpu useGpuFBufOps =
            stepWork.useGpuFBufferOps ? BufferOpsUseGpu::True : BufferOpsUseGpu::False;

    /* At a search step we need to start the first balancing region
     * somewhere early inside the step after communication during domain
     * decomposition (and not during the previous step as usual).
     */
    if (stepWork.doNeighborSearch)
    {
        ddBalanceRegionHandler.openBeforeForceComputationCpu(DdAllowBalanceRegionReopen::yes);
    }

    const int start  = 0;
    const int homenr = mdatoms->homenr;

    clear_mat(vir_force);

    if (stepWork.stateChanged)
    {
        if (inputrecNeedMutot(inputrec))
        {
            /* Calculate total (local) dipole moment in a temporary common array.
             * This makes it possible to sum them over nodes faster.
             */
            calc_mu(start, homenr, x.unpaddedArrayRef(), mdatoms->chargeA, mdatoms->chargeB,
                    mdatoms->nChargePerturbed, mu, mu + DIM);
        }
    }

    if (fr->ePBC != epbcNONE)
    {
        /* Compute shift vectors every step,
         * because of pressure coupling or box deformation!
         */
        if (stepWork.haveDynamicBox && stepWork.stateChanged)
        {
            calc_shifts(box, fr->shift_vec);
        }

        const bool fillGrid = (stepWork.doNeighborSearch && stepWork.stateChanged);
        const bool calcCGCM = (fillGrid && !DOMAINDECOMP(cr));
        if (calcCGCM)
        {
            put_atoms_in_box_omp(fr->ePBC, box, x.unpaddedArrayRef().subArray(0, homenr),
                                 gmx_omp_nthreads_get(emntDefault));
            inc_nrnb(nrnb, eNR_SHIFTX, homenr);
        }
        else if (EI_ENERGY_MINIMIZATION(inputrec->eI) && graph)
        {
            unshift_self(graph, box, as_rvec_array(x.unpaddedArrayRef().data()));
        }
    }

    nbnxn_atomdata_copy_shiftvec(stepWork.haveDynamicBox, fr->shift_vec, nbv->nbat.get());

#if GMX_MPI
    const bool pmeSendCoordinatesFromGpu =
            simulationWork.useGpuPmePpCommunication && !(stepWork.doNeighborSearch);
    const bool reinitGpuPmePpComms =
            simulationWork.useGpuPmePpCommunication && (stepWork.doNeighborSearch);
#endif

    const auto localXReadyOnDevice = (stateGpu != nullptr)
                                             ? stateGpu->getCoordinatesReadyOnDeviceEvent(
                                                       AtomLocality::Local, simulationWork, stepWork)
                                             : nullptr;

#if GMX_MPI
    // If coordinates are to be sent to PME task from CPU memory, perform that send here.
    // Otherwise the send will occur after H2D coordinate transfer.
    if (!thisRankHasDuty(cr, DUTY_PME) && !pmeSendCoordinatesFromGpu)
    {
        /* Send particle coordinates to the pme nodes.
         * Since this is only implemented for domain decomposition
         * and domain decomposition does not use the graph,
         * we do not need to worry about shifting.
         */
        if (!stepWork.doNeighborSearch && simulationWork.useGpuUpdate)
        {
            GMX_RELEASE_ASSERT(false,
                               "GPU update and separate PME ranks are only supported with GPU "
                               "direct communication!");
            // TODO: when this code-path becomes supported add:
            // stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
        }

        gmx_pme_send_coordinates(fr, cr, box, as_rvec_array(x.unpaddedArrayRef().data()), lambda[efptCOUL],
                                 lambda[efptVDW], (stepWork.computeVirial || stepWork.computeEnergy),
                                 step, simulationWork.useGpuPmePpCommunication, reinitGpuPmePpComms,
                                 pmeSendCoordinatesFromGpu, localXReadyOnDevice, wcycle);
    }
#endif /* GMX_MPI */

    // Coordinates on the device are needed if PME or BufferOps are offloaded.
    // The local coordinates can be copied right away.
    // NOTE: Consider moving this copy to right after they are updated and constrained,
    //       if the later is not offloaded.
    if (useGpuPmeOnThisRank || useGpuXBufOps == BufferOpsUseGpu::True)
    {
        if (stepWork.doNeighborSearch)
        {
            stateGpu->reinit(mdatoms->homenr,
                             cr->dd != nullptr ? dd_numAtomsZones(*cr->dd) : mdatoms->homenr);
            if (useGpuPmeOnThisRank)
            {
                // TODO: This should be moved into PME setup function ( pme_gpu_prepare_computation(...) )
                pme_gpu_set_device_x(fr->pmedata, stateGpu->getCoordinates());
            }
        }
        // We need to copy coordinates when:
        // 1. Update is not offloaded
        // 2. The buffers were reinitialized on search step
        if (!simulationWork.useGpuUpdate || stepWork.doNeighborSearch)
        {
            GMX_ASSERT(stateGpu != nullptr, "stateGpu should not be null");
            stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(), AtomLocality::Local);
        }
    }

    // TODO Update this comment when introducing SimulationWorkload
    //
    // The conditions for gpuHaloExchange e.g. using GPU buffer
    // operations were checked before construction, so here we can
    // just use it and assert upon any conditions.
    gmx::GpuHaloExchange* gpuHaloExchange =
            (havePPDomainDecomposition(cr) ? cr->dd->gpuHaloExchange.get() : nullptr);
    const bool ddUsesGpuDirectCommunication = (gpuHaloExchange != nullptr);
    GMX_ASSERT(!ddUsesGpuDirectCommunication || (useGpuXBufOps == BufferOpsUseGpu::True),
               "Must use coordinate buffer ops with GPU halo exchange");
    const bool useGpuForcesHaloExchange =
            ddUsesGpuDirectCommunication && (useGpuFBufOps == BufferOpsUseGpu::True);

    // Copy coordinate from the GPU if update is on the GPU and there
    // are forces to be computed on the CPU, or for the computation of
    // virial, or if host-side data will be transferred from this task
    // to a remote task for halo exchange or PME-PP communication. At
    // search steps the current coordinates are already on the host,
    // hence copy is not needed.
    const bool haveHostPmePpComms =
            !thisRankHasDuty(cr, DUTY_PME) && !simulationWork.useGpuPmePpCommunication;
    const bool haveHostHaloExchangeComms = havePPDomainDecomposition(cr) && !ddUsesGpuDirectCommunication;

    bool gmx_used_in_debug haveCopiedXFromGpu = false;
    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch
        && (runScheduleWork->domainWork.haveCpuLocalForceWork || stepWork.computeVirial
            || haveHostPmePpComms || haveHostHaloExchangeComms))
    {
        GMX_ASSERT(stateGpu != nullptr, "stateGpu should not be null");
        stateGpu->copyCoordinatesFromGpu(x.unpaddedArrayRef(), AtomLocality::Local);
        haveCopiedXFromGpu = true;
    }

#if GMX_MPI
    // If coordinates are to be sent to PME task from GPU memory, perform that send here.
    // Otherwise the send will occur before the H2D coordinate transfer.
    if (!thisRankHasDuty(cr, DUTY_PME) && pmeSendCoordinatesFromGpu)
    {
        /* Send particle coordinates to the pme nodes.
         * Since this is only implemented for domain decomposition
         * and domain decomposition does not use the graph,
         * we do not need to worry about shifting.
         */
        gmx_pme_send_coordinates(fr, cr, box, as_rvec_array(x.unpaddedArrayRef().data()), lambda[efptCOUL],
                                 lambda[efptVDW], (stepWork.computeVirial || stepWork.computeEnergy),
                                 step, simulationWork.useGpuPmePpCommunication, reinitGpuPmePpComms,
                                 pmeSendCoordinatesFromGpu, localXReadyOnDevice, wcycle);
    }
#endif /* GMX_MPI */

    if (useGpuPmeOnThisRank)
    {
        launchPmeGpuSpread(fr->pmedata, box, stepWork, pmeFlags, localXReadyOnDevice, wcycle);
    }

    /* do gridding for pair search */
    if (stepWork.doNeighborSearch)
    {
        if (graph && stepWork.stateChanged)
        {
            /* Calculate intramolecular shift vectors to make molecules whole */
            mk_mshift(fplog, graph, fr->ePBC, box, as_rvec_array(x.unpaddedArrayRef().data()));
        }

        // TODO
        // - vzero is constant, do we need to pass it?
        // - box_diag should be passed directly to nbnxn_put_on_grid
        //
        rvec vzero;
        clear_rvec(vzero);

        rvec box_diag;
        box_diag[XX] = box[XX][XX];
        box_diag[YY] = box[YY][YY];
        box_diag[ZZ] = box[ZZ][ZZ];

        wallcycle_start(wcycle, ewcNS);
        if (!DOMAINDECOMP(cr))
        {
            wallcycle_sub_start(wcycle, ewcsNBS_GRID_LOCAL);
            nbnxn_put_on_grid(nbv, box, 0, vzero, box_diag, nullptr, { 0, mdatoms->homenr }, -1,
                              fr->cginfo, x.unpaddedArrayRef(), 0, nullptr);
            wallcycle_sub_stop(wcycle, ewcsNBS_GRID_LOCAL);
        }
        else
        {
            wallcycle_sub_start(wcycle, ewcsNBS_GRID_NONLOCAL);
            nbnxn_put_on_grid_nonlocal(nbv, domdec_zones(cr->dd), fr->cginfo, x.unpaddedArrayRef());
            wallcycle_sub_stop(wcycle, ewcsNBS_GRID_NONLOCAL);
        }

        nbv->setAtomProperties(*mdatoms, fr->cginfo);

        wallcycle_stop(wcycle, ewcNS);

        /* initialize the GPU nbnxm atom data and bonded data structures */
        if (simulationWork.useGpuNonbonded)
        {
            wallcycle_start_nocount(wcycle, ewcLAUNCH_GPU);

            wallcycle_sub_start_nocount(wcycle, ewcsLAUNCH_GPU_NONBONDED);
            Nbnxm::gpu_init_atomdata(nbv->gpu_nbv, nbv->nbat.get());
            wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_NONBONDED);

            if (fr->gpuBonded)
            {
                /* Now we put all atoms on the grid, we can assign bonded
                 * interactions to the GPU, where the grid order is
                 * needed. Also the xq, f and fshift device buffers have
                 * been reallocated if needed, so the bonded code can
                 * learn about them. */
                // TODO the xq, f, and fshift buffers are now shared
                // resources, so they should be maintained by a
                // higher-level object than the nb module.
                fr->gpuBonded->updateInteractionListsAndDeviceBuffers(
                        nbv->getGridIndices(), top->idef, Nbnxm::gpu_get_xq(nbv->gpu_nbv),
                        Nbnxm::gpu_get_f(nbv->gpu_nbv), Nbnxm::gpu_get_fshift(nbv->gpu_nbv));
            }
            wallcycle_stop(wcycle, ewcLAUNCH_GPU);
        }
    }

    if (stepWork.doNeighborSearch)
    {
        // Need to run after the GPU-offload bonded interaction lists
        // are set up to be able to determine whether there is bonded work.
        runScheduleWork->domainWork = setupDomainLifetimeWorkload(
                *inputrec, *fr, pull_work, ed, top->idef, *fcd, *mdatoms, simulationWork, stepWork);

        wallcycle_start_nocount(wcycle, ewcNS);
        wallcycle_sub_start(wcycle, ewcsNBS_SEARCH_LOCAL);
        /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
        nbv->constructPairlist(InteractionLocality::Local, &top->excls, step, nrnb);

        nbv->setupGpuShortRangeWork(fr->gpuBonded, InteractionLocality::Local);

        wallcycle_sub_stop(wcycle, ewcsNBS_SEARCH_LOCAL);
        wallcycle_stop(wcycle, ewcNS);

        if (useGpuXBufOps == BufferOpsUseGpu::True)
        {
            nbv->atomdata_init_copy_x_to_nbat_x_gpu();
        }
        // For force buffer ops, we use the below conditon rather than
        // useGpuFBufOps to ensure that init is performed even if this
        // NS step is also a virial step (on which f buf ops are deactivated).
        if (simulationWork.useGpuBufferOps && simulationWork.useGpuNonbonded && (GMX_GPU == GMX_GPU_CUDA))
        {
            GMX_ASSERT(stateGpu, "stateGpu should be valid when buffer ops are offloaded");
            nbv->atomdata_init_add_nbat_f_to_f_gpu(stateGpu->fReducedOnDevice());
        }
    }
    else if (!EI_TPI(inputrec->eI))
    {
        if (useGpuXBufOps == BufferOpsUseGpu::True)
        {
            GMX_ASSERT(stateGpu, "stateGpu should be valid when buffer ops are offloaded");
            nbv->convertCoordinatesGpu(AtomLocality::Local, false, stateGpu->getCoordinates(),
                                       localXReadyOnDevice);
        }
        else
        {
            if (simulationWork.useGpuUpdate)
            {
                GMX_ASSERT(stateGpu, "need a valid stateGpu object");
                GMX_ASSERT(haveCopiedXFromGpu,
                           "a wait should only be triggered if copy has been scheduled");
                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
            }
            nbv->convertCoordinates(AtomLocality::Local, false, x.unpaddedArrayRef());
        }
    }

    const gmx::DomainLifetimeWorkload& domainWork = runScheduleWork->domainWork;

    if (simulationWork.useGpuNonbonded)
    {
        ddBalanceRegionHandler.openBeforeForceComputationGpu();

        wallcycle_start(wcycle, ewcLAUNCH_GPU);

        wallcycle_sub_start(wcycle, ewcsLAUNCH_GPU_NONBONDED);
        Nbnxm::gpu_upload_shiftvec(nbv->gpu_nbv, nbv->nbat.get());
        if (stepWork.doNeighborSearch || (useGpuXBufOps == BufferOpsUseGpu::False))
        {
            Nbnxm::gpu_copy_xq_to_gpu(nbv->gpu_nbv, nbv->nbat.get(), AtomLocality::Local);
        }
        wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_NONBONDED);
        // with X buffer ops offloaded to the GPU on all but the search steps

        // bonded work not split into separate local and non-local, so with DD
        // we can only launch the kernel after non-local coordinates have been received.
        if (domainWork.haveGpuBondedWork && !havePPDomainDecomposition(cr))
        {
            wallcycle_sub_start(wcycle, ewcsLAUNCH_GPU_BONDED);
            fr->gpuBonded->launchKernel(fr, stepWork, box);
            wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_BONDED);
        }

        /* launch local nonbonded work on GPU */
        wallcycle_sub_start_nocount(wcycle, ewcsLAUNCH_GPU_NONBONDED);
        do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::Local, enbvClearFNo, step, nrnb, wcycle);
        wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_NONBONDED);
        wallcycle_stop(wcycle, ewcLAUNCH_GPU);
    }

    if (useGpuPmeOnThisRank)
    {
        // In PME GPU and mixed mode we launch FFT / gather after the
        // X copy/transform to allow overlap as well as after the GPU NB
        // launch to avoid FFT launch overhead hijacking the CPU and delaying
        // the nonbonded kernel.
        launchPmeGpuFftAndGather(fr->pmedata, wcycle);
    }

    /* Communicate coordinates and sum dipole if necessary +
       do non-local pair search */
    if (havePPDomainDecomposition(cr))
    {
        if (stepWork.doNeighborSearch)
        {
            // TODO: fuse this branch with the above large stepWork.doNeighborSearch block
            wallcycle_start_nocount(wcycle, ewcNS);
            wallcycle_sub_start(wcycle, ewcsNBS_SEARCH_NONLOCAL);
            /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
            nbv->constructPairlist(InteractionLocality::NonLocal, &top->excls, step, nrnb);

            nbv->setupGpuShortRangeWork(fr->gpuBonded, InteractionLocality::NonLocal);
            wallcycle_sub_stop(wcycle, ewcsNBS_SEARCH_NONLOCAL);
            wallcycle_stop(wcycle, ewcNS);
            if (ddUsesGpuDirectCommunication)
            {
                gpuHaloExchange->reinitHalo(stateGpu->getCoordinates(), stateGpu->getForces());
            }
        }
        else
        {
            if (ddUsesGpuDirectCommunication)
            {
                // The following must be called after local setCoordinates (which records an event
                // when the coordinate data has been copied to the device).
                gpuHaloExchange->communicateHaloCoordinates(box, localXReadyOnDevice);

                if (domainWork.haveCpuBondedWork || domainWork.haveFreeEnergyWork)
                {
                    // non-local part of coordinate buffer must be copied back to host for CPU work
                    stateGpu->copyCoordinatesFromGpu(x.unpaddedArrayRef(), AtomLocality::NonLocal);
                }
            }
            else
            {
                // Note: GPU update + DD without direct communication is not supported,
                // a waitCoordinatesReadyOnHost() should be issued if it will be.
                GMX_ASSERT(!simulationWork.useGpuUpdate,
                           "GPU update is not supported with CPU halo exchange");
                dd_move_x(cr->dd, box, x.unpaddedArrayRef(), wcycle);
            }

            if (useGpuXBufOps == BufferOpsUseGpu::True)
            {
                // The condition here was (pme != nullptr && pme_gpu_get_device_x(fr->pmedata) != nullptr)
                if (!useGpuPmeOnThisRank && !ddUsesGpuDirectCommunication)
                {
                    stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(), AtomLocality::NonLocal);
                }
                nbv->convertCoordinatesGpu(AtomLocality::NonLocal, false, stateGpu->getCoordinates(),
                                           stateGpu->getCoordinatesReadyOnDeviceEvent(
                                                   AtomLocality::NonLocal, simulationWork, stepWork));
            }
            else
            {
                nbv->convertCoordinates(AtomLocality::NonLocal, false, x.unpaddedArrayRef());
            }
        }

        if (simulationWork.useGpuNonbonded)
        {
            wallcycle_start(wcycle, ewcLAUNCH_GPU);

            if (stepWork.doNeighborSearch || (useGpuXBufOps == BufferOpsUseGpu::False))
            {
                wallcycle_sub_start(wcycle, ewcsLAUNCH_GPU_NONBONDED);
                Nbnxm::gpu_copy_xq_to_gpu(nbv->gpu_nbv, nbv->nbat.get(), AtomLocality::NonLocal);
                wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_NONBONDED);
            }

            if (domainWork.haveGpuBondedWork)
            {
                wallcycle_sub_start(wcycle, ewcsLAUNCH_GPU_BONDED);
                fr->gpuBonded->launchKernel(fr, stepWork, box);
                wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_BONDED);
            }

            /* launch non-local nonbonded tasks on GPU */
            wallcycle_sub_start(wcycle, ewcsLAUNCH_GPU_NONBONDED);
            do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::NonLocal, enbvClearFNo, step,
                         nrnb, wcycle);
            wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_NONBONDED);

            wallcycle_stop(wcycle, ewcLAUNCH_GPU);
        }
    }

    if (simulationWork.useGpuNonbonded)
    {
        /* launch D2H copy-back F */
        wallcycle_start_nocount(wcycle, ewcLAUNCH_GPU);
        wallcycle_sub_start_nocount(wcycle, ewcsLAUNCH_GPU_NONBONDED);

        if (havePPDomainDecomposition(cr))
        {
            Nbnxm::gpu_launch_cpyback(nbv->gpu_nbv, nbv->nbat.get(), stepWork, AtomLocality::NonLocal);
        }
        Nbnxm::gpu_launch_cpyback(nbv->gpu_nbv, nbv->nbat.get(), stepWork, AtomLocality::Local);
        wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_NONBONDED);

        if (domainWork.haveGpuBondedWork && stepWork.computeEnergy)
        {
            fr->gpuBonded->launchEnergyTransfer();
        }
        wallcycle_stop(wcycle, ewcLAUNCH_GPU);
    }

    if (stepWork.stateChanged && inputrecNeedMutot(inputrec))
    {
        if (PAR(cr))
        {
            gmx_sumd(2 * DIM, mu, cr);

            ddBalanceRegionHandler.reopenRegionCpu();
        }

        for (i = 0; i < 2; i++)
        {
            for (j = 0; j < DIM; j++)
            {
                fr->mu_tot[i][j] = mu[i * DIM + j];
            }
        }
    }
    if (mdatoms->nChargePerturbed == 0)
    {
        copy_rvec(fr->mu_tot[0], mu_tot);
    }
    else
    {
        for (j = 0; j < DIM; j++)
        {
            mu_tot[j] = (1.0 - lambda[efptCOUL]) * fr->mu_tot[0][j] + lambda[efptCOUL] * fr->mu_tot[1][j];
        }
    }

    /* Reset energies */
    reset_enerdata(enerd);
    /* Clear the shift forces */
    // TODO: This should be linked to the shift force buffer in use, or cleared before use instead
    for (gmx::RVec& elem : fr->shiftForces)
    {
        elem = { 0.0_real, 0.0_real, 0.0_real };
    }

    if (DOMAINDECOMP(cr) && !thisRankHasDuty(cr, DUTY_PME))
    {
        wallcycle_start(wcycle, ewcPPDURINGPME);
        dd_force_flop_start(cr->dd, nrnb);
    }

    // For the rest of the CPU tasks that depend on GPU-update produced coordinates,
    // this wait ensures that the D2H transfer is complete.
    if ((simulationWork.useGpuUpdate)
        && (runScheduleWork->domainWork.haveCpuLocalForceWork || stepWork.computeVirial))
    {
        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
    }

    if (inputrec->bRot)
    {
        wallcycle_start(wcycle, ewcROT);
        do_rotation(cr, enforcedRotation, box, as_rvec_array(x.unpaddedArrayRef().data()), t, step,
                    stepWork.doNeighborSearch);
        wallcycle_stop(wcycle, ewcROT);
    }

    /* Start the force cycle counter.
     * Note that a different counter is used for dynamic load balancing.
     */
    wallcycle_start(wcycle, ewcFORCE);

    // Set up and clear force outputs.
    // We use std::move to keep the compiler happy, it has no effect.
    ForceOutputs forceOut =
            setupForceOutputs(fr, pull_work, *inputrec, std::move(force), stepWork, wcycle);

    /* We calculate the non-bonded forces, when done on the CPU, here.
     * We do this before calling do_force_lowlevel, because in that
     * function, the listed forces are calculated before PME, which
     * does communication.  With this order, non-bonded and listed
     * force calculation imbalance can be balanced out by the domain
     * decomposition load balancing.
     */

    const bool useOrEmulateGpuNb = simulationWork.useGpuNonbonded || fr->nbv->emulateGpu();

    if (!useOrEmulateGpuNb)
    {
        do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::Local, enbvClearFYes, step, nrnb, wcycle);
    }

    if (fr->efep != efepNO)
    {
        /* Calculate the local and non-local free energy interactions here.
         * Happens here on the CPU both with and without GPU.
         */
        nbv->dispatchFreeEnergyKernel(InteractionLocality::Local, fr,
                                      as_rvec_array(x.unpaddedArrayRef().data()),
                                      &forceOut.forceWithShiftForces(), *mdatoms, inputrec->fepvals,
                                      lambda.data(), enerd, stepWork, nrnb);

        if (havePPDomainDecomposition(cr))
        {
            nbv->dispatchFreeEnergyKernel(InteractionLocality::NonLocal, fr,
                                          as_rvec_array(x.unpaddedArrayRef().data()),
                                          &forceOut.forceWithShiftForces(), *mdatoms,
                                          inputrec->fepvals, lambda.data(), enerd, stepWork, nrnb);
        }
    }

    if (!useOrEmulateGpuNb)
    {
        if (havePPDomainDecomposition(cr))
        {
            do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::NonLocal, enbvClearFNo, step,
                         nrnb, wcycle);
        }

        if (stepWork.computeForces)
        {
            /* Add all the non-bonded force to the normal force array.
             * This can be split into a local and a non-local part when overlapping
             * communication with calculation with domain decomposition.
             */
            wallcycle_stop(wcycle, ewcFORCE);
            nbv->atomdata_add_nbat_f_to_f(AtomLocality::All, forceOut.forceWithShiftForces().force());
            wallcycle_start_nocount(wcycle, ewcFORCE);
        }

        /* If there are multiple fshift output buffers we need to reduce them */
        if (stepWork.computeVirial)
        {
            /* This is not in a subcounter because it takes a
               negligible and constant-sized amount of time */
            nbnxn_atomdata_add_nbat_fshift_to_fshift(*nbv->nbat,
                                                     forceOut.forceWithShiftForces().shiftForces());
        }
    }

    /* update QMMMrec, if necessary */
    if (fr->bQMMM)
    {
        update_QMMMrec(cr, fr, as_rvec_array(x.unpaddedArrayRef().data()), mdatoms, box);
    }

    // TODO Force flags should include haveFreeEnergyWork for this domain
    if (ddUsesGpuDirectCommunication && (domainWork.haveCpuBondedWork || domainWork.haveFreeEnergyWork))
    {
        /* Wait for non-local coordinate data to be copied from device */
        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::NonLocal);
    }
    /* Compute the bonded and non-bonded energies and optionally forces */
    do_force_lowlevel(fr, inputrec, &(top->idef), cr, ms, nrnb, wcycle, mdatoms, x, hist, &forceOut, enerd,
                      fcd, box, lambda.data(), graph, fr->mu_tot, stepWork, ddBalanceRegionHandler);

    wallcycle_stop(wcycle, ewcFORCE);

    computeSpecialForces(fplog, cr, inputrec, awh, enforcedRotation, imdSession, pull_work, step, t,
                         wcycle, fr->forceProviders, box, x.unpaddedArrayRef(), mdatoms, lambda.data(),
                         stepWork, &forceOut.forceWithVirial(), enerd, ed, stepWork.doNeighborSearch);


    // Will store the amount of cycles spent waiting for the GPU that
    // will be later used in the DLB accounting.
    float cycles_wait_gpu = 0;
    if (useOrEmulateGpuNb)
    {
        auto& forceWithShiftForces = forceOut.forceWithShiftForces();

        /* wait for non-local forces (or calculate in emulation mode) */
        if (havePPDomainDecomposition(cr))
        {
            if (simulationWork.useGpuNonbonded)
            {
                cycles_wait_gpu += Nbnxm::gpu_wait_finish_task(
                        nbv->gpu_nbv, stepWork, AtomLocality::NonLocal, enerd->grpp.ener[egLJSR].data(),
                        enerd->grpp.ener[egCOULSR].data(), forceWithShiftForces.shiftForces(), wcycle);
            }
            else
            {
                wallcycle_start_nocount(wcycle, ewcFORCE);
                do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::NonLocal, enbvClearFYes,
                             step, nrnb, wcycle);
                wallcycle_stop(wcycle, ewcFORCE);
            }

            if (useGpuFBufOps == BufferOpsUseGpu::True)
            {
                gmx::FixedCapacityVector<GpuEventSynchronizer*, 1> dependencyList;

                // TODO: move this into DomainLifetimeWorkload, including the second part of the
                // condition The bonded and free energy CPU tasks can have non-local force
                // contributions which are a dependency for the GPU force reduction.
                bool haveNonLocalForceContribInCpuBuffer =
                        domainWork.haveCpuBondedWork || domainWork.haveFreeEnergyWork;

                if (haveNonLocalForceContribInCpuBuffer)
                {
                    stateGpu->copyForcesToGpu(forceOut.forceWithShiftForces().force(),
                                              AtomLocality::NonLocal);
                    dependencyList.push_back(stateGpu->getForcesReadyOnDeviceEvent(
                            AtomLocality::NonLocal, useGpuFBufOps == BufferOpsUseGpu::True));
                }

                nbv->atomdata_add_nbat_f_to_f_gpu(AtomLocality::NonLocal, stateGpu->getForces(),
                                                  pme_gpu_get_device_f(fr->pmedata), dependencyList,
                                                  false, haveNonLocalForceContribInCpuBuffer);
                if (!useGpuForcesHaloExchange)
                {
                    // copy from GPU input for dd_move_f()
                    stateGpu->copyForcesFromGpu(forceOut.forceWithShiftForces().force(),
                                                AtomLocality::NonLocal);
                }
            }
            else
            {
                nbv->atomdata_add_nbat_f_to_f(AtomLocality::NonLocal, forceWithShiftForces.force());
            }


            if (fr->nbv->emulateGpu() && stepWork.computeVirial)
            {
                nbnxn_atomdata_add_nbat_fshift_to_fshift(*nbv->nbat, forceWithShiftForces.shiftForces());
            }
        }
    }

    if (havePPDomainDecomposition(cr))
    {
        /* We are done with the CPU compute.
         * We will now communicate the non-local forces.
         * If we use a GPU this will overlap with GPU work, so in that case
         * we do not close the DD force balancing region here.
         */
        ddBalanceRegionHandler.closeAfterForceComputationCpu();

        if (stepWork.computeForces)
        {

            if (useGpuForcesHaloExchange)
            {
                if (domainWork.haveCpuLocalForceWork)
                {
                    stateGpu->copyForcesToGpu(forceOut.forceWithShiftForces().force(), AtomLocality::Local);
                }
                gpuHaloExchange->communicateHaloForces(domainWork.haveCpuLocalForceWork);
            }
            else
            {
                if (useGpuFBufOps == BufferOpsUseGpu::True)
                {
                    stateGpu->waitForcesReadyOnHost(AtomLocality::NonLocal);
                }
                dd_move_f(cr->dd, &forceOut.forceWithShiftForces(), wcycle);
            }
        }
    }

    // With both nonbonded and PME offloaded a GPU on the same rank, we use
    // an alternating wait/reduction scheme.
    bool alternateGpuWait = (!c_disableAlternatingWait && useGpuPmeOnThisRank && simulationWork.useGpuNonbonded
                             && !DOMAINDECOMP(cr) && (useGpuFBufOps == BufferOpsUseGpu::False));
    if (alternateGpuWait)
    {
        alternatePmeNbGpuWaitReduce(fr->nbv.get(), fr->pmedata, &forceOut, enerd, stepWork, pmeFlags, wcycle);
    }

    if (!alternateGpuWait && useGpuPmeOnThisRank)
    {
        pme_gpu_wait_and_reduce(fr->pmedata, pmeFlags, wcycle, &forceOut.forceWithVirial(), enerd);
    }

    /* Wait for local GPU NB outputs on the non-alternating wait path */
    if (!alternateGpuWait && simulationWork.useGpuNonbonded)
    {
        /* Measured overhead on CUDA and OpenCL with(out) GPU sharing
         * is between 0.5 and 1.5 Mcycles. So 2 MCycles is an overestimate,
         * but even with a step of 0.1 ms the difference is less than 1%
         * of the step time.
         */
        const float gpuWaitApiOverheadMargin = 2e6F; /* cycles */
        const float waitCycles               = Nbnxm::gpu_wait_finish_task(
                nbv->gpu_nbv, stepWork, AtomLocality::Local, enerd->grpp.ener[egLJSR].data(),
                enerd->grpp.ener[egCOULSR].data(), forceOut.forceWithShiftForces().shiftForces(), wcycle);

        if (ddBalanceRegionHandler.useBalancingRegion())
        {
            DdBalanceRegionWaitedForGpu waitedForGpu = DdBalanceRegionWaitedForGpu::yes;
            if (stepWork.computeForces && waitCycles <= gpuWaitApiOverheadMargin)
            {
                /* We measured few cycles, it could be that the kernel
                 * and transfer finished earlier and there was no actual
                 * wait time, only API call overhead.
                 * Then the actual time could be anywhere between 0 and
                 * cycles_wait_est. We will use half of cycles_wait_est.
                 */
                waitedForGpu = DdBalanceRegionWaitedForGpu::no;
            }
            ddBalanceRegionHandler.closeAfterForceComputationGpu(cycles_wait_gpu, waitedForGpu);
        }
    }

    if (fr->nbv->emulateGpu())
    {
        // NOTE: emulation kernel is not included in the balancing region,
        // but emulation mode does not target performance anyway
        wallcycle_start_nocount(wcycle, ewcFORCE);
        do_nb_verlet(fr, ic, enerd, stepWork, InteractionLocality::Local,
                     DOMAINDECOMP(cr) ? enbvClearFNo : enbvClearFYes, step, nrnb, wcycle);
        wallcycle_stop(wcycle, ewcFORCE);
    }

    // If on GPU PME-PP comms or GPU update path, receive forces from PME before GPU buffer ops
    // TODO refactor this and unify with below default-path call to the same function
    if (PAR(cr) && !thisRankHasDuty(cr, DUTY_PME)
        && (simulationWork.useGpuPmePpCommunication || simulationWork.useGpuUpdate))
    {
        /* In case of node-splitting, the PP nodes receive the long-range
         * forces, virial and energy from the PME nodes here.
         */
        pme_receive_force_ener(fr, cr, &forceOut.forceWithVirial(), enerd,
                               simulationWork.useGpuPmePpCommunication,
                               stepWork.useGpuPmeFReduction, wcycle);
    }


    /* Do the nonbonded GPU (or emulation) force buffer reduction
     * on the non-alternating path. */
    if (useOrEmulateGpuNb && !alternateGpuWait)
    {
        // TODO simplify the below conditionals. Pass buffer and sync pointers at init stage rather than here. Unify getter fns for sameGPU/otherGPU cases.
        void* pmeForcePtr =
                stepWork.useGpuPmeFReduction
                        ? (thisRankHasDuty(cr, DUTY_PME) ? pme_gpu_get_device_f(fr->pmedata)
                                                         : // PME force buffer on same GPU
                                   fr->pmePpCommGpu->getGpuForceStagingPtr()) // buffer received from other GPU
                        : nullptr; // PME reduction not active on GPU

        GpuEventSynchronizer* const pmeSynchronizer =
                stepWork.useGpuPmeFReduction
                        ? (thisRankHasDuty(cr, DUTY_PME) ? pme_gpu_get_f_ready_synchronizer(fr->pmedata)
                                                         : // PME force buffer on same GPU
                                   static_cast<GpuEventSynchronizer*>(
                                           fr->pmePpCommGpu->getForcesReadySynchronizer())) // buffer received from other GPU
                        : nullptr; // PME reduction not active on GPU

        gmx::FixedCapacityVector<GpuEventSynchronizer*, 3> dependencyList;

        if (stepWork.useGpuPmeFReduction)
        {
            dependencyList.push_back(pmeSynchronizer);
        }

        gmx::ArrayRef<gmx::RVec> forceWithShift = forceOut.forceWithShiftForces().force();

        if (useGpuFBufOps == BufferOpsUseGpu::True)
        {
            // Flag to specify whether the CPU force buffer has contributions to
            // local atoms. This depends on whether there are CPU-based force tasks
            // or when DD is active the halo exchange has resulted in contributions
            // from the non-local part.
            const bool haveLocalForceContribInCpuBuffer =
                    (domainWork.haveCpuLocalForceWork || havePPDomainDecomposition(cr));

            // TODO: move these steps as early as possible:
            // - CPU f H2D should be as soon as all CPU-side forces are done
            // - wait for force reduction does not need to block host (at least not here, it's sufficient to wait
            //   before the next CPU task that consumes the forces: vsite spread or update)
            // - copy is not perfomed if GPU force halo exchange is active, because it would overwrite the result
            //   of the halo exchange. In that case the copy is instead performed above, before the exchange.
            //   These should be unified.
            if (haveLocalForceContribInCpuBuffer && !useGpuForcesHaloExchange)
            {
                // Note: AtomLocality::All is used for the non-DD case because, as in this
                // case copyForcesToGpu() uses a separate stream, it allows overlap of
                // CPU force H2D with GPU force tasks on all streams including those in the
                // local stream which would otherwise be implicit dependencies for the
                // transfer and would not overlap.
                auto locality = havePPDomainDecomposition(cr) ? AtomLocality::Local : AtomLocality::All;

                stateGpu->copyForcesToGpu(forceWithShift, locality);
                dependencyList.push_back(stateGpu->getForcesReadyOnDeviceEvent(
                        locality, useGpuFBufOps == BufferOpsUseGpu::True));
            }
            if (useGpuForcesHaloExchange)
            {
                dependencyList.push_back(gpuHaloExchange->getForcesReadyOnDeviceEvent());
            }
            nbv->atomdata_add_nbat_f_to_f_gpu(AtomLocality::Local, stateGpu->getForces(), pmeForcePtr,
                                              dependencyList, stepWork.useGpuPmeFReduction,
                                              haveLocalForceContribInCpuBuffer);
            // Copy forces to host if they are needed for update or if virtual sites are enabled.
            // If there are vsites, we need to copy forces every step to spread vsite forces on host.
            // TODO: When the output flags will be included in step workload, this copy can be combined with the
            //       copy call done in sim_utils(...) for the output.
            // NOTE: If there are virtual sites, the forces are modified on host after this D2H copy. Hence,
            //       they should not be copied in do_md(...) for the output.
            if (!simulationWork.useGpuUpdate || vsite)
            {
                stateGpu->copyForcesFromGpu(forceWithShift, AtomLocality::Local);
                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
            }
        }
        else
        {
            nbv->atomdata_add_nbat_f_to_f(AtomLocality::Local, forceWithShift);
        }
    }

    launchGpuEndOfStepTasks(nbv, fr->gpuBonded, fr->pmedata, enerd, *runScheduleWork,
                            simulationWork.useGpuNonbonded, useGpuPmeOnThisRank, step, wcycle);

    if (DOMAINDECOMP(cr))
    {
        dd_force_flop_stop(cr->dd, nrnb);
    }

    if (stepWork.computeForces)
    {
        rvec* f = as_rvec_array(forceOut.forceWithShiftForces().force().data());

        /* If we have NoVirSum forces, but we do not calculate the virial,
         * we sum fr->f_novirsum=forceOut.f later.
         */
        if (vsite && !(fr->haveDirectVirialContributions && !stepWork.computeVirial))
        {
            rvec* fshift = as_rvec_array(forceOut.forceWithShiftForces().shiftForces().data());
            spread_vsite_f(vsite, as_rvec_array(x.unpaddedArrayRef().data()), f, fshift, FALSE,
                           nullptr, nrnb, &top->idef, fr->ePBC, fr->bMolPBC, graph, box, cr, wcycle);
        }

        if (stepWork.computeVirial)
        {
            /* Calculation of the virial must be done after vsites! */
            calc_virial(0, mdatoms->homenr, as_rvec_array(x.unpaddedArrayRef().data()),
                        forceOut.forceWithShiftForces(), vir_force, graph, box, nrnb, fr, inputrec->ePBC);
        }
    }

    // TODO refactor this and unify with above GPU PME-PP / GPU update path call to the same function
    if (PAR(cr) && !thisRankHasDuty(cr, DUTY_PME) && !simulationWork.useGpuPmePpCommunication
        && !simulationWork.useGpuUpdate)
    {
        /* In case of node-splitting, the PP nodes receive the long-range
         * forces, virial and energy from the PME nodes here.
         */
        pme_receive_force_ener(fr, cr, &forceOut.forceWithVirial(), enerd,
                               simulationWork.useGpuPmePpCommunication, false, wcycle);
    }

    /* DeepMD */
    double               dener;
    std::vector<double > dforce;
    if (useDeepmd)
    {
        if (DIM != 3)
        {
            gmx_fatal(FARGS, "DeepMD does not support DIM < 3.");
        }
        else
        {
            std::vector<int >    dtype  = deepmdPlugin->dtype;
            std::vector<int >    dindex = deepmdPlugin->dindex;
            std::vector<double > dcoord;
            std::vector<double > dvirial;
            
            dcoord.resize(deepmdPlugin->natom * DIM);
            dforce.resize(deepmdPlugin->natom * DIM);
            dvirial.resize(9);
            
            deepmd::DeepPot nnp = deepmdPlugin->nnp;

            /* init coord */
            for (int i = 0; i < deepmdPlugin->natom; i++)
            {
                for (int j = 0; j < DIM; j++)
                {
                    dcoord[i * DIM + j] = as_rvec_array(x.unpaddedArrayRef().data())[dindex[i]][j] / c_dp2gmx;
                    // std::cout << dcoord[i * DIM + j] << " ";
                }
                // std::cout << std::endl;
            }
            /* init coord */

            /* init box */
            std::vector<double > dbox;
            if (deepmdPlugin->pbc)
            {
                dbox.resize(9);
                for (int i = 0; i < DIM; i++)
                {
                    for (int j = 0; j < DIM; j++)
                    {
                        dbox[i * DIM + j] = box[i][j] / c_dp2gmx;
                    }
                }
            }
            else
            {
                dbox = {};
            }
            /* init box */

            nnp.compute(dener, dforce, dvirial, dcoord, dtype, dbox);

            for (int i = 0; i < deepmdPlugin->natom; i++)
            {
                for (int j = 0; j < DIM; j++)
                {
                    as_rvec_array(forceOut.forceWithShiftForces().force().data())[dindex[i]][j] += dforce[i * DIM + j] * f_dp2gmx * deepmdPlugin->lmd;
                }
            }
        }
    }

    if (stepWork.computeForces)
    {
        post_process_forces(cr, step, nrnb, wcycle, top, box, as_rvec_array(x.unpaddedArrayRef().data()),
                            &forceOut, vir_force, mdatoms, graph, fr, vsite, stepWork);
    }

    if (stepWork.computeEnergy)
    {
        /* Sum the potential energy terms from group contributions */
        sum_epot(&(enerd->grpp), enerd->term);
        if (useDeepmd)
        {
            enerd->term[F_EPOT] += dener * e_dp2gmx * deepmdPlugin->lmd;
        }

        if (!EI_TPI(inputrec->eI))
        {
            checkPotentialEnergyValidity(step, *enerd, *inputrec);
        }
    }
    /* In case we don't have constraints and are using GPUs, the next balancing
     * region starts here.
     * Some "special" work at the end of do_force_cuts?, such as vsite spread,
     * virial calculation and COM pulling, is not thus not included in
     * the balance timing, which is ok as most tasks do communication.
     */
    ddBalanceRegionHandler.openBeforeForceComputationCpu(DdAllowBalanceRegionReopen::no);
}
