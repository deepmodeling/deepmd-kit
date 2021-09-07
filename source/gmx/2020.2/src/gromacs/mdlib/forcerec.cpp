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
#include "gmxpre.h"

#include "forcerec.h"

#include "config.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <memory>

#include "gromacs/commandline/filenm.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/ewald/ewald.h"
#include "gromacs/ewald/ewald_utils.h"
#include "gromacs/ewald/pme_pp_comm_gpu.h"
#include "gromacs/fileio/filetypes.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nonbonded/nonbonded.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/hardware/hw_info.h"
#include "gromacs/listed_forces/gpubonded.h"
#include "gromacs/listed_forces/manage_threading.h"
#include "gromacs/listed_forces/pairs.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/math/utilities.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/dispersioncorrection.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/forcerec_threading.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/md_support.h"
#include "gromacs/mdlib/qmmm.h"
#include "gromacs/mdlib/rf_util.h"
#include "gromacs/mdlib/wall.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/nbnxm_geometry.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/tables/forcetable.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/physicalnodecommunicator.h"
#include "gromacs/utility/pleasecite.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/strconvert.h"

#include "gromacs/mdlib/deepmd_plugin.h"

/*! \brief environment variable to enable GPU P2P communication */
static const bool c_enableGpuPmePpComms =
        (getenv("GMX_GPU_PME_PP_COMMS") != nullptr) && GMX_THREAD_MPI && (GMX_GPU == GMX_GPU_CUDA);

static real* mk_nbfp(const gmx_ffparams_t* idef, gmx_bool bBHAM)
{
    real* nbfp;
    int   i, j, k, atnr;

    atnr = idef->atnr;
    if (bBHAM)
    {
        snew(nbfp, 3 * atnr * atnr);
        for (i = k = 0; (i < atnr); i++)
        {
            for (j = 0; (j < atnr); j++, k++)
            {
                BHAMA(nbfp, atnr, i, j) = idef->iparams[k].bham.a;
                BHAMB(nbfp, atnr, i, j) = idef->iparams[k].bham.b;
                /* nbfp now includes the 6.0 derivative prefactor */
                BHAMC(nbfp, atnr, i, j) = idef->iparams[k].bham.c * 6.0;
            }
        }
    }
    else
    {
        snew(nbfp, 2 * atnr * atnr);
        for (i = k = 0; (i < atnr); i++)
        {
            for (j = 0; (j < atnr); j++, k++)
            {
                /* nbfp now includes the 6.0/12.0 derivative prefactors */
                C6(nbfp, atnr, i, j)  = idef->iparams[k].lj.c6 * 6.0;
                C12(nbfp, atnr, i, j) = idef->iparams[k].lj.c12 * 12.0;
            }
        }
    }

    return nbfp;
}

static real* make_ljpme_c6grid(const gmx_ffparams_t* idef, t_forcerec* fr)
{
    int   i, j, k, atnr;
    real  c6, c6i, c6j, c12i, c12j, epsi, epsj, sigmai, sigmaj;
    real* grid;

    /* For LJ-PME simulations, we correct the energies with the reciprocal space
     * inside of the cut-off. To do this the non-bonded kernels needs to have
     * access to the C6-values used on the reciprocal grid in pme.c
     */

    atnr = idef->atnr;
    snew(grid, 2 * atnr * atnr);
    for (i = k = 0; (i < atnr); i++)
    {
        for (j = 0; (j < atnr); j++, k++)
        {
            c6i  = idef->iparams[i * (atnr + 1)].lj.c6;
            c12i = idef->iparams[i * (atnr + 1)].lj.c12;
            c6j  = idef->iparams[j * (atnr + 1)].lj.c6;
            c12j = idef->iparams[j * (atnr + 1)].lj.c12;
            c6   = std::sqrt(c6i * c6j);
            if (fr->ljpme_combination_rule == eljpmeLB && !gmx_numzero(c6) && !gmx_numzero(c12i)
                && !gmx_numzero(c12j))
            {
                sigmai = gmx::sixthroot(c12i / c6i);
                sigmaj = gmx::sixthroot(c12j / c6j);
                epsi   = c6i * c6i / c12i;
                epsj   = c6j * c6j / c12j;
                c6     = std::sqrt(epsi * epsj) * gmx::power6(0.5 * (sigmai + sigmaj));
            }
            /* Store the elements at the same relative positions as C6 in nbfp in order
             * to simplify access in the kernels
             */
            grid[2 * (atnr * i + j)] = c6 * 6.0;
        }
    }
    return grid;
}

enum
{
    acNONE = 0,
    acCONSTRAINT,
    acSETTLE
};

static cginfo_mb_t* init_cginfo_mb(const gmx_mtop_t* mtop, const t_forcerec* fr, gmx_bool* bFEP_NonBonded)
{
    cginfo_mb_t* cginfo_mb;
    gmx_bool*    type_VDW;
    int*         cginfo;
    int*         a_con;

    snew(cginfo_mb, mtop->molblock.size());

    snew(type_VDW, fr->ntype);
    for (int ai = 0; ai < fr->ntype; ai++)
    {
        type_VDW[ai] = FALSE;
        for (int j = 0; j < fr->ntype; j++)
        {
            type_VDW[ai] = type_VDW[ai] || fr->bBHAM || C6(fr->nbfp, fr->ntype, ai, j) != 0
                           || C12(fr->nbfp, fr->ntype, ai, j) != 0;
        }
    }

    *bFEP_NonBonded = FALSE;

    int a_offset = 0;
    for (size_t mb = 0; mb < mtop->molblock.size(); mb++)
    {
        const gmx_molblock_t& molb = mtop->molblock[mb];
        const gmx_moltype_t&  molt = mtop->moltype[molb.type];
        const t_blocka&       excl = molt.excls;

        /* Check if the cginfo is identical for all molecules in this block.
         * If so, we only need an array of the size of one molecule.
         * Otherwise we make an array of #mol times #cgs per molecule.
         */
        gmx_bool bId = TRUE;
        for (int m = 0; m < molb.nmol; m++)
        {
            const int am = m * molt.atoms.nr;
            for (int a = 0; a < molt.atoms.nr; a++)
            {
                if (getGroupType(mtop->groups, SimulationAtomGroupType::QuantumMechanics, a_offset + am + a)
                    != getGroupType(mtop->groups, SimulationAtomGroupType::QuantumMechanics, a_offset + a))
                {
                    bId = FALSE;
                }
                if (!mtop->groups.groupNumbers[SimulationAtomGroupType::QuantumMechanics].empty())
                {
                    if (mtop->groups.groupNumbers[SimulationAtomGroupType::QuantumMechanics][a_offset + am + a]
                        != mtop->groups.groupNumbers[SimulationAtomGroupType::QuantumMechanics][a_offset + a])
                    {
                        bId = FALSE;
                    }
                }
            }
        }

        cginfo_mb[mb].cg_start = a_offset;
        cginfo_mb[mb].cg_end   = a_offset + molb.nmol * molt.atoms.nr;
        cginfo_mb[mb].cg_mod   = (bId ? 1 : molb.nmol) * molt.atoms.nr;
        snew(cginfo_mb[mb].cginfo, cginfo_mb[mb].cg_mod);
        cginfo = cginfo_mb[mb].cginfo;

        /* Set constraints flags for constrained atoms */
        snew(a_con, molt.atoms.nr);
        for (int ftype = 0; ftype < F_NRE; ftype++)
        {
            if (interaction_function[ftype].flags & IF_CONSTRAINT)
            {
                const int nral = NRAL(ftype);
                for (int ia = 0; ia < molt.ilist[ftype].size(); ia += 1 + nral)
                {
                    int a;

                    for (a = 0; a < nral; a++)
                    {
                        a_con[molt.ilist[ftype].iatoms[ia + 1 + a]] =
                                (ftype == F_SETTLE ? acSETTLE : acCONSTRAINT);
                    }
                }
            }
        }

        for (int m = 0; m < (bId ? 1 : molb.nmol); m++)
        {
            const int molculeOffsetInBlock = m * molt.atoms.nr;
            for (int a = 0; a < molt.atoms.nr; a++)
            {
                const t_atom& atom     = molt.atoms.atom[a];
                int&          atomInfo = cginfo[molculeOffsetInBlock + a];

                /* Store the energy group in cginfo */
                int gid = getGroupType(mtop->groups, SimulationAtomGroupType::EnergyOutput,
                                       a_offset + molculeOffsetInBlock + a);
                SET_CGINFO_GID(atomInfo, gid);

                bool bHaveVDW = (type_VDW[atom.type] || type_VDW[atom.typeB]);
                bool bHaveQ   = (atom.q != 0 || atom.qB != 0);

                bool haveExclusions = false;
                /* Loop over all the exclusions of atom ai */
                for (int j = excl.index[a]; j < excl.index[a + 1]; j++)
                {
                    if (excl.a[j] != a)
                    {
                        haveExclusions = true;
                        break;
                    }
                }

                switch (a_con[a])
                {
                    case acCONSTRAINT: SET_CGINFO_CONSTR(atomInfo); break;
                    case acSETTLE: SET_CGINFO_SETTLE(atomInfo); break;
                    default: break;
                }
                if (haveExclusions)
                {
                    SET_CGINFO_EXCL_INTER(atomInfo);
                }
                if (bHaveVDW)
                {
                    SET_CGINFO_HAS_VDW(atomInfo);
                }
                if (bHaveQ)
                {
                    SET_CGINFO_HAS_Q(atomInfo);
                }
                if (fr->efep != efepNO && PERTURBED(atom))
                {
                    SET_CGINFO_FEP(atomInfo);
                    *bFEP_NonBonded = TRUE;
                }
            }
        }

        sfree(a_con);

        a_offset += molb.nmol * molt.atoms.nr;
    }
    sfree(type_VDW);

    return cginfo_mb;
}

static std::vector<int> cginfo_expand(const int nmb, const cginfo_mb_t* cgi_mb)
{
    const int ncg = cgi_mb[nmb - 1].cg_end;

    std::vector<int> cginfo(ncg);

    int mb = 0;
    for (int cg = 0; cg < ncg; cg++)
    {
        while (cg >= cgi_mb[mb].cg_end)
        {
            mb++;
        }
        cginfo[cg] = cgi_mb[mb].cginfo[(cg - cgi_mb[mb].cg_start) % cgi_mb[mb].cg_mod];
    }

    return cginfo;
}

static void done_cginfo_mb(cginfo_mb_t* cginfo_mb, int numMolBlocks)
{
    if (cginfo_mb == nullptr)
    {
        return;
    }
    for (int mb = 0; mb < numMolBlocks; ++mb)
    {
        sfree(cginfo_mb[mb].cginfo);
    }
    sfree(cginfo_mb);
}

/* Sets the sum of charges (squared) and C6 in the system in fr.
 * Returns whether the system has a net charge.
 */
static bool set_chargesum(FILE* log, t_forcerec* fr, const gmx_mtop_t* mtop)
{
    /*This now calculates sum for q and c6*/
    double qsum, q2sum, q, c6sum, c6;

    qsum  = 0;
    q2sum = 0;
    c6sum = 0;
    for (const gmx_molblock_t& molb : mtop->molblock)
    {
        int            nmol  = molb.nmol;
        const t_atoms* atoms = &mtop->moltype[molb.type].atoms;
        for (int i = 0; i < atoms->nr; i++)
        {
            q = atoms->atom[i].q;
            qsum += nmol * q;
            q2sum += nmol * q * q;
            c6 = mtop->ffparams.iparams[atoms->atom[i].type * (mtop->ffparams.atnr + 1)].lj.c6;
            c6sum += nmol * c6;
        }
    }
    fr->qsum[0]  = qsum;
    fr->q2sum[0] = q2sum;
    fr->c6sum[0] = c6sum;

    if (fr->efep != efepNO)
    {
        qsum  = 0;
        q2sum = 0;
        c6sum = 0;
        for (const gmx_molblock_t& molb : mtop->molblock)
        {
            int            nmol  = molb.nmol;
            const t_atoms* atoms = &mtop->moltype[molb.type].atoms;
            for (int i = 0; i < atoms->nr; i++)
            {
                q = atoms->atom[i].qB;
                qsum += nmol * q;
                q2sum += nmol * q * q;
                c6 = mtop->ffparams.iparams[atoms->atom[i].typeB * (mtop->ffparams.atnr + 1)].lj.c6;
                c6sum += nmol * c6;
            }
            fr->qsum[1]  = qsum;
            fr->q2sum[1] = q2sum;
            fr->c6sum[1] = c6sum;
        }
    }
    else
    {
        fr->qsum[1]  = fr->qsum[0];
        fr->q2sum[1] = fr->q2sum[0];
        fr->c6sum[1] = fr->c6sum[0];
    }
    if (log)
    {
        if (fr->efep == efepNO)
        {
            fprintf(log, "System total charge: %.3f\n", fr->qsum[0]);
        }
        else
        {
            fprintf(log, "System total charge, top. A: %.3f top. B: %.3f\n", fr->qsum[0], fr->qsum[1]);
        }
    }

    /* A cut-off of 1e-4 is used to catch rounding errors due to ascii input */
    return (std::abs(fr->qsum[0]) > 1e-4 || std::abs(fr->qsum[1]) > 1e-4);
}

static real calcBuckinghamBMax(FILE* fplog, const gmx_mtop_t* mtop)
{
    const t_atoms *at1, *at2;
    int            i, j, tpi, tpj, ntypes;
    real           b, bmin;

    if (fplog)
    {
        fprintf(fplog, "Determining largest Buckingham b parameter for table\n");
    }
    ntypes = mtop->ffparams.atnr;

    bmin            = -1;
    real bham_b_max = 0;
    for (size_t mt1 = 0; mt1 < mtop->moltype.size(); mt1++)
    {
        at1 = &mtop->moltype[mt1].atoms;
        for (i = 0; (i < at1->nr); i++)
        {
            tpi = at1->atom[i].type;
            if (tpi >= ntypes)
            {
                gmx_fatal(FARGS, "Atomtype[%d] = %d, maximum = %d", i, tpi, ntypes);
            }

            for (size_t mt2 = mt1; mt2 < mtop->moltype.size(); mt2++)
            {
                at2 = &mtop->moltype[mt2].atoms;
                for (j = 0; (j < at2->nr); j++)
                {
                    tpj = at2->atom[j].type;
                    if (tpj >= ntypes)
                    {
                        gmx_fatal(FARGS, "Atomtype[%d] = %d, maximum = %d", j, tpj, ntypes);
                    }
                    b = mtop->ffparams.iparams[tpi * ntypes + tpj].bham.b;
                    if (b > bham_b_max)
                    {
                        bham_b_max = b;
                    }
                    if ((b < bmin) || (bmin == -1))
                    {
                        bmin = b;
                    }
                }
            }
        }
    }
    if (fplog)
    {
        fprintf(fplog, "Buckingham b parameters, min: %g, max: %g\n", bmin, bham_b_max);
    }

    return bham_b_max;
}

/*!\brief If there's bonded interactions of type \c ftype1 or \c
 * ftype2 present in the topology, build an array of the number of
 * interactions present for each bonded interaction index found in the
 * topology.
 *
 * \c ftype1 or \c ftype2 may be set to -1 to disable seeking for a
 * valid type with that parameter.
 *
 * \c count will be reallocated as necessary to fit the largest bonded
 * interaction index found, and its current size will be returned in
 * \c ncount. It will contain zero for every bonded interaction index
 * for which no interactions are present in the topology.
 */
static void count_tables(int ftype1, int ftype2, const gmx_mtop_t* mtop, int* ncount, int** count)
{
    int ftype, i, j, tabnr;

    // Loop over all moleculetypes
    for (const gmx_moltype_t& molt : mtop->moltype)
    {
        // Loop over all interaction types
        for (ftype = 0; ftype < F_NRE; ftype++)
        {
            // If the current interaction type is one of the types whose tables we're trying to count...
            if (ftype == ftype1 || ftype == ftype2)
            {
                const InteractionList& il     = molt.ilist[ftype];
                const int              stride = 1 + NRAL(ftype);
                // ... and there are actually some interactions for this type
                for (i = 0; i < il.size(); i += stride)
                {
                    // Find out which table index the user wanted
                    tabnr = mtop->ffparams.iparams[il.iatoms[i]].tab.table;
                    if (tabnr < 0)
                    {
                        gmx_fatal(FARGS, "A bonded table number is smaller than 0: %d\n", tabnr);
                    }
                    // Make room for this index in the data structure
                    if (tabnr >= *ncount)
                    {
                        srenew(*count, tabnr + 1);
                        for (j = *ncount; j < tabnr + 1; j++)
                        {
                            (*count)[j] = 0;
                        }
                        *ncount = tabnr + 1;
                    }
                    // Record that this table index is used and must have a valid file
                    (*count)[tabnr]++;
                }
            }
        }
    }
}

/*!\brief If there's bonded interactions of flavour \c tabext and type
 * \c ftype1 or \c ftype2 present in the topology, seek them in the
 * list of filenames passed to mdrun, and make bonded tables from
 * those files.
 *
 * \c ftype1 or \c ftype2 may be set to -1 to disable seeking for a
 * valid type with that parameter.
 *
 * A fatal error occurs if no matching filename is found.
 */
static bondedtable_t* make_bonded_tables(FILE*                            fplog,
                                         int                              ftype1,
                                         int                              ftype2,
                                         const gmx_mtop_t*                mtop,
                                         gmx::ArrayRef<const std::string> tabbfnm,
                                         const char*                      tabext)
{
    int            ncount, *count;
    bondedtable_t* tab;

    tab = nullptr;

    ncount = 0;
    count  = nullptr;
    count_tables(ftype1, ftype2, mtop, &ncount, &count);

    // Are there any relevant tabulated bond interactions?
    if (ncount > 0)
    {
        snew(tab, ncount);
        for (int i = 0; i < ncount; i++)
        {
            // Do any interactions exist that requires this table?
            if (count[i] > 0)
            {
                // This pattern enforces the current requirement that
                // table filenames end in a characteristic sequence
                // before the file type extension, and avoids table 13
                // being recognized and used for table 1.
                std::string patternToFind = gmx::formatString("_%s%d.%s", tabext, i, ftp2ext(efXVG));
                bool        madeTable     = false;
                for (gmx::index j = 0; j < tabbfnm.ssize() && !madeTable; ++j)
                {
                    if (gmx::endsWith(tabbfnm[j], patternToFind))
                    {
                        // Finally read the table from the file found
                        tab[i]    = make_bonded_table(fplog, tabbfnm[j].c_str(), NRAL(ftype1) - 2);
                        madeTable = true;
                    }
                }
                if (!madeTable)
                {
                    bool isPlural = (ftype2 != -1);
                    gmx_fatal(FARGS,
                              "Tabulated interaction of type '%s%s%s' with index %d cannot be used "
                              "because no table file whose name matched '%s' was passed via the "
                              "gmx mdrun -tableb command-line option.",
                              interaction_function[ftype1].longname, isPlural ? "' or '" : "",
                              isPlural ? interaction_function[ftype2].longname : "", i,
                              patternToFind.c_str());
                }
            }
        }
        sfree(count);
    }

    return tab;
}

void forcerec_set_ranges(t_forcerec* fr, int natoms_force, int natoms_force_constr, int natoms_f_novirsum)
{
    fr->natoms_force        = natoms_force;
    fr->natoms_force_constr = natoms_force_constr;

    if (fr->natoms_force_constr > fr->nalloc_force)
    {
        fr->nalloc_force = over_alloc_dd(fr->natoms_force_constr);
    }

    if (fr->haveDirectVirialContributions)
    {
        fr->forceBufferForDirectVirialContributions.resize(natoms_f_novirsum);
    }
}

static real cutoff_inf(real cutoff)
{
    if (cutoff == 0)
    {
        cutoff = GMX_CUTOFF_INF;
    }

    return cutoff;
}

/*! \brief Print Coulomb Ewald citations and set ewald coefficients */
static void initCoulombEwaldParameters(FILE*                fp,
                                       const t_inputrec*    ir,
                                       bool                 systemHasNetCharge,
                                       interaction_const_t* ic)
{
    if (!EEL_PME_EWALD(ir->coulombtype))
    {
        return;
    }

    if (fp)
    {
        fprintf(fp, "Will do PME sum in reciprocal space for electrostatic interactions.\n");

        if (ir->coulombtype == eelP3M_AD)
        {
            please_cite(fp, "Hockney1988");
            please_cite(fp, "Ballenegger2012");
        }
        else
        {
            please_cite(fp, "Essmann95a");
        }

        if (ir->ewald_geometry == eewg3DC)
        {
            if (fp)
            {
                fprintf(fp, "Using the Ewald3DC correction for systems with a slab geometry%s.\n",
                        systemHasNetCharge ? " and net charge" : "");
            }
            please_cite(fp, "In-Chul99a");
            if (systemHasNetCharge)
            {
                please_cite(fp, "Ballenegger2009");
            }
        }
    }

    ic->ewaldcoeff_q = calc_ewaldcoeff_q(ir->rcoulomb, ir->ewald_rtol);
    if (fp)
    {
        fprintf(fp, "Using a Gaussian width (1/beta) of %g nm for Ewald\n", 1 / ic->ewaldcoeff_q);
    }

    if (ic->coulomb_modifier == eintmodPOTSHIFT)
    {
        GMX_RELEASE_ASSERT(ic->rcoulomb != 0, "Cutoff radius cannot be zero");
        ic->sh_ewald = std::erfc(ic->ewaldcoeff_q * ic->rcoulomb) / ic->rcoulomb;
    }
    else
    {
        ic->sh_ewald = 0;
    }
}

/*! \brief Print Van der Waals Ewald citations and set ewald coefficients */
static void initVdwEwaldParameters(FILE* fp, const t_inputrec* ir, interaction_const_t* ic)
{
    if (!EVDW_PME(ir->vdwtype))
    {
        return;
    }

    if (fp)
    {
        fprintf(fp, "Will do PME sum in reciprocal space for LJ dispersion interactions.\n");
        please_cite(fp, "Essmann95a");
    }
    ic->ewaldcoeff_lj = calc_ewaldcoeff_lj(ir->rvdw, ir->ewald_rtol_lj);
    if (fp)
    {
        fprintf(fp, "Using a Gaussian width (1/beta) of %g nm for LJ Ewald\n", 1 / ic->ewaldcoeff_lj);
    }

    if (ic->vdw_modifier == eintmodPOTSHIFT)
    {
        real crc2       = gmx::square(ic->ewaldcoeff_lj * ic->rvdw);
        ic->sh_lj_ewald = (std::exp(-crc2) * (1 + crc2 + 0.5 * crc2 * crc2) - 1) / gmx::power6(ic->rvdw);
    }
    else
    {
        ic->sh_lj_ewald = 0;
    }
}

/* Generate Coulomb and/or Van der Waals Ewald long-range correction tables
 *
 * Tables are generated for one or both, depending on if the pointers
 * are non-null. The spacing for both table sets is the same and obeys
 * both accuracy requirements, when relevant.
 */
static void init_ewald_f_table(const interaction_const_t& ic,
                               EwaldCorrectionTables*     coulombTables,
                               EwaldCorrectionTables*     vdwTables)
{
    const bool useCoulombTable = (EEL_PME_EWALD(ic.eeltype) && coulombTables != nullptr);
    const bool useVdwTable     = (EVDW_PME(ic.vdwtype) && vdwTables != nullptr);

    /* Get the Ewald table spacing based on Coulomb and/or LJ
     * Ewald coefficients and rtol.
     */
    const real tableScale = ewald_spline3_table_scale(ic, useCoulombTable, useVdwTable);

    const int tableSize = static_cast<int>(ic.rcoulomb * tableScale) + 2;

    if (useCoulombTable)
    {
        *coulombTables =
                generateEwaldCorrectionTables(tableSize, tableScale, ic.ewaldcoeff_q, v_q_ewald_lr);
    }

    if (useVdwTable)
    {
        *vdwTables = generateEwaldCorrectionTables(tableSize, tableScale, ic.ewaldcoeff_lj, v_lj_ewald_lr);
    }
}

void init_interaction_const_tables(FILE* fp, interaction_const_t* ic)
{
    if (EEL_PME_EWALD(ic->eeltype) || EVDW_PME(ic->vdwtype))
    {
        init_ewald_f_table(*ic, ic->coulombEwaldTables.get(), ic->vdwEwaldTables.get());
        if (fp != nullptr)
        {
            fprintf(fp, "Initialized non-bonded Ewald tables, spacing: %.2e size: %zu\n\n",
                    1 / ic->coulombEwaldTables->scale, ic->coulombEwaldTables->tableF.size());
        }
    }
}

static void clear_force_switch_constants(shift_consts_t* sc)
{
    sc->c2   = 0;
    sc->c3   = 0;
    sc->cpot = 0;
}

static void force_switch_constants(real p, real rsw, real rc, shift_consts_t* sc)
{
    /* Here we determine the coefficient for shifting the force to zero
     * between distance rsw and the cut-off rc.
     * For a potential of r^-p, we have force p*r^-(p+1).
     * But to save flops we absorb p in the coefficient.
     * Thus we get:
     * force/p   = r^-(p+1) + c2*r^2 + c3*r^3
     * potential = r^-p + c2/3*r^3 + c3/4*r^4 + cpot
     */
    sc->c2   = ((p + 1) * rsw - (p + 4) * rc) / (pow(rc, p + 2) * gmx::square(rc - rsw));
    sc->c3   = -((p + 1) * rsw - (p + 3) * rc) / (pow(rc, p + 2) * gmx::power3(rc - rsw));
    sc->cpot = -pow(rc, -p) + p * sc->c2 / 3 * gmx::power3(rc - rsw)
               + p * sc->c3 / 4 * gmx::power4(rc - rsw);
}

static void potential_switch_constants(real rsw, real rc, switch_consts_t* sc)
{
    /* The switch function is 1 at rsw and 0 at rc.
     * The derivative and second derivate are zero at both ends.
     * rsw        = max(r - r_switch, 0)
     * sw         = 1 + c3*rsw^3 + c4*rsw^4 + c5*rsw^5
     * dsw        = 3*c3*rsw^2 + 4*c4*rsw^3 + 5*c5*rsw^4
     * force      = force*dsw - potential*sw
     * potential *= sw
     */
    sc->c3 = -10 / gmx::power3(rc - rsw);
    sc->c4 = 15 / gmx::power4(rc - rsw);
    sc->c5 = -6 / gmx::power5(rc - rsw);
}

/*! \brief Construct interaction constants
 *
 * This data is used (particularly) by search and force code for
 * short-range interactions. Many of these are constant for the whole
 * simulation; some are constant only after PME tuning completes.
 */
static void init_interaction_const(FILE*                 fp,
                                   interaction_const_t** interaction_const,
                                   const t_inputrec*     ir,
                                   const gmx_mtop_t*     mtop,
                                   bool                  systemHasNetCharge)
{
    interaction_const_t* ic = new interaction_const_t;

    ic->cutoff_scheme = ir->cutoff_scheme;

    ic->coulombEwaldTables = std::make_unique<EwaldCorrectionTables>();
    ic->vdwEwaldTables     = std::make_unique<EwaldCorrectionTables>();

    /* Lennard-Jones */
    ic->vdwtype         = ir->vdwtype;
    ic->vdw_modifier    = ir->vdw_modifier;
    ic->reppow          = mtop->ffparams.reppow;
    ic->rvdw            = cutoff_inf(ir->rvdw);
    ic->rvdw_switch     = ir->rvdw_switch;
    ic->ljpme_comb_rule = ir->ljpme_combination_rule;
    ic->useBuckingham   = (mtop->ffparams.functype[0] == F_BHAM);
    if (ic->useBuckingham)
    {
        ic->buckinghamBMax = calcBuckinghamBMax(fp, mtop);
    }

    initVdwEwaldParameters(fp, ir, ic);

    clear_force_switch_constants(&ic->dispersion_shift);
    clear_force_switch_constants(&ic->repulsion_shift);

    switch (ic->vdw_modifier)
    {
        case eintmodPOTSHIFT:
            /* Only shift the potential, don't touch the force */
            ic->dispersion_shift.cpot = -1.0 / gmx::power6(ic->rvdw);
            ic->repulsion_shift.cpot  = -1.0 / gmx::power12(ic->rvdw);
            break;
        case eintmodFORCESWITCH:
            /* Switch the force, switch and shift the potential */
            force_switch_constants(6.0, ic->rvdw_switch, ic->rvdw, &ic->dispersion_shift);
            force_switch_constants(12.0, ic->rvdw_switch, ic->rvdw, &ic->repulsion_shift);
            break;
        case eintmodPOTSWITCH:
            /* Switch the potential and force */
            potential_switch_constants(ic->rvdw_switch, ic->rvdw, &ic->vdw_switch);
            break;
        case eintmodNONE:
        case eintmodEXACTCUTOFF:
            /* Nothing to do here */
            break;
        default: gmx_incons("unimplemented potential modifier");
    }

    /* Electrostatics */
    ic->eeltype          = ir->coulombtype;
    ic->coulomb_modifier = ir->coulomb_modifier;
    ic->rcoulomb         = cutoff_inf(ir->rcoulomb);
    ic->rcoulomb_switch  = ir->rcoulomb_switch;
    ic->epsilon_r        = ir->epsilon_r;

    /* Set the Coulomb energy conversion factor */
    if (ic->epsilon_r != 0)
    {
        ic->epsfac = ONE_4PI_EPS0 / ic->epsilon_r;
    }
    else
    {
        /* eps = 0 is infinite dieletric: no Coulomb interactions */
        ic->epsfac = 0;
    }

    /* Reaction-field */
    if (EEL_RF(ic->eeltype))
    {
        GMX_RELEASE_ASSERT(ic->eeltype != eelGRF_NOTUSED, "GRF is no longer supported");
        ic->epsilon_rf = ir->epsilon_rf;

        calc_rffac(fp, ic->epsilon_r, ic->epsilon_rf, ic->rcoulomb, &ic->k_rf, &ic->c_rf);
    }
    else
    {
        /* For plain cut-off we might use the reaction-field kernels */
        ic->epsilon_rf = ic->epsilon_r;
        ic->k_rf       = 0;
        if (ir->coulomb_modifier == eintmodPOTSHIFT)
        {
            ic->c_rf = 1 / ic->rcoulomb;
        }
        else
        {
            ic->c_rf = 0;
        }
    }

    initCoulombEwaldParameters(fp, ir, systemHasNetCharge, ic);

    if (fp != nullptr)
    {
        real dispersion_shift;

        dispersion_shift = ic->dispersion_shift.cpot;
        if (EVDW_PME(ic->vdwtype))
        {
            dispersion_shift -= ic->sh_lj_ewald;
        }
        fprintf(fp, "Potential shift: LJ r^-12: %.3e r^-6: %.3e", ic->repulsion_shift.cpot, dispersion_shift);

        if (ic->eeltype == eelCUT)
        {
            fprintf(fp, ", Coulomb %.e", -ic->c_rf);
        }
        else if (EEL_PME(ic->eeltype))
        {
            fprintf(fp, ", Ewald %.3e", -ic->sh_ewald);
        }
        fprintf(fp, "\n");
    }

    *interaction_const = ic;
}

bool areMoleculesDistributedOverPbc(const t_inputrec& ir, const gmx_mtop_t& mtop, const gmx::MDLogger& mdlog)
{
    bool       areMoleculesDistributedOverPbc = false;
    const bool useEwaldSurfaceCorrection = (EEL_PME_EWALD(ir.coulombtype) && ir.epsilon_surface != 0);

    const bool bSHAKE =
            (ir.eConstrAlg == econtSHAKE
             && (gmx_mtop_ftype_count(mtop, F_CONSTR) > 0 || gmx_mtop_ftype_count(mtop, F_CONSTRNC) > 0));

    /* The group cut-off scheme and SHAKE assume charge groups
     * are whole, but not using molpbc is faster in most cases.
     * With intermolecular interactions we need PBC for calculating
     * distances between atoms in different molecules.
     */
    if (bSHAKE && !mtop.bIntermolecularInteractions)
    {
        areMoleculesDistributedOverPbc = ir.bPeriodicMols;

        if (areMoleculesDistributedOverPbc)
        {
            gmx_fatal(FARGS, "SHAKE is not supported with periodic molecules");
        }
    }
    else
    {
        /* Not making molecules whole is faster in most cases,
         * but with orientation restraints or non-tinfoil boundary
         * conditions we need whole molecules.
         */
        areMoleculesDistributedOverPbc =
                (gmx_mtop_ftype_count(mtop, F_ORIRES) == 0 && !useEwaldSurfaceCorrection);

        if (getenv("GMX_USE_GRAPH") != nullptr)
        {
            areMoleculesDistributedOverPbc = false;

            GMX_LOG(mdlog.warning)
                    .asParagraph()
                    .appendText(
                            "GMX_USE_GRAPH is set, using the graph for bonded "
                            "interactions");

            if (mtop.bIntermolecularInteractions)
            {
                GMX_LOG(mdlog.warning)
                        .asParagraph()
                        .appendText(
                                "WARNING: Molecules linked by intermolecular interactions "
                                "have to reside in the same periodic image, otherwise "
                                "artifacts will occur!");
            }
        }

        GMX_RELEASE_ASSERT(areMoleculesDistributedOverPbc || !mtop.bIntermolecularInteractions,
                           "We need to use PBC within molecules with inter-molecular interactions");

        if (bSHAKE && areMoleculesDistributedOverPbc)
        {
            gmx_fatal(FARGS,
                      "SHAKE is not properly supported with intermolecular interactions. "
                      "For short simulations where linked molecules remain in the same "
                      "periodic image, the environment variable GMX_USE_GRAPH can be used "
                      "to override this check.\n");
        }
    }

    return areMoleculesDistributedOverPbc;
}

void init_forcerec(FILE*                            fp,
                   const gmx::MDLogger&             mdlog,
                   t_forcerec*                      fr,
                   t_fcdata*                        fcd,
                   const t_inputrec*                ir,
                   const gmx_mtop_t*                mtop,
                   const t_commrec*                 cr,
                   matrix                           box,
                   const char*                      tabfn,
                   const char*                      tabpfn,
                   gmx::ArrayRef<const std::string> tabbfnm,
                   const gmx_hw_info_t&             hardwareInfo,
                   const gmx_device_info_t*         deviceInfo,
                   const bool                       useGpuForBonded,
                   const bool                       pmeOnlyRankUsesGpu,
                   real                             print_force,
                   gmx_wallcycle*                   wcycle)
{
    real     rtab;
    char*    env;
    double   dbl;
    gmx_bool bFEP_NonBonded;

    /* By default we turn SIMD kernels on, but it might be turned off further down... */
    fr->use_simd_kernels = TRUE;

    if (check_box(ir->ePBC, box))
    {
        gmx_fatal(FARGS, "%s", check_box(ir->ePBC, box));
    }

    /* Test particle insertion ? */
    if (EI_TPI(ir->eI))
    {
        /* Set to the size of the molecule to be inserted (the last one) */
        gmx::RangePartitioning molecules = gmx_mtop_molecules(*mtop);
        fr->n_tpi                        = molecules.block(molecules.numBlocks() - 1).size();
    }
    else
    {
        fr->n_tpi = 0;
    }

    if (ir->coulombtype == eelRF_NEC_UNSUPPORTED || ir->coulombtype == eelGRF_NOTUSED)
    {
        gmx_fatal(FARGS, "%s electrostatics is no longer supported", eel_names[ir->coulombtype]);
    }

    if (ir->bAdress)
    {
        gmx_fatal(FARGS, "AdResS simulations are no longer supported");
    }
    if (ir->useTwinRange)
    {
        gmx_fatal(FARGS, "Twin-range simulations are no longer supported");
    }
    /* Copy the user determined parameters */
    fr->userint1  = ir->userint1;
    fr->userint2  = ir->userint2;
    fr->userint3  = ir->userint3;
    fr->userint4  = ir->userint4;
    fr->userreal1 = ir->userreal1;
    fr->userreal2 = ir->userreal2;
    fr->userreal3 = ir->userreal3;
    fr->userreal4 = ir->userreal4;

    /* Shell stuff */
    fr->fc_stepsize = ir->fc_stepsize;

    /* Free energy */
    fr->efep        = ir->efep;
    fr->sc_alphavdw = ir->fepvals->sc_alpha;
    if (ir->fepvals->bScCoul)
    {
        fr->sc_alphacoul  = ir->fepvals->sc_alpha;
        fr->sc_sigma6_min = gmx::power6(ir->fepvals->sc_sigma_min);
    }
    else
    {
        fr->sc_alphacoul  = 0;
        fr->sc_sigma6_min = 0; /* only needed when bScCoul is on */
    }
    fr->sc_power      = ir->fepvals->sc_power;
    fr->sc_r_power    = ir->fepvals->sc_r_power;
    fr->sc_sigma6_def = gmx::power6(ir->fepvals->sc_sigma);

    env = getenv("GMX_SCSIGMA_MIN");
    if (env != nullptr)
    {
        dbl = 0;
        sscanf(env, "%20lf", &dbl);
        fr->sc_sigma6_min = gmx::power6(dbl);
        if (fp)
        {
            fprintf(fp, "Setting the minimum soft core sigma to %g nm\n", dbl);
        }
    }

    fr->bNonbonded = TRUE;
    if (getenv("GMX_NO_NONBONDED") != nullptr)
    {
        /* turn off non-bonded calculations */
        fr->bNonbonded = FALSE;
        GMX_LOG(mdlog.warning)
                .asParagraph()
                .appendText(
                        "Found environment variable GMX_NO_NONBONDED.\n"
                        "Disabling nonbonded calculations.");
    }

    if ((getenv("GMX_DISABLE_SIMD_KERNELS") != nullptr) || (getenv("GMX_NOOPTIMIZEDKERNELS") != nullptr))
    {
        fr->use_simd_kernels = FALSE;
        if (fp != nullptr)
        {
            fprintf(fp,
                    "\nFound environment variable GMX_DISABLE_SIMD_KERNELS.\n"
                    "Disabling the usage of any SIMD-specific non-bonded & bonded kernel routines\n"
                    "(e.g. SSE2/SSE4.1/AVX).\n\n");
        }
    }

    fr->bBHAM = (mtop->ffparams.functype[0] == F_BHAM);

    /* Neighbour searching stuff */
    fr->cutoff_scheme = ir->cutoff_scheme;
    fr->ePBC          = ir->ePBC;

    /* Determine if we will do PBC for distances in bonded interactions */
    if (fr->ePBC == epbcNONE)
    {
        fr->bMolPBC = FALSE;
    }
    else
    {
        const bool useEwaldSurfaceCorrection =
                (EEL_PME_EWALD(ir->coulombtype) && ir->epsilon_surface != 0);
        if (!DOMAINDECOMP(cr))
        {
            fr->bMolPBC = areMoleculesDistributedOverPbc(*ir, *mtop, mdlog);
            // The assert below is equivalent to fcd->orires.nr != gmx_mtop_ftype_count(mtop, F_ORIRES)
            GMX_RELEASE_ASSERT(!fr->bMolPBC || fcd->orires.nr == 0,
                               "Molecules broken over PBC exist in a simulation including "
                               "orientation restraints. "
                               "This likely means that the global topology and the force constant "
                               "data have gotten out of sync.");
            if (useEwaldSurfaceCorrection)
            {
                gmx_fatal(FARGS,
                          "In GROMACS 2020, Ewald dipole correction is disabled when not "
                          "using domain decomposition. With domain decomposition, it only works "
                          "when each molecule consists of a single update group (e.g. water). "
                          "This will be fixed in GROMACS 2021.");
            }
        }
        else
        {
            fr->bMolPBC = dd_bonded_molpbc(cr->dd, fr->ePBC);

            if (useEwaldSurfaceCorrection && !dd_moleculesAreAlwaysWhole(*cr->dd))
            {
                gmx_fatal(FARGS,
                          "You requested dipole correction (epsilon_surface > 0), but molecules "
                          "are broken "
                          "over periodic boundary conditions by the domain decomposition. "
                          "Run without domain decomposition instead.");
            }
        }

        if (useEwaldSurfaceCorrection)
        {
            GMX_RELEASE_ASSERT((!DOMAINDECOMP(cr) && !fr->bMolPBC)
                                       || (DOMAINDECOMP(cr) && dd_moleculesAreAlwaysWhole(*cr->dd)),
                               "Molecules can not be broken by PBC with epsilon_surface > 0");
        }
    }

    fr->rc_scaling = ir->refcoord_scaling;
    copy_rvec(ir->posres_com, fr->posres_com);
    copy_rvec(ir->posres_comB, fr->posres_comB);
    fr->rlist                  = cutoff_inf(ir->rlist);
    fr->ljpme_combination_rule = ir->ljpme_combination_rule;

    /* This now calculates sum for q and c6*/
    bool systemHasNetCharge = set_chargesum(fp, fr, mtop);

    /* fr->ic is used both by verlet and group kernels (to some extent) now */
    init_interaction_const(fp, &fr->ic, ir, mtop, systemHasNetCharge);
    init_interaction_const_tables(fp, fr->ic);

    const interaction_const_t* ic = fr->ic;

    /* TODO: Replace this Ewald table or move it into interaction_const_t */
    if (ir->coulombtype == eelEWALD)
    {
        init_ewald_tab(&(fr->ewald_table), ir, fp);
    }

    /* Electrostatics: Translate from interaction-setting-in-mdp-file to kernel interaction format */
    switch (ic->eeltype)
    {
        case eelCUT: fr->nbkernel_elec_interaction = GMX_NBKERNEL_ELEC_COULOMB; break;

        case eelRF:
        case eelRF_ZERO: fr->nbkernel_elec_interaction = GMX_NBKERNEL_ELEC_REACTIONFIELD; break;

        case eelSWITCH:
        case eelSHIFT:
        case eelUSER:
        case eelENCADSHIFT:
        case eelPMESWITCH:
        case eelPMEUSER:
        case eelPMEUSERSWITCH:
            fr->nbkernel_elec_interaction = GMX_NBKERNEL_ELEC_CUBICSPLINETABLE;
            break;

        case eelPME:
        case eelP3M_AD:
        case eelEWALD: fr->nbkernel_elec_interaction = GMX_NBKERNEL_ELEC_EWALD; break;

        default:
            gmx_fatal(FARGS, "Unsupported electrostatic interaction: %s", eel_names[ic->eeltype]);
    }
    fr->nbkernel_elec_modifier = ic->coulomb_modifier;

    /* Vdw: Translate from mdp settings to kernel format */
    switch (ic->vdwtype)
    {
        case evdwCUT:
            if (fr->bBHAM)
            {
                fr->nbkernel_vdw_interaction = GMX_NBKERNEL_VDW_BUCKINGHAM;
            }
            else
            {
                fr->nbkernel_vdw_interaction = GMX_NBKERNEL_VDW_LENNARDJONES;
            }
            break;
        case evdwPME: fr->nbkernel_vdw_interaction = GMX_NBKERNEL_VDW_LJEWALD; break;

        case evdwSWITCH:
        case evdwSHIFT:
        case evdwUSER:
        case evdwENCADSHIFT:
            fr->nbkernel_vdw_interaction = GMX_NBKERNEL_VDW_CUBICSPLINETABLE;
            break;

        default: gmx_fatal(FARGS, "Unsupported vdw interaction: %s", evdw_names[ic->vdwtype]);
    }
    fr->nbkernel_vdw_modifier = ic->vdw_modifier;

    if (ir->cutoff_scheme == ecutsVERLET)
    {
        if (!gmx_within_tol(ic->reppow, 12.0, 10 * GMX_DOUBLE_EPS))
        {
            gmx_fatal(FARGS, "Cut-off scheme %s only supports LJ repulsion power 12",
                      ecutscheme_names[ir->cutoff_scheme]);
        }
        /* Older tpr files can contain Coulomb user tables with the Verlet cutoff-scheme,
         * while mdrun does not (and never did) support this.
         */
        if (EEL_USER(fr->ic->eeltype))
        {
            gmx_fatal(FARGS, "Combination of %s and cutoff scheme %s is not supported",
                      eel_names[ir->coulombtype], ecutscheme_names[ir->cutoff_scheme]);
        }

        fr->bvdwtab  = FALSE;
        fr->bcoultab = FALSE;
    }

    /* 1-4 interaction electrostatics */
    fr->fudgeQQ = mtop->ffparams.fudgeQQ;

    fr->haveDirectVirialContributions =
            (EEL_FULL(ic->eeltype) || EVDW_PME(ic->vdwtype) || fr->forceProviders->hasForceProvider()
             || gmx_mtop_ftype_count(mtop, F_POSRES) > 0 || gmx_mtop_ftype_count(mtop, F_FBPOSRES) > 0
             || ir->nwall > 0 || ir->bPull || ir->bRot || ir->bIMD);

    if (fr->shift_vec == nullptr)
    {
        snew(fr->shift_vec, SHIFTS);
    }

    fr->shiftForces.resize(SHIFTS);

    if (fr->nbfp == nullptr)
    {
        fr->ntype = mtop->ffparams.atnr;
        fr->nbfp  = mk_nbfp(&mtop->ffparams, fr->bBHAM);
        if (EVDW_PME(ic->vdwtype))
        {
            fr->ljpme_c6grid = make_ljpme_c6grid(&mtop->ffparams, fr);
        }
    }

    /* Copy the energy group exclusions */
    fr->egp_flags = ir->opts.egp_flags;

    /* Van der Waals stuff */
    if ((ic->vdwtype != evdwCUT) && (ic->vdwtype != evdwUSER) && !fr->bBHAM)
    {
        if (ic->rvdw_switch >= ic->rvdw)
        {
            gmx_fatal(FARGS, "rvdw_switch (%f) must be < rvdw (%f)", ic->rvdw_switch, ic->rvdw);
        }
        if (fp)
        {
            fprintf(fp, "Using %s Lennard-Jones, switch between %g and %g nm\n",
                    (ic->eeltype == eelSWITCH) ? "switched" : "shifted", ic->rvdw_switch, ic->rvdw);
        }
    }

    if (fr->bBHAM && EVDW_PME(ic->vdwtype))
    {
        gmx_fatal(FARGS, "LJ PME not supported with Buckingham");
    }

    if (fr->bBHAM && (ic->vdwtype == evdwSHIFT || ic->vdwtype == evdwSWITCH))
    {
        gmx_fatal(FARGS, "Switch/shift interaction not supported with Buckingham");
    }

    if (fr->bBHAM && fr->cutoff_scheme == ecutsVERLET)
    {
        gmx_fatal(FARGS, "Verlet cutoff-scheme is not supported with Buckingham");
    }

    if (fp && fr->cutoff_scheme == ecutsGROUP)
    {
        fprintf(fp, "Cut-off's:   NS: %g   Coulomb: %g   %s: %g\n", fr->rlist, ic->rcoulomb,
                fr->bBHAM ? "BHAM" : "LJ", ic->rvdw);
    }

    if (ir->implicit_solvent)
    {
        gmx_fatal(FARGS, "Implict solvation is no longer supported.");
    }


    /* This code automatically gives table length tabext without cut-off's,
     * in that case grompp should already have checked that we do not need
     * normal tables and we only generate tables for 1-4 interactions.
     */
    rtab = ir->rlist + ir->tabext;

    /* We want to use unmodified tables for 1-4 coulombic
     * interactions, so we must in general have an extra set of
     * tables. */
    if (gmx_mtop_ftype_count(mtop, F_LJ14) > 0 || gmx_mtop_ftype_count(mtop, F_LJC14_Q) > 0
        || gmx_mtop_ftype_count(mtop, F_LJC_PAIRS_NB) > 0)
    {
        fr->pairsTable = make_tables(fp, ic, tabpfn, rtab, GMX_MAKETABLES_14ONLY);
    }

    /* Wall stuff */
    fr->nwall = ir->nwall;
    if (ir->nwall && ir->wall_type == ewtTABLE)
    {
        make_wall_tables(fp, ir, tabfn, &mtop->groups, fr);
    }

    if (fcd && !tabbfnm.empty())
    {
        // Need to catch std::bad_alloc
        // TODO Don't need to catch this here, when merging with master branch
        try
        {
            fcd->bondtab  = make_bonded_tables(fp, F_TABBONDS, F_TABBONDSNC, mtop, tabbfnm, "b");
            fcd->angletab = make_bonded_tables(fp, F_TABANGLES, -1, mtop, tabbfnm, "a");
            fcd->dihtab   = make_bonded_tables(fp, F_TABDIHS, -1, mtop, tabbfnm, "d");
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    else
    {
        if (debug)
        {
            fprintf(debug,
                    "No fcdata or table file name passed, can not read table, can not do bonded "
                    "interactions\n");
        }
    }

    // QM/MM initialization if requested
    fr->bQMMM = ir->bQMMM;
    if (fr->bQMMM)
    {
        // Initialize QM/MM if supported
        if (GMX_QMMM)
        {
            GMX_LOG(mdlog.info)
                    .asParagraph()
                    .appendText(
                            "Large parts of the QM/MM support is deprecated, and may be removed in "
                            "a future "
                            "version. Please get in touch with the developers if you find the "
                            "support useful, "
                            "as help is needed if the functionality is to continue to be "
                            "available.");
            fr->qr = mk_QMMMrec();
            init_QMMMrec(cr, mtop, ir, fr);
        }
        else
        {
            gmx_incons(
                    "QM/MM was requested, but is only available when GROMACS "
                    "is configured with QM/MM support");
        }
    }

    /* Set all the static charge group info */
    fr->cginfo_mb = init_cginfo_mb(mtop, fr, &bFEP_NonBonded);
    if (!DOMAINDECOMP(cr))
    {
        fr->cginfo = cginfo_expand(mtop->molblock.size(), fr->cginfo_mb);
    }

    if (!DOMAINDECOMP(cr))
    {
        forcerec_set_ranges(fr, mtop->natoms, mtop->natoms, mtop->natoms);
    }

    fr->print_force = print_force;

    /* Initialize the thread working data for bonded interactions */
    fr->bondedThreading = init_bonded_threading(
            fp, mtop->groups.groups[SimulationAtomGroupType::EnergyOutput].size());

    fr->nthread_ewc = gmx_omp_nthreads_get(emntBonded);
    snew(fr->ewc_t, fr->nthread_ewc);

    if (fr->cutoff_scheme == ecutsVERLET)
    {
        // We checked the cut-offs in grompp, but double-check here.
        // We have PME+LJcutoff kernels for rcoulomb>rvdw.
        if (EEL_PME_EWALD(ir->coulombtype) && ir->vdwtype == eelCUT)
        {
            GMX_RELEASE_ASSERT(ir->rcoulomb >= ir->rvdw,
                               "With Verlet lists and PME we should have rcoulomb>=rvdw");
        }
        else
        {
            GMX_RELEASE_ASSERT(
                    ir->rcoulomb == ir->rvdw,
                    "With Verlet lists and no PME rcoulomb and rvdw should be identical");
        }

        fr->nbv = Nbnxm::init_nb_verlet(mdlog, bFEP_NonBonded, ir, fr, cr, hardwareInfo, deviceInfo,
                                        mtop, box, wcycle);

        if (useGpuForBonded)
        {
            auto stream = havePPDomainDecomposition(cr)
                                  ? Nbnxm::gpu_get_command_stream(
                                            fr->nbv->gpu_nbv, gmx::InteractionLocality::NonLocal)
                                  : Nbnxm::gpu_get_command_stream(fr->nbv->gpu_nbv,
                                                                  gmx::InteractionLocality::Local);
            // TODO the heap allocation is only needed while
            // t_forcerec lacks a constructor.
            fr->gpuBonded = new gmx::GpuBonded(mtop->ffparams, stream, wcycle);
        }
    }

    if (ir->eDispCorr != edispcNO)
    {
        fr->dispersionCorrection = std::make_unique<DispersionCorrection>(
                *mtop, *ir, fr->bBHAM, fr->ntype,
                gmx::arrayRefFromArray(fr->nbfp, fr->ntype * fr->ntype * 2), *fr->ic, tabfn);
        fr->dispersionCorrection->print(mdlog);
    }

    if (fp != nullptr)
    {
        /* Here we switch from using mdlog, which prints the newline before
         * the paragraph, to our old fprintf logging, which prints the newline
         * after the paragraph, so we should add a newline here.
         */
        fprintf(fp, "\n");
    }

    if (pmeOnlyRankUsesGpu && c_enableGpuPmePpComms)
    {
        fr->pmePpCommGpu = std::make_unique<gmx::PmePpCommGpu>(cr->mpi_comm_mysim, cr->dd->pme_nodeid);
    }
    init_deepmd();
}

t_forcerec::t_forcerec() = default;

t_forcerec::~t_forcerec() = default;

/* Frees GPU memory and sets a tMPI node barrier.
 *
 * Note that this function needs to be called even if GPUs are not used
 * in this run because the PME ranks have no knowledge of whether GPUs
 * are used or not, but all ranks need to enter the barrier below.
 * \todo Remove physical node barrier from this function after making sure
 * that it's not needed anymore (with a shared GPU run).
 */
void free_gpu_resources(t_forcerec*                          fr,
                        const gmx::PhysicalNodeCommunicator& physicalNodeCommunicator,
                        const gmx_gpu_info_t&                gpu_info)
{
    bool isPPrankUsingGPU = (fr != nullptr) && (fr->nbv != nullptr) && fr->nbv->useGpu();

    /* stop the GPU profiler (only CUDA) */
    if (gpu_info.n_dev > 0)
    {
        stopGpuProfiler();
    }

    if (isPPrankUsingGPU)
    {
        /* Free data in GPU memory and pinned memory before destroying the GPU context */
        fr->nbv.reset();

        delete fr->gpuBonded;
        fr->gpuBonded = nullptr;
    }

    /* With tMPI we need to wait for all ranks to finish deallocation before
     * destroying the CUDA context in free_gpu() as some tMPI ranks may be sharing
     * GPU and context.
     *
     * This is not a concern in OpenCL where we use one context per rank which
     * is freed in nbnxn_gpu_free().
     *
     * Note: it is safe to not call the barrier on the ranks which do not use GPU,
     * but it is easier and more futureproof to call it on the whole node.
     */
    if (GMX_THREAD_MPI)
    {
        physicalNodeCommunicator.barrier();
    }
}

void done_forcerec(t_forcerec* fr, int numMolBlocks)
{
    if (fr == nullptr)
    {
        // PME-only ranks don't have a forcerec
        return;
    }
    done_cginfo_mb(fr->cginfo_mb, numMolBlocks);
    sfree(fr->nbfp);
    delete fr->ic;
    sfree(fr->shift_vec);
    sfree(fr->ewc_t);
    tear_down_bonded_threading(fr->bondedThreading);
    GMX_RELEASE_ASSERT(fr->gpuBonded == nullptr, "Should have been deleted earlier, when used");
    fr->bondedThreading = nullptr;
    delete fr;
}
