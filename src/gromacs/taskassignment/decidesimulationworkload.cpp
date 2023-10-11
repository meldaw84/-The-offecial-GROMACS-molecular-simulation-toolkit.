/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2019- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
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
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief Declares utility functions to manage step, domain-lifetime, and run workload data structures.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \author Szilárd Páll <pall.szilard@gmail.com>
 * \ingroup module_taskassignment
 */
#include "gmxpre.h"

#include "gromacs/taskassignment/decidesimulationworkload.h"

#include "gromacs/ewald/pme.h"
#include "gromacs/essentialdynamics/edsam.h"
#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/multipletimestepping.h"
#include "gromacs/pulling/pull.h"
#include "gromacs/taskassignment/decidegpuusage.h"
#include "gromacs/taskassignment/taskassignment.h"
#include "gromacs/utility/arrayref.h"

namespace gmx
{

SimulationWorkload createSimulationWorkload(const t_inputrec& inputrec,
                                            const bool        disableNonbondedCalculation,
                                            const DevelopmentFeatureFlags& devFlags,
                                            bool       havePpDomainDecomposition,
                                            bool       haveSeparatePmeRank,
                                            bool       useGpuForNonbonded,
                                            PmeRunMode pmeRunMode,
                                            bool       useGpuForBonded,
                                            bool       useGpuForUpdate,
                                            bool       useGpuDirectHalo,
                                            bool       canUseDirectGpuComm,
                                            bool       useGpuPmeDecomposition)
{
    SimulationWorkload simulationWorkload;
    simulationWorkload.computeNonbonded = !disableNonbondedCalculation;
    simulationWorkload.computeNonbondedAtMtsLevel1 =
            simulationWorkload.computeNonbonded && inputrec.useMts
            && inputrec.mtsLevels.back().forceGroups[static_cast<int>(MtsForceGroups::Nonbonded)];
    simulationWorkload.computeMuTot    = inputrecNeedMutot(&inputrec);
    simulationWorkload.useCpuNonbonded = !useGpuForNonbonded;
    simulationWorkload.useGpuNonbonded = useGpuForNonbonded;
    simulationWorkload.useCpuPme       = (pmeRunMode == PmeRunMode::CPU);
    simulationWorkload.useGpuPme = (pmeRunMode == PmeRunMode::GPU || pmeRunMode == PmeRunMode::Mixed);
    simulationWorkload.useGpuPmeFft              = (pmeRunMode == PmeRunMode::GPU);
    simulationWorkload.useGpuBonded              = useGpuForBonded;
    simulationWorkload.useGpuUpdate              = useGpuForUpdate;
    simulationWorkload.havePpDomainDecomposition = havePpDomainDecomposition;
    simulationWorkload.useCpuHaloExchange        = havePpDomainDecomposition && !useGpuDirectHalo;
    simulationWorkload.useGpuHaloExchange        = useGpuDirectHalo;
    if (pmeRunMode == PmeRunMode::None)
    {
        GMX_RELEASE_ASSERT(!haveSeparatePmeRank, "Can not have separate PME rank(s) without PME.");
    }
    simulationWorkload.haveSeparatePmeRank = haveSeparatePmeRank;
    simulationWorkload.useGpuPmePpCommunication =
            haveSeparatePmeRank && canUseDirectGpuComm
            && (pmeRunMode == PmeRunMode::GPU || pmeRunMode == PmeRunMode::Mixed);
    simulationWorkload.useCpuPmePpCommunication =
            haveSeparatePmeRank && !simulationWorkload.useGpuPmePpCommunication;
    GMX_RELEASE_ASSERT(!(simulationWorkload.useGpuPmePpCommunication
                         && simulationWorkload.useCpuPmePpCommunication),
                       "Cannot do PME-PP communication on both CPU and GPU");
    simulationWorkload.useGpuDirectCommunication =
            simulationWorkload.useGpuHaloExchange || simulationWorkload.useGpuPmePpCommunication;
    simulationWorkload.useGpuPmeDecomposition       = useGpuPmeDecomposition;
    simulationWorkload.haveEwaldSurfaceContribution = haveEwaldSurfaceContribution(inputrec);
    simulationWorkload.useMts                       = inputrec.useMts;
    const bool featuresRequireGpuBufferOps = useGpuForUpdate || simulationWorkload.useGpuDirectCommunication;
    simulationWorkload.useGpuXBufferOpsWhenAllowed =
            (devFlags.enableGpuBufferOps || featuresRequireGpuBufferOps) && !inputrec.useMts;
    simulationWorkload.useGpuFBufferOpsWhenAllowed =
            (devFlags.enableGpuBufferOps || featuresRequireGpuBufferOps) && !inputrec.useMts;
    if (simulationWorkload.useGpuXBufferOpsWhenAllowed || simulationWorkload.useGpuFBufferOpsWhenAllowed)
    {
        GMX_ASSERT(simulationWorkload.useGpuNonbonded,
                   "Can only offload X/F buffer ops if nonbonded computation is also offloaded");
    }
    simulationWorkload.useMdGpuGraph =
            devFlags.enableCudaGraphs && useGpuForUpdate
            && (simulationWorkload.haveSeparatePmeRank ? simulationWorkload.useGpuPmePpCommunication : true)
            && (havePpDomainDecomposition ? simulationWorkload.useGpuHaloExchange : true)
            && (havePpDomainDecomposition ? (GMX_THREAD_MPI > 0) : true);
    return simulationWorkload;
}


/*! \brief Return true if there are special forces computed.
 *
 * The conditionals exactly correspond to those in sim_util.cpp:computeSpecialForces().
 */
static bool haveSpecialForces(const t_inputrec&          inputrec,
                              const gmx::ForceProviders& forceProviders,
                              const pull_t*              pull_work,
                              const gmx_edsam*           ed)
{

    return ((forceProviders.hasForceProvider()) ||                 // forceProviders
            (inputrec.bPull && pull_have_potential(*pull_work)) || // pull
            inputrec.bRot ||                                       // enforced rotation
            (ed != nullptr) ||                                     // flooding
            (inputrec.bIMD));                                      // IMD
}

/*! \brief Set up flags that have the lifetime of the domain indicating what type of work is there to compute.
 */
DomainLifetimeWorkload setupDomainLifetimeWorkload(const t_inputrec&         inputrec,
                                                   const t_forcerec&         fr,
                                                   const pull_t*             pull_work,
                                                   const gmx_edsam*          ed,
                                                   const t_mdatoms&          mdatoms,
                                                   const SimulationWorkload& simulationWork)
{
    DomainLifetimeWorkload domainWork;
    // Note that haveSpecialForces is constant over the whole run
    domainWork.haveSpecialForces = haveSpecialForces(inputrec, *fr.forceProviders, pull_work, ed);
    domainWork.haveCpuListedForceWork = false;
    domainWork.haveCpuBondedWork      = false;
    for (const auto& listedForces : fr.listedForces)
    {
        if (listedForces.haveCpuListedForces(*fr.fcdata))
        {
            domainWork.haveCpuListedForceWork = true;
        }
        if (listedForces.haveCpuBondeds())
        {
            domainWork.haveCpuBondedWork = true;
        }
    }
    domainWork.haveGpuBondedWork =
            ((fr.listedForcesGpu != nullptr) && fr.listedForcesGpu->haveInteractions());
    // Note that haveFreeEnergyWork is constant over the whole run
    domainWork.haveFreeEnergyWork =
            (fr.efep != FreeEnergyPerturbationType::No && mdatoms.nPerturbed != 0);
    // We assume we have local force work if there are CPU
    // force tasks including PME or nonbondeds.
    domainWork.haveCpuLocalForceWork =
            domainWork.haveSpecialForces || domainWork.haveCpuListedForceWork
            || domainWork.haveFreeEnergyWork || simulationWork.useCpuNonbonded || simulationWork.useCpuPme
            || simulationWork.haveEwaldSurfaceContribution || inputrec.nwall > 0;
    domainWork.haveCpuNonLocalForceWork = domainWork.haveCpuBondedWork || domainWork.haveFreeEnergyWork;
    domainWork.haveLocalForceContribInCpuBuffer =
            domainWork.haveCpuLocalForceWork || simulationWork.havePpDomainDecomposition;

    return domainWork;
}


} // namespace gmx
