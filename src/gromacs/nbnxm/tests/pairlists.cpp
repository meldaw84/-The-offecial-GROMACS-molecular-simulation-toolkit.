/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2021- The GROMACS Authors
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
 * \brief
 * Unit tests for pair lists
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include "gromacs/mdtypes/interaction_const.h"
//#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/nbnxm/kernel_common.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/pairlistparams.h"
#include "gromacs/nbnxm/pairlistset.h"
#include "gromacs/nbnxm/pairlistsets.h"
#include "gromacs/nbnxm/pairsearch.h"
#include "gromacs/utility/logger.h"

#include "testutils/testasserts.h"

namespace gmx
{

namespace test
{

namespace
{

TEST(PairlistTest, CanConstruct)
{

    const size_t               numParticleTypes = 1;
    const std::vector<real>   nonbondedParameters = {2.0, 3.0};
    ASSERT_EQ(nonbondedParameters.size(), numParticleTypes * 2);
    const interaction_const_t interactionConst;
    //const gmx::DeviceStreamManager& deviceStreamManager;

    const auto pinPolicy       = gmx::PinningPolicy::PinnedIfSupported;
    const int  combinationRule = 0; // TODO what to choose?

    const bool useTabulatedEwaldCorrection = false;
    Nbnxm::KernelSetup kernelSetup;
    kernelSetup.kernelType         = Nbnxm::KernelType::Gpu8x8x8;
    kernelSetup.ewaldExclusionType = useTabulatedEwaldCorrection ? Nbnxm::EwaldExclusionType::Table
                                                           : Nbnxm::EwaldExclusionType::Analytical;

    const real pairlistCutoff = 1.0;
    PairlistParams pairlistParams(kernelSetup.kernelType, false, pairlistCutoff, false);


    // nbnxn_atomdata is always initialized with 1 thread if the GPU is used
    constexpr int numThreadsInit = 1;
    // multiple energy groups are not supported on the GPU
    constexpr int numEnergyGroups = 1;
    auto          atomData        = std::make_unique<nbnxn_atomdata_t>(pinPolicy,
                                                       gmx::MDLogger(),
                                                       kernelSetup.kernelType,
                                                       combinationRule,
                                                       numParticleTypes,
                                                       nonbondedParameters,
                                                       numEnergyGroups,
                                                       numThreadsInit);
    
    //NbnxmGpu* nbnxmGpu = Nbnxm::gpu_init(
    //        deviceStreamManager, &interactionConst, pairlistParams, atomData.get(), false);

    // minimum iList count for GPU balancing
    int iListCount = 0; // Nbnxm::gpu_min_ci_balanced(nbnxmGpu);

    auto pairlistSets = std::make_unique<PairlistSets>(pairlistParams, false, iListCount);
    const int numOpenMPThreads = 1;
    auto pairSearch   = std::make_unique<PairSearch>(
            PbcType::Xyz, false, nullptr, nullptr, pairlistParams.pairlistType, false, numOpenMPThreads, pinPolicy);

    // Put everything together
    auto nbv = std::make_unique<nonbonded_verlet_t>(
                                                    std::move(pairlistSets), std::move(pairSearch), std::move(atomData), kernelSetup, nullptr/*nbnxmGpu*/, nullptr);

    // Some paramters must be copied to NbnxmGpu to have a fully constructed nonbonded_verlet_t
    //Nbnxm::gpu_init_atomdata(nbv->gpu_nbv, nbv->nbat.get());
}

} // namespace
} // namespace test
} // namespace gmx
