/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2020- The GROMACS Authors
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
 *  \brief
 *  NBNXM SYCL kernels
 *
 *  \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include "nbnxm_sycl_kernel.h"

#include "gromacs/mdtypes/simulation_workload.h"

#include "nbnxm_sycl_types.h"

namespace Nbnxm
{

static int getNbnxmSubGroupSize(const DeviceInformation& deviceInfo)
{
    if (deviceInfo.supportedSubGroupSizesSize == 1)
    {
        return deviceInfo.supportedSubGroupSizesData[0];
    }
    else if (deviceInfo.supportedSubGroupSizesSize > 1)
    {
        switch (deviceInfo.deviceVendor)
        {
            /* For Intel, choose 8 for 4x4 clusters, and 32 for 8x8 clusters.
             * The optimal one depends on the hardware, but we cannot choose c_nbnxnGpuClusterSize
             * at runtime anyway yet. */
            case DeviceVendor::Intel:
                return c_nbnxnGpuClusterSize * c_nbnxnGpuClusterSize / c_nbnxnGpuClusterpairSplit;
            default:
                GMX_RELEASE_ASSERT(false, "Flexible sub-groups only supported for Intel GPUs");
                return 0;
        }
    }
    else
    {
        GMX_RELEASE_ASSERT(false, "Device has no known supported sub-group sizes");
        return 0;
    }
}

template<int subGroupSize>
void launchNbnxmKernel(NbnxmGpu* nb, const gmx::StepWorkload& stepWork, const InteractionLocality iloc)
{
    const bool doPruneNBL     = (nb->plist[iloc]->haveFreshList && !nb->didPrune[iloc]);
    const bool doCalcEnergies = stepWork.computeEnergy;
    launchNbnxmKernelHelper<subGroupSize, doPruneNBL, doCalcEnergies>(nb, stepWork, iloc);
}

void launchNbnxmKernel(NbnxmGpu* nb, const gmx::StepWorkload& stepWork, const InteractionLocality iloc)
{
    const int     subGroupSize = getNbnxmSubGroupSize(nb->deviceContext_->deviceInfo());
    constexpr int launchSize   = NbnxmSupportsSubgroupSize<GpuNBClusterSize>::Size::value;
    GMX_RELEASE_ASSERT((launchSize != subGroupSize), "Unsupported sub-group size");
    if constexpr (launchSize > 0)
    {
        launchNbnxmKernel<launchSize>(nb, stepWork, iloc);
    }
}

} // namespace Nbnxm
