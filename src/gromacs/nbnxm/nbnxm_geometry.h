/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2012- The GROMACS Authors
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
 *
 * \brief
 * Declares the geometry-related functionality
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */
#ifndef GMX_NBNXM_NBNXM_GEOMETRY_H
#define GMX_NBNXM_NBNXM_GEOMETRY_H

#include "gromacs/math/vectypes.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/simd/simd.h"
#include "gromacs/utility/fatalerror.h"

#include "pairlist.h"
//#include "pairlistparams.h"

/*! \brief Returns the base-2 log of n.
 * *
 * Generates a fatal error when n is not an integer power of 2.
 */
static inline int get_2log(int n)
{
    if (!gmx::isPowerOfTwo(n))
    {
        gmx_fatal(FARGS, "nbnxn na_c (%d) is not a power of 2", n);
    }

    return gmx::log2I(n);
}

namespace Nbnxm
{

//! The nbnxn i-cluster size in atoms for the given NBNxM kernel type
static inline constexpr int c_iClusterSize(const KernelType kernelType)
{
    switch (kernelType)
    {
        case KernelType::Cpu4x4_PlainC:
        case KernelType::Cpu4xN_Simd_4xN:
        case KernelType::Cpu4xN_Simd_2xNN: return 4;
        case KernelType::Cpu2xN_Simd_2xN: return 2;
        case KernelType::Cpu8xN_Simd_8xN: return 8;
        case KernelType::Gpu8x8x8:
        case KernelType::Cpu8x8x8_PlainC: return c_nbnxnGpuClusterSize;
        case KernelType::NotSet:
        case KernelType::Count: return 0;
    }

    GMX_RELEASE_ASSERT(false, "Unhandled case");

    return 0;
}

//! The nbnxn j-cluster size in atoms for the given NBNxM kernel type
static inline constexpr int c_jClusterSize(const KernelType kernelType)
{
    switch (kernelType)
    {
        case KernelType::Cpu4x4_PlainC: return 4;
#if GMX_SIMD
        case KernelType::Cpu4xN_Simd_4xN: return GMX_SIMD_REAL_WIDTH;
        case KernelType::Cpu4xN_Simd_2xNN: return GMX_SIMD_REAL_WIDTH / 2;
        case KernelType::Cpu2xN_Simd_2xN: return GMX_SIMD_REAL_WIDTH;
        case KernelType::Cpu8xN_Simd_8xN: return GMX_SIMD_REAL_WIDTH;
#else
        case KernelType::Cpu4xN_Simd_4xN:
        case KernelType::Cpu4xN_Simd_2xNN:
        case KernelType::Cpu8xN_Simd_8xN: return 0;
#endif
        case KernelType::Gpu8x8x8: return c_nbnxnGpuClusterSize;
        case KernelType::Cpu8x8x8_PlainC: return c_nbnxnGpuClusterSize / 2;
        case KernelType::NotSet:
        case KernelType::Count: return 0;
    }

    GMX_RELEASE_ASSERT(false, "Unhandled case");

    return 0;
}

/*! \brief Returns whether the pair-list corresponding to nb_kernel_type is simple */
static constexpr bool kernelTypeUsesSimplePairlist(const KernelType kernelType)
{
    return (kernelType == KernelType::Cpu4x4_PlainC || kernelType == KernelType::Cpu4xN_Simd_4xN
            || kernelType == KernelType::Cpu4xN_Simd_2xNN || kernelType == KernelType::Cpu8xN_Simd_8xN);
}

//! Returns whether a SIMD kernel is in use
static constexpr bool kernelTypeIsSimd(const KernelType kernelType)
{
    return (kernelType == KernelType::Cpu4xN_Simd_4xN || kernelType == KernelType::Cpu4xN_Simd_2xNN
            || kernelType == KernelType::Cpu8xN_Simd_8xN);
}

} // namespace Nbnxm

/*! \brief Returns the increase in pairlist radius when including volume of pairs beyond rlist
 *
 * Due to the cluster size the total volume of the pairlist is (much) more
 * than 4/3*pi*rlist^3. This function returns the increase in radius
 * required to match the volume of the pairlist including the atoms pairs
 * that are beyond rlist.
 */
real nbnxmPairlistVolumeRadiusIncrease(bool useGpu, real atomDensity);

/*! \brief Returns the effective list radius of the pair-list
 *
 * Due to the cluster size the effective pair-list is longer than
 * that of a simple atom pair-list. This function gives the extra distance.
 */
real nbnxn_get_rlist_effective_inc(int clusterSize, const gmx::RVec& averageClusterBoundingBox);

#endif
