/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2014- The GROMACS Authors
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
 * Defines constants used to know which nbNxM kernel flavours (4xM or 2xMM)
 * can be supported by the SIMD layer in use.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */
#ifndef GMX_NBNXM_NBNXM_SIMD_H
#define GMX_NBNXM_NBNXM_SIMD_H

#include "config.h"

#include "gromacs/math/vectypes.h"
#include "gromacs/simd/simd.h"
#include "gromacs/utility/real.h"

#include "pairlist.h"
#include "pairlistparams.h"

//! The types of nbNxM SIMD kernel layout
enum class KernelLayout
{
    r4xM,  //!< 4 'i'-registers each containing data for interaction with M j-atoms
    r2xMM, //!< 2 'i'-registers each containing duplicated data, { M, M }, for interaction with M j-atoms
    r8xM   //!< 8 'i'-registers each containing data for interaction with M j-atoms
};

//! The nbnxn i-cluster size in atoms for the given NBNxM kernel layout
static inline constexpr int c_iClusterSize(const KernelLayout kernelLayout)
{
    switch (kernelLayout)
    {
        case KernelLayout::r4xM: return 4;
        case KernelLayout::r2xMM: return 4;
        case KernelLayout::r8xM: return 8;
    }

    GMX_RELEASE_ASSERT(false, "Unhandled case");

    return 0;
}

#if GMX_SIMD && GMX_USE_SIMD_KERNELS

//! The nbnxn i-cluster size in atoms for the given NBNxM kernel layout
static inline constexpr int c_jClusterSize(const KernelLayout kernelLayout)
{
    switch (kernelLayout)
    {
        case KernelLayout::r4xM: return GMX_SIMD_REAL_WIDTH;
        case KernelLayout::r2xMM: return GMX_SIMD_REAL_WIDTH / 2;
        case KernelLayout::r8xM: return GMX_SIMD_REAL_WIDTH;
    }

    GMX_RELEASE_ASSERT(false, "Unhandled case");

    return 0;
}

/*! \brief The nbnxn SIMD 4xN and 2x(N+N) kernels can be added independently.
 * Currently the 2xNN SIMD kernels only make sense with:
 *  8-way SIMD: 4x4 setup, performance wise only useful on CPUs without FMA or on AMD Zen1
 * 16-way SIMD: 4x8 setup, used in single precision with 512 bit wide SIMD
 */
#    if GMX_SIMD_REAL_WIDTH == 2 || GMX_SIMD_REAL_WIDTH == 4 || GMX_SIMD_REAL_WIDTH == 8
#        define GMX_NBNXN_SIMD_4XN
#    endif
#    if GMX_SIMD_REAL_WIDTH == 8 || GMX_SIMD_REAL_WIDTH == 16
#        define GMX_NBNXN_SIMD_2XNN
#    endif
#    if GMX_SIMD_REAL_WIDTH == 4
#        define GMX_NBNXN_SIMD_8XN
#    endif

#    if !(defined GMX_NBNXN_SIMD_4XN || defined GMX_NBNXN_SIMD_2XNN || defined GMX_NBNXN_SIMD_8XN)
#        error "No SIMD kernel type defined"
#    endif

// We use the FDV0 tables for width==4 (when we can load it in one go), or if we don't have any unaligned loads
#    if GMX_SIMD_REAL_WIDTH == 4 || !GMX_SIMD_HAVE_GATHER_LOADU_BYSIMDINT_TRANSPOSE_REAL
static constexpr bool c_useTableFormatFDV0 = true;
#    else
static constexpr bool c_useTableFormatFDV0 = false;
#    endif

/*! \brief Helper function to generate an std::array of SimdReal using a lambda
 *
 * An example of intended usage is:
 * auto rV = genArr<4>([&](int i) { return rSquaredV[i] * rInvV[i]; });
 *
 * \tparam N  The size of the array
 * \tparam F  The function type, deduced
 * \param[in] f  The function
 */
template<int N, class F>
std::array<gmx::SimdReal, N> genArr(F f)
{
    static_assert(N <= 2 || N == 4 || N == 8);

    if constexpr (N == 0)
    {
        return std::array<gmx::SimdReal, 0>{};
    }
    else if constexpr (N == 1)
    {
        return std::array<gmx::SimdReal, 1>{ f(0) };
    }
    else if constexpr (N == 2)
    {
        return std::array<gmx::SimdReal, 2>{ f(0), f(1) };
    }
    else if constexpr (N == 4)
    {
        return std::array<gmx::SimdReal, 4>{ f(0), f(1), f(2), f(3) };
    }
    else
    {
        return std::array<gmx::SimdReal, 8>{ f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7) };
    }
}

/*! \brief Helper function to generate an std::array of SimdBool using a lambda
 *
 * An example of intended usage is:
 * auto withinCutoffV = genBoolArr<nR>([&](int i) { return rSquaredV[i] < rc2_S; });
 *
 * \tparam N  The size of the array
 * \tparam F  The function type, deduced
 * \param[in] f  The function
 */
template<int N, class F>
std::array<gmx::SimdBool, N> genBoolArr(F f)
{
    static_assert(N <= 2 || N == 4 || N == 8);

    if constexpr (N == 0)
    {
        return std::array<gmx::SimdBool, 0>{};
    }
    else if constexpr (N == 1)
    {
        return std::array<gmx::SimdBool, 1>{ f(0) };
    }
    else if constexpr (N == 2)
    {
        return std::array<gmx::SimdBool, 2>{ f(0), f(1) };
    }
    else if constexpr (N == 4)
    {
        return std::array<gmx::SimdBool, 4>{ f(0), f(1), f(2), f(3) };
    }
    else
    {
        return std::array<gmx::SimdBool, 8>{ f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7) };
    }
}

#endif // GMX_SIMD && GMX_USE_SIMD_KERNELS

#endif // GMX_NBNXM_NBNXM_SIMD
