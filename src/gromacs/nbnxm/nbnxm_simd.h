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

#include "pairlistparams.h"

//! The types of nbNxM SIMD kernel layout
enum class KernelLayout
{
    r4xM, //!< 4 'i'-registers each containing data for interaction with M j-atoms
    r2xMM //!< 2 'i'-registers each containing duplicated data, { M, M }, for interaction with M j-atoms
};

#if GMX_SIMD && GMX_USE_SIMD_KERNELS
/*! \brief The nbnxn SIMD 4xN and 2x(N+N) kernels can be added independently.
 * Currently the 2xNN SIMD kernels only make sense with:
 *  8-way SIMD: 4x4 setup, performance wise once useful on CPUs without FMA or AMD Zen1
 * 16-way SIMD: 4x8 setup, used by AVX-512
 */
#    if GMX_SIMD_REAL_WIDTH == 2 || GMX_SIMD_REAL_WIDTH == 4 || GMX_SIMD_REAL_WIDTH == 8
#        define GMX_NBNXN_SIMD_4XN
#    endif
#    if GMX_SIMD_REAL_WIDTH == 8 || GMX_SIMD_REAL_WIDTH == 16
#        define GMX_NBNXN_SIMD_2XNN
#    endif

#    if !(defined GMX_NBNXN_SIMD_4XN || defined GMX_NBNXN_SIMD_2XNN)
#        error "No SIMD kernel type defined"
#    endif

#endif // GMX_SIMD && GMX_USE_SIMD_KERNELS

#endif // GMX_NBNXM_NBNXM_SIMD
