/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2022- The GROMACS Authors
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

/*! \internal
 * \ingroup __module_nbnxm
 *
 * \brief Defines function for computing Coulomb interaction using SIMD
 *
 * The philosophy is as follows. The flavor of Coulomb interactions types
 * is set using constexpr variables. This is then used to select the appropriate
 * templated functions in this files through template specialization.
 * Functions in this file take C-style arrays of SIMD of size \p nR registers
 * as arguments, indicated by a suffix with the letter 'V', which are a list
 * of registers with one for each i-particle (4xM kernels) or
 * pair of i-particles (2xMM kernels) that have LJ interactions.
 * Note that we do not use functions for single SIMD registers because this limits
 * the instruction parallelism that compilers can extract.
 *
 * \author Berk Hess <hess@kth.se>
 */

#ifndef GMX_NBNXM_SIMD_COULOMB_FUNCTIONS_H
#define GMX_NBNXM_SIMD_COULOMB_FUNCTIONS_H

#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/simd/simd.h"

#include "atomdata.h"

namespace gmx
{

enum class KernelCoulombType
{
    RF,
    EwaldAnalytical,
    EwaldTabulatedFDV0,
    EwaldTabulatedFAndV
};

//! Computes Coulomb forces for reaction-field
template<int nR, KernelCoulombType coulombType>
inline std::enable_if_t<coulombType == KernelCoulombType::RF, void>
coulombForce(SimdReal* rSquaredV,
             SimdReal gmx_unused* dummyRInvV,
             SimdReal*            rInvExclV,
             SimdBool gmx_unused* withinCutoffV,
             SimdReal             minusTwoTimesRFCoeff,
             SimdReal gmx_unused  dummyParam,
             const real gmx_unused* dummyTable,
             SimdReal*              forceV)
{
    for (int i = 0; i < nR; i++)
    {
        forceV[i] = fma(rSquaredV[i], minusTwoTimesRFCoeff, rInvExclV[i]);
    }
}

//! Computes Coulomb forces and energies without 1/r term for reaction-field
template<int nR, KernelCoulombType coulombType>
inline std::enable_if_t<coulombType == KernelCoulombType::RF, void>
coulombForceAndCorrectionEnergy(SimdReal* rSquaredV,
                                SimdReal gmx_unused* dummyRInvV,
                                SimdReal*            rInvExclV,
                                SimdBool gmx_unused* withinCutoffV,
                                SimdReal             minusTwoTimesRFCoeff,
                                SimdReal             rfOffset,
                                const real gmx_unused* dummyTable1,
                                const real gmx_unused* dummyTable2,
                                SimdReal*              forceV,
                                SimdReal*              correctionEnergyV)
{
    for (int i = 0; i < nR; i++)
    {
        forceV[i] = fma(rSquaredV[i], minusTwoTimesRFCoeff, rInvExclV[i]);
    }
    SimdReal factor = SimdReal(0.5_real) * minusTwoTimesRFCoeff;
    for (int i = 0; i < nR; i++)
    {
        correctionEnergyV[i] = fma(rSquaredV[i], factor, rfOffset);
    }
}

//! Computes Coulomb forces for Ewald using an analytic approximation
template<int nR, KernelCoulombType coulombType>
inline std::enable_if_t<coulombType == KernelCoulombType::EwaldAnalytical, void>
coulombForce(SimdReal* rSquaredV,
             SimdReal gmx_unused* dummyRInvV,
             SimdReal*            rInvExclV,
             SimdBool*            withinCutoffV,
             SimdReal             beta,
             SimdReal             betaSquared,
             const real gmx_unused* dummyTable,
             SimdReal*              forceV)
{
    SimdReal brsqV[nR];
    for (int i = 0; i < nR; i++)
    {
        brsqV[i] = betaSquared * selectByMask(rSquaredV[i], withinCutoffV[i]);
    }
    SimdReal ewcorrV[nR];
    for (int i = 0; i < nR; i++)
    {
        ewcorrV[i] = beta * pmeForceCorrection(brsqV[i]);
    }
    for (int i = 0; i < nR; i++)
    {
        forceV[i] = fma(ewcorrV[i], brsqV[i], rInvExclV[i]);
    }
}

//! Computes Coulomb forces and correction energies for Ewald using an analytic approximation
template<int nR, KernelCoulombType coulombType>
inline std::enable_if_t<coulombType == KernelCoulombType::EwaldAnalytical, void>
coulombForceAndCorrectionEnergy(SimdReal*   rSquaredV,
                                SimdReal*   rInvV,
                                SimdReal*   rInvExclV,
                                SimdBool*   withinCutoffV,
                                SimdReal    beta,
                                SimdReal    betaSquared,
                                const real* table1,
                                const real* table2,
                                SimdReal*   forceV,
                                SimdReal*   correctionEnergyV)
{
    SimdReal brsqV[nR];
    for (int i = 0; i < nR; i++)
    {
        brsqV[i] = betaSquared * selectByMask(rSquaredV[i], withinCutoffV[i]);
    }
    SimdReal ewcorrV[nR];
    for (int i = 0; i < nR; i++)
    {
        ewcorrV[i] = beta * pmeForceCorrection(brsqV[i]);
    }
    for (int i = 0; i < nR; i++)
    {
        forceV[i] = fma(ewcorrV[i], brsqV[i], rInvExclV[i]);
    }
    for (int i = 0; i < nR; i++)
    {
        correctionEnergyV[i] = beta * pmePotentialCorrection(brsqV[i]);
    }

    GMX_UNUSED_VALUE(rInvV);
    GMX_UNUSED_VALUE(table1);
    GMX_UNUSED_VALUE(table2);
}

//! Computes Coulomb forces for Ewald using tabulated functions
template<int nR, KernelCoulombType coulombType>
inline std::enable_if_t<coulombType == KernelCoulombType::EwaldTabulatedFDV0 || coulombType == KernelCoulombType::EwaldTabulatedFAndV, void>
coulombForce(SimdReal* rSquaredV,
             SimdReal* rInvV,
             SimdReal* rInvExclV,
             SimdBool gmx_unused* withinCutoffV,
             SimdReal             invTableSpacing,
             SimdReal gmx_unused  dummy,
             const real*          tableForce,
             SimdReal*            forceV)
{
    /* We use separate registers for r for tabulated Ewald and LJ to keep the code simpler */
    SimdReal rV[nR];
    for (int i = 0; i < nR; i++)
    {
        rV[i] = rSquaredV[i] * rInvV[i];
    }
    /* Convert r to scaled table units */
    SimdReal rScaledV[nR];
    for (int i = 0; i < nR; i++)
    {
        rScaledV[i] = rV[i] * invTableSpacing;
    }
    /* Truncate scaled r to an int */
    SimdInt32 tableIndexV[nR];
    for (int i = 0; i < nR; i++)
    {
        tableIndexV[i] = cvttR2I(rScaledV[i]);
    }

    /* Convert r to scaled table units */
    SimdReal rScaledTruncatedV[nR];
    for (int i = 0; i < nR; i++)
    {
        rScaledTruncatedV[i] = trunc(rScaledV[i]);
    }
    SimdReal rScaledFractionV[nR];
    for (int i = 0; i < nR; i++)
    {
        rScaledFractionV[i] = rScaledV[i] - rScaledTruncatedV[i];
    }

    /* Load and interpolate table forces and possibly energies.
     * Force and energy can be combined in one table, stride 4: FDV0
     * or in two separate tables with stride 1: F and V
     * Currently single precision uses FDV0, double F and V.
     */
    SimdReal coulombTable0V[nR];
    SimdReal coulombTable1V[nR];
    if constexpr (coulombType == KernelCoulombType::EwaldTabulatedFDV0)
    {
        for (int i = 0; i < nR; i++)
        {
            gatherLoadBySimdIntTranspose<4>(
                    tableForce, tableIndexV[i], &coulombTable0V[i], &coulombTable1V[i]);
        }
    }
    else
    {
        for (int i = 0; i < nR; i++)
        {
            gatherLoadBySimdIntTranspose<1>(
                    tableForce, tableIndexV[i], &coulombTable0V[i], &coulombTable1V[i]);
        }
        for (int i = 0; i < nR; i++)
        {
            coulombTable1V[i] = coulombTable1V[i] - coulombTable0V[i];
        }
    }
    SimdReal forceCorrectionV[nR];
    for (int i = 0; i < nR; i++)
    {
        forceCorrectionV[i] = fma(rScaledFractionV[i], coulombTable1V[i], coulombTable0V[i]);
    }
    for (int i = 0; i < nR; i++)
    {
        forceV[i] = fnma(forceCorrectionV[i], rV[i], rInvExclV[i]);
    }
}

//! Computes Coulomb forces and correction energies for Ewald using tabulated functions
template<int nR, KernelCoulombType coulombType>
inline std::enable_if_t<coulombType == KernelCoulombType::EwaldTabulatedFDV0 || coulombType == KernelCoulombType::EwaldTabulatedFAndV, void>
coulombForceAndCorrectionEnergy(SimdReal* rSquaredV,
                                SimdReal* rInvV,
                                SimdReal* rInvExclV,
                                SimdBool gmx_unused* withinCutoffV,
                                SimdReal             invTableSpacing,
                                SimdReal             minusHalfTableSpacing,
                                const real*          tableForce,
                                const real*          tablePotential,
                                SimdReal*            forceV,
                                SimdReal*            correctionEnergyV)
{
    /* We use separate registers for r for tabulated Ewald and LJ to keep the code simpler */
    SimdReal rV[nR];
    for (int i = 0; i < nR; i++)
    {
        rV[i] = rSquaredV[i] * rInvV[i];
    }
    /* Convert r to scaled table units */
    SimdReal rScaledV[nR];
    for (int i = 0; i < nR; i++)
    {
        rScaledV[i] = rV[i] * invTableSpacing;
    }
    /* Truncate scaled r to an int */
    SimdInt32 tableIndexV[nR];
    for (int i = 0; i < nR; i++)
    {
        tableIndexV[i] = cvttR2I(rScaledV[i]);
    }

    /* Convert r to scaled table units */
    SimdReal rScaledTruncatedV[nR];
    for (int i = 0; i < nR; i++)
    {
        rScaledTruncatedV[i] = trunc(rScaledV[i]);
    }
    SimdReal rScaledFractionV[nR];
    for (int i = 0; i < nR; i++)
    {
        rScaledFractionV[i] = rScaledV[i] - rScaledTruncatedV[i];
    }

    /* Load and interpolate table forces and possibly energies.
     * Force and energy can be combined in one table, stride 4: FDV0
     * or in two separate tables with stride 1: F and V
     * Currently single precision uses FDV0, double F and V.
     */
    SimdReal coulombTable0V[nR];
    SimdReal coulombTable1V[nR];
    SimdReal coulombTablePotV[nR];
    SimdReal dumV[nR];
    if constexpr (coulombType == KernelCoulombType::EwaldTabulatedFDV0)
    {
        for (int i = 0; i < nR; i++)
        {
            gatherLoadBySimdIntTranspose<4>(tableForce,
                                            tableIndexV[i],
                                            &coulombTable0V[i],
                                            &coulombTable1V[i],
                                            &coulombTablePotV[i],
                                            &dumV[i]);
        }
    }
    else
    {
        for (int i = 0; i < nR; i++)
        {
            gatherLoadUBySimdIntTranspose<1>(
                    tableForce, tableIndexV[i], &coulombTable0V[i], &coulombTable1V[i]);
            gatherLoadUBySimdIntTranspose<1>(
                    tablePotential, tableIndexV[i], &coulombTablePotV[i], &dumV[i]);
        }
        for (int i = 0; i < nR; i++)
        {
            coulombTable1V[i] = coulombTable1V[i] - coulombTable0V[i];
        }
    }
    SimdReal forceCorrectionV[nR];
    for (int i = 0; i < nR; i++)
    {
        forceCorrectionV[i] = fma(rScaledFractionV[i], coulombTable1V[i], coulombTable0V[i]);
    }
    for (int i = 0; i < nR; i++)
    {
        forceV[i] = fnma(forceCorrectionV[i], rV[i], rInvExclV[i]);
    }

    for (int i = 0; i < nR; i++)
    {
        correctionEnergyV[i] = fma((minusHalfTableSpacing * rScaledFractionV[i]),
                                   (coulombTable0V[i] + forceCorrectionV[i]),
                                   coulombTablePotV[i]);
    }
}

} // namespace gmx

#endif // GMX_NBNXM_SIMD_COULOMB_FUNCTIONS_H
