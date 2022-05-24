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
 * \brief Defines function for computing Lennard-Jones interaction using SIMD
 *
 * The philosophy is as follows. The different flavors of LJ interactions types
 * and calculating energies or not are set using constexpr variables. These are
 * then used to select the appropriate templated functions in this files through
 * template specialization. Functions in this file take C-style arrays of SIMD
 * of size \p nR registers as arguments, indicated by a suffix with the letter 'V',
 * which are a list of registers with one for each i-particle (4xM kernels) or
 * pair of i-particles (2xMM kernels) that have LJ interactions. This can be half
 * the number of total registers when only part of the i-atoms have LJ.
 * Note that we do not use functions for single SIMD registers because this limits
 * the instruction parallelism that compilers can extract.
 *
 * The call sequence for functions in this file is:
 *
 * The calling code declares a C-style array with \p numLJShiftOrSwitchParams()
 * LJ shift/switch parameters.
 *
 * The LJ shift/switch parameters are set with \p setLJShiftOrSwitchParameters()
 *
 * For each cluster pair, \p lennardJonesInteractionsC612() or
 * \p lennardJonesInteractionsSigmaEpsilon() is called, depending on the combination
 * rule. Additionaly \p addLennardJonesEwaldCorrections() needs to be called when
 * using Ewald for Lennard-Jones.
 *
 * Note that when atoms can get very close to each other, which we assume only
 * happens when atoms are excluded, the masking template parameter should be set
 * to true to avoid overflows when calculating r^-6.
 *
 * Note that only plain or potential-shifted LJ interactions are supported with
 * Lorentz-Berthelot combination rules. For switched LJ interations choose no
 * combination rule.
 *
 * \author Berk Hess <hess@kth.se>
 */

#ifndef GMX_NBNXM_SIMD_LENNARDJONES_FUNCTIONS_H
#define GMX_NBNXM_SIMD_LENNARDJONES_FUNCTIONS_H

#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/simd/simd.h"

#include "atomdata.h"

namespace gmx
{

//! The fraction of i-particles for which LJ interactions need to be computed
enum class ILJInteractions
{
    All,  //!< all i-particles
    Half, //!< the first half of the i-particles
    None  //!< none of i-particles
};

//! Returns the number of shift parameters for LJ with (un)shifted potential
template<InteractionModifiers vdwModifier, bool calculateEnergies>
inline constexpr std::enable_if_t<vdwModifier == InteractionModifiers::None || vdwModifier == InteractionModifiers::PotShift, int>
numLJShiftOrSwitchParams()
{
    if constexpr (calculateEnergies)
    {
        return 2;
    }
    else
    {
        return 0;
    }
}

//! Sets the shift parameters for plain LJ or LJ with shifted potential
template<InteractionModifiers vdwModifier, bool calculateEnergies>
inline std::enable_if_t<vdwModifier == InteractionModifiers::None || vdwModifier == InteractionModifiers::PotShift, void>
setLJShiftOrSwitchParameters(const interaction_const_t& ic, SimdReal* shiftParams)
{
    // We shift the potentials by cpot, which can be zero
    shiftParams[0] = SimdReal(ic.dispersion_shift.cpot);
    shiftParams[1] = SimdReal(ic.repulsion_shift.cpot);
}

//! Computes r^-6 and r^-12, masked when requested
template<int nR, bool maskInteractions>
inline void rInvSixAndRInvTwelve(SimdReal* rInvSquaredV, SimdBool* interactV, SimdReal* rInvSixV, SimdReal* rInvTwelveV)
{
    for (int i = 0; i < nR; i++)
    {
        rInvSixV[i] = rInvSquaredV[i] * rInvSquaredV[i] * rInvSquaredV[i];
    }
    if constexpr (maskInteractions)
    {
        for (int i = 0; i < nR; i++)
        {
            rInvSixV[i] = selectByMask(rInvSixV[i], interactV[i]);
        }
    }

    for (int i = 0; i < nR; i++)
    {
        rInvTwelveV[i] = rInvSixV[i] * rInvSixV[i];
    }
}

//! Computes F*r for LJ with (un)shifted potential with C6/C12 parameters
template<int nR, InteractionModifiers vdwModifier, bool maskInteractions, bool calculateEnergies>
inline std::enable_if_t<(vdwModifier == InteractionModifiers::None || vdwModifier == InteractionModifiers::PotShift) && !calculateEnergies, void>
lennardJonesInteractionsC6C12(SimdReal* rSquaredV,
                              SimdReal* rInvV,
                              SimdReal* rInvSquaredV,
                              SimdBool* interactV,
                              SimdReal* c6V,
                              SimdReal* c12V,
                              SimdReal* shiftParams,
                              SimdReal  sixth,
                              SimdReal  twelfth,
                              SimdReal* frLJV,
                              SimdReal* vLJV)
{
    SimdReal rInvSixV[nR];
    SimdReal rInvTwelveV[nR];
    rInvSixAndRInvTwelve<nR, maskInteractions>(rInvSquaredV, interactV, rInvSixV, rInvTwelveV);

    for (int i = 0; i < nR; i++)
    {
        frLJV[i] = fms(c12V[i], rInvTwelveV[i], c6V[i] * rInvSixV[i]);
    }

    GMX_UNUSED_VALUE(rSquaredV);
    GMX_UNUSED_VALUE(rInvV);
    GMX_UNUSED_VALUE(rInvSquaredV);
    GMX_UNUSED_VALUE(shiftParams);
    GMX_UNUSED_VALUE(sixth);
    GMX_UNUSED_VALUE(twelfth);
    GMX_UNUSED_VALUE(vLJV);
}

//! Computes F*r and the potential for LJ with (un)shifted potential with C6/C12 parameters
template<int nR, InteractionModifiers vdwModifier, bool maskInteractions, bool calculateEnergies>
inline std::enable_if_t<(vdwModifier == InteractionModifiers::None || vdwModifier == InteractionModifiers::PotShift) && calculateEnergies, void>
lennardJonesInteractionsC6C12(SimdReal* rSquaredV,
                              SimdReal* rInvV,
                              SimdReal* rInvSquaredV,
                              SimdBool* interactV,
                              SimdReal* c6V,
                              SimdReal* c12V,
                              SimdReal* shiftParams,
                              SimdReal  sixth,
                              SimdReal  twelfth,
                              SimdReal* frLJV,
                              SimdReal* vLJV)
{
    SimdReal frLJ6V[nR];
    SimdReal frLJ12V[nR];
    rInvSixAndRInvTwelve<nR, maskInteractions>(rInvSquaredV, interactV, frLJ6V, frLJ12V);

    for (int i = 0; i < nR; i++)
    {
        frLJ6V[i]  = c6V[i] * frLJ6V[i];
        frLJ12V[i] = c12V[i] * frLJ12V[i];
    }
    for (int i = 0; i < nR; i++)
    {
        frLJV[i] = frLJ12V[i] - frLJ6V[i];
    }
    for (int i = 0; i < nR; i++)
    {
        vLJV[i] = sixth * fma(c6V[i], shiftParams[0], frLJ6V[i]);
    }
    for (int i = 0; i < nR; i++)
    {
        vLJV[i] = fms(twelfth, fma(c12V[i], shiftParams[1], frLJ12V[i]), vLJV[i]);
    }

    GMX_UNUSED_VALUE(rSquaredV);
    GMX_UNUSED_VALUE(rInvV);
}

//! Returns F*r and optionally the potential for LJ with (un)shifted potential with sigma/epsilon
template<int nR, bool maskInteractions, bool haveCutoffCheck, bool calculateEnergies>
inline void lennardJonesInteractionsSigmaEpsilon(SimdReal* rInvV,
                                                 SimdBool* interactV,
                                                 SimdBool* withinCutoffV,
                                                 SimdReal* sigmaV,
                                                 SimdReal* epsilonV,
                                                 SimdReal* shiftParams,
                                                 SimdReal  sixth,
                                                 SimdReal  twelfth,
                                                 SimdReal* frLJV,
                                                 SimdReal* vLJV)
{
    SimdReal sigmaInvRV[nR];
    for (int i = 0; i < nR; i++)
    {
        sigmaInvRV[i] = sigmaV[i] * rInvV[i];
    }
    SimdReal sigmaInvR2V[nR];
    for (int i = 0; i < nR; i++)
    {
        sigmaInvR2V[i] = sigmaInvRV[i] * sigmaInvRV[i];
    }
    SimdReal sigmaInvR6V[nR];
    for (int i = 0; i < nR; i++)
    {
        sigmaInvR6V[i] = sigmaInvR2V[i] * sigmaInvR2V[i] * sigmaInvR2V[i];
        if constexpr (maskInteractions)
        {
            sigmaInvR6V[i] = selectByMask(sigmaInvR6V[i], interactV[i]);
        }
    }

    if constexpr (haveCutoffCheck)
    {
        for (int i = 0; i < nR; i++)
        {
            sigmaInvR6V[i] = selectByMask(sigmaInvR6V[i], withinCutoffV[i]);
        }
    }

    SimdReal frLJ6V[nR];
    for (int i = 0; i < nR; i++)
    {
        frLJ6V[i] = epsilonV[i] * sigmaInvR6V[i];
    }
    SimdReal frLJ12V[nR];
    for (int i = 0; i < nR; i++)
    {
        frLJ12V[i] = frLJ6V[i] * sigmaInvR6V[i];
        frLJV[i]   = frLJ12V[i] - frLJ6V[i];
    }

    if constexpr (calculateEnergies)
    {
        /* We need C6 and C12 to calculate the LJ potential shift */
        SimdReal sigma2V[nR];
        for (int i = 0; i < nR; i++)
        {
            sigma2V[i] = sigmaV[i] * sigmaV[i];
        }
        SimdReal sigma6V[nR];
        for (int i = 0; i < nR; i++)
        {
            sigma6V[i] = sigma2V[i] * sigma2V[i] * sigma2V[i];
        }
        SimdReal c6V[nR];
        for (int i = 0; i < nR; i++)
        {
            c6V[i] = epsilonV[i] * sigma6V[i];
        }
        SimdReal c12V[nR];
        for (int i = 0; i < nR; i++)
        {
            c12V[i] = c6V[i] * sigma6V[i];
        }

        /* Calculate the LJ energies, with constant potential shift */
        for (int i = 0; i < nR; i++)
        {
            vLJV[i] = sixth * fma(c6V[i], shiftParams[0], frLJ6V[i]);
        }
        for (int i = 0; i < nR; i++)
        {
            vLJV[i] = fms(twelfth, fma(c12V[i], shiftParams[1], frLJ12V[i]), vLJV[i]);
        }
    }
    else
    {
        GMX_UNUSED_VALUE(sixth);
        GMX_UNUSED_VALUE(twelfth);
    }
}

//! Returns the number of switch parameters for LJ with force switch
template<InteractionModifiers vdwModifier, bool calculateEnergies>
inline constexpr std::enable_if_t<vdwModifier == InteractionModifiers::ForceSwitch, int> numLJShiftOrSwitchParams()
{
    if constexpr (calculateEnergies)
    {
        return 11;
    }
    else
    {
        return 5;
    }
}

//! Sets the switch parameters for LJ with force switch
template<InteractionModifiers vdwModifier, bool calculateEnergies>
inline std::enable_if_t<vdwModifier == InteractionModifiers::ForceSwitch, void>
setLJShiftOrSwitchParameters(const interaction_const_t& ic, SimdReal* switchParams)
{
    switchParams[0] = SimdReal(ic.rvdw_switch);
    switchParams[1] = SimdReal(ic.dispersion_shift.c2);
    switchParams[2] = SimdReal(ic.dispersion_shift.c3);
    switchParams[3] = SimdReal(ic.repulsion_shift.c2);
    switchParams[4] = SimdReal(ic.repulsion_shift.c3);

    if constexpr (calculateEnergies)
    {
        SimdReal mthird_S(-1.0_real / 3.0_real);
        SimdReal mfourth_S(-1.0_real / 4.0_real);

        switchParams[5]  = mthird_S * switchParams[1];
        switchParams[6]  = mfourth_S * switchParams[2];
        switchParams[7]  = SimdReal(ic.dispersion_shift.cpot / 6.0_real);
        switchParams[8]  = mthird_S * switchParams[3];
        switchParams[9]  = mfourth_S * switchParams[4];
        switchParams[10] = SimdReal(ic.repulsion_shift.cpot / 12.0_real);
    }
}

//! Computes (r - r_switch), (r - r_switch)^2 and (r - r_switch)^2 * r
template<int nR>
inline void computeForceSwitchVariables(SimdReal* rSquaredV,
                                        SimdReal* rInvV,
                                        SimdReal  rSwitch,
                                        SimdReal* rSwitchedV,
                                        SimdReal* rSwitchedSquaredV,
                                        SimdReal* rSwitchedSquaredTimesRV)
{
    SimdReal rV[nR];
    for (int i = 0; i < nR; i++)
    {
        rV[i]                = rSquaredV[i] * rInvV[i];
        rSwitchedV[i]        = max(rV[i] - rSwitch, setZero());
        rSwitchedSquaredV[i] = rSwitchedV[i] * rSwitchedV[i];
    }

    for (int i = 0; i < nR; i++)
    {
        rSwitchedSquaredTimesRV[i] = rSwitchedSquaredV[i] * rV[i];
    }
}

//! Adds the force switch term to \p force
inline SimdReal addLJForceSwitch(SimdReal force,
                                 SimdReal rSwitched,
                                 SimdReal rSwitchedSquaredTimesR,
                                 SimdReal c2,
                                 SimdReal c3)
{
    return fma(fma(c3, rSwitched, c2), rSwitchedSquaredTimesR, force);
}

//! Returns the LJ force switch function for the potential
inline SimdReal ljForceSwitchPotential(SimdReal rSwitched,
                                       SimdReal rSwitchedSquaredTimesR,
                                       SimdReal c0,
                                       SimdReal c3,
                                       SimdReal c4)
{
    return fma(fma(c4, rSwitched, c3), rSwitchedSquaredTimesR * rSwitched, c0);
}

//! Computes F*r and optionally the potential for LJ with force switch and C6/C12 parameters
template<int nR, InteractionModifiers vdwModifier, bool maskInteractions, bool calculateEnergies>
inline std::enable_if_t<vdwModifier == InteractionModifiers::ForceSwitch, void>
lennardJonesInteractionsC6C12(SimdReal* rSquaredV,
                              SimdReal* rInvV,
                              SimdReal* rInvSquaredV,
                              SimdBool* interactV,
                              SimdReal* c6V,
                              SimdReal* c12V,
                              SimdReal* switchParams,
                              SimdReal  sixth,
                              SimdReal  twelfth,
                              SimdReal* frLJV,
                              SimdReal* vLJV)
{
    SimdReal rInvSixV[nR];
    SimdReal rInvTwelveV[nR];
    rInvSixAndRInvTwelve<nR, maskInteractions>(rInvSquaredV, interactV, rInvSixV, rInvTwelveV);

    SimdReal rSwitchedV[nR];
    SimdReal rSwitchedSquaredV[nR];
    SimdReal rSwitchedSquaredTimesRV[nR];
    computeForceSwitchVariables<nR>(
            rSquaredV, rInvV, switchParams[0], rSwitchedV, rSwitchedSquaredV, rSwitchedSquaredTimesRV);

    for (int i = 0; i < nR; i++)
    {
        frLJV[i] = c6V[i]
                   * addLJForceSwitch(rInvSixV[i],
                                      rSwitchedV[i],
                                      rSwitchedSquaredTimesRV[i],
                                      switchParams[1],
                                      switchParams[2]);
    }
    for (int i = 0; i < nR; i++)
    {
        frLJV[i] = c12V[i]
                           * addLJForceSwitch(rInvTwelveV[i],
                                              rSwitchedV[i],
                                              rSwitchedSquaredTimesRV[i],
                                              switchParams[3],
                                              switchParams[4])
                   - frLJV[i];
    }

    if constexpr (calculateEnergies)
    {
        for (int i = 0; i < nR; i++)
        {
            vLJV[i] = c6V[i]
                      * fma(sixth,
                            rInvSixV[i],
                            ljForceSwitchPotential(rSwitchedV[i],
                                                   rSwitchedSquaredV[i],
                                                   switchParams[7],
                                                   switchParams[5],
                                                   switchParams[6]));
        }
        for (int i = 0; i < nR; i++)
        {
            vLJV[i] = c12V[i]
                              * fma(twelfth,
                                    rInvSixV[i] * rInvSixV[i],
                                    ljForceSwitchPotential(rSwitchedV[i],
                                                           rSwitchedSquaredV[i],
                                                           switchParams[10],
                                                           switchParams[8],
                                                           switchParams[9]))
                      - vLJV[i];
        }
    }
    else
    {
        GMX_UNUSED_VALUE(sixth);
        GMX_UNUSED_VALUE(twelfth);
    }
}

//! Returns the number of switch parameters for LJ with potential switch
template<InteractionModifiers vdwModifier, bool calculateEnergies>
inline constexpr std::enable_if_t<vdwModifier == InteractionModifiers::PotSwitch, int> numLJShiftOrSwitchParams()
{
    return 7;
}

//! Sets the switch parameters for LJ with potential switch
template<InteractionModifiers vdwModifier, bool calculateEnergies>
inline std::enable_if_t<vdwModifier == InteractionModifiers::PotSwitch, void>
setLJShiftOrSwitchParameters(const interaction_const_t& ic, SimdReal* switchParams)
{
    switchParams[0] = SimdReal(ic.rvdw_switch);
    switchParams[1] = SimdReal(ic.vdw_switch.c3);
    switchParams[2] = SimdReal(ic.vdw_switch.c4);
    switchParams[3] = SimdReal(ic.vdw_switch.c5);
    switchParams[4] = SimdReal(3.0_real * ic.vdw_switch.c3);
    switchParams[5] = SimdReal(4.0_real * ic.vdw_switch.c4);
    switchParams[6] = SimdReal(5.0_real * ic.vdw_switch.c5);
}

//! Computes (r - r_switch) and (r - r_switch)^2
template<int nR>
inline void computePotentialSwitchVariables(SimdReal* rSquaredV,
                                            SimdReal* rInvV,
                                            SimdReal  rSwitch,
                                            SimdReal* rSwitchedV,
                                            SimdReal* rSwitchedSquaredV)
{
    SimdReal rV[nR];
    for (int i = 0; i < nR; i++)
    {
        rV[i]                = rSquaredV[i] * rInvV[i];
        rSwitchedV[i]        = max(rV[i] - rSwitch, setZero());
        rSwitchedSquaredV[i] = rSwitchedV[i] * rSwitchedV[i];
    }
}

//! Returns the potential switch function
inline SimdReal potentialSwitchFunction(SimdReal rsw, SimdReal rsw2, SimdReal c3, SimdReal c4, SimdReal c5)
{
    return fma(fma(fma(c5, rsw, c4), rsw, c3), rsw2 * rsw, SimdReal(1.0_real));
}

//! Returns the derivative of the potential switch function
inline SimdReal potentialSwitchFunctionDerivative(SimdReal rsw, SimdReal rsw2, SimdReal c2, SimdReal c3, SimdReal c4)
{
    return fma(fma(c4, rsw, c3), rsw, c2) * rsw2;
}

//! Computes F*r and optionally the potential for LJ with potential switch and C6/C12 parameters
template<int nR, InteractionModifiers vdwModifier, bool maskInteractions, bool calculateEnergies>
inline std::enable_if_t<vdwModifier == InteractionModifiers::PotSwitch, void>
lennardJonesInteractionsC6C12(SimdReal* rSquaredV,
                              SimdReal* rInvV,
                              SimdReal* rInvSquaredV,
                              SimdBool* interactV,
                              SimdReal* c6V,
                              SimdReal* c12V,
                              SimdReal* switchParams,
                              SimdReal  sixth,
                              SimdReal  twelfth,
                              SimdReal* frLJV,
                              SimdReal* vLJV)
{
    /* We always need the potential, since it is needed for the force */
    SimdReal shiftParams[2];
    shiftParams[0] = setZero();
    shiftParams[1] = setZero();
    SimdReal vLJTmpV[nR];
    lennardJonesInteractionsC6C12<nR, InteractionModifiers::None, maskInteractions, true>(
            rSquaredV, rInvV, rInvSquaredV, interactV, c6V, c12V, shiftParams, sixth, twelfth, frLJV, vLJTmpV);

    SimdReal rSwitchedV[nR];
    SimdReal rSwitchedSquaredV[nR];
    computePotentialSwitchVariables<nR>(rSquaredV, rInvV, switchParams[0], rSwitchedV, rSwitchedSquaredV);

    SimdReal switchV[nR];
    SimdReal dSwitchV[nR];
    for (int i = 0; i < nR; i++)
    {
        switchV[i] = potentialSwitchFunction(
                rSwitchedV[i], rSwitchedSquaredV[i], switchParams[1], switchParams[2], switchParams[3]);
        dSwitchV[i] = potentialSwitchFunctionDerivative(
                rSwitchedV[i], rSwitchedSquaredV[i], switchParams[4], switchParams[5], switchParams[6]);
    }
    for (int i = 0; i < nR; i++)
    {
        SimdReal r = rSquaredV[i] * rInvV[i];
        frLJV[i]   = fnma(dSwitchV[i] * vLJTmpV[i], r, switchV[i] * frLJV[i]);
    }
    if constexpr (calculateEnergies)
    {
        for (int i = 0; i < nR; i++)
        {
            vLJV[i] = switchV[i] * vLJTmpV[i];
        }
    }
}

//! Adds the Ewald long-range correction for r^-6
template<int nR, bool maskInteractions, bool calculateEnergies>
inline void addLennardJonesEwaldCorrections(SimdReal*           rSquaredV,
                                            SimdReal*           rInvSquaredV,
                                            SimdBool*           interactV,
                                            SimdBool*           withinCutoffV,
                                            SimdReal*           c6GridV,
                                            SimdReal*           ljEwaldParams,
                                            SimdReal gmx_unused sixth,
                                            SimdReal*           frLJV,
                                            SimdReal*           vLJV)
{
    /* Recalculate r^-6 not masked for exclusions.
     * Note that we could reuse the previously calculated r^-6 which is unmasked
     * for exclusions when not calculating energies.
     */
    SimdReal rInvSixV[nR];
    for (int i = 0; i < nR; i++)
    {
        rInvSixV[i] = rInvSquaredV[i] * rInvSquaredV[i] * rInvSquaredV[i];
    }

    /* Mask for the cut-off to avoid overflow of cr2^2 */
    SimdReal rSquaredMaskedV[nR];
    for (int i = 0; i < nR; i++)
    {
        rSquaredMaskedV[i] = ljEwaldParams[2] * selectByMask(rSquaredV[i], withinCutoffV[i]);
    }
    // Unsafe version of our exp() should be fine, since these arguments should never
    // be smaller than -127 for any reasonable choice of cutoff or ewald coefficients.
    SimdReal expRSquaredMaskedV[nR];
    for (int i = 0; i < nR; i++)
    {
        expRSquaredMaskedV[i] = exp<MathOptimization::Unsafe>(-rSquaredMaskedV[i]);
    }

    /* 1 + cr2 + 1/2*cr2^2 */
    SimdReal polyV[nR];
    for (int i = 0; i < nR; i++)
    {
        polyV[i] = fma(fma(ljEwaldParams[1], rSquaredMaskedV[i], ljEwaldParams[0]),
                       rSquaredMaskedV[i],
                       ljEwaldParams[0]);
    }

    /* We calculate LJ F*r = (6*C6)*(r^-6 - F_mesh/6), we use:
     * r^-6*cexp*(1 + cr2 + cr2^2/2 + cr2^3/6) = cexp*(r^-6*poly + c^6/6)
     */
    for (int i = 0; i < nR; i++)
    {
        frLJV[i] = fma(
                c6GridV[i],
                fnma(expRSquaredMaskedV[i], fma(rInvSixV[i], polyV[i], ljEwaldParams[3]), rInvSixV[i]),
                frLJV[i]);
    }

    if constexpr (calculateEnergies)
    {
        SimdReal shiftMaskedV[nR];
        if constexpr (maskInteractions)
        {
            for (int i = 0; i < nR; i++)
            {
                shiftMaskedV[i] = selectByMask(ljEwaldParams[4], interactV[i]);
            }
        }
        else
        {
            for (int i = 0; i < nR; i++)
            {
                shiftMaskedV[i] = ljEwaldParams[4];
            }
        }

        for (int i = 0; i < nR; i++)
        {
            vLJV[i] = fma(sixth * c6GridV[i],
                          fma(rInvSixV[i],
                              fnma(expRSquaredMaskedV[i], polyV[i], ljEwaldParams[0]),
                              shiftMaskedV[i]),
                          vLJV[i]);
        }
    }
}

} // namespace gmx

#endif // GMX_NBNXM_SIMD_LENNARDJONES_FUNCTIONS_H
