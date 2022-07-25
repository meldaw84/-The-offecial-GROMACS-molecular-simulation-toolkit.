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

/* Doxygen gets confused (buggy) about the block in this file in combination with
 * the  namespace prefix, and thinks store is documented here.
 * This will solve itself with the second-generation nbnxn kernels, so for now
 * we just tell Doxygen to stay out.
 */
#ifndef DOXYGEN

/* This is the innermost loop contents for the 4 x N atom simd kernel.
 * This flavor of the kernel calculates interactions of 4 i-atoms
 * with N j-atoms stored in N wide simd registers.
 */

{
    /* Inner loop specific constexpr variables */
    constexpr int c_nRLJ = nR / (c_iLJInteractions == ILJInteractions::Half ? 2 : 1);
    /* When calculating RF or Ewald interactions we calculate the electrostatic/LJ
     * forces on excluded atom pairs here in the non-bonded loops.
     * But when energies and/or virial is required we calculate them
     * separately to as then it is easier to separate the energy and virial
     * contributions.
     */
    constexpr bool c_haveExclusionForces =
            (c_calculateCoulombInteractions || haveLJEwaldGeometric) && c_needToCheckExclusions;

    /* Energy group indices for two atoms packed into one int */
    int egp_jj[useEnergyGroups ? UNROLLJ / 2 : c_emptyCArraySize];

    /* The force times 1/r */
    SimdReal fScalarV[nR];

    /* j-cluster index */
    const int cj = l_cj[cjind].cj;

    /* Atom indices (of the first atom in the cluster) */
    const int gmx_unused aj = cj * UNROLLJ;
#    if UNROLLJ == STRIDE
    const int ajx = aj * DIM;
#    else
    const int ajx = (cj >> 1) * DIM * STRIDE + (cj & 1) * UNROLLJ;
#    endif
    const int ajy = ajx + STRIDE;
    const int ajz = ajy + STRIDE;

    /* Interaction (non-exclusion) mask of all 1's or 0's */
    // std::array<SimdBool, c_needToCheckExclusions ? nR : c_emptyCArraySize> interactV;
    std::array<SimdBool, nR> interactV;
    if constexpr (c_needToCheckExclusions)
    {
        interactV = loadSimdPairInteractionMasks<kernelLayout>(static_cast<int>(l_cj[cjind].excl),
                                                               exclusionFilterV,
                                                               nbat->simdMasks.interaction_array.data());
    }

    /* load j atom coordinates */
    SimdReal jx_S = loadJAtomData<kernelLayout>(x, ajx);
    SimdReal jy_S = loadJAtomData<kernelLayout>(x, ajy);
    SimdReal jz_S = loadJAtomData<kernelLayout>(x, ajz);

    /* Calculate distance */
    SimdReal dxV[nR];
    SimdReal dyV[nR];
    SimdReal dzV[nR];
    for (int i = 0; i < nR; i++)
    {
        dxV[i] = ixV[i] - jx_S;
        dyV[i] = iyV[i] - jy_S;
        dzV[i] = izV[i] - jz_S;
    }

    /* rsq = dx*dx + dy*dy + dz*dz */
    SimdReal rSquaredV[nR];
    for (int i = 0; i < nR; i++)
    {
        rSquaredV[i] = norm2(dxV[i], dyV[i], dzV[i]);
    }

    /* Do the cut-off check */
    SimdBool withinCutoffV[nR];
    for (int i = 0; i < nR; i++)
    {
        withinCutoffV[i] = (rSquaredV[i] < rc2_S);
    }

    if constexpr (c_needToCheckExclusions)
    {
        if constexpr (c_haveExclusionForces)
        {
            /* Only remove the (sub-)diagonal to avoid double counting */
            if constexpr (UNROLLJ == UNROLLI)
            {
                if (cj == ci_sh)
                {
                    for (int i = 0; i < nR; i++)
                    {
                        withinCutoffV[i] = withinCutoffV[i] && diagonalMaskV[i];
                    }
                }
            }
            else if constexpr (UNROLLJ < UNROLLI)
            {
                if (cj == ci_sh * 2)
                {
                    for (int i = 0; i < nR; i++)
                    {
                        withinCutoffV[i] = withinCutoffV[i] && diagonalMask0V[i];
                    }
                }
                if (cj == ci_sh * 2 + 1)
                {
                    for (int i = 0; i < nR; i++)
                    {
                        withinCutoffV[i] = withinCutoffV[i] && diagonalMask1V[i];
                    }
                }
            }
            else
            {
                if (cj * 2 == ci_sh)
                {
                    for (int i = 0; i < nR; i++)
                    {
                        withinCutoffV[i] = withinCutoffV[i] && diagonalMask0V[i];
                    }
                }
                else if (cj * 2 + 1 == ci_sh)
                {
                    for (int i = 0; i < nR; i++)
                    {
                        withinCutoffV[i] = withinCutoffV[i] && diagonalMask1V[i];
                    }
                }
            }
        }
        else
        {
            /* No exclusion forces: remove all excluded atom pairs from the list */
            for (int i = 0; i < nR; i++)
            {
                withinCutoffV[i] = withinCutoffV[i] && interactV[i];
            }
        }
    }

#    ifdef COUNT_PAIRS
    npair += pairCountWithinCutoff(rSquaredV, rc2_S);
#    endif

    // Ensure the distances do not fall below the limit where r^-12 overflows.
    // This should never happen for normal interactions.
    for (int i = 0; i < nR; i++)
    {
        rSquaredV[i] = max(rSquaredV[i], minRsq_S);
    }

    /* Calculate 1/r */
    SimdReal rInvV[nR];
#    if !GMX_DOUBLE
    for (int i = 0; i < nR; i++)
    {
        rInvV[i] = invsqrt(rSquaredV[i]);
    }
#    else
    for (int i = 0; i < nR; i += 2)
    {
        invsqrtPair(rSquaredV[i], rSquaredV[i + 1], &rInvV[i], &rInvV[i + 1]);
    }
#    endif

    SimdReal qqV[c_calculateCoulombInteractions ? nR : c_emptyCArraySize];
    if constexpr (c_calculateCoulombInteractions)
    {
        /* Load parameters for j atom */
        SimdReal jq_S = loadJAtomData<kernelLayout>(q, aj);
        // SimdReal qqV[nR];
        for (int i = 0; i < nR; i++)
        {
            qqV[i] = chargeIV[i] * jq_S;
        }
    }

    /* Set rinv to zero for r beyond the cut-off */
    for (int i = 0; i < nR; i++)
    {
        rInvV[i] = selectByMask(rInvV[i], withinCutoffV[i]);
    }

    SimdReal rInvSquaredV[nR];
    for (int i = 0; i < nR; i++)
    {
        rInvSquaredV[i] = rInvV[i] * rInvV[i];
    }

    /* frcoul = qi*qj*(1/r - fsub)*r */
    SimdReal            frCoulombV[nR];
    gmx_unused SimdReal vCoulombV[nR];

    if constexpr (c_calculateCoulombInteractions)
    {
        /* Note that here we calculate force*r, not the usual force/r.
         * This allows avoiding masking the reaction-field contribution,
         * as frcoul is later multiplied by rinvsq which has been
         * masked with the cut-off check.
         */


        /* Only add 1/r for non-excluded atom pairs */
        SimdReal rInvExclV[nR];
        if constexpr (c_haveExclusionForces)
        {
            for (int i = 0; i < nR; i++)
            {
                rInvExclV[i] = selectByMask(rInvV[i], interactV[i]);
            }
        }
        else
        {
            /* We hope that the compiler optimizes rInvExclV away */
            for (int i = 0; i < nR; i++)
            {
                rInvExclV[i] = rInvV[i];
            }
        }

        /* The potential (PME mesh) we need to subtract from 1/r */
        SimdReal vCoulombCorrectionV[calculateEnergies ? nR : c_emptyCArraySize];

        /* Electrostatic interactions, frcoul =  qi*qj*(1/r - fsub)*r */
        if constexpr (!calculateEnergies)
        {
            coulombForce<nR, coulombType>(
                    rSquaredV, rInvV, rInvExclV, withinCutoffV, coulombParam1, coulombParam2, tab_coul_F, frCoulombV);
        }
        else
        {
            coulombForceAndCorrectionEnergy<nR, coulombType>(rSquaredV,
                                                             rInvV,
                                                             rInvExclV,
                                                             withinCutoffV,
                                                             coulombParam1,
                                                             coulombParam2,
                                                             tab_coul_F,
                                                             tab_coul_V,
                                                             frCoulombV,
                                                             vCoulombCorrectionV);
        }
        for (int i = 0; i < nR; i++)
        {
            frCoulombV[i] = qqV[i] * frCoulombV[i];
        }

        if constexpr (calculateEnergies)
        {
            if constexpr (coulombType != KernelCoulombType::RF)
            {
                if constexpr (c_needToCheckExclusions)
                {
                    for (int i = 0; i < nR; i++)
                    {
                        vCoulombCorrectionV[i] =
                                vCoulombCorrectionV[i] + selectByMask(ewaldShift, interactV[i]);
                    }
                }
                else
                {
                    for (int i = 0; i < nR; i++)
                    {
                        vCoulombCorrectionV[i] = vCoulombCorrectionV[i] + ewaldShift;
                    }
                }
            }

            /* Combine Coulomb and correction terms */
            for (int i = 0; i < nR; i++)
            {
                vCoulombV[i] = qqV[i] * (rInvExclV[i] - vCoulombCorrectionV[i]);
            }

            /* Mask energy for cut-off and diagonal */
            for (int i = 0; i < nR; i++)
            {
                vCoulombV[i] = selectByMask(vCoulombV[i], withinCutoffV[i]);
            }
        }
    }

    /* Lennard-Jones interaction */
    constexpr bool calculateLJInteractions = (c_iLJInteractions != ILJInteractions::None);
    SimdReal       frLJV[calculateLJInteractions ? c_nRLJ : c_emptyCArraySize];
    SimdReal vLJV[(calculateLJInteractions && calculateEnergies) ? c_nRLJ : c_emptyCArraySize];
    if constexpr (calculateLJInteractions)
    {
        SimdBool withinVdwCutoffV[haveVdwCutoffCheck ? c_nRLJ : c_emptyCArraySize];
        if constexpr (haveVdwCutoffCheck)
        {
            for (int i = 0; i < c_nRLJ; i++)
            {
                withinVdwCutoffV[i] = (rSquaredV[i] < rcvdw2_S);
            }
        }

        constexpr int c_numLJCParams =
                (ljCombinationRule == LJCombinationRule::LorentzBerthelot && !calculateEnergies)
                        ? c_emptyCArraySize
                        : c_nRLJ;
        SimdReal c6V[c_numLJCParams];
        SimdReal c12V[c_numLJCParams];

        /* Index for loading LJ parameters, complicated when interleaving */
        int aj2;
        if constexpr (ljCombinationRule != LJCombinationRule::None || haveLJEwaldGeometric)
        {
            if constexpr (GMX_SIMD_REAL_WIDTH == GMX_SIMD_J_UNROLL_SIZE * STRIDE)
            {
                aj2 = aj * 2;
            }
            else
            {
                aj2 = (cj >> 1) * 2 * STRIDE + (cj & 1) * UNROLLJ;
            }
        }

        if constexpr (ljCombinationRule != LJCombinationRule::LorentzBerthelot)
        {
            /* We use C6 and C12 */

            if constexpr (ljCombinationRule == LJCombinationRule::None)
            {
                for (int i = 0; i < c_nRLJ; i++)
                {
                    gatherLoadTranspose<c_simdBestPairAlignment>(nbfpV[i], type + aj, &c6V[i], &c12V[i]);
                }
            }

            if constexpr (ljCombinationRule == LJCombinationRule::Geometric)
            {
                SimdReal c6s_j_S  = loadJAtomData<kernelLayout>(ljc, aj2 + 0);
                SimdReal c12s_j_S = loadJAtomData<kernelLayout>(ljc, aj2 + STRIDE);
                for (int i = 0; i < c_nRLJ; i++)
                {
                    c6V[i] = c6GeomV[i] * c6s_j_S;
                }
                for (int i = 0; i < c_nRLJ; i++)
                {
                    c12V[i] = c12GeomV[i] * c12s_j_S;
                }
            }

            lennardJonesInteractionsC6C12<c_nRLJ, vdwModifier, c_haveExclusionForces, calculateEnergies>(
                    rSquaredV,
                    rInvV,
                    rInvSquaredV,
                    interactV.data(),
                    c6V,
                    c12V,
                    ljShiftOrSwitchParams,
                    sixth_S,
                    twelveth_S,
                    frLJV,
                    vLJV);
        }

        if constexpr (ljCombinationRule == LJCombinationRule::LorentzBerthelot)
        {
            SimdReal halfSigmaJ   = loadJAtomData<kernelLayout>(ljc, aj2 + 0);
            SimdReal sqrtEpsilonJ = loadJAtomData<kernelLayout>(ljc, aj2 + STRIDE);

            SimdReal sigmaV[c_nRLJ];
            SimdReal epsilonV[c_nRLJ];
            for (int i = 0; i < c_nRLJ; i++)
            {
                sigmaV[i]   = halfSigmaIV[i] + halfSigmaJ;
                epsilonV[i] = sqrtEpsilonIV[i] * sqrtEpsilonJ;
            }

            lennardJonesInteractionsSigmaEpsilon<c_nRLJ, c_haveExclusionForces, haveVdwCutoffCheck, calculateEnergies>(
                    rInvV,
                    interactV.data(),
                    withinVdwCutoffV,
                    sigmaV,
                    epsilonV,
                    ljShiftOrSwitchParams,
                    sixth_S,
                    twelveth_S,
                    frLJV,
                    vLJV);
        }

        if constexpr (calculateEnergies && c_needToCheckExclusions)
        {
            /* The potential shift should be removed for excluded pairs */
            for (int i = 0; i < c_nRLJ; i++)
            {
                vLJV[i] = selectByMask(vLJV[i], interactV[i]);
            }
        }

        if constexpr (haveLJEwaldGeometric)
        {
            /* Determine C6 for the grid using the geometric combination rule */
            SimdReal c6s_j_S = loadJAtomData<kernelLayout>(ljc, aj2 + 0);
            SimdReal c6GridV[c_nRLJ];
            for (int i = 0; i < c_nRLJ; i++)
            {
                c6GridV[i] = c6GeomV[i] * c6s_j_S;
            }

            addLennardJonesEwaldCorrections<c_nRLJ, c_needToCheckExclusions, calculateEnergies>(
                    rSquaredV, rInvSquaredV, interactV.data(), withinCutoffV, c6GridV, ljEwaldParams, sixth_S, frLJV, vLJV);
        }

        if constexpr (haveVdwCutoffCheck)
        {
            /* frLJ is multiplied later by rinvsq, which is masked for the Coulomb
             * cut-off, but if the VdW cut-off is shorter, we need to mask with that.
             */
            for (int i = 0; i < c_nRLJ; i++)
            {
                frLJV[i] = selectByMask(frLJV[i], withinVdwCutoffV[i]);
            }
        }

        if constexpr (calculateEnergies)
        {
            /* The potential shift should be removed for pairs beyond cut-off */
            SimdBool* withinLJCutoffV = haveVdwCutoffCheck ? withinVdwCutoffV : withinCutoffV;
            for (int i = 0; i < c_nRLJ; i++)
            {
                vLJV[i] = selectByMask(vLJV[i], withinLJCutoffV[i]);
            }
        }

    } // calculateLJInteractions

    if constexpr (calculateEnergies)
    {
        if constexpr (useEnergyGroups)
        {
            /* Extract the group pair index per j pair.
             * Energy groups are stored per i-cluster, so things get
             * complicated when the i- and j-cluster size don't match.
             */
#    if UNROLLJ == 2
            const int egps_j = nbatParams.energrp[cj >> 1];
            egp_jj[0]        = ((egps_j >> ((cj & 1) * egps_jshift)) & egps_jmask) * egps_jstride;
#    else
            /* We assume UNROLLI <= UNROLLJ */
            for (int jdi = 0; jdi < UNROLLJ / UNROLLI; jdi++)
            {
                const int egps_j = nbatParams.energrp[cj * (UNROLLJ / UNROLLI) + jdi];
                for (int jj = 0; jj < (UNROLLI / 2); jj++)
                {
                    egp_jj[jdi * (UNROLLI / 2) + jj] =
                            ((egps_j >> (jj * egps_jshift)) & egps_jmask) * egps_jstride;
                }
            }
#    endif
        }

        if constexpr (c_calculateCoulombInteractions)
        {
            if constexpr (!useEnergyGroups)
            {
                for (int i = 0; i < nR; i++)
                {
                    vctot_S = vctot_S + vCoulombV[i];
                }
            }
            else
            {
                for (int i = 0; i < nR; i++)
                {
                    if constexpr (kernelLayout == KernelLayout::r4xM)
                    {
                        accumulateGroupPairEnergies4xM<kernelLayout>(vCoulombV[i], vctp[i], egp_jj);
                    }
                    else
                    {
                        accumulateGroupPairEnergies2xMM<kernelLayout>(
                                vCoulombV[i], vctp[i * 2], vctp[i * 2 + 1], egp_jj);
                    }
                }
            }
        }

        if constexpr (c_iLJInteractions != ILJInteractions::None)
        {
            if constexpr (!useEnergyGroups)
            {
                for (int i = 0; i < c_nRLJ; i++)
                {
                    Vvdwtot_S = Vvdwtot_S + vLJV[i];
                }
            }
            else
            {
                for (int i = 0; i < c_nRLJ; i++)
                {
                    if constexpr (kernelLayout == KernelLayout::r4xM)
                    {
                        accumulateGroupPairEnergies4xM<kernelLayout>(vLJV[i], vvdwtp[i], egp_jj);
                    }
                    else
                    {
                        accumulateGroupPairEnergies2xMM<kernelLayout>(
                                vLJV[i], vvdwtp[i * 2], vvdwtp[i * 2 + 1], egp_jj);
                    }
                }
            }
        }

    } // calculateEnergies

    if constexpr (c_iLJInteractions != ILJInteractions::None)
    {
        if constexpr (c_calculateCoulombInteractions)
        {
            for (int i = 0; i < c_nRLJ; i++)
            {
                fScalarV[i] = rInvSquaredV[i] * (frCoulombV[i] + frLJV[i]);
            }
            for (int i = c_nRLJ; i < nR; i++)
            {
                fScalarV[i] = rInvSquaredV[i] * frCoulombV[i];
            }
        }
        else
        {
            for (int i = 0; i < c_nRLJ; i++)
            {
                fScalarV[i] = rInvSquaredV[i] * frLJV[i];
            }
        }
    }
    else
    {
        for (int i = c_nRLJ; i < nR; i++)
        {
            fScalarV[i] = rInvSquaredV[i] * frCoulombV[i];
        }
    }

    /* Calculate temporary vectorial force */
    SimdReal txV[nR];
    for (int i = 0; i < nR; i++)
    {
        txV[i] = fScalarV[i] * dxV[i];
    }
    SimdReal tyV[nR];
    for (int i = 0; i < nR; i++)
    {
        tyV[i] = fScalarV[i] * dyV[i];
    }
    SimdReal tzV[nR];
    for (int i = 0; i < nR; i++)
    {
        tzV[i] = fScalarV[i] * dzV[i];
    }

    /* Increment i atom force */
    for (int i = 0; i < nR; i++)
    {
        forceIXV[i] = forceIXV[i] + txV[i];
    }
    for (int i = 0; i < nR; i++)
    {
        forceIYV[i] = forceIYV[i] + tyV[i];
    }
    for (int i = 0; i < nR; i++)
    {
        forceIZV[i] = forceIZV[i] + tzV[i];
    }

    /* Decrement j atom force */
#    if GMX_SIMD_J_UNROLL_SIZE == 1
    store(f + ajx, load<SimdReal>(f + ajx) - (txV[0] + txV[1] + txV[2] + txV[3]));
    store(f + ajy, load<SimdReal>(f + ajy) - (tyV[0] + tyV[1] + tyV[2] + tyV[3]));
    store(f + ajz, load<SimdReal>(f + ajz) - (tzV[0] + tzV[1] + tzV[2] + tzV[3]));
#    else
    decr3Hsimd(f + aj * DIM, txV[0] + txV[1], tyV[0] + tyV[1], tzV[0] + tzV[1]);
#    endif
}

#endif // !DOXYGEN
