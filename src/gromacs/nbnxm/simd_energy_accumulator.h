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
 * \brief Defines a class for accumulating Coulomb and VdW energy contributions
 *
 * Call sequence:
 * - EnergyAccumulator()
 * - for (i-cluster ...) {
 * -   initICluster()
 * -   for (j-cluster ...) {
 * -     addEnergy/ies()
 * -   }
 * -   reduceEnergies()
 * - }
 *
 * Note that with multiple energy groups the energy are accumulated into
 * a list of bins in nbnxn_atomdata_output_t and need to be reduced
 * to the final energy group pair matrix after calling the non-bonded kernel.
 *
 * \author Berk Hess <hess@kth.se>
 */

#ifndef GMX_NBNXM_ENERGY_ACCUMULATOR_H
#define GMX_NBNXM_ENERGY_ACCUMULATOR_H

#include "gromacs/simd/simd.h"

#include "atomdata.h"

namespace gmx
{

//! Adds energies to temporary energy group pair buffers for the 4xM kernel layout
template<std::size_t offsetJJSize>
inline void accumulateGroupPairEnergies4xM(SimdReal energies,
                                           real*    groupPairEnergyBuffersPtr,
                                           const std::array<int, offsetJJSize>& offsetJJ)
{
    static_assert(offsetJJSize == GMX_SIMD_REAL_WIDTH / 2);

    /* We need to balance the number of store operations with
     * the rapidly increasing number of combinations of energy groups.
     * We add to a temporary buffer for 1 i-group vs 2 j-groups.
     */
    for (int jj = 0; jj < GMX_SIMD_REAL_WIDTH / 2; jj++)
    {
        SimdReal groupPairEnergyBuffers =
                load<SimdReal>(groupPairEnergyBuffersPtr + offsetJJ[jj] + jj * GMX_SIMD_REAL_WIDTH);

        store(groupPairEnergyBuffersPtr + offsetJJ[jj] + jj * GMX_SIMD_REAL_WIDTH,
              groupPairEnergyBuffers + energies);
    }
}

//! Adds energies to temporary energy group pair buffers for the 2xMM kernel layout
template<std::size_t offsetJJSize>
inline void accumulateGroupPairEnergies2xMM(SimdReal energies,
                                            real*    groupPairEnergyBuffersPtr0,
                                            real*    groupPairEnergyBuffersPtr1,
                                            const std::array<int, offsetJJSize>& offsetJJ)
{
    static_assert(offsetJJSize == GMX_SIMD_REAL_WIDTH / 4);

    for (int jj = 0; jj < GMX_SIMD_REAL_WIDTH / 4; jj++)
    {
        incrDualHsimd(groupPairEnergyBuffersPtr0 + offsetJJ[jj] + jj * GMX_SIMD_REAL_WIDTH / 2,
                      groupPairEnergyBuffersPtr1 + offsetJJ[jj] + jj * GMX_SIMD_REAL_WIDTH / 2,
                      energies);
    }
}

//! Base energy accumulator class, only specializations are used
template<int, bool, bool>
class EnergyAccumulator;

//! Specialized energy accumulator class for no energy calculation
template<int iClusterSize, bool useEnergyGroups>
class EnergyAccumulator<iClusterSize, useEnergyGroups, false>
{
public:
    //! Constructor
    EnergyAccumulator(const nbnxn_atomdata_t::Params gmx_unused& nbatParams,
                      const int gmx_unused                       jClusterSize,
                      nbnxn_atomdata_output_t gmx_unused* nbatOutput)
    {
    }

    //! Does nothing
    inline void initICluster(const int gmx_unused iCluster) {}

    //! Does nothing
    template<int nRCoulomb, int nRVdw, KernelLayout kernelLayout, std::size_t cSize, std::size_t vdwSize>
    inline void addEnergies(const int gmx_unused              jCluster,
                            const std::array<SimdReal, cSize> gmx_unused& coulombEnergy,
                            const std::array<SimdReal, vdwSize> gmx_unused& vdwEnergy)
    {
    }

    //! Does nothing
    inline void addCoulombEnergy(const int gmx_unused iAtomInCluster, const real gmx_unused energy)
    {
    }

    //! Does nothing
    inline void addVdwEnergy(const int gmx_unused iAtomInCluster, const real gmx_unused energy) {}

    //! Does nothing
    inline void reduceIEnergies(const bool gmx_unused calculateCoulomb) {}
};

/*! \brief Specialized energy accumulator class for energy accumulation without energy groups
 *
 * Note that this specialization accumulates over each j-list to internal buffers with an entry
 * per i-particle and then reduces to the final buffers. This is done as to mimimize the rounding
 * errors in the reductions.
 */
template<int iClusterSize>
class EnergyAccumulator<iClusterSize, false, true>
{
public:
    //! Constructor
    EnergyAccumulator(const nbnxn_atomdata_t::Params gmx_unused& nbatParams,
                      const int gmx_unused                       jClusterSize,
                      nbnxn_atomdata_output_t gmx_unused* nbatOutput) :
        coulombEnergySum_(setZero()),
        vdwEnergySum_(setZero()),
        coulombEnergyPtr_(&(nbatOutput->Vc[0])),
        vdwEnergyPtr_(&(nbatOutput->Vvdw[0]))
    {
    }

    //! Clears the accumulation buffer which is used per i-cluster
    inline void initICluster(const int gmx_unused iCluster)
    {
        coulombEnergySum_ = setZero();
        vdwEnergySum_     = setZero();
    }

    //! Adds a single Coulomb energy contribution
    inline void addCoulombEnergy(const int gmx_unused iAtomInCluster, const real energy)
    {
        *coulombEnergyPtr_ += energy;
    }

    //! Adds a single VdW energy contribution
    inline void addVdwEnergy(const int gmx_unused iAtomInCluster, const real energy)
    {
        *vdwEnergyPtr_ += energy;
    }

    /*! \brief Adds Coulomb and/or VdW contributions for interactions of a j-cluster with an i-cluster
     *
     * The first \p nRCoulomb entries of \p coulombEnergy are reduced.
     * The first \p nRVdw entries of \p vdwEnergy are reduced.
     */
    template<int nRCoulomb, int nRVdw, KernelLayout kernelLayout, std::size_t cSize, std::size_t vdwSize>
    inline void addEnergies(const int gmx_unused                 jCluster,
                            const std::array<SimdReal, cSize>&   coulombEnergy,
                            const std::array<SimdReal, vdwSize>& vdwEnergy)
    {
        static_assert(cSize >= nRCoulomb);
        static_assert(vdwSize >= nRVdw);

        for (int i = 0; i < nRCoulomb; i++)
        {
            coulombEnergySum_ = coulombEnergySum_ + coulombEnergy[i];
        }
        for (int i = 0; i < nRVdw; i++)
        {
            vdwEnergySum_ = vdwEnergySum_ + vdwEnergy[i];
        }
    }

    //! Reduces the accumulated energies to the final output buffer
    inline void reduceIEnergies(const bool calculateCoulomb)
    {
        if (calculateCoulomb)
        {
            *coulombEnergyPtr_ += reduce(coulombEnergySum_);
        }

        *vdwEnergyPtr_ += reduce(vdwEnergySum_);
    }

private:
    //! Coulomb energy accumulation buffers for a j-list for one i-cluster
    SimdReal coulombEnergySum_;
    //! VdW energy accumulation buffers for a j-list for one i-cluster
    SimdReal vdwEnergySum_;
    //! Pointer to the output Coulomb energy
    real* coulombEnergyPtr_;
    //! Pointer to the output VdW energy
    real* vdwEnergyPtr_;
};

/*! \brief Specialized energy accumulator class for energy accumulation with energy groups
 *
 * Sums energies into a temporary buffer with bins for each combination of an i-atom energy group
 * with a pair of energy groups for two j-atoms. Reduction of this list of bins into the final
 * energy group pair matrix is done outside the non-bonded kernel.
 */
template<int iClusterSize>
class EnergyAccumulator<iClusterSize, true, true>
{
public:
    //! Constructor
    EnergyAccumulator(const nbnxn_atomdata_t::Params& nbatParams,
                      const int                       jClusterSize,
                      nbnxn_atomdata_output_t*        nbatOutput) :
        iShift_(nbatParams.neg_2log),
        iMask_((1 << iShift_) - 1),
        jShift_(nbatParams.neg_2log * 2),
        jMask_((1 << jShift_) - 1),
        jStride_((jClusterSize >> 1) * jClusterSize),
        iStride_(nbatParams.nenergrp * (1 << nbatParams.neg_2log) * jStride_),
        energyGroups_(nbatParams.energrp.data()),
        coulombEnergyGroupPairBins_(nbatOutput->VSc.data()),
        vdwEnergyGroupPairBins_(nbatOutput->VSvdw.data())
    {
    }

    //! Sets (internal) parameters for the atom in i-cluster \p iCluster
    inline void initICluster(const int iCluster)
    {
        energyGroupsICluster_ = energyGroups_[iCluster];
        for (int iAtom = 0; iAtom < iClusterSize; iAtom++)
        {
            const int iAtomIndex        = (energyGroupsICluster_ >> (iAtom * iShift_)) & iMask_;
            coulombBinIAtomPtrs_[iAtom] = coulombEnergyGroupPairBins_ + iAtomIndex * iStride_;
            vdwBinIAtomPtrs_[iAtom]     = vdwEnergyGroupPairBins_ + iAtomIndex * iStride_;
        }
    }

    //! Adds a single Coulomb energy contribution for atom with index in cluster: \p iAtomInCluster
    inline void addCoulombEnergy(const int iAtomInCluster, const real energy)
    {
        const int pairIndex = ((energyGroupsICluster_ >> (iAtomInCluster * iShift_)) & iMask_) * jStride_;

        coulombBinIAtomPtrs_[iAtomInCluster][pairIndex] += energy;
    }

    //! Adds a single VdW energy contribution for atom with index in cluster: \p iAtomInCluster
    inline void addVdwEnergy(const int iAtomInCluster, const real energy)
    {
        const int pairIndex = ((energyGroupsICluster_ >> (iAtomInCluster * iShift_)) & iMask_) * jStride_;

        vdwBinIAtomPtrs_[iAtomInCluster][pairIndex] += energy;
    }

    /*! \brief Adds Coulomb and/or VdW contributions for interactions of a j-cluster with an i-cluster
     *
     * The first \p nRCoulomb entries of \p coulombEnergy are reduced.
     * The first \p nRVdw entries of \p vdwEnergy are reduced.
     */
    template<int nRCoulomb, int nRVdw, KernelLayout kernelLayout, std::size_t cSize, std::size_t vdwSize>
    inline void addEnergies(const int                            jCluster,
                            const std::array<SimdReal, cSize>&   coulombEnergy,
                            const std::array<SimdReal, vdwSize>& vdwEnergy)
    {
        static_assert(cSize >= nRCoulomb);
        static_assert(vdwSize >= nRVdw);

        constexpr int jClusterSize =
                (kernelLayout == KernelLayout::r4xM ? GMX_SIMD_REAL_WIDTH : GMX_SIMD_REAL_WIDTH / 2);

        /* Energy group indices for two atom pairs packed into one int, one int for each i-atom */
        std::array<int, jClusterSize / 2> ijGroupPair;

        /* Extract the group index per j pair.
         * Energy groups are stored per i-cluster, so things get
         * complicated when the i- and j-cluster sizes don't match.
         */
        if constexpr (jClusterSize == 2)
        {
            const int jPairGroups = energyGroups_[jCluster >> 1];
            ijGroupPair[0] = ((jPairGroups >> ((jCluster & 1) * jShift_)) & jMask_) * jStride_;
        }
        else
        {
            static_assert(iClusterSize <= jClusterSize);

            for (int jdi = 0; jdi < jClusterSize / iClusterSize; jdi++)
            {
                const int jPairGroups = energyGroups_[jCluster * (jClusterSize / iClusterSize) + jdi];
                for (int jj = 0; jj < (iClusterSize / 2); jj++)
                {
                    ijGroupPair[jdi * (iClusterSize / 2) + jj] =
                            ((jPairGroups >> (jj * jShift_)) & jMask_) * jStride_;
                }
            }
        }

        for (int i = 0; i < nRCoulomb; i++)
        {
            if constexpr (kernelLayout == KernelLayout::r4xM)
            {
                accumulateGroupPairEnergies4xM(coulombEnergy[i], coulombBinIAtomPtrs_[i], ijGroupPair);
            }
            else
            {
                accumulateGroupPairEnergies2xMM(coulombEnergy[i],
                                                coulombBinIAtomPtrs_[i * 2],
                                                coulombBinIAtomPtrs_[i * 2 + 1],
                                                ijGroupPair);
            }
        }

        for (int i = 0; i < nRVdw; i++)
        {
            if constexpr (kernelLayout == KernelLayout::r4xM)
            {
                accumulateGroupPairEnergies4xM(vdwEnergy[i], vdwBinIAtomPtrs_[i], ijGroupPair);
            }
            else
            {
                accumulateGroupPairEnergies2xMM(
                        vdwEnergy[i], vdwBinIAtomPtrs_[i * 2], vdwBinIAtomPtrs_[i * 2 + 1], ijGroupPair);
            }
        }
    }

    //! Nothing do to here, reduction happens after the kernel call
    inline void reduceIEnergies(const bool gmx_unused calculateCoulomb) {}

private:
    //! i-cluster shift
    const int iShift_;
    //! i-cluster mask
    const int iMask_;
    //! j-cluster shift
    const int jShift_;
    //! j-cluster mask
    const int jMask_;
    //! j-cluster stride
    const int jStride_;
    //! Major division is over i-particle energy groups, this gives the bin stride for an i-atom
    const int iStride_;

    //! Pointer to a list of energy groups for j-clusters, packed into an int
    const int* energyGroups_;

    //! Energy groups for the i-cluster, packed into an int
    int energyGroupsICluster_;
    //! Pointers to the Coulomb energy bins for the atoms in the current i-cluster
    std::array<real*, iClusterSize> coulombBinIAtomPtrs_;
    //! Pointers to the VdW energy bins for the atoms in the current i-cluster
    std::array<real*, iClusterSize> vdwBinIAtomPtrs_;

    //! Pointer to the complete list of Coulomb energy bins for all energy group pair combinations
    real* coulombEnergyGroupPairBins_;
    //! Pointer to the complete list of VdW energy bins for all energy group pair combinations
    real* vdwEnergyGroupPairBins_;
};

} // namespace gmx

#endif // GMX_NBNXM_ENERGY_ACCUMULATOR_H
