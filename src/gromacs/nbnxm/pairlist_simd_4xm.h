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
 * Declares inline-friendly code for making 4xN pairlists
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

//! Stride of the packed x coordinate array
template<KernelLayout kernelLayout>
static constexpr int c_xStride4xN()
{
    return std::max(GMX_SIMD_REAL_WIDTH, c_iClusterSize(kernelLayout));
}

template<int clusterSize>
constexpr int log2ClusterSize()
{
    static_assert(clusterSize == 4 || clusterSize == 8, "Only 4 and 8 are supported");

    if constexpr (clusterSize == 4)
    {
        return 2;
    }
    else
    {
        return 3;
    }
}

//! Copies PBC shifted i-cell packed atom coordinates to working array
template<KernelLayout kernelLayout>
static inline void icell_set_x_simd_4xn(int                   ci,
                                        real                  shx,
                                        real                  shy,
                                        real                  shz,
                                        int gmx_unused        stride,
                                        const real*           x,
                                        NbnxnPairlistCpuWork* work)
{
    constexpr int iClusterSize = c_iClusterSize(kernelLayout);

    real* x_ci_simd = work->iClusterData.xSimd.data();

    const int ia = xIndexFromCi<kernelLayout>(ci);

    for (int i = 0; i < iClusterSize; i++)
    {
        store(x_ci_simd + (3 * i + 0) * GMX_SIMD_REAL_WIDTH,
              SimdReal(x[ia + 0 * c_xStride4xN<kernelLayout>() + i] + shx));
        store(x_ci_simd + (3 * i + 1) * GMX_SIMD_REAL_WIDTH,
              SimdReal(x[ia + 1 * c_xStride4xN<kernelLayout>() + i] + shy));
        store(x_ci_simd + (3 * i + 2) * GMX_SIMD_REAL_WIDTH,
              SimdReal(x[ia + 2 * c_xStride4xN<kernelLayout>() + i] + shz));
    }
}

/*! \brief SIMD code for checking and adding cluster-pairs to the list using coordinates in packed format.
 *
 * Checks bounding box distances and possibly atom pair distances.
 * This is an accelerated version of make_cluster_list_simple.
 *
 * \param[in]     jGrid               The j-grid
 * \param[in,out] nbl                 The pair-list to store the cluster pairs in
 * \param[in]     icluster            The index of the i-cluster
 * \param[in]     firstCell           The first cluster in the j-range, using i-cluster size indexing
 * \param[in]     lastCell            The last cluster in the j-range, using i-cluster size indexing
 * \param[in]     excludeSubDiagonal  Exclude atom pairs with i-index > j-index
 * \param[in]     x_j                 Coordinates for the j-atom, in SIMD packed format
 * \param[in]     rlist2              The squared list cut-off
 * \param[in]     rbb2                The squared cut-off for putting cluster-pairs in the list based on bounding box distance only
 * \param[in,out] numDistanceChecks   The number of distance checks performed
 */
template<KernelLayout kernelLayout>
static inline void makeClusterListSimd4xn(const Grid&              jGrid,
                                          NbnxnPairlistCpu*        nbl,
                                          int                      icluster,
                                          int                      firstCell,
                                          int                      lastCell,
                                          bool                     excludeSubDiagonal,
                                          const real* gmx_restrict x_j,
                                          real                     rlist2,
                                          float                    rbb2,
                                          int* gmx_restrict        numDistanceChecks)
{
    using namespace gmx;

    constexpr int iClusterSize = c_iClusterSize(kernelLayout);

    const real* gmx_restrict        x_ci_simd = nbl->work->iClusterData.xSimd.data();
    const BoundingBox* gmx_restrict bb_ci     = nbl->work->iClusterData.bb.data();

    /* Convert the j-range from i-cluster size indexing to j-cluster indexing */
    int jclusterFirst = cjFromCi<kernelLayout, 0>(firstCell);
    int jclusterLast  = cjFromCi<kernelLayout, 1>(lastCell);
    GMX_ASSERT(jclusterLast >= jclusterFirst,
               "We should have a non-empty j-cluster range, since the calling code should have "
               "ensured a non-empty cell range");

    const SimdReal rc2_S = SimdReal(rlist2);

    bool InRange = false;
    while (!InRange && jclusterFirst <= jclusterLast)
    {
        const float d2 = clusterBoundingBoxDistance2(bb_ci[0], jGrid.jBoundingBoxes()[jclusterFirst]);
        *numDistanceChecks += 2;

        /* Check if the distance is within the distance where
         * we use only the bounding box distance rbb,
         * or within the cut-off and there is at least one atom pair
         * within the cut-off.
         */
        if (d2 < rbb2)
        {
            InRange = true;
        }
        else if (d2 < rlist2)
        {
            const int xind_f = xIndexFromCj<kernelLayout>(
                    cjFromCi<kernelLayout, 0>(jGrid.cellOffset()) + jclusterFirst);

            const SimdReal jx_S = load<SimdReal>(x_j + xind_f + 0 * c_xStride4xN<kernelLayout>());
            const SimdReal jy_S = load<SimdReal>(x_j + xind_f + 1 * c_xStride4xN<kernelLayout>());
            const SimdReal jz_S = load<SimdReal>(x_j + xind_f + 2 * c_xStride4xN<kernelLayout>());


            /* Calculate distance */
            std::array<std::array<SimdReal, 3>, iClusterSize> d;
            for (int i = 0; i < iClusterSize; i++)
            {
                d[i][0] = load<SimdReal>(x_ci_simd + (i * 3 + 0) * GMX_SIMD_REAL_WIDTH) - jx_S;
                d[i][1] = load<SimdReal>(x_ci_simd + (i * 3 + 1) * GMX_SIMD_REAL_WIDTH) - jy_S;
                d[i][2] = load<SimdReal>(x_ci_simd + (i * 3 + 2) * GMX_SIMD_REAL_WIDTH) - jz_S;
            }

            /* rsq = dx*dx+dy*dy+dz*dz */
            std::array<SimdReal, iClusterSize> rsq;
            for (int i = 0; i < iClusterSize; i++)
            {
                rsq[i] = norm2(d[i][0], d[i][1], d[i][2]);
            }

            std::array<SimdBool, iClusterSize> wco;
            for (int i = 0; i < iClusterSize; i++)
            {
                wco[i] = (rsq[i] < rc2_S);
            }

            const int numBitShifts = log2ClusterSize<iClusterSize>();
            for (int bitShift = 0; bitShift < numBitShifts; bitShift++)
            {
                const int offset = (1 << bitShift);
                for (int i = 0; i < iClusterSize; i += 2 * offset)
                {
                    wco[i] = wco[i] || wco[i + offset];
                }
            }

            InRange = anyTrue(wco[0]);

            *numDistanceChecks += iClusterSize * GMX_SIMD_REAL_WIDTH;
        }
        if (!InRange)
        {
            jclusterFirst++;
        }
    }
    if (!InRange)
    {
        return;
    }

    InRange = false;
    while (!InRange && jclusterLast > jclusterFirst)
    {
        const float d2 = clusterBoundingBoxDistance2(bb_ci[0], jGrid.jBoundingBoxes()[jclusterLast]);
        *numDistanceChecks += 2;

        /* Check if the distance is within the distance where
         * we use only the bounding box distance rbb,
         * or within the cut-off and there is at least one atom pair
         * within the cut-off.
         */
        if (d2 < rbb2)
        {
            InRange = true;
        }
        else if (d2 < rlist2)
        {
            const int xind_l = xIndexFromCj<kernelLayout>(
                    cjFromCi<kernelLayout, 0>(jGrid.cellOffset()) + jclusterLast);

            const SimdReal jx_S = load<SimdReal>(x_j + xind_l + 0 * c_xStride4xN<kernelLayout>());
            const SimdReal jy_S = load<SimdReal>(x_j + xind_l + 1 * c_xStride4xN<kernelLayout>());
            const SimdReal jz_S = load<SimdReal>(x_j + xind_l + 2 * c_xStride4xN<kernelLayout>());

            /* Calculate distance */
            std::array<std::array<SimdReal, 3>, iClusterSize> d;
            for (int i = 0; i < iClusterSize; i++)
            {
                d[i][0] = load<SimdReal>(x_ci_simd + (i * 3 + 0) * GMX_SIMD_REAL_WIDTH) - jx_S;
                d[i][1] = load<SimdReal>(x_ci_simd + (i * 3 + 1) * GMX_SIMD_REAL_WIDTH) - jy_S;
                d[i][2] = load<SimdReal>(x_ci_simd + (i * 3 + 2) * GMX_SIMD_REAL_WIDTH) - jz_S;
            }

            /* rsq = dx*dx+dy*dy+dz*dz */
            std::array<SimdReal, iClusterSize> rsq;
            for (int i = 0; i < iClusterSize; i++)
            {
                rsq[i] = norm2(d[i][0], d[i][1], d[i][2]);
            }

            std::array<SimdBool, iClusterSize> wco;
            for (int i = 0; i < iClusterSize; i++)
            {
                wco[i] = (rsq[i] < rc2_S);
            }

            const int numBitShifts = log2ClusterSize<iClusterSize>() + 1;
            for (int bitShift = 1; bitShift < numBitShifts; bitShift++)
            {
                const int offset = (1 << bitShift);
                for (int i = 0; i < iClusterSize; i += 2 * offset)
                {
                    wco[i] = wco[i] || wco[i + offset];
                }
            }

            InRange = anyTrue(wco[0]);

            *numDistanceChecks += iClusterSize * GMX_SIMD_REAL_WIDTH;
        }
        if (!InRange)
        {
            jclusterLast--;
        }
    }

    if (jclusterFirst <= jclusterLast)
    {
        for (int jcluster = jclusterFirst; jcluster <= jclusterLast; jcluster++)
        {
            /* Store cj and the interaction mask */
            const int           cj = cjFromCi<kernelLayout, 0>(jGrid.cellOffset()) + jcluster;
            JClusterList::IMask excl;
#if GMX_HAVE_NBNXM_SIMD_4XM
            if constexpr (kernelLayout == KernelLayout::r4xM)
            {
                excl = get_imask_simd_4xn(excludeSubDiagonal, icluster, jcluster);
            }
#if GMX_HAVE_NBNXM_SIMD_8XM
            else
            {
                excl = get_imask_simd_8xn(excludeSubDiagonal, icluster, jcluster);
            }
#endif
#endif
#if GMX_HAVE_NBNXM_SIMD_2XM
            if constexpr (kernelLayout == KernelLayout::r2xM)
            {
                excl = get_imask_simd_2xn(excludeSubDiagonal, icluster, jcluster);
            }
#endif

            nbl->cj.push_back(cj, excl);
        }
        /* Increase the closing index in the i list */
        nbl->ci.back().cj_ind_end = nbl->cj.size();
    }
}
