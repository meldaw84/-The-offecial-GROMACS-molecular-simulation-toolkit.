/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2016- The GROMACS Authors
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
 *  CUDA bucket sci sort kernel.
 *
 *  \author Zhengru Wang <zhengruw@nvidia.com>
 *  \author Ania Brown <anbrown@nvidia.com>
 *  \ingroup module_nbnxm
 */


/*! \brief CUDA bucket sci sort kernel.
 *
 *  Sorts sci in order from most to least neighbours, using the count sort algorithm
 *
 *  Unlike the cpu version of sci sort, this kernel uses counts which only contain pairs which have
 *  not been masked out, giving an ordering which more accurately represents the work which will be
 *  done in the non bonded force kernel. The counts themselves are generated in the prune kernel.
 *
 *  Inputs:
 *   - plist.sci = unsorted pair list
 *   - plist.sorting.sciCount[i] = total number of sci with exactly i neighbours
 *   - plist.sorting.sciOffset = exclusive prefix sum of sciCount.
 *     plist.sorting.sciOffset[i] represents the offset that the first sci with i neighbours will
 *     have in the sorted sci list. All other sci with i neighbours will be placed randomly in
 * positions plist.sorting.sciOffset[i] to plist.sorting.sciOffset[i+1] exclusive
 */

#ifndef NBNXM_CUDA_KERNEL_SCI_SORT_CUH
#define NBNXM_CUDA_KERNEL_SCI_SORT_CUH

__launch_bounds__(c_sciSortingThreadsPerBlock) __global__
        void nbnxnKernelBucketSciSort(Nbnxm::gpu_plist plist)
{
    int size = plist.nsci;

    const unsigned int flat_id = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * c_sciSortingThreadsPerBlock * c_sciSortingItemsPerThread;

    const nbnxn_sci_t* pl_sci        = plist.sci;
    nbnxn_sci_t*       pl_sci_sort   = plist.sorting.sciSorted;
    const int*         pl_sci_count  = plist.sorting.sciCount;
    int*               pl_sci_offset = plist.sorting.sciOffset;

    int         sci_count[c_sciSortingItemsPerThread];
    int         sci_offset[c_sciSortingItemsPerThread];
    nbnxn_sci_t sci[c_sciSortingItemsPerThread];

#pragma unroll
    for (unsigned int i = 0; i < c_sciSortingItemsPerThread; i++)
    {
        if (size > (block_offset + c_sciSortingItemsPerThread * flat_id + i))
        {
            // Copy plist to intermediate registers for clarity
            sci[i]       = pl_sci[block_offset + c_sciSortingItemsPerThread * flat_id + i];
            sci_count[i] = pl_sci_count[block_offset + c_sciSortingItemsPerThread * flat_id + i];
        }
    }

#pragma unroll
    for (unsigned int i = 0; i < c_sciSortingItemsPerThread; i++)
    {
        if (size > (block_offset + c_sciSortingItemsPerThread * flat_id + i))
        {
            // Choose an index for each sci with exactly sci_count[i] neighbours, using
            // offsets from pl_sci_offset[count] to pl_sci_offset[count+1] exclusive.
            // As the amount of work for each of these in the non bonded force kernel will
            // be equivalent, we don't care what order these are placed in.
            sci_offset[i] = atomicAdd(&pl_sci_offset[sci_count[i]], 1);
            // Insert the sci into the sorted list at the chosen index
            pl_sci_sort[sci_offset[i]] = sci[i];
        }
    }
}

#endif /* NBNXM_CUDA_KERNEL_SCI_SORT_CUH */
