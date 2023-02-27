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
 *  CUDA non-bonded prune-only kernel.
 *
 *  Unlike the non-bonded interaction kernels, this is not preprocessor-generated,
 *  the two flavors achieved by templating.
 *
 *  \author Szilárd Páll <pall.szilard@gmail.com>
 *  \author Berk Hess <hess@kth.se>
 *  \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/math/utilities.h"
#include "gromacs/pbcutil/ishift.h"

#include "nbnxm_cuda_kernel_utils.cuh"
#include "nbnxm_cuda_types.h"

/* Note that floating-point constants in CUDA code should be suffixed
 * with f (e.g. 0.5f), to stop the compiler producing intermediate
 * code that is in double precision.
 */

/**@{*/
/*! \brief Compute capability dependent definition of kernel launch configuration parameters.
 *
 * Kernel launch bounds for different compute capabilities. The value of NTHREAD_Z
 * represents the j-concurrency, hence it determines the number of threads per block.
 * It is chosen such that 100% occupancy is maintained (on Maxwell and later for any NTHREAD_Z,
 * requires >=4 warp/block, NTHREAD_Z>=2 on Kepler).
 *
 * Hence, values NTHREAD_Z >= 2 trade inter- for intra-block parallelism
 * which has the advantage of lowering the overhead of starting up a block, filling shmem
 * and registers, etc. Ideally we'd want to expose as much intra-block work as possible
 * As we also split lists to cater for the block-parallelization needed by the register-
 * limited non-bonded kernels, for very short j-loops large NTHREAD_Z will cause slowdown
 * as it leads to intra-block warp imbalance. Ideally, we'd want to auto-tune the choice
 * of NTHREAD_Z, but for now we instead pick a reasonable tradeoff-value.
 *
 * Note that given the above input size tradeoffs and that performance depends on
 * additional factors including GPU arch, #SM's, we'll accept performance tradeoffs
 * of using a fixed NTHREAD_Z=4. The following outliers have been observed:
 *   - up to 25% faster (rolling) prune kernels with NTHREAD_Z=8 in the regime where lists
 *     are not split (much), but the rolling chunks are small;
 *   - with large inputs NTHREAD_Z=1 is 2-3% faster (on CC>=5.0)
 */
#define NTHREAD_Z (GMX_NBNXN_PRUNE_KERNEL_J4_CONCURRENCY)
#define THREADS_PER_BLOCK (c_clSize * c_clSize/2 * NTHREAD_Z)
#define MIN_BLOCKS_PER_MP (GMX_CUDA_MAX_THREADS_PER_MP / THREADS_PER_BLOCK)
/**@}*/


__forceinline__ __device__ float convert_f32_to_tf32(float f32_input){
    float tf32_output;
    asm ("{.reg .b32 %mr;\n"
        "cvt.rna.tf32.f32 %mr, %1;\n"
        "mov.b32 %0, %mr;}\n" : "=f"(tf32_output) : "f"(f32_input));
    return tf32_output;
}
/*! \brief Nonbonded list pruning kernel.
 *
 *  The \p haveFreshList template parameter defines the two flavors of the kernel; when
 *  true a new list from immediately after pair-list generation is pruned using rlistOuter,
 *  the pruned masks are stored in a separate buffer and the outer-list is pruned
 *  using the rlistInner distance; when false only the pruning with rlistInner is performed.
 *
 *  Kernel launch parameters:
 *   - #blocks   = #pair lists, blockId = pair list Id
 *   - #threads  = NTHREAD_Z * c_clSize^2
 *   - shmem     = see nbnxn_cuda.cu:calc_shmem_required_prune()
 *
 *   Each thread calculates an i-j atom distance..
 */
template<bool haveFreshList>
__launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP) __global__
        void nbnxn_kernel_prune_cuda(NBAtomDataGpu    atdat,
                                     NBParamGpu       nbparam,
                                     Nbnxm::gpu_plist plist,
                                     int              numParts,
                                     int              part)
#ifdef FUNCTION_DECLARATION_ONLY
                ; /* Only do function declaration, omit the function body. */

// Add extern declarations so each translation unit understands that
// there will be a definition provided.
extern template __global__ void
nbnxn_kernel_prune_cuda<true>(const NBAtomDataGpu, const NBParamGpu, const Nbnxm::gpu_plist, int, int);
extern template __global__ void
nbnxn_kernel_prune_cuda<false>(const NBAtomDataGpu, const NBParamGpu, const Nbnxm::gpu_plist, int, int);
#else
{

    /* convenience variables */
    const nbnxn_sci_t* pl_sci    = plist.sci;
    nbnxn_cj4_t*       pl_cj4    = plist.cj4;
    const float4*      xq        = atdat.xq;
    const float3*      shift_vec = asFloat3(atdat.shiftVec);

    float rlistOuter_sq = nbparam.rlistOuter_sq;
    float rlistInner_sq = nbparam.rlistInner_sq;

    /* thread/block/warp id-s */
    unsigned int tidxi = threadIdx.y;
    unsigned int tidxj = threadIdx.x;
#    if NTHREAD_Z == 1
    unsigned int tidxz = 0;
#    else
    unsigned int tidxz = threadIdx.z;
#    endif
    unsigned int bidx  = blockIdx.x;
    unsigned int widx  = (threadIdx.y * c_clSize) / warp_size; /* warp index */

    // cj preload is off in the following cases:
    // - sm_70 (V100), sm_8x (A100, GA100), sm_75 (TU102)
    // - for future arch (> 8.6 at the time of writing) we assume it is better to keep it off
    constexpr bool c_preloadCj = (GMX_PTX_ARCH < 700);

    /*********************************************************************
     * Set up shared memory pointers.
     * sm_nextSlotPtr should always be updated to point to the "next slot",
     * that is past the last point where data has been stored.
     */
    extern __shared__ char sm_dynamicShmem[];
    char*                  sm_nextSlotPtr = sm_dynamicShmem;
    static_assert(sizeof(char) == 1,
                  "The shared memory offset calculation assumes that char is 1 byte");

    /* shmem buffer for i x+q pre-loading */
    float4* xib = reinterpret_cast<float4*>(sm_nextSlotPtr);
    sm_nextSlotPtr += (c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(*xib));

    /* shmem buffer for cj, for each warp separately */
    int* cjs = reinterpret_cast<int*>(sm_nextSlotPtr);
    if (c_preloadCj)
    {
        /* the cjs buffer's use expects a base pointer offset for pairs of warps in the j-concurrent execution */
        cjs += tidxz * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize;
        sm_nextSlotPtr += (NTHREAD_Z * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize * sizeof(*cjs));
    }
    /*********************************************************************/


    nbnxn_sci_t nb_sci =
            pl_sci[bidx * numParts + part]; /* my i super-cluster's index = sciOffset + current bidx * numParts + part */
    int sci        = nb_sci.sci;           /* super-cluster */
    int cij4_start = nb_sci.cj4_ind_start; /* first ...*/
    int cij4_end   = nb_sci.cj4_ind_end;   /* and last index of j clusters */

    // We may need only a subset of threads active for preloading i-atoms
    // depending on the super-cluster and cluster / thread-block size.
    constexpr bool c_loadUsingAllXYThreads = (c_clSize == c_nbnxnGpuNumClusterPerSupercluster);
    if (tidxz == 0 && (c_loadUsingAllXYThreads || tidxi < c_nbnxnGpuNumClusterPerSupercluster))
    {
        /* Pre-load i-atom x and q into shared memory */
        int ci = sci * c_nbnxnGpuNumClusterPerSupercluster + tidxi*2;
        int ai = ci * c_clSize + tidxj;

        /* We don't need q, but using float4 in shmem avoids bank conflicts.
           (but it also wastes L2 bandwidth). */
        float4 tmp                    = xq[ai];
        float4 xi                     = tmp + shift_vec[nb_sci.shift];
        float  normi = norm2(make_float3(xi.x, xi.y, xi.z));
        // use last element in float4 to store norm, as we're reading the full float4 anyway
        xi.w = normi;
        // (a-b)^2 = a^2 + b^2 - 2ab -- do the multiplication by -2 here
        xi.x = -2*xi.x;
        xi.y = -2*xi.y;
        xi.z = -2*xi.z;
        xib[tidxi*2 * c_clSize + tidxj] = xi;

        // as we have 8x4 threads with tidxz==0, have each thread load two values to get up to 64 points
        int ci2 = sci * c_nbnxnGpuNumClusterPerSupercluster + tidxi*2+1;
        ai = ci2 * c_clSize + tidxj;

        /* We don't need q, but using float4 in shmem avoids bank conflicts.
           (but it also wastes L2 bandwidth). */
        tmp                    = xq[ai];
        xi                     = tmp + shift_vec[nb_sci.shift];
        normi = norm2(make_float3(xi.x, xi.y, xi.z));
        // use last element in float4 to store norm, as we're reading the full float4 anyway
        xi.w = normi;
        // (a-b)^2 = a^2 + b^2 - 2ab -- do the multiplication by -2 here
        xi.x = -2*xi.x;
        xi.y = -2*xi.y;
        xi.z = -2*xi.z;
        xib[(tidxi*2+1) * c_clSize + tidxj] = xi;

    }
    __syncthreads();

    /* loop over the j clusters = seen by any of the atoms in the current super-cluster;
     * The loop stride NTHREAD_Z ensures that consecutive warps-pairs are assigned
     * consecutive j4's entries.
     */
    for (int j4 = cij4_start + tidxz; j4 < cij4_end; j4 += NTHREAD_Z)
    {
        unsigned int imaskFull, imaskCheck, imaskNew;
        unsigned int imaskFull_firstJs, imaskCheck_firstJs, imaskNew_firstJs;
        unsigned int imaskFull_secondJs, imaskCheck_secondJs, imaskNew_secondJs;

        if (haveFreshList)
        {
            /* Read the mask from the list transferred from the CPU */
            imaskFull_firstJs = pl_cj4[j4].imei[0].imask;
            imaskFull_secondJs = pl_cj4[j4].imei[1].imask;
            /* We attempt to prune all pairs present in the original list */
            imaskCheck_firstJs = imaskFull_firstJs;
            imaskCheck_secondJs = imaskFull_secondJs;
            imaskNew_firstJs   = 0;
            imaskNew_secondJs   = 0;
        }
        else
        {
            /* Read the mask from the "warp-pruned" by rlistOuter mask array */
            // each warp now handles 8 j values, so is responsible for both masks
            // TODO fix hard code
            imaskFull_firstJs = plist.imask[j4 * c_nbnxnGpuClusterpairSplit + 0];
            imaskFull_secondJs = plist.imask[j4 * c_nbnxnGpuClusterpairSplit + 1];
            /* Read the old rolling pruned mask, use as a base for new */
            imaskNew_firstJs = pl_cj4[j4].imei[0].imask;
            imaskNew_secondJs = pl_cj4[j4].imei[1].imask;
            /* We only need to check pairs with different mask */
            imaskCheck_firstJs = (imaskNew_firstJs ^ imaskFull_firstJs);
            imaskCheck_secondJs = (imaskNew_secondJs ^ imaskFull_secondJs);
        }

        if (imaskCheck_firstJs || imaskCheck_secondJs)
        {
            if (c_preloadCj)
            {
                /* Pre-load cj into shared memory on both warps separately */
                if ((tidxj == 0 || tidxj == 4) && tidxi < c_nbnxnGpuJgroupSize)
                {
                    cjs[tidxi + tidxj * c_nbnxnGpuJgroupSize / c_splitClSize] = pl_cj4[j4].cj[tidxi];
                }
                __syncwarp(c_fullWarpMask);
            }
#    pragma unroll c_nbnxnGpuJgroupSize
            for (int jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
            {
                if ((imaskCheck_firstJs & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster)))
                ||(imaskCheck_secondJs & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster))))
                {
                    unsigned int mask_ji = (1U << (jm * c_nbnxnGpuNumClusterPerSupercluster));
                    int cj = c_preloadCj ? cjs[jm + (tidxj & 4) * c_nbnxnGpuJgroupSize / c_splitClSize]
                                         : pl_cj4[j4].cj[jm];
                    int aj = cj * c_clSize + tidxj;

                    /* load j atom data */
                    float4 tmp = xq[aj];
                    float3 xj  = make_float3(tmp.x, tmp.y, tmp.z);
                    float  normj = tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z;
#    pragma unroll c_nbnxnGpuNumClusterPerSupercluster
                    for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i+=2)
                    {
                        if ((imaskCheck_firstJs & mask_ji)
                        || (imaskCheck_secondJs & (mask_ji+mask_ji)))
                        {
                            /* distance between i and j atoms */
// Tensor core implementation
#if GMX_PTX_ARCH >= 800
                            int lid = threadIdx.x + threadIdx.y*blockDim.x;
                            // load i values using entire warp
                            int a0_index = lid >> 2; 
                            // first 8 i values
                            float4 xi1 = xib[i * c_clSize + a0_index];
                            // second 8 i values
                            float4 xi2 = xib[(i+1) * c_clSize + a0_index];
        
                            float normi1 = xi1.w;
                            float normi2 = xi2.w;

                            // vars for tf32 m16n8k4 instruction -- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1684
                            // A stores 16x3 i values
                            // B stores 8x3 j values
                            // D stores 16x8 ixj distances
                            // C stores 16x8 normi + normj values
                            float af0=0, af1=0;
                            float bf0=0;
                            float c0=0, c1=0, c2=0, c3=0;
                            float d0=0, d1=0, d2=0, d3=0;

                            // choose xi component based on matrix B k value
                            af0 = xi1.x*((lid&3)==0) + xi1.y*((lid&3)==1) + xi1.z*((lid&3)==2); // lid%4==3 (k=3) is 0
                            af1 = xi2.x*((lid&3)==0) + xi2.y*((lid&3)==1) + xi2.z*((lid&3)==2); // lid%4==3 (k=3) is 0

                            // choose xj component based on matrix A k value
                            float shfl_xvar_j = xj.x*(threadIdx.y==0) + xj.y*(threadIdx.y==1) + xj.z*(threadIdx.y==2);
                            bf0 = __shfl_sync(0xffffffff, shfl_xvar_j, lid>>2 + 8*(lid&3), 32);

                            // we need to add normi + normj to the MMA:
                            // (a-b)^2 = a^2 + b^2 - 2ab
                            float normj_c0_contrib = __shfl_sync(0xffffffff, normj, lid*2, 32);
                            float normj_c1_contrib = __shfl_sync(0xffffffff, normj, lid*2+1, 32);
                            float normj_c2_contrib = normj_c0_contrib; 
                            float normj_c3_contrib = normj_c1_contrib;

                            c0 = normj_c0_contrib + normi1;
                            c1 = normj_c1_contrib + normi1;
                            c2 = normj_c2_contrib + normi2;
                            c3 = normj_c3_contrib + normi2;

                            // 3xTF32 needed to reach FP32 accuracy:
                            // a x b = (a_big + a_small) x (b_big + b_small) = a_big x b_big + a_big x b_small + a_small x b_big
                            // big = fp32_to_tf32(fp32)
                            // small = fp32_to_tf32(fp32-big)

                            // uint32_t format required for a, b -- see https://github.com/NVIDIA/cutlass/blob/master/include/cutlass/arch/mma_sm80.h#L179
                            float af0_big = convert_f32_to_tf32(af0);
                            uint32_t *a0_big = reinterpret_cast<uint32_t *>(&af0_big);
                            float af1_big = convert_f32_to_tf32(af1);
                            uint32_t *a1_big = reinterpret_cast<uint32_t *>(&af1_big);
                            float bf0_big = convert_f32_to_tf32(bf0);
                            uint32_t *b0_big = reinterpret_cast<uint32_t *>(&bf0_big);

                            float af0_small = af0 - af0_big;
                            uint32_t *a0_small = reinterpret_cast<uint32_t *>(&af0_small);
                            float af1_small = af1 - af1_big;
                            uint32_t *a1_small = reinterpret_cast<uint32_t *>(&af1_small);
                            float bf0_small = bf0 - bf0_big;
                            uint32_t *b0_small = reinterpret_cast<uint32_t *>(&bf0_small);

                            // do MMA x 3
                            asm volatile(
                            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                            : 
                              "r"(*a0_big), "r"(*a1_big), 
                              "r"(*b0_big), 
                              "f"(c0), "f"(c1), "f"(c2), "f"(c3)
                            ); 
                            asm volatile(
                            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                            : 
                              "r"(*a0_big), "r"(*a1_big), 
                              "r"(*b0_small), 
                              "f"(d0), "f"(d1), "f"(d2), "f"(d3)
                            ); 
                            asm volatile(
                            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                            : 
                              "r"(*a0_small), "r"(*a1_small), 
                              "r"(*b0_big), 
                              "f"(d0), "f"(d1), "f"(d2), "f"(d3)
                            ); 

                            int j0To3Mask = 0x33333333;
                            int j4To7Mask = 0xCCCCCCCC;

                            int anyInOuterRange1_firstJs, anyInOuterRange2_firstJs, 
                                anyInInnerRange1_firstJs, anyInInnerRange2_firstJs, 
                                anyInOuterRange1_secondJs, anyInOuterRange2_secondJs, 
                                anyInInnerRange1_secondJs, anyInInnerRange2_secondJs;

                        // TODO tried masking out unneeded ops here, branching appears to lead to worse performance
                        // investigate why
                        //if (imaskCheck_firstJs & mask_ji){
                             anyInOuterRange1_firstJs = __any_sync(j0To3Mask, d0 < rlistOuter_sq) | 
                                        __any_sync(j0To3Mask, d1 < rlistOuter_sq);
                             anyInOuterRange2_firstJs = __any_sync(j0To3Mask, d2 < rlistOuter_sq) | 
                                        __any_sync(j0To3Mask, d3 < rlistOuter_sq);

                             anyInInnerRange1_firstJs = __any_sync(j0To3Mask, d0 < rlistInner_sq) | 
                                        __any_sync(j0To3Mask, d1 < rlistInner_sq);
                             anyInInnerRange2_firstJs = __any_sync(j0To3Mask, d2 < rlistInner_sq) | 
                                        __any_sync(j0To3Mask, d3 < rlistInner_sq);
                        //}

                        // TODO tried masking out unneeded ops here, branching appears to lead to worse performance
                        // investigate why
                        //if (imaskCheck_secondJs & (mask_ji+mask_ji)){
                             anyInOuterRange1_secondJs = __any_sync(j4To7Mask, d0 < rlistOuter_sq) | 
                                        __any_sync(j0To3Mask, d1 < rlistOuter_sq);
                             anyInOuterRange2_secondJs = __any_sync(j4To7Mask, d2 < rlistOuter_sq) | 
                                        __any_sync(j4To7Mask, d3 < rlistOuter_sq);

                             anyInInnerRange1_secondJs = __any_sync(j4To7Mask, d0 < rlistInner_sq) | 
                                        __any_sync(j4To7Mask, d1 < rlistInner_sq);
                             anyInInnerRange2_secondJs = __any_sync(j4To7Mask, d2 < rlistInner_sq) | 
                                        __any_sync(j4To7Mask, d3 < rlistInner_sq);
                        //}

#else
                            // Version that does not use tensor cores but does pack 16 i, 8 j into one warp
                            // for comparing data only; not optimised
                            float4 xi1_front4 = xib[i * c_clSize + tidxi];
                            float4 xi1_back4 = xib[i * c_clSize + tidxi + c_clSize/2];
                            float4 xi2_front4 = xib[(i+1) * c_clSize + tidxi];
                            float4 xi2_back4 = xib[(i+1) * c_clSize + tidxi + c_clSize/2];
                            float3 rv1_front4 = make_float3(xi1_front4.x/-2, xi1_front4.y/-2, xi1_front4.z/-2) - xj;
                            float3 rv1_back4 = make_float3(xi1_back4.x/-2, xi1_back4.y/-2, xi1_back4.z/-2) - xj;
                            float3 rv2_front4 = make_float3(xi2_front4.x/-2, xi2_front4.y/-2, xi2_front4.z/-2) - xj;
                            float3 rv2_back4 = make_float3(xi2_back4.x/-2, xi2_back4.y/-2, xi2_back4.z/-2) - xj;
                            float  r21_front4 = norm2(rv1_front4);
                            float  r21_back4 = norm2(rv1_back4);
                            float  r22_front4 = norm2(rv2_front4);
                            float  r22_back4 = norm2(rv2_back4);

                            int j0To3Mask = 0x0F0F0F0F;
                            int j4To7Mask = 0xF0F0F0F0;
                            int anyInOuterRange1_firstJs = __any_sync(j0To3Mask, r21_front4 < rlistOuter_sq) | __any_sync(j0To3Mask, r21_back4 < rlistOuter_sq);
                            int anyInInnerRange1_firstJs = __any_sync(j0To3Mask, r21_front4 < rlistInner_sq) | __any_sync(j0To3Mask, r21_back4 < rlistInner_sq);
                            int anyInOuterRange2_firstJs = __any_sync(j0To3Mask, r22_front4 < rlistOuter_sq) | __any_sync(j0To3Mask, r22_back4 < rlistOuter_sq);
                            int anyInInnerRange2_firstJs = __any_sync(j0To3Mask, r22_front4 < rlistInner_sq) | __any_sync(j0To3Mask, r22_back4 < rlistInner_sq);
                            int anyInOuterRange1_secondJs = __any_sync(j4To7Mask, r21_front4 < rlistOuter_sq) | __any_sync(j4To7Mask, r21_back4 < rlistOuter_sq);
                            int anyInInnerRange1_secondJs = __any_sync(j4To7Mask, r21_front4 < rlistInner_sq) | __any_sync(j4To7Mask, r21_back4 < rlistInner_sq);
                            int anyInOuterRange2_secondJs = __any_sync(j4To7Mask, r22_front4 < rlistOuter_sq) | __any_sync(j4To7Mask, r22_back4 < rlistOuter_sq);
                            int anyInInnerRange2_secondJs = __any_sync(j4To7Mask, r22_front4 < rlistInner_sq) | __any_sync(j4To7Mask, r22_back4 < rlistInner_sq);

#endif

                        // TODO tried masking out unneeded ops here, branching appears to lead to worse performance
                        // investigate why
                        //if (imaskCheck_firstJs & mask_ji){
                            /* If _none_ of the atoms pairs are in rlistOuter
                               range, the bit corresponding to the current
                               cluster-pair in imask gets set to 0. */
                            if (haveFreshList && !anyInOuterRange1_firstJs) imaskFull_firstJs &= ~mask_ji;
                            /* If any atom pair is within range, set the bit
                               corresponding to the current cluster-pair. */
                            if (anyInInnerRange1_firstJs) imaskNew_firstJs |= mask_ji;
                            /* If _none_ of the atoms pairs are in rlistOuter
                               range, the bit corresponding to the current
                               cluster-pair in imask gets set to 0. */
                            if (haveFreshList && !anyInOuterRange2_firstJs) imaskFull_firstJs &= ~(mask_ji+mask_ji);
                            /* If any atom pair is within range, set the bit
                               corresponding to the current cluster-pair. */
                            if (anyInInnerRange2_firstJs) imaskNew_firstJs |= (mask_ji);
                        //}

                        // TODO tried masking out unneeded ops here, branching appears to lead to worse performance
                        // investigate why
                        //if (imaskCheck_secondJs & (mask_ji+mask_ji)){
                            if (haveFreshList && !anyInOuterRange1_secondJs) imaskFull_secondJs &= ~mask_ji;
                            /* If any atom pair is within range, set the bit
                               corresponding to the current cluster-pair. */
                            if (anyInInnerRange1_secondJs) imaskNew_secondJs |= mask_ji;

                            if (haveFreshList && !anyInOuterRange2_secondJs) imaskFull_secondJs &= ~(mask_ji+mask_ji);
                            // TODO calculating definitely incorect mask which somehow prevents a mem error and crash 
                            // later in the program, for profiling the existing code only. This suggests there is 
                            // still a bug somewhere in this code
                            // the mask that should be correct is commented out below
                            if (anyInInnerRange2_secondJs) imaskNew_secondJs |= (mask_ji);
                            // if (anyInInnerRange2_secondJs) imaskNew_secondJs |= (mask_ji+mask_ji); 
                        //}
                        } // END if mask

                        // mask for next i cluster in supercluster
                        /* shift the mask bit by 2 */
                        mask_ji = mask_ji >> 2;
                    } // END LOOP OVER i
                } // END if mask
            } // END LOOP OVER jm

            if (haveFreshList)
            {
                /* copy the list pruned to rlistOuter to a separate buffer */
                plist.imask[j4 * c_nbnxnGpuClusterpairSplit + 0] = imaskFull_firstJs;
                plist.imask[j4 * c_nbnxnGpuClusterpairSplit + 1] = imaskFull_secondJs;
            }
            /* update the imask with only the pairs up to rlistInner */
            plist.cj4[j4].imei[0].imask = imaskNew_firstJs;
            plist.cj4[j4].imei[1].imask = imaskNew_secondJs;

        } // END if mask
        if (c_preloadCj)
        {
            // avoid shared memory WAR hazards on sm_cjs between loop iterations
            __syncwarp(c_fullWarpMask);
        }
    } // END LOOP OVER j4
}
#endif /* FUNCTION_DECLARATION_ONLY */

#undef NTHREAD_Z
#undef MIN_BLOCKS_PER_MP
#undef THREADS_PER_BLOCK
