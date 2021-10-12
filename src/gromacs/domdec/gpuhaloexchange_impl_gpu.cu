/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2019- The GROMACS Authors
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
 * \brief Implements backend-specific part of GPU halo exchange, namely pack and unpack
 * kernels and the host code for scheduling them, using CUDA.
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_domdec
 */
#include "gmxpre.h"

#include "gpuhaloexchange_impl_gpu.h"

#include "config.h"

#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/gpu_utils/vectype_ops.cuh"

#include "domdec_struct.h"

#if GMX_NVSHMEM
#include "nvshmem.h"
#include "nvshmemx.h"
#include <nvshmem_api.h>
#include <omp.h>
#include "gromacs/gpu_utils/nvshmembuffer.cuh"
#include <cuda/barrier>
#endif

namespace gmx
{

//! Number of CUDA threads in a block
// TODO Optimize this through experimentation
constexpr static int c_threadsPerBlock = 256;

template<bool usePBC>
__global__ void packSendBufKernel(float3* __restrict__ dataPacked,
                                  const float3* __restrict__ data,
                                  const int* __restrict__ map,
                                  const int    mapSize,
                                  const float3 coordinateShift)
{
    int           threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float3*       gm_dataDest = &dataPacked[threadIndex];
    const float3* gm_dataSrc  = &data[map[threadIndex]];

    if (threadIndex < mapSize)
    {
        if (usePBC)
        {
            *gm_dataDest = *gm_dataSrc + coordinateShift;
        }
        else
        {
            *gm_dataDest = *gm_dataSrc;
        }
    }
}

/*! \brief unpack non-local force data buffer on the GPU using pre-populated "map" containing index
 * information
 * \param[out] data        full array of force values
 * \param[in]  dataPacked  packed array of force values to be transferred
 * \param[in]  map         array of indices defining mapping from full to packed array
 * \param[in]  mapSize     number of elements in map array
 */
template<bool accumulate>
__global__ void unpackRecvBufKernel(float3* __restrict__ data,
                                    const float3* __restrict__ dataPacked,
                                    const int* __restrict__ map,
                                    const int mapSize)
{

    int           threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const float3* gm_dataSrc  = &dataPacked[threadIndex];
    float3*       gm_dataDest = &data[map[threadIndex]];

    if (threadIndex < mapSize)
    {
        if (accumulate)
        {
            *gm_dataDest += *gm_dataSrc;
        }
        else
        {
            *gm_dataDest = *gm_dataSrc;
        }
    }
}

#if GMX_NVSHMEM
template<bool usePBC>
__global__ void packSendMultiBlockNvshmemBufKernel(const float3* data,
                                  float3* __restrict__ dataPacked,
                                  const int* map,
                                  const int    mapSize,
                                  const float3 coordinateShift,
                                  const int sendRank,
                                  const int remoteAtomOffset,
                                  uint64_t *sync_arr,
                                  uint64_t *sync_recv_arr,
                                  const int recvRank,
                                  uint64_t counter,
                                  cuda::barrier<cuda::thread_scope_device> *bar,
                                  int pulse,
                                  int npulses,
                                  int dimIndex,
                                  int ndim)
{
    uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    using barrier = cuda::barrier<cuda::thread_scope_device>;
    if (threadIdx.x == 0 && ((pulse > 0) || (dimIndex > 0))) {
        /* Wait for neighbors notification */
        nvshmem_signal_wait_until(sync_arr - 1, NVSHMEM_CMP_EQ, counter);
    }

    /* Notify neighbors about arrival */
    if (threadIndex == 0) {
        nvshmemx_signal_op(sync_recv_arr, counter, NVSHMEM_SIGNAL_SET, recvRank);
    }

    const uint32_t gridSize = blockDim.x * gridDim.x;

    uint32_t gid = threadIndex;
    const float3* gm_dataSrc = nullptr;
    float3* gm_dataDest = nullptr;
    if (gid < mapSize) {
        int atomIndex = map[gid];
        gm_dataSrc  = &data[atomIndex];
        gm_dataDest = &dataPacked[gid];
    }
    if ((pulse > 0)  || (dimIndex > 0)) {
        __syncthreads();
    }
    if (gid < mapSize) {
        if (usePBC) {
            *gm_dataDest = *gm_dataSrc + coordinateShift;
        }
        else {
            *gm_dataDest = *gm_dataSrc;
        }
    }
    for (uint32_t idx = gid + gridSize; idx < mapSize; idx += gridSize)
    {
        int atomIndex = map[idx];
        gm_dataSrc  = &data[atomIndex];
        gm_dataDest = &dataPacked[idx];
        if (usePBC) {
            *gm_dataDest = *gm_dataSrc + coordinateShift;
        }
        else {
            *gm_dataDest = *gm_dataSrc;
        }
    }
    __syncthreads();

    // More study is needed to understand why more than 1 block nvshmem_put performs bad
    constexpr size_t numFinalBlocks = 1;
    barrier::arrival_token token;
    if (threadIdx.x == 0) {
        token = bar->arrive();
    }

    if (blockIdx.x < numFinalBlocks) {
        size_t const chunkSize = mapSize / numFinalBlocks;
        size_t blockOffset = blockIdx.x * chunkSize;
        size_t dataSize = blockIdx.x != (numFinalBlocks - 1) ? chunkSize : max(mapSize - blockOffset, chunkSize);

        float3 *recvPtr = (float3*)&data[remoteAtomOffset +  blockOffset];
        if (threadIdx.x == 0) {
            bar->wait(std::move(token));
            nvshmem_signal_wait_until(sync_recv_arr, NVSHMEM_CMP_EQ, counter);
        }
        __syncthreads();

        nvshmemx_float_put_signal_nbi_block((float*)recvPtr, (float*)&dataPacked[blockOffset], dataSize * 3, sync_arr, counter, NVSHMEM_SIGNAL_SET, sendRank);

        if (threadIdx.x == 0 && ((pulse == npulses - 1) && (dimIndex == ndim - 1))) {
            nvshmem_signal_wait_until(sync_arr, NVSHMEM_CMP_EQ, counter);
        }
    }
}


/*! \brief unpack non-local force data buffer on the GPU using pre-populated "map" containing index
 * information
 * \param[out] data        full array of force values
 * \param[in]  dataPacked  packed array of force values to be transferred
 * \param[in]  map         array of indices defining mapping from full to packed array
 * \param[in]  mapSize     number of elements in map array
 */
template<bool accumulate>
__global__ void unpackRecvBufNvshmemKernel(float3* __restrict__ data,
                                    const float3* __restrict__ dataPacked,
                                    const int* __restrict__ map,
                                    const int mapSize,
                                    const int sendSize,
                                    const int atomOffset,
                                    const int sendRank,
                                    uint64_t *sync_f,
                                    uint64_t counter)
{
    // put approach
    if (blockIdx.x  == 0) {
        nvshmemx_float_put_signal_nbi_block((float*)dataPacked, (float*)&data[atomOffset],
        sendSize * 3, sync_f, counter, NVSHMEM_SIGNAL_SET, sendRank);
    }

    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const float3* gm_dataSrc;
    float3* gm_dataDest;
    if (threadIndex < mapSize) {
        gm_dataSrc  = &dataPacked[threadIndex];
        gm_dataDest = &data[map[threadIndex]];
    }

    if (threadIdx.x == 0) {
        nvshmem_signal_wait_until(sync_f, NVSHMEM_CMP_EQ, counter);
    }
    __syncthreads();

    if (threadIndex < mapSize)
    {
        if (accumulate)
        {
            *gm_dataDest += *gm_dataSrc;
        }
        else
        {
            *gm_dataDest = *gm_dataSrc;
        }
    }
}
#endif

void GpuHaloExchange::Impl::launchPackXKernel(const matrix box, 
                                              uint64_t synccounter,
                                              uint64_t *sync_arr)
{
    // launch kernel to pack send buffer
    KernelLaunchConfig config;
    config.blockSize[0]     = c_threadsPerBlock;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (xSendSize_ + c_threadsPerBlock - 1) / c_threadsPerBlock;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;

    const float3* sendBuf  = asFloat3(d_sendBuf_);
    const float3* d_x      = asFloat3(d_x_);
    const int*    indexMap = d_indexMap_;
    const int     size     = xSendSize_;
    // The coordinateShift changes between steps when we have
    // performed a DD partition, or have updated the box e.g. when
    // performing pressure coupling. So, for simplicity, the box
    // is used every step to pass the shift vector as an argument of
    // the packing kernel.
    const int    boxDimensionIndex = dd_->dim[dimIndex_];
    const float3 coordinateShift{ box[boxDimensionIndex][XX],
                                  box[boxDimensionIndex][YY],
                                  box[boxDimensionIndex][ZZ] };

#if !GMX_NVSHMEM
    // Avoid launching kernel when there is no work to do
    if (size > 0)
    {
        auto kernelFn = usePBC_ ? packSendBufKernel<true> : packSendBufKernel<false>;

        const auto kernelArgs = prepareGpuKernelArguments(
                kernelFn, config, &sendBuf, &d_x, &indexMap, &size, &coordinateShift);

        launchGpuKernel(kernelFn, config, *haloStream_, nullptr, "Domdec GPU Apply X Halo Exchange", kernelArgs);
    }
#else
    auto packNvShmemkernelFn = usePBC_ ? packSendMultiBlockNvshmemBufKernel<true> : packSendMultiBlockNvshmemBufKernel<false>;

    constexpr int newThreadblockSize = c_threadsPerBlock * 4;
    config.blockSize[0]     = newThreadblockSize;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (xSendSize_ + newThreadblockSize + 1) / newThreadblockSize;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;

    if (gridDimX != config.gridSize[0]) {
        gridDimX = config.gridSize[0];
        cuda::barrier<cuda::thread_scope_device> temp_bar(gridDimX);
        cudaMemcpyAsync(bar, &temp_bar, sizeof(cuda::barrier<cuda::thread_scope_device>), cudaMemcpyDefault, haloStream_->stream());
    } else {
        config.gridSize[0] = gridDimX;
    }
    const gmx_domdec_comm_t&     comm = *dd_->comm;
    const int nPulses =  comm.cd[dimIndex_].numPulses();
    const auto kernelArgs = prepareGpuKernelArguments(
            packNvShmemkernelFn, config, &d_x, &sendBuf, &indexMap, &size, &coordinateShift, &sendRankX_,
                    &remoteAtomOffset_, &sync_arr, &sync_recv_arr, &recvRankX_, &synccounter,
                    &bar, &pulse_, &nPulses, &dimIndex_, &dd_->ndim);

    launchGpuKernel(packNvShmemkernelFn, config, *haloStream_, nullptr, "Domdec GPU Apply X Halo Exchange", kernelArgs);
#endif
}

// The following method should be called after non-local buffer operations,
// and before the local buffer operations.
void GpuHaloExchange::Impl::launchUnpackFKernel(bool accumulateForces)
{
    KernelLaunchConfig config;
    config.blockSize[0]     = c_threadsPerBlock;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (fRecvSize_ + c_threadsPerBlock - 1) / c_threadsPerBlock;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;

    float3*       d_f      = asFloat3(d_f_);
    const float3* recvBuf  = asFloat3(d_recvBuf_);
    const int*    indexMap = d_indexMap_;
    const int     size     = fRecvSize_;

#if !GMX_NVSHMEM
    if (size > 0)
    {
        auto kernelFn = accumulateForces ? unpackRecvBufKernel<true> : unpackRecvBufKernel<false>;

        const auto kernelArgs =
                prepareGpuKernelArguments(kernelFn, config, &d_f, &recvBuf, &indexMap, &size);

        launchGpuKernel(kernelFn, config, *haloStream_, nullptr, "Domdec GPU Apply F Halo Exchange", kernelArgs);
    }
#else
    config.blockSize[0]     = c_threadsPerBlock * 4;
    config.gridSize[0]      = (fRecvSize_ + config.blockSize[0] + 1) / config.blockSize[0];
    auto kernelFn = accumulateForces ? unpackRecvBufNvshmemKernel<true> : unpackRecvBufNvshmemKernel<false>;
    const auto kernelArgs =
            prepareGpuKernelArguments(kernelFn, config, &d_f, &recvBuf, &indexMap, &size, &fSendSize_, &atomOffset_, 
                                    &sendRankF_, &sync_f, &synccounter_f);

    launchGpuKernel(kernelFn, config, *haloStream_, nullptr, "Domdec GPU Apply F Halo Exchange", kernelArgs);
    synccounter_f++;
#endif
}

} // namespace gmx
