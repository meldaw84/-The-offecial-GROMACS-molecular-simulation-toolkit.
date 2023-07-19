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
 * \brief Implements backend-agnostic part of GPU-direct PME-PP communication.
 *
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include "config.h"

#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/utility/gmxmpi.h"

#include "pme_force_sender_gpu_impl.h"

// for creating padded atom index arrays
#include <algorithm>
#include <numeric>

namespace gmx
{

/*! \brief Create PME-PP GPU communication object */
PmeForceSenderGpu::Impl::Impl(GpuEventSynchronizer*  pmeForcesReady,
                              MPI_Comm               comm,
                              const DeviceContext&   deviceContext,
                              gmx::ArrayRef<PpRanks> ppRanks) :
    pmeForcesReady_(pmeForcesReady), comm_(comm), ppRanks_(ppRanks), deviceContext_(deviceContext)
{
    // Create streams, events and flags to manage pushing of force buffers to remote PP ranks
    ppCommManagers_.reserve(ppRanks.size());
    for (size_t i = 0; i != ppRanks.size(); ++i)
    {
        ppCommManagers_.emplace_back(PpForceCommManager{
                std::make_unique<DeviceStream>(deviceContext_, DeviceStreamPriority::High, false),
                std::make_unique<GpuEventSynchronizer>(),
                std::make_unique<std::atomic<CacheLineAlignedFlag>>(),
                nullptr,
                nullptr,
                nullptr });
    }
    pmeForcesReady_->setConsumptionLimits(ppRanks_.size(), ppRanks_.size());
    pmeForcesReady_->reset();
    stageThreadMpiGpuCpuComm_ = (getenv("GMX_ENABLE_STAGED_GPU_TO_CPU_PMEPP_COMM") != nullptr);

    allocateDeviceBuffer(&d_pmeRemoteGpuForcePtrs, ppRanks.size() * sizeof(RVec*), deviceContext_);
    allocateDeviceBuffer(&d_atomsPerPpProc, (1 + ppRanks.size()) * sizeof(size_t), deviceContext_);
    cudaMalloc(&d_pmeToPpReadyAtomicFlagPtrs_, ppRanks.size() * sizeof(cuda::atomic<int>*));
}

// TODO need to free device buffers here
PmeForceSenderGpu::Impl::~Impl() = default;

/*! \brief Sets location of force to be sent to each PP rank  */
void PmeForceSenderGpu::Impl::setForceSendBuffer(DeviceBuffer<Float3> d_f)
{

    // Need to send address to PP rank only for thread-MPI as PP rank pulls
    // data using cudamemcpy
    if (!GMX_THREAD_MPI)
    {
        return;
    }
    GMX_ASSERT(!GMX_GPU_SYCL,
               "PmeForceSenderGpu does not support SYCL with threadMPI; use libMPI instead.");

#if GMX_MPI && GMX_GPU_CUDA

    int ind_start = 0;
    int ind_end   = 0;
    int i         = 0;
    int n_procs   = ppRanks_.size();
    // set first element in array to array size
    cudaMemcpy(&(d_atomsPerPpProc[0]), &(n_procs), sizeof(size_t), cudaMemcpyHostToDevice);

    for (const auto& receiver : ppRanks_)
    {
        ind_start = ind_end;
        ind_end   = ind_start + receiver.numAtoms;

        if (receiver.numAtoms > 0)
        {
            ppCommManagers_[i].localForcePtr = &d_f[ind_start];
            // NOLINTNEXTLINE(bugprone-sizeof-expression)
            MPI_Recv(&ppCommManagers_[i].pmeRemoteGpuForcePtr,
                     sizeof(Float3*),
                     MPI_BYTE,
                     receiver.rankId,
                     0,
                     comm_,
                     MPI_STATUS_IGNORE);

            MPI_Recv(&ppCommManagers_[i].pmeToPpReadyAtomicFlagPtr,
                     sizeof(cuda::atomic<int>*),
                     MPI_BYTE,
                     receiver.rankId,
                     0,
                     comm_,
                     MPI_STATUS_IGNORE);
            // NOLINTNEXTLINE(bugprone-sizeof-expression)
            MPI_Recv(&ppCommManagers_[i].pmeRemoteCpuForcePtr,
                     sizeof(Float3*),
                     MPI_BYTE,
                     receiver.rankId,
                     0,
                     comm_,
                     MPI_STATUS_IGNORE);
            // Send address of event and associated flag to PP rank, to allow remote enqueueing
            // NOLINTNEXTLINE(bugprone-sizeof-expression)
            MPI_Send(&ppCommManagers_[i].event, sizeof(GpuEventSynchronizer*), MPI_BYTE, receiver.rankId, 0, comm_);

            std::atomic<bool>* tmpPpCommEventRecordedPtr =
                    reinterpret_cast<std::atomic<bool>*>((ppCommManagers_[i].eventRecorded.get()));
            tmpPpCommEventRecordedPtr->store(false, std::memory_order_release);
            // NOLINTNEXTLINE(bugprone-sizeof-expression)
            MPI_Send(&tmpPpCommEventRecordedPtr, sizeof(std::atomic<bool>*), MPI_BYTE, receiver.rankId, 0, comm_);


            cudaMemcpy(&(d_pmeRemoteGpuForcePtrs[i]),
                       &(ppCommManagers_[i].pmeRemoteGpuForcePtr),
                       sizeof(RVec*),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_pmeToPpReadyAtomicFlagPtrs_[i]),
                       &(ppCommManagers_[i].pmeToPpReadyAtomicFlagPtr),
                       sizeof(cuda::atomic<int>*),
                       cudaMemcpyHostToDevice);

            cudaMemcpy(&(d_atomsPerPpProc[i + 1]), &(receiver.numAtoms), sizeof(size_t), cudaMemcpyHostToDevice);
        }
        else
        {
            size_t zeroVal = 0;
            cudaMemcpy(&(d_atomsPerPpProc[i + 1]), &(zeroVal), sizeof(size_t), cudaMemcpyHostToDevice);
        }
        i++;
    }
    int atomsPerBlock = 32;
    // int atomsPerBlock = 8;
    int blockForcesSize = atomsPerBlock * DIM;
    int totalAtoms      = ind_end + 1;
    // hack, assume max is every pp rank requiring one additional overflow block
    int maxPaddedAtoms           = totalAtoms + atomsPerBlock * ppRanks_.size();
    int maxForceEls              = maxPaddedAtoms * DIM;
    int totalPreviousPaddedAtoms = 0;
    int totalPreviousAtoms       = 0;
    int totalPreviousForceEls    = 0;
    // TODO turn this into a single vector of a struct
    std::vector<int> paddedPpRanks(maxForceEls);
    std::vector<int> paddedAtomIndices(maxForceEls);
    std::vector<int> paddedAtomOffsets(maxPaddedAtoms);
    std::fill(paddedPpRanks.begin(), paddedPpRanks.end(), -1);
    std::fill(paddedAtomIndices.begin(), paddedAtomIndices.end(), -1);
    std::fill(paddedAtomOffsets.begin(), paddedAtomOffsets.end(), -1);
    std::vector<int>::iterator rank_it_start, rank_it_end_unpadded;
    std::vector<int>::iterator indices_it_start, indices_it_end_unpadded;
    std::vector<int>::iterator offsets_it_start, offsets_it_end_unpadded;
    // TODO this needs to be a reallocate
    allocateDeviceBuffer(&d_paddedPpRanks, maxForceEls * sizeof(int), deviceContext_);
    allocateDeviceBuffer(&d_paddedAtomIndices, maxForceEls * sizeof(int), deviceContext_);
    allocateDeviceBuffer(&d_paddedAtomOffsets, maxPaddedAtoms * sizeof(int), deviceContext_);
    i = 0;
    for (const auto& receiver : ppRanks_)
    {
        int nBlocks             = ceil(receiver.numAtoms / (float)atomsPerBlock);
        int nPaddedAtoms_local  = nBlocks * atomsPerBlock;
        int nForceEls           = nPaddedAtoms_local * DIM;
        rank_it_start           = paddedPpRanks.begin() + totalPreviousForceEls;
        rank_it_end_unpadded    = rank_it_start + receiver.numAtoms * DIM;
        indices_it_start        = paddedAtomIndices.begin() + totalPreviousForceEls;
        indices_it_end_unpadded = indices_it_start + receiver.numAtoms * DIM;
        offsets_it_start        = paddedAtomOffsets.begin() + totalPreviousPaddedAtoms;
        offsets_it_end_unpadded = offsets_it_start + receiver.numAtoms;
        std::fill(rank_it_start, rank_it_end_unpadded, i);
        std::iota(indices_it_start, indices_it_end_unpadded, 0);
        std::iota(offsets_it_start, offsets_it_end_unpadded, totalPreviousAtoms);
        totalPreviousForceEls += nForceEls;
        totalPreviousPaddedAtoms += nPaddedAtoms_local;
        totalPreviousAtoms += receiver.numAtoms;
        i++;
    }
    nPaddedAtoms = totalPreviousPaddedAtoms;
    cudaMemcpy(d_paddedPpRanks, paddedPpRanks.data(), paddedPpRanks.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_paddedAtomIndices,
               paddedAtomIndices.data(),
               paddedAtomIndices.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_paddedAtomOffsets,
               paddedAtomOffsets.data(),
               paddedAtomOffsets.size() * sizeof(int),
               cudaMemcpyHostToDevice);

#else
    GMX_UNUSED_VALUE(d_f);
#endif
}

/*! \brief Send PME data directly using GPU-aware MPI */
void PmeForceSenderGpu::Impl::sendFToPpGpuAwareMpi(DeviceBuffer<RVec> sendbuf,
                                                   int                offset,
                                                   int                numBytes,
                                                   int                ppRank,
                                                   MPI_Request*       request)
{
    GMX_ASSERT(GMX_LIB_MPI, "sendFToPpCudaMpi is expected to be called only for Lib-MPI");

#if GMX_MPI
    // if using GPU direct comm with GPU-aware MPI, make sure forces are ready on device
    // before sending it to PP ranks
    pmeForcesReady_->waitForEvent();

    MPI_Isend(asMpiPointer(sendbuf) + offset, numBytes, MPI_BYTE, ppRank, 0, comm_, request);

#else
    GMX_UNUSED_VALUE(sendbuf);
    GMX_UNUSED_VALUE(offset);
    GMX_UNUSED_VALUE(numBytes);
    GMX_UNUSED_VALUE(ppRank);
    GMX_UNUSED_VALUE(request);
#endif
}

int PmeForceSenderGpu::Impl::getNPaddedAtoms()
{
    return nPaddedAtoms;
}

DeviceBuffer<RVec*> PmeForceSenderGpu::Impl::getPmeRemoteGpuForcePtrs()
{
    return d_pmeRemoteGpuForcePtrs;
}

DeviceBuffer<int> PmeForceSenderGpu::Impl::getPaddedPpRanks()
{
    return d_paddedPpRanks;
}

DeviceBuffer<int> PmeForceSenderGpu::Impl::getPaddedAtomIndices()
{
    return d_paddedAtomIndices;
}

DeviceBuffer<int> PmeForceSenderGpu::Impl::getPaddedAtomOffsets()
{
    return d_paddedAtomOffsets;
}

DeviceBuffer<size_t> PmeForceSenderGpu::Impl::getAtomsPerPpProc()
{
    return d_atomsPerPpProc;
}

cuda::atomic<int>** PmeForceSenderGpu::Impl::getPmeToPpReadyAtomicFlagPtrs()
{
    return d_pmeToPpReadyAtomicFlagPtrs_;
}

PmeForceSenderGpu::PmeForceSenderGpu(GpuEventSynchronizer*  pmeForcesReady,
                                     MPI_Comm               comm,
                                     const DeviceContext&   deviceContext,
                                     gmx::ArrayRef<PpRanks> ppRanks) :
    impl_(new Impl(pmeForcesReady, comm, deviceContext, ppRanks))
{
}

PmeForceSenderGpu::~PmeForceSenderGpu() = default;


void PmeForceSenderGpu::setForceSendBuffer(DeviceBuffer<RVec> d_f)
{
    impl_->setForceSendBuffer(d_f);
}

void PmeForceSenderGpu::sendFToPpGpuAwareMpi(DeviceBuffer<RVec> sendbuf,
                                             int                offset,
                                             int                numBytes,
                                             int                ppRank,
                                             MPI_Request*       request)
{
    impl_->sendFToPpGpuAwareMpi(sendbuf, offset, numBytes, ppRank, request);
}

void PmeForceSenderGpu::sendFToPpPeerToPeer(int ppRank, int numAtoms, bool sendForcesDirectToPpGpu)
{
    impl_->sendFToPpPeerToPeer(ppRank, numAtoms, sendForcesDirectToPpGpu);
}

int PmeForceSenderGpu::getNPaddedAtoms()
{
    return impl_->getNPaddedAtoms();
}

DeviceBuffer<RVec*> PmeForceSenderGpu::getPmeRemoteGpuForcePtrs()
{
    return impl_->getPmeRemoteGpuForcePtrs();
}

DeviceBuffer<int> PmeForceSenderGpu::getPaddedPpRanks()
{
    return impl_->getPaddedPpRanks();
}

DeviceBuffer<int> PmeForceSenderGpu::getPaddedAtomIndices()
{
    return impl_->getPaddedAtomIndices();
}

DeviceBuffer<int> PmeForceSenderGpu::getPaddedAtomOffsets()
{
    return impl_->getPaddedAtomOffsets();
}

DeviceBuffer<size_t> PmeForceSenderGpu::getAtomsPerPpProc()
{
    return impl_->getAtomsPerPpProc();
}

cuda::atomic<int>** PmeForceSenderGpu::getPmeToPpReadyAtomicFlagPtrs()
{
    return impl_->getPmeToPpReadyAtomicFlagPtrs();
}

} // namespace gmx
