/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2022, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
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
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
/*! \internal \file
 *
 * \brief Implements GPU halo exchange using SYCL.
 *
 *
 * \author Alan Gray <alang@nvidia.com>
 * \author Andrey Alekseenko <al42and@gmail.com>
 *
 * \ingroup module_domdec
 */
#include "gmxpre.h"

#include "gpuhaloexchange_impl_sycl.h"

#include "config.h"

#include <utility>

#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/gpuhaloexchange.h"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gmxsycl.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/gpu_utils/gputraits_sycl.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/gmxmpi.h"

#include "domdec_internal.h"

template<bool usePbc>
class PackSendBufKernel;

template<bool accumulate>
class UnpackRecvBufKernel;

namespace gmx
{

template<typename T>
static T* asMpiPointer(DeviceBuffer<T>& buffer)
{
    return buffer ? buffer.buffer_->ptr_ : nullptr;
}

template<bool usePbc>
auto packSendBufKernel(sycl::handler&                                        cgh,
                       DeviceAccessor<Float3, sycl::access_mode::read_write> a_dataPacked,
                       DeviceAccessor<Float3, sycl::access_mode::read>       a_data,
                       DeviceAccessor<int, sycl::access_mode::read>          a_map,
                       int                                                   mapSize,
                       Float3                                                coordinateShift)
{
    a_dataPacked.bind(cgh);
    a_data.bind(cgh);
    a_map.bind(cgh);

    return [=](sycl::id<1> itemIdx) {
        const int threadIndex = itemIdx;
        if (threadIndex < mapSize)
        {
            if constexpr (usePbc)
            {
                a_dataPacked[itemIdx] = a_data[a_map[itemIdx]] + coordinateShift;
            }
            else
            {
                a_dataPacked[itemIdx] = a_data[a_map[itemIdx]];
            }
        }
    };
}

/*! \brief unpack non-local force data buffer on the GPU using pre-populated "map" containing index
 * information
 * \param[out] a_data        full array of force values
 * \param[in]  a_dataPacked  packed array of force values to be transferred
 * \param[in]  a_map         array of indices defining mapping from full to packed array
 * \param[in]  mapSize       number of elements in map array
 */
template<bool accumulate>
auto unpackRecvBufKernel(sycl::handler&                                        cgh,
                         DeviceAccessor<Float3, sycl::access_mode::read_write> a_data,
                         DeviceAccessor<Float3, sycl::access_mode::read>       a_dataPacked,
                         DeviceAccessor<int, sycl::access_mode::read>          a_map,
                         int                                                   mapSize)
{
    a_dataPacked.bind(cgh);
    a_data.bind(cgh);
    a_map.bind(cgh);

    return [=](sycl::id<1> itemIdx) {
        const int threadIndex = itemIdx;
        if (threadIndex < mapSize)
        {
            if constexpr (accumulate)
            {
                a_data[a_map[itemIdx]] += a_dataPacked[itemIdx];
            }
            else
            {
                a_data[a_map[itemIdx]] = a_dataPacked[itemIdx];
            }
        }
    };
}


void GpuHaloExchange::Impl::reinitHalo(DeviceBuffer<Float3>& d_coordinatesBuffer,
                                       DeviceBuffer<Float3>& d_forcesBuffer)
{
    wallcycle_start(wcycle_, WallCycleCounter::Domdec);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::DDGpu);

    d_x_ = d_coordinatesBuffer;
    d_f_ = d_forcesBuffer;

    const gmx_domdec_comm_t&     comm = *dd_->comm;
    const gmx_domdec_comm_dim_t& cd   = comm.cd[dimIndex_];
    const gmx_domdec_ind_t&      ind  = cd.ind[pulse_];

    numHomeAtoms_ = comm.atomRanges.numHomeAtoms(); // offset for data received by this rank

    // Determine receive offset for the dimension index and pulse of this halo exchange object
    int numZoneTemp   = 1;
    int numZone       = 0;
    int numAtomsTotal = numHomeAtoms_;
    for (int i = 0; i <= dimIndex_; i++)
    {
        int pulseMax = (i == dimIndex_) ? pulse_ : (comm.cd[i].numPulses() - 1);
        for (int p = 0; p <= pulseMax; p++)
        {
            atomOffset_                     = numAtomsTotal;
            const gmx_domdec_ind_t& indTemp = comm.cd[i].ind[p];
            numAtomsTotal += indTemp.nrecv[numZoneTemp + 1];
        }
        numZone = numZoneTemp;
        numZoneTemp += numZoneTemp;
    }

    int newSize = ind.nsend[numZone + 1];

    GMX_ASSERT(cd.receiveInPlace, "Out-of-place receive is not yet supported in GPU halo exchange");

    // reallocates only if needed
    h_indexMap_.resize(newSize);
    // reallocate on device only if needed
    if (newSize > maxPackedBufferSize_)
    {
        reallocateDeviceBuffer(&d_indexMap_, newSize, &indexMapSize_, &indexMapSizeAlloc_, deviceContext_);
        reallocateDeviceBuffer(&d_sendBuf_, newSize, &sendBufSize_, &sendBufSizeAlloc_, deviceContext_);
        reallocateDeviceBuffer(&d_recvBuf_, newSize, &recvBufSize_, &recvBufSizeAlloc_, deviceContext_);
        maxPackedBufferSize_ = newSize;
    }

    xSendSize_ = newSize;
#if GMX_MPI
    MPI_Sendrecv(&xSendSize_,
                 sizeof(int),
                 MPI_BYTE,
                 sendRankX_,
                 0,
                 &xRecvSize_,
                 sizeof(int),
                 MPI_BYTE,
                 recvRankX_,
                 0,
                 mpi_comm_mysim_,
                 MPI_STATUS_IGNORE);
#endif
    fSendSize_ = xRecvSize_;
    fRecvSize_ = xSendSize_;

    if (newSize > 0)
    {
        GMX_ASSERT(ind.index.size() == h_indexMap_.size(),
                   "Size mismatch between domain decomposition communication index array and GPU "
                   "halo exchange index mapping array");
        std::copy(ind.index.begin(), ind.index.end(), h_indexMap_.begin());

        copyToDeviceBuffer(
                &d_indexMap_, h_indexMap_.data(), 0, newSize, *haloStream_, GpuApiCallBehavior::Async, nullptr);
    }

#if GMX_MPI
    // Exchange of remote addresses from neighboring ranks is needed only with CUDA-direct as cudamemcpy needs both src/dst pointer
    // MPI calls such as MPI_send doesn't worry about receiving address, that is taken care by MPI_recv call in neighboring rank
    if (GMX_THREAD_MPI)
    {
        // This rank will push data to its neighbor, so needs to know
        // the remote receive address and similarly send its receive
        // address to other neighbour. We can do this here in reinit fn
        // since the pointers will not change until the next NS step.

        // Coordinates buffer:
        Float3* recvPtr = &asMpiPointer(d_x_)[atomOffset_];
        MPI_Sendrecv(&recvPtr,
                     sizeof(void*),
                     MPI_BYTE,
                     recvRankX_,
                     0,
                     &remoteXPtr_,
                     sizeof(void*),
                     MPI_BYTE,
                     sendRankX_,
                     0,
                     mpi_comm_mysim_,
                     MPI_STATUS_IGNORE);

        // Force buffer:
        recvPtr = asMpiPointer(d_recvBuf_);
        MPI_Sendrecv(&recvPtr,
                     sizeof(void*),
                     MPI_BYTE,
                     recvRankF_,
                     0,
                     &remoteFPtr_,
                     sizeof(void*),
                     MPI_BYTE,
                     sendRankF_,
                     0,
                     mpi_comm_mysim_,
                     MPI_STATUS_IGNORE);
    }
#endif

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::DDGpu);
    wallcycle_stop(wcycle_, WallCycleCounter::Domdec);
}

void GpuHaloExchange::Impl::enqueueWaitRemoteCoordinatesReadyEvent(GpuEventSynchronizer* coordinatesReadyOnDeviceEvent)
{
#if GMX_MPI
    GMX_ASSERT(coordinatesReadyOnDeviceEvent != nullptr,
               "Co-ordinate Halo exchange requires valid co-ordinate ready event");

    // Wait for event from receiving task that remote coordinates are ready, and enqueue that event to stream used
    // for subsequent data push. This avoids a race condition with the remote data being written in the previous timestep.
    // Similarly send event to task that will push data to this task.
    GpuEventSynchronizer* remoteCoordinatesReadyOnDeviceEvent;
    MPI_Sendrecv(&coordinatesReadyOnDeviceEvent,
                 sizeof(GpuEventSynchronizer*), //NOLINT(bugprone-sizeof-expression)
                 MPI_BYTE,
                 recvRankX_,
                 0,
                 &remoteCoordinatesReadyOnDeviceEvent,
                 sizeof(GpuEventSynchronizer*), //NOLINT(bugprone-sizeof-expression)
                 MPI_BYTE,
                 sendRankX_,
                 0,
                 mpi_comm_mysim_,
                 MPI_STATUS_IGNORE);
    remoteCoordinatesReadyOnDeviceEvent->enqueueWaitEvent(*haloStream_);
#else
    GMX_UNUSED_VALUE(coordinatesReadyOnDeviceEvent);
#endif
}

template<bool usePbc, class... Args>
static sycl::event launchPackSendBufKernel(const DeviceStream& deviceStream, int xSendSize, Args&&... args)
{
    using kernelNameType = PackSendBufKernel<usePbc>;

    const sycl::range<1> range(xSendSize);
    sycl::queue          q = deviceStream.stream();

    sycl::event e = q.submit([&](sycl::handler& cgh) {
        auto kernel = packSendBufKernel<usePbc>(cgh, std::forward<Args>(args)...);
        cgh.parallel_for<kernelNameType>(range, kernel);
    });

    return e;
}

template<bool accumulateForces, class... Args>
static sycl::event launchUnpackRecvBufKernel(const DeviceStream& deviceStream, int fRecvSize, Args&&... args)
{
    using kernelNameType = UnpackRecvBufKernel<accumulateForces>;

    const sycl::range<1> range(fRecvSize);
    sycl::queue          q = deviceStream.stream();

    sycl::event e = q.submit([&](sycl::handler& cgh) {
        auto kernel = unpackRecvBufKernel<accumulateForces>(cgh, std::forward<Args>(args)...);
        cgh.parallel_for<kernelNameType>(range, kernel);
    });

    return e;
}

GpuEventSynchronizer* GpuHaloExchange::Impl::communicateHaloCoordinates(const matrix box,
                                                                        GpuEventSynchronizer* dependencyEvent)
{
    wallcycle_start(wcycle_, WallCycleCounter::LaunchGpu);

    // ensure stream waits until dependency has been satisfied
    dependencyEvent->enqueueWaitEvent(*haloStream_);

    wallcycle_sub_start(wcycle_, WallCycleSubCounter::LaunchGpuMoveX);

    const int size = xSendSize_;
    // The coordinateShift changes between steps when we have
    // performed a DD partition, or have updated the box e.g. when
    // performing pressure coupling. So, for simplicity, the box
    // is used every step to pass the shift vector as an argument of
    // the packing kernel.
    const int    boxDimensionIndex = dd_->dim[dimIndex_];
    const Float3 coordinateShift{ box[boxDimensionIndex][XX],
                                  box[boxDimensionIndex][YY],
                                  box[boxDimensionIndex][ZZ] };

    // Avoid launching kernel when there is no work to do
    if (size > 0)
    {
        if (usePBC_)
        {
            launchPackSendBufKernel<true>(
                    *haloStream_, size, d_sendBuf_, d_x_, d_indexMap_, size, coordinateShift);
        }
        else
        {
            launchPackSendBufKernel<false>(
                    *haloStream_, size, d_sendBuf_, d_x_, d_indexMap_, size, coordinateShift);
        }
    }

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuMoveX);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpu);

    // Consider time spent in communicateHaloData as Comm.X counter
    // ToDo: We need further refinement here as communicateHaloData includes launch time for cudamemcpyasync
    wallcycle_start(wcycle_, WallCycleCounter::MoveX);

    // wait for remote co-ordinates is implicit with process-MPI as non-local stream is synchronized before MPI calls
    // and MPI_Waitall call makes sure both neighboring ranks' non-local stream is synchronized before data transfer is initiated
    // For multi-dimensional halo exchanges, this needs to be done for every dimIndex_, since the remote ranks will be different
    // for each. But different pulses within a dimension will communicate with the same remote ranks so we can restrict to the first pulse.
    if (GMX_THREAD_MPI && pulse_ == 0)
    {
        enqueueWaitRemoteCoordinatesReadyEvent(dependencyEvent);
    }

    Float3* recvPtr = GMX_THREAD_MPI ? asMpiPointer(remoteXPtr_) : &asMpiPointer(d_x_)[atomOffset_];
    communicateHaloData(asMpiPointer(d_sendBuf_), xSendSize_, sendRankX_, recvPtr, xRecvSize_, recvRankX_);

    coordinateHaloLaunched_.markEvent(*haloStream_);

    wallcycle_stop(wcycle_, WallCycleCounter::MoveX);

    return &coordinateHaloLaunched_;
}

// The following method should be called after non-local buffer operations,
// and before the local buffer operations.
void GpuHaloExchange::Impl::communicateHaloForces(bool accumulateForces,
                                                  FixedCapacityVector<GpuEventSynchronizer*, 2>* dependencyEvents)
{

    // Consider time spent in communicateHaloData as Comm.F counter
    // ToDo: We need further refinement here as communicateHaloData includes launch time for cudamemcpyasync
    wallcycle_start(wcycle_, WallCycleCounter::MoveF);

    while (!dependencyEvents->empty())
    {
        auto* dependency = dependencyEvents->back();
        dependency->enqueueWaitEvent(*haloStream_);
        dependencyEvents->pop_back();
    }

    Float3* recvPtr = asMpiPointer(GMX_THREAD_MPI ? remoteFPtr_ : d_recvBuf_);

    // Communicate halo data
    communicateHaloData(
            &(asMpiPointer(d_f_)[atomOffset_]), fSendSize_, sendRankF_, recvPtr, fRecvSize_, recvRankF_);

    wallcycle_stop(wcycle_, WallCycleCounter::MoveF);

    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpu);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::LaunchGpuMoveF);

    Float3*   d_f  = asMpiPointer(d_f_);
    const int size = fRecvSize_;

    // Unpack halo buffer into force array


    if (pulse_ > 0 || dd_->ndim > 1)
    {
        // We need to accumulate rather than set, since it is possible
        // that, in this pulse/dim, a value could be written to a location
        // corresponding to the halo region of a following pulse/dim.
        accumulateForces = true;
    }

    if (size > 0)
    {
        if (accumulateForces)
        {
            launchUnpackRecvBufKernel<true>(*haloStream_, size, d_f, d_recvBuf_, d_indexMap_, size);
        }
        else
        {
            launchUnpackRecvBufKernel<false>(*haloStream_, size, d_f, d_recvBuf_, d_indexMap_, size);
        }
    }

    fReadyOnDevice_.markEvent(*haloStream_);

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuMoveF);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpu);
}

void GpuHaloExchange::Impl::communicateHaloData(Float3* sendPtr,
                                                int     sendSize,
                                                int     sendRank,
                                                Float3* recvPtr,
                                                int     recvSize,
                                                int     recvRank)
{
    if (GMX_THREAD_MPI)
    {
        // no need to explicitly sync with GMX_THREAD_MPI as all operations are
        // anyway launched in correct stream
        communicateHaloDataWithSyclDirect(sendPtr, sendSize, sendRank, recvPtr, recvRank);
    }
    else
    {
        communicateHaloDataWithSyclMpi(sendPtr, sendSize, sendRank, recvPtr, recvSize, recvRank);
    }
}

void GpuHaloExchange::Impl::communicateHaloDataWithSyclMpi(Float3* sendPtr,
                                                           int     sendSize,
                                                           int     sendRank,
                                                           Float3* recvPtr,
                                                           int     recvSize,
                                                           int     recvRank)
{
    // no need to wait for haloDataReadyOnDevice event if this rank is not sending any data
    if (sendSize > 0)
    {
        // wait for halo stream to complete all outstanding
        // activities, to ensure that buffer is up-to-date in GPU memory
        // before transferring to remote rank

        // ToDo: Replace stream synchronize with event synchronize
        haloStream_->synchronize();
    }

    // perform halo exchange directly in device buffers
#if GMX_MPI
    MPI_Request request;

    // recv remote data into halo region
    MPI_Irecv(recvPtr, recvSize * DIM, MPI_FLOAT, recvRank, 0, mpi_comm_mysim_, &request);

    // send data to remote halo region
    MPI_Send(sendPtr, sendSize * DIM, MPI_FLOAT, sendRank, 0, mpi_comm_mysim_);

    MPI_Wait(&request, MPI_STATUS_IGNORE);
#else
    GMX_UNUSED_VALUE(sendPtr);
    GMX_UNUSED_VALUE(sendRank);
    GMX_UNUSED_VALUE(recvPtr);
    GMX_UNUSED_VALUE(recvSize);
    GMX_UNUSED_VALUE(recvRank);
#endif
}

void GpuHaloExchange::Impl::communicateHaloDataWithSyclDirect(Float3*, int, int, Float3*, int)
{
    GMX_RELEASE_ASSERT(false, "GPU Halo exchange with SYCL not implemented for threadMPI");
}

GpuEventSynchronizer* GpuHaloExchange::Impl::getForcesReadyOnDeviceEvent()
{
    return &fReadyOnDevice_;
}

/*! \brief Create Domdec GPU object */
GpuHaloExchange::Impl::Impl(gmx_domdec_t*        dd,
                            int                  dimIndex,
                            MPI_Comm             mpi_comm_mysim,
                            const DeviceContext& deviceContext,
                            int                  pulse,
                            gmx_wallcycle*       wcycle) :
    dd_(dd),
    sendRankX_(dd->neighbor[dimIndex][1]),
    recvRankX_(dd->neighbor[dimIndex][0]),
    sendRankF_(dd->neighbor[dimIndex][0]),
    recvRankF_(dd->neighbor[dimIndex][1]),
    usePBC_(dd->ci[dd->dim[dimIndex]] == 0),
    haloDataTransferLaunched_(GMX_THREAD_MPI ? new GpuEventSynchronizer() : nullptr),
    mpi_comm_mysim_(mpi_comm_mysim),
    deviceContext_(deviceContext),
    haloStream_(new DeviceStream(deviceContext, DeviceStreamPriority::High, false)),
    dimIndex_(dimIndex),
    pulse_(pulse),
    wcycle_(wcycle)
{
    if (usePBC_ && dd->unitCellInfo.haveScrewPBC)
    {
        gmx_fatal(FARGS, "Error: screw is not yet supported in GPU halo exchange\n");
    }

    changePinningPolicy(&h_indexMap_, gmx::PinningPolicy::PinnedIfSupported);

    allocateDeviceBuffer(&d_fShift_, 1, deviceContext_);
}

GpuHaloExchange::Impl::~Impl()
{
    freeDeviceBuffer(&d_indexMap_);
    freeDeviceBuffer(&d_sendBuf_);
    freeDeviceBuffer(&d_recvBuf_);
    freeDeviceBuffer(&d_fShift_);
    delete haloDataTransferLaunched_;
}

GpuHaloExchange::GpuHaloExchange(gmx_domdec_t*        dd,
                                 int                  dimIndex,
                                 MPI_Comm             mpi_comm_mysim,
                                 const DeviceContext& deviceContext,
                                 int                  pulse,
                                 gmx_wallcycle*       wcycle) :
    impl_(new Impl(dd, dimIndex, mpi_comm_mysim, deviceContext, pulse, wcycle))
{
}

GpuHaloExchange::GpuHaloExchange(GpuHaloExchange&&) noexcept = default;

GpuHaloExchange& GpuHaloExchange::operator=(GpuHaloExchange&& other) noexcept
{
    std::swap(impl_, other.impl_);
    return *this;
}

GpuHaloExchange::~GpuHaloExchange() = default;

void GpuHaloExchange::reinitHalo(DeviceBuffer<RVec> d_coordinatesBuffer, DeviceBuffer<RVec> d_forcesBuffer)
{
    impl_->reinitHalo(d_coordinatesBuffer, d_forcesBuffer);
}

GpuEventSynchronizer* GpuHaloExchange::communicateHaloCoordinates(const matrix          box,
                                                                  GpuEventSynchronizer* dependencyEvent)
{
    return impl_->communicateHaloCoordinates(box, dependencyEvent);
}

void GpuHaloExchange::communicateHaloForces(bool accumulateForces,
                                            FixedCapacityVector<GpuEventSynchronizer*, 2>* dependencyEvents)
{
    impl_->communicateHaloForces(accumulateForces, dependencyEvents);
}

GpuEventSynchronizer* GpuHaloExchange::getForcesReadyOnDeviceEvent()
{
    return impl_->getForcesReadyOnDeviceEvent();
}
} // namespace gmx
