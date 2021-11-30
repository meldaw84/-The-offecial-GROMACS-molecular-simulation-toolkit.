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
/*! \internal \file
 *
 * \brief Defines the MD Graph class
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 *
 * \ingroup module_mdlib
 */

#include "mdgraph_gpu_impl.h"

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/utility/gmxmpi.h"

#ifdef HAVECUDAGRAPHSUPPORT
const bool useGraph = (getenv("GMX_CUDA_GRAPH") != nullptr);
#else
const bool useGraph = false;
#endif

namespace gmx
{

MdGpuGraph::Impl::Impl(const DeviceStreamManager& deviceStreamManager,
                       SimulationWorkload         simulationWork,
                       MPI_Comm                   mpi_comm,
                       MdGraphEvenOrOddStep       evenOrOddStep,
                       gmx_wallcycle*             wcycle) :
    deviceStreamManager_(deviceStreamManager),
    simulationWork_(simulationWork),
    havePPDomainDecomposition_(simulationWork.havePpDomainDecomposition),
    mpi_comm_(mpi_comm),
    evenOrOddStep_(evenOrOddStep),
    wcycle_(wcycle)
{
    tmpEvent_              = std::make_unique<GpuEventSynchronizer>();
    ppTaskCompletionEvent_ = std::make_unique<GpuEventSynchronizer>();

    if (havePPDomainDecomposition_)
    {
        MPI_Barrier(mpi_comm_);
        MPI_Comm_size(mpi_comm_, &ppSize_);
        MPI_Comm_rank(mpi_comm_, &ppRank_);
    }
}

MdGpuGraph::Impl::~Impl() = default;


void MdGpuGraph::Impl::enqueueEventFromAllPpRanksToRank0Stream(GpuEventSynchronizer* event,
                                                               const DeviceStream&   stream)
{

    for (int remotePpRank = 1; remotePpRank < ppSize_; remotePpRank++)
    {
        if (ppRank_ == remotePpRank)
        {
            // send event to rank 0
            MPI_Send(&event,
                     sizeof(GpuEventSynchronizer*), //NOLINT(bugprone-sizeof-expression)
                     MPI_BYTE,
                     0,
                     0,
                     mpi_comm_);
        }
        else if (ppRank_ == 0)
        {
            // rank 0 enqueues recieved event
            GpuEventSynchronizer* eventToEnqueue;
            MPI_Recv(&eventToEnqueue,
                     sizeof(GpuEventSynchronizer*), //NOLINT(bugprone-sizeof-expression)
                     MPI_BYTE,
                     remotePpRank,
                     0,
                     mpi_comm_,
                     MPI_STATUS_IGNORE);
            eventToEnqueue->enqueueWaitEvent(stream);
        }
    }

    if (ppRank_ == 0)
    {
        // rank 0 also enqueues its local event
        event->enqueueWaitEvent(stream);
    }
}

void MdGpuGraph::Impl::enqueueRank0EventToAllPpStreams(GpuEventSynchronizer* event, const DeviceStream& stream)
{

    for (int remotePpRank = 1; remotePpRank < ppSize_; remotePpRank++)
    {
        if (ppRank_ == 0)
        {
            // rank 0 sends event to remote rank
            MPI_Send(&event,
                     sizeof(GpuEventSynchronizer*), //NOLINT(bugprone-sizeof-expression)
                     MPI_BYTE,
                     remotePpRank,
                     0,
                     mpi_comm_);
        }
        else if (ppRank_ == remotePpRank)
        {
            // remote rank enqueues recieved event to its stream
            GpuEventSynchronizer* eventToEnqueue;
            MPI_Recv(&eventToEnqueue,
                     sizeof(GpuEventSynchronizer*), //NOLINT(bugprone-sizeof-expression)
                     MPI_BYTE,
                     0,
                     0,
                     mpi_comm_,
                     MPI_STATUS_IGNORE);
            eventToEnqueue->enqueueWaitEvent(stream);
        }
    }

    if (ppRank_ == 0)
    {
        // rank 0 also enqueues event to its local stream
        event->enqueueWaitEvent(stream);
    }
}

void MdGpuGraph::Impl::reset()
{
    graphCreated_     = false;
    useGraphThisStep_ = false;
    graphIsCapturing_ = false;
    return;
}

void MdGpuGraph::Impl::start(bool                  bNS,
                             bool                  canUseGraphThisStep,
                             bool                  usedGraphLastStep,
                             GpuEventSynchronizer* xReadyOnDeviceEvent)
{
    if (!useGraph || bNS)
    {
        return;
    }

    useGraphThisStep_ = canUseGraphThisStep;
    graphIsCapturing_ = (!graphCreated_ && useGraphThisStep_);

    if (graphIsCapturing_)
    {
        if (havePPDomainDecomposition_)
        {
            MPI_Barrier(mpi_comm_);
        }
        wallcycle_start(wcycle_, WallCycleCounter::LaunchGpu);
        wallcycle_sub_start(wcycle_, WallCycleSubCounter::MdGpuGraphCapture);
    }

    if (useGraphThisStep_ && !usedGraphLastStep)
    {
        // Ensure NB local stream on Rank 0 (which will be used for graph capture and/or launch)
        // waits for coordinates to be ready on all ranks
        enqueueEventFromAllPpRanksToRank0Stream(
                xReadyOnDeviceEvent, deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
    }

    if (graphIsCapturing_)
    {
        graphCreated_ = true;

        if (ppRank_ == 0)
        {
            stat_ = cudaStreamBeginCapture(
                    deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal).stream(),
                    cudaStreamCaptureModeGlobal);
            CU_RET_ERR(stat_,
                       "cudaStreamBeginCapture in MD graph definition initialization failed.");
        }

        if (havePPDomainDecomposition_)
        {
            // Fork remote NB local streams from Rank 0 NB local stream
            tmpEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
            enqueueRank0EventToAllPpStreams(
                    tmpEvent_.get(), deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));

            // Fork NB non-local stream from NB local stream on each rank
            tmpEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
            tmpEvent_->enqueueWaitEvent(
                    deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedNonLocal));
        }

        alternateStepPpTaskCompletionEvent_->enqueueExternalWaitEventWhileCapturingGraph(
                deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));

        // Fork update stream from NB local stream on each rank
        tmpEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
        tmpEvent_->enqueueWaitEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::UpdateAndConstraints));

        if (simulationWork_.useGpuPme)
        {
            // Fork PME stream from NB local stream on each rank
            tmpEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
            tmpEvent_->enqueueWaitEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::Pme));
        }

        // Re-mark xReadyOnDeviceEvent to allow full isolation within graph capture
        xReadyOnDeviceEvent->markEvent(
                deviceStreamManager_.stream(gmx::DeviceStreamType::UpdateAndConstraints));
    }
};


void MdGpuGraph::Impl::end(GpuEventSynchronizer* xUpdatedOnDeviceEvent)
{

    if (!useGraphThisStep_)
    {
        return;
    }

    if (graphIsCapturing_)
    {

        if (simulationWork_.useGpuPme)
        {
            // Join PME stream to NB local stream on each rank
            tmpEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::Pme));
            tmpEvent_->enqueueWaitEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
        }

        // Join update stream to NB local stream on each rank
        tmpEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::UpdateAndConstraints));
        tmpEvent_->enqueueWaitEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));

        ppTaskCompletionEvent_->markExternalEventWhileCapturingGraph(
                deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));

        if (havePPDomainDecomposition_)
        {
            // Join NB non-local stream to NB local stream on each rank
            tmpEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedNonLocal));
            tmpEvent_->enqueueWaitEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));

            // Join remote NB local streams to Rank 0 NB local stream
            tmpEvent_->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
            enqueueEventFromAllPpRanksToRank0Stream(
                    tmpEvent_.get(), deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
        }


        if (ppRank_ == 0)
        {
            stat_ = cudaStreamEndCapture(
                    deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal).stream(), &graph_);
            CU_RET_ERR(stat_, "cudaStreamEndCapture in MD graph definition finalization failed.");
        }

        if (havePPDomainDecomposition_)
        {
            MPI_Barrier(mpi_comm_);
        }
        wallcycle_sub_stop(wcycle_, WallCycleSubCounter::MdGpuGraphCapture);
        wallcycle_sub_start(wcycle_, WallCycleSubCounter::MdGpuGraphInstantiateOrUpdate);
        wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpu);

        if (ppRank_ == 0)
        {
            // Instantiate graph, or update a previously instantiated graph (if possible)
            if (!updateGraph_)
            {
                stat_ = cudaGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
                CU_RET_ERR(stat_,
                           "cudaGraphInstantiate in MD graph definition finalization failed.");
            }
            else
            {
                cudaGraphNode_t           hErrorNode_out;
                cudaGraphExecUpdateResult updateResult_out;
                stat_ = cudaGraphExecUpdate(instance_, graph_, &hErrorNode_out, &updateResult_out);
                CU_RET_ERR(stat_, "cudaGraphExecUpdate in MD graph definition finalization failed.");
            }

            // With current CUDA, only single-threaded update is possible.
            // Multi-threaded update support will be available in a future CUDA release.
            if (ppSize_ == 1)
            {
                updateGraph_ = true;
            }
        }
        if (havePPDomainDecomposition_)
        {
            MPI_Barrier(mpi_comm_);
        }
        wallcycle_sub_stop(wcycle_, WallCycleSubCounter::MdGpuGraphInstantiateOrUpdate);
    }

    if (havePPDomainDecomposition_)
    {
        MPI_Barrier(mpi_comm_);
    }
    wallcycle_start(wcycle_, WallCycleCounter::LaunchGpu);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::MdGpuGraphLaunch);

    const DeviceStream* thisLaunchStream =
            (evenOrOddStep_ == MdGraphEvenOrOddStep::EvenStep)
                    ? &deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal)
                    : &deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedNonLocal);

    if (ppRank_ == 0)
    {
        stat_ = cudaGraphLaunch(instance_, thisLaunchStream->stream());
        CU_RET_ERR(stat_, "cudaGraphLaunch in MD graph definition finalization failed.");
        tmpEvent_->markEvent(*thisLaunchStream);
    }
    enqueueRank0EventToAllPpStreams(
            tmpEvent_.get(), deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));
    xUpdatedOnDeviceEvent->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal));

    if (havePPDomainDecomposition_)
    {
        MPI_Barrier(mpi_comm_);
    }
    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::MdGpuGraphLaunch);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpu);
};

void MdGpuGraph::Impl::setAlternateStepPpTaskCompletionEvent(GpuEventSynchronizer* event)
{
    alternateStepPpTaskCompletionEvent_ = event;
}

GpuEventSynchronizer* MdGpuGraph::Impl::getPpTaskCompletionEvent()
{
    return ppTaskCompletionEvent_.get();
}

MdGpuGraph::MdGpuGraph(const DeviceStreamManager& deviceStreamManager,
                       SimulationWorkload         simulationWork,
                       MPI_Comm                   mpi_comm,
                       MdGraphEvenOrOddStep       evenOrOddStep,
                       gmx_wallcycle*             wcycle) :
    impl_(new Impl(deviceStreamManager, simulationWork, mpi_comm, evenOrOddStep, wcycle))
{
}

MdGpuGraph::~MdGpuGraph() = default;

void MdGpuGraph::reset()
{
    impl_->reset();
}

void MdGpuGraph::start(bool bNS, bool canUseGraphThisStep, bool usedGraphLastStep, GpuEventSynchronizer* xReadyOnDeviceEvent)
{
    impl_->start(bNS, canUseGraphThisStep, usedGraphLastStep, xReadyOnDeviceEvent);
}

void MdGpuGraph::end(GpuEventSynchronizer* xUpdatedOnDeviceEvent)
{
    impl_->end(xUpdatedOnDeviceEvent);
}

bool MdGpuGraph::useGraphThisStep() const
{
    return impl_->useGraphThisStep();
}

bool MdGpuGraph::graphIsCapturing() const
{
    return impl_->graphIsCapturing();
}

void MdGpuGraph::setAlternateStepPpTaskCompletionEvent(GpuEventSynchronizer* event)
{
    impl_->setAlternateStepPpTaskCompletionEvent(event);
}

GpuEventSynchronizer* MdGpuGraph::getPpTaskCompletionEvent()
{
    return impl_->getPpTaskCompletionEvent();
}

} // namespace gmx
