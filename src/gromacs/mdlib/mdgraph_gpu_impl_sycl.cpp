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
 * \author Andrey Alekseenko <al42and@gmail.com>
 *
 *
 * \ingroup module_mdlib
 */

#include "gmxpre.h"

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/gmxsycl.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/utility/gmxmpi.h"

#include "mdgraph_gpu_impl.h"

namespace syclex = sycl::ext::oneapi::experimental;

namespace gmx
{

MdGpuGraph::Impl::Impl(const DeviceStreamManager& deviceStreamManager,
                       SimulationWorkload         simulationWork,
                       MPI_Comm                   mpiComm,
                       MdGraphEvenOrOddStep       evenOrOddStep,
                       gmx_wallcycle*             wcycle) :
    deviceStreamManager_(deviceStreamManager),
    launchStream_(new DeviceStream(deviceStreamManager.context(), DeviceStreamPriority::Normal, false)),
    launchStreamAlternate_(
            new DeviceStream(deviceStreamManager.context(), DeviceStreamPriority::Normal, false)),
    havePPDomainDecomposition_(simulationWork.havePpDomainDecomposition),
    haveGpuPmeOnThisPpRank_(simulationWork.haveGpuPmeOnPpRank()),
    haveSeparatePmeRank_(simulationWork.haveSeparatePmeRank),
    mpiComm_(mpiComm),
    evenOrOddStep_(evenOrOddStep),
    wcycle_(wcycle)
{
    helperEvent_           = std::make_unique<GpuEventSynchronizer>();
    ppTaskCompletionEvent_ = std::make_unique<GpuEventSynchronizer>();

    GMX_RELEASE_ASSERT(!havePPDomainDecomposition_,
                       "GPU Graphs with multiple ranks require threadMPI with GPU-direct "
                       "communication, but it is not supported in SYCL");
    GMX_RELEASE_ASSERT(!haveSeparatePmeRank_,
                       "GPU Graphs with separate PME rank require threadMPI with GPU-direct "
                       "communication, but it is not supported in SYCL");

    // Disable check for cycles in graph
    const sycl::property_list propList{ syclex::property::graph::no_cycle_check() };
    graph_ = std::make_unique<syclex::command_graph<syclex::graph_state::modifiable>>(
            deviceStreamManager.context().context(), deviceStreamManager.deviceInfo().syclDevice, propList);
}

MdGpuGraph::Impl::~Impl()
{
    /*
    stat_ = cudaDeviceSynchronize();
    CU_RET_ERR(stat_, "cudaDeviceSynchronize during MD graph cleanup failed.");
     */
}


void MdGpuGraph::Impl::enqueueEventFromAllPpRanksToRank0Stream(GpuEventSynchronizer*, const DeviceStream&)
{
    GMX_RELEASE_ASSERT(false, "enqueueEventFromAllPpRanksToRank0Stream not supported in SYCL");
}

void MdGpuGraph::Impl::enqueueRank0EventToAllPpStreams(GpuEventSynchronizer*, const DeviceStream&)
{
    GMX_RELEASE_ASSERT(false, "enqueueRank0EventToAllPpStreams not supported in SYCL");
}

void MdGpuGraph::Impl::reset()
{
    graphCreated_             = false;
    useGraphThisStep_         = false;
    graphIsCapturingThisStep_ = false;
    graphState_               = GraphState::Invalid;
}

void MdGpuGraph::Impl::disableForDomainIfAnyPpRankHasCpuForces(bool disableGraphAcrossAllPpRanks)
{
    disableGraphAcrossAllPpRanks_ = disableGraphAcrossAllPpRanks;
}

bool MdGpuGraph::Impl::captureThisStep(bool canUseGraphThisStep)
{
    useGraphThisStep_         = canUseGraphThisStep && !disableGraphAcrossAllPpRanks_;
    graphIsCapturingThisStep_ = useGraphThisStep_ && !graphCreated_;
    return graphIsCapturingThisStep_;
}

void MdGpuGraph::Impl::setUsedGraphLastStep(bool usedGraphLastStep)
{
    usedGraphLastStep_ = usedGraphLastStep;
}

void MdGpuGraph::Impl::startRecord(GpuEventSynchronizer* xReadyOnDeviceEvent)
{

    GMX_ASSERT(useGraphThisStep_,
               "startRecord should not have been called if graph is not in use this step");
    GMX_ASSERT(graphIsCapturingThisStep_,
               "startRecord should not have been called if graph is not capturing this step");
    GMX_ASSERT(graphState_ == GraphState::Invalid,
               "Graph should be in an invalid state before recording");
    GMX_RELEASE_ASSERT(ppRank_ == 0, "SYCL Graph does not support recording with PP decomposition");

    wallcycle_start(wcycle_, WallCycleCounter::MdGpuGraph);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::MdGpuGraphCapture);

    graphCreated_ = true;

    std::vector<sycl::queue> queuesToRecord;
    queuesToRecord.emplace_back(deviceStreamManager_.stream(gmx::DeviceStreamType::NonBondedLocal).stream());
    queuesToRecord.emplace_back(
            deviceStreamManager_.stream(gmx::DeviceStreamType::UpdateAndConstraints).stream());
    if (haveGpuPmeOnThisPpRank_)
    {
        queuesToRecord.emplace_back(deviceStreamManager_.stream(gmx::DeviceStreamType::Pme).stream());
    }

    bool result = graph_->begin_recording(queuesToRecord); // It can also throw
    GMX_RELEASE_ASSERT(result, "Failed to start graph recording");

    // Re-mark xReadyOnDeviceEvent to allow full isolation within graph capture
    // We explicitly want to replace an existing event, so we call the reset here
    if (xReadyOnDeviceEvent->isMarked())
    {
        xReadyOnDeviceEvent->reset();
    }
    xReadyOnDeviceEvent->markEvent(deviceStreamManager_.stream(gmx::DeviceStreamType::UpdateAndConstraints));

    graphState_ = GraphState::Recording;
};


void MdGpuGraph::Impl::endRecord()
{

    GMX_ASSERT(useGraphThisStep_,
               "endRecord should not have been called if graph is not in use this step");
    GMX_ASSERT(graphIsCapturingThisStep_,
               "endRecord should not have been called if graph is not capturing this step");
    GMX_ASSERT(graphState_ == GraphState::Recording,
               "Graph should be in a recording state before recording is ended");


    graph_->end_recording();

    graphState_ = GraphState::Recorded;

    // Sync all tasks before closing timing region, since the graph capture should be treated as a collective operation for timing purposes.
    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::MdGpuGraphCapture);
    wallcycle_stop(wcycle_, WallCycleCounter::MdGpuGraph);
};

void MdGpuGraph::Impl::createExecutableGraph(bool forceGraphReinstantiation)
{

    GMX_ASSERT(
            useGraphThisStep_,
            "createExecutableGraph should not have been called if graph is not in use this step");
    GMX_ASSERT(graphIsCapturingThisStep_,
               "createExecutableGraph should not have been called if graph is not capturing this "
               "step");
    GMX_ASSERT(graphState_ == GraphState::Recorded,
               "Graph should be in a recorded state before instantiation");

    // graph::update  API exists, but it is not supported, so we always re-instantiate
    GMX_UNUSED_VALUE(forceGraphReinstantiation);

    wallcycle_start(wcycle_, WallCycleCounter::MdGpuGraph);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::MdGpuGraphInstantiateOrUpdate);

    // Instantiate graph
    const auto instance = graph_->finalize();
    instance_ = std::make_unique<syclex::command_graph<syclex::graph_state::executable>>(std::move(instance));

    graphInstanceAllocated_ = true;
    graphState_             = GraphState::Instantiated;

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::MdGpuGraphInstantiateOrUpdate);
    wallcycle_stop(wcycle_, WallCycleCounter::MdGpuGraph);
};

void MdGpuGraph::Impl::launchGraphMdStep(GpuEventSynchronizer* xUpdatedOnDeviceEvent)
{

    GMX_ASSERT(useGraphThisStep_,
               "launchGraphMdStep should not have been called if graph is not in use this step");
    GMX_ASSERT(graphState_ == GraphState::Instantiated,
               "Graph should be in an instantiated state before launching");

    wallcycle_start(wcycle_, WallCycleCounter::MdGpuGraph);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::MdGpuGraphLaunch);

    const DeviceStream* thisLaunchStream = launchStream_.get();

    /* We cannot have the same graph instance enqueued twice in SYCL Graphs, so we use
     * helperEvent_ to prevent enqueueing the same graph twice.
     * This partially negates the benefits of using graph scheduling in the first place,
     * but, hopefully, the limitation will be lifted in the future. */
    if (helperEvent_->isMarked())
    {
        helperEvent_->waitForEvent();
    }

    thisLaunchStream->stream().ext_oneapi_graph(*instance_);

    helperEvent_->markEvent(*thisLaunchStream);

    if (xUpdatedOnDeviceEvent->isMarked())
    {
        // This reset should not be needed, dirty hack!
        xUpdatedOnDeviceEvent->reset();
    }
    xUpdatedOnDeviceEvent->markEvent(*thisLaunchStream);

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::MdGpuGraphLaunch);
    wallcycle_stop(wcycle_, WallCycleCounter::MdGpuGraph);
}

void MdGpuGraph::Impl::setAlternateStepPpTaskCompletionEvent(GpuEventSynchronizer*) {}

GpuEventSynchronizer* MdGpuGraph::Impl::getPpTaskCompletionEvent()
{
    return nullptr;
}

MdGpuGraph::MdGpuGraph(const DeviceStreamManager& deviceStreamManager,
                       SimulationWorkload         simulationWork,
                       MPI_Comm                   mpiComm,
                       MdGraphEvenOrOddStep       evenOrOddStep,
                       gmx_wallcycle*             wcycle) :
    impl_(new Impl(deviceStreamManager, simulationWork, mpiComm, evenOrOddStep, wcycle))
{
}

MdGpuGraph::~MdGpuGraph() = default;

void MdGpuGraph::reset()
{
    impl_->reset();
}

void MdGpuGraph::disableForDomainIfAnyPpRankHasCpuForces(bool disableGraphAcrossAllPpRanks)
{
    impl_->disableForDomainIfAnyPpRankHasCpuForces(disableGraphAcrossAllPpRanks);
}

bool MdGpuGraph::captureThisStep(bool canUseGraphThisStep)
{
    return impl_->captureThisStep(canUseGraphThisStep);
}

void MdGpuGraph::setUsedGraphLastStep(bool usedGraphLastStep)
{
    impl_->setUsedGraphLastStep(usedGraphLastStep);
}

void MdGpuGraph::startRecord(GpuEventSynchronizer* xReadyOnDeviceEvent)
{
    impl_->startRecord(xReadyOnDeviceEvent);
}

void MdGpuGraph::endRecord()
{
    impl_->endRecord();
}

void MdGpuGraph::createExecutableGraph(bool forceGraphReinstantiation)
{
    impl_->createExecutableGraph(forceGraphReinstantiation);
}

void MdGpuGraph::launchGraphMdStep(GpuEventSynchronizer* xUpdatedOnDeviceEvent)
{
    impl_->launchGraphMdStep(xUpdatedOnDeviceEvent);
}

bool MdGpuGraph::useGraphThisStep() const
{
    return impl_->useGraphThisStep();
}

bool MdGpuGraph::graphIsCapturingThisStep() const
{
    return impl_->graphIsCapturingThisStep();
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
