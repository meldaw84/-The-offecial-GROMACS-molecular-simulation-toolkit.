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
 * \brief Declares the MD Graph class
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 *
 * \ingroup module_mdlib
 */
#ifndef GMX_MDRUN_MDGRAPH_IMPL_H
#define GMX_MDRUN_MDGRAPH_IMPL_H

#include "gromacs/gpu_utils/device_stream_manager.h"
#include "gromacs/utility/classhelpers.h"

#include "mdgraph_gpu.h"

namespace gmx
{

class MdGpuGraph::Impl
{
public:
    /*! \brief Create MD graph object
     * \param [in] deviceStreamManager  Device stream manager object
     * \param [in] simulationWork       Simulation workload structure
     * \param [in] evenOrOddStep        Whether this graph corresponds to even or odd step
     * \param [in] mpi_comm             MPI communicator for PP domain decomposition
     * \param [in] wcycle               Wall cycle timer object
     */
    Impl(const DeviceStreamManager& deviceStreamManager,
         SimulationWorkload         simulationWork,
         MPI_Comm                   mpi_comm,
         MdGraphEvenOrOddStep       evenOrOddStep,
         gmx_wallcycle*             wcycle);
    // NOLINTNEXTLINE(performance-trivially-destructible)
    ~Impl();

    /*! \brief Reset graph */
    void reset();

    /*! \brief Denote start of graph region
     * \param [in] bNS                   Whether this is a search step
     * \param [in] canUseGraphThisStep   Whether graph can be used this step
     * \param [in] usedGraphLastStep     Whether graph was used in the last step
     * \param [in] xReadyOnDeviceEvent   Event marked when coordinates are ready on device
     */
    void start(bool bNS, bool canUseGraphThisStep, bool usedGraphLastStep, GpuEventSynchronizer* xReadyOnDeviceEvent);

    /*! \brief Denote end of graph region
     * \param [inout] xUpdatedOnDeviceEvent  Event marked when coordinates have been updated on device
     */
    void end(GpuEventSynchronizer* xUpdatedOnDeviceEvent);

    /*! \brief Whether graph is in use this step */
    bool useGraphThisStep() const { return useGraphThisStep_; }

    /*! \brief Whether graph is capturing */
    bool graphIsCapturing() const { return graphIsCapturing_; }

    /*! \brief Set PP task completion event for graph on alternate step */
    void setAlternateStepPpTaskCompletionEvent(GpuEventSynchronizer* event);

    /*! \brief Getter for task completion event for this graph
     * \returns ppTaskCompletionEvent_
     */
    GpuEventSynchronizer* getPpTaskCompletionEvent();

private:
    /*! \brief Collective operation to enqueue events from all PP ranks to a stream on PP rank 0
     * \param [in] event   Event to enqueue, valid on all PP ranks
     * \param [in] stream  Stream to enqueue events, valid on PP rank 0
     */
    void enqueueEventFromAllPpRanksToRank0Stream(GpuEventSynchronizer* event, const DeviceStream& stream);


    /*! \brief Collective operation to enqueue an event from PP rank 0 to streams on all PP ranks
     * \param [in] event   Event to enqueue, valid on PP rank 0
     * \param [in] stream  Stream to enqueue events, valid on all PP ranks
     */
    void enqueueRank0EventToAllPpStreams(GpuEventSynchronizer* event, const DeviceStream& stream);

    //! Captured graph object
    cudaGraph_t graph_;
    //! Instantiated graph object
    cudaGraphExec_t instance_;
    //! Whether existing graph should be updated rather than re-instantiated
    bool updateGraph_ = false;
    //! Whether graph has already been created
    bool graphCreated_ = false;
    //! Whether graph is capturing in this step
    bool graphIsCapturing_ = false;
    //! Whether graph should be used this step
    bool useGraphThisStep_ = false;
    //! Device stream manager object
    const DeviceStreamManager& deviceStreamManager_;
    //! Simulation workload structure
    SimulationWorkload simulationWork_;
    //! Whether PP domain decomposition is in use
    bool havePPDomainDecomposition_;
    //! MPI communicator for PP domain decomposition
    MPI_Comm mpi_comm_;
    //! PP Rank for this MD graph object
    int ppRank_ = 0;
    //! Number of PP ranks in use
    int ppSize_ = 1;
    //! CUDA status object
    cudaError_t stat_;
    //! Temporary event used for forking and joining streams in graph
    std::unique_ptr<GpuEventSynchronizer> tmpEvent_;
    //! Whether step is even or odd, where different graphs are used for each
    MdGraphEvenOrOddStep evenOrOddStep_;
    //! event marked on this step when this PP task has completed its tasks
    std::unique_ptr<GpuEventSynchronizer> ppTaskCompletionEvent_;
    //! event marked on alternate step when PP task has completed its tasks
    GpuEventSynchronizer* alternateStepPpTaskCompletionEvent_;
    //! Wall cycle timer object
    gmx_wallcycle* wcycle_;
};

} // namespace gmx
#endif
