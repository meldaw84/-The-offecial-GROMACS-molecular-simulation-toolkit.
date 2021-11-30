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
 * \brief Tests for MD GPU graph
 *
 * \author Alan Gray <alang@nvidia.com>
 */
#include "gmxpre.h"

#include "config.h"

#include <gtest/gtest.h>

#if GMX_GPU_CUDA

#    include "gromacs/gpu_utils/device_stream.h"
#    include "gromacs/gpu_utils/device_stream_manager.h"
#    include "gromacs/gpu_utils/devicebuffer.h"
#    include "gromacs/gpu_utils/gpueventsynchronizer.h"
#    include "gromacs/gpu_utils/hostallocator.h"
#    include "gromacs/mdlib/mdgraph_gpu.h"

#    include "testutils/refdata.h"
#    include "testutils/test_hardware_environment.h"
#    include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{

/*! \brief Clear all elements of buffer using a delayed mechanism via a second buffer
 *
 * \param [in] d_buffer1  Main buffer to be cleared
 * \param [in] d_buffer2  Second buffer to use in delayed clearting mechanism
 * \param [in] size       Number of elements in buffer
 * \param [in] stream     Device stream to use for operation
 */
static void delayedClearOfBuffer1onGpu(DeviceBuffer<int>   d_buffer1,
                                       DeviceBuffer<int>   d_buffer2,
                                       int                 size,
                                       const DeviceStream* stream)
{
    // clear all elements of buffer2, delaying subsequent copy
    clearDeviceBufferAsync(&d_buffer2, 0, size, *stream);
    // copy all of buffer2 to buffer1. After completion, all of buffer1 is zero
    copyBetweenDeviceBuffers(&d_buffer1, &d_buffer2, size, *stream, GpuApiCallBehavior::Async, nullptr);
}

TEST(MdGraphTest, MdGpuGraphExecutesActivities)
{

    const auto& testDevice    = getTestHardwareEnvironment()->getTestDeviceList()[0];
    const auto& deviceContext = testDevice->deviceContext();

    // Initialize required structures
    bool               havePpDomainDecomposition = false;
    SimulationWorkload simulationWork;
    simulationWork.useGpuPme    = true;
    simulationWork.useGpuUpdate = true;
    DeviceStreamManager deviceStreamManager(
            testDevice->deviceInfo(), havePpDomainDecomposition, simulationWork, false);
    ;
    GpuEventSynchronizer xReadyOnDeviceEvent;
    GpuEventSynchronizer xUpdatedOnDeviceEvent;
    gmx::MdGpuGraph      mdGpuGraph(
            deviceStreamManager, simulationWork, MPI_COMM_WORLD, MdGraphEvenOrOddStep::EvenStep, nullptr);

    // arbitrary size large enough to allow us to significantly delay one memory operation with another
    int size = 100000000;

    // Allocate 2 device buffers
    DeviceBuffer<int> d_buffer1;
    int               d_buffer1_size       = -1;
    int               d_buffer1_size_alloc = -1;
    reallocateDeviceBuffer(&d_buffer1, size, &d_buffer1_size, &d_buffer1_size_alloc, deviceContext);
    DeviceBuffer<int> d_buffer2;
    int               d_buffer2_size       = -1;
    int               d_buffer2_size_alloc = -1;
    reallocateDeviceBuffer(&d_buffer2, size, &d_buffer2_size, &d_buffer2_size_alloc, deviceContext);

    // Run test without, then with, use of graph, to check differing expected results
    for (bool useGraph : { false, true })
    {

        HostVector<int> h_one;
        changePinningPolicy(&h_one, PinningPolicy::PinnedIfSupported);
        h_one.resize(1);
        h_one.data()[0] = 1;

        HostVector<int> h_output;
        changePinningPolicy(&h_output, PinningPolicy::PinnedIfSupported);
        h_output.resize(1);

        // set first element of buffer1 as 1
        copyToDeviceBuffer(&d_buffer1,
                           h_one.data(),
                           0,
                           1,
                           deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal),
                           GpuApiCallBehavior::Sync,
                           nullptr);

        if (useGraph) // denote start of graph region
        {
            bool bNS               = false;
            bool usedGraphLastStep = true;
            mdGpuGraph.start(bNS, useGraph, usedGraphLastStep, &xReadyOnDeviceEvent);
        }

        // Asynchronously clear buffer1 in update stream, after a delay
        delayedClearOfBuffer1onGpu(
                d_buffer1, d_buffer2, size, &deviceStreamManager.stream(gmx::DeviceStreamType::UpdateAndConstraints));

        if (useGraph) // denote end of graph region
        {
            mdGpuGraph.end(&xUpdatedOnDeviceEvent);
        }

        // copy first element of buffer1 to cpu in local stream
        copyFromDeviceBuffer(h_output.data(),
                             &d_buffer1,
                             0,
                             1,
                             deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal),
                             GpuApiCallBehavior::Async,
                             nullptr);
        // wait for local stream
        deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal).synchronize();

        // Without graph, element is 1 since delayed clearing has not yet occured.
        // With graph, update stream is joined to local stream so element will be 0.
        if (mdGpuGraph.useGraphThisStep())
        {
            EXPECT_EQ(h_output.data()[0], 0);
        }
        else
        {
            EXPECT_EQ(h_output.data()[0], 1);
        }
    }
}

} // namespace
} // namespace test
} // namespace gmx

#endif // GMX_GPU_CUDA
