/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020,2021, by the GROMACS development team, led by
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
 * \brief
 * This implements basic nblib utility tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include "config.h"

#if GMX_GPU_CUDA
#    include "nblib/gmxcalculatorgpu.h"
#endif

#include <gtest/gtest.h>

#include "nblib/kerneloptions.h"
#include "nblib/simulationstate.h"
#include "nblib/tests/testhelpers.h"
#include "nblib/tests/testsystems.h"
#include "gromacs/hardware/device_management.h"
#include "gromacs/utility/arrayref.h"

#include "testutils/test_hardware_environment.h"

namespace nblib
{
namespace test
{
namespace
{

TEST(NBlibTest, CanCreateGPUfc)
{
    const auto& testDeviceList = gmx::test::getTestHardwareEnvironment()->getTestDeviceList();
    for (const auto& testDevice : testDeviceList)
    {
        const DeviceInformation& deviceInfo = testDevice->deviceInfo();
        setActiveDevice(deviceInfo);

        SpcMethanolSimulationStateBuilder spcMethanolSimulationStateBuilder;
        SimulationState simState = spcMethanolSimulationStateBuilder.setupSimulationState();
        NBKernelOptions options;
        options.useGpu           = true;
        options.nbnxmSimd        = SimdKernels::SimdNo;
        options.coulombType      = CoulombType::Cutoff;
        EXPECT_NO_THROW(GmxNBForceCalculatorGpu gmxForceCalculator =
                                setupGmxForceCalculatorGpu(simState.topology(), options, deviceInfo));
    }
}

TEST(NBlibTest, SpcMethanolForcesAreCorrectOnGpu)
{
    const auto& testDeviceList = gmx::test::getTestHardwareEnvironment()->getTestDeviceList();
    for (const auto& testDevice : testDeviceList)
    {
        const DeviceInformation& deviceInfo = testDevice->deviceInfo();
        setActiveDevice(deviceInfo);

        SpcMethanolSimulationStateBuilder spcMethanolSimulationStateBuilder;
        SimulationState simState = spcMethanolSimulationStateBuilder.setupSimulationState();
        NBKernelOptions options;
        options.coulombType      = CoulombType::Cutoff;
        auto gmxForceCalculator = setupGmxForceCalculatorGpu(simState.topology(), options, deviceInfo);
        gmxForceCalculator.updatePairlist(simState.coordinates(), simState.box());

        gmx::ArrayRef<Vec3> forces(simState.forces());
        ASSERT_NO_THROW(gmxForceCalculator.compute(simState.coordinates(), simState.box(), forces));

        RefDataChecker forcesOutputTest;
        forcesOutputTest.testArrays<Vec3>(forces, "SPC-methanol forces on GPU");
    }
}

/*! \brief reorder + undoReorder test
 *
 * We do    defaultOrder -> nbnxmOrderXQ -> nbnxmOrderX -> defaultOrderRecovered
 *                       |                              |
 *                   (reorder)                     (undoReorder)
 *
 * and then check that   defaultOrder == defaultOrderRecovered
 */
TEST(NBlibTest, ReorderIsInvertible)
{
    const auto& testDeviceList = gmx::test::getTestHardwareEnvironment()->getTestDeviceList();
    for (const auto& testDevice : testDeviceList)
    {
        const DeviceInformation& deviceInfo = testDevice->deviceInfo();
        setActiveDevice(deviceInfo);

        SpcMethanolSimulationStateBuilder spcMethanolSimulationStateBuilder;
        SimulationState simState = spcMethanolSimulationStateBuilder.setupSimulationState();

        NBKernelOptions options = NBKernelOptions();
        options.coulombType     = CoulombType::Cutoff;
        auto gmxForceCalculator = setupGmxForceCalculatorGpu(simState.topology(), options, deviceInfo);
        gmxForceCalculator.updatePairlist(simState.coordinates(), simState.box());

        int numParticles = simState.topology().numParticles();

        // default order is just a sequence 0, 1, ..., 3*numParticles
        std::vector<Vec3> defaultOrder(numParticles);
        for (int i = 0; i < numParticles; ++i)
        {
            defaultOrder[i] = { real(3 * i + 0), real(3 * i + 1), real(3 * i + 2) };
        }

        std::vector<real> nbnxmOrderXQ(4 * numParticles);
        gmxForceCalculator.reorder(defaultOrder, nbnxmOrderXQ);

        // throw away Q
        std::vector<gmx::RVec> nbnxmOrderX(numParticles);
        for (int i = 0; i < numParticles; ++i)
        {
            nbnxmOrderX[i] = { nbnxmOrderXQ[4 * i], nbnxmOrderXQ[4 * i + 1], nbnxmOrderXQ[4 * i + 2] };
        }

        std::vector<gmx::RVec> defaultOrderRecovered(numParticles, { -1, -1, -1 });
        gmxForceCalculator.undoReorder(nbnxmOrderX, defaultOrderRecovered);

        // original defaultOrder should be identical to defaultOrderRecovered
        for (int i = 0; i < numParticles; ++i)
        {
            EXPECT_EQ(defaultOrder[i][0], defaultOrderRecovered[i][0]);
            EXPECT_EQ(defaultOrder[i][1], defaultOrderRecovered[i][1]);
            EXPECT_EQ(defaultOrder[i][2], defaultOrderRecovered[i][2]);
        }
    }
}

/*! \brief test the DeviceBuffer compute interface
 *
 * We do:   X -> (reorder) -> XqNbnxm -> (copyToDevice) -> deviceXq -> (compute) -> deviceForces
 *          -> (copyFromDevice) -> forcesNbnxm -> (undoReorder) -> forcesDefaultOrder
 *
 * Then we test that forcesDefaultOrder == forces from CPU buffer interface of the gpu caluculator
 * which are tested against reference values.
 *
 * Note that the reference values are also coming from a gpu force calculator.
 */
TEST(NBlibTest, SpcMethanolForcesDeviceInterface)
{
    const auto& testDeviceList = gmx::test::getTestHardwareEnvironment()->getTestDeviceList();
    for (const auto& testDevice : testDeviceList)
    {
        const DeviceInformation& deviceInfo = testDevice->deviceInfo();
        setActiveDevice(deviceInfo);

        SpcMethanolSimulationStateBuilder spcMethanolSimulationStateBuilder;
        SimulationState simState = spcMethanolSimulationStateBuilder.setupSimulationState();
        NBKernelOptions options;
        options.coulombType      = CoulombType::Cutoff;
        auto forceCalculator = setupGmxForceCalculatorGpu(simState.topology(), options, deviceInfo);
        forceCalculator.updatePairlist(simState.coordinates(), simState.box());

        std::vector<Vec3> forcesReference(simState.forces().size());
        std::copy(simState.forces().begin(), simState.forces().end(), forcesReference.begin());

        // we know these forces to be correct from SpcMethanolForcesAreCorrectOnGpu test
        forceCalculator.compute(simState.coordinates(), simState.box(), forcesReference);

        int                  numParticles = simState.topology().numParticles();
        const DeviceContext& context      = forceCalculator.deviceContext();

        // reorder coordinates into nbnxm ordering on the CPU
        std::vector<real> coordinatesNbnxm(4 * numParticles);
        forceCalculator.reorder(simState.coordinates(), coordinatesNbnxm);

        // device coordinates
        DeviceBuffer<Float4> deviceXYZQ;
        allocateDeviceBuffer(&deviceXYZQ, numParticles, context);
        copyToDeviceBuffer(&deviceXYZQ,
                           reinterpret_cast<const Float4*>(coordinatesNbnxm.data()),
                           0,
                           numParticles,
                           forceCalculator.deviceStream(),
                           GpuApiCallBehavior::Sync,
                           nullptr);

        DeviceBuffer<Float3> deviceForces;
        allocateDeviceBuffer(&deviceForces, numParticles, context);
        clearDeviceBufferAsync(&deviceForces, 0, numParticles, forceCalculator.deviceStream());

        // launch compute directly with device buffers
        forceCalculator.compute(deviceXYZQ, simState.box(), deviceForces);

        // download forces from the GPU
        std::vector<gmx::RVec> forcesNbnxm(numParticles, { 0, 0, 0 });
        copyFromDeviceBuffer(reinterpret_cast<Float3*>(forcesNbnxm.data()),
                             &deviceForces,
                             0,
                             numParticles,
                             forceCalculator.deviceStream(),
                             GpuApiCallBehavior::Sync,
                             nullptr);

        // reorder downloaded forces from nbnxm into default ordering
        std::vector<gmx::RVec> forcesDefaultOrder(numParticles, gmx::RVec{ 0, 0, 0 });
        forceCalculator.undoReorder(forcesNbnxm, forcesDefaultOrder);

        // forcesDefaultOrder should be equal to the reference
        for (int i = 0; i < numParticles; ++i)
        {
            EXPECT_EQ(forcesDefaultOrder[i][0], forcesReference[i][0]);
            EXPECT_EQ(forcesDefaultOrder[i][1], forcesReference[i][1]);
            EXPECT_EQ(forcesDefaultOrder[i][2], forcesReference[i][2]);
        }

        freeDeviceBuffer(&deviceXYZQ);
        freeDeviceBuffer(&deviceForces);
    }
}

} // namespace
} // namespace test
} // namespace nblib
