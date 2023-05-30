/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2023- The GROMACS Authors
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
 * \brief
 * Implements tests for GPU pipelined PME spline and spread on 4 MPI ranks.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_ewald
 */

#include "gmxpre.h"

#include "config.h"

#include <cmath>

#include <string>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>

#include "gromacs/domdec/domdec.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/ewald/pme_coordinate_receiver_gpu.h"
#include "gromacs/ewald/pme_force_sender_gpu.h"
#include "gromacs/ewald/pme_gpu_internal.h"
#include "gromacs/ewald/pme_internal.h"
#include "gromacs/ewald/pme_pp_comm_gpu.h"
#include "gromacs/ewald/pme_pp_communication.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/math/matrix.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/state_propagator_data_gpu.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/message_string_collector.h"
#include "gromacs/utility/mpiinfo.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/unique_cptr.h"

#include "testutils/mpitest.h"
#include "testutils/refdata.h"
#include "testutils/test_device.h"
#include "testutils/test_hardware_environment.h"
#include "testutils/testasserts.h"
#include "testutils/testinit.h"

namespace gmx
{
namespace test
{
namespace
{

struct TestSystem
{
    std::vector<RVec> coordinates;
    std::vector<real> charges;
};

/*! \brief Test systems for spread pipelining */
const std::unordered_map<std::string, TestSystem> c_testSystems = {
    // Tests that a single particle on the first PP rank works
    { "1 particle", { std::vector<RVec>{ { 0.0F, 0.0F, 4.0F } }, std::vector<real>{ 1.0F } } },
    // Tests that a single particle on the third PP rank works
    { "2nd particle", { std::vector<RVec>{ { 0.0F, 0.0F, 16.0F } }, std::vector<real>{ -2.0F } } },
    // Tests that single non-overlapping particles on the first and third PP ranks works
    { "2 particles",
      { std::vector<RVec>{
                // 2 box lengths in x
                { 0.0F, 0.0F, 4.0F },
                { 0.0F, 0.0F, 16.0F },
        },
        std::vector<real>{ {
                1.0F,
                -2.0F,
        } } } },
    // Tests that single overlapping particles on the first and third PP ranks works
    { "2 overlapping particles",
      { std::vector<RVec>{
                // 2 box lengths in x
                { 0.0F, 0.0F, 2.0F },
                { 0.0F, 0.0F, 22.0F },
        },
        std::vector<real>{ {
                1.0F,
                -2.0F,
        } } } },
};

const std::map<std::string, Matrix3x3> c_inputBoxes = {
    { "rect", { { 24.0F, 0.0F, 0.0F, 0.0F, 24.0F, 0.0F, 0.0F, 0.0F, 24.0F } } },
    { "tric", { { 27.0F, 0.0F, 0.0F, 0.0F, 24.1F, 0.0F, 23.5F, 22.0F, 22.2F } } },
};

/*! \brief Convenience typedef of input parameters
 *
 * Parameters:
 * - key into map of unit cell vectors
 * - grid dimensions
 * - key into map of particle system (coordinates and charges)
 */
typedef std::tuple<std::string, IVec, std::string> SplineAndSpreadInputParameters;

//! Help GoogleTest name our test cases
std::string nameOfTest(const testing::TestParamInfo<SplineAndSpreadInputParameters>& info)
{
    std::string testName = formatString(
            "box_%s_"
            "grid_%d_%d_%d_"
            "system_%s",
            std::get<0>(info.param).c_str(),
            std::get<1>(info.param)[XX],
            std::get<1>(info.param)[YY],
            std::get<1>(info.param)[ZZ],
            std::get<2>(info.param).c_str());

    // Note that the returned names must be unique and may use only
    // alphanumeric ASCII characters. It's not supposed to contain
    // underscores (see the GoogleTest FAQ
    // why-should-test-suite-names-and-test-names-not-contain-underscore),
    // but doing so works for now, is likely to remain so, and makes
    // such test names much more readable.
    testName = replaceAll(testName, "-", "_");
    testName = replaceAll(testName, ".", "_");
    testName = replaceAll(testName, " ", "_");
    return testName;
}

//! Gather the reasons that the test cannot be run
MessageStringCollector getSkipMessagesIfNecessary(const t_inputrec& inputRec)
{
    // Note that we can't call GTEST_SKIP() from within this method,
    // because it only returns from the current function. So we
    // collect all the reasons why the test cannot run, return them
    // and skip in a higher stack frame.

    MessageStringCollector messages;
    messages.startContext("Test is being skipped because:");

    std::string errorMessage;
    messages.appendIf(!pme_gpu_supports_build(&errorMessage), errorMessage);
    messages.appendIf(!pme_gpu_supports_input(inputRec, &errorMessage), errorMessage);
    messages.appendIf(GMX_THREAD_MPI != 0 && GMX_GPU_CUDA == 0,
                      "thread-MPI build only works with CUDA");
    messages.appendIf(GMX_LIB_MPI != 0 && (GMX_GPU_CUDA == 0 && GMX_GPU_SYCL == 0),
                      "library MPI build works with CUDA or SYCL");
    messages.appendIf(GMX_GPU_CUDA != 0 && !isWorking(checkMpiCudaAwareSupport()),
                      "CUDA build requires an MPI library that supports CUDA awareness");
    messages.appendIf(
            GMX_SYCL_DPCPP != 0 && !isWorking(checkMpiZEAwareSupport()),
            "DPCPP SYCL build requires an MPI library that supports Level Zero awareness");
    messages.appendIf(GMX_SYCL_HIPSYCL != 0
                              && (GMX_HIPSYCL_HAVE_HIP_TARGET == 0 && GMX_HIPSYCL_HAVE_CUDA_TARGET == 0),
                      "HipSYCL build works with HIP and CUDA targets and matching GPU-aware MPI");
    messages.appendIf(
            GMX_SYCL_HIPSYCL != 0 && GMX_HIPSYCL_HAVE_HIP_TARGET != 0
                    && !isWorking(checkMpiHipAwareSupport()),
            "HipSYCL build with HIP target requires an MPI library that supports HIP awareness");
    messages.appendIf(
            GMX_SYCL_HIPSYCL != 0 && GMX_HIPSYCL_HAVE_CUDA_TARGET != 0
                    && !isWorking(checkMpiCudaAwareSupport()),
            "HipSYCL build with CUDA target requires an MPI library that supports CUDA awareness");
    return messages;
}

std::vector<PpRanks> makePpRanks()
{
    std::vector<PpRanks> ppRanks;
    ppRanks.push_back({ 0, -1 });
    ppRanks.push_back({ 1, -1 });
    ppRanks.push_back({ 2, -1 });
    return ppRanks;
}

/*! \brief Test fixture for testing both atom spline parameter computation and charge spreading.
 * These 2 stages of PME are tightly coupled in the code.
 */
class PipelineSplineAndSpreadTest : public ::testing::TestWithParam<SplineAndSpreadInputParameters>
{
};

//! \brief Test that spline and spread work
TEST_P(PipelineSplineAndSpreadTest, WorksWith)
{
    GMX_MPI_TEST(RequireRankCount<4>);

    // Organize the input
    const int   pmeOrder = 4;
    IVec        gridSize;
    std::string boxName, testSystemName;

    std::tie(boxName, gridSize, testSystemName) = GetParam();
    Matrix3x3                systemBox          = c_inputBoxes.at(boxName);
    const std::vector<RVec>& systemCoordinates  = c_testSystems.at(testSystemName).coordinates;
    const std::vector<real>& systemCharges      = c_testSystems.at(testSystemName).charges;
    const size_t             systemAtomCount    = systemCoordinates.size();

    // Fill and validate inputrec
    t_inputrec inputrec;
    inputrec.nkx                    = gridSize[XX];
    inputrec.nky                    = gridSize[YY];
    inputrec.nkz                    = gridSize[ZZ];
    inputrec.pme_order              = pmeOrder;
    inputrec.coulombtype            = CoulombInteractionType::Pme;
    inputrec.epsilon_r              = 1.0;
    MessageStringCollector messages = getSkipMessagesIfNecessary(inputrec);
    if (!messages.isEmpty())
    {
        GTEST_SKIP() << messages.toString();
    }

    // Get the first GPU available on each rank
    const auto& deviceList = getTestHardwareEnvironment()->getTestDeviceList();
    if (deviceList.empty())
    {
        GTEST_SKIP() << "No compatible GPUs detected";
    }
    TestDevice* testDevice = deviceList[0].get();
    testDevice->activate();

    // Make the data structure that holds the GPU buffers for coordinates
    gmx_wallcycle          wcycle;
    StatePropagatorDataGpu stateGpu(&testDevice->deviceStream(),
                                    testDevice->deviceContext(),
                                    GpuApiCallBehavior::Async,
                                    pme_gpu_get_atom_data_block_size(),
                                    &wcycle);

    // Prepare to do multiple-program multiple-data like
    // configurations with three PP and one PME-only ranks do.
    int      rank;
    MPI_Comm communicator = MPI_COMM_WORLD;
    MPI_Comm_rank(communicator, &rank);
    const int rankDoingPme = 3;
    if (rank != rankDoingPme)
    {
        // Do a simple one-dimensional domain decomposition in the
        // direction of the third box vector. First, find the domain
        // boundaries for this rank.
        const RVec xBoxVector(systemBox(0, 0), systemBox(0, 1), systemBox(0, 2));
        const RVec yBoxVector(systemBox(1, 0), systemBox(1, 1), systemBox(1, 2));
        const RVec zBoxVector(systemBox(2, 0), systemBox(2, 1), systemBox(2, 2));
        const RVec unitNormalOfPlaneOfXYBoxVectors = cross(xBoxVector, yBoxVector).unitVector();
        const real zBoxHeight = dot(zBoxVector, unitNormalOfPlaneOfXYBoxVectors);
        // rankDoingPme is the same as the number of ranks doing PP
        const real zBoxHeightPerDomain            = zBoxHeight / rankDoingPme;
        const real minimumZBoxHeightForThisDomain = zBoxHeightPerDomain * rank;
        const real maximumZBoxHeightForThisDomain = zBoxHeightPerDomain * (rank + 1);

        // Now partition the coordinates and charges
        HostVector<RVec> localCoordinates(0, { PinningPolicy::PinnedIfSupported });
        // PaddedVector does not support push_back, so we improvise by initially over-allocating
        PaddedHostVector<real> localCharges(systemCharges.size(), { PinningPolicy::PinnedIfSupported });
        auto                   currentLocalCharge  = localCharges.begin();
        auto                   currentSystemCharge = systemCharges.cbegin();
        for (const RVec& coordinate : systemCoordinates)
        {
            const real distanceToPlaneOfXYBoxVectors = dot(coordinate, unitNormalOfPlaneOfXYBoxVectors);
            if (distanceToPlaneOfXYBoxVectors >= minimumZBoxHeightForThisDomain
                && distanceToPlaneOfXYBoxVectors < maximumZBoxHeightForThisDomain)
            {
                // This particle belongs to the domain of this rank,
                // make a copy of the coordinates and charge.
                localCoordinates.push_back(coordinate);
                *currentLocalCharge++ = *currentSystemCharge;
            }
            ++currentSystemCharge;
        }
        // Fix the size of the local charges
        localCharges.resizeWithPadding(localCoordinates.size());

        // Send atom count to the PME rank
        const int numLocalAtoms = localCoordinates.size();
        MPI_Send(&numLocalAtoms, sizeof(numLocalAtoms), MPI_BYTE, rankDoingPme, eCommType_CNB, communicator);

        // Send any charges to the PME rank
        if (numLocalAtoms > 0)
        {
            MPI_Send(localCharges.data(),
                     numLocalAtoms * sizeof(real),
                     MPI_BYTE,
                     rankDoingPme,
                     eCommType_ChargeA,
                     communicator);
        }

        // Transfer the coordinates to the GPU like mdrun does.
        stateGpu.reinit(numLocalAtoms, numLocalAtoms);
        stateGpu.copyCoordinatesToGpu(localCoordinates, AtomLocality::Local, 1);
        const SimulationWorkload simulationWork;
        const StepWorkload       stepWork;
        GpuEventSynchronizer*    coordinatesReadyOnDeviceEvent =
                stateGpu.getCoordinatesReadyOnDeviceEvent(AtomLocality::Local, simulationWork, stepWork);

        // Send coordinates from the GPU to the PME rank.
        HostVector<RVec> pmeForceReceiveBuffer(numLocalAtoms, { PinningPolicy::PinnedIfSupported });
        PmePpCommGpu     pmePpCommGpu(communicator,
                                  rankDoingPme,
                                  &pmeForceReceiveBuffer,
                                  testDevice->deviceContext(),
                                  testDevice->deviceStream());
        pmePpCommGpu.reinit(numLocalAtoms);
        pmePpCommGpu.sendCoordinatesToPmeFromGpu(
                stateGpu.getCoordinates(), numLocalAtoms, coordinatesReadyOnDeviceEvent, pmeForceReceiveBuffer);
    }
    else
    {
        // Build the PME object
        std::unique_ptr<PmeGpuProgram> pmeGpuProgram = buildPmeGpuProgram(testDevice->deviceContext());
        const MDLogger                 dummyLogger;
        t_commrec                      commrec;
        const real                     ewaldCoeff_q                  = 1.0F;
        const real                     ewaldCoeff_lj                 = 1.0F;
        NumPmeDomains                  numPmeDomains                 = { 1, 1 };
        const real                     haloExtentForAtomDisplacement = 1.0;
        matrix                         legacySystemBox;
        fillLegacyMatrix(systemBox, legacySystemBox);
        PmeRunMode                              runMode = PmeRunMode::GPU;
        unique_cptr<gmx_pme_t, gmx_pme_destroy> pme(gmx_pme_init(&commrec,
                                                                 numPmeDomains,
                                                                 &inputrec,
                                                                 legacySystemBox,
                                                                 haloExtentForAtomDisplacement,
                                                                 false,
                                                                 false,
                                                                 true,
                                                                 ewaldCoeff_q,
                                                                 ewaldCoeff_lj,
                                                                 1,
                                                                 runMode,
                                                                 nullptr,
                                                                 &testDevice->deviceContext(),
                                                                 &testDevice->deviceStream(),
                                                                 pmeGpuProgram.get(),
                                                                 dummyLogger));
        // We want to run the pipeline path regardless of particle count.
        pme->gpu->minParticleCountToRecalculateSplines = 0;
        // Transfer the box to the GPU
        pme_gpu_update_input_box(pme->gpu, legacySystemBox);

        // Prepare for PP-PME communication
        std::vector<PpRanks>     ppRanks = makePpRanks();
        std::vector<MPI_Request> requests(ppRanks.size(), MPI_REQUEST_NULL);
        PaddedHostVector<real>   chargeA({ PinningPolicy::PinnedIfSupported });
        PaddedHostVector<real>   chargeB({ PinningPolicy::PinnedIfSupported });
        PmeCoordinateReceiverGpu pmeCoordinateReceiverGpu(communicator, testDevice->deviceContext(), ppRanks);
        PmeForceSenderGpu pmeForceSenderGpu(pme_gpu_get_f_ready_synchronizer(pme.get()),
                                            communicator,
                                            testDevice->deviceContext(),
                                            ppRanks);

        // Receive atom counts from PP ranks
        int messageCount = 0;
        for (auto& sender : ppRanks)
        {
            MPI_Irecv(&sender.numAtoms,
                      sizeof(sender.numAtoms),
                      MPI_BYTE,
                      sender.rankId,
                      eCommType_CNB,
                      communicator,
                      &requests[messageCount++]);
        }
        MPI_Waitall(messageCount, requests.data(), MPI_STATUSES_IGNORE);

        // Check the total atom count for sanity
        int totalNumAtoms = 0;
        for (const auto& sender : ppRanks)
        {
            totalNumAtoms += sender.numAtoms;
        }
        ASSERT_EQ(totalNumAtoms, systemAtomCount);

        // Prepare to receive charges from PP ranks
        chargeA.resizeWithPadding(totalNumAtoms);
        // Fill charges with a nonsense value that might help detect problems
        std::fill(chargeA.begin(), chargeA.end(), -12345);
        // Then receive charges on the host from ranks that should
        // send them.
        real* bufferPtr = chargeA.data();
        messageCount    = 0;
        for (const auto& sender : ppRanks)
        {
            if (sender.numAtoms > 0)
            {
                MPI_Irecv(bufferPtr,
                          sender.numAtoms * sizeof(real),
                          MPI_BYTE,
                          sender.rankId,
                          eCommType_ChargeA,
                          communicator,
                          &requests[messageCount++]);
                bufferPtr += sender.numAtoms;
            }
        }
        MPI_Waitall(messageCount, requests.data(), MPI_STATUSES_IGNORE);

        // Update sizes of GPU buffers and transfer charges to the GPU.
        stateGpu.reinit(totalNumAtoms, totalNumAtoms);
        gmx_pme_reinit_atoms(pme.get(), totalNumAtoms, chargeA, chargeB);

        // Post receives for coordinates.
        DeviceBuffer<RVec> coordinatesBuffer = stateGpu.getCoordinates();
        pme_gpu_set_device_x(pme.get(), coordinatesBuffer);
        pmeCoordinateReceiverGpu.reinitCoordinateReceiver(coordinatesBuffer);
        pmeForceSenderGpu.setForceSendBuffer(pme_gpu_get_device_f(pme.get()));
        int senderIndex = 0;
        int atomOffset  = 0;
        for (const auto& sender : ppRanks)
        {
            if (sender.numAtoms > 0)
            {
                if (GMX_THREAD_MPI)
                {
                    pmeCoordinateReceiverGpu.receiveCoordinatesSynchronizerFromPpPeerToPeer(sender.rankId);
                }
                else
                {
                    pmeCoordinateReceiverGpu.launchReceiveCoordinatesFromPpGpuAwareMpi(
                            coordinatesBuffer, atomOffset, sender.numAtoms * sizeof(rvec), sender.rankId, senderIndex);
                }
                atomOffset += sender.numAtoms;
            }
            senderIndex++;
        }

        // Then launch spread and wait for completion.
        auto       xReadyOnDevice   = nullptr;
        const real lambda_q         = 1.0;
        const bool useGpuDirectComm = true;
        pme_gpu_launch_spread(
                pme.get(), xReadyOnDevice, &wcycle, lambda_q, useGpuDirectComm, &pmeCoordinateReceiverGpu);
        testDevice->deviceStream().synchronize();

        // Now that the charges are on the PME grid, copy it back to the host.
        const int gridIndex = 0;
        real*     realGrid  = pme->fftgrid[gridIndex];
        ASSERT_NE(realGrid, nullptr);
        pme_gpu_copy_output_spread_grid(pme->gpu, realGrid, gridIndex);
        pme_gpu_sync_spread_grid(pme->gpu);

        // Convert the non-zero entries of the FFT grid to a map so
        // the reference data can be stored efficiently and (in simple
        // cases) it can be inspected for correctness.
        std::map<std::string, real> nonZeroGridValues;
        IVec                        paddedGridSize(0, 0, 0);
        pme_gpu_get_real_grid_sizes(pme->gpu, &gridSize, &paddedGridSize);
        for (int ix = 0; ix < gridSize[XX]; ix++)
        {
            for (int iy = 0; iy < gridSize[YY]; iy++)
            {
                for (int iz = 0; iz < gridSize[ZZ]; iz++)
                {
                    // Using XYZ grid ordering for real grid on GPU
                    const size_t gridValueIndex = (ix * paddedGridSize[YY] + iy) * paddedGridSize[ZZ] + iz;
                    const real value            = realGrid[gridValueIndex];
                    if (value != 0.0_real)
                    {
                        const auto key         = formatString("Cell %02d %02d %02d", ix, iy, iz);
                        nonZeroGridValues[key] = value;
                    }
                }
            }
        }

        // Construct some ulp tolerances for testing the grid contents
        const auto maxGridSize = std::max(std::max(gridSize[XX], gridSize[YY]), gridSize[ZZ]);
        // 4 is a modest estimate for amount of operations; (pmeOrder - 2) is a number of iterations;
        // maxGridSize is inverse of the smallest positive fractional coordinate (which are interpolated by the splines).
        const auto ulpToleranceSplineValues = 4 * (pmeOrder - 2) * maxGridSize;
        // 2 is empirical; sqrt(systemAtomCount) assumes all the input charges may spread onto the same cell
        const auto ulpToleranceGrid =
                2 * ulpToleranceSplineValues * int(std::ceil(std::sqrt(real(systemAtomCount))));
        SCOPED_TRACE(formatString("Testing grid values with tolerance of %d", ulpToleranceGrid));

        // Testing the results
        TestReferenceData    refData;
        TestReferenceChecker rootChecker(refData.rootChecker());
        TestReferenceChecker gridValuesChecker(
                rootChecker.checkCompound("NonZeroGridValues", "RealSpaceGrid"));
        gridValuesChecker.setDefaultTolerance(relativeToleranceAsUlp(1.0, ulpToleranceGrid));
        for (const auto& point : nonZeroGridValues)
        {
            gridValuesChecker.checkReal(point.second, point.first.c_str());
        }
    }
}

//! Moved out from test instantiations for readability
const auto c_inputBoxNames = ::testing::Values("rect", "tric");
//! Moved out from test instantiations for readability
const auto c_inputGridSizes = ::testing::Values(IVec{ 12, 12, 12 });
//! Moved out from test instantiations for readability
const auto c_inputTestSystemNames =
        ::testing::Values("1 particle", "2nd particle", "2 particles", "2 overlapping particles");

} // namespace

//! Instantiation of the tests
INSTANTIATE_TEST_SUITE_P(Pme,
                         PipelineSplineAndSpreadTest,
                         ::testing::Combine(c_inputBoxNames, c_inputGridSizes, c_inputTestSystemNames),
                         nameOfTest);

} // namespace test
} // namespace gmx
