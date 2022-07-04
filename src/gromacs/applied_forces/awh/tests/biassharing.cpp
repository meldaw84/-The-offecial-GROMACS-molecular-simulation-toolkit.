/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2021- The GROMACS Authors
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
#include "gmxpre.h"

#include "gromacs/applied_forces/awh/biassharing.h"

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "thread_mpi/tmpi.h"

#include "gromacs/applied_forces/awh/correlationgrid.h"
#include "gromacs/applied_forces/awh/pointstate.h"
#include "gromacs/applied_forces/awh/tests/awh_setup.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

namespace gmx
{

namespace test
{

// This test requires thread-MPI
#if GMX_THREAD_MPI

namespace
{

//! The number of thread-MPI ranks to run this test on
const int c_numRanks = 4;

//! The number of simulations sharing the same bias
const int c_numSharingBiases = 2;

/*! \brief The actual test body, executed by each MPI rank
 *
 * Sets ups sharing and sums over sahring simulations.
 */
void parallelTestFunction(const void gmx_unused* dummy)
{
    int numRanks;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    GMX_RELEASE_ASSERT(numRanks == c_numRanks, "Expect c_numRanks thread-MPI ranks");

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    const int shareGroup = 1 + (myRank / c_numSharingBiases);

    t_commrec commRecord = { 0 };
    commRecord.nnodes    = 1;

    const std::vector<char> serializedAwhParametersPerDim = awhDimParamSerialized();
    auto              awhDimArrayRef = gmx::arrayRefFromArray(&serializedAwhParametersPerDim, 1);
    AwhTestParameters params = getAwhTestParameters(AwhHistogramGrowthType::ExponentialLinear,
                                                    AwhPotentialType::Convolved,
                                                    awhDimArrayRef,
                                                    false,
                                                    0.4,
                                                    false,
                                                    0.5,
                                                    0,
                                                    shareGroup);

    BiasSharing biasSharing(params.awhParams, commRecord, MPI_COMM_WORLD);

    EXPECT_EQ(biasSharing.numSharingSimulations(0), c_numSharingBiases);
    EXPECT_EQ(biasSharing.sharingSimulationIndex(0), myRank % c_numSharingBiases);

    const std::array<int, c_numRanks> input{ 1, 2, 4, 8 };
    std::array<int, 1>                buffer{ input[myRank] };

    biasSharing.sumOverSharingMasterRanks(buffer, 0);
    int expectedSum = 0;
    for (int i = 0; i < c_numSharingBiases; i++)
    {
        expectedSum += input[(myRank / c_numSharingBiases) * c_numSharingBiases + i];
    }
    EXPECT_EQ(buffer[0], expectedSum);
}

/*! \brief The actual test body, executed by each MPI rank when testing sharing samples
 *
 * Sets up sharing and tests that the number of samples are counted correctly
 */
void sharingSamplesFrictionTest(const void* nStepsArg)
{
    int numRanks;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    ASSERT_EQ(numRanks, c_numRanks);

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    const int shareGroup = 1 + (myRank / c_numSharingBiases);

    t_commrec commRecord        = { 0 };
    commRecord.nnodes           = 1;
    const double myRankFraction = double(myRank + 1) / numRanks;

    const std::vector<char> serializedAwhParametersPerDim = awhDimParamSerialized();
    auto                awhDimArrayRef = gmx::arrayRefFromArray(&serializedAwhParametersPerDim, 1);
    AwhTestParameters   params = getAwhTestParameters(AwhHistogramGrowthType::ExponentialLinear,
                                                    AwhPotentialType::Convolved,
                                                    awhDimArrayRef,
                                                    false,
                                                    0.4,
                                                    false,
                                                    0.5,
                                                    0,
                                                    shareGroup);
    const AwhDimParams& awhDimParams = params.awhParams.awhBiasParams()[0].dimParams()[0];

    BiasSharing biasSharing(params.awhParams, commRecord, MPI_COMM_WORLD);

    constexpr double mdTimeStep = 0.1;

    const int biasIndex = 0;
    Bias      bias(biasIndex,
              params.awhParams,
              params.awhParams.awhBiasParams()[0],
              params.dimParams,
              params.beta,
              mdTimeStep,
              &biasSharing,
              "",
              Bias::ThisRankWillDoIO::No);

    const CorrelationGrid& forceCorrelation = bias.forceCorrelationGrid();

    const int64_t exitStep = *static_cast<const int64_t*>(nStepsArg);
    /* We use a trajectory of the sum of two sines to cover the reaction
     * coordinate range in a semi-realistic way.
     */
    const double midPoint  = 0.5 * (awhDimParams.end() + awhDimParams.origin());
    const double halfWidth = 0.5 * (awhDimParams.end() - awhDimParams.origin());

    for (int step = 0; step <= exitStep; step++)
    {
        double t = step * mdTimeStep;
        double coord =
                midPoint + myRankFraction * halfWidth * (0.5 * std::sin(t) + 0.55 * std::sin(1.5 * t));

        awh_dvec coordValue    = { coord, 0, 0, 0 };
        double   potential     = 0;
        double   potentialJump = 0;
        bias.calcForceAndUpdateBias(
                coordValue, {}, {}, &potential, &potentialJump, step, step, params.awhParams.seed(), nullptr);
    }
    bias.updateBiasStateSharedCorrelationTensorTimeIntegral();
    double sumLocalNumVisits = 0, sumPointNumVisitsTot = 0, sumPointNumVisitsIteration = 0;
    std::vector<double> rankNumVisitsIteration, rankNumVisitsTot, rankLocalNumVisits,
            rankLocalFriction, rankSharedFriction;
    for (size_t pointIndex = 0; pointIndex < bias.state().points().size(); pointIndex++)
    {
        rankNumVisitsIteration.push_back(bias.state().points()[pointIndex].numVisitsIteration());
        rankNumVisitsTot.push_back(bias.state().points()[pointIndex].numVisitsTot());
        rankLocalNumVisits.push_back(bias.state().points()[pointIndex].localNumVisits());
        rankLocalFriction.push_back(
                forceCorrelation.tensors()[pointIndex].getVolumeElement(forceCorrelation.dtSample));
        rankSharedFriction.push_back(bias.state().getSharedCorrelationTensorVolumeElement(
                pointIndex, forceCorrelation.tensorSize()));
        sumPointNumVisitsIteration += bias.state().points()[pointIndex].numVisitsIteration();
        sumPointNumVisitsTot += bias.state().points()[pointIndex].numVisitsTot();
        sumLocalNumVisits += bias.state().points()[pointIndex].localNumVisits();
    }
    int nSamples                      = exitStep / params.awhParams.nstSampleCoord();
    int expectedUnaccountedNumSamples = nSamples % params.awhParams.numSamplesUpdateFreeEnergy();
    int expectedAccountedNumSamples   = nSamples - expectedUnaccountedNumSamples;
    EXPECT_EQ(sumLocalNumVisits, expectedAccountedNumSamples);
    EXPECT_EQ(sumPointNumVisitsTot, c_numSharingBiases * expectedAccountedNumSamples);
    EXPECT_EQ(sumPointNumVisitsIteration, expectedUnaccountedNumSamples);

    size_t numPoints = bias.state().points().size();

    /* There can be only one simultaneous test open, so send all the data to MPI rank 0 and run the
     * tests only on that rank. */
    if (myRank == 0)
    {
        std::vector<double> numVisitsIteration(numPoints * numRanks);
        std::vector<double> numVisitsTot(numPoints * numRanks);
        std::vector<double> localNumVisits(numPoints * numRanks);
        std::vector<double> localFriction(numPoints * numRanks);
        std::vector<double> sharedFriction(numPoints * numRanks);
        std::copy(rankNumVisitsIteration.begin(), rankNumVisitsIteration.end(), numVisitsIteration.begin());
        std::copy(rankNumVisitsTot.begin(), rankNumVisitsTot.end(), numVisitsTot.begin());
        std::copy(rankLocalNumVisits.begin(), rankLocalNumVisits.end(), localNumVisits.begin());
        std::copy(rankLocalFriction.begin(), rankLocalFriction.end(), localFriction.begin());
        std::copy(rankSharedFriction.begin(), rankSharedFriction.end(), sharedFriction.begin());
        for (int i = 1; i < numRanks; i++)
        {
            MPI_Recv(numVisitsIteration.data() + numPoints * i, numPoints, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(numVisitsTot.data() + numPoints * i, numPoints, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(localNumVisits.data() + numPoints * i, numPoints, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(localFriction.data() + numPoints * i, numPoints, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(sharedFriction.data() + numPoints * i, numPoints, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        gmx::test::TestReferenceData    data;
        gmx::test::TestReferenceChecker checker(data.rootChecker());

        constexpr int ulpTol = 100;

        checker.checkSequence(
                numVisitsIteration.begin(), numVisitsIteration.end(), "numVisitsIteration");
        checker.checkSequence(numVisitsTot.begin(), numVisitsTot.end(), "numVisitsTot");
        checker.checkSequence(localNumVisits.begin(), localNumVisits.end(), "localNumVisits");
        checker.setDefaultTolerance(relativeToleranceAsUlp(1.0, ulpTol));
        checker.checkSequence(localFriction.begin(), localFriction.end(), "localFriction");
        checker.checkSequence(sharedFriction.begin(), sharedFriction.end(), "sharedFriction");
    }
    else
    {
        MPI_Send(rankNumVisitsIteration.data(), numPoints, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(rankNumVisitsTot.data(), numPoints, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Send(rankLocalNumVisits.data(), numPoints, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        MPI_Send(rankLocalFriction.data(), numPoints, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
        MPI_Send(rankSharedFriction.data(), numPoints, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
    }
}

} // namespace

TEST(BiasSharingTest, SharingWorks)
{
    if (tMPI_Init_fn(FALSE, c_numRanks, TMPI_AFFINITY_NONE, parallelTestFunction, static_cast<const void*>(this))
        != TMPI_SUCCESS)
    {
        GMX_THROW(gmx::InternalError("Failed to spawn thread-MPI threads"));
    }
}

TEST(BiasSharingTest, SharingSamplesAndFrictionWorks)
{
    int64_t nSteps = 2002;
    int     result = tMPI_Init_fn(FALSE,
                              c_numRanks,
                              TMPI_AFFINITY_NONE,
                              sharingSamplesFrictionTest,
                              static_cast<const void*>(&nSteps));
    ASSERT_EQ(result, TMPI_SUCCESS);
}

#endif

} // namespace test
} // namespace gmx
