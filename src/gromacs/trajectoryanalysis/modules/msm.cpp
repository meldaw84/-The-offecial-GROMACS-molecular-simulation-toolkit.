/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2021, by the GROMACS development team, led by
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
 * Implements gmx::analysismodules::MarkovModel.
 *
 * \author Cathrine Bergh <cathrine.bergh@gmail.com>
 * \ingroup module_trajectoryanalysis
 */

#include "gmxpre.h"
#include "msm.h"

#include "gromacs/math/multidimarray.h"
#include "gromacs/utility/filestream.h"
#include "gromacs/utility/loggerbuilder.h"

namespace gmx
{

namespace analysismodules
{

namespace
{


// WRITE MSM ESTIMATION STUFF HERE FOR NOW...
// INPUT: Trajectory with states
// Make TCM
// Make TPM
// Diagonalize
// OUTPUT: eigenvalues, eigenvectors

// WIP
void count_transitions()
{
    // Create state-assigned trajectory vector for initial tesing
    std::vector<int> discretizedTraj = {0, 1, 3, 2, 3, 3, 3, 2};
    // TODO: Take lag time as an argument
    int lag = 1;
    const int nstates = 4;

    //using static_extents = extents<3, 3>;
    //const double dynamicExtents2D extents<3,3>;

    // Initialize TCM as MultiDimArray with zeros
    // TODO: Move to initAnalysis
    // TODO: Make matrix unique?
    MultiDimArray<std::array<int, nstates*nstates>, extents<nstates, nstates>> transitionCountsMatrix = { { } };

    //Extract time-lagged trajectories
    std::vector<int> rows(discretizedTraj.begin(), discretizedTraj.end() - lag);
    std::vector<int> cols(discretizedTraj.begin() + lag, discretizedTraj.end());

    printf("\n ");

    const auto& dataView = transitionCountsMatrix.asConstView();
    const int numRows = transitionCountsMatrix.extent(0);
    const int numCols = transitionCountsMatrix.extent(1);

    // Iterate over trajectory and count transitions
    for (int i = 0; i < rows.size(); i++)
    {
      printf("row: %d ", rows[i]);
      printf("col: %d ", cols[i]);
      printf("\n");

      transitionCountsMatrix(rows[i], cols[i]) += 1;

    }

    for (int i = 0; i < numRows; i++)
    {
        printf("\n");
        for (int j=0; j < numCols; j++)
        {
            printf("%d ", dataView[i][j]);
        }
    }
}

/*
 * MarkovModel
 */

class MarkovModel : public TrajectoryAnalysisModule
{
public:
    MarkovModel();

    void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings) override;
    void optionsFinished(TrajectoryAnalysisSettings* settings) override;
    void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top) override;
    void analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata) override;

    void finishAnalysis(int nframes) override;
    void writeOutput() override;

private:
    std::unique_ptr<LoggerOwner>    loggerOwner_;
};

MarkovModel::MarkovModel()
{
    LoggerBuilder builder;
    builder.addTargetStream(gmx::MDLogger::LogLevel::Info, &gmx::TextOutputFile::standardOutput());
    builder.addTargetStream(gmx::MDLogger::LogLevel::Warning, &gmx::TextOutputFile::standardError());
    loggerOwner_ = std::make_unique<LoggerOwner>(builder.build());
}

void MarkovModel::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
}

void MarkovModel::optionsFinished(TrajectoryAnalysisSettings* settings)
{
}

void MarkovModel::initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top)
{
}

void MarkovModel::analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* /* pbc */, TrajectoryAnalysisModuleData* /*pdata*/)
{
// COLLECT TRANSITIONS INTO TPM HERE?
    printf("Doing stuff for every frame!\n");
}

void MarkovModel::finishAnalysis(int nframes)
{
// RUN ANALYSIS HERE
    count_transitions();
}

void MarkovModel::writeOutput()
{
// ALL FILE WRITING HERE
}

} // namespace

const char MsmInfo::name[]                 = "msm";
const char MsmInfo::shortDescription[]     = "Estimates MSM from TCM";

TrajectoryAnalysisModulePointer MsmInfo::create()
{
    return TrajectoryAnalysisModulePointer(new MarkovModel);
}

} // namespace analysismodules

} // namespace gmx
