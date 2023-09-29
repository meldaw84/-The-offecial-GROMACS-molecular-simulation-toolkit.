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

#include <numeric>

#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/analysisdata/paralleloptions.h"
#include "gromacs/math/multidimarray.h"
#include "gromacs/msm/msm_estimation.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/utility/filestream.h"
#include "gromacs/utility/loggerbuilder.h"
#include "gromacs/topology/index.h"


namespace gmx
{

namespace analysismodules
{

namespace
{


/*
 * MarkovModel
 */

class MarkovModelModule : public TrajectoryAnalysisModule
{
public:
    MarkovModelModule();

    void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings) override;
    void optionsFinished(TrajectoryAnalysisSettings* settings) override;
    void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top) override;
    void analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata) override;
    void finishAnalysis(int nframes) override;
    void writeOutput() override;

private:
    std::unique_ptr<LoggerOwner> loggerOwner_;
    AnalysisDataPlotSettings plotSettings_;

    std::string freeEnergyDataFileName_;
    std::string clusterIndexFileName_;

    int nstates_;
    int lag_;
    bool bLagSet_;
    std::optional<t_cluster_ndx> clusterIndex_;
    AnalysisData freeEnergies_;

    MarkovModel msm;
};

MarkovModelModule::MarkovModelModule()
{
    LoggerBuilder builder;
    builder.addTargetStream(gmx::MDLogger::LogLevel::Info, &gmx::TextOutputFile::standardOutput());
    builder.addTargetStream(gmx::MDLogger::LogLevel::Warning, &gmx::TextOutputFile::standardError());
    loggerOwner_ = std::make_unique<LoggerOwner>(builder.build());
}

// Create selelction variables here and they will be parsed
void MarkovModelModule::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    // TODO: Write documentation!
    static const char* const desc[] = {
        "[THISMODULE] shall be described here!"
    };

    settings->setHelpText(desc);

    options->addOption(FileNameOption("clusters")
    // TODO: change Index to AtomIndex to avoid shadowing (see extract-clusters)
                               .filetype(OptionFileType::Index)
                               .inputFile()
                               .required()
                               .store(&clusterIndexFileName_)
                               .defaultBasename("cluster")
                               .description("Name of index file containing frame indices for each "
                                            "cluster, obtained from gmx cluster -clndx."));

    // TODO: lag time should be set as time-based and not index-based later on
    options->addOption(IntegerOption("lag")
                                .store(&lag_)
                                .storeIsSet(&bLagSet_)
                                .required()
                                .defaultValue(1)
                                .description("Lag time (index-based)"));

    // TODO: add option to return stationary distribution?
    options->addOption(FileNameOption("energy")
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .store(&freeEnergyDataFileName_)
                               .defaultBasename("energy")
                               .description("Free energies for each microstate"));

    // TODO: implement option to output TPM/TCM
    // TODO: implement option to output transition rates
    // TODO: implement option to output timescales
    // TODO: implement option to output ITS test
}

// After parsing (command-line or files but not interactive), adjust selections
// The interactive prompt will be shown after this function returns
void MarkovModelModule::optionsFinished(TrajectoryAnalysisSettings* settings)
{
    clusterIndex_ = cluster_index(nullptr, clusterIndexFileName_.c_str());
    // TODO: we need to return an error each time we try to count transitions
    // between states not between 0 and nstates (-1 in particular, which
    // show up if we use the skip flag).

    nstates_ = gmx::ssize(clusterIndex_->clusters);

    for (int i = 0; i < gmx::ssize(clusterIndex_->inv_clust); ++i)
    {
        printf("Val: %d\n", clusterIndex_->inv_clust[i]);
    }

    msm.initializeMarkovModel(nstates_);
}

// Initialize analysis
void MarkovModelModule::initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top)
{
    plotSettings_ = settings.plotSettings();
}

// TODO: Call the state assignment here?
// TODO: Call transition counting here? Might be more efficient...
void MarkovModelModule::analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* /* pbc */, TrajectoryAnalysisModuleData* /*pdata*/)
{
    //printf("Doing stuff for every frame!\n");

    // TODO: Do we really need anything here?
    // * We need frames to project onto free energy landscape
    // * XTC files shouldn't be mandatory, though...
}

void MarkovModelModule::finishAnalysis(int nframes)
{
}

void MarkovModelModule::writeOutput()
{
    //std::vector<int> traj = {0, 0, 0, 0, 0, 3, 3, 2};
    //std::vector<int> traj = {0, 2, 0, 0, 0, 1, 1, 2};
    // TODO: check that this works with skip!
    msm.countTransitions(clusterIndex_->inv_clust, lag_);
    msm.computeTransitionProbabilities();
    auto tpm = msm.transitionProbabilityMatrix;
    msm.diagonalizeMatrix(tpm);
    auto freeEnergies = msm.getStationaryDistributionFromEigenvector(TRUE);

    // ONLY FOR DEBUG
    auto tcm = msm.transitionCountsMatrix;
    const auto& dataView = tcm.asConstView();
    const int numRows = tcm.extent(0);
    const int numCols = tcm.extent(1);

    printf("Transition Counts Matrix: \n");
    for (int i = 0; i < numRows; i++)
    {
        printf("\n");
        for (int j=0; j < numCols; j++)
        {
            printf("%d ", dataView[i][j]);
        }
    }
    printf("\n");

    const auto& dataView2 = tpm.asConstView();
    printf("Transition Probability Matrix: \n");
    for (int i = 0; i < numRows; i++)
    {
        printf("\n");
        for (int j=0; j < numCols; j++)
        {
            printf("%f ", dataView2[i][j]);
        }
    }
    printf("\n");

    // Write data relevant for microstates
    registerAnalysisDataset(&freeEnergies_, "Free Energies of Microstates");

    AnalysisDataPlotModulePointer plotm(new AnalysisDataPlotModule(plotSettings_));
    plotm->setFileName(freeEnergyDataFileName_);
    plotm->setTitle("Free Energies");
    plotm->setXLabel("Microstate index");
    // TODO: Fix Delta G here
    plotm->setYLabel("D G (kT)");

    freeEnergies_.addModule(plotm);
    freeEnergies_.setDataSetCount(1);
    freeEnergies_.setColumnCount(0, 1);

    AnalysisDataHandle dh = freeEnergies_.startData({});

    // TODO: decide whether to have zero or one indexing for states
    // 1-indexing agrees with Paul's clustering method

    // TODO: how to handle infinite free energies?
    for (int i = 0; i < nstates_; ++i)
    {
        dh.startFrame(i, i + 1);
        dh.setPoint(0, freeEnergies[i]);
        dh.finishFrame();
    }
    dh.finishData();

}

} // namespace

const char MarkovModelInfo::name[]                 = "msm";
const char MarkovModelInfo::shortDescription[]     = "Estimates MSM from TCM";

TrajectoryAnalysisModulePointer MarkovModelInfo::create()
{
    return TrajectoryAnalysisModulePointer(new MarkovModelModule);
}

} // namespace analysismodules

} // namespace gmx
