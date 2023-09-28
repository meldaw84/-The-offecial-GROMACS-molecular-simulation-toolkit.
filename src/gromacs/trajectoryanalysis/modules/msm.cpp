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
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/analysisdata/paralleloptions.h"
#include "gromacs/math/multidimarray.h"
#include "gromacs/msm/msm_estimation.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/utility/filestream.h"
#include "gromacs/utility/loggerbuilder.h"

// For clustering, move later?
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

    std::optional<t_cluster_ndx> clusterIndex_;
    AnalysisData freeEnergies_;

    // TODO: take nstates as input
    // TODO: we want to instantiate MSM after parsing the clustering data
    //MarkovModel* msmCallback;
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

    // Same as in Paul's extract-clusters
    options->addOption(FileNameOption("clusters")
    // TODO: change Index to AtomIndex to avoid shadowing (see extract-clusters)
                               .filetype(OptionFileType::Index)
                               .inputFile()
                               .required()
                               .store(&clusterIndexFileName_)
                               .defaultBasename("cluster")
                               .description("Name of index file containing frame indices for each "
                                            "cluster, obtained from gmx cluster -clndx."));

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
    // TODO: parse out nstates from data

    // TODO: More of this should be taken as input later on
    //int nstates = 4;
    //int lag = 2;
    //MarkovModel msm = MarkovModel();
}

// Initialize analysis
// TODO: create MSM here?
void MarkovModelModule::initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top)
{
    plotSettings_ = settings.plotSettings();
}

// TODO: Call the state assignment here?
// TODO: Call transition counting here?
void MarkovModelModule::analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* /* pbc */, TrajectoryAnalysisModuleData* /*pdata*/)
{
    printf("Doing stuff for every frame!\n");
}

void MarkovModelModule::finishAnalysis(int nframes)
{
}

void MarkovModelModule::writeOutput()
{
    int nstates = 4;
    int lag = 2;
    std::vector<int> traj = {0, 0, 0, 0, 0, 3, 3, 2};
    //MarkovModel msm = MarkovModel(nstates);
    MarkovModel msm = MarkovModel();
    msm.initializeMarkovModel(nstates);
    msm.assignStatesToFrames();
    msm.countTransitions(traj, lag);
    msm.computeTransitionProbabilities();
    auto tpm = msm.transitionProbabilityMatrix;
    msm.diagonalizeMatrix(tpm);
    auto freeEnergies = msm.getStationaryDistributionFromEigenvector(TRUE);

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
    for (int i = 0; i < nstates; ++i)
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
