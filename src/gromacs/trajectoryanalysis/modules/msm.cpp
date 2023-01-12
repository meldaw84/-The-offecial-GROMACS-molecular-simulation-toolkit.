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
    std::unique_ptr<LoggerOwner>    loggerOwner_;
    AnalysisDataPlotSettings  plotSettings_;

    std::string microstateDataFileName_;

    AnalysisData microstates_;
};

MarkovModelModule::MarkovModelModule()
{
    LoggerBuilder builder;
    builder.addTargetStream(gmx::MDLogger::LogLevel::Info, &gmx::TextOutputFile::standardOutput());
    builder.addTargetStream(gmx::MDLogger::LogLevel::Warning, &gmx::TextOutputFile::standardError());
    loggerOwner_ = std::make_unique<LoggerOwner>(builder.build());
}

void MarkovModelModule::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    // TODO: Write documentation!
    static const char* const desc[] = {
        "[THISMODULE] shall be described here!"
    };

    settings->setHelpText(desc);

    options->addOption(FileNameOption("microfe")
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .store(&microstateDataFileName_)
                               .defaultBasename("microfe")
                               .description("Free energies for each microstate"));
}

void MarkovModelModule::optionsFinished(TrajectoryAnalysisSettings* settings)
{
}

void MarkovModelModule::initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top)
{
    plotSettings_ = settings.plotSettings();
}

void MarkovModelModule::analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* /* pbc */, TrajectoryAnalysisModuleData* /*pdata*/)
{
    printf("Doing stuff for every frame!\n");
}

void MarkovModelModule::finishAnalysis(int nframes)
{
}

void MarkovModelModule::writeOutput()
{

    MarkovModel msm = MarkovModel(4);

    // Write data relevant for microstates
    registerAnalysisDataset(&microstates_, "Microstate IDs");

    AnalysisDataPlotModulePointer plotm(new AnalysisDataPlotModule(plotSettings_));
    plotm->setFileName(microstateDataFileName_);
    plotm->setTitle("Free Energies");

    microstates_.addModule(plotm);
    microstates_.setDataSetCount(1);
    microstates_.setColumnCount(0, 1);

    AnalysisDataHandle dh = microstates_.startData({});

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
