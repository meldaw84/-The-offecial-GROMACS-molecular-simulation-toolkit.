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

#include "gromacs/utility/filestream.h"
#include "gromacs/utility/loggerbuilder.h"

namespace gmx
{

namespace analysismodules
{

namespace
{


// WRITE MSM ESTIMATION STUFF HERE!
// INPUT: Trajectory with states
// Make TCM
// Make TPM
// Diagonalize
// OUTPUT: eigenvalues, eigenvectors

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
}

void MarkovModel::finishAnalysis(int nframes)
{
// RUN ANALYSIS HERE
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
