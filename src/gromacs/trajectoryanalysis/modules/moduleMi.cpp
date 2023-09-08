/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2013- The GROMACS Authors
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
/*
 * GROMACS WORKSHOP TASK
 */
#include "moduleMi.h"

#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"

using namespace gmx::analysismodules;

AnalysisMi::AnalysisMi() : cutoff_(0.0)
{
    ;
}

void AnalysisMi::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    static const char* const desc[] = { "This is the exercise doing by AMN (c1) for using the",
                                        "interface available in the modern GROMACS for writting",
                                        "your own analysis code!" };

    settings->setHelpText(desc);

    settings->setFlag(TrajectoryAnalysisSettings::efRequireIR);
}


void AnalysisMi::initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top)
{
    nb_.setCutoff(static_cast<real>(cutoff_));

    printf("User-int1: %d\n", top.mir()->userint1);
}


void AnalysisMi::analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata)
{
    printf("-- frame -- \n"); //
}


void AnalysisMi::finishAnalysis(int /*nframes*/) {}


void AnalysisMi::writeOutput()
{
    // We print out the average of the mean distances for each group.
    ;
}

const char AnalysisMiInfo::name[]             = "anami";
const char AnalysisMiInfo::shortDescription[] = "Calculate pretty NOTHING";

gmx::TrajectoryAnalysisModulePointer AnalysisMiInfo::create()
{
    return TrajectoryAnalysisModulePointer(new AnalysisMi);
}

/*

int main(int argc, char* argv[])
{
    return gmx::TrajectoryAnalysisCommandLineRunner::runAsMain<AnalysisMi>(argc, argv);
}

*/
