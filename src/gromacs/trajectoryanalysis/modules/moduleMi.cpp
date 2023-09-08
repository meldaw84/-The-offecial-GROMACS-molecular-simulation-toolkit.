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