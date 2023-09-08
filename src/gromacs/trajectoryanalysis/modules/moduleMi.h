#ifndef GMX_ANALYSISDATA_MODULES_MODULEMI_H
#define GMX_ANALYSISDATA_MODULES_MODULEMI_H

#include <string>
#include <vector>

#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/analysisdata/modules/average.h"
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/options.h"
#include "gromacs/selection/nbsearch.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/trajectoryanalysis/analysismodule.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/cmdlinerunner.h"

namespace gmx
{
namespace analysismodules
{


class AnalysisMi : public TrajectoryAnalysisModule
{
public:
    AnalysisMi();

    virtual void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings);
    virtual void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top);

    virtual void analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata);

    virtual void finishAnalysis(int nframes);
    virtual void writeOutput();

private:
    class ModuleData;

    std::string   fnDist_;
    double        cutoff_;
    Selection     refsel_;
    SelectionList sel_;

    AnalysisNeighborhood nb_;

    AnalysisData                     data_;
    AnalysisDataAverageModulePointer avem_;
};

class AnalysisMiInfo
{
public:
    static const char                      name[];
    static const char                      shortDescription[];
    static TrajectoryAnalysisModulePointer create();
};

} // namespace analysismodules
} // namespace gmx

#endif
