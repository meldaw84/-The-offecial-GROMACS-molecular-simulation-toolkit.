/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2022- The GROMACS Authors
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

#include "lie.h"

#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/analysisdata/paralleloptions.h"
#include "gromacs/commandline/viewit.h"
#include "gromacs/energyanalysis/energytermcontainer.h"
#include "gromacs/trajectory/energyframe.h"
#include "gromacs/utility/real.h"

namespace gmx
{

namespace
{

class Lie : public IEnergyAnalysis
{
public:
    Lie();
    void initOptions(IOptionsContainer* options, ICommandLineOptionsModuleSettings* settings) override;
    void initAnalysis(ArrayRef<const EnergyNameUnit> eNU, const gmx_output_env_t* oenv) override;
    void analyzeFrame(t_enxframe* fr, const gmx_output_env_t* oenv) override;
    void finalizeAnalysis(const gmx_output_env_t* oenv) override;
    void viewOutput(const gmx_output_env_t* oenv) override;

private:
    //! Output for the computer LIE energy as a function of time
    std::string outputXvgPath_;
    
    //! Mean Lennard-Jones interaction value between ligand and solvent
    real lie_lj_ = 0;
    //! Mean Coulomb interaction value between ligand and solvent
    real lie_qq_ = 0;
    //! LIE equation factor/weight for Lennard-Jones interaction
    real fac_lj_ = 0.181;
    //! LIE equation factor/weight for Coulomb interaction
    real fac_qq_ = 0.5;
    
    //! Name of the ligand 
    std::string ligandName_;
    
    //! Storage for the Coulomb energy term between the ligand and other energy groups
    EnergyTermContainer qqTerms_;
    
    //! Storage for the Lennard-Jones energy term between the ligand and other energy groups
    EnergyTermContainer ljTerms_;
    
    //! Storage for the computed LIE
    EnergyTerm lie_;
    
    //! Dataset for LIE values over the energy file
    AnalysisData                             lieOverTime_;
    
    //! Handle for the AnalysisData object (EnergyAnalysis is not multithreaded so no complication)
    AnalysisDataHandle lieOverTimeHandle_;
    
};

Lie::Lie () :
    // TODO: constant for gromacs energy unit?
    lie_(0, true, "Linear interaction energy", "kJ/mol")
{
    
}

void gmx::Lie::initOptions(IOptionsContainer* options, ICommandLineOptionsModuleSettings* settings)
{
    static const char* const desc[] = {
        "[THISMODULE] computes a free energy estimate based on an energy analysis",
        "from nonbonded energies. One needs an energy file with the following components:",
        "Coul-(A-B) LJ-SR (A-B) etc.[PAR]",
        "To utilize [TT]g_lie[tt] correctly, two simulations are required: one with the",
        "molecule of interest bound to its receptor and one with the molecule in water.",
        "Both need to utilize [TT]energygrps[tt] such that Coul-SR(A-B), LJ-SR(A-B), etc. terms",
        "are written to the [REF].edr[ref] file. Values from the molecule-in-water simulation",
        "are necessary for supplying suitable values for -Elj and -Eqq."
    };
    
    settings->setHelpText(desc);
    
    options->addOption(FileNameOption("o")
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .store(&outputXvgPath_)
                               .defaultBasename("lie")
                               .description("Computed Linear Interaction Energy (LIE) as function of time"));
    
    options->addOption(
            RealOption("Elj").store(&lie_lj_).required().description("Lennard-Jones interaction between ligand and solvent"));
    options->addOption(
            RealOption("Eqq").store(&lie_qq_).required().description("Coulomb interaction between ligand and solvent"));
    options->addOption(
            RealOption("Clj").store(&fac_lj_).description("Factor in the LIE equation for the Lennard-Jones component of the energy"));
    options->addOption(
            RealOption("Cqq").store(&fac_qq_).description("Factor in the LIE equation for the Coulomb component of the energy"));
    
    options->addOption(
            StringOption("ligand").store(&ligandName_).description("Name of the ligand in the energy file"));
    
    
}

void gmx::Lie::initAnalysis(ArrayRef<const EnergyNameUnit> eNU, const gmx_output_env_t* oenv) {
    
    
    lieOverTime_.setDataSetCount(1);
    lieOverTime_.setColumnCount(0, 1);
    if (!outputXvgPath_.empty())
    {
        AnalysisDataPlotModulePointer plotm(new AnalysisDataPlotModule());
        plotm->setFileName(outputXvgPath_);
        plotm->setTitle("LIE free energy estimate");
        plotm->setXAxisIsTime();
        plotm->setYLabel("DeltaG_bind (kJ/mol)");
        lieOverTime_.addModule(plotm);
    }
    
    // parallelizationFactor = 1, no MT
    lieOverTimeHandle_ = lieOverTime_.startData(AnalysisDataParallelOptions(1));
    
    std::string selfInteractionEnergy = ligandName_ + "-" + ligandName_;
    
    // Search the energy terms names for the ones that include the provided ligand name
    // but are not self interaction. Of those, store indices for Coulomb and LJ terms separately.
    for (int i = 0; i < gmx::ssize(eNU); i++)
    {
        std::string currentName = eNU[i].energyName;
        if(currentName.find(ligandName_) != std::string::npos
            && currentName.find(selfInteractionEnergy) == std::string::npos) {
            EnergyTerm term(i, true, eNU[i].energyName, eNU[i].energyUnit);
            if(currentName.find("LJ") != std::string::npos) {
                ljTerms_.addEnergyTerm(term);
            }
            if(currentName.find("Coul") != std::string::npos) {
                qqTerms_.addEnergyTerm(term);
            }
        }
    }
    
    printf("Using the following energy terms:\n");
    printf("LJ:  ");
    for(const auto& term: ljTerms_) {
        printf("  %s", term.name().c_str());
    }
    printf("\nCoul:");
    for(const auto& term: qqTerms_) {
        printf("  %s", term.name().c_str());
    }
    printf("\n");
}


void gmx::Lie::analyzeFrame(t_enxframe* fr, const gmx_output_env_t* oenv) {
    qqTerms_.addFrame(fr);
    ljTerms_.addFrame(fr);
    
    // Compute the sum of the term of interest for the current frame
    real sum_lj = 0;
    for(const auto& term: ljTerms_) {
        // TODO: add .back() method
        sum_lj += (term.end() - 1)->energy();
    }
    
    real sum_qq = 0;
    for(const auto& term: qqTerms_) {
        // TODO: add .back() method
        sum_qq += (term.end() - 1)->energy();
    }
    
    real lieForFrame =  fac_lj_ * (sum_lj - lie_lj_) + fac_qq_ * (sum_qq - lie_qq_);
    
    lieOverTimeHandle_.startFrame(fr->step, fr->t);
    lieOverTimeHandle_.setPoint(0, lieForFrame, true);
    lieOverTimeHandle_.finishFrame();
    
    lie_.addFrame(fr->t,
                  fr->step,
                  // Zero so that the EnergyTerm does the average/variance/... itself
                  0,
                  0,
                  0,
                  lieForFrame);
}

void gmx::Lie::finalizeAnalysis(const gmx_output_env_t* oenv) {
    
    if (lie_.numFrames() > 0)
    {
        printf("DeltaG_bind = %.3f (std. dev. %.3f)\n",
               lie_.average() / lie_.numFrames(), lie_.standardDeviation());
    }
    lieOverTime_.finishData(lieOverTimeHandle_);
    
}

void gmx::Lie::viewOutput(const gmx_output_env_t* oenv) {
    do_view(oenv,  outputXvgPath_.c_str(), "-nxy");
}

} // namespace

namespace analysismodules
{

const char LieInfo::name[]             = "lie";
const char LieInfo::shortDescription[] = "Calculate distances between pairs of positions";

IEnergyAnalysisPointer LieInfo::create()
{
    return IEnergyAnalysisPointer(new Lie);
}

} // namespace analysismodules

} // namespace gmx