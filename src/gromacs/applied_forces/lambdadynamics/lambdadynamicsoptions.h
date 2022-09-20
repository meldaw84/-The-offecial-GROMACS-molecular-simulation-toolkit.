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
/*! \internal \file
 * \brief
 * Declares options for Lambda Dynamics
 * LambdaDynamicsOptions class responsible for all parameters set up 
 * during pre-processing also modificatios of topology would be done here
 *
 * \author Pavel Buslaev <pavel.i.buslaev@jyu.fi>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_LAMBDADYNAMCSOPTIONS_H
#define GMX_APPLIED_FORCES_LAMBDADYNAMCSOPTIONS_H

#include <string>
#include <vector>

#include "gromacs/utility/real.h"
#include "gromacs/mdtypes/imdpoptionprovider.h"
#include "gromacs/utility/basedefinitions.h"

struct gmx_mtop_t;
struct warninp;

namespace gmx
{

class IndexGroupsAndNames;
class KeyValueTreeObject;
class KeyValueTreeBuilder;
class MDLogger;
struct MdRunInputFilename;
struct CoordinatesAndBoxPreprocessed;
struct LambdaDynamicsInputFileName;

/* Must correspond to strings in lambdadynamicsoptions.cpp */
enum class Directive : int
{
    d_lambdaDynamicsResidues,
    d_residue,
    d_state,
    d_atoms,
    d_end_atoms,
    d_parameters,
    d_end_parameters,
    d_end_state,
    d_end_residue,
    d_end,
    d_invalid,
    Count
};

struct GroupParameters
{
    //! pKa value of titratable group
    real pKa_ = 0.;
    //! dVdL coefficients for the state
    std::vector<real> dvdlCoefs_;
};

struct GroupState
{
    //! List of atom names
    std::vector<std::string> atomNames_;
    //! List of atom types
    std::vector<std::string> atomTypes_;
    //! List of atom charges
    std::vector<real> atomCharges_;
    //! State parameters
    GroupParameters groupParameters_;
};

struct LambdaDynamicsGroupType
{
    //! Name of residue
    std::string name_ = "";
    //! Number of states in the group
    int nStates_ = 0;
    //! Type of the group: single (0) or multi (1) site
    int type_ = 0;
    //! List of atom names
    std::vector<std::string> atomNames_;
    //! List of group states
    std::vector<GroupState> groupStates_;
};

struct LambdaDynamicsAtomSet
{
    //! Name of residue type to use
    std::string name_ = "";
    //! Name of index group to use
    std::string indexGroupName_ = "";
    //! Vector of global indices
    std::vector<int> atomIndices_;
    //! Index of corresponding group type in used group types vector
    int groupTypeIndex = -1;
    //! Vector of initial states
    std::vector<real> initialStates_;
    //! Barrier
    real barrier_ = 7.5;
    //! Charge constraint group number
    int chargeConstrintGroup = -1;
    //! Buffer residue multiplier
    int bufferMultiplier_ = 1;
};

/*! \internal
 * \brief Holding all parameters needed for Lambda Dynamics simulation.
 * Also used for setting all default parameter values.
 */
struct LambdaDynamicsParameters
{
    //! Indicate if density fitting is active
    bool active_ = false;
    //! Simulation pH
    real pH_ = 0.0;
    //! Write Lambda Dynamics output every n-stepts
    std::int64_t nst_ = 100;
    //! Mass of lambda-particles
    real lambdaMass_ = 5.;
    //! Lambda Dynamics thermostat tau
    real tau_ = 2.0;
    //! Whether to use charge constraints in Lambda Dynamics
    bool useChargeConstraints_ = false;
    //! Whether Lambda Dynamics is in the calibration mode
    bool isCalibration_ = false;
    //! Number of atom sets
    std::int64_t nAtomSets_ = 0;
    //! Vector of group types
    std::vector<LambdaDynamicsGroupType> groupTypes_;
    //! Vector of used group types
    std::vector<LambdaDynamicsGroupType> usedGroupTypes_;
    //! Vector of of atm sets
    std::vector<LambdaDynamicsAtomSet> atomSets_;
};

const std::string enumValueToString(Directive d);

Directive str2dir(std::string dstr);

/*! \internal
 * \brief Input data storage for Lambda Dynamics
 */
class LambdaDynamicsOptions final : public IMdpOptionProvider
{
public:
    //! Implementation of IMdpOptionProvider method
    void initMdpTransform(IKeyValueTreeTransformRules* rules) override;

    /*! \brief
     * Build mdp parameters for Lambda Dynamics to be output after pre-processing.
     * \param[in, out] builder the builder for the mdp options output KVT.
     */
    void buildMdpOutput(KeyValueTreeObjectBuilder* builder) const override;

    /*! \brief
     * Connects options names and data.
     */
    void initMdpOptions(IOptionsContainerWithSections* options) override;

    //! Report if this set of MDP options is active (i.e. Lambda Dynamics MdModule is active)
    bool active() const;

    //! Get parameters_ instance
    const LambdaDynamicsParameters& parameters();

    /*! \brief Evaluate and store atom indices.
     * During pre-processing, use the group string from the options to
     * evaluate the indices of both QM atoms and MM atoms, also stores them
     * as vectors into the parameters_
     * \param[in] indexGroupsAndNames object containing data about index groups and names
     */
    //void setLambdaDynamicsGroupIndices(const IndexGroupsAndNames& indexGroupsAndNames);

    /*! \brief Modifies topology in case of active QMMM module using QMMMTopologyPreprocessor
     * \param[in,out] mtop topology to modify for QMMM
     */
    //void setLambdaDynamicsCharges(gmx_mtop_t* mtop);

    //! Store the paramers that are not mdp options in the tpr file
    void writeInternalParametersToKvt(KeyValueTreeObjectBuilder treeBuilder);

    //! Set the internal parameters that are stored in the tpr file
    void readInternalParametersFromKvt(const KeyValueTreeObject& tree);

    //! Set the MDLogger instance
    void setLogger(const MDLogger& logger);

    //! Set the warninp instance
    void setWarninp(warninp* wi);

    /*! \brief Process Lambda Dynamics input file in case it is provided with -ldi option of grompp.
     * Reads lambda dynamics force field parameters
     * \param[in] lambdaDynamicsInputFileName structure with information about lambda dynamics 
     * force field parameters input
     */
    void setFFInputFile(const LambdaDynamicsInputFileName& lambdaDynamicsInputFileName);

private:
    //! Write message to the log
    void appendLog(const std::string& msg);

    //! Write grompp warning
    void appendWarning(const std::string& msg);

    /*! \brief Following Tags denotes names of parameters from .mdp file
     * \note Changing this strings will break .tpr backwards compability
     */
    //! \{
    const std::string c_activeTag_              = "active";
    const std::string c_pHTag_                  = "simulation-pH";
    const std::string c_nStepsTag_              = "nst";
    const std::string c_massTag_                = "particle-mass";
    const std::string c_tauTag_                 = "tau";
    const std::string c_chargeConstraintsTag_   = "charge-constraints";
    const std::string c_isCalibrationTag_       = "calibration";
    const std::string c_nAtomCollectionsTag_    = "n-atom-collections";
    //! \}

    /*! \brief This tags for parameters which will be generated during grompp
     * and stored into *.tpr file via KVT
     */
    //! \{
    const std::string c_lambdaGroupTag_  = "lambda-group";
    const std::string c_nLambdGroupsTag_ = "nLambdaGroups";
    //! \}

    //! Logger instance
    const MDLogger* logger_ = nullptr;

    //! Instance of warning bookkeeper
    warninp* wi_ = nullptr;

    //! LambdaDynamics parameters built from mdp input
    LambdaDynamicsParameters parameters_;

    /*! \brief LambdaDynamics module searches for the force field specific parameters
     * in grouptypes.ldp file. Currently, the supposed location of the file is 
     * working directory
     */
    std::string lambdaDynamicsGroupTypesFileName_;

    /*! \brief Process force field specific lambda dynamics input file.
     * Produces the vector of groupTypes in LambdaDynamicsParameters
     */
    void getGroupTypeInformationFromFF();

};

} // namespace gmx

#endif
