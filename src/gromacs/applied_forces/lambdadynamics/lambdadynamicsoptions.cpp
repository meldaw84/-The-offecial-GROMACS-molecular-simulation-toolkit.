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
 * Implements LambdaDynamicsOptions class
 *
 * \author Pavel Buslaev <pavel.i.buslaev@jyu.fi>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "lambdadynamicsoptions.h"

#include <map>

#include "gromacs/applied_forces/lambdadynamics/lambdadynamics.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/math/vec.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/optionsection.h"
#include "gromacs/selection/indexutil.h"
#include "gromacs/topology/mtop_lookup.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/keyvaluetreetransform.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mdmodulesnotifiers.h"
#include "gromacs/utility/path.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/stringcompare.h"
#include "gromacs/utility/stringtoenumvalueconverter.h"
#include "gromacs/utility/textreader.h"

#define OPENDIR  '['  /* starting sign for directive */
#define CLOSEDIR ']' /* ending sign for directive   */
#define COMMENT  '#' /* comment sign */

namespace gmx
{

namespace
{

/*! \brief Helper to declare mdp transform rules.
 *
 * Enforces uniform mdp options that are always prepended with the correct
 * string for the Lambda Dynamics mdp options.
 *
 * \tparam ToType type to be transformed to
 * \tparam TransformWithFunctionType type of transformation function to be used
 *
 * \param[in] rules KVT transformation rules
 * \param[in] transformationFunction the function to transform the flat kvt tree
 * \param[in] optionTag string tag that describes the mdp option, appended to the
 *                      default string for the Lambda Dynamics simulation
 */
template<class ToType, class TransformWithFunctionType>
void LambdaDynamicsMdpTransformFromString(IKeyValueTreeTransformRules* rules,
                                TransformWithFunctionType    transformationFunction,
                                const std::string&           optionTag)
{
    rules->addRule()
            .from<std::string>("/" + LambdaDynamicsModuleInfo::name_ + "-" + optionTag)
            .to<ToType>("/" + LambdaDynamicsModuleInfo::name_ + "/" + optionTag)
            .transformWith(transformationFunction);
}

/*! \brief Helper to declare mdp output.
 *
 * Enforces uniform mdp options output strings that are always prepended with the
 * correct string for the Lambda Dynamics mdp options and are consistent with the
 * options name and transformation type.
 *
 * \tparam OptionType the type of the mdp option
 * \param[in] builder the KVT builder to generate the output
 * \param[in] option the mdp option
 * \param[in] optionTag string tag that describes the mdp option, appended to the
 *                      default string for the Lambda Dynamics simulation
 */
template<class OptionType>
void addLambdaDynamicsMdpOutputValue(KeyValueTreeObjectBuilder* builder, 
                                     const OptionType& option, 
                                     const std::string& optionTag)
{
    builder->addValue<OptionType>(LambdaDynamicsModuleInfo::name_ + "-" + optionTag, option);
}

/*! \brief Helper to declare mdp output comments.
 *
 * Enforces uniform mdp options comment output strings that are always prepended
 * with the correct string for the Lambda Dynamics mdp options and are consistent
 * with the options name and transformation type.
 *
 * \param[in] builder the KVT builder to generate the output
 * \param[in] comment on the mdp option
 * \param[in] optionTag string tag that describes the mdp option
 */
void addLambdaDynamicsMdpOutputValueComment(KeyValueTreeObjectBuilder* builder,
                                            const std::string&         comment,
                                            const std::string&         optionTag)
{
    builder->addValue<std::string>("comment-" + LambdaDynamicsModuleInfo::name_ + "-" + optionTag, 
                                    comment);
}

} // namespace

void LambdaDynamicsOptions::initMdpTransform(IKeyValueTreeTransformRules* rules)
{
    LambdaDynamicsMdpTransformFromString<bool>(rules, &fromStdString<bool>, c_activeTag_);
    LambdaDynamicsMdpTransformFromString<real>(rules, &fromStdString<bool>, c_pHTag_);
    LambdaDynamicsMdpTransformFromString<int>(rules, &fromStdString<bool>, c_nStepsTag_);
    LambdaDynamicsMdpTransformFromString<real>(rules, &fromStdString<bool>, c_massTag_);
    LambdaDynamicsMdpTransformFromString<real>(rules, &fromStdString<bool>, c_tauTag_);
    LambdaDynamicsMdpTransformFromString<bool>(rules, &fromStdString<bool>, c_chargeConstraintsTag_);
    LambdaDynamicsMdpTransformFromString<bool>(rules, &fromStdString<bool>, c_isCalibrationTag_);
    LambdaDynamicsMdpTransformFromString<int>(rules, &fromStdString<bool>, c_nAtomCollectionsTag_);
        // We have parameters_
}

void LambdaDynamicsOptions::buildMdpOutput(KeyValueTreeObjectBuilder* builder) const
{

    addLambdaDynamicsMdpOutputValueComment(builder, "", "empty-line");

    // Active flag
    addLambdaDynamicsMdpOutputValueComment(builder, "; Lambda Dynamics", "module");
    addLambdaDynamicsMdpOutputValue(builder, parameters_.active_, c_activeTag_);

    if (parameters_.active_)
    {
        addLambdaDynamicsMdpOutputValueComment(builder, "; Lambda Dynamics pH", c_pHTag_);
        addLambdaDynamicsMdpOutputValue(builder, parameters_.pH_, c_pHTag_);

        addLambdaDynamicsMdpOutputValueComment(builder, "; Lambda Dynamics output nSteps", c_nStepsTag_);
        addLambdaDynamicsMdpOutputValue(builder, parameters_.nst_, c_nStepsTag_);

        addLambdaDynamicsMdpOutputValueComment(builder, "; Lambda Dynamics particle mass", c_massTag_);
        addLambdaDynamicsMdpOutputValue(builder, parameters_.lambdaMass_, c_massTag_);

        addLambdaDynamicsMdpOutputValueComment(builder, "; Lambda Dynamics thermostat tau", c_tauTag_);
        addLambdaDynamicsMdpOutputValue(builder, parameters_.tau_, c_tauTag_);

        addLambdaDynamicsMdpOutputValueComment(builder, 
                                               "; Does Lambda Dynamics charge constraint?", 
                                               c_chargeConstraintsTag_);
        addLambdaDynamicsMdpOutputValue(builder, 
                                        parameters_.useChargeConstraints_, 
                                        c_chargeConstraintsTag_);

        addLambdaDynamicsMdpOutputValueComment(builder, 
                                               "; Is Lambda Dynamics in calibration mode?", 
                                               c_isCalibrationTag_);
        addLambdaDynamicsMdpOutputValue(builder, parameters_.isCalibration_, c_isCalibrationTag_);

        addLambdaDynamicsMdpOutputValueComment(builder, 
                                               "; Lambda Dynamics number of atom sets", 
                                               c_nAtomCollectionsTag_);
        addLambdaDynamicsMdpOutputValue(builder, parameters_.nAtomSets_, c_nAtomCollectionsTag_);
    }
}

void LambdaDynamicsOptions::initMdpOptions(IOptionsContainerWithSections* options)
{
    auto section = options->addSection(OptionSection(LambdaDynamicsModuleInfo::name_.c_str()));

    section.addOption(BooleanOption(c_activeTag_.c_str()).store(&parameters_.active_));
    section.addOption(RealOption(c_pHTag_.c_str()).store(&parameters_.pH_));
    section.addOption(Int64Option(c_nStepsTag_.c_str()).store(&parameters_.nst_));
    section.addOption(RealOption(c_pHTag_.c_str()).store(&parameters_.lambdaMass_));
    section.addOption(RealOption(c_tauTag_.c_str()).store(&parameters_.tau_));
    section.addOption(BooleanOption(c_chargeConstraintsTag_.c_str()).store(&parameters_.useChargeConstraints_));
    section.addOption(BooleanOption(c_isCalibrationTag_.c_str()).store(&parameters_.isCalibration_));
    section.addOption(Int64Option(c_nAtomCollectionsTag_.c_str()).store(&parameters_.nAtomSets_));

    // add number of atom collections
    // store(&parameters ...)
    // iterate over num collections
}

bool LambdaDynamicsOptions::active() const
{
    return parameters_.active_;
}

const LambdaDynamicsParameters& LambdaDynamicsOptions::parameters()
{
    return parameters_;
}

void LambdaDynamicsOptions::setLogger(const MDLogger& logger)
{
    // Exit if QMMM module is not active
    if (!parameters_.active_)
    {
        return;
    }

    logger_ = &logger;
}

void LambdaDynamicsOptions::setWarninp(warninp* wi)
{
    // Exit if QMMM module is not active
    if (!parameters_.active_)
    {
        return;
    }

    wi_ = wi;
}

void LambdaDynamicsOptions::appendLog(const std::string& msg)
{
    if (logger_)
    {
        GMX_LOG(logger_->info).asParagraph().appendText(msg);
    }
}

void LambdaDynamicsOptions::appendWarning(const std::string& msg)
{
    if (wi_)
    {
        warning(wi_, msg);
    }
}

const std::string enumValueToString(Directive d)
{
    /* Must correspond to the Directive enum in lambdadynamicsoptions.h */
    std::map <Directive, const std::string> directiveNames = {
        {Directive::d_lambdaDynamicsResidues, "lambda_dynamics_residues"},
        {Directive::d_residue, "residue"},
        {Directive::d_state, "state"},
        {Directive::d_atoms, "atoms"},
        {Directive::d_end_atoms, "end_atoms"},
        {Directive::d_parameters, "parameters"},
        {Directive::d_end_parameters, "end_parameters"},
        {Directive::d_end_state, "end_state"},
        {Directive::d_end_residue, "end_residue"},
        {Directive::d_end, "end"},
        {Directive::d_invalid, "invalid"}
    }; 
    
    return directiveNames[d];
}

Directive str2dir(std::string dstr)
{
    std::map<std::string, Directive> stringToEnumValue;

    for (const auto type : EnumerationWrapper<Directive>{})
    {
        GMX_RELEASE_ASSERT(type != Directive::Count,
                               "EnumerationWrapper<EnumType> should never return EnumType::Count");
        std::string stringFromType = enumValueToString(type);
        stringFromType = stripString(stringFromType);
        stringToEnumValue[stringFromType] = type;
    }

    auto typeIt = stringToEnumValue.find(stripString(dstr));

    return (typeIt != stringToEnumValue.end()) ? typeIt->second : Directive::d_invalid;
}

namespace 
{

GroupState getGroupState(TextReader &fInp)
{
    GroupState groupState;

    std::string line;
    bool readingAtoms  = false;
    bool readingParams = false;
    while (fInp.readLine(&line))
    {
        line = stripString(line);
        // Skip comment and empty lines
        if ((line[0] == COMMENT) || (line.size() == 0))
        {
            continue;
        }

        if ((!readingAtoms) && (!readingParams) && (line[0] != OPENDIR))
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "Wrong formatting in the state entry of groupTypes.ldp file.")));
        }

        if ((readingAtoms) && (line[0] != OPENDIR))
        {
            auto words = splitString(line);
            if (words.size() != 3)
            {
                GMX_THROW(InconsistentInputError(formatString(
                        "Wrong formatting in the atoms section of groupTypes.ldp file."
                        "Atom section expects lines with 3 values: atom name, atom type, and charge.")));
            }
            groupState.atomNames_.push_back(words[0]);
            groupState.atomTypes_.push_back(words[1]);
            groupState.atomCharges_.push_back(std::stod(words[2]));
        }

        if ((readingParams) && (line[0] != OPENDIR))
        {
            auto words = splitString(line);
            if (words[0] == "pKa")
            {
                if (words.size() != 2)
                {
                    GMX_THROW(InconsistentInputError(formatString(
                            "Wrong formatting in the parameters section of groupTypes.ldp file."
                            "pKa entry expects only one value.")));
                }
                groupState.groupParameters_.pKa_ = std::stod(words[1]); 
            }
            else if (words[0] == "dvdl")
            {
                if (words.size() == 1)
                {
                    GMX_THROW(InconsistentInputError(formatString(
                            "Wrong formatting in the parameters section of groupTypes.ldp file."
                            "No dvdl values are provided")));
                }
                for (size_t i = 1; i < words.size(); ++i)
                {
                    groupState.groupParameters_.dvdlCoefs_.push_back(std::stod(words[i]));
                }
            }
            else
            {
                GMX_THROW(InconsistentInputError(formatString(
                        "Wrong formatting in the parameter section. Unexpected entry name."
                )));
            }
        }

        // Parse directive
        if (line[0] == OPENDIR)
        {
            Directive newd = str2dir(line.substr(1,line.size() - 2));
        
            if (newd == Directive::d_invalid) 
            {
                GMX_THROW(InconsistentInputError(formatString(
                    "%s directive is not allowed in groupTypes.ldp file",
                    stripString(line.substr(1,line.size() - 2)).c_str())));
            }

            if (newd == Directive::d_atoms)
            {
                readingAtoms = true;
            }

            if (newd == Directive::d_end_atoms)
            {
                readingAtoms = false;
            }

            if (newd == Directive::d_parameters)
            {
                readingParams = true;
                if ((readingAtoms) && (readingParams))
                {
                    GMX_THROW(InconsistentInputError(formatString(
                        "Wrong formatting of state section groupTypes.ldp file. "
                        "Two sections are open simultaneousely")));
                }
            }

            if (newd == Directive::d_end_parameters)
            {
                readingParams = false;
            }

            if (newd == Directive::d_end_state) 
            {
                // Checks: parameters and atoms
                return groupState;
            }

            GMX_THROW(InconsistentInputError(formatString(
                    "Wrong order of directives.")));
        }
    }

    // How we ended up here?
    return groupState;
}

LambdaDynamicsGroupType getGroupType(TextReader &fInp)
{
    LambdaDynamicsGroupType groupType;

    std::string line;
    while (fInp.readLine(&line))
    {
        line = stripString(line);
        // Skip comment and empty lines
        if ((line[0] == COMMENT) || (line.size() == 0))
        {
            continue;
        }

        if (line[0] != OPENDIR)
        {
            // Split line into words
            auto words = splitString(line);

            if (words.size() != 2)
            {
                GMX_THROW(InconsistentInputError(formatString(
                    "Wrong formatting in groupTypes.ldp file. Exactly two words are "
                    "expected in residue entry.")));
            }

            if (words[0] == "nstates")
            {
                groupType.nStates_ = std::stoi(words[1]);
            }
            else if (words[0] == "name")
            {
                groupType.name_ = words[1];
            }
            else if (words[0] == "type")
            {
                if (words[1] == "ssite")
                {
                    groupType.type_ = 0;
                }
                else if (words[1] == "msite")
                {
                    groupType.type_ = 1;
                }
                else
                {
                    GMX_THROW(InconsistentInputError(formatString(
                    "Wrong group type is provided in groupTypes.ldp file")));
                }
            }
            else
            {
                GMX_THROW(InconsistentInputError(formatString(
                    "Invalid entry is provided in residue block of groupTypes.ldp file")));
            }
        }
        // Parse directive
        if (line[0] == OPENDIR)
        {
            Directive newd = str2dir(line.substr(1,line.size() - 2));
        
            if (newd == Directive::d_invalid) 
            {
                GMX_THROW(InconsistentInputError(formatString(
                    "%s directive is not allowed in groupTypes.ldp file",
                    stripString(line.substr(1,line.size() - 2)).c_str())));
            }

            if (newd == Directive::d_state)
            {
                groupType.groupStates_.push_back(getGroupState(fInp));
            }

            if (newd == Directive::d_end_residue) 
            {
                // Checks: type and nstates; names; length and nstates; fill names
                return groupType;
            }

            GMX_THROW(InconsistentInputError(formatString(
                    "Wrong order of directives.")));

        }
    }
    //How we end up here?
    return groupType;
}

} // namespace end

void LambdaDynamicsOptions::getGroupTypeInformationFromFF()
{
    // Exit if LambdaDynamics module is not active
    if (!parameters_.active_)
    {
        return;
    }

    // First check if we could read lambdaDynamicsGroupTypesFileName_
    TextReader fInp(lambdaDynamicsGroupTypesFileName_);

    // Loop over all lines in the file
    std::string line;
    bool inLambdaGroupType = false;
    while (fInp.readLine(&line))
    {
        line = stripString(line);
        // Skip comment
        if (line[0] == COMMENT)
        {
            continue;
        }

        if ((!inLambdaGroupType) && (line[0] != OPENDIR))
        {
            GMX_THROW(InconsistentInputError(formatString(
                "The groupTypes.ldp file must start with the directives %s."
                "No information should be provided after %s directive. ",
                stripString(enumValueToString(Directive::d_lambdaDynamicsResidues)).c_str(),
                stripString(enumValueToString(Directive::d_end)).c_str())));
        }

        if ((inLambdaGroupType) && line[0] != OPENDIR)
        {
            GMX_THROW(InconsistentInputError(formatString(
                "No information should be provided before, after %s directive, "
                "or detween %s directives. ",
                stripString(enumValueToString(Directive::d_lambdaDynamicsResidues)).c_str(),
                stripString(enumValueToString(Directive::d_residue)).c_str())));
        }

        // Parse directive
        if (line[0] == OPENDIR)
        {
            Directive newd = str2dir(line.substr(1,line.size() - 2));

            if (newd == Directive::d_invalid) {
                GMX_THROW(InconsistentInputError(formatString(
                    "%s directive is not allowed in groupTypes.ldp file",
                    stripString(line.substr(1,line.size() - 2)).c_str())));
            }

            if ((!inLambdaGroupType) && 
                (newd != Directive::d_lambdaDynamicsResidues))
            {
                GMX_THROW(InconsistentInputError(formatString(
                    "The first directive in groupTypes.ldp file must be %s",
                    stripString(enumValueToString(Directive::d_lambdaDynamicsResidues)).c_str())));
            }

            if (newd == Directive::d_lambdaDynamicsResidues)
            {
                inLambdaGroupType = true;
            }

            if ((inLambdaGroupType) && (newd != Directive::d_residue))
            {
                GMX_THROW(InconsistentInputError(formatString(
                    "Wrong order of directives")));
            }

            if ((inLambdaGroupType) && (newd == Directive::d_residue))
            {
                parameters_.groupTypes_.push_back(getGroupType(fInp));
            }

            if (newd == Directive::d_end)
            {
                inLambdaGroupType = false;
            }
        }
    }
}

void LambdaDynamicsOptions::setFFInputFile(const LambdaDynamicsInputFileName& lambdaDynamicsInputFileName)
{
    // Exit if Lambda Dynamics module is not active
    if (!parameters_.active_)
    {
        return;
    }

    // Lambda dynamics requires force field specific input
    if (!lambdaDynamicsInputFileName.hasLambdaDynamicsInputFileName_)
    {
        // If parameters_.qmMethod_ != INPUT then user should not provide external input file
        GMX_THROW(InconsistentInputError(
                "Lambda dynamics requires the force field input file with group definitions "
                "provided to grompp with -ldi option, but it was not provided"));
    }

    // If external input is provided by the user then we should process it and save into the parameters_
    lambdaDynamicsGroupTypesFileName_ = lambdaDynamicsInputFileName.lambdaDynamicsFileName_;
    getGroupTypeInformationFromFF();
}

void LambdaDynamicsOptions::writeInternalParametersToKvt(KeyValueTreeObjectBuilder treeBuilder)
{    
    // Write number of lambda dynamics groups
    treeBuilder.addValue<std::int64_t>(LambdaDynamicsModuleInfo::name_ + "-" + c_nLambdGroupsTag_, 
                parameters_.groupTypes_.size());
    // Write lambda dynamics groups
    int groupNumber = 1;
    for (auto &group : parameters_.groupTypes_)
    {
        std::string header = LambdaDynamicsModuleInfo::name_ + "-" + 
                                c_lambdaGroupTag_ + "-" + std::to_string(groupNumber);
        treeBuilder.addValue<std::string>(header + "-name", group.name_);
        treeBuilder.addValue<std::int64_t>(header + "-nstates", group.nStates_);
        treeBuilder.addValue<std::int64_t>(header + "-type", group.type_);
        auto StringArrayAdder = 
                treeBuilder.addUniformArray<std::string>(header + "-atom-names");
        for (const auto& indexValue : group.atomNames_)
        {
            StringArrayAdder.addValue(indexValue);
        }
        int stateNumber = 1;
        for (auto &state : group.groupStates_)
        {
            std::string stateHeader = header + "-state-" + std::to_string(stateNumber);
            StringArrayAdder = 
                treeBuilder.addUniformArray<std::string>(stateHeader + "-atom-types");
            for (const auto& indexValue : state.atomTypes_)
            {
                StringArrayAdder.addValue(indexValue);
            }

            auto RealArrayAdder =
                treeBuilder.addUniformArray<real>(stateHeader + "-atom-charges");
            for (const auto& indexValue : state.atomCharges_)
            {
                RealArrayAdder.addValue(indexValue);
            }

            treeBuilder.addValue<real>(stateHeader + "-pka", state.groupParameters_.pKa_);

            RealArrayAdder =
                treeBuilder.addUniformArray<real>(stateHeader + "-dvdl_coefficients");
            for (const auto& indexValue : state.groupParameters_.dvdlCoefs_)
            {
                RealArrayAdder.addValue(indexValue);
            }
            ++stateNumber;
        }

        ++groupNumber;
    }
}

void LambdaDynamicsOptions::readInternalParametersFromKvt(const KeyValueTreeObject& tree)
{
    // Check if active
    if (!parameters_.active_)
    {
        return;
    }

    // Try to read number of Lambda Groups from tpr
    
    if (!tree.keyExists(LambdaDynamicsModuleInfo::name_ + "-" + c_nLambdGroupsTag_))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find the number of Lambda Groups types required for constant pH MD.\n"
                "This could be caused by incompatible or corrupted tpr input file."));
    }
    std::int64_t nLambdGroup = 
            tree[LambdaDynamicsModuleInfo::name_ + "-" + c_nLambdGroupsTag_].cast<std::int64_t>();
    
    // Read lambda dynamics groups from tpr
    for (std::int64_t i = 0; i != nLambdGroup; ++i)
    {
        int groupNumber = i + 1;

        LambdaDynamicsGroupType currentGroupType;

        std::string header = LambdaDynamicsModuleInfo::name_ + "-" + 
                                c_lambdaGroupTag_ + "-" + std::to_string(groupNumber);
        // Try to read Lambda Group type name
        if (!tree.keyExists(header + "-name"))
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "Cannot find the name of Lambda Group Type %s.\n"
                    "This could be caused by incompatible or corrupted tpr input file.",
                    std::to_string(groupNumber).c_str())));
        }
        currentGroupType.name_ = tree[header + "-name"].cast<std::string>();

        // Try to read Lambda Group type number of states
        if (!tree.keyExists(header + "-nstates"))
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "Cannot find the number of states for Lambda Group Type %s.\n"
                    "This could be caused by incompatible or corrupted tpr input file.",
                    std::to_string(groupNumber).c_str())));
        }
        currentGroupType.nStates_ = tree[header + "-nstates"].cast<int>();

        // Try to read the type of Lambda Group type
        if (!tree.keyExists(header + "-type"))
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "Cannot find the type data for Lambda Group Type %s.\n"
                    "This could be caused by incompatible or corrupted tpr input file.",
                    std::to_string(groupNumber).c_str())));
        }
        currentGroupType.type_ = tree[header + "-type"].cast<int>();

        // Try to read the atom names of Lambda Group type
        if (!tree.keyExists(header + "-atom-names"))
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "Cannot find the atom names for Lambda Group Type %s.\n"
                    "This could be caused by incompatible or corrupted tpr input file.",
                    std::to_string(groupNumber).c_str())));
        }
        auto kvtStringArray = tree[header + "-atom-names"].asArray().values();
        currentGroupType.atomNames_.resize(kvtStringArray.size());
        std::transform(std::begin(kvtStringArray),
                       std::end(kvtStringArray),
                       std::begin(currentGroupType.atomNames_),
                       [](const KeyValueTreeValue& val) { return val.cast<std::string>(); });
        
        for (int j = 0; j != currentGroupType.nStates_; ++j)
        {
            int stateNumber = j + 1;
            std::string stateHeader = header + "-state-" + std::to_string(stateNumber);
            
            GroupState currentGroupState;

            // Try to read the atom types of Lambda Group type state
            if (!tree.keyExists(stateHeader + "-atom-types"))
            {
                GMX_THROW(InconsistentInputError(formatString(
                        "Cannot find the atom types for Lambda Group Type %s state %s.\n"
                        "This could be caused by incompatible or corrupted tpr input file.",
                        std::to_string(groupNumber).c_str(), std::to_string(stateNumber).c_str())));
            }
            kvtStringArray = tree[stateHeader + "-atom-types"].asArray().values();
            currentGroupState.atomTypes_.resize(kvtStringArray.size());
            std::transform(std::begin(kvtStringArray),
                           std::end(kvtStringArray),
                           std::begin(currentGroupState.atomTypes_),
                           [](const KeyValueTreeValue& val) { return val.cast<std::string>(); });

            // Try to read the atom charges of Lambda Group type state
            if (!tree.keyExists(stateHeader + "-atom-charges"))
            {
                GMX_THROW(InconsistentInputError(formatString(
                        "Cannot find the atom charges for Lambda Group Type %s state %s.\n"
                        "This could be caused by incompatible or corrupted tpr input file.",
                        std::to_string(groupNumber).c_str(), std::to_string(stateNumber).c_str())));
            }
            auto kvtRealArray = tree[stateHeader + "-atom-charges"].asArray().values();
            currentGroupState.atomTypes_.resize(kvtRealArray.size());
            std::transform(std::begin(kvtRealArray),
                           std::end(kvtRealArray),
                           std::begin(currentGroupState.atomCharges_),
                           [](const KeyValueTreeValue& val) { return val.cast<real>(); });

            // Try to read Lambda Group type name
            if (!tree.keyExists(stateHeader + "-pKa"))
            {
                GMX_THROW(InconsistentInputError(formatString(
                        "Cannot find the pKa value for Lambda Group Type %s state %s.\n"
                        "This could be caused by incompatible or corrupted tpr input file.",
                        std::to_string(groupNumber).c_str(), std::to_string(stateNumber).c_str())));
            }
            currentGroupState.groupParameters_.pKa_ = tree[stateHeader + "-pka"].cast<real>();

            // Try to read the dvdl coefficients of Lambda Group type state
            if (!tree.keyExists(stateHeader + "-dvdl-coefficients"))
            {
                GMX_THROW(InconsistentInputError(formatString(
                        "Cannot find the dvdl coefficients for Lambda Group Type %s state %s.\n"
                        "This could be caused by incompatible or corrupted tpr input file.",
                        std::to_string(groupNumber).c_str(), std::to_string(stateNumber).c_str())));
            }
            kvtRealArray = tree[stateHeader + "-dvdl-coefficients"].asArray().values();
            currentGroupState.groupParameters_.dvdlCoefs_.resize(kvtRealArray.size());
            std::transform(std::begin(kvtRealArray),
                           std::end(kvtRealArray),
                           std::begin(currentGroupState.groupParameters_.dvdlCoefs_),
                           [](const KeyValueTreeValue& val) { return val.cast<real>(); });
            
            // Add atom names to group state
            currentGroupState.atomNames_ = currentGroupType.atomNames_;
            // Save group state to group type
            currentGroupType.groupStates_.push_back(currentGroupState);
        }

        // Save group type to parameters
        parameters_.groupTypes_.push_back(currentGroupType);
    }

}

} // namespace gmx
