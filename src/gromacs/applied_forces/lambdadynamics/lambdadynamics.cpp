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
 * Implements LambdaDynamics class that implements IMDModule interface
 *
 * \author Pavel Buslaev <pavel.i.buslaev@jyu.fi>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "lambdadynamics.h"

#include <memory>
#include <numeric>

#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/fileio/checkpoint.h"
#include "gromacs/math/multidimarray.h"
#include "gromacs/mdlib/broadcaststructs.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/imdmodule.h"
#include "gromacs/mdtypes/imdoutputprovider.h"
#include "gromacs/utility/classhelpers.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/logger.h"

#include "lambdadynamicsforceprovider.h"
#include "lambdadynamicsoptions.h"
#include "lambdadynamicsoutputprovider.h"

namespace gmx
{

namespace
{

/*! \internal
 * \brief Helper class that holds simulation data and
 * callback functions for simulation setup time notifications
 */
class LymbdaDynamicsSimulationParameterSetup
{
public:
    LymbdaDynamicsSimulationParameterSetup() = default;

    /*! \brief Set the local atom set for the lambda dynamics.
     * \param[in] localAtomSet of atoms to be in lambda coordinates
     */

    /*
    void setLocalAtomSet(const LocalAtomSet& localAtomSet)
    {
        localAtomSet_ = std::make_unique<LocalAtomSet>(localAtomSet);
    }
    */

    /*! \brief Return local atom set for lambda dynamics.
     * \throws InternalError if local atom set is not set
     * \returns local atom set for lambda dynamics.
     */

    /*
    const LocalAtomSet& localAtomSet() const
    {
        if (localAtomSet_ == nullptr)
        {
            GMX_THROW(
                    InternalError("Local atom set is not set for Lambda "
                                  "Dynamics simulation."));
        }
        return *localAtomSet_;
    }
    */

    /*! \brief Set the periodic boundary condition via MDModuleNotifier.
     *
     * The pbc type is wrapped in PeriodicBoundaryConditionType to
     * allow the MDModuleNotifier to statically distinguish the callback
     * function type from other 'int' function callbacks.
     *
     * \param[in] pbcType enumerates the periodic boundary condition.
     */
    void setPeriodicBoundaryConditionType(const PbcType& pbcType)
    {
        pbcType_ = std::make_unique<PbcType>(pbcType);
    }

    //! Get the periodic boundary conditions
    PbcType periodicBoundaryConditionType()
    {
        if (pbcType_ == nullptr)
        {
            GMX_THROW(InternalError(
                    "Periodic boundary condition enum not set for Lambda Dynamics simulation."));
        }
        return *pbcType_;
    }

    /*! \brief Set the logger for Lambda Dynamics during mdrun
     * \param[in] logger Logger instance to be used for output
     */
    void setLogger(const MDLogger& logger) { logger_ = &logger; }

    //! Get the logger instance
    const MDLogger& logger() const
    {
        if (logger_ == nullptr)
        {
            GMX_THROW(InternalError("Logger not set for Lambda Dynamics simulation."));
        }
        return *logger_;
    }

private:
    //! The local QM atom set to act on
    //! A vector of local atom sets
    // std::unique_ptr<LocalAtomSet> localAtomSet_;
    //! The type of periodic boundary conditions in the simulation
    std::unique_ptr<PbcType> pbcType_;
    //! MDLogger for notifications during mdrun
    const MDLogger* logger_ = nullptr;

    GMX_DISALLOW_COPY_AND_ASSIGN(LymbdaDynamicsSimulationParameterSetup);
};

/*! \internal
 * \brief LambdaDynamics module
 *
 * Class that implements the Lambda Dynamics module
 */
class LambdaDynamics final : public IMDModule
{
public:
    //! \brief Construct the LambdaDynamics module.
    explicit LambdaDynamics() = default;

    // Now callbacks for several kinds of MdModuleNotification are created
    // and subscribed, and will be dispatched correctly at run time
    // based on the type of the parameter required by the lambda.

    /*! \brief Requests to be notified during pre-processing.
     *
     * \param[in] notifier allows the module to subscribe to notifications from MdModules.
     *
     * The LambdaDynamics code subscribes to these notifications:
     *   - setting atom group indices in the lambdadynamicsOptions_ from an
     *     index group string by taking a parmeter const IndexGroupsAndNames &
     *   - storing its internal parameters in a tpr file by writing to a
     *     key-value-tree during pre-processing by a function taking a
     *     KeyValueTreeObjectBuilder as parameter
     *   - Modify charges according to input of LambdaDynamics
     *   - Access MDLogger for notifications output
     *   - Access warninp for for grompp warnings output
     *   - Coordinates, PBC and box for pH gradient/spatial dependency
     */
    void subscribeToPreProcessingNotifications(MDModulesNotifiers* notifier) override
    {
        if (!lambdadynamicsOptions_.active())
        {
            return;
        }

        // Notification of the Lambda Dynamics input file provided via -ldi option of grompp
        const auto setLambdaDynamicsExternalInputFileNameFunction =
                [this](const LambdaDynamicsInputFileName& ldInputFileName) {
                    lambdadynamicsOptions_.setFFInputFile(ldInputFileName);
                };
        notifier->preProcessingNotifier_.subscribe(setLambdaDynamicsExternalInputFileNameFunction);

        // Writing internal parameters during pre-processing
        const auto writeInternalParametersFunction = [this](KeyValueTreeObjectBuilder treeBuilder) {
            lambdadynamicsOptions_.writeInternalParametersToKvt(treeBuilder);
        };
        notifier->preProcessingNotifier_.subscribe(writeInternalParametersFunction);

        // Setting atom group indices
        // multiple groups
        /*
        const auto setLambdaDynamicsGroupIndicesFunction =
                                            [this](const IndexGroupsAndNames& indexGroupsAndNames) {
            lambdadynamicsOptions_.setLabdaDynamicsGroupIndices(indexGroupsAndNames);
        };
        notifier->preProcessingNotifier_.subscribe(setLambdaDynamicsGroupIndicesFunction);*/

        // Set Logger during pre-processing
        const auto setLoggerFunction = [this](const MDLogger& logger) {
            lambdadynamicsOptions_.setLogger(logger);
        };
        notifier->preProcessingNotifier_.subscribe(setLoggerFunction);

        // Set warning output during pre-processing
        const auto setWarninpFunction = [this](WarningHandler* wi) {
            lambdadynamicsOptions_.setWarninp(wi);
        };
        notifier->preProcessingNotifier_.subscribe(setWarninpFunction);

        // Notification of the Coordinates, box and pbc during pre-processing
        /*const auto processCoordinatesFunction = [this](const CoordinatesAndBoxPreprocessed& coord)
        { lambdadynamicsOptions_.processCoordinates(coord);
        };
        notifier->preProcessingNotifier_.subscribe(processCoordinatesFunction);*/

        // Setting charges during pre-processing
        // Here we should use charge setter???
        // Check if we need to add a ref to chset
        /*
        const auto setLambdaDynamicsChargesFunction = [this](gmx_mtop_t* mtop) {
            lambdadynamicsOptions_.setLambdaDynamicsCharges(mtop);
        };
        notifier->preProcessingNotifier_.subscribe(setLambdaDynamicsChargesFunction);*/
    }

    /*! \brief Requests to be notified during simulation setup.
     * The Lambda Dynamics code subscribes to these notifications:
     *   - reading its internal parameters from a key-value-tree during
     *     simulation setup by taking a const KeyValueTreeObject & parameter
     *   - constructing local atom sets in the simulation parameter setup
     *     by taking a LocalAtomSetManager * as parameter
     *     ??? should we instead send global indices to potential manager???
     *     potential manager should be a class here and an object of module
     *     or this can be done through construction of lambdadynamics options
     *     we need to use a list of local atom sets/ a map between coordinates
     *     and atoms in collective atom sett
     *   - the type of periodic boundary conditions that are used
     *     by taking a PeriodicBoundaryConditionType as parameter
     *   - Access MDLogger for notifications output
     *   - Request LambdaDynamics energy output to md.log
     *   - Should we request energy output to edr????
     */
    void subscribeToSimulationSetupNotifications(MDModulesNotifiers* notifier) override
    {
        if (!lambdadynamicsOptions_.active())
        {
            return;
        }

        // Reading internal parameters during simulation setup
        const auto readInternalParametersFunction = [this](const KeyValueTreeObject& tree) {
            lambdadynamicsOptions_.readInternalParametersFromKvt(tree);
        };
        notifier->simulationSetupNotifier_.subscribe(readInternalParametersFunction);

        // constructing local atom sets during simulation setup
        /*
        const auto setLocalAtomSetFunction = [this](LocalAtomSetManager* localAtomSetManager) {
            LocalAtomSet atomSet = localAtomSetManager->add(
                                   lambdadynamisOptions_.parameters().lambdaIndices_);
            this->lambdaDynamicsSimulationParameters_.setLocalAtomSet(atomSet);
        };
        notifier->simulationSetupNotifier_.subscribe(setLocalAtomSetFunction);*/

        // Reading PBC parameters during simulation setup
        const auto setPeriodicBoundaryContionsFunction = [this](const PbcType& pbc) {
            this->lambdaDynamicsSimulationParameters_.setPeriodicBoundaryConditionType(pbc);
        };
        notifier->simulationSetupNotifier_.subscribe(setPeriodicBoundaryContionsFunction);

        // Saving MDLogger during simulation setup
        const auto setLoggerFunction = [this](const MDLogger& logger) {
            this->lambdaDynamicsSimulationParameters_.setLogger(logger);
        };
        notifier->simulationSetupNotifier_.subscribe(setLoggerFunction);

        // Adding output to energy file
        const auto requestEnergyOutput =
                [](MDModulesEnergyOutputToLambdaDynamicsRequestChecker* energyOutputRequest) {
                    energyOutputRequest->energyOutputToLambdaDynamics_ = true;
                };
        notifier->simulationSetupNotifier_.subscribe(requestEnergyOutput);

        // Request to disable PME-only ranks, which are not compatible with
        // lambda dynamics currently
        const auto requestPmeRanks = [](SeparatePmeRanksPermitted* pmeRanksPermitted) {
            pmeRanksPermitted->disablePmeRanks(
                    "Separate PME-only ranks are not compatible with Lambda Dynamics MdModule");
        };
        notifier->simulationSetupNotifier_.subscribe(requestPmeRanks);
    }

    //! From IMDModule
    IMdpOptionProvider* mdpOptionProvider() override { return &lambdadynamicsOptions_; }

    //! Add this module to the force providers if active
    void initForceProviders(ForceProviders* forceProviders) override
    {
        if (!lambdadynamicsOptions_.active())
        {
            return;
        }

        //! managers needed for lambda coordinate update

        //! parameters and lambda coordinates
        // const auto& parameters = lambdadynamicsOptions_.parameters();

        forceProvider_ = std::make_unique<LambdaDynamicsForceProvider>(
                lambdaDynamicsSimulationParameters_.periodicBoundaryConditionType(),
                lambdaDynamicsSimulationParameters_.logger());
        forceProviders->addForceProvider(forceProvider_.get());
    }

    //! This MDModule provides its own output
    IMDOutputProvider* outputProvider() override { return &lambdaDynamicsOutputProvider_; }

private:
    //! The output provider
    LambdaDynamicsOutputProvider lambdaDynamicsOutputProvider_;
    //! The options provided for Lambda Dynamics
    LambdaDynamicsOptions lambdadynamicsOptions_;
    //! Object that evaluates the forces
    //! Though in our case there are no forces on cartesian coordinates
    std::unique_ptr<LambdaDynamicsForceProvider> forceProvider_;
    /*! \brief Parameters for LambdaDynamics that become available at
     * simulation setup time.
     */
    LymbdaDynamicsSimulationParameterSetup lambdaDynamicsSimulationParameters_;

    GMX_DISALLOW_COPY_AND_ASSIGN(LambdaDynamics);
};

} // namespace

std::unique_ptr<IMDModule> LambdaDynamicsModuleInfo::create()
{
    return std::make_unique<LambdaDynamics>();
}

const std::string LambdaDynamicsModuleInfo::name_ = "lambda-dynamics";

} // namespace gmx
