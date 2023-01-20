/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2019- The GROMACS Authors
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
 * Helper classes for tests that compare the results of equivalent
 * simulation runs. Currently used for the rerun and the simulator
 * tests
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdrun_integration_tests
 */

#ifndef GMX_PROGRAMS_MDRUN_TESTS_SIMULATORCOMPARISON_H
#define GMX_PROGRAMS_MDRUN_TESTS_SIMULATORCOMPARISON_H

#include <string>

#include "gromacs/utility/enumerationhelpers.h"

#include "testutils/setenv.h"

#include "comparison_helpers.h"
#include "energycomparison.h"
#include "trajectorycomparison.h"

namespace gmx
{
namespace test
{
class SimulationRunner;

typedef std::tuple<std::string, std::string> SimulationOptionTuple;

void runGrompp(SimulationRunner*                         runner,
               const std::vector<SimulationOptionTuple>& options = std::vector<SimulationOptionTuple>());

void runMdrun(SimulationRunner*                         runner,
              const std::vector<SimulationOptionTuple>& options = std::vector<SimulationOptionTuple>());

void compareEnergies(const std::string&          edr1Name,
                     const std::string&          edr2Name,
                     const EnergyTermsToCompare& energyTermsToCompare,
                     MaxNumFrames                maxNumFrams = MaxNumFrames::compareAllFrames());

void compareTrajectories(const std::string&          trajectory1Name,
                         const std::string&          trajectory2Name,
                         const TrajectoryComparison& trajectoryComparison);

//! Allow for type-safe choice of environment variables
enum class SimulatorChoice : int
{
    UseModularSimulator,
    DisableModularSimulator,
    //! Provide conventional end-of-enumeration marker.
    Count
};

const EnumerationArray<SimulatorChoice, const char*> c_environmentVariableNames = {
    "GMX_USE_MODULAR_SIMULATOR",
    "GMX_DISABLE_MODULAR_SIMULATOR"
};

/*! \brief RAII-style helper to preserve the old value of an
 * environment variable and restore it upon destruction. */
class EnvironmentVariablePreserver
{
public:
    EnvironmentVariablePreserver() = default;
    //! Constructor that works on environment variable \c name
    explicit EnvironmentVariablePreserver(const char* name) : name_(name), oldValue_(getenv(name_))
    {
    }
    //! Copy construction does not make sense
    EnvironmentVariablePreserver(const EnvironmentVariablePreserver&) = delete;
    //! Copy assignment does not make sense
    EnvironmentVariablePreserver& operator=(const EnvironmentVariablePreserver&) = delete;
    //! Move assignment
    EnvironmentVariablePreserver& operator=(EnvironmentVariablePreserver&& other) noexcept
    {
        if (&other != this)
        {
            // Ensure that other no longer manages a variable that
            // would be set in its destructor.
            EnvironmentVariablePreserver replacement;
            std::swap(other.name_, replacement.name_);
            std::swap(other.oldValue_, replacement.oldValue_);
        }
        return *this;
    }
    //! Move constructor
    EnvironmentVariablePreserver(EnvironmentVariablePreserver&& other) noexcept
    {
        *this = std::move(other);
    }
    //! Destructor that restores the old value, if any
    ~EnvironmentVariablePreserver()
    {
        if (oldValue_)
        {
            const int overWriteEnvironmentVariable = 1;
            gmxSetenv(name_, oldValue_, overWriteEnvironmentVariable);
        }
    }

private:
    //! The name of the environment variable with a value to preserve.
    const char* name_ = nullptr;
    //! The former value of the environment variable to restore.
    const char* oldValue_ = nullptr;
};

template<typename FirstCallable, typename SecondCallable>
void changeSimulatorBetweenSimulations(const SimulatorChoice& simulatorChoice,
                                       const FirstCallable    firstSimulation,
                                       const SecondCallable   secondSimulation)
{
    // Backup current state of both environment variables and unset them
    EnumerationArray<SimulatorChoice, EnvironmentVariablePreserver> environmentVariableValuePreservers;
    for (const SimulatorChoice choice : keysOf(c_environmentVariableNames))
    {
        environmentVariableValuePreservers[choice] =
                EnvironmentVariablePreserver(c_environmentVariableNames[choice]);
        gmxUnsetenv(c_environmentVariableNames[choice]);
    }

    // Run first simulation with default simulator
    firstSimulation();

    // Set tested environment variable to change simulator
    const int overWriteEnvironmentVariable = 1;
    gmxSetenv(c_environmentVariableNames[simulatorChoice], "ON", overWriteEnvironmentVariable);

    secondSimulation();

    // Unset tested environment variable
    gmxUnsetenv(c_environmentVariableNames[simulatorChoice]);
}

} // namespace test
} // namespace gmx

#endif // GMX_PROGRAMS_MDRUN_TESTS_SIMULATORCOMPARISON_H
