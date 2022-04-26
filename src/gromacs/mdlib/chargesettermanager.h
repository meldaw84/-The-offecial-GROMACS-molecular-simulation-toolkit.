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
/*! \libinternal \file
 * \brief
 * Declares gmx::ChargeSetterManager
 *
 * \author Christian Blau <cblau.mail@gmail.com>
 */

#ifndef GMX_MDLIB_CHARGESETTERMANAGER_H
#define GMX_MDLIB_CHARGESETTERMANAGER_H

#include <memory>
#include <vector>

#include "gromacs/utility/real.h"

namespace gmx
{
template<typename>
class ArrayRef;

class LocalAtomSet;
class ChargeSetter;

/*! \libinternal \brief
 * Hands out handles to ChargeSetters and performs charge interpolation between
 * two states for all ChargeSetters, once they request charges be interpolated
 * for a certain atomSet.
 */
class ChargeSetterManager
{
public:
    /*! \brief Construct an object to allow to set charges during a simulation.
     *
     * Needs to be able to write to the charges in the A-state which is
     * the reference charge used in all simulations, needs access to the
     * charges in the B-state. Stores the original charges in the A-state to
     * be able to perform the interpolation each time interpolationParameter changes.
     *
     * \param[in] chargeA writable view on the charges in the A-state
     * \param[in] chargeB view on the charges in the B-state
     */
    ChargeSetterManager(ArrayRef<real> chargeA, ArrayRef<const real> chargeB);

    /*! \brief Add a set of atoms whose charges are to be interpolated and
     *         return a handle to do just that.
     * \param[in] atoms whose charge shall be set
     * \returns Handle to ChargeSetter
     */
    ChargeSetter add(const LocalAtomSet& atoms);

    void updateCharges();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

};

} // namespace gmx

#endif
