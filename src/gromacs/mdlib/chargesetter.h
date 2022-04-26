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
/*! \file
 * \libinternal \brief
 * Declares gmx::ChargeSetter.
 *
 * \author Christian Blau <cblau.mail@gmail.com>
 */
#ifndef GMX_MDLIB_CHARGESETTER_H
#define GMX_MDLIB_CHARGESETTER_H

#include <memory>

#include "gromacs/utility/real.h"

namespace gmx
{
template<typename>
class ArrayRef;

/*! \libinternal \brief
 * A ChargeSetter allows to set charges with an interpolation parameter.
 *
 * To generate a ChargeSetter call gmx::ChargeSetterManager::add and keep the
 * handle to the ChargeSetter returned from this call.
 */
class ChargeSetter
{
public:
    // only the ChargeSetterManager gets to construct this class
    friend class ChargeSetterManager;

    /*! \brief Requests the ChargeSetterManager to set charges for the atoms in
     *         the localAtomSet that was used to build this chargesetter in the
     *         ChargeSettermanager
     *
     * NOTE better requestSetCharge is misleading, better use something like
     * requestInterpolateCharges
     */
    void requestSetCharge(real lambda) { *requestedLambda_ = lambda; }

private:
    /*! \brief Constructs a new ChargeSetter by setting a reference to the lambda.
     *         which is manged by \ref gmx::ChargeSetterManager.
     */
    ChargeSetter(real* requestedLambda) : requestedLambda_{ requestedLambda } {}

    /*! \brief Store access to the requested lambda via an observing pointer.
     *
     * It is the task of the ChargeSetterManager to set the charges according
     * to the requested lambda values, however the individual ChargeSetters need
     * not store a handle to the ChargeSetterManager to avoid circular
     * dependency between ChargeSetterManager and ChargeSetter.
     *
     * When the data that the ChargeSetter has a handle on becomes less trivial
     * as a real value, introduce a ChargeSetterData class that both
     * ChargeSetter and ChargeSetterManager implementations use to avoid
     * circular dependencies.
     */
    real* requestedLambda_;
};

} // namespace gmx

#endif
