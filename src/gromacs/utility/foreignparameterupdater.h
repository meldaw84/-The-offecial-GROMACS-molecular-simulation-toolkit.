/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2022- by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
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
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
 /*! \internal \file
 *
 * \brief 
 *
 * \author Christian Blau <cblau.mail@gmail.com>
 * \ingroup 
 */

#include <memory>

#include "gromacs/utility/real.h"

#ifndef GMX_FOREIGNPARAMETERUPDATER_H
#define GMX_FOREIGNPARAMETERUPDATER_H

namespace gmx
{

/*! \brief Provide a handle to set a parameter that is stored in this class,
 *         keep track if the parameter has changed, retreive and update
 *         the parameter upon retrieval.
 *
 * By using the handle that this class returns, a parameter in this class
 * can be set as long as the handle is valid. The object setting the parameter
 * and the object consuming the parameter can thus have only minimal coupling
 * with another.
 *
 * This class needs to outlive the handles that it hands out, because the
 * handle to its data must not expire.
 *
 * The typical application of this class is a managing object that wants to
 * allow other classes to make a change to its internal state.
 *
 * This class allows lazy updates of data. It stores the latest parameter
 * setting and updates the set parameter only when it is requested. It would
 * typically be used in patterns like this:
 *
 * if (updater.needsUpdate())
 * {
 *   real parameter = updater.updateAndGetParameter();
 *   someExpensiveCalculation(parameter);
 * }
 *
 * \note instances of this class have to outlive the handle to its data
 */
class ForeignParameterUpdater
{
public:
    /*! \brief Initialize the parameter, so we know if to update on the first
     *         request.
     * \param[in] initialValue the initial value
     */
    ForeignParameterUpdater(real initialValue);

    //! Handle that allows to request a parameter be set
    real* handleToRequestParameter();

    //! True if the parameters in this ForeignParameterUpdater need to be updated
    bool needsUpdate() const;

    //! Let the ForeignParameterUpdater know that Parameters have been updated accordingly
    real updateAndGetParameter();

private:
    //! The parameter that was set before the last update
    real currentParameter_;
    /*! \brief  Allocate a fixed place in memory for the data that external
     * modules can write to. That way the handle to the parameter is valid,
     * even if this class is moved to a different location in memory.
     * (eg upon vector reallocation)
     */
    std::unique_ptr<real> requestedParameter_;
};


} // namespace gmx
#endif