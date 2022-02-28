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
/*! \libinternal \file
 * \brief
 * Declares utilities for selecting energies.
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_energyanalysis
 */
#ifndef GMX_ENERGYANALYSIS_SELECT_H
#define GMX_ENERGYANALYSIS_SELECT_H

#include <string>
#include <vector>

#include "ienergyanalysis.h"

namespace gmx
{
class TextInputStream;

/*! \libinternal
 * \brief Return a set of indices in the energy file
 *
 * The indices can be specified interactively or by parsing a string.
 * The caller may need to check whether the set is empty on return.
 * \param[in]  eNU      The names (and units) of the energy terms
 * \param[in]  bVerbose Write help text to standard output
 * \param[in]  input    Input stream from file or command line string providing selections.
 * \param[out] set      The vector of indices selected by the input
 */
void select_energies(ArrayRef<const EnergyNameUnit> eNU,
                     bool                           bVerbose,
                     TextInputStream*               input,
                     std::vector<int>*              set);

} // namespace gmx

#endif
