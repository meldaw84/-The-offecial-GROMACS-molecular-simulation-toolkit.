/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2022, by the GROMACS development team, led by
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
 * \brief
 * Implements the MSM class.
 *
 * \author Cathrine Bergh
 */
#include "gmxpre.h"

#include "msm.h"

namespace gmx
{

// Constructor
MarkovModel::MarkovModel()
{
  // TODO: Move nstates to function argument
  // TODO: scan clustered trajectory of highest state value?
  const int nstates = 4;

  // Initialize TCM as MultiDimArray
  transitionCountsMatrix = { { } };
}


void MarkovModel::count_transitions(std::vector<int>& discretizedTraj, int lag)
{
    //Extract time-lagged trajectories
    std::vector<int> rows(discretizedTraj.begin(), discretizedTraj.end() - lag);
    std::vector<int> cols(discretizedTraj.begin() + lag, discretizedTraj.end());

    // Iterate over trajectory and count transitions
    for (int i = 0; i < rows.size(); i++)
    {
      transitionCountsMatrix(rows[i], cols[i]) += 1;
    }
}

} // namespace gmx
