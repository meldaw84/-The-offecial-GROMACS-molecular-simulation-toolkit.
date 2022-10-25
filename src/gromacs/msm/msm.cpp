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

#include <cstring>

#include "gromacs/linearalgebra/eigensolver.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/real.h"

namespace gmx
{

// Constructor
MarkovModel::MarkovModel(int nstates)
{
    // TODO: scan clustered trajectory of highest state value?
    // TODO: use brace initialization?

    // Initialize the TCM and TPM and set the size
    transitionCountsMatrix.resize(nstates, nstates);
    transitionProbabilityMatrix.resize(nstates, nstates);

    // Initialize eigenvalues and eigenvectors here?
    //eigenvalues.resize(nstates);
    //eigenvectors.resize(nstates * nstates);

}

// TODO: In function signatures use gmx::ArrayRef instead of std::vector
void MarkovModel::countTransitions(std::vector<int>& discretizedTraj, int lag)
{
    // Extract time-lagged trajectories
    std::vector<int> rows(discretizedTraj.begin(), discretizedTraj.end() - lag);
    std::vector<int> cols(discretizedTraj.begin() + lag, discretizedTraj.end());

    // Iterate over trajectory and count transitions
    for (int i = 0; i < rows.size(); i++)
    {
        transitionCountsMatrix(rows[i], cols[i]) += 1;
    }
}

void MarkovModel::computeTransitionProbabilities()
{
    // Construct a transition probability matrix from a transition counts matrix
    // T_ij = c_ij/c_i; where c_i=sum_j(c_ij)
    // TODO: implement reversibility

    // Use a float here to enable float division. Could there be issues having a float counter?
    // TODO: cast rowsum in for loop instead? Compiler might be able to sort inefficiencies
    // TODO: use std::accumulate to get counts
    float rowsum;
    for (int i = 0; i < transitionCountsMatrix.extent(0); i++)
    {
    rowsum = 0;
    for (int j = 0; j < transitionCountsMatrix.extent(1); j++)
    {
        rowsum += transitionCountsMatrix(i, j);
    }

    // Make sure sum is positive, ignore sums of zero to avoid zero-division
    GMX_ASSERT(rowsum >= 0, "Sum of transition counts must be positive!");
        if (rowsum != 0 ){
            // Once we have the rowsum, loop through the row again
            for (int k = 0; k < transitionCountsMatrix.extent(1); k++)
            {
                transitionProbabilityMatrix(i, k) = transitionCountsMatrix(i ,k) / rowsum;
            }
        }
    }
}

// TODO: add sparse solver
// TODO: maybe there's a better way to handle function argument?
void MarkovModel::diagonalizeMatrix(MultiDimArray<std::vector<real>, extents<dynamic_extent, dynamic_extent>> matrix)
{
    // Create vector to store eigenvectors and eigenvalues
    GMX_ASSERT(matrix.extent(0) == matrix.extent(1), "Matrix must be square!");
    int dim = matrix.extent(0);

    eigenvalues.resize(dim, 0);
    eigenvectors.resize(dim * dim, 0);

    auto tmpMat = matrix.toArrayRef().data();

    eigensolver(tmpMat, dim, 0, dim, eigenvalues.data(), eigenvectors.data());
}

} // namespace gmx
