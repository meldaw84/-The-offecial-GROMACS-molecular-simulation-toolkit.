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
  // TODO: use brace initialization

  // Initialize the TCM and TPM and set the size
  transitionCountsMatrix.resize(nstates, nstates);
  transitionProbabilityMatrix.resize(nstates, nstates);

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
  // TODO: use std::accumulate to get counts
  float rowsum;
  for (int i = 0; i < transitionCountsMatrix.extent(0); i++)
  {
    rowsum = 0;
    for (int j = 0; j < transitionCountsMatrix.extent(1); j++){
      rowsum += transitionCountsMatrix(i, j);
    }

    // Make sure sum is positive, ignore sums of zero to avoid zero-division
    if ( rowsum >= 0 ) {
      if (rowsum != 0 ){
        // Once we have the rowsum, loop through the row again
        for (int k = 0; k < transitionCountsMatrix.extent(1); k++){
          transitionProbabilityMatrix(i, k) = transitionCountsMatrix(i ,k) / rowsum;
        }
      }
    }
    else {
      // TODO: Better to check after value is created rather than when it's used
      // TODO: Assert error instead
      GMX_THROW(InternalError("Sum of transition counts must be positive!"));
    }
  }
}

// TODO: make it more general which attribute matrix we want to diagonalize?
// TODO: add sparse solver
// TODO: think about design here!
void MarkovModel::diagonalizeTPM()
{
  // Create vector to store eigenvectors and eigenvalues
  int dim = transitionProbabilityMatrix.extent(0);
  std::vector<real> eigenvalues(dim);
  std::vector<real> eigenvectors(dim * dim);

  auto tmpTPM = transitionProbabilityMatrix.toArrayRef().data();

  eigensolver(tmpTPM, dim, 0, dim, eigenvalues.data(), eigenvectors.data());

  // TODO: move to unit test
  for (int i=0; i<eigenvalues.size(); ++i){
    printf("Val elm %d: %f\n", i, eigenvalues[i]);
  }

  for (int i=0; i<eigenvectors.size(); ++i){
    printf("Vec elm %d: %f\n", i, eigenvectors[i]);
  }
}

} // namespace gmx
