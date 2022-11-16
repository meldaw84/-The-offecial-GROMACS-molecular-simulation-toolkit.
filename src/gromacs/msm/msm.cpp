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

#include "gromacs/gmxana/cmat.h"
//#include "gromacs/utility/smalloc.h"


namespace gmx
{

namespace
{

bool matrixIsSquare(const MatrixNxM & matrix)
{
    return matrix.extent(0) == matrix.extent(1);
};

} // namespace

DiagonalizationResult::DiagonalizationResult(int dimension):eigenvalues_(dimension), eigenvectors_(dimension, dimension){}

DiagonalizationResult diagonalize(MatrixNxM matrix)
{
    // Create vector to store eigenvectors and eigenvalues
    GMX_ASSERT(matrixIsSquare(matrix), "Matrix must be square!");

    const int dim = matrix.extent(0);
    DiagonalizationResult result(matrix.extent(0));

    printf("Matrix dimensions: %d, %d\n", matrix.extent(0), matrix.extent(1));

    eigensolver(matrix.asView().data(), dim, 0, dim, result.eigenvalues_.data(), result.eigenvectors_.asView().data());
    return result;
}

// Constructor
MarkovModel::MarkovModel(int nstates)
{
    // TODO: scan clustered trajectory of highest state value?
    // TODO: use brace initialization?

    // Initialize the TCM and TPM and set the size
    transitionCountsMatrix.resize(nstates, nstates);
    transitionProbabilityMatrix.resize(nstates, nstates);

    const auto& dataView = transitionProbabilityMatrix.asConstView();
    const int numRows = transitionProbabilityMatrix.extent(0);
    const int numCols = transitionProbabilityMatrix.extent(1);

    for (int i = 0; i < numRows; i++)
    {
        printf("\n");
        for (int j=0; j < numCols; j++)
        {
            printf("%f ", dataView[i][j]);
        }
    }
    printf("\n");

    // Initialize eigenvalues and eigenvectors here?
    //eigenvalues.resize(nstates);
    //eigenvectors.resize(nstates * nstates);
}

void MarkovModel::countTransitions(gmx::ArrayRef<int> discretizedTraj, int lag)
{
    // Extract time-lagged trajectories
    std::vector<int> rows(discretizedTraj.begin(), discretizedTraj.end() - lag);
    std::vector<int> cols(discretizedTraj.begin() + lag, discretizedTraj.end());

    // Iterate over trajectory and count transitions
    for (int i = 0; i < rows.size(); i++)
    {
        transitionCountsMatrix(rows[i], cols[i]) += 1;
    }

    const auto& dataView = transitionCountsMatrix.asConstView();
    const int numRows = transitionCountsMatrix.extent(0);
    const int numCols = transitionCountsMatrix.extent(1);

    for (int i = 0; i < numRows; i++)
    {
        printf("\n");
        for (int j=0; j < numCols; j++)
        {
            printf("%d ", dataView[i][j]);
        }
    }
    printf("\n");
}

void MarkovModel::computeTransitionProbabilities()
{
    // Construct a transition probability matrix from a transition counts matrix
    // T_ij = c_ij/c_i; where c_i=sum_j(c_ij)
    // TODO: implement reversibility

    // Use a float here to enable float division. Could there be issues having a float counter?
    // TODO: cast rowsum in for loop instead? Compiler might be able to sort inefficiencies
    // TODO: use std::accumulate to get counts

    const auto& dataView = transitionProbabilityMatrix.asConstView();
    const int numRows = transitionProbabilityMatrix.extent(0);
    const int numCols = transitionProbabilityMatrix.extent(1);

    printf("IN TPM\n");

    for (int i = 0; i < numRows; i++)
    {
        printf("\n");
        for (int j=0; j < numCols; j++)
        {
            printf("%f ", dataView[i][j]);
        }
    }
    printf("\n");

    real rowsum;
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
                printf("Rowsum: %f\n", rowsum);
                printf("TPM val: %f\n", transitionProbabilityMatrix(i,k));
                transitionProbabilityMatrix(i, k) = transitionCountsMatrix(i ,k) / rowsum;
            }
        }
    }
    const auto& dataView2 = transitionProbabilityMatrix.asConstView();
    for (int i = 0; i < numRows; i++)
    {
        printf("\n");
        for (int j=0; j < numCols; j++)
        {
            printf("%f ", dataView2[i][j]);
        }
    }
    printf("\n");
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

    const auto& dataView = transitionProbabilityMatrix.asConstView();
    const int numRows = transitionProbabilityMatrix.extent(0);
    const int numCols = transitionProbabilityMatrix.extent(1);

    printf("IN DIAG\n");

    for (int i = 0; i < numRows; i++)
    {
        printf("\n");
        for (int j=0; j < numCols; j++)
        {
            printf("%f ", dataView[i][j]);
        }
    }
    printf("\n");

    //auto tmpMat = matrix.toArrayRef().data();

    /*
    for (int i = 0; i < tmpMat.size(); i++)
    {
        printf("tmpMat elm: %f\n", tmpMat[i]);
    }
    */

    // We shouldn't need this...
    /*
    t_mat* t_matrix = init_mat(dim * dim, FALSE);

    for (int i = 0; i < dim * dim; i++)
    {
        printf("Matrix: %f\n", *t_matrix->mat[0]);
    }

    */
    //real* *mat;
    //snew(mat, 4 * 4);

    // Didn't make a difference
    //std::vector<real> mat = {0.8, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5};

    //std::memcpy(eigenvectors.data(), t_matrix->mat[0], dim * dim * sizeof(real));

    //eigensolver(eigenvectors.data(), dim, 0, dim, eigenvalues.data(), t_matrix->mat[0]);
    //std::memcpy(eigenvectors.data(), matrix.toArrayRef().data(), dim * dim * sizeof(real));
    //eigensolver(eigenvectors.data(), dim, 0, dim, eigenvalues.data(), matrix.toArrayRef().data());
    eigensolver(transitionProbabilityMatrix.asView().data(), dim, 0, dim, eigenvalues.data(), eigenvectors.data());
    //eigensolver(t_matrix->mat[0], dim, 0, dim, eigenvalues.data(), eigenvectors.data());
    //eigensolver(mat.data(), dim, 0, dim, eigenvalues.data(), eigenvectors.data());
    //eigensolver(mat.data(), dim, 0, dim, eigenvalues.data(), eigenvectors.data());
    printf("Number of eigenvalues: %d\n", eigenvalues.size());
    printf("Number of eigenvector components: %d\n", eigenvectors.size());
}

void MarkovModel::WriteOutput()
{

}

} // namespace gmx
