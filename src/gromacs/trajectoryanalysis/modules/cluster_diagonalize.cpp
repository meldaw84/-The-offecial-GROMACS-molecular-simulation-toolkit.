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

#include "gmxpre.h"

#include "cluster_diagonalize.h"

#include <cstring>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#include "gromacs/gmxana/cmat.h"
#include "gromacs/linearalgebra/eigensolver.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/logger.h"

namespace gmx
{

ArrayRef<const int> ClusterDiagonalize::clusterList() const
{
    GMX_THROW(InternalError("Cannot access cluster list for simple diagonalization"));
}

ArrayRef<const real> ClusterDiagonalize::eigenvalues() const
{
    if (!finished_)
    {
        GMX_THROW(InternalError("Cannot access eigenvalues until they have been calculated"));
    }
    return eigenvalues_;
}

void ClusterDiagonalize::makeClusters()
{
    if (finished_)
    {
        GMX_THROW(InternalError("Should not cluster again after doing it once"));
    }

    eigenvalues_.resize(numStructures_);
    std::vector<real> eigenvectors(numStructures_ * numStructures_);

    std::memcpy(eigenvectors.data(), matrix_->mat[0], numStructures_ * numStructures_ * sizeof(real));
    eigensolver(eigenvectors.data(), numStructures_, 0, numStructures_, eigenvalues_.data(), matrix_->mat[0]);
    finished_ = true;
}

} // namespace gmx
