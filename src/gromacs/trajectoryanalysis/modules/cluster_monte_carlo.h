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

#ifndef GMX_GMXANA_CLUSTER_MONTE_CARLO_H
#define GMX_GMXANA_CLUSTER_MONTE_CARLO_H

#include <stdio.h>

#include <vector>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

#include "icluster.h"

struct t_clusters;
struct t_mat;

namespace gmx
{

class MDLogger;

class ClusterMonteCarlo : public ICluster
{
public:
    explicit ClusterMonteCarlo(t_mat*               inputMatrix,
                               real                 kT,
                               int                  seed,
                               int                  maxIterations,
                               int                  randomIterations,
                               const MDLogger&      logger,
                               ArrayRef<const real> time) :
        finished_(false),
        kT_(kT),
        seed_(seed),
        maxIterations_(maxIterations),
        randomIterations_(randomIterations),
        matrix_(inputMatrix),
        logger_(logger),
        time_(time)
    {
        makeClusters();
    }
    ~ClusterMonteCarlo() override = default;

    ArrayRef<const int> clusterList() const override;

private:
    //! Perform actual clustering.
    void makeClusters();
    //! Did we perform the clustering?
    bool finished_;
    //! Boltzmann weigthing.
    const real kT_;
    //! Random seed for monte carlo.
    const int seed_;
    //! Maximum number of MC steps.
    const int maxIterations_;
    //! Number of fully random steps.
    const int randomIterations_;
    //! Handle to cluster matrix.
    t_mat* matrix_;
    //! Cluster indices
    std::vector<int> clusters_;
    //! Logger handle
    const MDLogger& logger_;
    //! Time points for frames.
    ArrayRef<const real> time_;
};

} // namespace gmx

#endif
