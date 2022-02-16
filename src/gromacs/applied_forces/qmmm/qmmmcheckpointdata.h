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
 * Declares QMMMCheckpointData strucutre responsible for holding and managing
 * all parameters which should be saved/loaded with checkpointing
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_QMMMCHECKPOINTDATA_H
#define GMX_APPLIED_FORCES_QMMMCHECKPOINTDATA_H

#include <string>

#include "gromacs/fileio/checkpoint.h"
#include "gromacs/math/vectypes.h"

#include "qmmmtypes.h"

namespace gmx
{

/*! \internal
 * \brief Storage of internal data, that needs to be saved/loaded
 * with checkpointing notifications
 */
class QMMMCheckpointData
{
public:
    /*! \brief Initialize internal QMMM data with QMMMParameters instance
     * \param[in] parameters QMMM parameters which have been read from the tpr file
     */
    void initData(const QMMMParameters& parameters);

    /*! \brief Write internal QMMM data into a checkpoint key value tree.
     * \param[in] kvtBuilder enables writing to the Key-Value-Tree
     * \param[in] identifier denotes the module that is checkpointing the data
     */
    void writeData(KeyValueTreeObjectBuilder kvtBuilder, const std::string& identifier) const;

    /*! \brief Read the internal parameters from the checkpoint file on master
     * \param[in] kvtData holding the checkpoint information
     * \param[in] identifier identifies the data in a key-value-tree
     */
    void readData(const KeyValueTreeObject& kvtData, const std::string& identifier);

    /*! \brief Broadcast the internal parameters.
     * \param[in] communicator to broadcast the state information
     * \param[in] isParallelRun to determine if anything has to be broadcast at all
     */
    void broadcastData(MPI_Comm communicator, bool isParallelRun);

    /*! \brief Accessor methods for stored data
     */
    //! Get qmTrans_ instance
    const RVec& qmTrans() const;

    //! Set qmTrans_ instance
    void setQmTrans(const RVec& trans);

private:
    //! Translation vector to center QM subsystem inside the QM Box
    RVec qmTrans_;

    /*! \brief This tags for parameters which will be stored in *.cpt file via KVT
     * \note Changing this tags will break backwards compability for checkpoint file writing.
     */
    //! \{
    const std::string c_qmTransTag = "qmtrans";
    //! \}
};

} // namespace gmx

#endif
