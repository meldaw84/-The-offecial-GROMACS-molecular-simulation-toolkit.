/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2017,2018,2019,2021, by the GROMACS development team, led by
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
 *  \brief Declares the GPU 3D FFT routines.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \author Gaurav Garg <gaugarg@nvidia.com>
 *  \ingroup module_fft
 */

#ifndef GMX_FFT_GPU_3DFFT_CUFFT_H
#define GMX_FFT_GPU_3DFFT_CUFFT_H

#include <memory>

#include "gromacs/fft/fft.h"
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/utility/gmxmpi.h"
#include "gpu_3dfft_impl.h"

#include <cufft.h>

#define UCX_MPIALLTOALLV_BUG_HACK 1

class DeviceContext;
class DeviceStream;

namespace gmx
{

/*! \internal \brief
 * A 3D FFT wrapper class for performing R2C/C2R transforms using cuFFT
 */
class Gpu3dFft::ImplCuFft : public Gpu3dFft::Impl
{
public:
    //! \copydoc Gpu3dFft::Impl::Impl
    ImplCuFft(bool                 allocateGrids,
              MPI_Comm             comm,
              ArrayRef<const int>  gridSizesInXForEachRank,
              ArrayRef<const int>  gridSizesInYForEachRank,
              int                  nz,
              bool                 performOutOfPlaceFFT,
              const DeviceContext& context,
              const DeviceStream&  pmeStream,
              ivec                 realGridSize,
              ivec                 realGridSizePadded,
              ivec                 complexGridSizePadded,
              DeviceBuffer<float>* realGrid,
              DeviceBuffer<float>* complexGrid);

    //! \copydoc Gpu3dFft::Impl::~Impl
    ~ImplCuFft() override;

    //! \copydoc Gpu3dFft::Impl::perform3dFft
    void perform3dFft(gmx_fft_direction dir, CommandEvent* timingEvent) override;

private:
    cufftHandle   planR2C_;
    cufftHandle   planC2R_;
    cufftReal*    realGrid_;
    cufftComplex* complexGrid_;
    cufftComplex* complexGrid2_;
    ivec complexGridSizePadded_;

    std::vector<int> _gridSizesInX;

    /*! \brief
     * CUDA stream used for PME computation
     */
    const DeviceStream& stream_;

    /*! \brief
     * 2D and 1D cufft plans used for distributed fft implementation
     */
    cufftHandle planR2C2D_;
    cufftHandle planC2R2D_;
    cufftHandle planC2C1D_;

    /*! \brief
     * MPI complex type
     */
    MPI_Datatype complexType_;

    /*! \brief
     * MPI communicator for PME ranks
     */
    MPI_Comm mpi_comm_;

    /*! \brief
     * total ranks within PME group
     */
    int mpiSize_;

    /*! \brief
     * current local mpi rank within PME group
     */
    int mpiRank_;

    /*! \brief
     * Max local grid size in X-dim (used during transposes in forward pass)
     */
    int xMax_;

    /*! \brief
     * Max local grid size in Y-dim (used during transposes in reverse pass)
     */
    int yMax_;

    /*! \brief
     * device array containing 1D decomposition size in X-dim (forwarad pass)
     */
    DeviceBuffer<int> d_xBlockSizes_;

    /*! \brief
     * device array containing 1D decomposition size in Y-dim (reverse pass)
     */
    DeviceBuffer<int> d_yBlockSizes_;

    /*! \brief
     * device arrays for local interpolation grid start values in X-dim
     * (used during transposes in forward pass)
     */
    DeviceBuffer<int> d_s2g0x_;

    /*! \brief
     * device arrays for local interpolation grid start values in Y-dim
     * (used during transposes in reverse pass)
     */
    DeviceBuffer<int> d_s2g0y_;

    /*! \brief
     * host array containing 1D decomposition size in X-dim (forwarad pass)
     */
    gmx::HostVector<int> h_xBlockSizes_;

    /*! \brief
     * host array containing 1D decomposition size in Y-dim (reverse pass)
     */
    gmx::HostVector<int> h_yBlockSizes_;

    /*! \brief
     * host array for local interpolation grid start values in Y-dim
     */
    gmx::HostVector<int> h_s2g0y_;

    /*! \brief
     * device array big enough to hold grid overlapping region
     * used during grid halo exchange
     */
    DeviceBuffer<float> d_transferGrid_;

    /*! \brief
     * count and displacement arrays used in MPI_Alltoall call
     *
     */
    int *sendCount_, *sendDisp_;
    int *recvCount_, *recvDisp_;

#        if UCX_MPIALLTOALLV_BUG_HACK
    /*! \brief
     * count arrays used in MPI_Alltoall call which has no self copies
     *
     */
    int *sendCountTemp_, *recvCountTemp_;
#        endif
};

} // namespace gmx

#endif
