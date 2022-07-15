/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2022- The GROMACS Authors
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

/*! \internal \file
 *  \brief Implements GPU 3D FFT routines for SYCL.
 *
 *  \author Carsten Uphoff <carsten.uphoff@intel.com>
 *  \ingroup module_fft
 */

#include "gmxpre.h"

#include "gpu_3dfft_sycl_bbfft.h"

#include "config.h"

#include "gromacs/gpu_utils/devicebuffer_sycl.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"

#include <bbfft/bad_configuration.hpp>
#include <bbfft/configuration.hpp>

class DeviceContext;

#if (!GMX_SYCL_DPCPP)
#    error This file is only supported with Intel DPC++ compiler
#endif

#include <cstddef>

namespace gmx
{

Gpu3dFft::ImplSyclBbfft::ImplSyclBbfft(bool allocateGrids,
                                       MPI_Comm /*comm*/,
                                       ArrayRef<const int> gridSizesInXForEachRank,
                                       ArrayRef<const int> gridSizesInYForEachRank,
                                       int /*nz*/,
                                       const bool /*performOutOfPlaceFFT*/,
                                       const DeviceContext& context,
                                       const DeviceStream&  pmeStream,
                                       ivec                 realGridSize,
                                       ivec                 realGridSizePadded,
                                       ivec                 complexGridSizePadded,
                                       DeviceBuffer<float>* realGrid,
                                       DeviceBuffer<float>* complexGrid) :
    realGrid_(*realGrid->buffer_), queue_(pmeStream.stream())
{
    GMX_RELEASE_ASSERT(!allocateGrids, "Grids needs to be pre-allocated");
    GMX_RELEASE_ASSERT(gridSizesInXForEachRank.size() == 1 && gridSizesInYForEachRank.size() == 1,
                       "Multi-rank FFT decomposition not implemented with SYCL MKL backend");

    GMX_ASSERT(checkDeviceBuffer(*realGrid,
                                 realGridSizePadded[XX] * realGridSizePadded[YY] * realGridSizePadded[ZZ]),
               "Real grid buffer is too small for the declared padded size");

    GMX_ASSERT(checkDeviceBuffer(*complexGrid,
                                 complexGridSizePadded[XX] * complexGridSizePadded[YY]
                                         * complexGridSizePadded[ZZ] * 2),
               "Complex grid buffer is too small for the declared padded size");
    allocateComplexGrid(complexGridSizePadded, realGrid, complexGrid, context);
    complexGrid_ = *complexGrid->buffer_;

    std::array<size_t, bbfft::max_tensor_dim> shape   = { 1,
                                                        static_cast<size_t>(realGridSize[ZZ]),
                                                        static_cast<size_t>(realGridSize[YY]),
                                                        static_cast<size_t>(realGridSize[XX]),
                                                        1 };
    std::array<size_t, bbfft::max_tensor_dim> rstride = {
        1,
        1,
        static_cast<size_t>(realGridSizePadded[ZZ]),
        static_cast<size_t>(realGridSizePadded[ZZ] * realGridSizePadded[YY]),
        static_cast<size_t>(realGridSizePadded[ZZ] * realGridSizePadded[YY] * realGridSizePadded[XX])
    };
    std::array<size_t, bbfft::max_tensor_dim> cstride = {
        1,
        1,
        static_cast<size_t>(complexGridSizePadded[ZZ]),
        static_cast<size_t>(complexGridSizePadded[ZZ] * complexGridSizePadded[YY]),
        static_cast<size_t>(complexGridSizePadded[ZZ] * complexGridSizePadded[YY] * complexGridSizePadded[XX])
    };

    try
    {
        bbfft::configuration cfg = {
            3, shape, bbfft::direction::forward, bbfft::transform_type::r2c, rstride, cstride
        };
        r2cDescriptor_ = bbfft::make_plan<bbfft::runtime::sycl, float>(cfg, queue_);
    }
    catch (bbfft::bad_configuration& exc)
    {
        GMX_THROW(InternalError(
                formatString("bbfft failure while configuring R2C descriptor: %s", exc.what())));
    }

    try
    {
        bbfft::configuration cfg = {
            3, shape, bbfft::direction::backward, bbfft::transform_type::c2r, cstride, rstride
        };
        c2rDescriptor_ = bbfft::make_plan<bbfft::runtime::sycl, float>(cfg, queue_);
    }
    catch (bbfft::bad_configuration& exc)
    {
        GMX_THROW(InternalError(
                formatString("bbfft failure while configuring C2R descriptor: %s", exc.what())));
    }
}

Gpu3dFft::ImplSyclBbfft::~ImplSyclBbfft() {
    deallocateComplexGrid();
}

void Gpu3dFft::ImplSyclBbfft::perform3dFft(gmx_fft_direction dir, CommandEvent* /*timingEvent*/)
{
    switch (dir)
    {
        case GMX_FFT_REAL_TO_COMPLEX: r2cDescriptor_.execute(realGrid_, complexGrid_); break;
        case GMX_FFT_COMPLEX_TO_REAL: c2rDescriptor_.execute(complexGrid_, realGrid_); break;
        default:
            GMX_THROW(NotImplementedError("The chosen 3D-FFT case is not implemented on GPUs"));
    }
}

} // namespace gmx
