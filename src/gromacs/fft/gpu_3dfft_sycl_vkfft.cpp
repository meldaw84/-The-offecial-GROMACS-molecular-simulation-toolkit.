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

/*! \internal \file
 *  \brief Implements GPU 3D FFT routines for HIP.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \ingroup module_fft
 */

#include "gmxpre.h"
//#include <vkfft/vkFFT.h>
//#include <vkfft/vkFFT.h>
//#include <vkfft/vkFFT.h>

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"

#include "gpu_3dfft_sycl_vkfft.h"

namespace gmx
{

Gpu3dFft::ImplSyclVkfft::ImplSyclVkfft(bool allocateGrids,
                                       MPI_Comm /*comm*/,
                                       ArrayRef<const int> gridSizesInXForEachRank,
                                       ArrayRef<const int> gridSizesInYForEachRank,
                                       const int /*nz*/,
                                       bool                 performOutOfPlaceFFT,
                                       const DeviceContext& context,
                                       const DeviceStream&  pmeStream,
                                       ivec                 realGridSize,
                                       ivec                 realGridSizePadded,
                                       ivec                 complexGridSizePadded,
                                       DeviceBuffer<float>* realGrid,
                                       DeviceBuffer<float>* complexGrid) :
    Gpu3dFft::Impl::Impl(performOutOfPlaceFFT), realGrid_(*realGrid), queue_(pmeStream.stream())
{
    GMX_RELEASE_ASSERT(allocateGrids == false, "Grids needs to be pre-allocated");
    GMX_RELEASE_ASSERT(gridSizesInXForEachRank.size() == 1 && gridSizesInYForEachRank.size() == 1,
                       "FFT decomposition not implemented with cuFFT backend");

    allocateComplexGrid(complexGridSizePadded, realGrid, complexGrid, context);


    GMX_RELEASE_ASSERT(realGrid, "Bad (null) input real-space grid");
    GMX_RELEASE_ASSERT(complexGrid, "Bad (null) input complex grid");


    configuration         = {};
    launchParams          = {};
    appR2C                = {};
    configuration.FFTdim  = 3;
    configuration.size[0] = realGridSize[ZZ];
    configuration.size[1] = realGridSize[YY];
    configuration.size[2] = realGridSize[XX];

    configuration.performR2C  = 1;
    queue_device              = sycl::get_native<sycl::backend::hip>(queue_.get_device());
    configuration.device      = &queue_device;
    configuration.stream      = &backend_stream;
    configuration.num_streams = 1;

    bufferSize = complexGridSizePadded[XX] * complexGridSizePadded[YY] * complexGridSizePadded[ZZ]
                 * sizeof(float) * 2;
    configuration.bufferSize      = &bufferSize;
    configuration.aimThreads      = 64;
    configuration.bufferStride[0] = complexGridSizePadded[ZZ];
    configuration.bufferStride[1] = complexGridSizePadded[ZZ] * complexGridSizePadded[YY];
    configuration.bufferStride[2] =
            complexGridSizePadded[ZZ] * complexGridSizePadded[YY] * complexGridSizePadded[XX];

    configuration.isInputFormatted           = 1;
    configuration.inverseReturnToInputBuffer = 1;
    inputBufferSize =
            realGridSizePadded[XX] * realGridSizePadded[YY] * realGridSizePadded[ZZ] * sizeof(float);
    configuration.inputBufferSize      = &inputBufferSize;
    configuration.inputBufferStride[0] = realGridSizePadded[ZZ];
    configuration.inputBufferStride[1] = realGridSizePadded[ZZ] * realGridSizePadded[YY];
    configuration.inputBufferStride[2] =
            realGridSizePadded[ZZ] * realGridSizePadded[YY] * realGridSizePadded[XX];
    queue_.submit(sycl::property::command_group::hipSYCL_coarse_grained_events{},[&](sycl::handler& cgh) {
              cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle& gmx_unused h) {
                  VkFFTResult resFFT = initializeVkFFT(&appR2C, configuration);
                  if (resFFT != VKFFT_SUCCESS)
                      printf("VkFFT error: %d\n", resFFT);
              });
          }).wait();
}

Gpu3dFft::ImplSyclVkfft::~ImplSyclVkfft()
{
    deleteVkFFT(&appR2C);
    deallocateComplexGrid();
}

void Gpu3dFft::ImplSyclVkfft::perform3dFft(gmx_fft_direction dir, CommandEvent* /*timingEvent*/)
{
#if !GMX_SYCL_USE_USM
    sycl::buffer<float, 1> complexGidBuffer = *complexGrid_.buffer_.get();
    sycl::buffer<float, 1> realGridBuffer   = *realGrid_.buffer_.get();
#endif
    queue_.submit(sycl::property::command_group::hipSYCL_coarse_grained_events{},[&](sycl::handler& cgh) {
#if !GMX_SYCL_USE_USM
        auto complexGridAccessor = complexGidBuffer.get_access(cgh, sycl::read_write);
        auto realGridAccessor    = realGridBuffer.get_access(cgh, sycl::read_write);
#endif
        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle& h) {
#if GMX_SYCL_USE_USM
            void* d_complexGrid = reinterpret_cast<void*>(complexGrid_.buffer_->ptr_);
            void* d_realGrid    = reinterpret_cast<void*>(realGrid_.buffer_->ptr_);
#else
            void* d_complexGrid = h.get_native_mem<sycl::backend::hip>(complexGridAccessor);
            void* d_realGrid    = h.get_native_mem<sycl::backend::hip>(realGridAccessor);
#endif
            hipStream_t stream = h.get_native_queue<sycl::backend::hip>();
            // based on: https://github.com/DTolm/VkFFT/issues/78
            appR2C.configuration.stream = &stream;
            launchParams.inputBuffer    = &d_realGrid;
            launchParams.buffer         = &d_complexGrid;
            VkFFTResult resFFT          = VKFFT_SUCCESS;
            if (dir == GMX_FFT_REAL_TO_COMPLEX)
            {
                resFFT = VkFFTAppend(&appR2C, -1, &launchParams);
                if (resFFT != VKFFT_SUCCESS)
                    printf("VkFFT error: %d\n", resFFT);
            }
            else
            {
                resFFT = VkFFTAppend(&appR2C, 1, &launchParams);
                if (resFFT != VKFFT_SUCCESS)
                    printf("VkFFT error: %d\n", resFFT);
            }
        });
    });
}

} // namespace gmx
