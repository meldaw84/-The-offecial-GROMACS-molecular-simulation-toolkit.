/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2016- The GROMACS Authors
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
 *  \brief Implements GPU 3D FFT routines for CUDA.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \ingroup module_fft
 */

#include "gmxpre.h"
#include <cufftXt.h>
#include "gpu_3dfft_cufft.h"

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"

#include <iostream>

namespace gmx
{
static void handleCufftError(cufftResult_t status, const char* msg)
{
    if (status != CUFFT_SUCCESS)
    {
        gmx_fatal(FARGS, "%s (error code %d)\n", msg, status);
    }
}

Gpu3dFft::ImplCuFft::ImplCuFft(bool allocateRealGrid,
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
                               DeviceBuffer<__half>* realGrid,
                               DeviceBuffer<__half>* complexGrid) :
    Gpu3dFft::Impl::Impl(performOutOfPlaceFFT)
{
    GMX_RELEASE_ASSERT(allocateRealGrid == true, "Grids cannot be pre-allocated");
    GMX_RELEASE_ASSERT(gridSizesInXForEachRank.size() == 1 && gridSizesInYForEachRank.size() == 1,
                       "FFT decomposition not implemented with cuFFT backend");

    const int complexGridSizePaddedTotal =
            complexGridSizePadded[XX] * complexGridSizePadded[YY] * complexGridSizePadded[ZZ];
    const int realGridSizePaddedTotal =
            realGridSizePadded[XX] * realGridSizePadded[YY] * realGridSizePadded[ZZ];

    allocateDeviceBuffer(realGrid, realGridSizePaddedTotal, context);
    allocateComplexGrid(complexGridSizePadded, realGrid, complexGrid, context);

    realGrid_ = reinterpret_cast<__half*>(*realGrid);

    GMX_RELEASE_ASSERT(realGrid_, "Bad (null) input real-space grid");
    GMX_RELEASE_ASSERT(complexGrid_, "Bad (null) input complex grid");

    std::cout << "FFT grid " << realGridSize[XX] << " " << realGridSize[YY] << " " << realGridSize[ZZ] << std::endl;

    cufftResult_t result;
    /* Commented code for a simple 3D grid with no padding */
    /*
       result = cufftPlan3d(&planR2C_, realGridSize[XX], realGridSize[YY], realGridSize[ZZ],
       CUFFT_R2C); handleCufftError(result, "cufftPlan3d R2C plan failure");

       result = cufftPlan3d(&planC2R_, realGridSize[XX], realGridSize[YY], realGridSize[ZZ],
       CUFFT_C2R); handleCufftError(result, "cufftPlan3d C2R plan failure");
     */

    const int rank = 3, batch = 1;
    size_t workSize = 0;
    result = cufftCreate(&planR2C_);
    handleCufftError(result, "cufftCreate failure");

    long long int rgsize[DIM];
    rgsize[0] = realGridSize[0];
    rgsize[1] = realGridSize[1];
    rgsize[2] = realGridSize[2];

    long long int rgsizepad[DIM];
    rgsizepad[0] = realGridSizePadded[0];
    rgsizepad[1] = realGridSizePadded[1];
    rgsizepad[2] = realGridSizePadded[2];

    long long int cgsizepad[DIM];
    cgsizepad[0] = complexGridSizePadded[0];
    cgsizepad[1] = complexGridSizePadded[1];
    cgsizepad[2] = complexGridSizePadded[2];

    result = cufftXtMakePlanMany(planR2C_, rank, rgsize, rgsizepad, 1, realGridSizePaddedTotal,
                                CUDA_R_16F, cgsizepad, 1, complexGridSizePaddedTotal, CUDA_C_16F, batch, 
                                &workSize, CUDA_C_16F);
    handleCufftError(result, "cufftXtMakePlanMany R2C plan failure");

    // result = cufftPlanMany(&planR2C_,
    //                        rank,
    //                        realGridSize,
    //                        realGridSizePadded,
    //                        1,
    //                        realGridSizePaddedTotal,
    //                        complexGridSizePadded,
    //                        1,
    //                        complexGridSizePaddedTotal,
    //                        CUFFT_R2C,
    //                        batch);
    // handleCufftError(result, "cufftPlanMany R2C plan failure");

    result = cufftCreate(&planC2R_);
    handleCufftError(result, "cufftCreate failure");

    result = cufftXtMakePlanMany(planC2R_, rank, rgsize, cgsizepad, 1, complexGridSizePaddedTotal,
                                CUDA_C_16F, rgsizepad, 1, realGridSizePaddedTotal, CUDA_R_16F, batch, 
                                &workSize, CUDA_R_16F);
    handleCufftError(result, "cufftXtMakePlanMany C2R plan failure");

    // result = cufftPlanMany(&planC2R_,
    //                        rank,
    //                        realGridSize,
    //                        complexGridSizePadded,
    //                        1,
    //                        complexGridSizePaddedTotal,
    //                        realGridSizePadded,
    //                        1,
    //                        realGridSizePaddedTotal,
    //                        CUFFT_C2R,
    //                        batch);
    // handleCufftError(result, "cufftPlanMany C2R plan failure");

    cudaStream_t stream = pmeStream.stream();
    GMX_RELEASE_ASSERT(stream, "Can not use the default CUDA stream for PME cuFFT");

    result = cufftSetStream(planR2C_, stream);
    handleCufftError(result, "cufftSetStream R2C failure");

    result = cufftSetStream(planC2R_, stream);
    handleCufftError(result, "cufftSetStream C2R failure");
}

Gpu3dFft::ImplCuFft::~ImplCuFft()
{
    deallocateComplexGrid();

    cufftResult_t result;
    result = cufftDestroy(planR2C_);
    handleCufftError(result, "cufftDestroy R2C failure");
    result = cufftDestroy(planC2R_);
    handleCufftError(result, "cufftDestroy C2R failure");
}

void Gpu3dFft::ImplCuFft::perform3dFft(gmx_fft_direction dir, CommandEvent* /*timingEvent*/)
{
    cufftResult_t result;
    if (dir == GMX_FFT_REAL_TO_COMPLEX)
    {
        result = cufftXtExec(planR2C_, realGrid_, complexGrid_, CUFFT_FORWARD);
        handleCufftError(result, "cuFFT R2C execution failure");
    }
    else
    {
        result = cufftXtExec(planC2R_, complexGrid_, realGrid_, CUFFT_INVERSE);
        handleCufftError(result, "cuFFT C2R execution failure");
    }
}

} // namespace gmx
