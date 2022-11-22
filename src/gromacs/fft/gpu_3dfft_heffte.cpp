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
 *  \brief Implements GPU 3D FFT routines using HeFFTe.
 *
 *  \author Gaurav Garg <gaugarg@nvidia.com>
 *  \ingroup module_fft
 */

#include "gmxpre.h"

#include "gpu_3dfft_heffte.h"

#include <iostream>

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"

#if GMX_GPU_SYCL
#    include "gromacs/gpu_utils/devicebuffer_sycl.h"
#endif

namespace gmx
{

static auto getNativeStream(sycl::queue q)
{
#if GMX_SYCL_HIPSYCL
#    if GMX_HIPSYCL_HAVE_CUDA_TARGET
    cudaStream_t   stream;
    constexpr auto backend = sycl::backend::cuda;
#    elif GMX_HIPSYCL_HAVE_HIP_TARGET
    hipStream_t    stream;
    constexpr auto backend = sycl::backend::hip;
#    endif
    q.submit([&](sycl::handler& cgh) {
         cgh.hipSYCL_enqueue_custom_operation(
                 [=, &stream](sycl::interop_handle& h) { stream = h.get_native_queue<backend>(); });
     }).wait();
    return stream;
#else
    return q;
#endif
}


template<typename backend_tag>
Gpu3dFft::ImplHeFfte<backend_tag>::ImplHeFfte(bool                 allocateRealGrid,
                                              MPI_Comm             comm,
                                              ArrayRef<const int>  gridSizesInXForEachRank,
                                              ArrayRef<const int>  gridSizesInYForEachRank,
                                              const int            nz,
                                              bool                 performOutOfPlaceFFT,
                                              const DeviceContext& context,
                                              const DeviceStream&  pmeStream,
                                              ivec                 realGridSize,
                                              ivec                 realGridSizePadded,
                                              ivec                 complexGridSizePadded,
                                              DeviceBuffer<float>* realGrid,
                                              DeviceBuffer<float>* complexGrid) :
    pmeRawStream_(pmeStream.stream())
{
    const int numDomainsX = gridSizesInXForEachRank.size();
    const int numDomainsY = gridSizesInYForEachRank.size();

    GMX_RELEASE_ASSERT(allocateRealGrid == true, "Grids cannot be pre-allocated");
    GMX_RELEASE_ASSERT(performOutOfPlaceFFT == true, "Only out-of-place FFT supported");
    GMX_RELEASE_ASSERT(numDomainsX * numDomainsY > 1,
                       "HeFFTe backend is expected to be used only with more than 1 rank");

    // calculate grid offsets
    std::vector<int> gridOffsetsInX(numDomainsX + 1);
    std::vector<int> gridOffsetsInY(numDomainsY + 1);

    gridOffsetsInX[0] = 0;
    for (unsigned int i = 0; i < gridSizesInXForEachRank.size(); ++i)
    {
        gridOffsetsInX[i + 1] = gridOffsetsInX[i] + gridSizesInXForEachRank[i];
    }

    gridOffsetsInY[0] = 0;
    for (unsigned int i = 0; i < gridSizesInYForEachRank.size(); ++i)
    {
        gridOffsetsInY[i + 1] = gridOffsetsInY[i] + gridSizesInYForEachRank[i];
    }

    int rank, nProcs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nProcs);

    GMX_RELEASE_ASSERT(nProcs == numDomainsX * numDomainsY,
                       "Mismatch in communicator size and expected domain decomposition");

    // define how ranks are mapped to 2d domain
    int procY = rank % numDomainsY;
    int procX = rank / numDomainsY;

    // local real grid boxes
    heffte::box3d<> const realBox = { { 0, gridOffsetsInY[procY], gridOffsetsInX[procX] },
                                      { nz - 1, gridOffsetsInY[procY + 1] - 1, gridOffsetsInX[procX + 1] - 1 } };

    const int nx = gridOffsetsInX[numDomainsX];
    const int ny = gridOffsetsInY[numDomainsY];

    const int complexZDim = nz / 2 + 1;

    // if possible, keep complex data in slab decomposition along Z
    // this allows heffte to have single communication phase
    if (numDomainsY > 1 && complexZDim >= nProcs)
    {
        // define shape of local complex grid boxes
        std::vector<int> gridOffsetsInZ_transformed(nProcs + 1);

        for (int i = 0; i < nProcs; i++)
        {
            gridOffsetsInZ_transformed[i] = (i * complexZDim) / nProcs;
        }
        gridOffsetsInZ_transformed[nProcs] = complexZDim;

        heffte::box3d<> const complexBox = { { gridOffsetsInZ_transformed[rank], 0, 0 },
                                             { gridOffsetsInZ_transformed[rank + 1] - 1, ny - 1, nx - 1 } };

        // Define 3D FFT plan
#if GMX_SYCL_HIPSYCL
        pmeRawStream_
                .submit([&, &fftPlanRef = fftPlan_](sycl::handler& cgh) {
                    constexpr auto c_hipsyclBackend = sycl::backend::hip;
                    cgh.hipSYCL_enqueue_custom_operation([=, &fftPlanRef](sycl::interop_handle& gmx_unused h) {
                        auto stream = h.get_native_queue<c_hipsyclBackend>();
                        fftPlanRef  = std::make_unique<heffte::fft3d_r2c<backend_tag, int>>(
                                stream, realBox, complexBox, 0, comm, heffte::default_options<backend_tag>());
                    });
                })
                .wait();
#else
        // Define 3D FFT plan
        fftPlan_ = std::make_unique<heffte::fft3d_r2c<backend_tag, int>>(
                pmeRawStream_, realBox, complexBox, 0, comm, heffte::default_options<backend_tag>());
#endif
    }
    else
    {
        // define shape of local complex grid boxes
        std::vector<int> gridOffsetsInY_transformed(numDomainsX + 1);
        std::vector<int> gridOffsetsInZ_transformed(numDomainsY + 1);

        for (int i = 0; i < numDomainsX; i++)
        {
            gridOffsetsInY_transformed[i] = (i * ny + 0) / numDomainsX;
        }
        gridOffsetsInY_transformed[numDomainsX] = ny;

        for (int i = 0; i < numDomainsY; i++)
        {
            gridOffsetsInZ_transformed[i] = (i * complexZDim + 0) / numDomainsY;
        }
        gridOffsetsInZ_transformed[numDomainsY] = complexZDim;

        heffte::box3d<> const complexBox = {
            { gridOffsetsInZ_transformed[procY], gridOffsetsInY_transformed[procX], 0 },
            { gridOffsetsInZ_transformed[procY + 1] - 1, gridOffsetsInY_transformed[procX + 1] - 1, nx - 1 }
        };

        // Define 3D FFT plan
#if GMX_SYCL_HIPSYCL
        pmeRawStream_
                .submit([&, &fftPlanRef = fftPlan_](sycl::handler& cgh) {
                    constexpr auto c_hipsyclBackend = sycl::backend::hip;
                    cgh.hipSYCL_enqueue_custom_operation([=, &fftPlanRef](sycl::interop_handle& gmx_unused h) {
                        auto stream = h.get_native_queue<c_hipsyclBackend>();
                        fftPlanRef  = std::make_unique<heffte::fft3d_r2c<backend_tag, int>>(
                                stream, realBox, complexBox, 0, comm, heffte::default_options<backend_tag>());
                    });
                })
                .wait();
#else
        // Define 3D FFT plan
        fftPlan_ = std::make_unique<heffte::fft3d_r2c<backend_tag, int>>(
                pmeRawStream_, realBox, complexBox, 0, comm, heffte::default_options<backend_tag>());
#endif
    }

    workspace_ = heffte::gpu::vector<std::complex<float>>(fftPlan_->size_workspace());

    // allocate grid and return handles to it
#if GMX_GPU_CUDA
    localRealGrid_    = heffte::gpu::vector<float>(fftPlan_->size_inbox());
    localComplexGrid_ = heffte::gpu::vector<std::complex<float>>(fftPlan_->size_outbox());
    *realGrid         = localRealGrid_.data();
    *complexGrid      = (float*)localComplexGrid_.data();
    GMX_UNUSED_VALUE(context);
#elif GMX_GPU_SYCL
    allocateDeviceBuffer(&localRealGrid_, fftPlan_->size_inbox(), context);
    allocateDeviceBuffer(&localComplexGrid_, fftPlan_->size_outbox(), context);
    *realGrid                     = localRealGrid_;
    *complexGrid                  = localComplexGrid_;
#else
#    error "HeFFTe build only supports CUDA and SYCL"
#endif

    realGridSize[XX] = gridSizesInXForEachRank[procX];
    realGridSize[YY] = gridSizesInYForEachRank[procY];
    realGridSize[ZZ] = nz;

    realGridSizePadded[XX] = fftPlan_->inbox().size[2];
    realGridSizePadded[YY] = fftPlan_->inbox().size[1];
    realGridSizePadded[ZZ] = fftPlan_->inbox().size[0];

    complexGridSizePadded[XX] = fftPlan_->outbox().size[2];
    complexGridSizePadded[YY] = fftPlan_->outbox().size[1];
    complexGridSizePadded[ZZ] = fftPlan_->outbox().size[0];
}

template<typename backend_tag>
Gpu3dFft::ImplHeFfte<backend_tag>::~ImplHeFfte<backend_tag>() = default;

template<typename backend_tag>
void Gpu3dFft::ImplHeFfte<backend_tag>::perform3dFft(gmx_fft_direction dir, CommandEvent* /*timingEvent*/)
{
#if GMX_GPU_CUDA
    float*               realGrid    = localRealGrid_.data();
    std::complex<float>* complexGrid = localComplexGrid_.data();
#elif GMX_GPU_SYCL
    float*               realGrid = localRealGrid_.buffer_->ptr_;
    std::complex<float>* complexGrid =
            reinterpret_cast<std::complex<float>*>(localComplexGrid_.buffer_->ptr_);
#endif
    switch (dir)
    {
        case GMX_FFT_REAL_TO_COMPLEX:
#if GMX_SYCL_HIPSYCL
            pmeRawStream_.submit(GMX_SYCL_DISCARD_EVENT[&](sycl::handler & cgh) {
                cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle& gmx_unused h) {
                    fftPlan_->forward(realGrid, complexGrid, workspace_.data());
                });
            });
#else
            fftPlan_->forward(realGrid, complexGrid, workspace_.data());
#endif
            break;
        case GMX_FFT_COMPLEX_TO_REAL:
#if GMX_SYCL_HIPSYCL
            pmeRawStream_.submit(GMX_SYCL_DISCARD_EVENT[&](sycl::handler & cgh) {
                cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle& gmx_unused h) {
                    fftPlan_->backward(complexGrid, realGrid, workspace_.data());
                });
            });
#else
            fftPlan_->backward(complexGrid, realGrid, workspace_.data());
#endif
            break;
        default:
            GMX_THROW(NotImplementedError("The chosen 3D-FFT case is not implemented on GPUs"));
    }
}

// instantiate relevant HeFFTe backend
#if GMX_GPU_FFT_CUFFT
template class Gpu3dFft::ImplHeFfte<heffte::backend::cufft>;
#endif
#if GMX_GPU_FFT_MKL
template class Gpu3dFft::ImplHeFfte<heffte::backend::onemkl>;
#endif
#if GMX_GPU_FFT_ROCFFT
template class Gpu3dFft::ImplHeFfte<heffte::backend::rocfft>;
#endif

} // namespace gmx
