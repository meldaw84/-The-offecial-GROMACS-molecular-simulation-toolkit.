/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2018,2019,2020,2021, by the GROMACS development team, led by
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

/*! \libinternal \file
 *  \brief Implements the DeviceBuffer type and routines for CUDA.
 *  Should only be included directly by the main DeviceBuffer file devicebuffer.h.
 *  TODO: the intent is for DeviceBuffer to become a class.
 *
 *  \author Mahesh Doijade mdoijade@nvidia.com
 *
 *  \inlibraryapi
 */


#ifndef GMX_GPU_UTILS_NVSHMEMBUFFER_CUH
#define GMX_GPU_UTILS_NVSHMEMBUFFER_CUH

#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gpu_utils.h" //only for GpuApiCallBehavior
#include "gromacs/gpu_utils/gputraits.cuh"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

#if GMX_NVSHMEM

#include <nvshmem.h>

/*! \brief
 * Allocates a NVSHMEM buffer.
 * It is currently a caller's responsibility to call it only on not-yet allocated buffers.
 *
 * \tparam        ValueType            Raw value type of the \p buffer.
 * \param[in,out] buffer               Pointer to the device-side buffer.
 * \param[in]     numValues            Number of values to accommodate.
 * \param[in]     deviceContext        The buffer's dummy device  context - not managed explicitly in CUDA RT.
 */
template<typename ValueType>
void allocateNvshmemBuffer(DeviceBuffer<ValueType>* buffer, size_t numValues, const DeviceContext& /* deviceContext */)
{
    GMX_ASSERT(buffer, "needs a buffer pointer");
    *buffer = (ValueType*) nvshmem_malloc(numValues * sizeof(ValueType));

    GMX_RELEASE_ASSERT(
            *buffer != nullptr, "Allocation of the nvshmem buffer failed.");
}

/*! \brief
 * Frees a device-side buffer.
 * This does not reset separately stored size/capacity integers,
 * as this is planned to be a destructor of DeviceBuffer as a proper class,
 * and no calls on \p buffer should be made afterwards.
 *
 * \param[in] buffer  Pointer to the buffer to free.
 */
template<typename DeviceBuffer>
void freeNvshmemBuffer(DeviceBuffer* buffer)
{
    GMX_ASSERT(buffer, "needs a buffer pointer");
    if (*buffer)
    {
        nvshmem_free(*buffer);
    }
}

/*! \brief
 *  Reallocates the device-side buffer.
 *
 *  Reallocates the device-side memory pointed by \p buffer.
 *  Allocation is buffered and therefore freeing is only needed
 *  if the previously allocated space is not enough.
 *  \p currentNumValues and \p currentMaxNumValues are updated.
 *  TODO: \p currentNumValues, \p currentMaxNumValues, \p deviceContext
 *  should all be encapsulated in a host-side class together with the buffer.
 *
 *  \tparam        ValueType            Raw value type of the \p buffer.
 *  \param[in,out] buffer               Pointer to the device-side buffer
 *  \param[in]     numValues            Number of values to accommodate.
 *  \param[in,out] currentNumValues     The pointer to the buffer's number of values.
 *  \param[in,out] currentMaxNumValues  The pointer to the buffer's capacity.
 *  \param[in]     deviceContext        The buffer's device context.
 */
template<typename ValueType>
void reallocateNvshmemBuffer(DeviceBuffer<ValueType>* buffer,
                            size_t                   numValues,
                            int*                     currentNumValues,
                            int*                     currentMaxNumValues,
                            const DeviceContext&     deviceContext)
{
    GMX_ASSERT(buffer, "needs a buffer pointer");
    GMX_ASSERT(currentNumValues, "needs a size pointer");
    GMX_ASSERT(currentMaxNumValues, "needs a capacity pointer");

    /* reallocate only if the data does not fit */
    if (static_cast<int>(numValues) > *currentMaxNumValues)
    {
        if (*currentMaxNumValues >= 0)
        {
            freeNvshmemBuffer(buffer);
        }

        *currentMaxNumValues = over_alloc_large(numValues);
        allocateNvshmemBuffer(buffer, *currentMaxNumValues, deviceContext);
    }
    /* size could have changed without actual reallocation */
    *currentNumValues = numValues;
}
#endif

#endif
