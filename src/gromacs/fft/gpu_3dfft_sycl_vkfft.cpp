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
 *  \brief Implements GPU 3D FFT routines for SYCL with VkFFT.
 *
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \ingroup module_fft
 *
 *  In DPC++, we can use VkFFT to perform the FFT.
 *
 */
// TODO use VkFFTGetVersion
#include "gmxpre.h"

#include "gpu_3dfft_sycl_vkfft.h"

#include "config.h"

#include <optional>

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/devicebuffer_sycl.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/strconvert.h"

//#include <level_zero/ze_api.h>

class DeviceContext;

#if (!GMX_SYCL_DPCPP)
#    error This file can only be compiled with Intel DPC++ compiler
#endif

#include <cstddef>
//#pragma clang diagnostic ignored "-Wsuggest-override" // can be removed when support for 2022.0 is dropped
//#pragma clang diagnostic ignored "-Wundefined-func-template"

namespace gmx
{

namespace
{

//! Helper for consistent error handling
void handleFftError(VkFFTResult result, const std::string& msg)
{
    if (result == VKFFT_SUCCESS)
    {
        return;
    }
    const char *c_vkfftErrorStrings[VKFFT_ERROR_FAILED_TO_SUBMIT_BARRIER+1];
	c_vkfftErrorStrings[0] = "VKFFT_SUCCESS";
	c_vkfftErrorStrings[1] = "VKFFT_ERROR_MALLOC_FAILED";
	c_vkfftErrorStrings[2] = "VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER";
	c_vkfftErrorStrings[3] = "VKFFT_ERROR_INSUFFICIENT_TEMP_BUFFER";
	c_vkfftErrorStrings[4] = "VKFFT_ERROR_PLAN_NOT_INITIALIZED";
	c_vkfftErrorStrings[5] = "VKFFT_ERROR_NULL_TEMP_PASSED";
	c_vkfftErrorStrings[1001] = "VKFFT_ERROR_INVALID_PHYSICAL_DEVICE";
	c_vkfftErrorStrings[1002] = "VKFFT_ERROR_INVALID_DEVICE";
	c_vkfftErrorStrings[1003] = "VKFFT_ERROR_INVALID_QUEUE";
	c_vkfftErrorStrings[1004] = "VKFFT_ERROR_INVALID_COMMAND_POOL";
	c_vkfftErrorStrings[1005] = "VKFFT_ERROR_INVALID_FENCE";
	c_vkfftErrorStrings[1006] = "VKFFT_ERROR_ONLY_FORWARD_FFT_INITIALIZED";
	c_vkfftErrorStrings[1007] = "VKFFT_ERROR_ONLY_INVERSE_FFT_INITIALIZED";
	c_vkfftErrorStrings[1008] = "VKFFT_ERROR_INVALID_CONTEXT";
	c_vkfftErrorStrings[1009] = "VKFFT_ERROR_INVALID_PLATFORM";
	c_vkfftErrorStrings[1010] = "VKFFT_ERROR_ENABLED_saveApplicationToString";
	c_vkfftErrorStrings[1011] = "VKFFT_ERROR_EMPTY_FILE";
	c_vkfftErrorStrings[2001] = "VKFFT_ERROR_EMPTY_FFTdim";
	c_vkfftErrorStrings[2002] = "VKFFT_ERROR_EMPTY_size";
	c_vkfftErrorStrings[2003] = "VKFFT_ERROR_EMPTY_bufferSize";
	c_vkfftErrorStrings[2004] = "VKFFT_ERROR_EMPTY_buffer";
	c_vkfftErrorStrings[2005] = "VKFFT_ERROR_EMPTY_tempBufferSize";
	c_vkfftErrorStrings[2006] = "VKFFT_ERROR_EMPTY_tempBuffer";
	c_vkfftErrorStrings[2007] = "VKFFT_ERROR_EMPTY_inputBufferSize";
	c_vkfftErrorStrings[2008] = "VKFFT_ERROR_EMPTY_inputBuffer";
	c_vkfftErrorStrings[2009] = "VKFFT_ERROR_EMPTY_outputBufferSize";
	c_vkfftErrorStrings[2010] = "VKFFT_ERROR_EMPTY_outputBuffer";
	c_vkfftErrorStrings[2011] = "VKFFT_ERROR_EMPTY_kernelSize";
	c_vkfftErrorStrings[2012] = "VKFFT_ERROR_EMPTY_kernel";
	c_vkfftErrorStrings[2013] = "VKFFT_ERROR_EMPTY_applicationString";
	c_vkfftErrorStrings[3001] = "VKFFT_ERROR_UNSUPPORTED_RADIX";
	c_vkfftErrorStrings[3002] = "VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH";
	c_vkfftErrorStrings[3003] = "VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_R2C";
	c_vkfftErrorStrings[3004] = "VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_DCT";
	c_vkfftErrorStrings[3005] = "VKFFT_ERROR_UNSUPPORTED_FFT_OMIT";
	c_vkfftErrorStrings[4001] = "VKFFT_ERROR_FAILED_TO_ALLOCATE";
	c_vkfftErrorStrings[4002] = "VKFFT_ERROR_FAILED_TO_MAP_MEMORY";
	c_vkfftErrorStrings[4003] = "VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS";
	c_vkfftErrorStrings[4004] = "VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER";
	c_vkfftErrorStrings[4005] = "VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER";
	c_vkfftErrorStrings[4006] = "VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE";
	c_vkfftErrorStrings[4007] = "VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES";
	c_vkfftErrorStrings[4008] = "VKFFT_ERROR_FAILED_TO_RESET_FENCES";
	c_vkfftErrorStrings[4009] = "VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_POOL";
	c_vkfftErrorStrings[4010] = "VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_SET_LAYOUT";
	c_vkfftErrorStrings[4011] = "VKFFT_ERROR_FAILED_TO_ALLOCATE_DESCRIPTOR_SETS";
	c_vkfftErrorStrings[4012] = "VKFFT_ERROR_FAILED_TO_CREATE_PIPELINE_LAYOUT";
	c_vkfftErrorStrings[4013] = "VKFFT_ERROR_FAILED_SHADER_PREPROCESS";
	c_vkfftErrorStrings[4014] = "VKFFT_ERROR_FAILED_SHADER_PARSE";
	c_vkfftErrorStrings[4015] = "VKFFT_ERROR_FAILED_SHADER_LINK";
	c_vkfftErrorStrings[4016] = "VKFFT_ERROR_FAILED_SPIRV_GENERATE";
	c_vkfftErrorStrings[4017] = "VKFFT_ERROR_FAILED_TO_CREATE_SHADER_MODULE";
	c_vkfftErrorStrings[4018] = "VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE";
	c_vkfftErrorStrings[4019] = "VKFFT_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER";
	c_vkfftErrorStrings[4020] = "VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE";
	c_vkfftErrorStrings[4021] = "VKFFT_ERROR_FAILED_TO_CREATE_DEVICE";
	c_vkfftErrorStrings[4022] = "VKFFT_ERROR_FAILED_TO_CREATE_FENCE";
	c_vkfftErrorStrings[4023] = "VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL";
	c_vkfftErrorStrings[4024] = "VKFFT_ERROR_FAILED_TO_CREATE_BUFFER";
	c_vkfftErrorStrings[4025] = "VKFFT_ERROR_FAILED_TO_ALLOCATE_MEMORY";
	c_vkfftErrorStrings[4026] = "VKFFT_ERROR_FAILED_TO_BIND_BUFFER_MEMORY";
	c_vkfftErrorStrings[4027] = "VKFFT_ERROR_FAILED_TO_FIND_MEMORY";
	c_vkfftErrorStrings[4028] = "VKFFT_ERROR_FAILED_TO_SYNCHRONIZE";
	c_vkfftErrorStrings[4029] = "VKFFT_ERROR_FAILED_TO_COPY";
	c_vkfftErrorStrings[4030] = "VKFFT_ERROR_FAILED_TO_CREATE_PROGRAM";
	c_vkfftErrorStrings[4031] = "VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM";
	c_vkfftErrorStrings[4032] = "VKFFT_ERROR_FAILED_TO_GET_CODE_SIZE";
	c_vkfftErrorStrings[4033] = "VKFFT_ERROR_FAILED_TO_GET_CODE";
	c_vkfftErrorStrings[4034] = "VKFFT_ERROR_FAILED_TO_DESTROY_PROGRAM";
	c_vkfftErrorStrings[4035] = "VKFFT_ERROR_FAILED_TO_LOAD_MODULE";
	c_vkfftErrorStrings[4036] = "VKFFT_ERROR_FAILED_TO_GET_FUNCTION";
	c_vkfftErrorStrings[4037] = "VKFFT_ERROR_FAILED_TO_SET_DYNAMIC_SHARED_MEMORY";
	c_vkfftErrorStrings[4038] = "VKFFT_ERROR_FAILED_TO_MODULE_GET_GLOBAL";
	c_vkfftErrorStrings[4039] = "VKFFT_ERROR_FAILED_TO_LAUNCH_KERNEL";
	c_vkfftErrorStrings[4040] = "VKFFT_ERROR_FAILED_TO_EVENT_RECORD";
	c_vkfftErrorStrings[4041] = "VKFFT_ERROR_FAILED_TO_ADD_NAME_EXPRESSION";
	c_vkfftErrorStrings[4042] = "VKFFT_ERROR_FAILED_TO_INITIALIZE";
	c_vkfftErrorStrings[4043] = "VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID";
	c_vkfftErrorStrings[4044] = "VKFFT_ERROR_FAILED_TO_GET_DEVICE";
	c_vkfftErrorStrings[4045] = "VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT";
	c_vkfftErrorStrings[4046] = "VKFFT_ERROR_FAILED_TO_CREATE_PIPELINE";
	c_vkfftErrorStrings[4047] = "VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG";
	c_vkfftErrorStrings[4048] = "VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE";
	c_vkfftErrorStrings[4049] = "VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE";
	c_vkfftErrorStrings[4050] = "VKFFT_ERROR_FAILED_TO_ENUMERATE_DEVICES";
	c_vkfftErrorStrings[4051] = "VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE";
	c_vkfftErrorStrings[4052] = "VKFFT_ERROR_FAILED_TO_CREATE_EVENT";
	c_vkfftErrorStrings[4053] = "VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST";
	c_vkfftErrorStrings[4054] = "VKFFT_ERROR_FAILED_TO_DESTROY_COMMAND_LIST";
	c_vkfftErrorStrings[4055] = "VKFFT_ERROR_FAILED_TO_SUBMIT_BARRIER";
    GMX_THROW(gmx::InternalError(gmx::formatString("%s: (error code %d - %s)\n", msg.c_str(), result, c_vkfftErrorStrings[result])));
}

//! Helper for consistent error handling
void handleFftError(VkFFTResult result, const std::string& direction, const std::string& msg)
{
    if (result == VKFFT_SUCCESS)
    {
        return;
    }
    handleFftError(result, msg + " doing " + direction);
}

//! Helper for consistent error handling
void handleFftError(ze_result_t result, const std::string& msg)
{
    if (result == ZE_RESULT_SUCCESS)
    {
        return;
    }
    const char *c_zeErrorStrings[1];
	c_zeErrorStrings[0] = "ZE_RESULT_SUCCESS";
    GMX_THROW(gmx::InternalError(gmx::formatString("%s: (error code %d - %s)\n", msg.c_str(), result, c_zeErrorStrings[result])));
}

} // namespace

    /*
Gpu3dFft::ImplSyclVkfft::Descriptor Gpu3dFft::ImplSyclVkfft::initDescriptor(const ivec realGridSize)
{
    try
    {
        const std::vector<std::int64_t> realGridDimensions{ realGridSize[XX],
                                                            realGridSize[YY],
                                                            realGridSize[ZZ] };
        return { realGridDimensions };
    }
    catch (oneapi::mkl::exception& exc)
    {
        GMX_THROW(InternalError(formatString("MKL failure while constructing descriptor: %s", exc.what())));
    }
}
    */

static std::optional<uint32_t> getEnvironmentVariableValue(const char *environmentVariable)
{
    const char* valueString = std::getenv(environmentVariable);
    if (valueString)
    {
        return intFromString(valueString);
    }
    return std::nullopt;
}

static uint32_t get_command_queue_group_ordinal(ze_device_handle_t device,
                                         ze_command_queue_group_property_flags_t flags)
{
    ze_result_t zeResult;
    uint32_t cmdqueue_group_count = 0;
    zeResult = zeDeviceGetCommandQueueGroupProperties(device, &cmdqueue_group_count, nullptr);
    handleFftError(zeResult, "Error getting command queue group properties count");
    auto cmdqueue_group_properties =
        std::vector<ze_command_queue_group_properties_t>(cmdqueue_group_count);
    zeResult = zeDeviceGetCommandQueueGroupProperties(device, &cmdqueue_group_count,
                                           cmdqueue_group_properties.data());
    handleFftError(zeResult, "Error getting command queue group properties count");

    uint32_t ordinal = cmdqueue_group_count;
    for (uint32_t i = 0; i < cmdqueue_group_count; ++i) {
        if ((~cmdqueue_group_properties[i].flags & flags) == 0) {
            ordinal = i;
            break;
        }
    }

    return ordinal;
}

Gpu3dFft::ImplSyclVkfft::ImplSyclVkfft(bool allocateRealGrid,
                                   MPI_Comm /*comm*/,
                                   ArrayRef<const int> gridSizesInXForEachRank,
                                   ArrayRef<const int> gridSizesInYForEachRank,
                                   int /*nz*/,
                                   const bool           performOutOfPlaceFFT,
                                   const DeviceContext& context,
                                   const DeviceStream&  pmeStream,
                                   ivec                 realGridSize,
                                   ivec                 realGridSizePadded,
                                   ivec                 complexGridSizePadded,
                                   DeviceBuffer<float>* realGrid,
                                   DeviceBuffer<float>* complexGrid) :
    Gpu3dFft::Impl::Impl(performOutOfPlaceFFT),
    realGrid_(*realGrid->buffer_),
    queue_(pmeStream.stream()),
    configuration_{0}
    /*
    ,
    r2cDescriptor_(initDescriptor(realGridSize)),
    c2rDescriptor_(initDescriptor(realGridSize))
    */
{
    GMX_RELEASE_ASSERT(!allocateRealGrid, "Grids needs to be pre-allocated");
    GMX_RELEASE_ASSERT(gridSizesInXForEachRank.size() == 1 && gridSizesInYForEachRank.size() == 1,
                       "Multi-rank FFT decomposition not implemented with SYCL VKFFT backend");

    GMX_ASSERT(checkDeviceBuffer(*realGrid,
                                 realGridSizePadded[XX] * realGridSizePadded[YY] * realGridSizePadded[ZZ]),
               "Real grid buffer is too small for the declared padded size");

    allocateComplexGrid(complexGridSizePadded, realGrid, complexGrid, context);
    GMX_ASSERT(checkDeviceBuffer(*complexGrid,
                                 complexGridSizePadded[XX] * complexGridSizePadded[YY]
                                         * complexGridSizePadded[ZZ] * 2),
               "Complex grid buffer is too small for the declared padded size");

    // Configure the FFT plan
    configuration_.FFTdim = 3;
    device_ = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(pmeStream.stream().get_device());
    configuration_.device = &device_;
    context_ = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context.context());
    configuration_.context = &context_;
    stream_ = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(pmeStream.stream());
    configuration_.commandQueue = &stream_;

    // VkFFT needs to be passed a command list that targets a command
    // queue that supports compute (for the FFTs) and copy (for phase
    // vectors, etc). We'd like to re-use the SYCL queue, but we
    // cannot query the ordinal from it via the
    // ze_command_queue_handle_t. So we just take the first command
    // queue group that supports compute and copy. Then we must sync
    // the operations in the command lists on the two command queues.
    const ze_command_queue_group_property_flags_t commandQueueGroupPropertyFlags = ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE | ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
    // This field of configuration_ is misnamed. It is used to fill the ordinal field of
    // a ze_command_queue_desc_t.
    configuration_.commandQueueID = get_command_queue_group_ordinal(device_, commandQueueGroupPropertyFlags);
    
    //configuration_.userTempBuffer = 0; // Not applicable to our use case
    // Buffer sizes only mandatory for Vulkan
    //configuration_.bufferSize = 0; //?
    //configuration_.inputBufferSize = 
    // TODO should we use performConvolution?
    configuration_.bufferOffset = 0;
    if (std::optional<uint32_t> value = getEnvironmentVariableValue("GMX_VKFFT_COALESCED_MEMORY"))
    {
        configuration_.coalescedMemory = value.value_or(64); // TODO is this appropriate?
    }
    if (std::optional<uint32_t> value = getEnvironmentVariableValue("GMX_VKFFT_AIM_THREADS"))
    {
        configuration_.aimThreads = value.value_or(128); // TODO tune me
    }
    if (std::optional<uint32_t> value = getEnvironmentVariableValue("GMX_VKFFT_NUM_SHARED_BANKS"))
    {
        configuration_.numSharedBanks = value.value_or(64); // TODO tune me
    }
    configuration_.numberBatches = 1;
    configuration_.useUint64 = 1;
    configuration_.performBandwidthBoost = 0; // TODO tune me
    configuration_.doublePrecision = GMX_DOUBLE;
    configuration_.halfPrecision = 0;
    configuration_.performR2C = 1;
    configuration_.isInputFormatted = 1; // Use separate input buffer for out-of-place transform
    configuration_.inverseReturnToInputBuffer = 1; // Return the result of C2R to the input buffer
    configuration_.normalize = 1; // TODO try off
    //    configuration_.printMemoryLayout = 1; // Needed only for debugging
    configuration_.inputBufferSize = inputBufferSize_.data();
    configuration_.inputBufferSize[XX] = uint64_t(sizeof(float)) * realGridSize[ZZ];
    configuration_.inputBufferSize[YY] = uint64_t(sizeof(float)) * realGridSize[YY];
    configuration_.inputBufferSize[ZZ] = uint64_t(sizeof(float)) * realGridSize[XX];
    /*
    configuration_.inputBufferStride[XX] = configuration_.size[XX];
    configuration_.inputBufferStride[YY] = configuration_.inputBufferStride[XX] * configuration_.size[YY];
    configuration_.inputBufferStride[ZZ] = configuration_.inputBufferStride[YY] * configuration_.size[ZZ];
    */
    /*
    configuration_.inputBufferStride[XX] = 0;
    configuration_.inputBufferStride[YY] = 0;
    configuration_.inputBufferStride[ZZ] = 0;
    */
    configuration_.inputBufferStride[XX] = realGridSizePadded[ZZ];
    configuration_.inputBufferStride[YY] = configuration_.inputBufferStride[XX] * realGridSizePadded[YY];
    configuration_.inputBufferStride[ZZ] = configuration_.inputBufferStride[YY] * realGridSizePadded[XX];
    /*
    configuration_.inputBufferStride[XX] = 1;
    configuration_.inputBufferStride[YY] = realGridSizePadded[ZZ];
    configuration_.inputBufferStride[ZZ] = realGridSizePadded[YY] * realGridSizePadded[ZZ];
    */
    //configuration_.size[XX] = (realGridSize[ZZ] / 2) + 1;
    configuration_.size[XX] = realGridSize[ZZ];
    configuration_.size[YY] = realGridSize[YY];
    configuration_.size[ZZ] = realGridSize[XX];
    //configuration_.size[ZZ] = (realGridSize[XX] / 2) + 1;
    /*
    configuration_.bufferStride[XX] = (configuration_.size[XX] / 2) + 1;
    configuration_.bufferStride[YY] = configuration_.bufferStride[XX] * configuration_.size[YY];
    configuration_.bufferStride[ZZ] = configuration_.bufferStride[YY] * configuration_.size[ZZ];
    */
    /*
    configuration_.bufferStride[XX] = 0;
    configuration_.bufferStride[YY] = 0;
    configuration_.bufferStride[ZZ] = 0;
    */
    configuration_.bufferStride[XX] = complexGridSizePadded[ZZ];
    //configuration_.bufferStride[XX] = (complexGridSizePadded[ZZ] / 2) + 1;
    configuration_.bufferStride[YY] = configuration_.bufferStride[XX] * complexGridSizePadded[YY];
    configuration_.bufferStride[ZZ] = configuration_.bufferStride[YY] * complexGridSizePadded[XX];
    configuration_.performZeropadding[XX] = 0;
    configuration_.performZeropadding[YY] = 0;
    configuration_.performZeropadding[ZZ] = 0;
    configuration_.performConvolution = 0;
    /*
    configuration_.
    configuration_.
    configuration_.
    configuration_.
    */

    VkFFTResult result;
    result = initializeVkFFT(&application_, configuration_);
    handleFftError(result, "Initializating VkFFT");
    
    /*
    // MKL expects row-major
    const std::array<MKL_LONG, 4> realGridStrides = {
        0, static_cast<MKL_LONG>(realGridSizePadded[YY] * realGridSizePadded[ZZ]), realGridSizePadded[ZZ], 1
    };
    const std::array<MKL_LONG, 4> complexGridStrides = {
        0,
        static_cast<MKL_LONG>(complexGridSizePadded[YY] * complexGridSizePadded[ZZ]),
        complexGridSizePadded[ZZ],
        1
    };

    const auto placement = performOutOfPlaceFFT ? DFTI_NOT_INPLACE : DFTI_INPLACE;

    try
    {
        using oneapi::mkl::dft::config_param;
        r2cDescriptor_.set_value(config_param::INPUT_STRIDES, realGridStrides.data());
        r2cDescriptor_.set_value(config_param::OUTPUT_STRIDES, complexGridStrides.data());
        r2cDescriptor_.set_value(config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        r2cDescriptor_.set_value(config_param::PLACEMENT, placement);
        r2cDescriptor_.commit(queue_);
    }
    catch (oneapi::mkl::exception& exc)
    {
        GMX_THROW(InternalError(
                formatString("MKL failure while configuring R2C descriptor: %s", exc.what())));
    }

    try
    {
        using oneapi::mkl::dft::config_param;
        c2rDescriptor_.set_value(config_param::INPUT_STRIDES, complexGridStrides.data());
        c2rDescriptor_.set_value(config_param::OUTPUT_STRIDES, realGridStrides.data());
        c2rDescriptor_.set_value(config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        c2rDescriptor_.set_value(config_param::PLACEMENT, placement);
        c2rDescriptor_.commit(queue_);
    }
    catch (oneapi::mkl::exception& exc)
    {
        GMX_THROW(InternalError(
                formatString("MKL failure while configuring C2R descriptor: %s", exc.what())));
    }
    */

    ze_result_t zeResult;

    // TODO comment
    zeCommandListDescription_.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    zeCommandListDescription_.pNext = nullptr;

    // Use the same command queue group as VkFFT uses for its transfers
    zeCommandListDescription_.commandQueueGroupOrdinal = configuration_.commandQueueID;
    zeCommandListDescription_.flags = ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY;

    // Create event pool
    ze_event_pool_desc_t eventPoolDescription = {
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        nullptr,
        0, // default: events visible to device and peers
        4 // count of events in this pool
    };
    zeResult = zeEventPoolCreate(context_, &eventPoolDescription, 1, &device_, &eventPool_);
    handleFftError(zeResult, "Error creating event pool");

    EnumerationArray<FftDirection, ze_event_desc_t> descriptionOfEventBeforeFft, descriptionOfEventAfterFft;
    uint32_t eventPoolIndex = 0;
    for (FftDirection direction : EnumerationWrapper<FftDirection>{})
    {
        // Create the event to use before the FFT for this direction
        descriptionOfEventBeforeFft[direction] = {
            ZE_STRUCTURE_TYPE_EVENT_DESC,
            nullptr,
            eventPoolIndex++,
            // ensure memory coherency across device after event completes
            ZE_EVENT_SCOPE_FLAG_DEVICE,
            ZE_EVENT_SCOPE_FLAG_DEVICE
        };
        zeResult = zeEventCreate(eventPool_, &descriptionOfEventBeforeFft[direction], &eventBeforeFft_[direction]);
        handleFftError(zeResult, formatString("Error creating event before FFT for direction %d", static_cast<int>(direction)));

        // Create the event to use after the FFT for this direction
        descriptionOfEventAfterFft[direction] = {
            ZE_STRUCTURE_TYPE_EVENT_DESC,
            nullptr,
            eventPoolIndex++,
            // ensure memory coherency across device after event completes
            ZE_EVENT_SCOPE_FLAG_DEVICE,
            ZE_EVENT_SCOPE_FLAG_DEVICE
        };
        // TODO file bug, when eventPoolIndex is out of range, garbage error code is returned
        zeResult = zeEventCreate(eventPool_, &descriptionOfEventAfterFft[direction], &eventAfterFft_[direction]);
        handleFftError(zeResult, formatString("Error creating event after FFT for direction %d", static_cast<int>(direction)));
    }
}

Gpu3dFft::ImplSyclVkfft::~ImplSyclVkfft()
{
    deleteVkFFT(&application_);
    deallocateComplexGrid();
}

void Gpu3dFft::ImplSyclVkfft::perform3dFft(gmx_fft_direction dir, CommandEvent* /*timingEvent*/)
{
#if GMX_SYCL_USE_USM
    // TODO can we just use buffer_ directly?
    void* realGridPtr = realGrid_;
    void* complexGridPtr = *complexGrid_.buffer_;
#else
    float* realGridPtr = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(realGrid_);
    sycl::buffer<float, 1> complexGrid = *complexGrid_.buffer_;
    float* complexGridPtr = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(complexGrid);
#endif
    ze_result_t zeResult;
    ze_command_queue_handle_t zeQueue = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue_);
    fprintf(stderr, "gmx: executing command list\ngmx: zeQueue = %p\n", reinterpret_cast<void*>(zeQueue));

    // TODO this sync should not be needed
    //zeResult = zeCommandQueueSynchronize(zeQueue, UINT64_MAX);
    //handleFftError(zeResult, "Error synchronizing queue before VkFFT work");

    //zeResult = zeCommandListAppendSignalEvent(zeQueue, 
    
    VkFFTLaunchParams launchParameters;
    launchParameters.inputBuffer = &realGridPtr;
    launchParameters.buffer = &complexGridPtr;

    // Make a command list to put into the SYCL command queue that
    // will send the signal that the FFT can start and wait for the
    // signal that it has ended.
    ze_command_list_handle_t zeCommandListFftSignalling;
    // TODO zeCommandListDescription_.commandQueueGroupOrdinal could probably just be based on copy
    zeResult = zeCommandListCreate(context_, device_, &zeCommandListDescription_, &zeCommandListFftSignalling);
    handleFftError(zeResult, "Error creating LevelZero command list to signal FFT start");
       

    ze_command_list_handle_t zeCommandListFftCompute;
    zeResult = zeCommandListCreate(context_, device_, &zeCommandListDescription_, &zeCommandListFftCompute);
    handleFftError(zeResult, "Error creating LevelZero command list for FFT compute");
    launchParameters.commandList = &zeCommandListFftCompute;

    VkFFTResult result;
    const int forwardTransfer = -1;
    const int backwardTransfer = 1;
    switch (dir)
    {
        case GMX_FFT_REAL_TO_COMPLEX:
            /*
            zeResult = zeCommandListAppendSignalEvent(zeCommandListFftSignalling, eventBeforeFft_[FftDirection::RealToComplex]);
            handleFftError(zeResult, "Error appending signal before real-to-complex");
            zeResult = zeCommandListAppendWaitOnEvents(zeCommandListFftCompute, 1, &eventBeforeFft_[FftDirection::RealToComplex]);
            handleFftError(zeResult, "Error appending wait before real-to-complex");
            */
            fprintf(stderr, "gmx: submitting FFT work\n");
            result = VkFFTAppend(&application_, forwardTransfer, &launchParameters);
            handleFftError(result, "Transform failure doing real-to-complex");
            fprintf(stderr, "gmx: done submitting FFT work, enqueuing barrier\n");
            zeResult = zeCommandListAppendBarrier(zeCommandListFftCompute, nullptr, 0, nullptr);
            handleFftError(zeResult, "Error appending barrier after real-to-complex");
            /*
            zeResult = zeCommandListAppendSignalEvent(zeCommandListFftCompute, eventAfterFft_[FftDirection::RealToComplex]);
            handleFftError(zeResult, "Error appending signal after real-to-complex");
            zeResult = zeCommandListAppendWaitOnEvents(zeCommandListFftSignalling, 1, &eventAfterFft_[FftDirection::RealToComplex]);
            handleFftError(zeResult, "Error appending wait after real-to-complex");
            */
            break;
        case GMX_FFT_COMPLEX_TO_REAL:
            result = VkFFTAppend(&application_, backwardTransfer, &launchParameters);
            handleFftError(result, "Transform failure doing complex-to-real");
            //zeResult = zeCommandListAppendSignalEvent(zeQueue, &eventAfterFft_[FftDirection::ComplexToReal]);
            //handleFftError(zeResult, "Error appending signal after complex-to-real");
            break;
        default:
            GMX_THROW(NotImplementedError("The chosen 3D-FFT case is not implemented on GPUs"));
    }
    /*
    zeResult = zeCommandListClose(zeCommandListFftSignalling);
    handleFftError(zeResult, "Error closing command list for FFT signalling\n");
    zeResult = zeCommandQueueExecuteCommandLists(zeQueue, 1, &zeCommandListFftSignalling, nullptr);
    handleFftError(zeResult, "Error executing command list for FFT signalling\n");
    */
    zeResult = zeCommandListClose(zeCommandListFftCompute);
    handleFftError(zeResult, "Error closing command list for FFT compute\n");
    fprintf(stderr, "gmx: executing FFT work\n");
    zeResult = zeCommandQueueExecuteCommandLists(zeQueue, 1, &zeCommandListFftCompute, nullptr);
    handleFftError(zeResult, "Error executing command list for FFT compute\n");

    // TODO this sync should not be needed
    //zeResult = zeCommandQueueSynchronize(zeQueue, UINT64_MAX);
    //handleFftError(zeResult, "Error synchronizing queue after VkFFT work");
}

} // namespace gmx
